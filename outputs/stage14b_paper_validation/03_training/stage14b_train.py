"""Three capacity-matched NoGraph runs; original Stage11B loop and InnerDev selection.

The original CPU state, dev assessment and graph training function bodies are
reused through AST. Only artifact prefixes, the structural parameter count and
an additional historical batch-order assertion change those bodies.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_protocol'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_preflight'))
from stage14b_common import *
from stage14b_capacity_preflight import gradients

def config(fold,name):
    assert name in VARIANTS
    folder=ROOT/f'03_training/fold{fold}/{name}';folder.mkdir(parents=True,exist_ok=True)
    protocol=read_json(PROTOCOL)
    c={k:protocol[k] for k in ('optimizer','learning_rate','weight_decay','precision','AMP',
        'microbatch','accumulation','effective_batch','max_epochs','patience','carry','batch_order','ABC_selection')}
    c.update(Fold=fold,Model=name,PaperModel='Matched-NG-'+name,Architecture='Capacity matched own-node residual adapter; zero interaction message',
        protocol_sha256=sha256(PROTOCOL),
        split_sha256=sha256(S11B/f'02_splits/stage11b_fold{fold}_split.json'),
        normalization_sha256=sha256(S11B/f'02_splits/stage11b_fold{fold}_normalization.json'),
        training_indices_sha256=array_sha(indices(fold,'InnerTrain')),
        InnerDev_indices_sha256=array_sha(indices(fold,'InnerDev')),
        # Freeze training sources only. Evaluation code is isolated by the
        # pre-freeze read guard and is not part of the fitting computation.
        new_sources_sha256={str(p.relative_to(ROOT)):sha256(p) for directory in
            ('00_protocol','01_preflight','02_models','03_training')
            for p in sorted((ROOT/directory).glob('stage14b_*.py'))},
        OriginalTrainingSourceSHA256=sha256(S11B/'03_training/stage11b_train.py'))
    atomic_json(folder/'stage14b_training_config.json',c);return folder,c

def original_order_audit(fold,epoch,order_hash):
    for old_variant in ('A','C'):
        old=pd.read_csv(S11B/f'03_training/fold{fold}/{old_variant}/stage11b_batch_order.csv')
        if epoch<=len(old):assert old.iloc[epoch-1].OrderSHA256==order_hash

source=S11B/'03_training/stage11b_train.py'
tree=ast.parse(source.read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'cpu_state','assess_graph','train_graph'}]
class OwnArtifacts(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):return ast.copy_location(ast.Constant(node.value.replace('stage11b_','stage14b_')),node)
        if node.value==24066:return ast.copy_location(ast.Constant(24001),node)
        return node
nodes=[OwnArtifacts().visit(n) for n in nodes]
training=next(n for n in nodes if n.name=='train_graph')
epoch_loop=next(n for n in training.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch')
order_statement=next(i for i,n in enumerate(epoch_loop.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
    and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='append')
epoch_loop.body.insert(order_statement+1,ast.parse('original_order_audit(fold,epoch,order_hash)').body[0])
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source)+' [Stage14B controlled adaptation]','exec'),globals())

def main():
    seed();verify()
    tiny=read_json(ROOT/'01_preflight/stage14b_tiny_audit.json')
    assert tiny['Status']=='PASS' and tiny['FormalTrainingPermitted']
    assert read_json(ROOT/'01_preflight/stage14b_capacity_preflight.json')['Status']=='PASS'
    assert sha256(ROOT/'02_models/stage14b_matched_nograph.py')==read_json(REG)['ModelSourceSHA256']
    final=ROOT/'04_checkpoints/stage14b_all_frozen.json';assert not final.exists(),'preserve frozen experiment'
    predictor=frozen_predictor('cpu');ps=state_sha(predictor);summaries=[];checkpoints=[]
    for fold in (1,2,3):
        frozen=ROOT/f'04_checkpoints/fold{fold}/stage14b_frozen.json';assert not frozen.exists()
        store=Store(fold);r2=load_fold_r2(fold);rs=state_sha(r2)
        old_r2=read_json(S11B/f'03_training/fold{fold}/R2/stage11b_summary.json')
        refs={t:old_r2['DevMetrics'][t]['Top1FDE'] for t in ('Vehicle','Pedestrian')}
        fold_rows=[]
        for variant in VARIANTS:
            print('START_FORMAL',fold,'Matched-NG-'+variant,flush=True)
            result=train_graph(fold,variant,store,predictor,r2,refs)
            assert result['Status']=='COMPLETE' and not result['OuterTestUsed']
            result.update(PaperModel='Matched-NG-'+variant,TrainableParams=24001,HistoryBatchOrder='EXACT_MATCH_ON_COMMON_EPOCH_PREFIX',
                FrozenR2Reused=True,TinyWeightsUsed=False)
            atomic_json(ROOT/f'03_training/fold{fold}/{variant}/stage14b_summary.json',result)
            summaries.append(result)
            row=dict(Fold=fold,Model='Matched-NG-'+variant,Path=str(cp_path(fold,variant).relative_to(PROJECT)),
                SHA256=sha256(cp_path(fold,variant)),SelectedEpoch=result['SelectedEpoch'],CheckpointScore=result['CheckpointScore'],
                NormalizationSHA256=result['normalization_sha256'],SplitSHA256=result['split_sha256'],TrainingConfigSHA256=result['config_sha256'])
            checkpoints.append(row);fold_rows.append(row)
            assert state_sha(r2)==rs and state_sha(predictor)==ps
            dump('03_training/stage14b_training_summary.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in summaries])
            dump('04_checkpoints/stage14b_checkpoint_manifest.csv',checkpoints)
        atomic_json(frozen,dict(Status='FROZEN',Fold=fold,Checkpoints=fold_rows,OuterTestEvaluationPermitted=False,
            Reason='all3 global gate required',FurtherTrainingPermitted=False))
        verify();del store,r2;torch.cuda.empty_cache()
    assert len(checkpoints)==3
    verify(history=True)
    atomic_json(final,dict(Status='FROZEN_ALL_COMPLETE',Folds=3,Checkpoints=checkpoints,
        predictor_state_sha256=ps,TinyWeightsUsed=False,NoOuterTestModelSelection=True,
        HeadDevUsed=False,OfficialVALTestUsed=False,NewFormalTrainingRuns=3,HistoricalGACRetrained=False,
        FurtherTrainingPermitted=False,OuterTestEvaluationPermitted=True))
    print('STAGE14B_ALL_THREE_CHECKPOINTS_FROZEN',flush=True)

if __name__=='__main__':main()
