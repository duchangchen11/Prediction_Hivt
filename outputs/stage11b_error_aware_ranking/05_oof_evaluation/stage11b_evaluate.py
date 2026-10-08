"""One OOF score per HeadTrain actor after ALL fold checkpoints are frozen."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11b_common import *

FIELDS=('Top1FDE','Top1ADE','OracleGap','HitRate','MRR','SoftCE','ExpectedRegret','NormalizedExpectedRegret',
    'PredictionEntropy','Top1Probability','Top1Top2Margin','OracleModeProbability','minADEOracle6','minFDE6','MR6')

@torch.no_grad()
def main():
    seed();verify();frozen=read_json(ROOT/'04_checkpoints/stage11b_all_frozen.json');assert frozen['Status']=='FROZEN_ALL_COMPLETE'
    assert read_json(ROOT/'09_reports/stage11b_pre_oof_training_audit.json')['Status']=='PASS'
    for r in frozen['Checkpoints']:assert sha256(PROJECT/r['Path'])==r['SHA256']
    dest=ROOT/'05_oof_evaluation/cache';dest.mkdir(exist_ok=True);assert not (dest/'complete.json').exists()
    f=frame().copy();n=len(f);filled=np.zeros(n,bool);foldid=np.zeros(n,np.int8)
    logits=np.lib.format.open_memmap(dest/'stage11b_oof_predictions_logits.npy',mode='w+',dtype='float32',shape=(n,5,6))
    prob=np.lib.format.open_memmap(dest/'stage11b_oof_predictions_probabilities.npy',mode='w+',dtype='float32',shape=(n,5,6))
    values=np.lib.format.open_memmap(dest/'stage11b_oof_metrics.npy',mode='w+',dtype='float64',shape=(n,5,len(FIELDS)))
    candidates=np.load(S11/'01_identity_audit/cache/candidates.npy',mmap_mode='r');candidate_sha_before=sha256(S11/'01_identity_audit/cache/candidates.npy')
    foldrows=[];efficiency=[];peak={};times={};batchcounts={};bicyclecount=0
    for fold in (1,2,3):
        assert read_json(ROOT/f'04_checkpoints/fold{fold}/stage11b_frozen.json')['Status']=='FROZEN'
        store=Store(fold);ids=indices(fold,'OuterTest');assert not filled[ids].any()
        assert not set(f.iloc[ids].scene_token)&set(split(fold)['InnerTrain']+split(fold)['InnerDev'])
        models={};seed(2022);models['R2']=ReliabilityHead().cuda()
        for name in VARIANTS:models[name]=fresh(fold)
        for name,m in models.items():
            saved=torch.load(ROOT/f'04_checkpoints/fold{fold}/{name}_best.pt',map_location='cpu',weights_only=False)
            assert saved['Fold']==fold and saved['Model']==name
            assert saved['normalization_sha256']==sha256(store.normpath)
            m.load_state_dict(saved['state_dict']);m.eval().requires_grad_(False)
        before={name:state_sha(m) for name,m in models.items()}
        elapsed={name:0. for name in MODELS};prepare_seconds=0.;torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        for start in range(0,len(ids),128):
            ix=ids[start:start+128];begin=time.monotonic();args,fd,ad=store.batch(ix);x=store.r2(ix);torch.cuda.synchronize();prepare_seconds+=time.monotonic()-begin
            shared=candidates[store.source[ix]];shared_sha=hashlib.sha256(shared.tobytes()).hexdigest()
            zz={};pp={}
            begin=time.monotonic();zz['R0']=args[6];pp['R0']=args[6].softmax(-1);torch.cuda.synchronize();elapsed['R0']+=time.monotonic()-begin
            begin=time.monotonic();r=models['R2'](x,args[6]);zz['R2']=r['mode_logits'];pp['R2']=r['mode_prob'];torch.cuda.synchronize();elapsed['R2']+=time.monotonic()-begin
            bike=torch.as_tensor(store.types[ix]==2,device='cuda');bicyclecount+=int(bike.sum())
            for name in VARIANTS:
                begin=time.monotonic();out=models[name](*args);zz[name]=out['mode_logits'].clone();pp[name]=out['mode_prob'].clone()
                zz[name][bike]=zz['R2'][bike];pp[name][bike]=pp['R2'][bike];torch.cuda.synchronize();elapsed[name]+=time.monotonic()-begin
                assert torch.equal(zz[name][bike],zz['R2'][bike]) and torch.equal(pp[name][bike],pp['R2'][bike])
                assert torch.equal(pp[name][bike].argmax(-1),pp['R2'][bike].argmax(-1))
            assert hashlib.sha256(shared.tobytes()).hexdigest()==shared_sha
            ii=torch.arange(len(ix),device='cuda');best=fd.argmin(-1);oracle=fd.min(-1).values;oracleade=ad.min(-1).values
            costs=fd-oracle[:,None];scale=costs.mean(-1).clamp(min=1)
            for j,name in enumerate(MODELS):
                z,p=zz[name],pp[name];assert torch.isfinite(z).all() and torch.isfinite(p).all()
                assert torch.allclose(p.sum(-1),torch.ones(len(ix),device='cuda'),atol=1e-6,rtol=0.)
                top=p.argmax(-1);order=p.argsort(dim=-1,descending=True,stable=True);rank=(order==best[:,None]).long().argmax(-1)+1
                sortedp=p.sort(-1,descending=True).values
                v=torch.stack((fd[ii,top].double(),ad[ii,top].double(),fd[ii,top].double()-oracle.double(),(top==best).double(),
                    1./rank.double(),objective(z,fd,'A').double(),(p*costs).sum(-1).double(),
                    (p*(costs/scale[:,None])).sum(-1).double(),-(p*p.clamp(min=1e-30).log()).sum(-1).double(),
                    sortedp[:,0].double(),(sortedp[:,0]-sortedp[:,1]).double(),p[ii,best].double(),
                    oracleade.double(),oracle.double(),(oracle>2).double()),-1)
                logits[ix,j]=z.cpu().numpy();prob[ix,j]=p.cpu().numpy();values[ix,j]=v.cpu().numpy()
            assert np.array_equal(values[ix,0,12:],values[ix,1,12:])
            for j in range(2,5):assert np.array_equal(values[ix,j,12:],values[ix,1,12:])
            filled[ix]=True;foldid[ix]=fold
        assert before=={name:state_sha(m) for name,m in models.items()}
        foldframe=f.iloc[ids];g=groups(foldframe)
        for group,mask in g.items():
            if not mask.any():continue
            for j,name in enumerate(MODELS):foldrows.append({'Fold':fold,'Group':group,'Model':name,'Count':int(mask.sum()),
                **dict(zip(FIELDS,values[ids[mask],j].mean(0)))})
        for name in MODELS:
            summary={} if name=='R0' else read_json(ROOT/f'03_training/fold{fold}/{name}/stage11b_summary.json')
            efficiency.append({'Fold':fold,'Model':name,'Params':summary.get('Params',0),'TrainingSeconds':summary.get('Seconds',0),
                'MeanEpochSeconds':summary.get('MeanEpochSeconds',0),'ExecutedEpochs':summary.get('ExecutedEpochs',0),
                'TrainPeakGPUMemoryBytes':summary.get('PeakGPUMemoryBytes',0),'EvalPeakGPUMemoryBytes':torch.cuda.max_memory_allocated(),
                'OOFActors':len(ids),'HeadEvalSeconds':elapsed[name],'HeadLatencyMSPerActor':1000*elapsed[name]/len(ids),
                'ApproxRankingWithFeaturePreparationMSPerActor':1000*(elapsed[name]+prepare_seconds+(elapsed['R2'] if name in VARIANTS else 0))/len(ids),
                'TimingScope':'FP32 cached-feature ranking, CUDA synchronized batches128; frozen predictor forward excluded'})
        print('OUTER_FOLD_PASS',fold,len(ids),'seconds',time.monotonic()-started,flush=True)
        del models,store;torch.cuda.empty_cache()
    assert filled.all() and np.all(foldid>0) and len(set(f.scene_token))==630
    assert sha256(S11/'01_identity_audit/cache/candidates.npy')==candidate_sha_before and bicyclecount==2980
    f['Fold']=foldid;f['Partition']='OuterTest';f.to_csv(dest/'stage11b_oof_predictions_actor_records.csv',index=False)
    rows=[]
    for group,mask in groups(f).items():
        for j,name in enumerate(MODELS):rows.append({'Group':group,'Model':name,'Count':int(mask.sum()),**dict(zip(FIELDS,values[mask,j].mean(0)))})
    dump('05_oof_evaluation/stage11b_fold_metrics.csv',foldrows)
    dump('05_oof_evaluation/stage11b_oof_main_results.csv',[r for r in rows if r['Group']=='Overall'])
    dump('05_oof_evaluation/stage11b_oof_type_results.csv',[r for r in rows if r['Group'] in TYPES])
    dump('05_oof_evaluation/stage11b_oof_motion_results.csv',[r for r in rows if r['Group'] not in ('Overall',*TYPES)])
    dump('07_diagnostics/stage11b_probability_statistics.csv',[{k:r[k] for k in ('Group','Model','Count','PredictionEntropy','Top1Probability','Top1Top2Margin','OracleModeProbability','SoftCE')} for r in rows])
    dump('07_diagnostics/stage11b_efficiency.csv',efficiency)
    for array in (logits,prob,values):array.flush()
    audit={'Status':'PASS','Folds':3,'OOFScenes':630,'OOFActors':n,'EachSceneExactlyOneFold':True,'InnerDevInOOF':False,
        'OuterTestSeenByCorrespondingRankingTraining':False,'CheckpointsFrozenBeforeOOF':True,'CandidateIdentity':'PASS','CandidateCoordinateMaxDiff':0,
        'GeometryAllModelsBitwiseEqual':True,'BicyclePreserved':'YES','BicycleActors':bicyclecount,
        'BicycleLogitProbabilityModeMaxDiff':0,'ModelStatesUnchanged':True,'official_VAL_or_test_used':False,
        'fields':FIELDS,'models':MODELS,'cache_files':{p.name:sha256(p) for p in dest.glob('*') if p.suffix in ('.npy','.csv')}}
    atomic_json(dest/'complete.json',audit);atomic_json(ROOT/'05_oof_evaluation/stage11b_oof_identity_audit.json',audit)
    verify(history=True);print('ALL_OOF_IDENTITY_AND_METRICS_PASS',flush=True)
if __name__=='__main__':main()
