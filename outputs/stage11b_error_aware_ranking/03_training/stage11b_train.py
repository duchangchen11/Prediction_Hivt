"""Strict Fold1 R2/A/B/C then Fold2 then Fold3, no OuterTest reads."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_preflight'))
from stage11b_common import *
from stage11b_tiny import gradients

def cp_path(fold,name):return ROOT/f'04_checkpoints/fold{fold}/{name}_best.pt'
def cpu_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def config(fold,name):
    folder=ROOT/f'03_training/fold{fold}/{name}';folder.mkdir(parents=True,exist_ok=True)
    c={**read_json(PROTOCOL)['R2_protocol']} if name=='R2' else {k:read_json(PROTOCOL)[k] for k in
        ('optimizer','learning_rate','weight_decay','precision','AMP','microbatch','accumulation','effective_batch','max_epochs','patience','carry','batch_order','ABC_selection')}
    c.update({'Fold':fold,'Model':name,'protocol_sha256':sha256(PROTOCOL),
        'split_sha256':sha256(ROOT/f'02_splits/stage11b_fold{fold}_split.json'),
        'normalization_sha256':sha256(ROOT/f'02_splits/stage11b_fold{fold}_normalization.json'),
        'training_indices_sha256':hashlib.sha256(indices(fold,'InnerTrain').tobytes()).hexdigest(),
        'InnerDev_indices_sha256':hashlib.sha256(indices(fold,'InnerDev').tobytes()).hexdigest(),
        'new_sources_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sorted(ROOT.rglob('stage11b_*.py'))}})
    atomic_json(folder/'stage11b_training_config.json',c);return folder,c

def data_r2(store,ids):
    source=store.source[ids]
    return {'features_R2':store.r2(ids),'base_logits':torch.from_numpy(np.array(store.base[source],copy=True)).cuda(),
        'FDE_by_mode':torch.from_numpy(np.array(store.fde[source],copy=True)).cuda(),
        'ADE_by_mode':torch.from_numpy(np.array(store.ade[source],copy=True)).cuda()}

@torch.no_grad()
def assess_r2(model,data,types):
    model.eval();n=len(types);sums=np.zeros((4,4),np.float64);counts=np.array([n,*[(types==k).sum() for k in range(3)]])
    for start in range(0,n,4096):
        x=data['features_R2'][start:start+4096];z=data['base_logits'][start:start+4096];fd=data['FDE_by_mode'][start:start+4096];ad=data['ADE_by_mode'][start:start+4096]
        out=model(x,z);top=out['mode_prob'].argmax(-1);ii=torch.arange(len(top),device='cuda')
        vv=torch.stack((fd[ii,top],ad[ii,top],(top==fd.argmin(-1)).float(),objective(out['mode_logits'],fd,'A')),-1).double().cpu().numpy()
        for g in range(4):
            keep=np.ones(len(top),bool) if g==0 else types[start:start+len(top)]==g-1;sums[g]+=vv[keep].sum(0)
    return {name:{'Count':int(counts[j]),**dict(zip(('Top1FDE','Top1ADE','HitRate','SoftCE'),(sums[j]/counts[j]).tolist()))}
        for j,name in enumerate(('Overall',*TYPES))}

def train_r2(fold,store,predictor):
    folder,c=config(fold,'R2');complete=folder/'stage11b_summary.json'
    if complete.exists():return read_json(complete)
    assert not (folder/'last.pt').exists(),'resume requires explicit saved state, do not overwrite partial runs'
    seed(2022);m=ReliabilityHead().cuda();init=state_sha(m);optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    train=indices(fold,'InnerTrain');dev=indices(fold,'InnerDev');tr=data_r2(store,train);dv=data_r2(store,dev)
    zero=assess_r2(m,dv,store.types[dev]);atomic_json(folder/'stage11b_step0.json',zero)
    best=float('inf');bestep=None;bad=0;curve=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,51):
        begin=time.monotonic();m.train();gen=torch.Generator().manual_seed(2022+epoch);order=torch.randperm(len(train),generator=gen).cuda()
        total=0.;norms=[]
        for start in range(0,len(order),1024):
            ix=order[start:start+1024];out=m(tr['features_R2'][ix],tr['base_logits'][ix]);loss=ranking_loss(out['mode_logits'],tr['FDE_by_mode'][ix])
            assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();norms.append(gradients(m,predictor));optimizer.step();total+=float(loss.detach())*len(ix)
        metrics=assess_r2(m,dv,store.types[dev]);score=metrics['Overall']['Top1FDE'];better=score<best
        if better:best=score;bestep=epoch;bad=0
        else:bad+=1
        row={'Epoch':epoch,'TrainLoss':total/len(train),'InnerDevLoss':metrics['Overall']['SoftCE'],
            **{f'InnerDev{k}Top1FDE':metrics[k]['Top1FDE'] for k in ('Overall',*TYPES)},
            'DevSelectionScore':score,'GradientNorm':float(np.mean(norms)),'Selected':int(better),'PatienceCount':bad,'Seconds':time.monotonic()-begin}
        curve.append(row);dump(f'03_training/fold{fold}/R2/stage11b_training_curve.csv',curve)
        cp={'state_dict':cpu_state(m),'Fold':fold,'Model':'R2','Epoch':epoch,'Score':score,
            'selection':'original Stage6 Overall Top1FDE, now InnerDev42','initial_state_sha256':init,
            'config_sha256':sha256(folder/'stage11b_training_config.json'),'normalization_sha256':c['normalization_sha256'],'split_sha256':c['split_sha256']}
        if better:atomic_torch(cp_path(fold,'R2'),cp)
        done=bad>=5 or epoch==50
        atomic_torch(folder/'last.pt',{**cp,'optimizer_state':optimizer.state_dict(),'curve':curve,'bad':bad,'best_score':best,'best_epoch':bestep,'complete':done})
        print('EPOCH',fold,'R2',epoch,'SCORE',score,'BEST',bestep,'PATIENCE',bad,flush=True)
        if done:break
    saved=torch.load(cp_path(fold,'R2'),map_location='cpu',weights_only=False);m.load_state_dict(saved['state_dict']);metrics=assess_r2(m,dv,store.types[dev])
    assert metrics['Overall']['Top1FDE']==best and all(np.isfinite(metrics[t]['Top1FDE']) and metrics[t]['Top1FDE']>0 for t in ('Vehicle','Pedestrian'))
    result={'Status':'COMPLETE','Fold':fold,'Model':'R2','SelectedEpoch':bestep,'ExecutedEpochs':epoch,'CheckpointScore':best,
        'checkpoint_sha256':sha256(cp_path(fold,'R2')),'normalization_sha256':c['normalization_sha256'],'split_sha256':c['split_sha256'],
        'config_sha256':sha256(folder/'stage11b_training_config.json'),'initial_state_sha256':init,'DevMetrics':metrics,
        'Seconds':time.monotonic()-started,'MeanEpochSeconds':float(np.mean([r['Seconds'] for r in curve])),
        'PeakGPUMemoryBytes':torch.cuda.max_memory_allocated(),'Params':673,'R2OriginalProtocolPreserved':True,
        'HistoricalR2Loaded':False,'AllDistinctTrainingActorsUsed':len(train),'OuterTestUsed':False,'PredictorGradientCount':0}
    atomic_json(complete,result);m.eval().requires_grad_(False)
    del optimizer,tr,dv;torch.cuda.empty_cache();return result

@torch.no_grad()
def assess_graph(m,r2,store,dev,variant,refs):
    m.eval();r2.eval();f=frame().iloc[dev];sums=np.zeros((4,4),np.float64);counts=np.array([len(dev),*[(store.types[dev]==k).sum() for k in range(3)]])
    entropy=maximum=margin=oraprob=rawexpected=normalized=rawloss=routedloss=0.
    for start in range(0,len(dev),128):
        ix=dev[start:start+128];args,fd,ad=store.batch(ix);out=m(*args);rawloss+=float(objective(out['mode_logits'],fd,variant).double().sum())
        rz=r2(store.r2(ix),args[6]);bike=torch.as_tensor(store.types[ix]==2,device='cuda')
        z=out['mode_logits'].clone();p=out['mode_prob'].clone();z[bike]=rz['mode_logits'][bike];p[bike]=rz['mode_prob'][bike]
        assert torch.isfinite(z).all() and torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(len(ix),device='cuda'),atol=1e-6,rtol=0.)
        assert torch.equal(z[bike],rz['mode_logits'][bike]) and torch.equal(p[bike],rz['mode_prob'][bike])
        top=p.argmax(-1);best=fd.argmin(-1);ii=torch.arange(len(ix),device='cuda')
        val=torch.stack((fd[ii,top],ad[ii,top],(top==best).float(),objective(z,fd,'A')),-1).double().cpu().numpy()
        for g in range(4):
            keep=np.ones(len(ix),bool) if g==0 else store.types[ix]==g-1;sums[g]+=val[keep].sum(0)
        ep=-(p*p.clamp(min=1e-30).log()).sum(-1);ordered=p.sort(-1,descending=True).values
        entropy+=float(ep.double().sum());maximum+=float(ordered[:,0].double().sum());margin+=float((ordered[:,0]-ordered[:,1]).double().sum())
        oraprob+=float(p[ii,best].double().sum());cost=fd-fd.min(-1,keepdim=True).values;scale=cost.mean(-1).clamp(min=1)
        rawexpected+=float((p*cost).sum(-1).double().sum());normalized+=float((p*(cost/scale[:,None])).sum(-1).double().sum())
        routedloss+=float(objective(z,fd,variant).double().sum())
    metrics={name:{'Count':int(counts[j]),**dict(zip(('Top1FDE','Top1ADE','HitRate','SoftCE'),(sums[j]/counts[j]).tolist()))}
        for j,name in enumerate(('Overall',*TYPES))}
    score=.5*metrics['Vehicle']['Top1FDE']/refs['Vehicle']+.5*metrics['Pedestrian']['Top1FDE']/refs['Pedestrian']
    return {'InnerDevLoss':rawloss/len(dev),'InnerDevRoutedLoss':routedloss/len(dev),
        **{f'InnerDev{k}Top1FDE':metrics[k]['Top1FDE'] for k in ('Overall',*TYPES)},'DevSelectionScore':score,
        'ProbabilityEntropy':entropy/len(dev),'MaxProbability':maximum/len(dev),'Top1Top2Margin':margin/len(dev),
        'OracleModeProbability':oraprob/len(dev),'RawExpectedRegret':rawexpected/len(dev),'NormalizedExpectedRegret':normalized/len(dev)}

def train_graph(fold,variant,store,predictor,r2,refs):
    folder,c=config(fold,variant);complete=folder/'stage11b_summary.json'
    if complete.exists():return read_json(complete)
    assert not (folder/'last.pt').exists(),'preserve partial run; explicit resume needed'
    foldseed=2022+100*(fold-1);seed(foldseed);m=fresh(fold);init=state_sha(m)
    initrows=pd.read_csv(ROOT/'01_preflight/stage11b_initialization.csv');assert init==initrows[initrows.Fold==fold].iloc[0].StateSHA256
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);train=indices(fold,'InnerTrain');dev=indices(fold,'InnerDev')
    zero=assess_graph(m,r2,store,dev,variant,refs);atomic_json(folder/'stage11b_step0.json',zero)
    best=float('inf');bestep=None;bad=0;curve=[];pending=np.empty(0,np.int64);coverage=np.zeros(len(store.source),np.int32);orders=[]
    scales=np.maximum(1,(store.fde[store.source[train]]-store.fde[store.source[train]].min(-1,keepdims=True)).mean(-1))
    scale_summary={'ScaleMean':float(scales.mean()),'ScaleMin':float(scales.min()),'ScaleMedian':float(np.median(scales)),
        'ScaleP90':float(np.quantile(scales,.9)),'ScaleP99':float(np.quantile(scales,.99)),'ScaleMax':float(scales.max())}
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,51):
        begin=time.monotonic();order=np.concatenate((pending,np.random.default_rng(foldseed+epoch).permutation(train)))
        used=len(order)//1024*1024;pending=order[used:].copy();order_hash=hashlib.sha256(order.tobytes()).hexdigest();orders.append({'Epoch':epoch,'OrderSHA256':order_hash,'Used':used,'Pending':len(pending)})
        # Same epoch prefix must be bitwise identical in every completed earlier variant.
        for oldvariant in VARIANTS[:VARIANTS.index(variant)]:
            path=ROOT/f'03_training/fold{fold}/{oldvariant}/stage11b_batch_order.csv'
            previous=pd.read_csv(path)
            if epoch<=len(previous):assert previous.iloc[epoch-1].OrderSHA256==order_hash
        total=rawtotal=0.;norms=[];m.train()
        for pos in range(0,used,1024):
            batch=order[pos:pos+1024];optimizer.zero_grad(set_to_none=True)
            for offset in range(0,1024,128):
                ix=batch[offset:offset+128];args,fd,_=store.batch(ix);assert not any(a.requires_grad for a in args if a is not None)
                out=m(*args);value=objective(out['mode_logits'],fd,variant);loss=value.mean();assert torch.isfinite(loss)
                assert torch.isfinite(out['mode_prob']).all() and torch.allclose(out['mode_prob'].sum(-1),torch.ones(len(ix),device='cuda'),atol=1e-6,rtol=0.)
                (loss/8).backward();total+=float(value.detach().double().sum())
                if variant=='C':
                    cost=fd-fd.min(-1,keepdim=True).values;rawtotal+=float((out['mode_prob'].detach()*cost).sum(-1).double().sum())
            norms.append(gradients(m,predictor));assert not any(p.grad is not None or p.requires_grad for p in r2.parameters());optimizer.step();np.add.at(coverage,batch,1)
            if pos and pos%65536==0:print('TRAIN',fold,variant,epoch,pos,'/',used,flush=True)
        measured=assess_graph(m,r2,store,dev,variant,refs);score=measured['DevSelectionScore'];better=score<best
        if better:best=score;bestep=epoch;bad=0
        else:bad+=1
        row={'Epoch':epoch,'TrainLoss':total/used,**measured,'GradientNorm':float(np.mean(norms)),
            'Selected':int(better),'PatienceCount':bad,'OptimizerTargets':used,'CarryTargets':len(pending),
            'Seconds':time.monotonic()-begin,'OrderSHA256':order_hash}
        if variant=='C':row.update({'TrainRawExpectedRegret':rawtotal/used,'TrainNormalizedExpectedRegret':total/used,**scale_summary})
        curve.append(row);dump(f'03_training/fold{fold}/{variant}/stage11b_training_curve.csv',curve);dump(f'03_training/fold{fold}/{variant}/stage11b_batch_order.csv',orders)
        cp={'state_dict':cpu_state(m),'Fold':fold,'Model':variant,'Epoch':epoch,'Score':score,
            'selection':'fixed .5 Vehicle / .5 Pedestrian relative InnerDev Top1FDE','initial_state_sha256':init,
            'config_sha256':sha256(folder/'stage11b_training_config.json'),'normalization_sha256':c['normalization_sha256'],
            'split_sha256':c['split_sha256'],'R2_DevReferences':refs,'BicycleRoute':'frozen fold-specific R2'}
        if better:atomic_torch(cp_path(fold,variant),cp)
        done=bad>=5 or epoch==50
        atomic_torch(folder/'last.pt',{**cp,'optimizer_state':optimizer.state_dict(),'pending':pending,'curve':curve,'coverage':coverage,
            'bad':bad,'best_score':best,'best_epoch':bestep,'complete':done})
        print('EPOCH',fold,variant,epoch,'SCORE',score,'V',measured['InnerDevVehicleTop1FDE'],'P',measured['InnerDevPedestrianTop1FDE'],'BEST',bestep,'PATIENCE',bad,flush=True)
        if done:break
    assert (coverage[train]>0).all();saved=torch.load(cp_path(fold,variant),map_location='cpu',weights_only=False);m.load_state_dict(saved['state_dict'])
    reloaded=assess_graph(m,r2,store,dev,variant,refs);assert abs(reloaded['DevSelectionScore']-best)<1e-12
    result={'Status':'COMPLETE','Fold':fold,'Model':variant,'SelectedEpoch':bestep,'ExecutedEpochs':epoch,'CheckpointScore':best,
        'checkpoint_sha256':sha256(cp_path(fold,variant)),'normalization_sha256':c['normalization_sha256'],'split_sha256':c['split_sha256'],
        'config_sha256':sha256(folder/'stage11b_training_config.json'),'initial_state_sha256':init,'SelectedMetrics':reloaded,
        'Seconds':time.monotonic()-started,'MeanEpochSeconds':float(np.mean([r['Seconds'] for r in curve])),
        'PeakGPUMemoryBytes':torch.cuda.max_memory_allocated(),'Params':24066,'AllDistinctTrainingActorsUsed':len(train),
        'FinalPendingOccurrences':len(pending),'PendingSavedNotActorDropped':True,'OuterTestUsed':False,
        'PredictorGradientCount':0,'BicycleTargetsTrained':int((store.types[train]==2).sum()),'R2_DevReferences':refs,
        'AMP':False,'Precision':'FP32'}
    atomic_json(complete,result);del m,optimizer;torch.cuda.empty_cache();return result

def main():
    seed();verify();tiny=read_json(ROOT/'01_preflight/stage11b_tiny_audit.json');assert tiny['Status']=='PASS' and tiny['FormalCVPermitted']
    predictor=old.frozen_predictor('cpu');ps=state_sha(predictor);summaries=[];checkpoints=[]
    for fold in (1,2,3):
        frozen=ROOT/f'04_checkpoints/fold{fold}/stage11b_frozen.json';assert not frozen.exists(),'preserve frozen fold; do not retrain'
        store=Store(fold);r=train_r2(fold,store,predictor);summaries.append(r)
        refs={t:r['DevMetrics'][t]['Top1FDE'] for t in ('Vehicle','Pedestrian')};seed(2022);r2=ReliabilityHead().cuda()
        r2.load_state_dict(torch.load(cp_path(fold,'R2'),map_location='cpu',weights_only=False)['state_dict']);r2.eval().requires_grad_(False);rs=state_sha(r2)
        for variant in VARIANTS:summaries.append(train_graph(fold,variant,store,predictor,r2,refs))
        assert state_sha(r2)==rs and state_sha(predictor)==ps;verify()
        cp_rows=[]
        for result in summaries[-4:]:
            name=result['Model'];row={'Fold':fold,'Model':name,'Path':str(cp_path(fold,name).relative_to(PROJECT)),
                'SHA256':sha256(cp_path(fold,name)),'SelectedEpoch':result['SelectedEpoch'],'CheckpointScore':result['CheckpointScore'],
                'NormalizationSHA256':result['normalization_sha256'],'SplitSHA256':result['split_sha256'],'TrainingConfigSHA256':result['config_sha256']}
            checkpoints.append(row);cp_rows.append(row)
        atomic_json(frozen,{'Status':'FROZEN','Fold':fold,'Checkpoints':cp_rows,'OuterTestEvaluationPermitted':True,'FurtherTrainingPermitted':False})
        dump('03_training/stage11b_training_summary.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in summaries])
        dump('04_checkpoints/stage11b_checkpoint_manifest.csv',checkpoints)
        print('FOLD_ALL_FOUR_CHECKPOINTS_FROZEN',fold,flush=True);del r2,store;torch.cuda.empty_cache()
    verify(history=True);atomic_json(ROOT/'04_checkpoints/stage11b_all_frozen.json',{'Status':'FROZEN_ALL_COMPLETE','Folds':3,'Checkpoints':checkpoints,
        'predictor_state_sha256':ps,'TinyWeightsUsed':False,'NoOuterTestModelSelection':True,'HeadDevUsed':False,'OfficialVALTestUsed':False})
    print('ALL_THREE_CV_FOLDS_TRAINING_COMPLETE',flush=True)
if __name__=='__main__':main()
