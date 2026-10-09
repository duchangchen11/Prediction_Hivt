"""Registered nine independent pedestrian heads; no residual OuterTest access."""
from pathlib import Path
import os,sys
os.environ['STAGE12B_PHASE']='train'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12b_common import *
from stage12b_model import SemanticResidual,objective
from stage6a_common import frozen_predictor

def dev_metrics(m,x,base,fd,ad):
    m.eval()
    with torch.no_grad():
        out=m(x,base);choice=out['logits'].argmax(-1);old=base.argmax(-1);ar=torch.arange(len(fd),device=fd.device)
        chosen=fd[ar,choice];oldfd=fd[ar,old];change=choice!=old;harm=(chosen-oldfd).clamp(min=0)
        d=out['delta'];p=out['probabilities']
        return dict(DevPedestrianTop1FDE=float(chosen.double().mean()),DevPedestrianTop1ADE=float(ad[ar,choice].double().mean()),
            DevPedestrianHitRate=float((choice==fd.argmin(-1)).double().mean()),DevPedestrianWrongSwitchCount=int((change&(harm>0)).sum()),
            DevPedestrianGrossHarm=float(harm.double().sum()),DevSelectionScore=float(chosen.double().mean()),
            PredictionEntropy=float(-(p*p.clamp(min=1e-30).log()).sum(-1).double().mean()),ResidualMean=float(d.double().mean()),
            ResidualStd=float(d.double().std(unbiased=False)),ResidualMaxAbs=float(d.abs().max()),ModeChangedRatio=float(change.double().mean()))

def main():
    seed();verify();pre=read_json(ROOT/'01_preflight/stage12b_preflight_audit.json');assert pre['FormalTrainingPermitted'] and pre['Status']=='PASS'
    feature=ROOT/'02_input_cache/stage12b_feature_manifest.json'
    for rel,h in {**read_json(feature)['Files'],**read_json(feature)['Normalizations']}.items():assert sha256(ROOT/rel)==h,rel
    config=read_json(PROTOCOL);config['SourceSHA256']={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'00_manifest/stage12b_model.py',ROOT/'00_manifest/stage12b_common.py']}
    config['ProtocolSHA256']=sha256(PROTOCOL);configpath=ROOT/'04_training/stage12b_training_config.json'
    if configpath.exists():assert read_json(configpath)==config
    else:atomic_json(configpath,config)
    fdall,adall=label_arrays();predictor=frozen_predictor('cpu');predictor_state=state_sha(predictor)
    allcurves=[];manifest=[];sequence=[]
    for fold in (1,2,3):
        foldseed=2022+100*(fold-1);train=indices(fold,'InnerTrain',True);dev=indices(fold,'InnerDev',True)
        baseall=np.load(CACHE/f'fold{fold}/stage12b_base_logits.npy',mmap_mode='r')
        bt=torch.from_numpy(np.array(baseall[train],copy=True)).cuda()/TEMPERATURES[fold];bd=torch.from_numpy(np.array(baseall[dev],copy=True)).cuda()/TEMPERATURES[fold]
        ft=torch.from_numpy(fdall[train]).cuda();fd=torch.from_numpy(fdall[dev]).cuda();ad=torch.from_numpy(adall[dev]).cuda()
        initial=torch.load(ROOT/f'01_preflight/cache/stage12b_fold{fold}_initial.pt',map_location='cpu',weights_only=False)
        c,r2=frozen_models(fold);oldstates=[state_sha(z) for z in [c,r2]]
        for name in VARIANTS:
            seed(foldseed);m=SemanticResidual().cuda();m.load_state_dict(initial['state_dict']);assert state_sha(m)==initial['StateSHA256']
            optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            assert sum(p.numel() for group in optimizer.param_groups for p in group['params'])==641
            xt=variant_input(fold,'InnerTrain',name,train);xd=variant_input(fold,'InnerDev',name,dev)
            assert all(not v.requires_grad for v in [xt,xd,ft,fd,ad,bt,bd]);assert len(train)>512
            folder=ROOT/f'04_training/fold{fold}';checkpoint=ROOT/f'05_checkpoints/fold{fold}/{name}_best.pt';resume=folder/f'cache/stage12b_{name}_last.pt'
            curves=[];pending=np.empty(0,np.int64);seen=np.zeros(len(train),bool);best=float('inf');bad=0;first=1;updates=0;total_time=0.
            if resume.exists():
                saved=torch.load(resume,map_location='cpu',weights_only=False);assert saved['ConfigSHA256']==sha256(configpath)
                m.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer']);curves=saved['curves'];pending=saved['pending'];seen=saved['seen'];best=saved['best'];bad=saved['bad'];first=saved['epoch']+1;updates=saved['updates'];total_time=saved['Seconds']
            torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            for epoch in range(first,51):
                if bad>=5:break
                order=np.random.default_rng(foldseed+epoch).permutation(len(train));joined=np.concatenate([pending,order]);used=len(joined)//512*512;work=joined[:used];pending=joined[used:]
                sums=np.zeros(4,np.float64);norms=[];m.train()
                for pos in range(0,used,512):
                    optimizer.zero_grad(set_to_none=True)
                    for micro in range(4):
                        ids=work[pos+micro*128:pos+(micro+1)*128];ii=torch.from_numpy(ids).cuda();out=m(xt[ii],bt[ii]);loss,parts=objective(out,ft[ii],bt[ii]);assert torch.isfinite(loss).all()
                        (loss.mean()/4).backward();sums+=np.r_[float(loss.detach().double().sum()),parts.detach().double().sum(0).cpu().numpy()];seen[ids]=True
                    norm=float(torch.stack([p.grad.detach().square().sum() for p in m.parameters()]).sum().sqrt());assert np.isfinite(norm) and norm>0
                    assert all(p.grad is None and not p.requires_grad for old in [predictor,c,r2] for p in old.parameters())
                    norms.append(norm);optimizer.step();updates+=1
                result=dev_metrics(m,xd,bd,fd,ad);selected=result['DevSelectionScore']<best
                row=dict(Fold=fold,Model=name,Epoch=epoch,TrainLoss=sums[0]/used,TrainRisk=sums[1]/used,TrainAnchorKL=sums[2]/used,
                    TrainResidualPenalty=sums[3]/used,GradientNorm=float(np.mean(norms)),OptimizerUpdates=updates,TrainingOccurrences=used,
                    PendingOccurrences=len(pending),UniqueTrainTargetsSeen=int(seen.sum()),BatchOrderSHA256=array_sha(train[work]),Selected=selected,**result)
                curves.append(row)
                if selected:
                    best=result['DevSelectionScore'];bad=0
                    atomic_torch(checkpoint,dict(state_dict=m.state_dict(),Fold=fold,Model=name,SelectedEpoch=epoch,SelectionScore=best,Params=641,
                        InitialStateSHA256=initial['StateSHA256'],ProtocolSHA256=sha256(PROTOCOL),ConfigSHA256=sha256(configpath),
                        NormalizationSHA256=sha256(ROOT/f'02_input_cache/stage12b_fold{fold}_normalization.json'),SplitSHA256=sha256(S11B/f'02_splits/stage11b_fold{fold}_split.json'),
                        BaseCSHA256=sha256(S11B/f'04_checkpoints/fold{fold}/C_best.pt'),Temperature=TEMPERATURES[fold],SemanticCacheSHA256=sha256(S12A/'02_semantic_cache/stage12a_semantic_cache_manifest.json'),
                        FeatureManifestSHA256=sha256(feature)))
                else:bad+=1
                elapsed=total_time+time.monotonic()-started
                atomic_torch(resume,dict(state_dict=m.state_dict(),optimizer=optimizer.state_dict(),curves=curves,pending=pending,seen=seen,best=best,bad=bad,
                    epoch=epoch,updates=updates,Seconds=elapsed,ConfigSHA256=sha256(configpath)))
                dump(f'04_training/fold{fold}/stage12b_{name}_curves.csv',curves)
                print('EPOCH',fold,name,epoch,'loss',round(row['TrainLoss'],6),'devFDE',round(best,6),'bad',bad,flush=True)
            assert seen.all(),'An InnerTrain pedestrian never used for optimization'
            assert state_sha(predictor)==predictor_state and [state_sha(z) for z in [c,r2]]==oldstates
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False);duration=total_time+time.monotonic()-started
            allcurves.extend(curves);sequence.append(f'Fold{fold} {name}')
            manifest.append(dict(Fold=fold,Model=name,Path=str(checkpoint.relative_to(ROOT)),SHA256=sha256(checkpoint),SelectedEpoch=saved['SelectedEpoch'],
                SelectionScore=saved['SelectionScore'],Params=641,TrainingEpochs=len(curves),OptimizerUpdates=updates,TrainingSeconds=duration,
                PeakGPUMemoryMB=torch.cuda.max_memory_allocated()/2**20,NormalizationSHA256=saved['NormalizationSHA256'],SplitSHA256=saved['SplitSHA256'],
                ConfigSHA256=saved['ConfigSHA256'],BaseCSHA256=saved['BaseCSHA256'],Temperature=saved['Temperature'],SemanticCacheSHA256=saved['SemanticCacheSHA256'],
                InitialStateSHA256=saved['InitialStateSHA256'],AllUniqueInnerTrainTargetsUsed=True,FrozenGradients=0))
            dump('04_training/stage12b_training_curves.csv',allcurves);dump('05_checkpoints/stage12b_checkpoint_manifest.csv',manifest)
            print('TRAINING_COMPLETE',fold,name,'selected',saved['SelectedEpoch'],'seconds',round(duration,2),flush=True)
            del m,optimizer,xt,xd;torch.cuda.empty_cache()
        del c,r2,bt,bd,ft,fd,ad;torch.cuda.empty_cache()
    assert len(manifest)==9 and sequence==config['Training']['Sequence'];verify()
    atomic_json(ROOT/'05_checkpoints/stage12b_all9_frozen.json',dict(Status='FROZEN_ALL9',TimestampUTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        Checkpoints={r['Path']:r['SHA256'] for r in manifest},Sequence=sequence,ProtocolSHA256=sha256(PROTOCOL),ConfigurationSHA256=sha256(configpath),
        NoResidualOuterTestReadDuringTraining=True,FrozenGradients=0,FrozenStatesUnchanged=True,CheckpointManifestSHA256=sha256(ROOT/'05_checkpoints/stage12b_checkpoint_manifest.csv')))
    print('STAGE12B_ALL9_FROZEN',flush=True)
if __name__=='__main__':main()
