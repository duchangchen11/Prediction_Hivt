"""Unified residual OuterTest evaluation, permitted only after all9 freeze."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12b_common import *
from stage12b_model import SemanticResidual
FIELDS=('Top1FDE','Top1ADE','OracleGap','HitRate','minFDE6','minADEOracle6','MR6','ExpectedRegret','PredictionEntropy')

def summarize(f,values,walk,fold=None):
    rows=[]
    for group,mask in groups(f,walk).items():
        if not mask.any():continue
        for j,name in enumerate(MODELS):rows.append(dict(Group=group,Model=name,Count=int(mask.sum()),**({'Fold':fold} if fold else {}),**dict(zip(FIELDS,values[mask,j].mean(0)))))
    return rows

@torch.no_grad()
def main():
    seed();verify();frozen=read_json(ROOT/'05_checkpoints/stage12b_all9_frozen.json');assert frozen['Status']=='FROZEN_ALL9' and len(frozen['Checkpoints'])==9
    for path,h in frozen['Checkpoints'].items():assert sha256(ROOT/path)==h
    frozen_sha=sha256(ROOT/'05_checkpoints/stage12b_all9_frozen.json');f=frame().copy();n=len(f);dest=ROOT/'06_oof_evaluation/cache';dest.mkdir(exist_ok=True)
    filled=np.zeros(n,bool);foldids=np.zeros(n,np.int8)
    z=np.lib.format.open_memmap(dest/'stage12b_logits.npy',mode='w+',dtype=np.float32,shape=(n,4,6));p=np.lib.format.open_memmap(dest/'stage12b_probabilities.npy',mode='w+',dtype=np.float32,shape=(n,4,6))
    delta=np.lib.format.open_memmap(dest/'stage12b_residuals.npy',mode='w+',dtype=np.float32,shape=(n,3,6));delta[:]=0
    fd,ad=label_arrays();walk=np.load(CACHE/'stage12b_walkway.npy',mmap_mode='r');efficiency=[];foldrows=[];peak=[]
    before_candidate=sha256(S11A/'01_identity_audit/cache/candidates.npy')
    for fold in (1,2,3):
        ids=indices(fold,'OuterTest');ped=indices(fold,'OuterTest',True);assert not filled[ids].any()
        basez=np.load(CACHE/f'fold{fold}/stage12b_base_logits.npy',mmap_mode='r');basep=np.load(CACHE/f'fold{fold}/stage12b_base_probabilities.npy',mmap_mode='r')
        for j in range(4):z[ids,j]=basez[ids];p[ids,j]=basep[ids]
        for j,name in enumerate(VARIANTS,1):
            saved=torch.load(ROOT/f'05_checkpoints/fold{fold}/{name}_best.pt',map_location='cpu',weights_only=False);assert saved['Fold']==fold and saved['Model']==name
            m=SemanticResidual().cuda().eval().requires_grad_(False);m.load_state_dict(saved['state_dict']);before=state_sha(m)
            features=variant_input(fold,'OuterTest',name,ped);base=torch.from_numpy(np.array(basez[ped],copy=True)).cuda()/TEMPERATURES[fold]
            # Warm up only the already frozen residual; no checkpoint selection.
            for repeat in range(10):m(features[:128],base[:128])
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            for start in range(0,len(ped),128):
                ix=ped[start:start+128];out=m(features[start:start+128],base[start:start+128]);assert torch.isfinite(out['logits']).all()
                z[ix,j]=out['logits'].cpu().numpy();p[ix,j]=out['probabilities'].cpu().numpy();delta[ix,j-1]=out['delta'].cpu().numpy()
            torch.cuda.synchronize();elapsed=time.monotonic()-started
            assert state_sha(m)==before and abs(delta[ped,j-1]).max()<=2.
            # A synchronized device-only measurement separates host export overhead.
            torch.cuda.synchronize();beg=time.monotonic()
            for start in range(0,len(ped),128):m(features[start:start+128],base[start:start+128])
            torch.cuda.synchronize();device_elapsed=time.monotonic()-beg
            efficiency.append(dict(Fold=fold,Model=name,Parameters=641,PedestrianTargets=len(ped),ResidualForwardExportSeconds=elapsed,
                ResidualDeviceForwardSeconds=device_elapsed,ResidualDeviceMSPerPedestrian=1000*device_elapsed/len(ped),ResidualExportMSPerPedestrian=1000*elapsed/len(ped),
                EvalPeakGPUMemoryMB=torch.cuda.max_memory_allocated()/2**20,CPUMapMatchingSecondsThisStage=0,
                Scope='Cached features and base logits; GPU FP32 batches128; excludes frozen predictor/C/R2 and HD Map extraction'))
            del features,base,m;torch.cuda.empty_cache()
        filled[ids]=True;foldids[ids]=fold
        print('OUTERTEST_FOLD_DONE',fold,len(ids),len(ped),flush=True)
    assert filled.all() and np.all(foldids>0);assert sha256(ROOT/'05_checkpoints/stage12b_all9_frozen.json')==frozen_sha
    oldz=np.load(OOF/'stage11b_oof_predictions_logits.npy',mmap_mode='r');oldp=np.load(OOF/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    assert np.array_equal(z[:,0],oldz[:,4]) and np.array_equal(p[:,0],oldp[:,4])
    vehicle=f.agent_type_id.to_numpy()==0;bike=f.agent_type_id.to_numpy()==2
    for j in range(4):
        assert np.array_equal(z[vehicle,j],oldz[vehicle,4]) and np.array_equal(p[vehicle,j],oldp[vehicle,4])
        assert np.array_equal(z[bike,j],oldz[bike,1]) and np.array_equal(p[bike,j],oldp[bike,1])
    assert np.isfinite(z).all() and np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-6,rtol=0)
    top=p.argmax(-1);oracle=fd.argmin(-1);ar=np.arange(n);minfd=fd.min(-1);minad=ad.min(-1)
    values=np.empty((n,4,len(FIELDS)),np.float64)
    for j,name in enumerate(MODELS):
        chosen=fd[ar,top[:,j]];prob=np.asarray(p[:,j],np.float64)
        values[:,j]=np.stack([chosen,ad[ar,top[:,j]],chosen.astype(np.float64)-minfd,top[:,j]==oracle,minfd,minad,minfd>2,
            (prob*(fd-minfd[:,None])).sum(-1),-(prob*np.log(np.maximum(prob,1e-30))).sum(-1)],-1)
        assert np.array_equal(values[vehicle,j,0],values[vehicle,0,0]) and np.array_equal(values[bike,j,0],values[bike,0,0])
        assert np.array_equal(values[:,j,4:7],values[:,0,4:7])
    oldmetrics=np.load(OOF/'stage11b_oof_metrics.npy',mmap_mode='r');assert np.array_equal(values[:,0,0],oldmetrics[:,4,0])
    assert sha256(S11A/'01_identity_audit/cache/candidates.npy')==before_candidate
    np.save(dest/'stage12b_metrics.npy',values);np.save(dest/'stage12b_modes.npy',top)
    f['Fold']=foldids;f['Partition']='OuterTest';f.to_csv(dest/'stage12b_actor_records.csv',index=False)
    for fold in (1,2,3):
        ix=np.flatnonzero(foldids==fold);foldrows.extend(summarize(f.iloc[ix],values[ix],walk[ix],fold))
    rows=summarize(f,values,walk);dump('06_oof_evaluation/stage12b_fold_metrics.csv',foldrows)
    dump('06_oof_evaluation/stage12b_oof_metrics.csv',rows);dump('06_oof_evaluation/stage12b_pedestrian_groups.csv',[r for r in rows if r['Group'].startswith('Pedestrian')])
    probability=[];coverage=[];gm=groups(f,walk)
    for group,mask in gm.items():
        if not group.startswith('Pedestrian'):continue
        for j,name in enumerate(MODELS):
            pp=np.array(p[mask,j],np.float64);zz=np.array(z[mask,j],np.float64);zz-=zz.max(-1,keepdims=True);logp=zz-np.log(np.exp(zz).sum(-1,keepdims=True))
            target=oracle[mask];one=np.eye(6)[target];conf=pp.max(-1);hit=pp.argmax(-1)==target;bins=np.minimum((conf*10).astype(int),9)
            ece=sum(float((bins==b).mean())*abs(float(conf[bins==b].mean())-float(hit[bins==b].mean())) for b in range(10) if (bins==b).any())
            probability.append(dict(Group=group,Model=name,Count=int(mask.sum()),RawNLL=float(-logp[np.arange(len(pp)),target].mean()),
                Brier=float(((pp-one)**2).sum(-1).mean()),ECE10=ece,ExpectedRegret=float(values[mask,j,7].mean()),PredictionEntropy=float(values[mask,j,8].mean()),
                CalibrationClaim='none; original Stage11C calibration does not transfer'))
            coverage.append(dict(Group=group,Model=name,Count=int(mask.sum()),SelectedModeWalkwayValidFraction=float(walk[np.flatnonzero(mask),top[mask,j],1].mean()),
                ActorAnyCandidateCoverage=float((walk[mask,:,1]>0).any(-1).mean())))
    dump('08_diagnostics/stage12b_probability_diagnostics.csv',probability);dump('08_diagnostics/stage12b_selected_coverage.csv',coverage)
    training=pd.read_csv(ROOT/'05_checkpoints/stage12b_checkpoint_manifest.csv')
    for row in efficiency:
        tr=training[(training.Fold==row['Fold'])&(training.Model==row['Model'])].iloc[0];row.update(TrainingSeconds=float(tr.TrainingSeconds),TrainPeakGPUMemoryMB=float(tr.PeakGPUMemoryMB))
    dump('08_diagnostics/stage12b_efficiency.csv',efficiency)
    z.flush();p.flush();delta.flush()
    atomic_json(ROOT/'06_oof_evaluation/stage12b_oof_identity_audit.json',dict(Status='PASS',Actors=n,Scenes=630,Pedestrian=66145,Vehicle=191026,Bicycle=2980,
        FrozenModelIntegrity='PASS',SemanticCacheIntegrity='PASS',FoldIsolation='PASS',CandidateIdentity='PASS',CandidateMaxDiff=0,
        VehiclePreserved='YES',BicyclePreserved='YES',AllCandidateOracleMetricsBitwiseIdentical=True,RawC0OOFBitwiseIdentical=True,
        All9FrozenBeforeOuterTest=True,All9FreezeSHA256=frozen_sha,Models=list(MODELS),Fields=list(FIELDS),CandidateFileSHA256=before_candidate,
        CacheFiles={str(q.relative_to(ROOT)):sha256(q) for q in dest.iterdir() if q.is_file()},CompletedUTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    verify();print('STAGE12B_OOF_PASS',flush=True)
if __name__=='__main__':main()
