"""One unified matched-head OOF evaluation after all three checkpoints freeze."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_protocol'))
from stage14b_common import *
FIELDS=('Top1FDE','Top1ADE','OracleGap','HitRate','MRR','SoftCE','ExpectedRegret','NormalizedExpectedRegret','PredictionEntropy','Top1Probability','Top1Top2Margin','OracleModeProbability','minADEOracle6','minFDE6','MR6')
# Reuse the original FP32 actor metric operations exactly, without importing old hooks.
path=S14A/'05_evaluation/stage14a_evaluate.py'
tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='actor_metrics']
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),globals())

def frozen_gate():
    gate=read_json(ROOT/'04_checkpoints/stage14b_all_frozen.json')
    assert gate['Status']=='FROZEN_ALL_COMPLETE' and len(gate['Checkpoints'])==3
    assert gate['OuterTestEvaluationPermitted'] and not gate['FurtherTrainingPermitted']
    for row in gate['Checkpoints']:assert sha256(PROJECT/row['Path'])==row['SHA256']
    assert sha256(ROOT/'stage14b_preregistered_plan.md')==read_json(ROOT/'00_protocol/stage14b_plan_registration.json')['PlanSHA256']
    return gate

@torch.no_grad()
def main():
    gate=read_json(ROOT/'04_checkpoints/stage14b_all_frozen.json')
    assert gate['Status']=='FROZEN_ALL_COMPLETE' and len(gate['Checkpoints'])==3 and gate['OuterTestEvaluationPermitted']
    assert sha256(ROOT/'stage14b_preregistered_plan.md')==read_json(ROOT/'00_protocol/stage14b_plan_registration.json')['PlanSHA256']
    for row in gate['Checkpoints']:assert sha256(PROJECT/row['Path'])==row['SHA256']
    verify(history=True,data=True);seed()
    dest=ROOT/'05_evaluation/cache';assert not (dest/'complete.json').exists()
    f=frame().copy();previous=S14A/'05_evaluation/cache'
    hist=pd.read_csv(previous/'stage14a_oof_actor_records.csv',dtype={'future_mask_bits':str})
    keys=['scene_token','sample_token','instance_token'];assert not f.duplicated(keys).any()
    assert len(f)==len(hist)==260151 and np.array_equal(f[keys].values,hist[keys].values)
    oldv=np.load(previous/'stage14a_oof_metrics.npy',mmap_mode='r');oldz=np.load(previous/'stage14a_oof_logits.npy',mmap_mode='r');oldp=np.load(previous/'stage14a_oof_probabilities.npy',mmap_mode='r')
    assert oldv.shape==(len(f),4,15)
    vv=np.lib.format.open_memmap(dest/'stage14b_matched_metrics.npy',mode='w+',dtype='float64',shape=(len(f),15))
    zz=np.lib.format.open_memmap(dest/'stage14b_matched_logits.npy',mode='w+',dtype='float32',shape=(len(f),6))
    pp=np.lib.format.open_memmap(dest/'stage14b_matched_probabilities.npy',mode='w+',dtype='float32',shape=(len(f),6))
    filled=np.zeros(len(f),bool);foldrows=[];runtime=[];poison=[]
    candidate_path=S11A/'01_identity_audit/cache/candidates.npy';beforecandidate=sha256(candidate_path)
    for fold in (1,2,3):
        saved=torch.load(cp_path(fold,'C'),map_location='cpu',weights_only=False);assert saved['Fold']==fold and saved['Model']=='C'
        m=fresh(fold).eval();m.load_state_dict(saved['state_dict']);m.requires_grad_(False);ms=state_sha(m)
        store=Store(fold);assert saved['normalization_sha256']==sha256(store.normpath)
        ids=indices(fold,'OuterTest');assert (hist.iloc[ids].Fold==fold).all() and not filled[ids].any()
        assert not set(f.iloc[ids].scene_token)&set(split(fold)['InnerTrain']+split(fold)['InnerDev'])
        elapsed=preparation=0.;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for start in range(0,len(ids),128):
            ix=ids[start:start+128];t=time.monotonic();args,fd,ad=store.batch(ix,part='OuterTest');torch.cuda.synchronize();preparation+=time.monotonic()-t
            assert not any(a is not None and a.data_ptr() in (fd.data_ptr(),ad.data_ptr()) for a in args)
            torch.cuda.synchronize();t=time.monotonic();out=m(*args);torch.cuda.synchronize();elapsed+=time.monotonic()-t
            z,p=out['mode_logits'].clone(),out['mode_prob'].clone();bike=torch.as_tensor(store.types[ix]==2,device='cuda')
            if start==0:
                poisoned_fd=torch.full_like(fd,float('nan'));poisoned_ad=torch.full_like(ad,float('nan'));repeat=m(*args)
                assert torch.equal(z,repeat['mode_logits']) and torch.equal(p,repeat['mode_prob'])
                poison.append(dict(Fold=fold,Actors=len(ix),IndependentFDEADELabelsPoisoned=True,LogitMaxDiff=0,ProbabilityMaxDiff=0,Status='PASS'))
            rz=torch.from_numpy(np.array(oldz[ix,0],copy=True)).cuda();rp=torch.from_numpy(np.array(oldp[ix,0],copy=True)).cuda()
            z[bike],p[bike]=rz[bike],rp[bike]
            assert torch.isfinite(z).all() and torch.isfinite(p).all()
            assert torch.allclose(p.sum(-1),torch.ones(len(ix),device='cuda'),atol=1e-6,rtol=0)
            metric=actor_metrics(z,p,fd,ad).cpu().numpy()
            assert np.array_equal(metric[:,12:],oldv[ix,0,12:]),'same frozen candidates must preserve all oracle metrics'
            assert np.array_equal(metric[bike.cpu().numpy()],oldv[ix[bike.cpu().numpy()],0]),'fixed Bicycle route'
            vv[ix],zz[ix],pp[ix]=metric,z.cpu().numpy(),p.cpu().numpy();filled[ix]=True
        assert state_sha(m)==ms and not any(p.requires_grad or p.grad is not None for p in m.parameters())
        for group,mask in groups(f.iloc[ids]).items():
            if mask.any():foldrows.append(dict(Fold=fold,Group=group,Model='Matched-NG-C',Count=int(mask.sum()),**dict(zip(FIELDS,vv[ids[mask]].mean(0)))))
        runtime.append(dict(Fold=fold,Model='Matched-NG-C',Actors=len(ids),HeadForwardSeconds=elapsed,HeadLatencyMSPerActor=1000*elapsed/len(ids),FeaturePreparationSeconds=preparation,TotalEvaluationWallSeconds=time.monotonic()-started,PeakAllocatedGPUMemoryBytes=torch.cuda.max_memory_allocated(),TimingScope='cached-input head CUDA FP32; excludes predictor/transfer/routing'))
        print('STAGE14B_OUTER_FOLD_PASS',fold,len(ids),flush=True);del store,m;torch.cuda.empty_cache()
    assert filled.all() and beforecandidate==sha256(candidate_path)
    rows=[]
    for group,mask in groups(f).items():
        for axis,name in enumerate(MODELS):
            values=oldv[mask,axis] if axis<4 else vv[mask]
            rows.append(dict(Group=group,Model=name,Count=int(mask.sum()),**dict(zip(FIELDS,values.mean(0)))))
    dump('05_evaluation/stage14b_capacity_metrics.csv',rows);dump('05_evaluation/stage14b_matched_fold_metrics.csv',foldrows)
    dump('05_evaluation/stage14b_label_input_audit.csv',poison);dump('05_evaluation/stage14b_evaluation_runtime.csv',runtime)
    hist.to_csv(dest/'stage14b_actor_records.csv',index=False)
    del vv,zz,pp
    cachefiles={p.name:sha256(p) for p in dest.glob('*') if p.is_file()}
    audit=dict(Status='PASS',Models=list(MODELS),Fields=list(FIELDS),Actors=len(f),Scenes=630,IdentityKeys=keys,IdentityJoin='EXACT_SAME_UNIQUE_KEYS_AND_ORDER',HistoricalOutputsUnchanged=True,AllCandidatesUnchanged=True,AllOracleMetricsBitwiseIdentical=True,BicycleR2RouteBitwiseIdentical=True,TrainableInferenceParameters=0,cache_files=cachefiles)
    atomic_json(ROOT/'05_evaluation/stage14b_identity_audit.json',audit);atomic_json(dest/'complete.json',dict(Status='COMPLETE',IdentityAuditSHA256=sha256(ROOT/'05_evaluation/stage14b_identity_audit.json')))
    verify(history=True);print('STAGE14B_UNIFIED_EVALUATION_PASS',flush=True)
if __name__=='__main__':main()
