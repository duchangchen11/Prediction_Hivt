"""TRAIN-only source/identity/GT-poison audit before initialization registration."""
from stage9a_common import *
import copy,pandas as pd

def main():
    seed();verify();assert not REG.exists(),'Registration immutable'
    source=S8/'01_training/cache';old=read_json(source/'manifest.json')
    for name in ('arg0.npy','arg1.npy','arg2.npy','fde.npy','ade.npy','partition.npy','identities.csv'):
        assert sha256(source/name)==old['files'][name],name
    store=Store();ids=pd.read_csv(source/'identities.csv');split=read_json(SPLIT)
    trainset=set(split['HeadTrain']);devset=set(split['HeadDev']);assert len(trainset)==630 and len(devset)==70 and not trainset&devset
    assert np.array_equal(np.array([int(s in trainset) for s in ids.scene_token]),store.partition)
    assert set(ids.scene_token)==trainset|devset
    key=lambda r:(r.scene_token,r.sample_token,r.instance_token)
    lookup={key(r):i for i,r in enumerate(ids.itertuples())}
    rec=next(r for r in read_json(S6/'01_cache/stage6a_cache_manifest.json')['batches'] if r['split']=='train')
    payload=torch.load(S6/rec['path'],map_location='cpu',weights_only=False);maximum=0.;r2=frozen_r2();r2norm=read_json(S6/'02_features/stage6a_normalization.json');checked=0
    for w in payload['windows']:
        targets=torch.where(w['target_mask']&w['future_mask'].all(-1))[0]
        if not len(targets):continue
        args,gate,context,_,_=graph_from_window(w,targets)
        poisoned=copy.deepcopy(w);poisoned['GT'].fill_(float('nan'));poisoned['future_mask']=~poisoned['future_mask'];poisoned['target_mask']=~poisoned['target_mask']
        poisoned['lane_semantic']=torch.full((1,9),float('nan'))
        aa,gg,cc,_,_=graph_from_window(poisoned,targets)
        assert all(torch.equal(x,y) for x,y in zip(args,aa)) and torch.equal(gate,gg) and torch.equal(context,cc)
        ix=np.array([lookup[(w['scene_token'],w['sample_token'],w['instance_tokens'][t])] for t in targets.tolist()])
        cached,_,_=store.batch(ix,'cpu');assert all(torch.equal(x,y) for x,y in zip(args,cached[:3])) and torch.equal(gate,cached[4])
        with torch.no_grad():
            feature,flags,_,_=observable_features(*[w[k].cuda() for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
            out=r2(r2normalize(feature,flags,r2norm,'R2')[targets],w['mode_logits'][targets].cuda())
            maximum=max(maximum,float((out['mode_logits'].cpu()-cached[3]).abs().max()))
        checked+=len(targets)
    assert maximum<1e-6,maximum
    # Exact macro objective and gradient, independent of microbatch partition.
    torch.manual_seed(2022);z=torch.randn(1024,6,requires_grad=True);fd=torch.rand(1024,6);ty=torch.arange(1024)%3
    p=per_actor_loss(z,fd);full=objective(p,ty,True)[0];g1=torch.autograd.grad(full,z,retain_graph=True)[0]
    counts=torch.bincount(ty,minlength=3);weight=.5/1024.+.5/(3*counts.float());accum=sum((p[s:s+128]*weight[ty[s:s+128]]).sum() for s in range(0,1024,128));g2=torch.autograd.grad(accum,z)[0]
    assert abs(float(full-accum))<1e-6 and float((g1-g2).abs().max())<1e-7
    models={v:fresh(v,'cpu') for v in VARIANTS};g1base=SparseGraphReranker('G1')
    for name in ('node_encoder','norm','head','interaction_encoder','interaction_attention'):
        baseline=getattr(g1base,name).state_dict()
        for model in models.values():assert all(torch.equal(x,getattr(model,name).state_dict()[k]) for k,x in baseline.items())
    assert all(sum(p.numel() for p in m.parameters())<40000 for m in models.values())
    for v in ('T2','T3'):
        assert all(torch.count_nonzero(a[-1].weight)==0 and torch.count_nonzero(a[-1].bias)==0 for a in models[v].adapters)
    atomic_json(ROOT/'01_training/stage9a_no_leakage_audit.json',{'status':'PASS','TRAIN_windows':len(payload['windows']),'TRAIN_targets_checked':checked,'GT_nan_future_target_mask_poison':True,'semantic_nan_ignored':True,'observable_inputs_bitwise_identical':True,'graph_gate_cache_identity':True,'fresh_R2_logit_cache_maxdiff':maximum,'effective_batch_macro_loss_equivalence':True,'effective_batch_macro_gradient_maxdiff':float((g1-g2).abs().max()),'HeadTrain630_HeadDev70_exact':True,'official_VAL_opened':False})
    atomic_json(REG,{'status':'REGISTERED_BEFORE_TINY_FORMAL_AND_VAL','base_commit':BASE,'protocol_sha256':sha256(ROOT/'00_manifest/stage9a_protocol.json'),'normalization_sha256':sha256(NORM),'split_sha256':sha256(SPLIT),'cache_manifest_sha256':sha256(ROOT/'01_training/cache/manifest.json'),'training_sources':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('stage9a_*.py'))},'initial_state_sha256':{v:state_sha(m) for v,m in models.items()},'parameters':PARAMS,'shared_G1_initialization_bitwise_identical':True,'type_adapter_up_zero_initialized':True,'final_score_zero_initialized':True,'Official_VAL_role':'development-reuse evidence','semantic_in_Stage9':False,'test_used':False})
    print('STAGE9_AUDIT_AND_REGISTRATION_PASS',PARAMS,flush=True)

if __name__=='__main__':main()
