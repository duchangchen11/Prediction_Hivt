"""After G1 fails, finish the other authorized tiny gates; never fit full heads."""
from stage8a1_common import *
from stage8a1_train import gradient

def main():
    seed();verify();assert not FROZEN.exists()
    assert read_json(ROOT/'01_training/G1/stage8a1_tiny_audit.json')['status']=='FAIL'
    registration=read_json(ROOT/'00_manifest/stage8a1_registration.json')
    for p,h in registration['new_sources_sha256'].items():assert sha256(Path(__file__).parent/p)==h,p
    sys.path.insert(0,str(S6/'00_manifest'));from stage6a_common import frozen_predictor
    predictor=frozen_predictor('cpu');predictorsha=state_sha(predictor)
    store=Store('train');train=np.flatnonzero(store.partition==1)
    ids=np.array(read_json(ROOT/'01_training/stage8a1_tiny_targets.json')['indices']);assert (store.partition[ids]==1).all()
    neutral=np.random.default_rng(2022).choice(train,1024,replace=False);audit=[]
    args,fde,_=store.batch(ids);q=(-fde).softmax(-1);entropy=float(-(q*q.log()).sum(-1).mean())
    with torch.no_grad():initial=float(loss(args[-1],fde))
    assert entropy<=initial+1e-6
    max_reduction=1-entropy/initial
    atomic_json(ROOT/'01_training/stage8a1_tiny_feasibility.json',{
        'status':'AUDITED','target_count':128,'soft_target_entropy_lower_bound':entropy,'initial_ranking_loss':initial,
        'required_final_loss_strictly_below':.8*initial,'maximum_mathematically_possible_relative_reduction':max_reduction,
        'twenty_percent_gate_feasible':bool(entropy<.8*initial),'proof':'SoftCE(q,p)=H(q)+KL(q||p)>=H(q); common zero-residual initialization fixes initial loss for all variants',
        'sampling':'unchanged original uniform HeadTrain fixed128 seed2022; no failure-driven resampling','temperature_m':1.,
        'new_audit_source_sha256':sha256(Path(__file__))})
    for variant in PARAMS:
        model=fresh(variant);assert state_sha(model)==registration['initial_state_sha256'][variant]
        maxlog=maxprob=maxinstrument=0.
        for start in range(0,1024,128):
            aa,_,_=store.batch(neutral[start:start+128])
            with torch.no_grad():
                out=model(*aa);hooked,_=capture(model,aa)
                maxlog=max(maxlog,float((out['mode_logits']-aa[-1]).abs().max()));maxprob=max(maxprob,float((out['mode_prob']-aa[-1].softmax(-1)).abs().max()))
                maxinstrument=max(maxinstrument,float((out['mode_logits']-hooked['mode_logits']).abs().max()))
        assert maxlog<1e-6 and maxprob<1e-6 and maxinstrument==0
        audit.append({'Variant':variant,'Targets':1024,'LogitMaxDiff':maxlog,'ProbabilityMaxDiff':maxprob,'CandidateMaxDiff':0.,'CaptureMaxDiff':maxinstrument,'Status':'PASS'})
        if variant=='G1':del model;continue # this failed variant receives no further optimizer updates
        root=ROOT/'01_training'/variant;assert not (root/'stage8a1_tiny_audit.json').exists(),'Tiny cannot be repeated'
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);curve=[];grad=None
        model.train()
        for step in range(1,301):
            optimizer.zero_grad(set_to_none=True);out=model(*args);l=loss(out['mode_logits'],fde);assert torch.isfinite(l)
            l.backward();grad=gradient(model,predictor);optimizer.step()
            if step%10==0:curve.append({'Update':step,'Loss':float(l.detach())})
        model.eval()
        with torch.no_grad():out=model(*args);final=float(loss(out['mode_logits'],fde));delta=float(out['delta_logits'].abs().max())
        passed=final<.8*initial and delta>0 and bool(torch.isfinite(out['delta_logits']).all())
        write_csv(root/'stage8a1_tiny_curve.csv',curve);atomic_json(root/'gradient_audit.json',grad)
        atomic_json(root/'stage8a1_tiny_audit.json',{'status':'PASS' if passed else 'FAIL','initial_loss':initial,'final_loss':final,
            'relative_decrease':1-final/initial,'delta_abs_max':delta,'updates':300,'precision':'FP32','predictor_gradient_count':0,
            'candidate_requires_grad':False,'full_training_initialization':'no full training; tiny weights discarded',
            'source_sha256':sha256(Path(__file__))})
        print('TINY',variant,initial,final,'PASS' if passed else 'FAIL',flush=True)
        del model,optimizer;torch.cuda.empty_cache()
    write_csv(ROOT/'06_tables/stage8a1_neutral_identity.csv',audit)
    assert state_sha(predictor)==predictorsha;verify()
    result={v:read_json(ROOT/'01_training'/v/'stage8a1_tiny_audit.json') for v in PARAMS}
    atomic_json(ROOT/'01_training/stage8a1_stop_gate.json',{'status':'STOP_BEFORE_FORMAL_TRAINING','tiny':result,
        'reason':'at least one required tiny gate failed; fixed128 target soft-label entropy also makes20% decrease infeasible',
        'full_training_updates':0,'formal_checkpoints_created':0,'official_VAL_evaluated':False,'test_used':False,
        'predictor_gradient_count':0,'waiting_for_user_brain_AI_review':True})
    print('STOP_NO_FORMAL_TRAINING',max_reduction,flush=True)

if __name__=='__main__':main()
