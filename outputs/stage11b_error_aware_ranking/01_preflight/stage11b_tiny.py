"""Mandatory 300-update gates; any scientific gate failure stops formal CV."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11b_common import *

def gradients(model,predictor):
    gs=[p.grad for p in model.parameters() if p.grad is not None]
    assert gs and all(torch.isfinite(g).all() for g in gs)
    norm=float(torch.stack([g.square().sum() for g in gs]).sum().sqrt());assert norm>0
    assert not predictor.training and not any(p.requires_grad or p.grad is not None for p in predictor.parameters())
    return norm

def main():
    seed();verify();assert read_json(CACHE/'prepared.json')['Status']=='PASS'
    dest=ROOT/'01_preflight/stage11b_tiny_audit.json';assert not dest.exists()
    store=Store(1);train=indices(1,'InnerTrain');ids=np.random.default_rng(2022).choice(train,128,replace=False)
    atomic_json(ROOT/'01_preflight/stage11b_tiny_targets.json',{'fold':1,'partition':'InnerTrain','local_indices':ids.tolist(),'seed':2022,'count':128})
    args,fd,ad=store.batch(ids,part='InnerTrain');assert not fd.requires_grad and not any(a.requires_grad for a in args if a is not None)
    predictor=old.frozen_predictor('cpu');predsha=state_sha(predictor);init=state_sha(fresh(1,'cpu'));entropy=-((-fd).softmax(-1)*(-fd).log_softmax(-1)).sum(-1).mean()
    # Independent closed forms, tie and detached-label checks on isolated logits.
    z=torch.tensor([[.1,.2,.3,.4,.5,.6],[1.,2.,3.,4.,5.,6.]],device='cuda',requires_grad=True)
    errors=torch.tensor([[1.,1.,2.,3.,4.,5.],[2.,3.,4.,5.,6.,7.]],device='cuda',requires_grad=True)
    costs=errors.detach()-errors.detach().min(-1,keepdim=True).values;scale=costs.mean(-1).clamp(min=1)
    expected={'A':-((-errors.detach()).softmax(-1)*z.log_softmax(-1)).sum(-1),
        'B':-z.log_softmax(-1)[:,0],'C':(z.softmax(-1)*(costs/scale[:,None])).sum(-1)}
    for variant in VARIANTS:
        v=objective(z,errors,variant);assert torch.equal(v,expected[variant]);assert v.requires_grad
        torch.autograd.grad(v.sum(),z,retain_graph=True)
        assert torch.autograd.grad(v.sum(),errors,allow_unused=True,retain_graph=True)[0] is None
    results=[]
    for variant in VARIANTS:
        seed(2022);m=fresh(1);assert state_sha(m)==init;m.eval()
        with torch.no_grad():
            out=m(*args);assert torch.equal(out['mode_logits'],args[6])
            initial=float(objective(out['mode_logits'],fd,variant).mean())
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);curve=[];norm=0.;started=time.monotonic();m.train()
        for update in range(1,301):
            optimizer.zero_grad(set_to_none=True);out=m(*args);loss=objective(out['mode_logits'],fd,variant).mean()
            assert torch.isfinite(loss);loss.backward();norm=gradients(m,predictor);optimizer.step()
            assert torch.isfinite(out['mode_prob']).all() and torch.allclose(out['mode_prob'].sum(-1),torch.ones(128,device='cuda'),atol=1e-6,rtol=0.)
            if update%10==0:curve.append({'Variant':variant,'Update':update,'Loss':float(loss.detach()),'GradientNorm':norm})
        m.eval()
        with torch.no_grad():final=float(objective(m(*args)['mode_logits'],fd,variant).mean())
        if variant=='A':reduction=1-(final-float(entropy))/(initial-float(entropy));threshold=.9
        else:reduction=1-final/initial;threshold=.8
        passed=reduction>=threshold;results.append({'Variant':variant,'Status':'PASS' if passed else 'FAIL','Updates':300,
            'InitialLoss':initial,'FinalLoss':final,'EntropyFloor':float(entropy) if variant=='A' else 0.,
            'Reduction':reduction,'Threshold':threshold,'PredictorGradientCount':0,'FinalGradientNorm':norm,
            'InitialStateSHA256':init,'FullTrainingUsesTinyWeights':False,'Seconds':time.monotonic()-started})
        dump(f'01_preflight/stage11b_{variant}_tiny_curve.csv',curve)
        atomic_json(dest,{'Status':'PASS' if len(results)==3 and all(r['Status']=='PASS' for r in results) else 'FAIL' if not passed else 'RUNNING',
            'results':results,'LossImplementation':'PASS','CandidateIdentity':'PASS','GTLabelsDetached':True,
            'FormalCVPermitted':len(results)==3 and all(r['Status']=='PASS' for r in results)})
        print('TINY',variant,results[-1],flush=True);del m,optimizer;torch.cuda.empty_cache()
        if not passed:
            assert state_sha(predictor)==predsha;verify();print('STOP_TINY_FAILED_NO_FORMAL_CV',variant,flush=True);return
    assert state_sha(predictor)==predsha;verify();dump('01_preflight/stage11b_tiny_results.csv',results)
    print('ALL_THREE_TINY_GATES_PASS',flush=True)
if __name__=='__main__':main()
