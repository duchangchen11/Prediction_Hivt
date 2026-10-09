"""Original Stage11B tiny targets, budgets and gates; no formal run on failure."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_protocol'))
from stage14a_common import *
def gradients(model,predictor):
    gs=[p.grad for p in model.parameters() if p.grad is not None]
    assert gs and all(torch.isfinite(g).all() for g in gs)
    norm=float(torch.stack([g.square().sum() for g in gs]).sum().sqrt());assert norm>0
    assert not predictor.training and not any(p.requires_grad or p.grad is not None for p in predictor.parameters())
    return norm

def main():
    seed();verify();assert read_json(ROOT/'01_preflight/stage14a_input_loss_integrity.json')['Status']=='PASS'
    dest=ROOT/'01_preflight/stage14a_tiny_audit.json';assert not dest.exists()
    old=read_json(S11B/'01_preflight/stage11b_tiny_targets.json');ids=np.array(old['local_indices'],np.int64)
    assert np.array_equal(ids,np.random.default_rng(2022).choice(indices(1,'InnerTrain'),128,replace=False))
    atomic_json(ROOT/'01_preflight/stage14a_tiny_targets.json',old)
    store=Store(1);args,fd,ad=store.batch(ids,part='InnerTrain')
    assert not fd.requires_grad and not any(a.requires_grad for a in args if a is not None)
    predictor=frozen_predictor('cpu');predictor_sha=state_sha(predictor);init=state_sha(fresh(1,'cpu'))
    entropy=-((-fd).softmax(-1)*(-fd).log_softmax(-1)).sum(-1).mean()
    candidate_before=sha256(S11A/'01_identity_audit/cache/candidates.npy');results=[]
    for variant in VARIANTS:
        seed(2022);m=fresh(1);assert state_sha(m)==init;m.eval()
        with torch.no_grad():
            out=m(*args);assert torch.equal(out['mode_logits'],args[6]);initial=float(objective(out['mode_logits'],fd,variant).mean())
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);curve=[];started=time.monotonic();m.train()
        for update in range(1,301):
            optimizer.zero_grad(set_to_none=True);out=m(*args);loss=objective(out['mode_logits'],fd,variant).mean()
            assert torch.isfinite(loss);loss.backward();norm=gradients(m,predictor);optimizer.step()
            assert torch.isfinite(out['mode_prob']).all() and torch.allclose(out['mode_prob'].sum(-1),torch.ones(128,device='cuda'),atol=1e-6,rtol=0)
            if update%10==0:curve.append(dict(Model='NG-'+variant,Update=update,Loss=float(loss.detach()),GradientNorm=norm))
        m.eval()
        with torch.no_grad():final=float(objective(m(*args)['mode_logits'],fd,variant).mean())
        reduction=1-(final-float(entropy))/(initial-float(entropy)) if variant=='A' else 1-final/initial
        threshold=.9 if variant=='A' else .8;passed=reduction>=threshold
        results.append(dict(Model='NG-'+variant,Status='PASS' if passed else 'FAIL',Updates=300,InitialLoss=initial,FinalLoss=final,
            EntropyFloor=float(entropy) if variant=='A' else 0,Reduction=reduction,Threshold=threshold,PredictorGradientCount=0,
            FinalGradientNorm=norm,InitialStateSHA256=init,FullTrainingUsesTinyWeights=False,Seconds=time.monotonic()-started))
        dump(f'01_preflight/stage14a_NG-{variant}_tiny_curve.csv',curve);dump('01_preflight/stage14a_tiny_results.csv',results)
        audit=dict(Status='PASS' if len(results)==2 and all(r['Status']=='PASS' for r in results) else 'FAIL' if not passed else 'RUNNING',
            Results=results,FormalTrainingPermitted=len(results)==2 and all(r['Status']=='PASS' for r in results),
            NoTinyCheckpointCreated=True,SameTargetsAndUpdatesAsStage11B=True,LossAndLabels='PASS',PredictorGradientCount=0,
            CandidateIdentity='PASS',CandidateCoordinateMaxDiff=0,FailureAction='STOP_WITHOUT_TUNING_OR_FORMAL_TRAINING')
        atomic_json(dest,audit);print('TINY',variant,results[-1],flush=True)
        assert state_sha(predictor)==predictor_sha and sha256(S11A/'01_identity_audit/cache/candidates.npy')==candidate_before
        del m,optimizer;torch.cuda.empty_cache()
        if not passed:verify(history=True);print('STOP_STAGE14A_TINY_FAILED',variant,flush=True);return
    verify(history=True);print('STAGE14A_TINY_BOTH_PASS',flush=True)

if __name__=='__main__':main()
