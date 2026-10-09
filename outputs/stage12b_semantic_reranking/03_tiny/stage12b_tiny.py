"""Fixed128 Pedestrian targets, 300 updates per variant, no tuning rescue."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12b_common import *
from stage12b_model import SemanticResidual,objective
from stage6a_common import frozen_predictor
def grad_norm(m):
    values=[p.grad.detach().float().square().sum() for p in m.parameters() if p.grad is not None]
    assert values and all(torch.isfinite(v) for v in values)
    norm=float(torch.stack(values).sum().sqrt());assert norm>0;return norm
def main():
    seed();verify();audit=read_json(ROOT/'01_preflight/stage12b_preflight_audit.json');assert audit['Status']=='PASS_PRE_TINY'
    f=frame();train=indices(1,'InnerTrain',True);ids=np.array(sorted(train,key=lambda i:f.iloc[i].actor_id)[:128],np.int64)
    assert len(ids)==128;folder=ROOT/'03_tiny/cache';folder.mkdir(exist_ok=True);np.save(folder/'stage12b_tiny_indices.npy',ids)
    predictor=frozen_predictor('cpu');c,r2=frozen_models(1);frozen=[predictor,c,r2];before=[state_sha(m) for m in frozen]
    fd,ad=label_arrays();fd=torch.from_numpy(fd[ids]).cuda();base=np.load(CACHE/'fold1/stage12b_base_logits.npy',mmap_mode='r')
    base=torch.from_numpy(np.array(base[ids],copy=True)).cuda()/TEMPERATURES[1]
    initial=torch.load(ROOT/'01_preflight/cache/stage12b_fold1_initial.pt',map_location='cpu',weights_only=False)
    rows=[];curves=[]
    for variant in VARIANTS:
        seed(2022);m=SemanticResidual().cuda();m.load_state_dict(initial['state_dict']);x=variant_input(1,'InnerTrain',variant,ids)
        optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
        with torch.no_grad():begin=float(objective(m(x,base),fd,base)[0].mean())
        norms=[];started=time.monotonic()
        for update in range(1,301):
            optimizer.zero_grad(set_to_none=True)
            for micro in range(4):
                out=m(x,base);loss,parts=objective(out,fd,base);assert torch.isfinite(loss).all();(loss.mean()/4).backward()
            norms.append(grad_norm(m));assert not any(p.requires_grad or p.grad is not None for old in frozen for p in old.parameters())
            optimizer.step()
            if update in [1,25,50,100,150,200,250,300]:
                with torch.no_grad():now=float(objective(m(x,base),fd,base)[0].mean())
                curves.append({'Variant':variant,'Update':update,'Loss':now,'InitialLoss':begin,'GradientNorm':norms[-1]})
        with torch.no_grad():out=m(x,base);end=float(objective(out,fd,base)[0].mean());maxdelta=float(out['delta'].abs().max())
        decrease=(begin-end)/begin;passed=np.isfinite(end) and decrease>=.05 and maxdelta>0 and min(norms)>0
        row={'Variant':variant,'Targets':128,'Updates':300,'InitialLoss':begin,'FinalLoss':end,'RelativeLossDecrease':decrease,
            'MinimumGradientNorm':min(norms),'ResidualMaxAbs':maxdelta,'FrozenGradients':0,'Params':641,
            'InitialStateSHA256':initial['StateSHA256'],'Seconds':time.monotonic()-started,'Status':'PASS' if passed else 'FAIL'}
        rows.append(row);print('TINY',variant,json.dumps(row),flush=True);del m,optimizer
    assert before==[state_sha(m) for m in frozen]
    dump('03_tiny/stage12b_tiny_results.csv',rows);dump('03_tiny/stage12b_tiny_curves.csv',curves)
    passed=all(r['Status']=='PASS' for r in rows)
    atomic_json(ROOT/'03_tiny/stage12b_tiny_audit.json',{'Status':'PASS' if passed else 'FAIL','TinyOverfit':'PASS' if passed else 'FAIL',
        'FrozenPredictorCAndR2StatesUnchanged':True,'FrozenGradients':0,'InitialWeightsOnly':'same per-fold state; tiny weights discarded',
        'FormalTrainingPermitted':passed,'TargetsSHA256':array_sha(ids),'LossDecreaseThreshold':.05,'NoHyperparameterAdjustment':True})
    audit.update(Status='PASS' if passed else 'FAIL',TinyOverfit='PASS' if passed else 'FAIL',FormalTrainingPermitted=passed)
    atomic_json(ROOT/'01_preflight/stage12b_preflight_audit.json',audit);verify();assert passed,'STOP: TinyOverfit failed; formal training prohibited'
    print('STAGE12B_TINY_ALL_PASS',flush=True)
if __name__=='__main__':main()
