"""No optimizer: verify half-macro gradient on T3's exact first formal batch."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage9a_common import *

def main():
    seed();verify();assert not FROZEN.exists();store=Store();train=np.flatnonzero(store.partition==1)
    ids=np.random.default_rng(2023).permutation(train)[:1024];counts=np.bincount(store.types[ids],minlength=3);present=int((counts>0).sum())
    m=fresh('T3');assert state_sha(m)==read_json(REG)['initial_state_sha256']['T3'];total=[torch.zeros_like(p) for p in m.parameters()]
    for start in range(0,1024,128):
        ix=ids[start:start+128];args,fde,_=store.batch(ix);out=m(*args);ty=torch.as_tensor(store.types[ix],device='cuda')
        per=per_actor_loss(out['mode_logits'],fde);weight=torch.as_tensor(np.divide(.5,counts*present,out=np.zeros(3),where=counts>0),dtype=per.dtype,device='cuda')[ty]
        grad=torch.autograd.grad((per*weight).sum(),tuple(m.parameters()),allow_unused=True)
        for dst,g in zip(total,grad):
            if g is not None:dst.add_(g)
    assert all(torch.isfinite(g).all() for g in total);norm=float(torch.stack([g.square().sum() for g in total]).sum().sqrt());assert norm>0
    atomic_json(ROOT/'05_diagnostics/stage9a_macro_gradient_audit.json',{'status':'PASS','Variant':'T3','batch_targets':1024,'type_counts':counts.tolist(),'half_macro_gradient_norm':norm,'finite_nonzero':True,'weights':'.5/(present_type_count * effective_batch_type_count)','optimizer_updates':0,'exact_T3_first_formal_batch':True,'ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'HeadTrain_only':True,'official_VAL_used':False})
    print('FORMAL_MACRO_GRADIENT_PASS',norm,flush=True)

if __name__=='__main__':main()
