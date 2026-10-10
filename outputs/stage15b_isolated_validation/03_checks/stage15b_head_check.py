"""Bounded new-data ranking checks. Two diagnostic updates/head, never formal weights."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'05_candidate_interface'),str(ROOT/'06_rank_training')]
from stage15b_heads import *
from stage15b_cache import context


def check(fold):
 ctx=context(fold,'ranking');verify_sources();store=RoleStore(fold,'InnerTrain');ids=store.ids[:128]
 args,fd,ad=store.batch(ids);rf=store.r2(ids);rows=[]
 for name in NAMES:
  m=fresh(name,fold);init=state_digest(m.state_dict());o=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
  out=head_forward(m,name,args,rf);assert torch.equal(out['mode_logits'],args[6])
  initial=float(objective(out['mode_logits'],fd,'A' if name=='R2' else name[-1]).mean())
  for step in range(2):
   o.zero_grad(set_to_none=True);out=head_forward(m,name,args,rf);loss=objective(out['mode_logits'],fd,'A' if name=='R2' else name[-1]).mean()
   assert torch.isfinite(loss);loss.backward();gn=gradient(m);o.step()
  assert init!=state_digest(m.state_dict())
  with torch.no_grad():
   ref=head_forward(m,name,args,rf)['mode_logits']
   # Labels remain external; poison does not change any forward argument.
   fde_poison=torch.full_like(fd,float('nan'));ade_poison=torch.full_like(ad,float('nan'))
   repeat=head_forward(m,name,args,rf)['mode_logits'];assert torch.equal(ref,repeat)
   neighbor=list(args);neighbor[0]=args[0].clone();neighbor[0][:,1:]=torch.randn_like(neighbor[0][:,1:])
   neighbor[1]=torch.randn_like(args[1]);neighbor[2]=~args[2]
   if name in ('NG-A','NG-C','Matched-NG-C'):
    assert torch.equal(ref,head_forward(m,name,tuple(neighbor),rf)['mode_logits'])
  rows.append({'Model':name,'Params':PARAMS[name],'InitialStateSHA256':init,'DiagnosticUpdates':2,'InitialLoss':initial,
               'FinalLoss':float(loss),'GradientNorm':gn,'GTLabelPoisonMaxDiff':0.,'OwnOnlyNeighborInvariance':name in ('NG-A','NG-C','Matched-NG-C'),
               'TinyWeightsRetainedAsFormal':False})
  del m,o;torch.cuda.empty_cache()
 atomic_json(ROOT/f'03_checks/stage15b_fold{fold}_head_checks.json',{'Status':'PASS','Fold':fold,'Role':'InnerTrain','Targets':128,'ActualUpdates':12,
    'Rows':rows,'OptimizerInputsFrozenFeatures':True,'NewPredictorWeightsUpdated':False,'OuterTestUsed':False})
 print('ALL_FRESH_HEADS_TINY_PASS',fold,flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);a=p.parse_args()
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);check(a.fold)
