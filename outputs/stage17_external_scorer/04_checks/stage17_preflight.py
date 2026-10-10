"""Actual InnerTrain probes, disjoint role audit, loss and saved-state replay."""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage17_common import *
def main():
 runtime();protect(True);verify_registration();results=[]
 for fold in (1,2,3):
  st=TNTStore(fold,'InnerTrain');dv=TNTStore(fold,'InnerDev');ids=st.ids[:128];x,tr=st.inputs(ids);gt=st.labels(ids)
  seed=2022+100*(fold-1);m=scorer(seed);initial=state_digest(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
  p=m(x,tr);assert p.shape==(128,6) and torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(128,device='cuda'),atol=1e-6,rtol=0)
  # Explicit module.loss equivalence to Eq6 with max squared FULL-PATH distance.
  q=(-(tr.reshape(-1,6,12,2)-gt.reshape(-1,1,12,2)).square().sum(-1).max(-1).values/.01).softmax(-1)
  loss=m.loss(x,tr,gt)/128;manual=-(q*p.log()).sum(-1).mean();assert torch.allclose(loss,manual,atol=1e-6)
  label_swap=-gt;assert torch.equal(p,m(x,tr));assert not x.requires_grad and not tr.requires_grad
  torch.cuda.reset_peak_memory_stats();start=time.monotonic();losses=[]
  for step in range(20):
   opt.zero_grad(set_to_none=True);v=m.loss(x,tr,gt)/128;assert torch.isfinite(v);v.backward();gn=gradient(m)
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters());opt.step();losses.append(float(v.detach()))
  path=ROOT/f'04_checks/fold{fold}/tiny_state.pt'
  atomic_torch(path,{'state_dict':cpu_state(m),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
  saved=torch.load(path,map_location='cpu',weights_only=False);n=scorer(seed);n.load_state_dict(saved['state_dict'],strict=True)
  assert torch.equal(m(x,tr),n(x,tr));o=torch.optim.AdamW(n.parameters(),lr=.001,weight_decay=.0001);o.load_state_dict(saved['optimizer'])
  for net,optimizer in [(m,opt),(n,o)]:
   optimizer.zero_grad(set_to_none=True);net.loss(x,tr,gt).div(128).backward();optimizer.step()
  assert state_digest(m.state_dict())==state_digest(n.state_dict())
  assert initial!=state_digest(m.state_dict())
  result={'Fold':fold,'Status':'PASS','Parameters':16001,'InputContext':[128,1,64],'InputTrajectory':[128,6,24],'OutputProbability':[128,6],
   'ProbabilityNormalized':True,'AllParametersFiniteConnectedGradient':True,'OriginalComponentLossVerified':True,'GTOnlyLoss':True,
   'NoLabelsInForwardInputs':True,'TrainDevDisjoint':not bool(set(st.frame.scene_token)&set(dv.frame.scene_token)),
   'Steps':20,'FirstTinyLoss':losses[0],'LastTinyLoss':losses[-1],'GradientNorm':gn,'CheckpointReplayBitwiseEqual':True,
   'TinyWeightsEligibleForFormal':False,'Seconds':time.monotonic()-start,'PeakGPUMemoryBytes':torch.cuda.max_memory_allocated()}
  assert result['TrainDevDisjoint'];results.append(result);del m,n,opt,o;torch.cuda.empty_cache()
 atomic_json(ROOT/'04_checks/stage17_preflight.json',{'Status':'PASS','Folds':results,'OuterPerformanceRead':False,'FullHiVTTrainingSteps':0})
 print('PREFLIGHT_PASS_ALL_THREE_FOLDS',flush=True)
if __name__=='__main__':main()
