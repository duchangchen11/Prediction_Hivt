"""Three bounded component-only fits. Outer data denied until global freeze."""
from pathlib import Path
import sys,time,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage17_common import *
@torch.no_grad()
def assess(m,store,refs):
 m.eval();n=len(store.ids);fd=np.empty(n,np.float32);ad=fd.copy();top=np.empty(n,np.int64);ps=np.empty((n,6),np.float32);loss=0.
 for start in range(0,n,1024):
  ix=store.ids[start:start+1024];x,tr=store.inputs(ix);p=m(x,tr);gt=store.labels(ix);l=m.loss(x,tr,gt);assert torch.isfinite(l) and torch.isfinite(p).all()
  assert torch.allclose(p.sum(-1),torch.ones(len(ix),device='cuda'),atol=1e-6,rtol=0)
  mode=p.argmax(-1).cpu().numpy();top[ix]=mode;ps[ix]=p.cpu().numpy()
  fd[ix]=store.base.fde[ix,mode];ad[ix]=store.base.ade[ix,mode];loss+=float(l)
 metrics={k:{'Count':int(g.sum()),'Top1FDE':float(fd[g].astype(np.float64).mean()),'Top1ADE':float(ad[g].astype(np.float64).mean())} for k,g in groups(store.frame).items()}
 score=.5*metrics['Vehicle']['Top1FDE']/refs['Vehicle']+.5*metrics['Pedestrian']['Top1FDE']/refs['Pedestrian']
 return {'DevSelectionScore':score,'Loss':loss/n,'Metrics':metrics},(fd,ad,top,ps)
def train(fold):
 runtime();protect(True);reg=verify_registration();assert read_json(ROOT/'04_checks/stage17_preflight.json')['Status']=='PASS'
 assert read_json(ROOT/'00_manifest/stage17_pretraining_gate.json')['Status']=='PASS'
 tr=TNTStore(fold,'InnerTrain');dv=TNTStore(fold,'InnerDev');seed=2022+100*(fold-1)
 folder=ROOT/f'05_training/fold{fold}';summary=folder/'stage17_summary.json'
 if summary.exists():assert sha256(cp_path(fold))==read_json(summary)['CheckpointSHA256'];return
 assert not (folder/'last.pt').exists(),'Incomplete fit retained; stop for review'
 m=scorer(seed);initial=state_digest(m.state_dict());opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
 refs={t:read_json(OLD/f'06_rank_training/fold{fold}/R2/stage15b_summary.json')['DevMetrics'][t]['Top1FDE'] for t in ('Vehicle','Pedestrian')}
 cfg={'Fold':fold,'Seed':seed,'RegistrationSHA256':sha256(REG),'TrainManifestSHA256':sha256(tr.folder/'manifest.json'),
 'DevManifestSHA256':sha256(dv.folder/'manifest.json'),'Parameters':16001,'Optimizer':'AdamW','LR':.001,'WeightDecay':.0001,
 'EffectiveBatch':1024,'Microbatch':128,'Accumulation':8,'MaxEpochs':50,'Patience':5,'Temperature':.01,'ScoreTemperature':1.,
 'Loss':'pinned TrajScoreSelection.loss / effectivebatch; soft CE of max squared full-trajectory distance','R2DevReferences':refs,
 'BicycleRoute':'pure TNT for primary; same-fold frozen R2 sensitivity reported separately','FreshInitialization':True,'TinyWeightsLoaded':False}
 atomic_json(folder/'stage17_training_config.json',cfg);zero,_=assess(m,dv,refs);atomic_json(folder/'stage17_step0.json',zero)
 pending=np.empty(0,np.int64);coverage=np.zeros(len(tr.ids),np.int32);best=float('inf');bad=0;curve=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,51):
  t=time.monotonic();m.train();order=np.concatenate((pending,np.random.default_rng(seed+epoch).permutation(tr.ids)));used=len(order)//1024*1024;pending=order[used:].copy();total=0.;norms=[]
  for s in range(0,used,1024):
   ix=order[s:s+1024];opt.zero_grad(set_to_none=True)
   for off in range(0,1024,128):
    j=ix[off:off+128];x,trajectory=tr.inputs(j);gt=tr.labels(j);value=m.loss(x,trajectory,gt);assert torch.isfinite(value),'Nonfinite genuine component loss; STOP without changing loss'
    (value/1024).backward();total+=float(value.detach())
   norms.append(gradient(m));opt.step();np.add.at(coverage,ix,1)
  orderhash=hashlib.sha256(order.tobytes()).hexdigest()
  old=pd.read_csv(OLD/f'06_rank_training/fold{fold}/G-C/stage15b_batch_order.csv')
  if epoch<=len(old):assert old.iloc[epoch-1].OrderSHA256==orderhash
  result,_=assess(m,dv,refs);score=result['DevSelectionScore'];better=score<best
  if better:best=score;bestep=epoch;bad=0
  else:bad+=1
  curve.append({'Epoch':epoch,'TrainLoss':total/used,'DevLoss':result['Loss'],'DevSelectionScore':score,'Selected':int(better),'PatienceCount':bad,
   'GradientNorm':float(np.mean(norms)),'OptimizerTargets':used,'CarryTargets':len(pending),'OrderSHA256':orderhash,'Seconds':time.monotonic()-t,
   **{k+'Top1FDE':v['Top1FDE'] for k,v in result['Metrics'].items()}})
  dump(folder/'stage17_training_curve.csv',curve)
  cp={'state_dict':cpu_state(m),'Fold':fold,'Seed':seed,'Epoch':epoch,'Score':score,'initial_state_sha256':initial,
   'ConfigSHA256':sha256(folder/'stage17_training_config.json'),'RegistrationSHA256':sha256(REG),'PredictorCheckpointSHA256':tr.base.manifest['PredictorCheckpointSHA256']}
  if better:atomic_torch(cp_path(fold),cp)
  done=bad>=5 or epoch==50
  atomic_torch(folder/'last.pt',{**cp,'optimizer_state':opt.state_dict(),'pending':pending,'coverage':coverage,'curve':curve,'bad':bad,
   'best_epoch':bestep,'best_score':best,'complete':done,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
  print('TNT_FIT',fold,epoch,'DEV',score,'BEST',bestep,'BAD',bad,flush=True)
  if done:break
 assert (coverage>0).all();m.load_state_dict(torch.load(cp_path(fold),map_location='cpu',weights_only=False)['state_dict']);selected,_=assess(m,dv,refs)
 assert abs(selected['DevSelectionScore']-best)<1e-12
 atomic_json(summary,{'Status':'COMPLETE','Fold':fold,'Seed':seed,'SelectedEpoch':bestep,'ExecutedEpochs':epoch,'DevSelectionScore':best,
 'CheckpointSHA256':sha256(cp_path(fold)),'ConfigSHA256':sha256(folder/'stage17_training_config.json'),'InitialStateSHA256':initial,
 'PredictorCheckpointSHA256':tr.base.manifest['PredictorCheckpointSHA256'],'AllTrainTargetsUsed':int((coverage>0).sum()),'FinalPending':len(pending),
 'DevMetrics':selected['Metrics'],'Seconds':time.monotonic()-start,'PeakGPUMemoryBytes':torch.cuda.max_memory_allocated(),'OuterRead':False,'HiVTTrainingSteps':0})
 print('TNT_FOLD_COMPLETE',fold,flush=True)
def freeze():
 verify_registration();summaries=[read_json(ROOT/f'05_training/fold{k}/stage17_summary.json') for k in (1,2,3)]
 for k,s in enumerate(summaries,1):assert s['Status']=='COMPLETE' and s['CheckpointSHA256']==sha256(cp_path(k))
 atomic_json(ROOT/'05_training/stage17_all_frozen.json',{'Status':'FROZEN_ALL_COMPLETE','Checkpoints':summaries,'FrozenUTC':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'OuterEvaluationAllowed':True,'FurtherFittingAllowed':False})
 print('ALL_THREE_TNT_CHECKPOINTS_FROZEN',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int);p.add_argument('--freeze',action='store_true');a=p.parse_args();freeze() if a.freeze else train(a.fold)
