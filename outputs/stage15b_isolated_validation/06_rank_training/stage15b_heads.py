"""Fresh six-head training on new isolated candidates, original frozen protocols."""
from pathlib import Path
import sys,os,time,json,hashlib,random,csv
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'05_candidate_interface')]
from stage15b_common import read_json,sha256,atomic_json,PROTOCOL,state_digest,seed_all,PROJECT
from stage15b_ranking import *
import pandas as pd
NAMES=('R2','NG-A','NG-C','G-A','G-C','Matched-NG-C');TYPES=('Vehicle','Pedestrian','Bicycle')
PARAMS={'R2':673,'NG-A':7425,'NG-C':7425,'G-A':24066,'G-C':24066,'Matched-NG-C':24001}


def atomic_torch(path,value):
 path=Path(path);assert path.resolve().is_relative_to(ROOT);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix('.pt.tmp');torch.save(value,tmp);os.replace(tmp,path)


def dump(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(path,index=False,float_format='%.12g')


def cp_path(fold,name):return ROOT/f'07_rank_checkpoints/fold{fold}/{name}_best.pt'

def cpu_state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}


def fresh(name,fold):
 seed=2022+100*(fold-1)
 if name=='R2':
  seed_all(2022);m=ReliabilityHead()
 else:
  seed_all(seed)
  m=NoGraphReranker(seed) if name.startswith('NG-') else MatchedNoGraphReranker(seed) if name=='Matched-NG-C' else SparseGraphReranker('G1',seed)
 assert sum(p.numel() for p in m.parameters())==PARAMS[name]
 return m.cuda()


def verify_sources():
 reg=read_json(ROOT/'00_manifest/stage15b_ranking_registration.json')
 for path,h in reg['sources'].items():assert sha256(ROOT/path)==h,'Head/cache source changed '+path
 assert sha256(PROTOCOL)==reg['protocol_sha256']
 for path,h in read_json(PROTOCOL)['source_sha256'].items():assert sha256(PROJECT/path)==h


class RoleStore:
 def __init__(self,fold,role):
  assert role in ('InnerTrain','InnerDev','OuterTest')
  if role=='OuterTest':assert read_json(ROOT/'07_rank_checkpoints/stage15b_all_frozen.json')['Status']=='FROZEN_ALL_COMPLETE'
  self.fold=fold;self.role=role;self.folder=ROOT/f'05_candidate_interface/cache/fold{fold}/{role}'
  self.manifest=read_json(self.folder/'manifest.json');self.normpath=ROOT/f'05_candidate_interface/stage15b_fold{fold}_normalization.json'
  self.norm=read_json(self.normpath);assert self.norm['fit_partition']=='InnerTrain' and self.norm['formal_eligible']
  self.frame=pd.read_csv(self.folder/'targets.csv');self.types=self.frame.agent_type_id.to_numpy();self.ids=np.arange(len(self.frame),dtype=np.int64)
  parts=read_json(PROTOCOL)['folds'][fold-1]['parts'];assert set(self.frame.scene_token)==set(parts[role]);assert (self.frame.role==role).all()
  cp=read_json(ROOT/'04_predictor_checkpoints/stage15b_all_frozen.json')['Checkpoints'][fold-1]
  assert self.manifest['PredictorCheckpointSHA256']==cp['sha256'] and (self.frame.predictor_checkpoint_sha256==cp['sha256']).all()
  assert sha256(self.folder/'targets.csv')==self.manifest['IdentityCSV']['sha256']
  self.arrays={k:np.load(self.folder/(k+'.npy'),mmap_mode='r') for k in self.manifest['Arrays']}
  self.fde=self.arrays['fde'];self.ade=self.arrays['ade'];self.base=self.arrays['arg6']
  for k,a in self.arrays.items():assert list(a.shape)==self.manifest['Arrays'][k]['shape']
 def raw(self,k,ids):
  ids=np.asarray(ids,dtype=np.int64);assert ids.ndim==1 and np.all((ids>=0)&(ids<len(self.ids)))
  return torch.from_numpy(np.array(self.arrays[k][ids],copy=True))
 def batch(self,ids,device='cuda'):
  node,edge,mask=graph_normalize(*(self.raw(k,ids) for k in ('arg0','arg1','arg2')),self.norm['graph'])
  args=(node.to(device),edge.to(device),mask.to(device),None,None,None,self.raw('arg6',ids).to(device))
  return args,self.raw('fde',ids).to(device),self.raw('ade',ids).to(device)
 def r2(self,ids,device='cuda'):
  return normalize_r2(self.raw('r2_raw',ids),self.raw('r2_flags',ids),self.norm['R2']).to(device)


def gradient(m):
 values=[p.grad for p in m.parameters() if p.grad is not None]
 assert values and all(torch.isfinite(v).all() for v in values),'Nonfinite/disconnected head gradient'
 return float(torch.sqrt(sum(v.detach().square().sum() for v in values)))


@torch.no_grad()
def assess(m,name,store,r2=None,refs=None):
 m.eval()
 if r2 is not None:r2.eval()
 n=len(store.ids);sums=np.zeros((4,4));counts=np.array([n,*[(store.types==k).sum() for k in range(3)]])
 entropy=normalized_loss=0.;out_fde=np.empty(n,np.float32);out_ade=np.empty(n,np.float32);out_top=np.empty(n,np.int64)
 for start in range(0,n,4096 if name=='R2' else 128):
  ix=store.ids[start:start+(4096 if name=='R2' else 128)]
  if name=='R2':
   fd=store.raw('fde',ix).cuda();ad=store.raw('ade',ix).cuda();base=store.raw('arg6',ix).cuda();out=m(store.r2(ix),base)
  else:
   args,fd,ad=store.batch(ix);out=m(*args)
  z=out['mode_logits'].clone();p=out['mode_prob'].clone()
  if name!='R2' and r2 is not None:
   rout=r2(store.r2(ix),args[6]);bike=torch.as_tensor(store.types[ix]==2,device='cuda')
   z[bike]=rout['mode_logits'][bike];p[bike]=rout['mode_prob'][bike]
   assert torch.equal(z[bike],rout['mode_logits'][bike])
  assert torch.isfinite(p).all() and torch.allclose(p.sum(-1),torch.ones(len(ix),device='cuda'),atol=1e-6,rtol=0)
  top=p.argmax(-1);best=fd.argmin(-1);rows=torch.arange(len(ix),device='cuda')
  v=torch.stack((fd[rows,top],ad[rows,top],(top==best).float(),objective(z,fd,'A')),-1).double().cpu().numpy()
  for g in range(4):
   keep=np.ones(len(ix),bool) if g==0 else store.types[ix]==g-1;sums[g]+=v[keep].sum(0)
  entropy+=float((-(p*p.clamp(min=1e-30).log()).sum(-1)).double().sum())
  variant='A' if name=='R2' else name[-1];normalized_loss+=float(objective(out['mode_logits'],fd,variant).double().sum())
  out_fde[ix]=v[:,0];out_ade[ix]=v[:,1];out_top[ix]=top.cpu().numpy()
 metrics={k:{'Count':int(counts[g]),**dict(zip(('Top1FDE','Top1ADE','HitRate','SoftCE'),(sums[g]/counts[g]).tolist()))} for g,k in enumerate(('Overall',*TYPES))}
 score=metrics['Overall']['Top1FDE'] if name=='R2' else .5*metrics['Vehicle']['Top1FDE']/refs['Vehicle']+.5*metrics['Pedestrian']['Top1FDE']/refs['Pedestrian']
 return {'Metrics':metrics,'DevSelectionScore':score,'InnerDevLoss':normalized_loss/n,'ProbabilityEntropy':entropy/n},(out_fde,out_ade,out_top)


def config(fold,name,tr,dv):
 folder=ROOT/f'06_rank_training/fold{fold}/{name}';folder.mkdir(parents=True,exist_ok=True)
 c={'Fold':fold,'Model':name,'seed':2022 if name=='R2' else 2022+100*(fold-1),'parameters':PARAMS[name],
    'ProtocolSHA256':sha256(PROTOCOL),'TrainingPartition':'InnerTrain378','SelectionPartition':'InnerDev42',
    'TrainingCacheManifestSHA256':sha256(tr.folder/'manifest.json'),'DevCacheManifestSHA256':sha256(dv.folder/'manifest.json'),
    'NormalizationSHA256':sha256(tr.normpath),'HeadRegistrationSHA256':sha256(ROOT/'00_manifest/stage15b_ranking_registration.json'),
    'PredictorCheckpointSHA256':tr.manifest['PredictorCheckpointSHA256'],'HistoricalWeightsLoaded':False,
    'R2Protocol':read_json(PROTOCOL)['ranking']['R2_protocol'],'GraphProtocol':{k:read_json(PROTOCOL)['ranking'][k] for k in
       ('optimizer','learning_rate','weight_decay','microbatch','accumulation','effective_batch','max_epochs','patience','carry','ABC_selection')}}
 atomic_json(folder/'stage15b_training_config.json',c);return folder,c


def train_head(fold,name,tr,dv,r2=None,refs=None):
 verify_sources();assert tr.role=='InnerTrain' and dv.role=='InnerDev'
 folder,c=config(fold,name,tr,dv);summarypath=folder/'stage15b_summary.json'
 if summarypath.exists():
  result=read_json(summarypath);assert sha256(cp_path(fold,name))==result['CheckpointSHA256'];return result
 assert not (folder/'last.pt').exists(),'Failed/partial head run is preserved; no automatic restart'
 m=fresh(name,fold);initial=state_digest(m.state_dict());optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
 zero,_=assess(m,name,dv,r2,refs);atomic_json(folder/'stage15b_step0.json',zero)
 best=float('inf');bestep=None;bad=0;curve=[];pending=np.empty(0,np.int64);coverage=np.zeros(len(tr.ids),np.int32);orders=[]
 # R2's original whole-partition tensor protocol is preserved.
 data=None
 if name=='R2':data={'x':tr.r2(tr.ids),'base':tr.raw('arg6',tr.ids).cuda(),'fd':tr.raw('fde',tr.ids).cuda()}
 started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for epoch in range(1,51):
  begin=time.monotonic();m.train();total=0.;norms=[]
  if name=='R2':
   order=torch.randperm(len(tr.ids),generator=torch.Generator().manual_seed(2022+epoch));order_np=order.numpy();used=len(order_np);pending=np.empty(0,np.int64)
   for start in range(0,used,1024):
    ix=order[start:start+1024].cuda();out=m(data['x'][ix],data['base'][ix]);loss=ranking_loss(out['mode_logits'],data['fd'][ix])
    assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();norms.append(gradient(m));optimizer.step();total+=float(loss.detach())*len(ix)
   coverage[order_np]+=1
  else:
   order_np=np.concatenate((pending,np.random.default_rng(2022+100*(fold-1)+epoch).permutation(tr.ids)))
   used=len(order_np)//1024*1024;pending=order_np[used:].copy()
   for pos in range(0,used,1024):
    batch=order_np[pos:pos+1024];optimizer.zero_grad(set_to_none=True)
    for offset in range(0,1024,128):
     ix=batch[offset:offset+128];args,fd,_=tr.batch(ix);assert not any(a.requires_grad for a in args if a is not None)
     out=m(*args);value=objective(out['mode_logits'],fd,name[-1]);loss=value.mean();assert torch.isfinite(loss)
     (loss/8).backward();total+=float(value.detach().double().sum())
    norms.append(gradient(m));assert not any(p.grad is not None or p.requires_grad for p in r2.parameters());optimizer.step();np.add.at(coverage,batch,1)
  order_hash=hashlib.sha256(order_np.tobytes()).hexdigest();orders.append({'Epoch':epoch,'OrderSHA256':order_hash,'Used':used,'Pending':len(pending)})
  if name!='R2':
   for earlier in NAMES[1:NAMES.index(name)]:
    oldpath=ROOT/f'06_rank_training/fold{fold}/{earlier}/stage15b_batch_order.csv'
    old=pd.read_csv(oldpath)
    if epoch<=len(old):assert old.iloc[epoch-1].OrderSHA256==order_hash
  measured,_=assess(m,name,dv,r2,refs);score=measured['DevSelectionScore'];better=score<best
  if better:best=score;bestep=epoch;bad=0
  else:bad+=1
  row={'Epoch':epoch,'TrainLoss':total/used,'InnerDevLoss':measured['InnerDevLoss'],
      'DevSelectionScore':score,'Selected':int(better),'PatienceCount':bad,'GradientNorm':float(np.mean(norms)),
      'OptimizerTargets':used,'CarryTargets':len(pending),'Seconds':time.monotonic()-begin,'OrderSHA256':order_hash,
      **{f'InnerDev{k}Top1FDE':v['Top1FDE'] for k,v in measured['Metrics'].items()}}
  curve.append(row);dump(folder/'stage15b_training_curve.csv',curve);dump(folder/'stage15b_batch_order.csv',orders)
  cp={'state_dict':cpu_state(m),'Fold':fold,'Model':name,'Epoch':epoch,'Score':score,'initial_state_sha256':initial,
      'ConfigSHA256':sha256(folder/'stage15b_training_config.json'),'NormalizationSHA256':c['NormalizationSHA256'],
      'PredictorCheckpointSHA256':c['PredictorCheckpointSHA256'],'R2DevReferences':refs,'BicycleRoute':'new samefold R2'}
  if better:atomic_torch(cp_path(fold,name),cp)
  done=bad>=5 or epoch==50
  atomic_torch(folder/'last.pt',{**cp,'optimizer_state':optimizer.state_dict(),'curve':curve,'coverage':coverage,'pending':pending,
       'bad':bad,'best_score':best,'best_epoch':bestep,'complete':done,'torch_rng':torch.get_rng_state(),
       'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state()})
  print('HEAD',fold,name,epoch,'DEV',score,'BEST',bestep,'BAD',bad,flush=True)
  if done:break
 assert (coverage>0).all();saved=torch.load(cp_path(fold,name),map_location='cpu',weights_only=False);m.load_state_dict(saved['state_dict'],strict=True)
 measured,_=assess(m,name,dv,r2,refs);assert abs(measured['DevSelectionScore']-best)<1e-12
 result={'Status':'COMPLETE','Fold':fold,'Model':name,'SelectedEpoch':bestep,'ExecutedEpochs':epoch,'CheckpointScore':best,
         'CheckpointSHA256':sha256(cp_path(fold,name)),'Params':PARAMS[name],'InitialStateSHA256':initial,
         'DevMetrics':measured['Metrics'],'Selection':c['R2Protocol']['selection'] if name=='R2' else c['GraphProtocol']['ABC_selection'],
         'NormalizationSHA256':c['NormalizationSHA256'],'PredictorCheckpointSHA256':c['PredictorCheckpointSHA256'],
         'AllDistinctTrainActorWindowsUsed':len(tr.ids),'FinalPendingOccurrences':len(pending),'OuterTestUsed':False,
         'HistoricalOrTinyWeightsLoaded':False,'PredictorGradientCount':0,'Seconds':time.monotonic()-started,
         'PeakGPUMemoryBytes':torch.cuda.max_memory_allocated(),'ConfigSHA256':sha256(folder/'stage15b_training_config.json')}
 atomic_json(summarypath,result);del m,optimizer,data;torch.cuda.empty_cache();return result


def run_fold(fold):
 verify_sources()
 from stage15b_cache import context
 ctx=context(fold,'ranking')
 def forbid_outer(event,args):
  if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
   path=Path(os.fsdecode(args[0])).resolve()
   assert not path.is_relative_to(ROOT/f'05_candidate_interface/cache/fold{fold}/OuterTest'),'Outer cache read forbidden during head fitting'
 sys.addaudithook(forbid_outer)
 tr=RoleStore(fold,'InnerTrain');dv=RoleStore(fold,'InnerDev');summaries=[]
 for name in NAMES:
  r2=refs=None
  if name!='R2':
   r2=fresh('R2',fold);r2.load_state_dict(torch.load(cp_path(fold,'R2'),map_location='cpu',weights_only=False)['state_dict']);r2.eval().requires_grad_(False)
   refs={t:summaries[0]['DevMetrics'][t]['Top1FDE'] for t in ('Vehicle','Pedestrian')}
  try:result=train_head(fold,name,tr,dv,r2,refs)
  except BaseException as e:
   atomic_json(ROOT/f'06_rank_training/fold{fold}/{name}/stage15b_failure.json',{'Status':'FAILED_STOP','Exception':repr(e),'Traceback':__import__('traceback').format_exc(),'NoProtocolChangePermitted':True});raise
  summaries.append(result)
  if r2 is not None:del r2;torch.cuda.empty_cache()
 atomic_json(ROOT/f'07_rank_checkpoints/stage15b_fold{fold}_frozen.json',{'Status':'FROZEN','Fold':fold,'Checkpoints':summaries,'OuterEvaluationPermitted':False})
 print('ALL_SIX_FOLD_HEADS_FROZEN',fold,flush=True)


def freeze_heads():
 summaries=[]
 for fold in (1,2,3):
  record=read_json(ROOT/f'07_rank_checkpoints/stage15b_fold{fold}_frozen.json');assert len(record['Checkpoints'])==6
  for x in record['Checkpoints']:
   assert x['Status']=='COMPLETE' and not x['OuterTestUsed'] and sha256(cp_path(fold,x['Model']))==x['CheckpointSHA256']
  summaries+=record['Checkpoints']
 atomic_json(ROOT/'07_rank_checkpoints/stage15b_all_frozen.json',{'Status':'FROZEN_ALL_COMPLETE','Checkpoints':summaries,'FurtherTrainingPermitted':False,'OuterEvaluationPermitted':True})
 print('ALL18_HEADS_FROZEN',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=(1,2,3));p.add_argument('--freeze',action='store_true');args=p.parse_args()
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 if args.freeze:freeze_heads()
 else:assert args.fold;run_fold(args.fold)
