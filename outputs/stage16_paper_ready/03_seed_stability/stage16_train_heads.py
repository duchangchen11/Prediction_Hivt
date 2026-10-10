"""Original Stage15B head optimizer loop, new init seeds and Stage16 output scope only."""
from pathlib import Path
import sys,os,time,json,hashlib,random,csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage16_common import *
from stage15b_heads import verify_sources
REG=read_json(ROOT/'00_manifest/stage16_supplement_registration.json')
JOB=None
NAMES=original.NAMES
PROTOCOL=original.PROTOCOL

def cp_path(fold,name):return ROOT/f"03_seed_stability/checkpoints/fold{fold}/seed{JOB['initialization_seed']}/{name}_best.pt"

def fresh(name,fold):
 assert JOB['fold']==fold and JOB['model']==name
 seed=JOB['initialization_seed'];seed_all(seed)
 m=NoGraphReranker(seed) if name.startswith('NG-') else MatchedNoGraphReranker(seed) if name=='Matched-NG-C' else SparseGraphReranker('G1',seed)
 assert sum(p.numel() for p in m.parameters())==PARAMS[name]
 return m.cuda()

def config(fold,name,tr,dv):
 folder=ROOT/f"03_seed_stability/fold{fold}/seed{JOB['initialization_seed']}/{name}"
 c={**JOB,'Fold':fold,'Model':name,'parameters':PARAMS[name], 'ProtocolSHA256':sha256(PROTOCOL),
 'TrainingPartition':'InnerTrain378','SelectionPartition':'InnerDev42','TrainingCacheManifestSHA256':sha256(tr.folder/'manifest.json'),
 'DevCacheManifestSHA256':sha256(dv.folder/'manifest.json'),'NormalizationSHA256':sha256(tr.normpath),
 'PredictorCheckpointSHA256':tr.manifest['PredictorCheckpointSHA256'],'HistoricalWeightsLoaded':False,
 'OriginalOptimizerLoopSHA256':sha256(Path(original.__file__)), 'SupplementRegistrationSHA256':sha256(ROOT/'00_manifest/stage16_supplement_registration.json'),
 'R2CheckpointSHA256':sha256(original.cp_path(fold,'R2')),
 'R2Protocol':read_json(PROTOCOL)['ranking']['R2_protocol'],
 'GraphProtocol':{k:read_json(PROTOCOL)['ranking'][k] for k in ('optimizer','learning_rate','weight_decay','microbatch','accumulation','effective_batch','max_epochs','patience','carry','ABC_selection')}}
 atomic_json(folder/'stage16_training_config.json',c);return folder,c

def check_original_order(fold,epoch,value):
 for name in REG['Models']:
  p=OLD/f'06_rank_training/fold{fold}/{name}/stage15b_batch_order.csv'
  df=pd.read_csv(p)
  if epoch<=len(df):assert df.iloc[epoch-1].OrderSHA256==value,'Data permutation changed'

def train_head(fold,name,tr,dv,r2=None,refs=None):
 verify_sources();assert tr.role=='InnerTrain' and dv.role=='InnerDev'
 folder,c=config(fold,name,tr,dv);summarypath=folder/'stage16_summary.json'
 if summarypath.exists():
  result=read_json(summarypath);assert sha256(cp_path(fold,name))==result['CheckpointSHA256'];return result
 assert not (folder/'last.pt').exists(),'Failed/partial head run is preserved; no automatic restart'
 m=fresh(name,fold);initial=state_digest(m.state_dict());optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
 zero,_=assess(m,name,dv,r2,refs);atomic_json(folder/'stage16_step0.json',zero)
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
  check_original_order(fold,epoch,order_hash)
  measured,_=assess(m,name,dv,r2,refs);score=measured['DevSelectionScore'];better=score<best
  if better:best=score;bestep=epoch;bad=0
  else:bad+=1
  row={'Epoch':epoch,'TrainLoss':total/used,'InnerDevLoss':measured['InnerDevLoss'],
      'DevSelectionScore':score,'Selected':int(better),'PatienceCount':bad,'GradientNorm':float(np.mean(norms)),
      'OptimizerTargets':used,'CarryTargets':len(pending),'Seconds':time.monotonic()-begin,'OrderSHA256':order_hash,
      **{f'InnerDev{k}Top1FDE':v['Top1FDE'] for k,v in measured['Metrics'].items()}}
  curve.append(row);dump(folder/'stage16_training_curve.csv',curve);dump(folder/'stage16_batch_order.csv',orders)
  cp={'state_dict':cpu_state(m),'Fold':fold,'Model':name,'Epoch':epoch,'Score':score,'initial_state_sha256':initial,
      'ConfigSHA256':sha256(folder/'stage16_training_config.json'),'NormalizationSHA256':c['NormalizationSHA256'],
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
         'PeakGPUMemoryBytes':torch.cuda.max_memory_allocated(),'ConfigSHA256':sha256(folder/'stage16_training_config.json')}
 atomic_json(summarypath,result);del m,optimizer,data;torch.cuda.empty_cache();return result

def main():
 global JOB
 runtime();verify_sources();protect_fit()
 completed=[]
 for fold in (1,2,3):
  tr=RoleStore(fold,'InnerTrain');dv=RoleStore(fold,'InnerDev')
  r2=original.fresh('R2',fold);r2.load_state_dict(torch.load(original.cp_path(fold,'R2'),map_location='cpu',weights_only=False)['state_dict'],strict=True);r2.eval().requires_grad_(False)
  rs=read_json(OLD/f'06_rank_training/fold{fold}/R2/stage15b_summary.json');refs={t:rs['DevMetrics'][t]['Top1FDE'] for t in ('Vehicle','Pedestrian')}
  for JOB in (j for j in REG['Jobs'] if j['fold']==fold):
   print('START_SEED_JOB',JOB,flush=True)
   result=train_head(fold,JOB['model'],tr,dv,r2,refs);result.update(JOB);completed.append(result)
   atomic_json(ROOT/'03_seed_stability/stage16_progress.json',{'Status':'FITTING','Completed':len(completed),'Total':24,'Results':completed})
  del tr,dv,r2;torch.cuda.empty_cache()
 assert len(completed)==24
 atomic_json(ROOT/'03_seed_stability/stage16_all_frozen.json',{'Status':'FROZEN_ALL_COMPLETE','RegistrationSHA256':sha256(ROOT/'00_manifest/stage16_supplement_registration.json'),'TrainingSourceSHA256':sha256(Path(__file__)),'Results':completed,'OuterReadDuringFitting':False,'FurtherTrainingPermitted':False})
 print('ALL24_SUPPLEMENT_HEADS_FROZEN',flush=True)
if __name__=='__main__':main()
