"""Post-freeze distribution, negative transfer, cases and scoped compute overhead."""
from pathlib import Path
import sys,time,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'02_training'),str(ROOT/'05_candidate_interface'),str(ROOT/'06_rank_training'),str(ROOT/'08_evaluation')]
from stage15b_evaluate import *
from stage15b_common import FoldContext,FoldDataset,model_input
from stage15b_train import restore
from torch_geometric.data import Batch


def distributions():
 rows=[]
 for fold in (1,2,3):
  for role in ('InnerTrain','InnerDev','OuterTest'):
   s=RoleStore(fold,role);oracle=np.asarray(s.fde).min(-1);gap=np.asarray(s.fde).max(-1)-oracle
   for group,keep in groups(s.frame).items():
    x=oracle[keep]
    if len(x):rows.append({'Fold':fold,'Role':role,'Group':group,'Count':len(x),'Scenes':s.frame.loc[keep,'scene_token'].nunique(),
      'OracleMinFDEMean':float(x.mean(dtype=np.float64)),'OracleMinFDEMedian':float(np.median(x)),
      'OracleMinFDEP90':float(np.quantile(x,.9)),'OracleMinFDEP99':float(np.quantile(x,.99)),
      'MeanMaxMinusMinCandidateFDE':float(gap[keep].mean(dtype=np.float64)),
      'Interpretation':'distribution description after all models frozen; InnerTrain is predictor training-in, not OOF'})
 dump(ROOT/'09_statistics/stage15b_candidate_error_distributions.csv',rows)


def failures():
 folder=ROOT/'08_evaluation/cache';f=pd.read_csv(folder/'actor_records.csv');v=np.load(folder/'metrics.npy',mmap_mode='r');z=np.load(folder/'logits.npy',mmap_mode='r')
 contrasts=(('G-C','NG-C'),('G-C','Matched-NG-C'),('NG-C','NG-A'));rows=[];case_records=[];seen=set()
 casesdir=ROOT/'08_evaluation/stage15b_cases';casesdir.mkdir(parents=True,exist_ok=True)
 for a,b in contrasts:
  delta=v[:,MODELS.index(a),0]-v[:,MODELS.index(b),0]
  for group,keep in groups(f).items():
   d=delta[keep];harm=d[d>0];gain=d[d<0];largest=harm[np.argsort(harm)[-max(1,int(np.ceil(.1*len(harm)))):]] if len(harm) else harm
   rows.append({'Group':group,'Comparison':a+'-'+b,'Count':len(d),'Improved':int((d<0).sum()),'Worsened':int((d>0).sum()),'Equal':int((d==0).sum()),
     'DeltaTop1FDE':float(d.mean()),'PositiveSwitchHarmSum':float(harm.sum()),'Largest10PercentPositiveHarmSum':float(largest.sum()),
     'Largest10PercentPositiveHarmCount':len(largest),'NegativeTransferReported':True})
 # Deterministic extreme harms, transparently illustrative and not representative statistics.
 specs=[('G-C','NG-C','MovingVehicle'),('G-C','NG-C','Pedestrian'),('G-C','Matched-NG-C','Overall'),('NG-C','NG-A','Vehicle')]
 for a,b,group in specs:
  delta=v[:,MODELS.index(a),0]-v[:,MODELS.index(b),0];keep=groups(f)[group];available=np.flatnonzero(keep)
  order=available[np.argsort(delta[available],kind='stable')[::-1]];i=next(int(i) for i in order if int(i) not in seen);seen.add(i)
  row=f.iloc[i];fold=int(row.fold);scene=str(row.scene_token);rolefolder=ROOT/f'05_candidate_interface/cache/fold{fold}/OuterTest'
  scene_cache=torch.load(rolefolder/(scene+'.pt'),map_location='cpu',weights_only=False)
  w=next(w for w in scene_cache if w['observable']['sample_token']==row.sample_token)
  obs=w['observable'];labels=w['labels'];node=int(row.node_index);assert obs['instance_tokens'][node]==row.instance_token
  tops={name:int(z[i,axis].argmax()) for axis,name in enumerate(MODELS)}
  cs={name:{'mode':tops[name],'Top1FDE':float(v[i,axis,0]),'Top1ADE':float(v[i,axis,1])} for axis,name in enumerate(MODELS)}
  trajectory=[]
  for t in range(12):
   r={'time_seconds_actual':float(obs['future_sample_times_metadata'][t]),'GT_x':float(labels['future_xy'][node,t,0]),'GT_y':float(labels['future_xy'][node,t,1])}
   for mode in range(6):r.update({f'candidate{mode}_x':float(obs['ego_prediction'][node,mode,t,0]),f'candidate{mode}_y':float(obs['ego_prediction'][node,mode,t,1])})
   trajectory.append(r)
  cid=f'case{len(case_records)+1}';dump(casesdir/(cid+'_trajectories.csv'),trajectory)
  detail={'Case':cid,'Selection':'largest observed harm in stated fixed contrast/group, unique actor-window','Comparison':a+'-'+b,'Group':group,
      'Fold':fold,'SceneToken':scene,'SampleToken':row.sample_token,'InstanceToken':row.instance_token,'NodeIndex':node,
      'DeltaTop1FDE':float(delta[i]),'OracleMinFDE6':float(v[i,0,13]),'Models':cs,
      'PredictorCheckpointSHA256':row.predictor_checkpoint_sha256,'CoordinateFrame':'t0 ego xy meters','Times':'exact source timestamps'}
  atomic_json(casesdir/(cid+'.json'),detail);case_records.append(detail)
 dump(ROOT/'09_statistics/stage15b_switch_harm.csv',rows)
 text=['# Stage15B failure cases','', 'Cases are selected after all checkpoints and evaluation outputs freeze. They illustrate extreme observed harms and do not estimate their prevalence. All degradation and tie counts appear in `09_statistics/stage15b_switch_harm.csv`. Shared-candidate oracle metrics remain unchanged.','']
 for r in case_records:
  text.extend([f"## {r['Case']}: {r['Comparison']} / {r['Group']}",'',f"Fold {r['Fold']}; scene `{r['SceneToken']}`, sample `{r['SampleToken']}`, instance `{r['InstanceToken']}`. Delta Top1FDE={r['DeltaTop1FDE']:.6f} m; shared oracle minFDE6={r['OracleMinFDE6']:.6f} m.",'',
    f"Full selected modes, errors and predictor SHA: [case metadata](08_evaluation/stage15b_cases/{r['Case']}.json). Exact GT and all six candidate coordinates/times: [trajectories](08_evaluation/stage15b_cases/{r['Case']}_trajectories.csv). These cases show ranking errors under common geometry; they do not establish a causal mechanism.",''])
 (ROOT/'stage15b_failure_cases.md').write_text('\n'.join(text)+'\n')


@torch.no_grad()
def compute():
 rows=[];system=[]
 for fold in (1,2,3):
  record=read_json(PROTOCOL)['folds'][fold-1]
  ctx=FoldContext(fold,record['seed'],ROOT/record['files']['InnerTrain']['path'],ROOT/record['files']['InnerDev']['path'],
                  ROOT/f'04_predictor_checkpoints/fold{fold}',stage='evaluation',install=fold==1)
  store=RoleStore(fold,'InnerDev');ids=store.ids[:1024];pieces=[]
  for start in range(0,len(ids),128):
   ix=ids[start:start+128];args,_,_=store.batch(ix);pieces.append((args,store.r2(ix)))
  heads={}
  for name in NAMES:
   m=fresh(name,fold);m.load_state_dict(torch.load(cp_path(fold,name),map_location='cpu',weights_only=False)['state_dict']);m.eval().requires_grad_(False);heads[name]=m
   for j in range(20):head_forward(m,name,*pieces[0])
   torch.cuda.synchronize();timings=[];torch.cuda.reset_peak_memory_stats()
   for rep in range(20):
    torch.cuda.synchronize();t=time.perf_counter()
    for args,rf in pieces:head_forward(m,name,args,rf)
    torch.cuda.synchronize();timings.append(time.perf_counter()-t)
   rows.append({'Fold':fold,'Model':name,'Params':PARAMS[name],'TargetsPerTimedCycle':1024,'BatchSize':128,'WarmupCalls':20,'TimedCycles':20,
      'MeanSecondsPer1024Targets':float(np.mean(timings)),'MedianMSPerTarget':1000*float(np.median(timings))/1024,
      'PeakAllocatedBytes':torch.cuda.max_memory_allocated(),'Scope':'preloaded normalized same InnerDev pool; CUDA forward only, excludes predictor/features/H2D/routing'})
  ds=FoldDataset(ctx,'InnerDev');graphs=[ds[i] for i in range(16)];batch=Batch.from_data_list(graphs);data=batch.cuda();observed=model_input(data)
  chosen=read_json(ROOT/'04_predictor_checkpoints/stage15b_all_frozen.json')['Checkpoints'][fold-1]
  model,optimizer,saved=restore(ROOT/chosen['path'],ctx,'FORMAL_SCENE_ISOLATED');del optimizer;model.eval().requires_grad_(False)
  for _ in range(3):model(observed)
  baseline=[]
  for rep in range(20):
   torch.cuda.synchronize();t=time.perf_counter();out=model(observed);pred=model.ego_predictions(out,observed);torch.cuda.synchronize();baseline.append(time.perf_counter()-t)
  def full(name):
   out=model(observed);pred=model.ego_predictions(out,observed);ptr=batch.ptr.tolist();targets=0
   for j,g in enumerate(graphs):
    lo,hi=ptr[j:j+2];w={'history':g.positions[:,:5], 'history_padding':g.padding_mask[:,:5],'agent_type':g.agent_type,
       'ego_prediction':pred[lo:hi].cpu(),'mode_logits':out['mode_logits'][lo:hi].cpu(),'mode_prob':out['mode_prob'][lo:hi].cpu(),
       'scene_token':g.scene_token,'sample_token':g.sample_token,'instance_tokens':g.instance_tokens,'map_location':g.map_location,
       'origin':g.origin.numpy(),'yaw':float(g.ego_yaw),'source_role':'InnerDev'}
    # Past-only eligibility; GT/full-horizon labels are not consulted in this forward.
    pack=pack_window(w);targets+=len(pack['targets']);args,rf=normalized(pack,store.norm)
    args=tuple(a.cuda() if a is not None else None for a in args);rf=rf.cuda()
    outhead=head_forward(heads[name],name,args,rf)
    if name!='R2':
     r2=heads['R2'](rf,args[6]);bike=g.agent_type[pack['targets']].cuda()==2
     outhead['mode_logits'][bike]=r2['mode_logits'][bike]
   return targets
  for name in ('R2','G-C'):
   for _ in range(3):full(name)
   timings=[];torch.cuda.reset_peak_memory_stats()
   for rep in range(20):
    torch.cuda.synchronize();t=time.perf_counter();targets=full(name);torch.cuda.synchronize();timings.append(time.perf_counter()-t)
   system.append({'Fold':fold,'Model':name,'Windows':16,'CurrentActors':batch.num_nodes,'PastEligibleTargets':targets,
       'PurePredictorMeanSeconds':float(np.mean(baseline)),'FullSystemMeanSeconds':float(np.mean(timings)),
       'AddedSeconds':float(np.mean(timings)-np.mean(baseline)),'TimedRepeats':20,'PeakAllocatedBytes':torch.cuda.max_memory_allocated(),
       'Scope':'fixed same16 InnerDev windows; preloaded predictor input, predictor+ego conversion+CPU graph/R2 features+normalization+H2D+head+Bicycle route; excludes raw data I/O and audit SHA capture; engineering path, not optimized deployment FPS'})
  del model,heads,pieces,store;ds.clear();torch.cuda.empty_cache()
 dump(ROOT/'09_statistics/stage15b_head_compute.csv',rows);dump(ROOT/'09_statistics/stage15b_system_compute.csv',system)

if __name__=='__main__':
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 distributions();failures();compute();print('EXTRA_ANALYSIS_COMPLETE',flush=True)
