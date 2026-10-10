"""All-predictor gate then shared, role-separated streamed candidate/features cache."""
from pathlib import Path
import sys,csv,json,time,os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'02_training'),str(ROOT/'05_candidate_interface')]
from stage15b_common import *
from stage15b_train import restore
from stage15b_ranking import candidate_windows,pack_window
from torch_geometric.data import Batch


def context(fold,stage='candidate'):
 c=read_json(PROTOCOL)['folds'][fold-1]
 return FoldContext(fold,c['seed'],ROOT/c['files']['InnerTrain']['path'],ROOT/c['files']['InnerDev']['path'],
                    ROOT/f'04_predictor_checkpoints/fold{fold}',stage=stage)


def freeze_predictors():
 records=[]
 for fold in (1,2,3):
  folder=ROOT/f'04_predictor_checkpoints/fold{fold}'
  assert not (folder/'stage15b_failure.json').exists()
  w=read_json(folder/'stage15b_warmup_summary.json');n=read_json(folder/'stage15b_nll_summary.json')
  assert w['status']==n['status']=='COMPLETE' and w['phase_steps']==5000 and 500<=n['phase_steps']<=16000
  for s in (w,n):
   f=read_json(PROTOCOL)['folds'][fold-1]
   assert set(s['TrainScenes'])==set(f['parts']['InnerTrain']) and set(s['DevScenes'])==set(f['parts']['InnerDev'])
   assert not s['OuterTestUsed'] and not s['HeadDevUsed']
  cp=ROOT/n['selected_checkpoint'];assert sha256(cp)==n['selected_checkpoint_sha256']
  # Structural/readback integrity; no evaluation on OuterTest.
  saved=torch.load(cp,map_location='cpu',weights_only=False)
  assert saved['config']['scope']=='FORMAL_SCENE_ISOLATED' and saved['config']['fold']==fold
  assert saved['config']['seed']==f['seed'] and saved['phase_state']['phase']=='original_nll'
  m=model_new(f['seed'],'cpu');m.load_state_dict(saved['state_dict'],strict=True)
  assert sum(p.numel() for p in m.parameters())==650403
  assert state_digest(m.state_dict())==state_digest(saved['state_dict'])
  records.append({'fold':fold,'seed':f['seed'],'path':str(cp.relative_to(ROOT)), 'sha256':sha256(cp),
                  'state_sha256':state_digest(m.state_dict()),'params':650403,'warm_steps':5000,
                  'nll_steps_executed':n['phase_steps'],'selected_global_step':n['best_global_step'],
                  'checkpoint_dev_minFDE6':n['best_overall_FDE'],'transition':read_json(folder/'stage15b_formal_transition.json'),
                  'TrainSceneSHA256':f['files']['InnerTrain']['sha256'],'DevSceneSHA256':f['files']['InnerDev']['sha256']})
 atomic_json(ROOT/'04_predictor_checkpoints/stage15b_all_frozen.json',{'Status':'FROZEN_ALL_COMPLETE','Checkpoints':records,
   'NewFromFresh':True,'TinyOrHistoricalWeightsUsed':False,'OuterEvaluationPerformed':False,'FurtherPredictorTrainingPermitted':False})
 print('ALL_THREE_PREDICTORS_FROZEN',flush=True)


class RunningMoments:
 def __init__(self):self.n=0;self.mean=0.;self.M2=0.
 def update(self,x):
  v=np.asarray(x,dtype=np.float64).reshape(-1)
  if not len(v):return
  assert np.isfinite(v).all();n=len(v);u=float(v.mean());m2=float(((v-u)**2).sum())
  d=u-self.mean;total=self.n+n
  self.M2+=m2+d*d*self.n*n/total;self.mean+=d*n/total;self.n=total
 def result(self):
  assert self.n>0
  return {'count':self.n,'mean':self.mean,'std':float(np.sqrt(max(0.,self.M2/self.n)))}


def create_cache(fold):
 ctx=context(fold);record=read_json(ROOT/'04_predictor_checkpoints/stage15b_all_frozen.json')['Checkpoints'][fold-1]
 cp=ROOT/record['path'];assert sha256(cp)==record['sha256']
 model,optimizer,saved=restore(cp,ctx,'FORMAL_SCENE_ISOLATED');del optimizer
 model.eval().requires_grad_(False)
 stats={str(k):{str(c):RunningMoments() for c in cols} for k,cols in ((0,range(3,15)),(1,range(11)))}
 rstats=[RunningMoments() for _ in range(19)]
 manifests=[];started=time.monotonic()
 for role in ('InnerTrain','InnerDev','OuterTest'):
  folder=ROOT/f'05_candidate_interface/cache/fold{fold}/{role}';folder.mkdir(parents=True,exist_ok=True)
  assert not (folder/'manifest.json').exists(),'Frozen cache must not be overwritten'
  ds=FoldDataset(ctx,role);count=ctx.record['metadata'][role]['full_targets']
  specs={'arg0':(np.float32,(count,9,6,15)),'arg1':(np.float32,(count,6,8,6,17)),
         'arg2':(np.bool_,(count,8)),'arg6':(np.float32,(count,6)),
         'fde':(np.float32,(count,6)),'ade':(np.float32,(count,6)),
         'r2_raw':(np.float32,(count,6,19)),'r2_flags':(np.bool_,(count,3))}
  arrays={k:np.lib.format.open_memmap(folder/(k+'.npy'),mode='w+',dtype=dt,shape=shape) for k,(dt,shape) in specs.items()}
  pos=0;scenes=set();context_rows=[];stored_windows=0;context_actors=0;exclude_partial=0
  path=folder/'targets.csv';fields=['source_index','fold','seed','role','scene_token','scene_name','sample_token','instance_token','node_index','agent_type_id','agent_type','motion_state','GT_displacement','predictor_checkpoint_sha256','coordinate_frame']
  with path.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for scene,window_ids in ds.scene_indices.items():
    # One small scene object, saved once per fold, shared by all six heads.
    scene_windows=[]
    for start in range(0,len(window_ids),16):
     ids=window_ids[start:start+16];graphs=[ds[int(i)] for i in ids];batch=Batch.from_data_list(graphs)
     windows=candidate_windows(model,batch,graphs,ctx,role)
     for g,w in zip(graphs,windows):
      full=g.target_mask & g.future_mask.all(-1);targets=torch.where(full)[0]
      pack=pack_window(w,targets);n=len(targets);sl=slice(pos,pos+n)
      args=pack['args'];pred=w['ego_prediction'][targets];gt=g.positions[targets,5:]
      distance=(pred-gt[:,None]).norm(dim=-1);fd=distance[:,:,-1];ad=distance.mean(-1)
      for k,val in [('arg0',args[0]),('arg1',args[1]),('arg2',args[2]),('arg6',args[6]),
                    ('fde',fd),('ade',ad),('r2_raw',pack['r2']),('r2_flags',pack['flags'])]:
       arrays[k][sl]=val.numpy()
      if role=='InnerTrain' and n:
       mask=args[2];nv=torch.cat((torch.ones((n,1),dtype=torch.bool),mask),1)[...,None].expand(-1,9,6)
       ev=mask[:,None,:,None].expand(-1,6,8,6)
       for k,x,valid,cols in ((0,args[0],nv,range(3,15)),(1,args[1],ev,range(11))):
        for col in cols:stats[str(k)][str(col)].update(x[...,col][valid].numpy())
       for col in range(19):rstats[col].update((pack['r2'][pack['flags'][:,0],:,col] if 12<=col<15 else pack['r2'][:,:,col]).numpy())
      for row,t in enumerate(targets.tolist()):
       writer.writerow({'source_index':pos+row,'fold':fold,'seed':ctx.seed,'role':role,'scene_token':scene,
        'scene_name':g.scene_name,'sample_token':g.sample_token,'instance_token':g.instance_tokens[t],
        'node_index':t,'agent_type_id':int(g.agent_type[t]),'agent_type':('Vehicle','Pedestrian','Bicycle')[int(g.agent_type[t])],
        'motion_state':g.t0_motion_state[t],'GT_displacement':float((g.positions[t,-1]-g.positions[t,4]).norm()),
        'predictor_checkpoint_sha256':record['sha256'],'coordinate_frame':'t0_ego_xy_m'})
      pos+=n;stored_windows+=1;context_actors+=g.num_nodes;exclude_partial+=int((g.target_mask & ~g.future_mask.all(-1)).sum())
      # Keep supervised fields in a separate sidecar, never in the observable feature object.
      obs={k:v for k,v in w.items() if k not in ('raw_prediction','rotation')}
      obs['predictor_checkpoint_sha256']=record['sha256']
      labels={'future_xy':g.positions[:,5:].clone(),'future_mask':g.future_mask.clone(),'target_mask':g.target_mask.clone(),
              'full_horizon_mask':full.clone(),'motion_state':g.t0_motion_state,'annotation_tokens':g.annotation_tokens,
              'input_signature':getattr(g,'input_signature',None)}
      scene_windows.append({'observable':obs,'labels':labels,'target_cache_start':pos-n,'target_count':n})
      assert torch.all(g.future_times[1:] > g.future_times[:-1]) and torch.all(g.history_times[1:] > g.history_times[:-1])
    scpath=folder/(scene+'.pt');torch.save(scene_windows,scpath)
    context_rows.append({'scene_token':scene,'path':str(scpath.relative_to(ROOT)),'sha256':sha256(scpath),'bytes':scpath.stat().st_size})
    scenes.add(scene)
    if len(scenes)%25==0:print('CACHE',fold,role,len(scenes),'/',len(ds.scene_indices),'targets',pos,flush=True)
   assert pos==count and scenes==set(ctx.parts[role])
  for a in arrays.values():a.flush()
  manifest={'Status':'FROZEN','Fold':fold,'Seed':ctx.seed,'Role':role,'Scenes':sorted(scenes),'SceneCount':len(scenes),
    'Targets':count,'StoredWindows':stored_windows,'ContextActors':context_actors,'PartialTargetsExcluded':exclude_partial,
    'EmptySupervisionWindowsExcluded':len(ds.empty_rows),'PredictorCheckpointSHA256':record['sha256'],
    'CoordinateFrame':'t0 ego x-forward y-left meters','Clock':'original source keyframes, nominal2s/6s, actual per-window timestamps retained',
    'Arrays':{k:{'path':str((folder/(k+'.npy')).relative_to(ROOT)),'sha256':sha256(folder/(k+'.npy')),'bytes':(folder/(k+'.npy')).stat().st_size,'shape':list(a.shape),'dtype':str(a.dtype)} for k,a in arrays.items()},
    'IdentityCSV':{'path':str(path.relative_to(ROOT)),'sha256':sha256(path)},'Contexts':context_rows,'SixHeadsShareCache':True,
    'GTUse':'separate labels; never passed to predictor or graph/R2 observable feature computation'}
  atomic_json(folder/'manifest.json',manifest);manifests.append(manifest)
  ds.clear();del arrays;torch.cuda.empty_cache()
 norm={'Fold':fold,'fit_partition':'InnerTrain','formal_eligible':True,'fit_scene_tokens':sorted(ctx.parts['InnerTrain']),
       'fit_actor_windows':manifests[0]['Targets'],'graph':{k:{c:x.result() for c,x in cols.items()} for k,cols in stats.items()},
       'R2':{'mean':[x.result()['mean'] for x in rstats],'std':[x.result()['std'] for x in rstats]},
       'epsilon':1e-6,'std_ddof':0,'InnerDev_or_Outer_or_HeadDev_used':False,'PredictorCheckpointSHA256':record['sha256']}
 atomic_json(ROOT/f'05_candidate_interface/stage15b_fold{fold}_normalization.json',norm)
 atomic_json(ROOT/f'05_candidate_interface/stage15b_fold{fold}_provenance.json',{'Status':'COMPLETE','Fold':fold,'Roles':manifests,
      'NormalizationSHA256':sha256(ROOT/f'05_candidate_interface/stage15b_fold{fold}_normalization.json'),'Seconds':time.monotonic()-started})
 print('FOLD_CACHE_COMPLETE',fold,flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--fold',type=int,choices=(1,2,3));args=p.parse_args()
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 if args.freeze:freeze_predictors()
 else:assert args.fold;create_cache(args.fold)
