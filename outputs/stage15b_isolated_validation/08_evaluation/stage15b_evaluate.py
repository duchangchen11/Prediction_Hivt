"""Unified Outer evaluation after every new predictor and all18 heads freeze."""
from pathlib import Path
import sys,ast,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'05_candidate_interface'),str(ROOT/'06_rank_training')]
from stage15b_heads import *
from stage15b_cache import context
FIELDS=('Top1FDE','Top1ADE','OracleGap','HitRate','MRR','SoftCE','ExpectedRegret','NormalizedExpectedRegret','PredictionEntropy','Top1Probability','Top1Top2Margin','OracleModeProbability','minADEOracle6','minFDE6','MR6')
MODELS=('R0',*NAMES)
source=PROJECT/'outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py'
tree=ast.parse(source.read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='actor_metrics']
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),globals())


def groups(f):
 typ=f.agent_type.to_numpy();state=f.motion_state.to_numpy()
 return {'Overall':np.ones(len(f),bool),**{t:typ==t for t in TYPES},
    **{g:(typ=='Vehicle')&(state==s) for g,s in [('MovingVehicle','vehicle.moving'),('StoppedVehicle','vehicle.stopped'),('ParkedVehicle','vehicle.parked')]},
    'OtherVehicleState':(typ=='Vehicle')&~np.isin(state,['vehicle.moving','vehicle.stopped','vehicle.parked'])}


def main():
 gate=read_json(ROOT/'07_rank_checkpoints/stage15b_all_frozen.json');assert gate['Status']=='FROZEN_ALL_COMPLETE' and len(gate['Checkpoints'])==18
 verify_sources()
 for x in gate['Checkpoints']:assert sha256(cp_path(x['Fold'],x['Model']))==x['CheckpointSHA256']
 folder=ROOT/'08_evaluation/cache';folder.mkdir(parents=True,exist_ok=True);assert not (folder/'complete.json').exists()
 frames=[pd.read_csv(ROOT/f'05_candidate_interface/cache/fold{k}/OuterTest/targets.csv') for k in (1,2,3)]
 f=pd.concat(frames,ignore_index=True);n=len(f);assert n==260151 and f.scene_token.nunique()==630
 assert not f.duplicated(['scene_token','sample_token','instance_token']).any()
 values=np.lib.format.open_memmap(folder/'metrics.npy',mode='w+',dtype=np.float64,shape=(n,len(MODELS),len(FIELDS)))
 logits=np.lib.format.open_memmap(folder/'logits.npy',mode='w+',dtype=np.float32,shape=(n,len(MODELS),6))
 probs=np.lib.format.open_memmap(folder/'probabilities.npy',mode='w+',dtype=np.float32,shape=(n,len(MODELS),6))
 offset=0;foldrows=[];runtime=[];poison=[]
 for fold in (1,2,3):
  record=read_json(PROTOCOL)['folds'][fold-1]
  from stage15b_common import FoldContext
  ctx=FoldContext(fold,record['seed'],ROOT/record['files']['InnerTrain']['path'],ROOT/record['files']['InnerDev']['path'],
                  ROOT/f'04_predictor_checkpoints/fold{fold}',stage='evaluation',install=fold==1)
  store=RoleStore(fold,'OuterTest');heads={}
  for k,v in store.manifest['Arrays'].items():assert sha256(store.folder/(k+'.npy'))==v['sha256']
  for name in NAMES:
   m=fresh(name,fold);saved=torch.load(cp_path(fold,name),map_location='cpu',weights_only=False)
   assert saved['Fold']==fold and saved['Model']==name and saved['NormalizationSHA256']==sha256(store.normpath)
   assert saved['PredictorCheckpointSHA256']==store.manifest['PredictorCheckpointSHA256']
   m.load_state_dict(saved['state_dict'],strict=True);m.eval().requires_grad_(False);heads[name]=m
  states={k:state_digest(m.state_dict()) for k,m in heads.items()};elapsed=dict.fromkeys(MODELS,0.);prepare_seconds=0.;startfold=time.monotonic()
  for start in range(0,len(store.ids),128):
   ix=store.ids[start:start+128];t=time.monotonic();args,fd,ad=store.batch(ix);rf=store.r2(ix);torch.cuda.synchronize();prepare_seconds+=time.monotonic()-t
   assert not any(a is not None and a.data_ptr() in (fd.data_ptr(),ad.data_ptr()) for a in args)
   outputs={'R0':{'mode_logits':args[6],'mode_prob':args[6].softmax(-1)}}
   for name,m in heads.items():
    torch.cuda.synchronize();t=time.monotonic();outputs[name]=head_forward(m,name,args,rf);torch.cuda.synchronize();elapsed[name]+=time.monotonic()-t
    if start==0:
     fdp=torch.full_like(fd,float('nan'));adp=torch.full_like(ad,float('nan'));again=head_forward(m,name,args,rf)
     assert torch.equal(outputs[name]['mode_logits'],again['mode_logits']) and torch.equal(outputs[name]['mode_prob'],again['mode_prob'])
     poison.append({'Fold':fold,'Model':name,'LabelPoisonMaxLogitDiff':0.,'LabelPoisonMaxProbabilityDiff':0.,'Status':'PASS'})
   bike=torch.as_tensor(store.types[ix]==2,device='cuda');r2=outputs['R2'];oracle=None;bike_metrics=None
   # All five new ablations route Bicycle to this fold's new R2.
   for axis,name in enumerate(MODELS):
    z=outputs[name]['mode_logits'].clone();p=outputs[name]['mode_prob'].clone()
    if name not in ('R0','R2'):z[bike]=r2['mode_logits'][bike];p[bike]=r2['mode_prob'][bike]
    assert torch.isfinite(z).all() and torch.isfinite(p).all()
    v=actor_metrics(z,p,fd,ad).cpu().numpy();rows=offset+ix
    if oracle is None:oracle=v[:,12:].copy()
    else:assert np.array_equal(oracle,v[:,12:]),'Common-candidate oracle metrics changed'
    if name=='R2':bike_metrics=v[bike.cpu().numpy()].copy()
    elif name not in ('R0','R2'):assert np.array_equal(bike_metrics,v[bike.cpu().numpy()]),'Bicycle routing changed'
    values[rows,axis]=v;logits[rows,axis]=z.cpu().numpy();probs[rows,axis]=p.cpu().numpy()
  for name,m in heads.items():assert state_digest(m.state_dict())==states[name] and not any(p.grad is not None for p in m.parameters())
  for group,keep in groups(store.frame).items():
   if keep.any():
    for axis,name in enumerate(MODELS):foldrows.append({'Fold':fold,'Group':group,'Model':name,'Count':int(keep.sum()),'Scenes':int(store.frame.loc[keep,'scene_token'].nunique()),**dict(zip(FIELDS,values[offset+store.ids[keep],axis].mean(0)))})
  for name in NAMES:runtime.append({'Fold':fold,'Model':name,'Actors':len(store.ids),'Parameters':PARAMS[name],'HeadForwardSeconds':elapsed[name],
       'HeadMSPerActor':1000*elapsed[name]/len(store.ids),'SharedFeaturePreparationSeconds':prepare_seconds,'UnifiedFoldEvaluationSeconds':time.monotonic()-startfold,
       'Scope':'cached inputs; head CUDA FP32; excludes predictor and scene feature construction and Bicycle route'})
  offset+=len(store.ids);del heads,store;torch.cuda.empty_cache();print('OUTER_FOLD_COMPLETE',fold,'rows',offset,flush=True)
 assert offset==n
 for a in (values,logits,probs):a.flush()
 f.to_csv(folder/'actor_records.csv',index=False);rows=[]
 for group,keep in groups(f).items():
  if keep.any():
   for axis,name in enumerate(MODELS):rows.append({'Fold':'Pooled','Group':group,'Model':name,'Count':int(keep.sum()),'Scenes':int(f.loc[keep,'scene_token'].nunique()),**dict(zip(FIELDS,values[keep,axis].mean(0)))})
 dump(ROOT/'stage15b_end_to_end_metrics.csv',foldrows+rows);dump(ROOT/'stage15b_ablation_metrics.csv',[x for x in foldrows+rows if x['Model'] not in ('R0','R2')]);dump(ROOT/'08_evaluation/stage15b_runtime.csv',runtime);dump(ROOT/'08_evaluation/stage15b_label_input_audit.csv',poison)
 audit={'Status':'PASS','ActorWindows':n,'Scenes':630,'Fields':FIELDS,'Models':MODELS,'IdentitiesUnique':True,'AllOracleMetricsBitwiseEqual':True,'BicycleNewR2RouteBitwiseEqual':True,
        'All18HeadsFrozenBeforeEvaluation':True,'HistoricalPredictionsLoaded':False,'OfficialProtocol':'custom internal evaluation, not nuScenes leaderboard',
        'SHA256':{p.name:sha256(p) for p in folder.iterdir() if p.is_file()}}
 atomic_json(folder/'complete.json',audit);atomic_json(ROOT/'08_evaluation/stage15b_identity_audit.json',audit)
 print('UNIFIED_OUTER_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 with torch.no_grad():main()
