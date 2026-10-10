"""Freeze sources and experiment choices BEFORE external Outer performance exists."""
from pathlib import Path
import sys,subprocess,datetime,shutil
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage17_common import *
def main():
 protect();assert not REG.exists(),'Registration is immutable'
 source=read_json(S16/'02_literature/TNT-Trajectory-Prediction_code_audit.json')
 for p,r in source['code'].items():assert sha256(S16/'02_literature/cache/TNT-Trajectory-Prediction'/p)==r['sha256']
 atomic_json(ROOT/'01_source/stage17_original_scoring_source.json',source)
 protocol=read_json(pc.PROTOCOL);folds=[]
 for f in protocol['folds']:
  p=f['parts'];assert [len(p[k]) for k in ('InnerTrain','InnerDev','OuterTest','QuarantinedHeadDev')]==[378,42,210,70]
  assert len(set.union(*(set(v) for v in p.values())))==700
  for a in p:
   for b in p:
    if a!=b:assert not set(p[a])&set(p[b])
  folds.append({'Fold':f['fold'],'Seed':f['seed'],'Parts':p,'OriginalSceneFiles':f['files']})
 paths=subprocess.check_output(['git','ls-files'],text=True).splitlines()
 frozen={p:sha256(PROJECT/p) for p in paths if not p.startswith(str(ROOT.relative_to(PROJECT))+'/') and (PROJECT/p).is_file()}
 for root in (OLD/'04_predictor_checkpoints',OLD/'07_rank_checkpoints',S16/'03_seed_stability/checkpoints'):
  for p in root.rglob('*.pt'):frozen[str(p.relative_to(PROJECT))]=sha256(p)
 for p in (OLD/'08_evaluation/cache').glob('*.npy'):frozen[str(p.relative_to(PROJECT))]=sha256(p)
 frozen[str((OLD/'08_evaluation/cache/actor_records.csv').relative_to(PROJECT))]=sha256(OLD/'08_evaluation/cache/actor_records.csv')
 for p in (PROJECT/'outputs/stage2c_trainval_vehicle_baseline/04_evaluation').glob('stage2c_qualitative_main_case_new*'):frozen[str(p.relative_to(PROJECT))]=sha256(p)
 atomic_json(ROOT/'00_manifest/stage17_historical_manifest.json',{'BaseCommit':'e6682e4855d99eb598a3a40d117a81a41448e940','SHA256':frozen})
 sources={str(p.relative_to(ROOT)):sha256(p) for p in ROOT.rglob('*.py') if 'cache' not in p.parts}
 registration={'Stage':'17','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'BaseCommit':'e6682e4855d99eb598a3a40d117a81a41448e940',
 'Status':'FROZEN_BEFORE_FIRST_NEW_OUTER_METRIC','Stage15BProtocolSHA256':sha256(pc.PROTOCOL),'FrozenSources':sources,'TNTCommit':source['commit'],
 'Predictors':read_json(OLD/'04_predictor_checkpoints/stage15b_all_frozen.json')['Checkpoints'],'Folds':folds,
 'Model':'Unmodified pinned TrajScoreSelection(C64,T12,H64),16001 params','Context':'detached decoder pre-hook local input, after type fusion; all observed local lane/actor context',
 'Candidates':'K6 same order and exact immutable ego candidates; rigid transform only for scorer input','Coordinates':'current actor t0 origin and observed actor orientation, meters, absolute future positions not increments',
 'Padding':'history mask handled by frozen HiVT; all 6 candidates valid; original full-horizon evaluation mask; no filled GT or candidate filtering',
 'Normalization':'no external centering/scaling of context; LayerNorm in actual source. Rigid actor-frame transform only, no learned moments',
 'Training':{'Folds':3,'Seed':[2022,2122,2222],'Optimizer':'AdamW','LR':.001,'WeightDecay':.0001,'Betas':[.9,.999],'Microbatch':128,'Accumulation':8,'EffectiveBatch':1024,
 'MaxEpochs':50,'EarlyStopPatience':5,'Selection':'strict minimum .5*VehicleDevFDE/R2VehicleDevFDE + .5*PedestrianDevFDE/R2PedestrianDevFDE; epoch1 first eligible',
 'EpochOrder':'same fold seed+epoch numpy permutations with continuous remainder carry as Stage15B','Precision':'FP32 deterministic, no gradient clipping',
 'Loss':'pinned scoring COMPONENT soft CE sum divided by effective batch; q=softmax(-max_t squared 2D candidate-GT distance/.01)',
 'LabelTemperature':.01,'InferenceSoftmaxTemperature':1.,'TemperatureTuning':False,'FullRepositoryTrainerDifference':'active TNTLoss uses BCE; not used here; explicitly reported',
 'BicyclePrimary':'pure TNT all types','BicycleSecondary':'same-fold frozen R2 routing sensitivity; never selects primary model',
 'Failure':'any nonfinite loss, identity/invariance/isolation failure stops; no candidate count/loss change or Outer repair; incomplete files preserved'},
 'Evaluation':{'Groups':['Overall','Vehicle','Pedestrian','Bicycle','MovingVehicle'],'Metrics':['Top1FDE','Top1ADE','HitRate','minFDE6'],
 'Bootstrap':'2000 paired scene clusters, seed2022, 210 scene resamples in each fold; actor weighted; same frozen Stage15B weights if exact scene order matches',
 'Comparisons':['G-C minus Adapted TNT','NG-C minus Adapted TNT'],'CI':'95% descriptive','Multiplicity':'supplementary, no confirmatory claims; outside original family3; no new G-C tuning',
 'NegativeDeltaMeaning':'existing method lower error','EmptyGroups':'report count and omit unavailable fold class comparisons; never delete difficult actors'},
 'Efficiency':{'HeadBatch':1024,'Warmup':10,'Repetitions':50,'WholeWindows':16,'Partition':'InnerDev only, original Stage16 fixed windows',
 'WholeWarmup':3,'WholeRepetitions':10,'GPU':'same RTX3080; CUDA synchronize; raw I/O and source hashes excluded; context extraction included'},
 'Licensing':'No license file in pinned 139-entry complete tree; upstream code kept local, no redistribution or open-source-license claim',
 'HiVTTrainingAllowed':False,'NewTrainableEncoderAllowed':False,'OuterGate':'all three selected scorer checkpoints frozen before first TNT Outer metrics',
 'FullTNTReproduction':False,'Budget':'reuse caches, export <1GB added arrays, 3 bounded small scorer fits; no HiVT fitting'}
 atomic_json(REG,registration);print('REGISTERED',sha256(REG),'historical files',len(frozen),flush=True)
if __name__=='__main__':main()
