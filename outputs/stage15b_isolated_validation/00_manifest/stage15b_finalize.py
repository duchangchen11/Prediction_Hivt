"""Evidence-first reports and immutable-history audit; never fits a model."""
from pathlib import Path
import sys,json,csv,hashlib,shutil,time
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
sys.path[:0]=[str(ROOT/'00_manifest')]
from stage15b_infrastructure import numpy_scalar_default
from stage15b_common import read_json,atomic_json,sha256,PROTOCOL
import pandas as pd


def write(name,body):
 (ROOT/name).write_text(body.rstrip()+'\n')


def table(rows):
 if not rows:return '(no rows)'
 columns=list(rows[0])
 def cell(value):
  if isinstance(value,float):return f'{value:.6f}'
  return str(value).replace('|','\\|').replace('\n',' ')
 return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',
   *['| '+' | '.join(cell(row.get(col,'')) for col in columns)+' |' for row in rows]])


def history():
 frozen=read_json(ROOT/'00_manifest/stage15b_frozen_history.json')
 for section in ('historical_files','checkpoints','preserved_untracked'):
  for path,h in frozen[section].items():assert sha256(PROJECT/path)==h,'Historical change '+path
 return {k:len(frozen[k]) for k in ('historical_files','checkpoints','preserved_untracked')}


def finalize():
 h=history();pred=ROOT/'04_predictor_checkpoints/stage15b_all_frozen.json';heads=ROOT/'07_rank_checkpoints/stage15b_all_frozen.json';stats=ROOT/'09_statistics/stage15b_decisions.json'
 complete=pred.exists() and heads.exists() and stats.exists() and (ROOT/'08_evaluation/cache/complete.json').exists()
 if not complete:
  status='INCOMPLETE';live=read_json(ROOT/'00_manifest/stage15b_live_status.json') if (ROOT/'00_manifest/stage15b_live_status.json').exists() else {}
  for name in ('stage15b_predictor_training_report.md','stage15b_predictor_checkpoint_audit.md','stage15b_candidate_provenance.md','stage15b_ranking_training_report.md','stage15b_failure_cases.md'):
   write(name,'# Stage15B INCOMPLETE\n\nNo complete three-fold experiment is available. Historical models/results are not used to fill missing runs.\n\nFailure/live status:\n\n```json\n'+json.dumps(live,indent=2)+'\n```\n')
  for name in ('stage15b_end_to_end_metrics.csv','stage15b_ablation_metrics.csv','stage15b_bootstrap_ci.csv'):
   if not (ROOT/name).exists():pd.DataFrame([{'Status':'INCOMPLETE','NoMissingMetricsImputed':True}]).to_csv(ROOT/name,index=False)
  write('stage15b_final_report.md','# Stage15B INCOMPLETE\n\nGraphIncrementSupported=INCOMPLETE\n\nLossIncrementSupported=INCOMPLETE\n\nCapacityAlternativeNotSufficient=INCOMPLETE\n\nNo OuterTest success claim. No historical checkpoint substitute. Frozen history audit PASS.\n\n'+json.dumps(live,indent=2)+'\n')
  atomic_json(ROOT/'00_manifest/stage15b_final_audit.json',{'Status':'INCOMPLETE','History':h,'NoHistoricalSubstitute':True,'STOP':True});return
 p=read_json(pred);s=read_json(heads);d=read_json(stats);identity=read_json(ROOT/'08_evaluation/stage15b_identity_audit.json')
 assert len(p['Checkpoints'])==3 and len(s['Checkpoints'])==18 and identity['Status']=='PASS'
 curves=[];trainrows=[]
 for fold in (1,2,3):
  folder=ROOT/f'04_predictor_checkpoints/fold{fold}'
  for phase in ('warmup','nll'):
   summary=read_json(folder/f'stage15b_{phase}_summary.json');assert summary['status']=='COMPLETE'
   trainrows.append({k:summary[k] for k in ('fold','seed','phase','phase_steps','best_global_step','best_overall_FDE','stop_reason','elapsed_seconds')})
   for row in summary['Monitoring']:
    curves.append({'Fold':fold,'Phase':phase,'GlobalStep':row['global_step'],'TrainLoss':row['training_loss_mean_last500_steps'],'LR':row['lr'],
     'BestStep':row['best_step'],'BestFDE':row['best_FDE'],'BadValidations':row['bad_validations'],'PeakAllocatedBytes':row['gpu_peak_allocated_bytes'],
     'PeakReservedBytes':row['gpu_peak_reserved_bytes'],'BlockSeconds':row['block_wall_seconds'],
     **{f'{g}_{m}':v[m] for g,v in row['measured']['metrics'].items() for m in ('Count','minADE6','minFDE6')}})
 pd.DataFrame(curves).to_csv(ROOT/'02_training/stage15b_predictor_curves.csv',index=False);pd.DataFrame(trainrows).to_csv(ROOT/'02_training/stage15b_predictor_training_summary.csv',index=False)
 cp_rows=[]
 for cp in p['Checkpoints']:
  assert sha256(ROOT/cp['path'])==cp['sha256'];check=read_json(ROOT/f"03_checks/stage15b_fold{cp['fold']}_checkpoint_checks.json");assert check['Status']=='PASS'
  cp_rows.append({k:cp[k] for k in ('fold','seed','path','sha256','params','warm_steps','nll_steps_executed','selected_global_step','checkpoint_dev_minFDE6','TrainSceneSHA256','DevSceneSHA256')})
 pd.DataFrame(cp_rows).to_csv(ROOT/'04_predictor_checkpoints/stage15b_checkpoint_manifest.csv',index=False)
 rankingrows=[]
 for cp in s['Checkpoints']:
  assert sha256(ROOT/f"07_rank_checkpoints/fold{cp['Fold']}/{cp['Model']}_best.pt")==cp['CheckpointSHA256']
  rankingrows.append({k:v for k,v in cp.items() if not isinstance(v,(dict,list))})
 pd.DataFrame(rankingrows).to_csv(ROOT/'06_rank_training/stage15b_ranking_summary.csv',index=False)
 write('stage15b_predictor_training_report.md','# Stage15B predictor training\n\nThree fresh Stage5A HiVTMotionAwareDecoder predictors, 650403 parameters each. Original type embedding, motion-conditioned residual decoder and HiVT backbone/K6 are retained. Scheme A exclusively uses InnerTrain378 and InnerDev42; Outer210 and HeadDev70 never contribute optimizer updates or selection. FP32, batch16, original AdamW decay groups/weight_decay0.0001; warmup5000/LR0.001, original NLL max16000/LR0.0001, validation every500, strict full-horizon Overall minFDE6, NLL patience5, final NLL best only. ADE monitoring uses the FDE-best mode as in the original metric function.\n\n'+table(trainrows)+'\n\nFull monitor curves: `02_training/stage15b_predictor_curves.csv`. Fresh source/config and every-batch source witness are retained locally. No old or tiny weights enter formal fitting.\n\nFold1 logging interruption after durable step1000: NumPy bool JSON serialization failed. The original fitting sources and protocol remain unchanged; a registered JSON-scalar wrapper repairs serialization. Exact same-checkpoint next-step loss, model, optimizer, batch, cursor and all RNG replay PASS before resuming at1001. The first replay diagnostic omitted frozen deterministic runtime flags and failed; its source/failure is retained, followed by a corrected deterministic check PASS. Four discarded diagnostic updates are separate from formal steps. No tuning or budget extension. See `00_manifest/stage15b_infrastructure_correction_receipt.json` and `03_checks/stage15b_resume_replay_detail.json`.\n')
 write('stage15b_predictor_checkpoint_audit.md','# Stage15B predictor checkpoints\n\n'+table(cp_rows)+'\n\nAll three new NLL checkpoints were frozen before candidate generation. Strict readback,650403 parameters,K6, GT-poison forward invariance, exact next-step optimizer/RNG replay and original raw timestamp/coordinate identity checks pass per-fold (`03_checks/stage15b_foldN_checkpoint_checks.json`). Own best warmup model/optimizer/RNG are restored; only LR changes to0.0001, NLL sampler resets with100000 epoch offset. Timestamps preserve original keyframe clock jitter; nominal5-past/12-future is not resampled to an invented exact six-second grid.\n')
 provenance=[];cache_bytes=0
 for fold in (1,2,3):
  report=read_json(ROOT/f'05_candidate_interface/stage15b_fold{fold}_provenance.json')
  for role in report['Roles']:
   cache_bytes+=sum(x['bytes'] for x in role['Arrays'].values())+sum(x['bytes'] for x in role['Contexts'])
   provenance.append({'Fold':fold,'Role':role['Role'],'Scenes':role['SceneCount'],'ActorWindows':role['Targets'],'Windows':role['StoredWindows'],
       'ContextActors':role['ContextActors'],'PartialTargetsExcluded':role['PartialTargetsExcluded'],'EmptySupervisionWindowsExcluded':role['EmptySupervisionWindowsExcluded'],
       'PredictorSHA':role['PredictorCheckpointSHA256']})
 write('stage15b_candidate_provenance.md','# Stage15B candidate provenance\n\n'+table(provenance)+'\n\nEach fold has one role-separated streamed cache shared by all six heads: scene/sample/instance/node identity, fold/seed/checkpoint SHA, original logits/probabilities, all6 trajectories, original timestamps, ego frame, history/candidate masks and separate complete supervised masks/labels. No sixfold cache duplication. All current actors remain graph context, including partial-future actors; only predetermined full-horizon eligible targets are scored. No difficult case exclusions. Empty-supervision windows contain no qualifying target and are transparently counted.\n\nFeature computation uses only observable history/type/candidates/logits/probabilities. GT is separate label data. Full graph/R2 normalization is fitted only on corresponding InnerTrain; `stage15b_foldN_normalization.json` includes source lists, counts and SHA. InnerTrain candidates are predictor training-in under Scheme A; this is disclosed rather than presented as inner OOF. Outer labels/errors are cached without aggregate evaluation or fitting access; a read guard forbids the Outer cache during head optimization.\n\nSmall per-fold provenance manifests retain full local cache/context SHA/size/identity lists. Large `.pt`/`.npy`/identity CSV caches stay local.\n')
 write('stage15b_ranking_training_report.md','# Stage15B ranking training\n\nAll18 heads are fresh and trained only on fold-specific InnerTrain candidates. R2 uses673 parameters, seed2022 in every fold, original19→32→1 softCE, batch1024 including final partial batch, AdamW0.001/wd0.0001,max50epoch/patience5, strict InnerDev Overall Top1FDE.\n\nOther heads use original15D nodes/17D edges/K6/radius50m/max8 neighbors; seeds2022/2122/2222, micro128×8=1024, continuous carry, common epoch order hashes, AdamW0.001/wd0.0001,max50/patience5. Strict Sdev=.5*VehicleDevFDE/R2VehicleDevFDE+.5*PedestrianDevFDE/R2PedestrianDevFDE. A is detached softCE q=softmax(-FDE/1m). C is mean sum softmax(z)*(FDE-minFDE)/max(1m,mean(FDE-minFDE)). No loss/architecture search. NG-A/NG-C7425; G-A/G-C24066; Matched-NG-C24001 active target-only parameters. Approximate capacity difference65 (0.2701%) and structural/optimization differences remain. Common additive score bias is softmax-unidentifiable, as in historical architecture.\n\nBicycle participates in all training targets/neighborhoods; all five ablation outputs route Bicycle to newly trained samefold R2. All18 heads freeze before unified Outer evaluation. No old head/predictor weights. Bounded new-data head checks discard their weights; original structural/capacity checks remain unchanged.\n\n'+table([{k:r[k] for k in ('Fold','Model','Params','SelectedEpoch','ExecutedEpochs','CheckpointScore','Seconds')} for r in rankingrows])+'\n')
 metrics=pd.read_csv(ROOT/'stage15b_end_to_end_metrics.csv');pooled=metrics[(metrics.Fold=='Pooled')&(metrics.Group.isin(['Overall','Vehicle','Pedestrian','MovingVehicle']))]
 ci=pd.read_csv(ROOT/'stage15b_bootstrap_ci.csv');primary=ci[ci.CoPrimary==True]
 free=shutil.disk_usage(ROOT).free;timings=pd.read_csv(ROOT/'09_statistics/stage15b_system_compute.csv')
 limitations='Internal training-isolated three-fold CV on historically developed scenes; not pristine confirmation. Scheme A uses predictor training-in candidates for ranker fitting. Bootstrap is conditional on three selected predictors/heads and does not quantify retraining seed uncertainty. Custom eligibility/t0 ego coordinates/K6/end-point miss/HitRate differ from official nuScenes top5/global-coordinate/full-horizon miss/OffRoadRate leaderboard requirements. No cross-protocol paper-number comparison. Capacity is approximately matched and no causal proof is claimed.'
 write('stage15b_final_report.md','# Stage15B Scene-Isolated End-to-End Validation\n\nStatus=COMPLETE\n\nThree fresh scene-isolated predictors and18 fresh heads are frozen; the exact378/42/210/70 partitions and seeds remain registered. Official VAL and HeadDev70 are excluded from fitting/selection. All historical tracked files, checkpoints and five untracked Stage2C artifacts pass immutable SHA checks.\n\n'+table(pooled[['Group','Model','Count','Scenes','Top1FDE','Top1ADE','minFDE6','HitRate']].to_dict('records'))+'\n\n## Registered primary comparisons\n\nNegative FDE delta means the first model improves. Three co-primary Overall Top1FDE comparisons;2000 paired whole-scene bootstrap,210 scenes independently resampled within each fold,seed2022. Report95% descriptive intervals and Bonferroni family3 coverage98.333333%. Support requires negative point, adjusted upper<0 and at least2/3 negative folds; no thresholds were changed.\n\n'+table(primary[['Comparison','Delta','CI95Lower','CI95Upper','BonferroniCILower','BonferroniCIUpper','Fold1Delta','Fold2Delta','Fold3Delta','NegativeFolds']].to_dict('records'))+'\n\n'+''.join(f"{k}={x['Decision']}\n\n" for k,x in d['Decisions'].items())+'## Integrity and supplementary analyses\n\nAll7 scoring outputs share exact candidate oracle metrics; all ablations share bitwise Bicycle R2 routing.260151 unique full-horizon actor-window targets,630 outer scenes, no ad hoc difficult-sample removal. Full fold/type/motion metrics and exclusions are reported. `stage15b_switch_harm.csv` exposes negative transfer/ties and high-cost switching; the failure case report includes exact identities, GT/candidates and selected modes. `stage15b_candidate_error_distributions.csv` reports train-in,Dev,Outer oracle distributions after freezes; it does not justify tuning. Head-only and complete engineering inference scopes are separated in compute tables.\n\n'+table(timings.to_dict('records'))+f'\n\nShared arrays/context cache measured bytes={cache_bytes}; free disk after experiment={free}. Predictor/head training times and peaks are recorded in training summaries and monitor curves; all large checkpoints/caches remain local.\n\n## Scientific limitations\n\n'+limitations+'\n\nSTOP. No new method-improvement stage is authorized by this report. Wait for brain-AI review.\n')
 atomic_json(ROOT/'00_manifest/stage15b_final_audit.json',{'Status':'PASS','Predictors':3,'RankingHeads':18,'History':h,'Identity':identity,
   'Decisions':{k:x['Decision'] for k,x in d['Decisions'].items()},'CacheBytes':cache_bytes,'DiskFreeBytes':free,
   'ScientificLimitations':limitations,'ProtocolSHA256':sha256(PROTOCOL),'AllRequiredDeliverables':True,'STOP':True})
 print('STAGE15B_REPORTS_COMPLETE',flush=True)

if __name__=='__main__':finalize()
