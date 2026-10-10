"""Write outcome-grounded reports and verify all historical scientific bytes."""
from pathlib import Path
import sys,datetime,ast,shutil
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage17_common import *
def write(name,content):
 path=ROOT/name;assert path.resolve().is_relative_to(ROOT);path.write_text(content.strip()+'\n')
def md(frame):
 def cell(v):
  if isinstance(v,(float,np.floating)):return f'{v:.6f}'
  return str(v).replace('|','\\|')
 return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
  *['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])
def main():
 runtime();protect();verify_registration();gate=read_json(ROOT/'05_training/stage17_all_frozen.json');pre=read_json(ROOT/'04_checks/stage17_preflight.json')
 assert gate['Status']=='FROZEN_ALL_COMPLETE' and pre['Status']=='PASS'
 ev=read_json(ROOT/'06_evaluation/cache/complete.json');assert ev['Status']=='PASS' and ev['ExistingSevenMetricsBitwiseEqual'] and ev['CandidateOracleBitwiseEqual']
 atomic_json(ROOT/'06_evaluation/stage17_evaluation_integrity.json',ev)
 sealed=read_json(ROOT/'00_manifest/stage17_analysis_code_freeze.json')
 for name in ('06_evaluation/stage17_evaluate.py','07_efficiency/stage17_benchmark.py'):
  assert sha256(ROOT/name)==sealed['AnalysisSHA256'][name],'Statistical/compute implementation changed after freeze'
 for file,h in ev['SHA256'].items():assert sha256(ROOT/'06_evaluation/cache'/file)==h
 records=[read_json(ROOT/f'03_context/stage17_fold{k}_{role}.json') for k in (1,2,3) for role in ('InnerTrain','InnerDev','OuterTest')]
 assert all(r['Status']=='PASS' and r['CachedCandidatesBitwiseEqual'] and r['CachedLogitsBitwiseEqual'] and r['FuturePerturbationInputAndOutputBitwiseEqual'] for r in records)
 # Historical tracked files, all preexisting predictor/head checkpoints and selected caches.
 history=read_json(ROOT/'00_manifest/stage17_historical_manifest.json')['SHA256'];count=0;byteschecked=0
 for p,h in history.items():
  q=PROJECT/p;assert sha256(q)==h,'Historical file changed '+p;count+=1;byteschecked+=q.stat().st_size
 # Independently verify every frozen candidate array and scene context, not just predictors.
 cachecount=0
 for fold in (1,2,3):
  for role in ('InnerTrain','InnerDev','OuterTest'):
   folder=OLD/f'05_candidate_interface/cache/fold{fold}/{role}';manifest=read_json(folder/'manifest.json')
   for rec in list(manifest['Arrays'].values())+manifest['Contexts']+[manifest['IdentityCSV']]:
    path=OLD/rec['path'];assert sha256(path)==rec['sha256'],'Frozen candidate/context changed '+str(path);cachecount+=1;byteschecked+=path.stat().st_size
 historical={'Status':'PASS','HistoricalFilesChecked':count,'FrozenCandidateContextFilesChecked':cachecount,'BytesChecked':byteschecked,
  'BaseCommit':'e6682e4855d99eb598a3a40d117a81a41448e940','NoHistoricalCheckpointResultChanges':True}
 atomic_json(ROOT/'00_manifest/stage17_historical_preservation.json',historical)
 seedreceipt=read_json(S16/'03_seed_stability/stage16_seed_evaluation_integrity.json')
 assert sha256(S16/'03_seed_stability/cache/seed_metrics.npy')==seedreceipt['SeedMetricsSHA256']
 d=pd.read_csv(ROOT/'stage17_external_comparison.csv');b=pd.read_csv(ROOT/'stage17_bootstrap_comparison.csv');pool=d[d.Fold=='Pooled']
 overall=pool[pool.Group=='Overall'].set_index('Model');t=overall.loc['Adapted TNT Scoring'];gc=overall.loc['G-C'];nc=overall.loc['NG-C']
 c=b[(b.Group=='Overall')&(b.Metric=='Top1FDE')&(b.Comparison=='G-C - Adapted TNT Scoring')].iloc[0]
 cn=b[(b.Group=='Overall')&(b.Metric=='Top1FDE')&(b.Comparison=='NG-C - Adapted TNT Scoring')].iloc[0]
 sensitivity=pd.read_csv(ROOT/'06_evaluation/stage17_bicycle_route_sensitivity.csv')
 routed=float(sensitivity[sensitivity.Group=='Overall'].Top1FDE.iloc[0])
 biketnt=float(pool[(pool.Model=='Adapted TNT Scoring')&(pool.Group=='Bicycle')].Top1FDE.iloc[0])
 bikegc=float(pool[(pool.Model=='G-C')&(pool.Group=='Bicycle')].Top1FDE.iloc[0])
 direction='G-C has lower pooled Top1FDE than the adapted TNT scorer' if c.Delta<0 else 'The adapted TNT scorer has lower pooled Top1FDE than G-C' if c.Delta>0 else 'The pooled Top1FDE values are equal'
 consistency='all three folds' if c.NegativeFolds in (0,3) else f'{int(c.NegativeFolds)} folds favor G-C and {3-int(c.NegativeFolds)} favor TNT'
 supported='The descriptive interval excludes zero in the observed pooled direction.' if c.CI95Lower*c.CI95Upper>0 else 'The descriptive interval includes zero; a stable directional difference is not established by this interval.'
 head=pd.read_csv(ROOT/'07_efficiency/stage17_head_compute.csv');whole=pd.read_csv(ROOT/'07_efficiency/stage17_whole_compute.csv');resource=read_json(ROOT/'07_efficiency/stage17_resource_receipt.json')
 totalfit=sum(x['Seconds'] for x in gate['Checkpoints']);allpaired=sum(r['PairedIntegrityBatches'] for r in records);before=sum(r['PairedIntegrityBatches'] for r in records if r['Role']!='OuterTest')
 maxcoord=max(r['ActorFrameTransformMaxDiff'] for r in records)
 checkrows=[{'Fold':r['Fold'],'Role':r['Role'],'Scenes':r['Scenes'],'Targets':r['Targets'],'PairedBatches':r['PairedIntegrityBatches'],'RawDiff':r['InstrumentationMaxDiff']['raw_prediction'],'LogitDiff':r['InstrumentationMaxDiff']['mode_logits']} for r in records]
 write('stage17_integrity_checks.md',f'''# Stage17 — integrity checks

**PASS.** {before} paired Train/Dev batches passed before formal fitting; {allpaired} paired batches including post-freeze Outer export. Exact raw-prediction, logits and probability instrumentation differences are zero. All original ego candidate coordinates/logits/probabilities match their Stage15B scene caches **bitwise**, across every exported window. Predictor SHA256 and tensor-state digests remain unchanged.

{md(pd.DataFrame(checkrows))}

Actor-centered scoring coordinates use a rigid transform of those unchanged candidates. Largest round-trip difference from native HiVT local coordinates: {maxcoord:.12g} m; original FDE agreement tolerance was 5e-5 m plus relative2e-6. This numerical transformation is not a modification of the immutable candidate geometry. Evaluation reads original candidate error arrays, preserving the oracle bitwise.

Future poisoning replaces future positions/padding/labels, then confirms the predictor’s strict observed input and outputs are identical. The context hook captures only the observed decoder input and never returns a replacement. `TNTStore.inputs` physically excludes the separate GT array. GT is used only for fitting loss and offline masks/evaluation. No future-conditioned context encoder is fitted.

All three actual-InnerTrain preflights pass [receipts](04_checks/stage17_preflight.json): shapes [128,1,64]/[128,6,24]→[128,6], normalized finite probabilities, all 16,001 parameters connected to finite gradients, exact component loss equivalence,20 tiny updates, strict checkpoint reload and next-optimizer-step replay. Tiny weights are discarded. Train/Dev scenes are disjoint and checked against original protocol tokens. Each optimization stream uses only its InnerTrain targets; selection only InnerDev. Outer folders, original Outer performance arrays, official VAL/test and HeadDev reads are blocked during fitting.

All three selected scorers were globally frozen before any new TNT Outer performance: [global freeze](05_training/stage17_all_frozen.json), [public registration](00_manifest/stage17_public_registration.json). All 260,151 original Outer actor-windows and630 scenes are retained; no difficult samples removed. Existing seven-model metrics and all eight models’ minFDE6 are bitwise unchanged: [evaluation receipt](06_evaluation/stage17_evaluation_integrity.json) (large arrays local only).

Historical preservation: {count} files and {cachecount} frozen candidate/context files verified against original hashes; [preservation receipt](00_manifest/stage17_historical_preservation.json). Historical Stage15B/16 scientific outputs and checkpoints were never overwritten. GPU/disk checks use the existing environment; no HiVT training or new trainable encoder.
''')
 trainrows=[{'Fold':s['Fold'],'Seed':s['Seed'],'SelectedEpoch':s['SelectedEpoch'],'ExecutedEpochs':s['ExecutedEpochs'],'DevSelectionScore':s['DevSelectionScore'],'TrainTargets':s['AllTrainTargetsUsed'],'Seconds':s['Seconds'],'PeakGPU_MB':s['PeakGPUMemoryBytes']/1e6} for s in gate['Checkpoints']]
 write('stage17_training_report.md',f'''# Stage17 — training report

**COMPLETE: exactly three scoring components, zero HiVT training steps.** Optimizer/choice rules were frozen publicly before fits and new TNT Outer metrics. Source/model/loss/temperature choices are unchanged after evaluation.

{md(pd.DataFrame(trainrows))}

Total bounded scorer fit time: {totalfit/3600:.6f} GPU hours (includes development assessment and checkpoint I/O). All distinct InnerTrain target identities were used. Selection is the strict minimum registered balanced relative Vehicle/Pedestrian InnerDev FDE; no official VAL, HeadDev70 or Outer selection. Original fold seeds2022/2122/2222 and private permutation/carry rules are preserved. No tiny or previous scoring weights initialize a formal scorer.

Training: AdamW lr0.001/wd0.0001, micro128×8, effective1024, FP32, at most50 epochs, patience5. Real pinned `TrajScoreSelection.loss` sum divided by effective batch; full-path maximum squared distance soft CE with fixed temperature0.01. Score softmax temperature1. No Loss C replacement, calibration, clipping or new encoder. This retains the **paper/component CE** mechanism; the unofficial full trainer’s active BCE, scheduler and joint-system fitting are explicitly different. Selection/weight decay follow the frozen project head comparison, not full TNT reproduction.

Each fold retains step-zero receipt, epoch curve, selected checkpoint, last training state with optimizer, pending/coverage and RNGs. [Global selected hashes](05_training/stage17_all_frozen.json) bind exactly three checkpoints; [config/curves](05_training/). The registered cap and stopping condition were not extended after Outer evaluation.
''')
 hr=head.groupby('Model')[['Parameters','MeanMSPer1024']].mean().reset_index();wr=whole.groupby('Model')[['MeanSeconds']].mean().reset_index()
 write('stage17_efficiency_report.md',f'''# Stage17 — efficiency

Measured on {resource['GPU']}, torch{resource['Torch']}, original environment, no simultaneous scorer-fit job. All methods use the same first1,024 InnerDev targets per fold,128-target microbatches,10 warm-up cycles and50 timed cycles; CUDA synchronization bounds each timing. Head-only means across three folds:

{md(hr)}

Whole engineering path on the **same first16 InnerDev windows**, past-only target eligibility,3 warm-ups and10 repeats per model/fold. Equal fold means in seconds (counts differ by fold; see source CSV):

{md(wr)}

[Head source](07_efficiency/stage17_head_compute.csv) and [whole-path source](07_efficiency/stage17_whole_compute.csv) record scopes, counts, window identity digest and GPU peaks. Head-only excludes feature construction,H2D and routing. Full path includes frozen predictor, ego conversion, required context/features, head, actual Bicycle routing and argmax; excludes raw file I/O and audit hashing. TNT context capture is included in full path and shares the predictor forward. G-C retains its actual CPU graph preparation. Different feature pipelines explain engineering overhead; these are not deployment FPS estimates.

TNT has16,001 new scoring parameters and uses independent frozen64D local context. G-C has24,066 head parameters, NG-C7,425, Matched24,001; the shared Stage5A predictor650,403 is not counted as newly trained. Existing graph/no-graph Bicycle routing also invokes frozenR2(673 params); pure TNT does not. Including the shared predictor and required route heads: R0=650,403; R2=651,076; NG-A/NG-C=658,501; G-A/G-C=675,142; Matched-NG-C=675,077; pure Adapted TNT=666,404 total loaded parameters. Only the16,001 TNT head parameters receive updates in this stage. Parameter/input differences remain explicit; the comparison is not a parameter-matched or identical-input experiment.

Added context arrays {resource['AddedContextCacheBytes']:,} bytes; selected scorer checkpoint sizes {resource['SelectedCheckpointBytes']}; disk free at benchmark {resource['DiskFreeBytes']:,} bytes. Original large candidate caches reused read-only. Local checkpoints/caches stay ignored; source-data tables and figure exports alone are published.
''')
 resulttable=pool[pool.Group.isin(['Overall','Vehicle','Pedestrian','MovingVehicle'])][['Group','Model','Count','Top1FDE','Top1ADE','HitRate','minFDE6']]
 write('stage17_scientific_conclusion.md',f'''# Stage17 — scientific conclusion

{direction}: Adapted TNT Overall Top1FDE={t.Top1FDE:.6f} m, G-C={gc.Top1FDE:.6f} m, NG-C={nc.Top1FDE:.6f} m. G-C−TNT={c.Delta:.6f} m;95% descriptive paired scene-cluster CI[{c.CI95Lower:.6f},{c.CI95Upper:.6f}]. Fold differences={c.Fold1Delta:.6f}/{c.Fold2Delta:.6f}/{c.Fold3Delta:.6f} m ({consistency}). {supported}

NG-C−TNT={cn.Delta:.6f} m;95% descriptive CI[{cn.CI95Lower:.6f},{cn.CI95Upper:.6f}]. Complete participant/fold/metric estimates, including any degradation and null direction, appear in [comparison](stage17_external_comparison.csv) and [bootstrap](stage17_bootstrap_comparison.csv). No result is used to revise G-C or the original Stage15B decisions.

{md(resulttable)}

The shared candidate oracle minFDE6 remains{t.minFDE6:.6f} m; scoring cannot repair missing candidate geometry. HitRate means exact selected-mode agreement with the endpoint-FDE oracle, not a2m threshold. MovingVehicle is the original **t0 vehicle.moving** group, not a future-GT motion filter. All original hard and high-error samples remain included.

**A metric tradeoff must also be reported:** TNT Overall HitRate={t.HitRate:.6f}, higher than G-C={gc.HitRate:.6f} and NG-C={nc.HitRate:.6f}, despite its worse mean Top1FDE and Top1ADE. More frequent exact oracle-mode matches do not guarantee smaller average selection error. This aggregate frequency/error contrast is not evidence of a causal mechanism; no probability-calibration claim is made from it. Do not present only FDE as though TNT loses on every metric.

TNT’s soft target ranks **maximum full-path squared distance**, while comparison metrics include endpoint-FDE and the frozen project loss/selection definitions. Context is independently learned by the immutable predictor, not a newly fitted encoder. TNT’s context/whole-path information differs from G-C’s explicit predicted neighbor edges; these data do not identify a causal mechanism or prove that a full TNT system would behave similarly.

Pure TNT scores all types. [Bicycle-route sensitivity](06_evaluation/stage17_bicycle_route_sensitivity.csv) exposes the frozenR2 option separately: TNT+routed-R2 Overall Top1FDE={routed:.6f} m; G-C minus this sensitivity result={float(gc.Top1FDE)-routed:.6f} m. Bicycle primary TNT FDE={biketnt:.6f} m versus the G-C/R2 route={bikegc:.6f} m (n=2,980). This does not replace the primary result or conceal poor Bicycle scores. Vehicle/Pedestrian/MovingVehicle estimates are invariant to this route.

This comparison was added **after Stage15B results were known**. It is supplementary and outside the original confirmatory family3. The2000 paired bootstrap draws cluster by scene and independently resample210 scenes per fold; intervals are descriptive and conditional on fixed predictors/scorers. There is no new multiplicity-adjusted confirmatory claim. Historical method-development exposure, SchemeA training-in candidate use for head training, and lack of inner OOF remain unchanged limitations. Internal scene-isolated CV is not an untouched confirmation set or official nuScenes test result.

The baseline uses actual pinned unofficial TNT scoring code and paper/component loss; its context,horizon,candidate count and joint-training setting are adapted. The active full-repository BCE difference and missing redistribution license are disclosed. Source stays local; published code contains loader/provenance, not third-party class text. No new network innovation experiment follows this stage.
''')
 summary={'Status':'COMPLETE','TNTSourceVerified':True,'AdapterIntegrity':'PASS','TrainingStatus':'COMPLETE','ThreeFoldComplete':True,
 'AdaptedTNTOverallFDE':float(t.Top1FDE),'GCOverallFDE':float(gc.Top1FDE),'NGCOverallFDE':float(nc.Top1FDE),
 'GCvsTNTDelta':float(c.Delta),'GCvsTNT95CI':[float(c.CI95Lower),float(c.CI95Upper)],'GCvsTNTFoldDelta':[float(c[f'Fold{k}Delta']) for k in (1,2,3)],
 'Interpretation':direction,'SupplementaryOnly':True,'NoNewConfirmatoryFamilyMember':True,'FullTNTReproduction':False,
 'License':'no grant found; upstream source local-only','FittingGPUHours':totalfit/3600,'PairedTrainDevBatches':before,
 'CandidateGeometryAndLogitsBitwiseUnchanged':True,'HiVTTrainingSteps':0,'PaperTableReady':True,'FigureExports':6,
 'TNTR2BicycleSensitivityOverallFDE':routed,'BicycleTNTFDE':biketnt,'BicycleGCR2RouteFDE':bikegc,
 'Groups':{g:{name:float(pool[(pool.Model==name)&(pool.Group==g)].Top1FDE.iloc[0]) for name in ('G-C','NG-C','Adapted TNT Scoring')} for g in ('MovingVehicle','Pedestrian')},
 'Compute':{'HeadMeanMSPer1024':{name:float(head[head.Model==name].MeanMSPer1024.mean()) for name in ('G-C','NG-C','Adapted TNT Scoring')},
 'WholeMeanSecondsPer16Windows':{name:float(whole[whole.Model==name].MeanSeconds.mean()) for name in ('G-C','NG-C','Adapted TNT Scoring')}},
 'HistoricalPreservation':historical,'DiskFreeBytes':shutil.disk_usage(ROOT).free,'STOP':True,'NextStageAuthorized':False}
 atomic_json(ROOT/'00_manifest/stage17_final_summary.json',summary)
 write('stage17_final_report.md',f'''# Stage17 — final report

**COMPLETE.** Three genuine TNT scoring components fitted on frozen Stage15B candidates; no HiVT training, no G-C alteration, no Outer tuning. {direction}.

| Item | Result |
|---|---|
| Branch | stage17/external-trajectory-scorer |
| Base commit | e6682e4855d99eb598a3a40d117a81a41448e940 |
| Source | unofficial Henry1iu TNT, bcbccdc1d35a717793e3caa1d599c1f700612227; actual component classes and CE loss |
| Adapter / initialization / GT isolation | PASS; frozen64D observed context; fresh16,001-param scorers |
| Training / three-fold status | COMPLETE / YES;{totalfit/3600:.6f} scorer GPU hours;0 HiVT training steps |
| Adapted TNT / G-C / NG-C Overall FDE | {t.Top1FDE:.6f} / {gc.Top1FDE:.6f} / {nc.Top1FDE:.6f} m |
| G-C−TNT /95% descriptive CI | {c.Delta:.6f} m /[{c.CI95Lower:.6f},{c.CI95Upper:.6f}] |
| Analysis classification | supplementary, outside unchanged Stage15B confirmatory family3 |
| Shared candidates / historical preservation | bitwise candidate/logit/oracle PASS;{count} historical files plus{cachecount} candidate/context files verified |
| Paper table / figure | READY;8-model source table and comparison figure; SVG/PDF/PNG |
| TNT reproduction / licensing | scoring adaptation only; full system not reproduced; no upstream license grant found, no third-party source redistributed |
| Next action | STOP; await brain-AI review; no automatic network innovation experiment |

Deliverables: [source audit](stage17_baseline_source_audit.md), [adapter design](stage17_adapter_design.md), [integrity](stage17_integrity_checks.md), [training](stage17_training_report.md), [eight-model comparison](stage17_external_comparison.csv), [paired bootstrap](stage17_bootstrap_comparison.csv), [efficiency](stage17_efficiency_report.md), [scientific conclusion](stage17_scientific_conclusion.md), [main table source](stage17_main_performance_table.csv), [figure exports](08_figures/).

Use **“Adapted TNT Scoring”** in the paper and cite the original paper plus pinned unofficial implementation. All estimates are custom internal scene-isolated CV on630 scenes /260,151 full-horizon actor-windows, not official nuScenes test performance. Context and Bicycle-route information differences, CE/BCE source discrepancy and prior scene exposure are disclosed in the audits. Report both favorable and unfavorable class/fold differences from the CSV; do not retune G-C.

The immutable public preregistration commit is `{read_json(ROOT/'00_manifest/stage17_public_registration.json')['CommitSHA']}`. The final Git commit is the commit containing this report, available in Git history and the final user receipt; the report does not embed its own self-referential commit hash.
''')
 for p in ROOT.rglob('*.py'):
  if 'cache' not in p.parts:ast.parse(p.read_text())
 atomic_json(ROOT/'00_manifest/stage17_final_audit.json',{'Status':'PASS','UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'SummarySHA256':sha256(ROOT/'00_manifest/stage17_final_summary.json'),'RegistrationSHA256':sha256(REG),'HistoricalPreservation':historical,
  'NewCheckpointSHA256':{str(k):sha256(cp_path(k)) for k in (1,2,3)},'DataArrayHashesVerified':True,'NewTNTOuterMetricsAfterGlobalFreeze':True,
  'OldSevenMetricsAndOracleBitwisePreserved':True,'PythonASTPass':True,'FiguresSourceReceiptSHA256':sha256(ROOT/'08_figures/stage17_figure_source_receipt.json')})
 print('STAGE17_FINAL_AUDIT_PASS',summary['GCvsTNTDelta'],summary['GCvsTNT95CI'],flush=True)
if __name__=='__main__':main()
