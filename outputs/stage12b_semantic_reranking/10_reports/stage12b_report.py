"""Evidence-led Stage12B report; no further model or calibration fitting."""
from pathlib import Path
import sys,platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12b_common import *
def markdown(rows,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in rows:
        values=[row[c] for c in columns];lines.append('| '+' | '.join(('—' if not np.isfinite(v) else f'{v:.6f}') if isinstance(v,(float,np.floating)) else str(v) for v in values)+' |')
    return '\n'.join(lines)
@torch.no_grad()
def main():
    seed();verify();decision=read_json(ROOT/'10_reports/stage12b_scientific_decision.json');assert read_json(ROOT/'10_reports/stage12b_verification.json')['Status']=='PASS'
    met=pd.read_csv(ROOT/'06_oof_evaluation/stage12b_oof_metrics.csv');boot=pd.read_csv(ROOT/'07_bootstrap/stage12b_bootstrap_ci.csv');prob=pd.read_csv(ROOT/'08_diagnostics/stage12b_probability_diagnostics.csv')
    cps=pd.read_csv(ROOT/'05_checkpoints/stage12b_checkpoint_manifest.csv');sw=pd.read_csv(ROOT/'08_diagnostics/stage12b_mode_switch_cost.csv');eff=pd.read_csv(ROOT/'08_diagnostics/stage12b_efficiency.csv')
    f=pd.read_csv(ROOT/'06_oof_evaluation/cache/stage12b_actor_records.csv');fd,ad=label_arrays();z=np.load(ROOT/'06_oof_evaluation/cache/stage12b_logits.npy',mmap_mode='r');n=len(f)
    # This diagnostic accounts for already fixed temperature scaling. It is not
    # a fifth trained variant, a new calibration or an alternative selector.
    teacher_z=np.zeros((n,6),np.float32);teacher_p=np.zeros_like(teacher_z)
    for fold in (1,2,3):
        ix=indices(fold,'OuterTest',True)
        for start in range(0,len(ix),128):
            ids=ix[start:start+128];base=torch.from_numpy(np.array(z[ids,0],copy=True)).cuda()/TEMPERATURES[fold]
            teacher_z[ids]=base.cpu().numpy();teacher_p[ids]=base.softmax(-1).cpu().numpy()
    keep=f.agent_type_id.to_numpy()==1;pp=teacher_p[keep].astype(np.float64);zz=teacher_z[keep].astype(np.float64);zz-=zz.max(-1,keepdims=True);lp=zz-np.log(np.exp(zz).sum(-1,keepdims=True));oracle=fd[keep].argmin(-1)
    conf=pp.max(-1);hit=pp.argmax(-1)==oracle;bins=np.minimum((conf*10).astype(int),9)
    teacher=dict(Diagnostic='C0 with existing fixed foldT, neutral residual0',Count=int(keep.sum()),RawNLL=float(-lp[np.arange(len(pp)),oracle].mean()),
        Brier=float(((pp-np.eye(6)[oracle])**2).sum(-1).mean()),ECE10=sum(float((bins==b).mean())*abs(float(conf[bins==b].mean())-float(hit[bins==b].mean())) for b in range(10) if (bins==b).any()),
        ExpectedRegret=float((pp*(fd[keep]-fd[keep].min(-1,keepdims=True))).sum(-1).mean()),PredictionEntropy=float(-(pp*np.log(np.maximum(pp,1e-30))).sum(-1).mean()))
    dump('08_diagnostics/stage12b_fixed_temperature_teacher_diagnostic.csv',[teacher])
    original_top=np.load(ROOT/'06_oof_evaluation/cache/stage12b_modes.npy',mmap_mode='r');assert np.array_equal(teacher_p[keep].argmax(-1),original_top[keep,0])
    runtime=dict(Python=platform.python_version(),Environment=sys.executable,Torch=torch.__version__,CUDA=torch.version.cuda,GPU=torch.cuda.get_device_name(0),
        Numpy=np.__version__,Device='FP32 CUDA; AMP disabled',FrozenProtocolSHA256=sha256(PROTOCOL))
    atomic_json(ROOT/'00_manifest/stage12b_runtime.json',runtime)
    extraction=read_json(S12A/'02_semantic_cache/stage12a_semantic_cache_manifest.json')['Seconds']
    atomic_json(ROOT/'08_diagnostics/stage12b_semantic_extraction_cost.json',dict(Stage12BMatchingCallsDuringTraining=0,Stage12BMapMatchingSeconds=0,
        SourceStage12AManifestSHA256=sha256(S12A/'02_semantic_cache/stage12a_semantic_cache_manifest.json'),RecordedStage12AExtractionWallSeconds=extraction,
        Scope='historical cached extraction run, three CPU workers plus earlier cached scenes; excludes earlier runs/cache preparation; not cold production inference latency',
        ColdFeatureExtractionLatency='not measured in Stage12B; no HD Map extraction repeated during training'))
    main_rows=[]
    for group in ['Overall','Vehicle','Pedestrian','Bicycle','MovingVehicle','Pedestrian<5m','Pedestrian>5m','Pedestrian5-8m','Pedestrian_walkway_valid_1','Pedestrian_walkway_valid_0']:
        subset=met[met.Group==group].set_index('Model');main_rows.append(dict(Group=group,Count=int(subset.loc['C0','Count']),**{name:float(subset.loc[name,'Top1FDE']) for name in MODELS}))
    comparison=boot[(boot.Group=='Pedestrian')&boot.Comparison.isin(['S-G','S-P','S-C0','G-C0','P-C0'])]
    switch=sw[(sw.Group=='Pedestrian')&sw.Comparison.isin(['G-C0','S-C0','P-C0'])]
    probability=prob[prob.Group=='Pedestrian'];sub=boot[(boot.Group=='Pedestrian5-8m')&(boot.Comparison=='S-C0')].iloc[0]
    cases=pd.read_csv(ROOT/'09_figures/stage12b_case_manifest.csv');loss=pd.read_csv(ROOT/'04_training/stage12b_training_curves.csv')
    report=f'''# Stage12B Motion-Conditioned Pedestrian Semantic Reranking

Stage12B=STOP; SemanticIncrement=NOT_SUPPORTED; ReadyForNextStage=NO. Frozen-model, semantic-cache, candidate-identity, GT-feature isolation and fold-isolation checks pass. Vehicle outputs exactly preserve frozen Fold C Raw; Bicycle outputs exactly preserve Fold R2. No further training is authorized or performed.

S does not improve the primary outcome: pedestrian Top1FDE is1.318903m compared with C0=1.304968m, G=1.316236m and P=1.316583m. S−C0=+0.013935m (95% development-stage descriptive interval [+0.008777,+0.019396]). The predeclared control-failure rule determines NOT_SUPPORTED. The more permissive PARTIAL wording does not override the explicit failure rule; this conservative precedence was recorded before training in stage12b_protocol.json.

## Frozen sources and integrity

Base commit: `{BASE}`. Branch: `stage12b/motion-conditioned-semantic-reranking`. All Stage12B artifacts and scripts stay in this stage's root. No historical source, report, checkpoint, temperature, normalization or Stage2C untracked drawing is modified. Final audit checks2775 historical files,20 frozen checkpoints,5 preserved untracked files and every frozen source-data hash.

Stage5A candidate predictor SHA256=`88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`. Each fold uses its own original C and R2 checkpoint for InnerTrain,InnerDev and OuterTest. Original graph normalizers are executed unchanged. C's full OuterTest raw logits and probabilities reproduce the Stage11B outputs bitwise; Bicycle routing reproduces R2 bitwise. Concatenated historical OOF C logits are never used as InnerTrain/InnerDev input. The residual training process forbids reading both historical OOF prediction files and new OOF results.

Temperatures remain T1=93.47459065453494,T2=41.89311387594937,T3=77.85669786831706. There is no temperature refit. The residual uses C_raw/T; Vehicle and Bicycle preserve raw frozen probabilities through direct copying.

Candidate coordinates, mode indices and per-candidate geometry hashes match all1560906 Stage12A semantic records. Every original630-scene NPZ and merged semantic array is verified. All four variants reference the same frozen six-coordinate trajectories, without generating altered copies. Candidate maxdiff=0; minFDE6, minADEOracle6 and MR6 remain bitwise identical. Neutral residual0 preserves the C0 Top1 for595305 fold/partition/variant pedestrian evaluations.

GT is used for training costs and offline evaluation only. The input function accepts frozen observed/candidate geometry and frozen semantic arrays, with no GT/error/motion-label arguments. The historical Stage12A upstream GT-poison audit remains verified. The new metadata-poison exercise documents schema isolation; poisoned future columns are not arguments to the feature extractor. Normalization, matching and semantic permutations do not consult GT. GT displacement-based motion groups are offline diagnostics, never inputs or selectors.

## Registered controlled experiment

Population=630 HeadTrain scenes/260151 full-horizon actor windows: Vehicle191026,Pedestrian66145,Bicycle2980. Original threefold splits remain378 InnerTrain/42 InnerDev/210 OuterTest scenes; each scene is tested in exactly one fold. HeadDev70, official VAL150 and test data are unused.

All G/S/P heads have18 inputs,32 hidden units,ReLU,two Linear layers and641 parameters. Inputs are the same eight frozen Stage12A geometry/history/interaction fields, eight validity masks and two semantic slots. G sets both semantic slots to zero; S uses walkway_inside_fraction and walkway_valid; P jointly permutes their six candidate records within each actor, independently by fold and partition with seed2022. Training, development and testing all use P's shuffled semantics. No missing actor is dropped; fraction0+mask0 represents missing walkway. Actor coverage groups use the same fixed ANY-of-original-six valid-mask population for all models; selected-mode coverage is a separate diagnostic.

The eight continuous variables are normalized from valid InnerTrain pedestrian candidate records only, with population standard deviation and floor1e−6. G/S/P share each fold's normalizer; fraction and masks are unscaled. Trajectory curvature retains the frozen absolute wrapped turn-angle proxy in radians, without redefining it as inverse-meter curvature.

Delta=2*tanh(MLP(x)). The loss is normalized expected FDE regret+0.1 KL(frozen scaled teacher||student)+0.001 mean(delta²). Only pedestrians contribute gradients. AdamW lr0.001,weight_decay0.0001,FP32,noAMP,microbatch128,effectivebatch512,accumulation4,max50epochs,patience5. Fold seeds2022/2122/2222 and actor order are fixed. Pending occurrences below512 are carried to the next epoch; all unique InnerTrain pedestrians receive updates. No architecture, coefficients, features, seed or training budget is searched.

The fixed128-target tiny gate runs300 updates per variant. Loss decreases G36.70%,S37.17%,P37.26%; gradients are finite and nonzero and all frozen parameter gradients remain zero. Tiny weights are discarded. Every variant starts from the same neutral initialization within its fold, rather than another trained head.

All nine trained checkpoints are selected only by minimum InnerDev pedestrian Top1FDE; exact ties keep the earliest epoch. Step0 is an initialization audit, not a selectable formal checkpoint. All nine checkpoint SHAs, selected epochs and training/split/normalization/base-C/temperature/semantic-cache identities are frozen before residual OuterTest inference. Final verification independently replays all nine outputs bitwise.

{markdown(cps.to_dict('records'),['Fold','Model','SelectedEpoch','TrainingEpochs','SelectionScore','TrainingSeconds'])}

Training loss decreases in all variants, while InnerDev Top1FDE often stops improving early. Most selected checkpoints are epoch1; this follows the registered strict selector and patience. No initialization fallback or training rescue is introduced after results are seen.

## Development OOF results

Top1FDE in meters, lower is better:

{markdown(main_rows,['Group','Count',*MODELS])}

Top1ADE and all fixed observable-speed groups are supplied in stage12b_oof_metrics.csv and stage12b_pedestrian_groups.csv. Overall S−C0=+0.003543m, entirely from pedestrian reranking because Vehicle/Bicycle are unchanged. The GT5–8m group includes22483 pedestrians; S−C0={sub.DeltaTop1FDE:+.6f}m with95% descriptive interval[{sub.CI95Lower:+.6f},{sub.CI95Upper:+.6f}]. It is significantly worse under the registered descriptive rule, rather than hidden by a pooled result.

## Paired scene uncertainty and controls

2000 whole-scene bootstrap repetitions,seed2022,draw210 scenes with replacement within each frozen outerfold. All actors/windows in a drawn scene remain paired. Pooled means weight actor windows, not scenes equally. The exact historical weight matrix is independently regenerated. Reported intervals are development-stage descriptive intervals. The S−G and S−P family uses Bonferroni97.5% individual intervals (percentiles1.25/98.75) for95% family coverage. No actor-independent bootstrap is used.

{markdown(comparison.to_dict('records'),['Comparison','Count','DeltaTop1FDE','CI95Lower','CI95Upper','FamilyAdjustedLower','FamilyAdjustedUpper'])}

Fold S−G pedestrian differences are {decision['FoldPedestrianSMinusG']}. Two folds point downward, but the pooled primary difference points upward, so the required pooled conditions fail. S−P remains positive even under the adjusted interval. No independent semantic prediction gain is supported. GeometryControl=PASS and ShuffledControl=PASS mean the controls follow the matched experimental protocol; they do not mean S beats either control.

## Mode switches and cost

{markdown(switch.to_dict('records'),['Comparison','changed_count','improved_count','worsened_count','gross_gain','gross_harm','net_delta_FDE','mean_harm','p90_harm','p95_harm','p99_harm','HighCostHarmAbove5mCount'])}

S has4844 wrong switches versus G5342 and P4957, reductions498 and113 respectively. Counts alone do not establish benefit: S loses more useful switches and its mean harm is higher. S gross harm2850.527m exceeds its gross gain1928.826m. Seventeen C0→S switches incur more than5m extra FDE; this fixed diagnostic threshold is unused for selection. Net pedestrian error worsens. Full paired S−G/S−P switch tables accompany the baseline-relative statistics.

## Probability diagnostics

The hard oracle label is the minimum-FDE candidate, with lowest-index tie breaking. Brier sums six squared errors. ECE uses ten equal-width confidence bins and hard-oracle Top1 agreement. NLL uses stable log-softmax without probability clipping saturation. No new calibration is fitted or claimed.

{markdown(probability.to_dict('records'),['Model','Count','RawNLL','Brier','ECE10','ExpectedRegret','PredictionEntropy'])}

C0's raw probability is the original sharp C distribution; G/S/P start from the already fixed temperature-scaled teacher. Consequently, raw NLL/entropy differences mix known temperature scaling and the residual. The neutral scaled teacher is reported separately, with unchanged Top1: NLL={teacher['RawNLL']:.6f},Brier={teacher['Brier']:.6f},ECE10={teacher['ECE10']:.6f},ExpectedRegret={teacher['ExpectedRegret']:.6f},entropy={teacher['PredictionEntropy']:.6f}. It is a diagnostic of the frozen scale, not a fifth trained variant. Improvements in raw NLL over unscaled C0 do not establish semantic value or transfer Stage11C calibration to these changed logits.

## Efficiency and artifacts

Each new head has641 trainable parameters. The nine formal runs take {float(cps.TrainingSeconds.sum()):.3f}s; this excludes preflight, tiny validation, frozen-C replay and reporting. Peak allocated CUDA memory during a formal run is{float(cps.PeakGPUMemoryMB.max()):.3f}MiB, rather than total desktop/device reserved memory. FP32 cached residual device-forward latency ranges{float(eff.ResidualDeviceMSPerPedestrian.min()):.6f}–{float(eff.ResidualDeviceMSPerPedestrian.max()):.6f}ms per pedestrian with batch128. Export-inclusive timings are separate. These timings exclude the frozen predictor/C/R2 and HD Map geometry extraction.

CPU semantic matching calls during Stage12B training=0; all semantic inputs are reused from the frozen Stage12A cache. The prior extraction manifest records{extraction:.3f}s wall time for its cached extraction run, with three CPU workers and earlier cached scenes. This is not a cold production feature latency and omits earlier preparation/runs. Cold online HD Map extraction cost is not measured in this stage. The feature extraction cost record is separate from residual inference latency.

Six required figure families are delivered: training curves, pedestrian FDE, fixed motion/coverage groups, real versus shuffled contrasts, switch counts/cost, and ten BEV cases. The cases are five largest gains and five largest harms, with actor-ID ties and distinct-scene preference, without semantic coverage filtering. Each shows all six candidates, GT, original walkway/map polygons, C0 and S choices in the same t0 ego coordinate system. These extreme examples are illustrative, not a representative sample. PNG/SVG/PDF exports and complete case identities are saved in09_figures.

## Scientific interpretation and boundary

This fixed lightweight residual is not supported: S worsens pooled pedestrian and Overall Top1FDE, fails both matched controls and degrades the5–8m subgroup. Walkway association observed in Stage12A does not establish incremental ranking benefit in this controlled structure. All three new heads worsen Top1FDE versus the strong frozen C baseline despite declining training objectives; the experiment does not isolate a causal explanation for that loss/outcome mismatch. Static walkway occupancy alone cannot be interpreted as live traffic-light state or yielding behavior.

The hypothesis originates in Stage12A, which already inspected these630 OOF scenes. This stage reuses the same development population and frozen splits. Stage5A previously encountered related historical training scenes. The evidence is internal development OOF, not a fully independent end-to-end test or unbiased confirmatory dataset. Descriptive intervals quantify paired scene variation without removing prior hypothesis inspection or training dependence.

Historical conclusions stay frozen: Stage11B ErrorAwareRanking=STRONG_SUPPORTED,VehicleImproved=YES,PedestrianImproved=YES,BicyclePreserved=YES; Stage11C CalibrationUseful=YES,Top1Identity=PASS; Stage12A PedestrianSemanticSignal=PROMISING,VehicleSemanticSignal=WEAK,SemanticIncrementalValue=SUGGESTED,GenuineIndependentPredictionGain=UNRESOLVED. Stage12B does not rewrite those conclusions.

Final fields: FrozenModelIntegrity=PASS;SemanticCacheIntegrity=PASS;GTLeakage=PASS;FoldIsolation=PASS;CandidateIdentity=PASS;VehiclePreserved=YES;BicyclePreserved=YES;GeometryControl=PASS;ShuffledControl=PASS;SemanticIncrement=NOT_SUPPORTED;PedestrianImproved=NO;Pedestrian5_8mDegraded=YES;OverallImproved=NO;Stage12B=STOP;ReadyForNextStage=NO.

STOP. Await 大脑AI review. No Stage12C, new semantic model, new feature/loss/seed search, official VAL/test evaluation or HiVT retraining is executed.
'''
    (ROOT/'10_reports/stage12b_final_report.md').write_text(report)
    atomic_json(ROOT/'10_reports/stage12b_delivery_audit.json',dict(Status='PASS',Science=decision,IndependentVerification='PASS',TrainingSeconds=float(cps.TrainingSeconds.sum()),
        ParamsPerHead=641,FormalCheckpoints=9,All9CheckpointSHA256=read_json(ROOT/'05_checkpoints/stage12b_all9_frozen.json')['Checkpoints'],
        ReportSHA256=sha256(ROOT/'10_reports/stage12b_final_report.md'),FiguresSHA256=sha256(ROOT/'09_figures/stage12b_figure_audit.json'),Runtime=runtime))
    print('STAGE12B_REPORT_COMPLETE_STOP',flush=True)
if __name__=='__main__':main()
