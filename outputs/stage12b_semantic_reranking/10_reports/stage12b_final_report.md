# Stage12B Motion-Conditioned Pedestrian Semantic Reranking

Stage12B=STOP; SemanticIncrement=NOT_SUPPORTED; ReadyForNextStage=NO. Frozen-model, semantic-cache, candidate-identity, GT-feature isolation and fold-isolation checks pass. Vehicle outputs exactly preserve frozen Fold C Raw; Bicycle outputs exactly preserve Fold R2. No further training is authorized or performed.

S does not improve the primary outcome: pedestrian Top1FDE is1.318903m compared with C0=1.304968m, G=1.316236m and P=1.316583m. S−C0=+0.013935m (95% development-stage descriptive interval [+0.008777,+0.019396]). The predeclared control-failure rule determines NOT_SUPPORTED. The more permissive PARTIAL wording does not override the explicit failure rule; this conservative precedence was recorded before training in stage12b_protocol.json.

## Frozen sources and integrity

Base commit: `c2ef16631f6eb67da7d98097967558e7a65925db`. Branch: `stage12b/motion-conditioned-semantic-reranking`. All Stage12B artifacts and scripts stay in this stage's root. No historical source, report, checkpoint, temperature, normalization or Stage2C untracked drawing is modified. Final audit checks2775 historical files,20 frozen checkpoints,5 preserved untracked files and every frozen source-data hash.

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

| Fold | Model | SelectedEpoch | TrainingEpochs | SelectionScore | TrainingSeconds |
| --- | --- | --- | --- | --- | --- |
| 1 | G | 1 | 6 | 1.360602 | 13.709669 |
| 1 | S | 1 | 6 | 1.361183 | 13.163428 |
| 1 | P | 1 | 6 | 1.360975 | 13.211945 |
| 2 | G | 4 | 9 | 1.272573 | 19.662055 |
| 2 | S | 1 | 6 | 1.272526 | 13.129824 |
| 2 | P | 3 | 8 | 1.274786 | 17.426682 |
| 3 | G | 1 | 6 | 1.340808 | 12.883833 |
| 3 | S | 1 | 6 | 1.339656 | 12.833543 |
| 3 | P | 1 | 6 | 1.337923 | 12.761621 |

Training loss decreases in all variants, while InnerDev Top1FDE often stops improving early. Most selected checkpoints are epoch1; this follows the registered strict selector and patience. No initialization fallback or training rescue is introduced after results are seen.

## Development OOF results

Top1FDE in meters, lower is better:

| Group | Count | C0 | G | S | P |
| --- | --- | --- | --- | --- | --- |
| Overall | 260151 | 2.199512 | 2.202377 | 2.203055 | 2.202466 |
| Vehicle | 191026 | 2.516273 | 2.516273 | 2.516273 | 2.516273 |
| Pedestrian | 66145 | 1.304968 | 1.316236 | 1.318903 | 1.316583 |
| Bicycle | 2980 | 1.749867 | 1.749867 | 1.749867 | 1.749867 |
| MovingVehicle | 41728 | 9.906479 | 9.906479 | 9.906479 | 9.906479 |
| Pedestrian<5m | 24005 | 0.880861 | 0.889270 | 0.884659 | 0.886774 |
| Pedestrian>5m | 42140 | 1.546560 | 1.559456 | 1.566269 | 1.561423 |
| Pedestrian5-8m | 22483 | 1.529405 | 1.566389 | 1.577015 | 1.570819 |
| Pedestrian_walkway_valid_1 | 52800 | 1.321296 | 1.334476 | 1.334914 | 1.333799 |
| Pedestrian_walkway_valid_0 | 13345 | 1.240367 | 1.244067 | 1.255552 | 1.248470 |

Top1ADE and all fixed observable-speed groups are supplied in stage12b_oof_metrics.csv and stage12b_pedestrian_groups.csv. Overall S−C0=+0.003543m, entirely from pedestrian reranking because Vehicle/Bicycle are unchanged. The GT5–8m group includes22483 pedestrians; S−C0=+0.047610m with95% descriptive interval[+0.040135,+0.056100]. It is significantly worse under the registered descriptive rule, rather than hidden by a pooled result.

## Paired scene uncertainty and controls

2000 whole-scene bootstrap repetitions,seed2022,draw210 scenes with replacement within each frozen outerfold. All actors/windows in a drawn scene remain paired. Pooled means weight actor windows, not scenes equally. The exact historical weight matrix is independently regenerated. Reported intervals are development-stage descriptive intervals. The S−G and S−P family uses Bonferroni97.5% individual intervals (percentiles1.25/98.75) for95% family coverage. No actor-independent bootstrap is used.

| Comparison | Count | DeltaTop1FDE | CI95Lower | CI95Upper | FamilyAdjustedLower | FamilyAdjustedUpper |
| --- | --- | --- | --- | --- | --- | --- |
| S-G | 66145 | 0.002667 | -0.000943 | 0.006296 | -0.001522 | 0.006871 |
| S-P | 66145 | 0.002319 | 0.000403 | 0.004174 | 0.000133 | 0.004398 |
| S-C0 | 66145 | 0.013935 | 0.008777 | 0.019396 | — | — |
| G-C0 | 66145 | 0.011267 | 0.006458 | 0.016765 | — | — |
| P-C0 | 66145 | 0.011615 | 0.006765 | 0.017058 | — | — |

Fold S−G pedestrian differences are [-0.0003309215405655763, 0.01016394115267194, -0.0017483036313785671]. Two folds point downward, but the pooled primary difference points upward, so the required pooled conditions fail. S−P remains positive even under the adjusted interval. No independent semantic prediction gain is supported. GeometryControl=PASS and ShuffledControl=PASS mean the controls follow the matched experimental protocol; they do not mean S beats either control.

## Mode switches and cost

| Comparison | changed_count | improved_count | worsened_count | gross_gain | gross_harm | net_delta_FDE | mean_harm | p90_harm | p95_harm | p99_harm | HighCostHarmAbove5mCount |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S-C0 | 9571 | 4727 | 4844 | 1928.825824 | 2850.526682 | 0.013935 | 0.588465 | 1.339029 | 2.754977 | 4.397431 | 17 |
| G-C0 | 10736 | 5394 | 5342 | 2114.206386 | 2859.490893 | 0.011267 | 0.535285 | 1.209755 | 2.316362 | 4.348083 | 18 |
| P-C0 | 10013 | 5056 | 4957 | 2001.746460 | 2770.030277 | 0.011615 | 0.558812 | 1.260896 | 2.589927 | 4.396971 | 17 |

S has4844 wrong switches versus G5342 and P4957, reductions498 and113 respectively. Counts alone do not establish benefit: S loses more useful switches and its mean harm is higher. S gross harm2850.527m exceeds its gross gain1928.826m. Seventeen C0→S switches incur more than5m extra FDE; this fixed diagnostic threshold is unused for selection. Net pedestrian error worsens. Full paired S−G/S−P switch tables accompany the baseline-relative statistics.

## Probability diagnostics

The hard oracle label is the minimum-FDE candidate, with lowest-index tie breaking. Brier sums six squared errors. ECE uses ten equal-width confidence bins and hard-oracle Top1 agreement. NLL uses stable log-softmax without probability clipping saturation. No new calibration is fitted or claimed.

| Model | Count | RawNLL | Brier | ECE10 | ExpectedRegret | PredictionEntropy |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | 66145 | 33.915948 | 1.278961 | 0.626255 | 0.397968 | 0.110419 |
| G | 66145 | 1.672099 | 0.780202 | 0.050550 | 0.582050 | 1.420103 |
| S | 66145 | 1.650000 | 0.778370 | 0.051799 | 0.599509 | 1.440779 |
| P | 66145 | 1.664881 | 0.779400 | 0.052495 | 0.586957 | 1.426441 |

C0's raw probability is the original sharp C distribution; G/S/P start from the already fixed temperature-scaled teacher. Consequently, raw NLL/entropy differences mix known temperature scaling and the residual. The neutral scaled teacher is reported separately, with unchanged Top1: NLL=1.673582,Brier=0.787080,ECE10=0.019813,ExpectedRegret=0.736332,entropy=1.567238. It is a diagnostic of the frozen scale, not a fifth trained variant. Improvements in raw NLL over unscaled C0 do not establish semantic value or transfer Stage11C calibration to these changed logits.

## Efficiency and artifacts

Each new head has641 trainable parameters. The nine formal runs take 128.783s; this excludes preflight, tiny validation, frozen-C replay and reporting. Peak allocated CUDA memory during a formal run is93.475MiB, rather than total desktop/device reserved memory. FP32 cached residual device-forward latency ranges0.002221–0.002252ms per pedestrian with batch128. Export-inclusive timings are separate. These timings exclude the frozen predictor/C/R2 and HD Map geometry extraction.

CPU semantic matching calls during Stage12B training=0; all semantic inputs are reused from the frozen Stage12A cache. The prior extraction manifest records1092.400s wall time for its cached extraction run, with three CPU workers and earlier cached scenes. This is not a cold production feature latency and omits earlier preparation/runs. Cold online HD Map extraction cost is not measured in this stage. The feature extraction cost record is separate from residual inference latency.

Six required figure families are delivered: training curves, pedestrian FDE, fixed motion/coverage groups, real versus shuffled contrasts, switch counts/cost, and ten BEV cases. The cases are five largest gains and five largest harms, with actor-ID ties and distinct-scene preference, without semantic coverage filtering. Each shows all six candidates, GT, original walkway/map polygons, C0 and S choices in the same t0 ego coordinate system. These extreme examples are illustrative, not a representative sample. PNG/SVG/PDF exports and complete case identities are saved in09_figures.

## Scientific interpretation and boundary

This fixed lightweight residual is not supported: S worsens pooled pedestrian and Overall Top1FDE, fails both matched controls and degrades the5–8m subgroup. Walkway association observed in Stage12A does not establish incremental ranking benefit in this controlled structure. All three new heads worsen Top1FDE versus the strong frozen C baseline despite declining training objectives; the experiment does not isolate a causal explanation for that loss/outcome mismatch. Static walkway occupancy alone cannot be interpreted as live traffic-light state or yielding behavior.

The hypothesis originates in Stage12A, which already inspected these630 OOF scenes. This stage reuses the same development population and frozen splits. Stage5A previously encountered related historical training scenes. The evidence is internal development OOF, not a fully independent end-to-end test or unbiased confirmatory dataset. Descriptive intervals quantify paired scene variation without removing prior hypothesis inspection or training dependence.

Historical conclusions stay frozen: Stage11B ErrorAwareRanking=STRONG_SUPPORTED,VehicleImproved=YES,PedestrianImproved=YES,BicyclePreserved=YES; Stage11C CalibrationUseful=YES,Top1Identity=PASS; Stage12A PedestrianSemanticSignal=PROMISING,VehicleSemanticSignal=WEAK,SemanticIncrementalValue=SUGGESTED,GenuineIndependentPredictionGain=UNRESOLVED. Stage12B does not rewrite those conclusions.

Final fields: FrozenModelIntegrity=PASS;SemanticCacheIntegrity=PASS;GTLeakage=PASS;FoldIsolation=PASS;CandidateIdentity=PASS;VehiclePreserved=YES;BicyclePreserved=YES;GeometryControl=PASS;ShuffledControl=PASS;SemanticIncrement=NOT_SUPPORTED;PedestrianImproved=NO;Pedestrian5_8mDegraded=YES;OverallImproved=NO;Stage12B=STOP;ReadyForNextStage=NO.

STOP. Await 大脑AI review. No Stage12C, new semantic model, new feature/loss/seed search, official VAL/test evaluation or HiVT retraining is executed.
