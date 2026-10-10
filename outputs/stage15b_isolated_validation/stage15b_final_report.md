# Stage15B Scene-Isolated End-to-End Validation

Status=COMPLETE

Three fresh scene-isolated predictors and18 fresh heads are frozen; the exact378/42/210/70 partitions and seeds remain registered. Official VAL and HeadDev70 are excluded from fitting/selection. All historical tracked files, checkpoints and five untracked Stage2C artifacts pass immutable SHA checks.

| Group | Model | Count | Scenes | Top1FDE | Top1ADE | minFDE6 | HitRate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | R0 | 260151 | 630 | 2.372741 | 1.056338 | 1.266202 | 0.451830 |
| Overall | R2 | 260151 | 630 | 2.346574 | 1.044426 | 1.266202 | 0.455017 |
| Overall | NG-A | 260151 | 630 | 2.375782 | 1.064636 | 1.266202 | 0.439276 |
| Overall | NG-C | 260151 | 630 | 2.306543 | 1.020819 | 1.266202 | 0.464492 |
| Overall | G-A | 260151 | 630 | 2.361641 | 1.063987 | 1.266202 | 0.440198 |
| Overall | G-C | 260151 | 630 | 2.261161 | 0.999894 | 1.266202 | 0.470369 |
| Overall | Matched-NG-C | 260151 | 630 | 2.283684 | 1.009464 | 1.266202 | 0.467475 |
| Vehicle | R0 | 191026 | 629 | 2.731294 | 1.198128 | 1.390441 | 0.489389 |
| Vehicle | R2 | 191026 | 629 | 2.698962 | 1.183550 | 1.390441 | 0.491310 |
| Vehicle | NG-A | 191026 | 629 | 2.741044 | 1.212796 | 1.390441 | 0.466460 |
| Vehicle | NG-C | 191026 | 629 | 2.648431 | 1.153161 | 1.390441 | 0.497199 |
| Vehicle | G-A | 191026 | 629 | 2.720192 | 1.209955 | 1.390441 | 0.464895 |
| Vehicle | G-C | 191026 | 629 | 2.588150 | 1.125391 | 1.390441 | 0.502497 |
| Vehicle | Matched-NG-C | 191026 | 629 | 2.616192 | 1.137288 | 1.390441 | 0.501680 |
| Pedestrian | R0 | 66145 | 542 | 1.353283 | 0.652980 | 0.914364 | 0.341356 |
| Pedestrian | R2 | 66145 | 542 | 1.346393 | 0.649446 | 0.914364 | 0.348054 |
| Pedestrian | NG-A | 66145 | 542 | 1.339737 | 0.644469 | 0.914364 | 0.357911 |
| Pedestrian | NG-C | 66145 | 542 | 1.334883 | 0.644362 | 0.914364 | 0.368312 |
| Pedestrian | G-A | 66145 | 542 | 1.344339 | 0.650122 | 0.914364 | 0.366059 |
| Pedestrian | G-C | 66145 | 542 | 1.330484 | 0.642264 | 0.914364 | 0.376128 |
| Pedestrian | Matched-NG-C | 66145 | 542 | 1.338082 | 0.645543 | 0.914364 | 0.367103 |
| MovingVehicle | R0 | 41728 | 597 | 10.874690 | 4.687916 | 5.343627 | 0.214101 |
| MovingVehicle | R2 | 41728 | 597 | 10.724894 | 4.619718 | 5.343627 | 0.221338 |
| MovingVehicle | NG-A | 41728 | 597 | 10.863803 | 4.722059 | 5.343627 | 0.248179 |
| MovingVehicle | NG-C | 41728 | 597 | 10.470631 | 4.470499 | 5.343627 | 0.222153 |
| MovingVehicle | G-A | 41728 | 597 | 10.714765 | 4.681714 | 5.343627 | 0.273989 |
| MovingVehicle | G-C | 41728 | 597 | 10.210756 | 4.350547 | 5.343627 | 0.234087 |
| MovingVehicle | Matched-NG-C | 41728 | 597 | 10.324218 | 4.398571 | 5.343627 | 0.227138 |

## Registered primary comparisons

Negative FDE delta means the first model improves. Three co-primary Overall Top1FDE comparisons;2000 paired whole-scene bootstrap,210 scenes independently resampled within each fold,seed2022. Report95% descriptive intervals and Bonferroni family3 coverage98.333333%. Support requires negative point, adjusted upper<0 and at least2/3 negative folds; no thresholds were changed.

| Comparison | Delta | CI95Lower | CI95Upper | BonferroniCILower | BonferroniCIUpper | Fold1Delta | Fold2Delta | Fold3Delta | NegativeFolds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G-C-NG-C | -0.045382 | -0.064219 | -0.029747 | -0.068976 | -0.026489 | -0.047455 | -0.029450 | -0.057778 | 3 |
| G-C-Matched-NG-C | -0.022523 | -0.032962 | -0.013724 | -0.035387 | -0.011727 | -0.042529 | -0.015297 | -0.011335 | 3 |
| NG-C-NG-A | -0.069239 | -0.098942 | -0.039383 | -0.104990 | -0.031506 | -0.029504 | -0.152368 | -0.030058 | 3 |

GraphIncrementSupported=SUPPORTED

LossIncrementSupported=SUPPORTED

CapacityAlternativeNotSufficient=SUPPORTED

## Integrity and supplementary analyses

All7 scoring outputs share exact candidate oracle metrics; all ablations share bitwise Bicycle R2 routing.260151 unique full-horizon actor-window targets,630 outer scenes, no ad hoc difficult-sample removal. Full fold/type/motion metrics and exclusions are reported. `stage15b_switch_harm.csv` exposes negative transfer/ties and high-cost switching; the failure case report includes exact identities, GT/candidates and selected modes. `stage15b_candidate_error_distributions.csv` reports train-in,Dev,Outer oracle distributions after freezes; it does not justify tuning. Head-only and complete engineering inference scopes are separated in compute tables.

| Fold | Model | Windows | CurrentActors | PastEligibleTargets | PurePredictorMeanSeconds | FullSystemMeanSeconds | AddedSeconds | TimedRepeats | PeakAllocatedBytes | Scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | R2 | 16 | 1014 | 962 | 0.084562 | 0.280898 | 0.196336 | 20 | 948196864 | fixed same16 InnerDev windows; preloaded predictor input, predictor+ego conversion+CPU graph/R2 features+normalization+H2D+head+Bicycle route; excludes raw data I/O and audit SHA capture; engineering path, not optimized deployment FPS |
| 1 | G-C | 16 | 1014 | 962 | 0.084562 | 0.301171 | 0.216608 | 20 | 948196864 | fixed same16 InnerDev windows; preloaded predictor input, predictor+ego conversion+CPU graph/R2 features+normalization+H2D+head+Bicycle route; excludes raw data I/O and audit SHA capture; engineering path, not optimized deployment FPS |
| 2 | R2 | 16 | 392 | 369 | 0.061592 | 0.185335 | 0.123743 | 20 | 1177153536 | fixed same16 InnerDev windows; preloaded predictor input, predictor+ego conversion+CPU graph/R2 features+normalization+H2D+head+Bicycle route; excludes raw data I/O and audit SHA capture; engineering path, not optimized deployment FPS |
| 2 | G-C | 16 | 392 | 369 | 0.061592 | 0.210328 | 0.148736 | 20 | 1177153536 | fixed same16 InnerDev windows; preloaded predictor input, predictor+ego conversion+CPU graph/R2 features+normalization+H2D+head+Bicycle route; excludes raw data I/O and audit SHA capture; engineering path, not optimized deployment FPS |
| 3 | R2 | 16 | 163 | 159 | 0.044128 | 0.145458 | 0.101330 | 20 | 413116416 | fixed same16 InnerDev windows; preloaded predictor input, predictor+ego conversion+CPU graph/R2 features+normalization+H2D+head+Bicycle route; excludes raw data I/O and audit SHA capture; engineering path, not optimized deployment FPS |
| 3 | G-C | 16 | 163 | 159 | 0.044128 | 0.172570 | 0.128441 | 20 | 413116416 | fixed same16 InnerDev windows; preloaded predictor input, predictor+ego conversion+CPU graph/R2 features+normalization+H2D+head+Bicycle route; excludes raw data I/O and audit SHA capture; engineering path, not optimized deployment FPS |

Shared arrays/context cache measured bytes=19639145590; free disk after experiment=25919733760. Predictor/head training times and peaks are recorded in training summaries and monitor curves; all large checkpoints/caches remain local.

## Scientific limitations

Internal training-isolated three-fold CV on historically developed scenes; not pristine confirmation. Scheme A uses predictor training-in candidates for ranker fitting. Bootstrap is conditional on three selected predictors/heads and does not quantify retraining seed uncertainty. Custom eligibility/t0 ego coordinates/K6/end-point miss/HitRate differ from official nuScenes default K=[1,5,10], official targets/splits, global-coordinate output, whole-trajectory miss and OffRoadRate definitions. No cross-protocol paper-number comparison. Capacity is approximately matched and no causal proof is claimed.

## Post-experiment delivery review

The [actual data-usage audit](01_data_isolation/stage15b_actual_data_usage.md) independently reconstructs all 36,500 formal predictor optimization batches from original scene/sample identities. All three folds use exactly 378 InnerTrain scenes and 42 InnerDev scenes; OuterTest and HeadDev70 contribute zero fitting batches. Source-witness prefix/full-file SHA matches and every 500-step full-development target count pass. Final selected model ancestry is warmup-best4500 → NLL-best9000 for Fold1, warmup-best4000 → NLL-best9500 for Fold2, and warmup-best5000 → NLL-best10500 for Fold3. Executed budgets remain5000+6500,5000+7000,5000+8000; discarded replay/tiny updates never become formal weights.

The root Stage15B protocol and candidate/checkpoint provenance govern the current experiment. Nested `ranking.Stage`, `ranking.branch`, and the historical `ranking.limitations` phrase about a TRAIN700 predictor are preserved Stage14B head-protocol provenance. They do not describe the current predictor: all three actual generators are fresh, training-isolated Stage15B checkpoints with the recorded SHA. Frozen protocol bytes were not edited after fitting. R0 denotes each adapted Stage5A predictor's original mode scoring, not an independently reproduced unadapted third-party HiVT model. See [pre-outcome reporting clarifications](00_manifest/stage15b_reporting_clarifications.md).

### Secondary groups and negative transfer

These are descriptive95% scene-bootstrap intervals, outside the three-primary corrected family; no new confirmatory group claims are introduced.

| Group | Contrast | Delta Top1FDE (m) | Descriptive95% CI | Fold1 / Fold2 / Fold3 deltas (m) |
| --- | --- | --- | --- | --- |
| MovingVehicle | G-C−NG-C | -0.259876 | [-0.364893,-0.173174] | -0.264687 / -0.136299 / -0.374967 |
| MovingVehicle | G-C−Matched-NG-C | -0.113463 | [-0.172796,-0.060984] | -0.214698 / -0.072140 / -0.065387 |
| MovingVehicle | NG-C−NG-A | -0.393172 | [-0.573589,-0.212412] | -0.174780 / -0.847180 / -0.144732 |
| Pedestrian | G-C−NG-C | -0.004399 | [-0.008611,-0.000027] | -0.007345 / -0.006548 / +0.000107 |
| Pedestrian | G-C−Matched-NG-C | -0.007598 | [-0.011500,-0.003801] | -0.016735 / -0.006827 / -0.000537 |
| Pedestrian | NG-C−NG-A | -0.004855 | [-0.012593,+0.003618] | -0.017861 / -0.008159 / +0.009289 |

MovingVehicle G-C mean Top1FDE remains10.210756 m despite the mean improvements. Pedestrian gains are small; Fold3 G-C vs NG-C and NG-C vs NG-A deteriorate. StoppedVehicle/OtherVehicleState intervals cross zero, and ParkedVehicle G-C vs NG-C deteriorates in two folds despite its small negative pooled point. Full subgroup rows remain in the delivered metric and CI tables.

G-C vs NG-C worsens19,956 of260,151 actor-windows, including5,718 MovingVehicle and5,756 Pedestrian cases. G-C vs Matched-NG-C worsens18,599; NG-C vs NG-A worsens31,769. These mean improvements do not protect every actor. The largest illustrated switching harms are25.088585 m for a graph-vs-NoGraph moving case and5.705440 m for a pedestrian case; all four deterministic extreme failure cases are retained. Approximately matched capacity supports the registered comparison but does not eliminate architecture/optimization alternatives or prove causality.

Loss C also produces much sharper mode probabilities. G-C entropy is0.132957, mean top1 probability0.942559 and softCE23.422898, versus G-A1.459550/0.299848/1.498611. NG-C softCE11.352449 also exceeds NG-A1.504201. This softCE evaluates fit to the frozen q=softmax(-FDE/1m) reference, not an empirical calibration metric; it reflects a different optimization target. The improved ranking errors do not establish probability calibration, and no calibration or planning claim is made. These existing auxiliary metrics remain visible in the full delivered tables.

### Training-in and Outer candidate distributions

No consistent overall oracle-error gap appears across all three folds:

| Fold | InnerTrain mean minFDE6 | InnerDev mean minFDE6 | Outer mean minFDE6 | Outer−InnerTrain (m) |
| --- | --- | --- | --- | --- |
| 1 | 1.261050 | 1.180323 | 1.251966 | -0.009084 |
| 2 | 1.225950 | 1.277962 | 1.310884 | +0.084934 |
| 3 | 1.313329 | 1.125768 | 1.238864 | -0.074465 |

This descriptive result neither establishes identical distributions nor supplies inner OOF candidates. Fold1/2/3 outer mean candidate max-minus-min FDE spreads are22.073729/13.891226/11.622250 m, respectively. Tail quantiles, type/motion groups and exact counts are in [candidate distributions](09_statistics/stage15b_candidate_error_distributions.csv). They are not used to tune or refit any model.

### Compute scope

G-C head-only forward takes10.315–10.397 ms per1024 cached normalized targets; NG-C takes4.731–4.784 ms and Matched-NG-C6.480–6.540 ms. These preloaded-input measurements exclude predictor, feature construction, H2D and Bicycle routing. Full engineering G-C inference takes0.173–0.301 s per fixed16-window batch versus0.044–0.085 s for the predictor alone, including CPU feature construction and routing but excluding raw-data I/O/audit SHA capture. It adds0.020–0.027 s over the full R2 pipeline. Neither range is deployment FPS or representative of every scene, and the recorded peak includes resident benchmark inputs/models. System G-C+Bicycle-R2 has675142 parameters; head-only G-C has24066.

Small case identities/errors are versioned. Exact GT/candidate coordinate CSVs remain local, as do all large checkpoints and candidate caches. The [delivery consistency audit](00_manifest/stage15b_delivery_review.json) checks required files, frozen source SHA, weighted fold-to-pooled metrics and registered decisions without fitting or introducing another hypothesis test.

STOP. Wait for brain-AI review; no further model-improvement experiment is started.
