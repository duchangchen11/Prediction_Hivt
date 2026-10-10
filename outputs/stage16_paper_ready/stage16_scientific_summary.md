# Stage16 scientific summary

The two implemented methods remain supported by the frozen **internal scene-isolated CV** comparisons: candidate-conditioned future interaction messages and normalized regret-aware mode ranking. They reduce selected-mode error without changing the six candidate trajectories. Stage15B resolves predictor–Outer training overlap; it does not create a previously untouched research confirmation set, establish literature novelty, or constitute official nuScenes test performance.

## Evidence scope: Stage14A / Stage14B / Stage15B

Stage14A held one TRAIN700-trained Stage5A predictor fixed. Its ranking heads were scene OOF, but all630 evaluated scenes were in predictor training. Stage14B audited this limitation, designed scheme A, and added the 24,001-parameter target-only capacity control on the same old candidates; it did not retrain HiVT. Stage15B fitted three fresh 650,403-parameter Stage5A generators on378 InnerTrain scenes each;42 InnerDev selected checkpoints. Both predictor and ranker fitting, normalization and model selection excluded eachfold210 Outer scenes and70 quarantined HeadDev scenes. Outer union is630 scenes/260,151 actor windows. History-derived features and predicted futures enter the scorer; GT futures only provide detached training labels, Dev selection and offline evaluation.

Official VAL150 historically informed predictor selection and method development. The TRAIN scenes and earlier internal results also informed development. Therefore even scene-isolated retraining is an internal validation of the frozen procedure, not pristine independent scientific confirmation. Scheme A uses predictor training-in candidates for the head's InnerTrain; it does not remove all train-in versus inference candidate distribution shift. Predictors differ across folds in both training scenes and seeds, so these folds cannot isolate predictor-seed variance.

## Pooled Overall metrics, meters

| Stage | Model | Count | Top1FDE | Top1ADE | minFDE6 | HitRate |
| --- | --- | --- | --- | --- | --- | --- |
| Stage14A | NG-A | 260151 | 2.380943 | 1.073364 | 1.258135 | 0.385914 |
| Stage14A | NG-C | 260151 | 2.254734 | 0.994880 | 1.258135 | 0.419872 |
| Stage14A | G-A | 260151 | 2.384061 | 1.089466 | 1.258135 | 0.385069 |
| Stage14A | G-C | 260151 | 2.199512 | 0.973628 | 1.258135 | 0.420121 |
| Stage14B | NG-A | 260151 | 2.380943 | 1.073364 | 1.258135 | 0.385914 |
| Stage14B | NG-C | 260151 | 2.254734 | 0.994880 | 1.258135 | 0.419872 |
| Stage14B | G-A | 260151 | 2.384061 | 1.089466 | 1.258135 | 0.385069 |
| Stage14B | G-C | 260151 | 2.199512 | 0.973628 | 1.258135 | 0.420121 |
| Stage14B | Matched-NG-C | 260151 | 2.258354 | 0.995979 | 1.258135 | 0.421390 |
| Stage15B | NG-A | 260151 | 2.375782 | 1.064636 | 1.266202 | 0.439276 |
| Stage15B | NG-C | 260151 | 2.306543 | 1.020819 | 1.266202 | 0.464492 |
| Stage15B | G-A | 260151 | 2.361641 | 1.063987 | 1.266202 | 0.440198 |
| Stage15B | G-C | 260151 | 2.261161 | 0.999894 | 1.266202 | 0.470369 |
| Stage15B | Matched-NG-C | 260151 | 2.283684 | 1.009464 | 1.266202 | 0.467475 |

## Frozen Stage15B primary evidence

| Comparison | Delta | BonferroniCILower | BonferroniCIUpper | NegativeFolds |
| --- | --- | --- | --- | --- |
| G-C-NG-C | -0.045382 | -0.068976 | -0.026489 | 3 |
| G-C-Matched-NG-C | -0.022523 | -0.035387 | -0.011727 | 3 |
| NG-C-NG-A | -0.069239 | -0.104990 | -0.031506 | 3 |

Each negative delta favors the first model. All three frozen co-primary comparisons retain their original support decision: 2000 paired whole-scene bootstrap draws, seed2022,210 sampled scenes independently in each fold, actor-window-weighted aggregate; family3 Bonferroni98.333% intervals. These intervals condition on trained models. Stage14A family4 and Stage14B family3 intervals are not interchangeable; old and new estimates must not be pooled as independent replications because they reuse scenes and development history.

G-C−NG-C Overall is −0.055221m in Stage14A/14B versus −0.045382m in Stage15B; graph direction is retained with a smaller absolute gain. NG-C−NG-A is −0.126209m before versus −0.069239m after isolation. G-C−Matched-NG-C is −0.058842m before versus −0.022523m after isolation. Matching active parameter count retains evidence beyond this particular capacity control, but cannot prove all capacity/optimization confounding is eliminated: matching counts does not match function class or training dynamics. The nominal65-parameter difference is0.2701%, and the common scalar score bias is softmax-unidentifiable in all heads.

The isolated common oracle minFDE6 is1.266202m, versus1.258135m on the old generator. Different predictor geometry and Dev selection prevent attributing the between-stage difference solely to isolation. The original graph/loss ablations and capacity control remain historical analyses; Stage15B supplies the principal end-to-end internal CV evidence.

## MovingVehicle absolute error

MovingVehicle uses t0 `vehicle.moving`, not a GT-speed-filtered selection. N=41,728. G-C Top1FDE=10.210756m, versus NG-C10.470631m and oracle5.343627m. Its graph gain −0.259876m has descriptive95% scene CI[−0.364893,−0.173174] and negative direction in all three folds; this secondary group is outside the registered primary family. The oracle's large absolute error means fixed-candidate quality is itself limiting; ranking cannot close missing-candidate error. Residual selection gap remains4.867129m. Six-second horizon and heterogeneous moving scenes are relevant context, but these summaries do not establish the cause of the large errors. Report distributions/median/tails from frozen source data rather than hiding them behind an Overall mean dominated by parked vehicles.

## Pedestrian modest/uneven gain

N=66,145. G-C1.330484m vs NG-C1.334883m gives −0.004399m; descriptive95% scene CI[−0.008611,−0.000027], with fold3 slightly positive (+0.000107m). The old graph gain was −0.012584m. NG-C−NG-A Pedestrian CI crosses zero and fold3 is worse (+0.009289m). Do not describe uniformly strong multi-type gains. Bicycle uses identical samefold R2 selections across all NG/G/Matched variants; its result validates routing consistency, not an interaction-graph improvement.

## Concentration and costly mode switches

Loss C's pooled G-C entropy=0.132957nats and Top1Probability=0.942559, compared with G-A entropy1.459550 and probability0.299848. The dedicated probability report establishes a tradeoff against the explicitly defined endpoint-oracle label, using fixed-bin ECE and categorical Brier; no calibration was fitted. These are ranking scores and must not be sold as well-calibrated maneuver probabilities.

G-C vs NG-C improves23,914 actor windows, worsens19,956 and ties216,281. In MovingVehicle,7,813 improve and5,718 worsen; in Pedestrian,6,577 improve and5,756 worsen. Four real-coordinate examples deliberately include extreme improvements and failures. They illustrate mode selection under identical geometry and are not representative prevalence estimates. High confidence can coexist with a costly wrong switch; switching summaries and harm tails remain disclosed. No causal claim follows from one case.

## Compute burden

Frozen Stage15B CUDA head-only forward for1024 normalized cached targets is approximately G-C10.3–10.4ms, NG-C4.7–4.8ms, Matched6.5ms and R2≈2.0ms on RTX3080. This excludes predictor, feature construction, transfers and Bicycle route. The engineering full-system benchmark for16 fixed InnerDev windows measures G-C0.3012/0.2103/0.1726s in folds1/2/3, versus R20.2809/0.1853/0.1455s; G-C adds20–27ms relative to R2 in that path. Actor counts differ byfold (1014/392/163 current actors); these are not comparable scene-independent deployment FPS or optimized real-time guarantees. Full timings include predictor, ego conversion, CPU feature construction, normalization, transfer, score head and route, but exclude raw I/O/audit hashing. Benchmark peak memory includes resident pools/models.

Per-fold total inference parameters: R0 650,403; R2 651,076; NG658,501; G675,142; Matched675,077, including Bicycle R2 where applicable. The generator is frozen during head optimization; actual head trainable counts are673/7,425/24,066/24,001. No ensemble of the three folds is claimed.

## Paper framing and remaining boundaries

Keep the paper focused on fixed-candidate selection, the future interaction graph and normalized regret-aware ranking. Describe the adapted Stage5A generator and R2 routing honestly as existing components; NG means no **post-prediction** graph message, not an interaction-free HiVT backbone. Show common oracle geometry, paired scene uncertainty, grouped harms, initialization sensitivity and calibration limitations. No third module, changed loss, predictor retraining, new checkpoint choice or Outer-based tuning is introduced.

Two external scoring-component adaptations were reviewed, not evaluated; their architecture/loss/source differences are explicit. The seven current baselines support controlled mechanism claims, but a reproduced external scorer or another genuinely comparable external prediction baseline remains a reviewer-facing limitation. Cross-protocol literature numbers are excluded. An official submission would additionally need official targets/split, global-coordinate export, supported K and official metrics; current custom K6 three-type endpoint protocol is not official test.

Supplementary seed results are reported separately after all24 new heads freeze; they measure only initialization sensitivity conditional on fixed predictors and fixed sample order, and cannot overwrite Stage15B's preregistered outcomes. Paper-readiness is a bounded internal-evidence package, subject to brain-AI review, literature novelty appraisal and journal selection.

## Actor-weighted composition of the Overall change

| ExclusiveGroup | Count | FractionOfOverall | MeanDeltaTop1FDE | WeightedContributionMeters |
| --- | --- | --- | --- | --- |
| MovingVehicle | 41728 | 0.160399 | -0.259876 | -0.041684 |
| OtherVehicle | 149298 | 0.573890 | -0.004496 | -0.002580 |
| Pedestrian | 66145 | 0.254256 | -0.004399 | -0.001118 |
| Bicycle | 2980 | 0.011455 | 0.000000 | 0.000000 |

The four groups are disjoint and exhaustive; OtherVehicle means Vehicle without the t0 moving attribute, including stopped, parked and other observed states. WeightedContributionMeters equals group fraction times group mean paired FDE difference. These signed contributions sum to the frozen Overall G-C minus NG-C delta. This is an arithmetic decomposition of the actor-window-weighted mean, not a causal attribution or separate statistical test. The MovingVehicle contribution is large despite its smaller window fraction; it does not remove the large remaining absolute errors. The full source table also records the capacity and loss contrasts.

## Completed initialization supplement

| Group | Comparison | FoldInitializationPairs | NegativePairs | PositivePairs | EqualPairs | UnweightedMeanPairDelta |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | G-C-NG-C | 9 | 9 | 0 | 0 | -0.041756 |
| Overall | G-C-Matched-NG-C | 9 | 8 | 1 | 0 | -0.030443 |
| Overall | NG-C-NG-A | 9 | 9 | 0 | 0 | -0.076693 |

All24 new head fits and the12 original replicates are disclosed in the seed report. This is conditional initialization evidence; mixed directions are retained and do not revise the original preregistered Stage15B decisions.

Overall contrasts with a positive difference:

| Fold | InitializationReplicate | Comparison | DeltaTop1FDE |
| --- | --- | --- | --- |
| 3 | 1 | G-C-Matched-NG-C | 0.006112 |

The capacity comparison is therefore not uniformly favorable under initialization changes. This limits a universal superiority claim while leaving the original preregistered conditional bootstrap result intact. No initialization is selected or discarded using these outcomes.

## MovingVehicle error distribution

| Model | Count | MedianTop1FDE | P90Top1FDE | P99Top1FDE | MedianOracleMinFDE6 | P90OracleMinFDE6 |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 41728 | 8.544700 | 23.995763 | 38.957117 | 2.718320 | 14.922475 |
| NG-C | 41728 | 8.439326 | 22.759357 | 37.049940 | 2.718320 | 14.922475 |
| G-C | 41728 | 8.134652 | 22.395326 | 36.953451 | 2.718320 | 14.922475 |

All distances are meters and distributions use the frozen actor-window denominator, not independent trajectories. The long upper tail remains after ranking.
