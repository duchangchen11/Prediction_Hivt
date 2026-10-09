# Stage12A Candidate Trajectory Semantic Consistency Audit

## Scientific decision and evidence boundary

Stage12A = **GO**; VehicleSemanticSignal = **WEAK**; PedestrianSemanticSignal = **PROMISING**; SemanticIncrementalValue = **SUGGESTED**. ReadyForStage12B = **YES** is a review recommendation only. Execution stops here. No model was trained, no score/candidate was changed, and no new checkpoint was created.

**CONFIRMED:** static map extraction, correct ego/global coordinates, complete actor/candidate identity, and frozen R0/FoldR2/A/C Raw/C Calibrated Top1 results. **STRONGLY_SUGGESTED** applies only to features meeting the registered cross-fold and both-control screen shown below. **UNRESOLVED:** independent prediction gain beyond existing motion/interaction features. A descriptive semantic association, even after coarse displacement matching, does not establish causality or useful predictive gain.

This is HeadTrain630 development exploration using previously inspected Stage11B/11C OOF results, not an independent confirmatory test. No historical HeadDev70, official VAL150 or test evaluation was performed. All 2000 scene-cluster intervals are descriptive, unadjusted for the exploratory feature family. The registered protocol was hashed before extraction and label analysis; all630 semantic scene caches were frozen before FDE/ADE/GT analysis.

The qualifying signal is Pedestrian **−walkway inside fraction**: pooled oriented rho=0.110663 on 22283 paired eligible actors/455 scenes, positive in all3folds, with real-minus-both-control contrasts passing the registered screen. After same-actor displacement-bin matching, rho=0.088528 on 11452 actors; real−NC1=0.098798 [95% descriptiveCI 0.049068,0.146305], real−NC2=0.073866 [0.028282,0.118997]. This supports a development-stage semantic screen, not a rule to force all pedestrians onto walkways.

Vehicle lane distance and −drivable occupancy correlate with six-mode errors, but their C-minus-oracle signed differences are positive in only onefold each, so they remain **WEAK** under the predeclared rule. Vehicle heading is **NONE**. Pedestrian crossing-distance/intersection hypotheses are also **NONE** and often show the opposite orientation. The primary walkway relation is nearzero in the important Pedestrian5–8m subgroup; the pooled Pedestrian result cannot be extrapolated to these moving targets. No new threshold or semantic reranker was derived from these findings.

## Frozen history and source identity

Base commit: `165c497961afe9f3a196a0987128f9eed48da144`. Branch: `stage12a/candidate-semantic-consistency-audit`. All new files reside in `outputs/stage12a_candidate_semantic_audit/`. The source arrays and20 checkpoint hashes are recorded in `00_manifest/stage12a_frozen_manifest.json`; final verification rehashed 2600 historical tracked files and 5 preserved Stage2C files.

Historical conclusions remain unchanged: Stage8 AgentGraph=SUPPORTED, SemanticGraph=NOT_SUPPORTED, SemanticContributionToJoint=NOT_SUPPORTED; Stage9 TAFIG=NOT_SUPPORTED; Stage10 DualExpert=STOP; Stage11A LossRankingMismatch=CONFIRMED, CostlyModeSwitchProblem=CONFIRMED, CandidateGeometryBottleneck=PARTIAL; Stage11B ErrorAwareRanking=STRONG_SUPPORTED, VehicleImproved=YES, PedestrianImproved=YES, BicyclePreserved=YES; Stage11C CalibrationEngineering=PASS, CalibrationUseful=YES, Top1Identity=PASS, Stage11C=GO.

Frozen Stage5 predictor SHA256: `88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`. K=6, Th=5, Tf=12. Population:630 scenes,260151 actor-window targets, including191026 Vehicle,66145 Pedestrian and2980 Bicycle. Three outer folds retain210 scenes each; no resplit. All1,560,906 geometry hashes and per-mode FP32 FDE/ADE were independently checked; maxdiff FDE=0.0, ADE=0.0. Positive-temperature C Calibrated has exactly the C Raw Top1 for all actors. Probability changes from calibration are auxiliary and provide no new trajectory improvement.

Frozen ranking metrics, pooled:

| Group | Model | Count | Top1FDE | Top1ADE | minFDE6 | OracleGap | MeanTop1Confidence |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | R0 | 260151 | 2.36294 | 1.0634 | 1.25813 | 1.1048 | 0.229731 |
| Overall | FoldR2 | 260151 | 2.3443 | 1.05401 | 1.25813 | 1.08617 | 0.259518 |
| Overall | A SoftCE | 260151 | 2.38406 | 1.08947 | 1.25813 | 1.12593 | 0.271761 |
| Overall | C Raw | 260151 | 2.19951 | 0.973628 | 1.25813 | 0.941377 | 0.957226 |
| Overall | C Calibrated | 260151 | 2.19951 | 0.973628 | 1.25813 | 0.941377 | 0.327244 |
| Vehicle | R0 | 191026 | 2.73357 | 1.21674 | 1.38159 | 1.35198 | 0.237585 |
| Vehicle | FoldR2 | 191026 | 2.70838 | 1.20375 | 1.38159 | 1.32678 | 0.268038 |
| Vehicle | A SoftCE | 191026 | 2.75108 | 1.24468 | 1.38159 | 1.36949 | 0.281721 |
| Vehicle | C Raw | 191026 | 2.51627 | 1.09657 | 1.38159 | 1.13468 | 0.969143 |
| Vehicle | C Calibrated | 191026 | 2.51627 | 1.09657 | 1.38159 | 1.13468 | 0.333108 |
| Pedestrian | R0 | 66145 | 1.32002 | 0.632193 | 0.907948 | 0.412072 | 0.207103 |
| Pedestrian | FoldR2 | 66145 | 1.31964 | 0.632774 | 0.907948 | 0.411688 | 0.234935 |
| Pedestrian | A SoftCE | 66145 | 1.35268 | 0.653994 | 0.907948 | 0.444735 | 0.243569 |
| Pedestrian | C Raw | 66145 | 1.30497 | 0.626157 | 0.907948 | 0.39702 | 0.954262 |
| Pedestrian | C Calibrated | 66145 | 1.30497 | 0.626157 | 0.907948 | 0.39702 | 0.313382 |
| Bicycle | R0 | 2980 | 1.75322 | 0.80473 | 1.11696 | 0.636261 | 0.228459 |
| Bicycle | FoldR2 | 2980 | 1.74987 | 0.805404 | 1.11696 | 0.632905 | 0.259054 |
| Bicycle | A SoftCE | 2980 | 1.74987 | 0.805404 | 1.11696 | 0.632905 | 0.259054 |
| Bicycle | C Raw | 2980 | 1.74987 | 0.805404 | 1.11696 | 0.632905 | 0.259054 |
| Bicycle | C Calibrated | 2980 | 1.74987 | 0.805404 | 1.11696 | 0.632905 | 0.259054 |
| MovingVehicle | R0 | 41728 | 10.8464 | 4.74398 | 5.32716 | 5.51922 | 0.218285 |
| MovingVehicle | FoldR2 | 41728 | 10.7364 | 4.68782 | 5.32716 | 5.40921 | 0.230722 |
| MovingVehicle | A SoftCE | 41728 | 10.9053 | 4.86009 | 5.32716 | 5.5781 | 0.284195 |
| MovingVehicle | C Raw | 41728 | 9.90648 | 4.22742 | 5.32716 | 4.57932 | 0.957077 |
| MovingVehicle | C Calibrated | 41728 | 9.90648 | 4.22742 | 5.32716 | 4.57932 | 0.313519 |
| Pedestrian5-8m | R0 | 22483 | 1.55404 | 0.726626 | 1.15135 | 0.402692 | 0.199189 |
| Pedestrian5-8m | FoldR2 | 22483 | 1.54507 | 0.722668 | 1.15135 | 0.39372 | 0.219016 |
| Pedestrian5-8m | A SoftCE | 22483 | 1.68836 | 0.799061 | 1.15135 | 0.537007 | 0.228678 |
| Pedestrian5-8m | C Raw | 22483 | 1.52941 | 0.717004 | 1.15135 | 0.378053 | 0.946726 |
| Pedestrian5-8m | C Calibrated | 22483 | 1.52941 | 0.717004 | 1.15135 | 0.378053 | 0.303778 |

## Map integrity, coordinate audit and GT isolation

All engineering gates PASS. Reused the original Stage8A-0C SparseSemanticIndex, exact six static entities, 3/3/1/1 Vehicle quotas, 0/0/1/0/2/3 Pedestrian quotas and3/3/1/0 Bicycle quotas, 10m centerlines and2m polygons. No nearest fallback, buffer, geometry repair or GT selection was added. Four original nuScenes expansion maps retain the frozen Stage8 hashes; the local source directory name contains `nuscenes-mini` but includes all four full-region expansion maps. The existing single invalid Boston walkway component is excluded by unchanged Stage8 policy, retaining other valid components of that entity.

The 114 fixed preflight actors/684 candidates, selected without error labels, reproduce source entity IDs/types and raw FP32 map_node/map_edge bitwise. Ego x-forward/y-left, positive yaw counterclockwise; each sample uses its original t0 origin and yaw. Roundtrip maximum=1.63425e-13 m <1e-5m. Polygon strict inside, polyline intersection, original-component boundary distance/crossing, true segment-interior distance and multipart OR tests pass. Poisoning GT toNaN, future/target masks and future motion labels changes neither entity selection nor semantic features.

Source caches include opaque future labels; the extractor uses an explicit observation/prediction allowlist and never uses those fields to choose entities or features. Standalone future-label array reads are blocked during extraction. Label access starts only after `FROZEN_ALL_630`. Three fork CPU workers write disjoint per-scene caches; three sequential-cache replay scenes pass, and spatial-index optimization maximum replay difference=0. Spatial pruning changes no definition. Stationary predicted headings have an explicit missing mask; original Stage8 degenerate-line warnings do not change finite cached values. Case visual clipping affects display size only.

## Feature definitions and completeness

Every mode stores scene/sample/actor/type/fold/mode, original geometry SHA256, traceable selected entity IDs and feature-valid bitmask. The unchanged source entity dictionary maps IDs to type, token and map region. The local full CSV `02_semantic_cache/cache/stage12a_mode_semantics.csv` contains1,560,906 rows; SHA256 `9778a391b1d863edf9b5218e9d61fc57e727503550507b2bfb0c7373b5b17f38`. Its small versioned summary is `03_feature_statistics/stage12a_mode_semantics.csv`. Large arrays/CSV remain local and ignored by Git.

The nearest selected lane/connector is chosen by whole-polyline minimum distance; its cached mean12-point distance, endpoint distance and acos clipped heading cosine are retained. These are distances to true centerline polylines, not just start points. Candidate map matching uses12 predicted future points, without adding the observed-t0-to-first-future segment. Polygon distance is minimum over selected entities; strict sampled-point containment is OR over original valid components, not polygon union. Drivable boundary crossing and distance use original component boundaries. These details can differ from containment in a repaired/unioned polygon and are frozen here.

All26 semantic fields and24 observable candidate/history/interaction fields are finite. Missing features are zero with an explicit validity mask, and all empty-map actors remain. Centerline headings use the frozen final1s secant; undefined stationary headings are masked. History groups use only observed speed, net displacement and heading. Future displacement groups are offline labels only. SemanticCoverage=PASS means complete finite/masked extraction and retention; it does not claim100% map coverage.

Candidate endpoints/current positions are in the original t0 ego frame. `trajectory_curvature` is explicitly a turn-magnitude proxy: sum of absolute wrapped heading increments over consecutive nonzero candidate steps, in radians, not inverse-meter differential curvature. Frozen interaction summaries retain Stage8 raw physical units (distances in meters and closing distance/6s) and mask unavailable neighbors; they are not recomputed from future GT.

| Scope | Group | Actors | Candidates | CandidateMapCoverage | ActorAnyModeCoverage | ActorAllModesCoverage | RouteCenterlineCoverage | PedestrianSpecificCoverage | EmptyCandidates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pooled | Vehicle | 191026 | 1146156 | 0.886876 | 0.913698 | 0.875823 | 0.793065 | 0 | 129658 |
| Pooled | Pedestrian | 66145 | 396870 | 0.897931 | 0.914173 | 0.887338 | 0 | 0.818835 | 40508 |
| Pooled | Bicycle | 2980 | 17880 | 0.921309 | 0.951007 | 0.908725 | 0.917897 | 0 | 1407 |
| Fold1 | Vehicle | 61556 | 369336 | 0.897237 | 0.91991 | 0.887663 | 0.792298 | 0 | 37954 |
| Fold1 | Pedestrian | 20250 | 121500 | 0.877292 | 0.903753 | 0.862123 | 0 | 0.793942 | 14909 |
| Fold1 | Bicycle | 867 | 5202 | 0.949827 | 0.986159 | 0.933103 | 0.946367 | 0 | 261 |
| Fold2 | Vehicle | 60451 | 362706 | 0.897887 | 0.923326 | 0.888405 | 0.809267 | 0 | 37037 |
| Fold2 | Pedestrian | 22108 | 132648 | 0.897903 | 0.913289 | 0.88719 | 0 | 0.816831 | 13543 |
| Fold2 | Bicycle | 1152 | 6912 | 0.877749 | 0.91059 | 0.864583 | 0.877749 | 0 | 845 |
| Fold3 | Vehicle | 69019 | 414114 | 0.86799 | 0.899723 | 0.854243 | 0.779558 | 0 | 54667 |
| Fold3 | Pedestrian | 23787 | 142722 | 0.915528 | 0.923866 | 0.908942 | 0 | 0.841888 | 12056 |
| Fold3 | Bicycle | 961 | 5766 | 0.947797 | 0.967742 | 0.939646 | 0.94034 | 0 | 301 |

Stage8 comparison uses the identical selector but a different population (TRAIN700 includes historical HeadDev70; Stage12 analyzes only HeadTrain630). This is a comparison of existing audit summaries, not a new HeadDev evaluation:

| Group | Stage8Actors | Stage12Actors | Stage8CandidateCoverage | Stage12CandidateCoverage | DifferenceDueToPopulation |
| --- | --- | --- | --- | --- | --- |
| Vehicle | 213745 | 191026 | 0.886942 | 0.886876 | same selector and candidate matching; different actor/scene population, not assumed equal |
| Pedestrian | 73202 | 66145 | 0.902009 | 0.897931 | same selector and candidate matching; different actor/scene population, not assumed equal |
| Bicycle | 3138 | 2980 | 0.925271 | 0.921309 | same selector and candidate matching; different actor/scene population, not assumed equal |

## Prespecified semantic associations and negative controls

Within each actor, signed feature vs frozen six-mode FDE is tied-rank Spearman over at least3 valid varying modes. Undefined correlations remain marked ineligible; all actors/modes stay in coverage and identity outputs. Primary signs encode a screening hypothesis, not a rule that off-lane vehicles or road-entering pedestrians are wrong. Means average eligible actors within each scene, then scenes equally. Continuous feature/error tables also include actor-level mean,median,P90/P95/P99; association tables include scene median/P90/P95.

NC1 permutes entire semantic+mask records among each actor's six modes. NC2 permutes candidate semantic+mask records within exactly the same static map region and observed actor type. Independent fixed seed2022 generators preserve complete record distributions. Candidates, FDE, ADE, logits, probabilities and mode choices never change. All comparisons use the intersection of eligible actors across REAL/NC1/NC2, with explicit denominators. NegativeControl=PASS certifies this construction, not the existence of a semantic effect.

Scene bootstrap resamples whole scenes within each frozen210-scene fold,2000 repetitions, seed2022; the exact historical weight matrix was independently regenerated. The same weights and eligible actor intersection are used for real-minus-control paired contrasts. No independent-actor bootstrap is used.

| Scope | ActorType | Feature | CommonActors | Scenes | Real_SceneMeanRho | NC1_SceneMeanRho | NC2_SceneMeanRho | DeltaRealMinusNC1 | DeltaNC1_CI95Low | DeltaNC1_CI95High | DeltaRealMinusNC2 | DeltaNC2_CI95Low | DeltaNC2_CI95High | CMinusOracleSignedSceneMean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pooled | Vehicle | centerline_mean_distance | 145963 | 629 | 0.169313 | -0.000275811 | -2.45183e-05 | 0.169589 | 0.150715 | 0.188588 | 0.169338 | 0.150963 | 0.188318 | 0.000826577 |
| Fold1 | Vehicle | centerline_mean_distance | 47078 | 209 | 0.165103 | -0.00193693 | 0.00152279 | 0.16704 | 0.13258 | 0.202498 | 0.16358 | 0.12754 | 0.200285 | -0.0444938 |
| Fold2 | Vehicle | centerline_mean_distance | 47089 | 210 | 0.175526 | 0.000393065 | -0.00178424 | 0.175133 | 0.141432 | 0.207011 | 0.177311 | 0.143819 | 0.20853 | 0.0491037 |
| Fold3 | Vehicle | centerline_mean_distance | 51796 | 210 | 0.16729 | 0.000708527 | 0.000195267 | 0.166581 | 0.134383 | 0.200768 | 0.167095 | 0.135985 | 0.201242 | -0.00256173 |
| Pooled | Vehicle | centerline_heading_error | 145963 | 629 | 0.0102622 | 0.000363176 | 0.00175804 | 0.00989906 | -0.00308996 | 0.0229508 | 0.00850419 | -0.00497526 | 0.0214166 | 0.0666014 |
| Fold1 | Vehicle | centerline_heading_error | 47078 | 209 | 0.00388373 | 0.00337875 | -0.0048989 | 0.00050498 | -0.0226007 | 0.0249079 | 0.00878263 | -0.0146578 | 0.0334152 | 0.0783127 |
| Fold2 | Vehicle | centerline_heading_error | 47089 | 210 | 0.00616866 | -0.0010721 | 0.00864642 | 0.00724076 | -0.01147 | 0.0257144 | -0.00247776 | -0.0217202 | 0.0166817 | 0.0644024 |
| Fold3 | Vehicle | centerline_heading_error | 51796 | 210 | 0.0207039 | -0.00120276 | 0.00149491 | 0.0219067 | -0.00156241 | 0.0448273 | 0.019209 | -0.00472399 | 0.0430195 | 0.0572007 |
| Pooled | Vehicle | drivable_inside_fraction | 46668 | 619 | 0.484498 | 0.00553593 | 0.00496997 | 0.478962 | 0.452465 | 0.503104 | 0.479528 | 0.453429 | 0.504398 | -0.000492081 |
| Fold1 | Vehicle | drivable_inside_fraction | 17283 | 209 | 0.486228 | 0.00510022 | 0.000304198 | 0.481127 | 0.432123 | 0.529205 | 0.485923 | 0.436247 | 0.533563 | -0.000995015 |
| Fold2 | Vehicle | drivable_inside_fraction | 14655 | 204 | 0.47713 | -0.00439521 | 0.0139057 | 0.481525 | 0.440993 | 0.523065 | 0.463224 | 0.420074 | 0.504871 | -0.00100475 |
| Fold3 | Vehicle | drivable_inside_fraction | 14730 | 206 | 0.490038 | 0.0158127 | 0.000854677 | 0.474226 | 0.427135 | 0.518021 | 0.489184 | 0.444618 | 0.534206 | 0.000523564 |
| Pooled | Pedestrian | crosswalk_distance | 1651 | 197 | -0.143822 | 0.00296192 | 0.00156133 | -0.146784 | -0.234445 | -0.0621979 | -0.145384 | -0.240093 | -0.052999 | -0.0167716 |
| Fold1 | Pedestrian | crosswalk_distance | 428 | 64 | -0.168298 | 0.0240998 | -0.057829 | -0.192398 | -0.327228 | -0.0398869 | -0.110469 | -0.278394 | 0.057053 | -0.0173324 |
| Fold2 | Pedestrian | crosswalk_distance | 571 | 64 | -0.0962546 | 0.0204684 | 0.0458191 | -0.116723 | -0.267774 | 0.025254 | -0.142074 | -0.300162 | 0.0127242 | -0.0242161 |
| Fold3 | Pedestrian | crosswalk_distance | 652 | 69 | -0.165241 | -0.0328821 | 0.0155974 | -0.132359 | -0.289049 | 0.0188183 | -0.180838 | -0.349747 | -0.0178686 | -0.0100584 |
| Pooled | Pedestrian | walkway_inside_fraction | 22283 | 455 | 0.110663 | 0.00824353 | 0.0100927 | 0.102419 | 0.0642584 | 0.141618 | 0.10057 | 0.0601127 | 0.143091 | 0.00710949 |
| Fold1 | Pedestrian | walkway_inside_fraction | 6990 | 156 | 0.159516 | 0.00121956 | 0.0144624 | 0.158296 | 0.090005 | 0.224696 | 0.145053 | 0.0672437 | 0.21894 | 0.00793094 |
| Fold2 | Pedestrian | walkway_inside_fraction | 6728 | 148 | 0.0841493 | 0.0257565 | 0.0222463 | 0.0583928 | -0.00948238 | 0.128168 | 0.061903 | -0.00761503 | 0.134372 | 0.00978016 |
| Fold3 | Pedestrian | walkway_inside_fraction | 8565 | 151 | 0.0861791 | -0.00166492 | -0.00633388 | 0.0878441 | 0.0278652 | 0.147905 | 0.092513 | 0.0297054 | 0.156741 | 0.00378445 |
| Pooled | Pedestrian | crosswalk_intersects | 616 | 146 | -0.259227 | -0.01172 | 0.0521217 | -0.247507 | -0.351978 | -0.135562 | -0.311349 | -0.430551 | -0.18848 | -0.0145271 |
| Fold1 | Pedestrian | crosswalk_intersects | 154 | 47 | -0.235024 | 0.000922193 | -0.0297676 | -0.235946 | -0.424495 | -0.0384351 | -0.205256 | -0.426421 | 0.0172627 | 0.00169588 |
| Fold2 | Pedestrian | crosswalk_intersects | 221 | 47 | -0.213919 | 0.00386103 | 0.129021 | -0.21778 | -0.41153 | -0.0222391 | -0.34294 | -0.547417 | -0.130875 | -0.0316712 |
| Fold3 | Pedestrian | crosswalk_intersects | 241 | 52 | -0.322054 | -0.0372294 | 0.0566315 | -0.284825 | -0.466758 | -0.096026 | -0.378686 | -0.582464 | -0.173743 | -0.013429 |

Registered PROMISING requires pooled signedrho>=.05,500 eligible actors/50scenes, positive direction in at least2 adequately sized folds, real-minus-each-control>=.03 with descriptive95%CI lower>0, and positive C-minus-oracle signed feature difference in at least2folds. WEAK retains the rho/reproducibility requirements but fails remaining controls/selection checks. NONE includes no registered signal or insufficient estimability; absent estimates are shown as not estimable, never filled as evidence of zero.

| ActorType | Feature | Grade | PositiveEligibleFolds | PositiveSelectedDifferenceFolds | ControlContrastPass | MatchedMotionIncrementalSuggested |
| --- | --- | --- | --- | --- | --- | --- |
| Vehicle | centerline_mean_distance | WEAK | 3 | 1 | True | False |
| Vehicle | centerline_heading_error | NONE | 3 | 3 | False | False |
| Vehicle | drivable_inside_fraction | WEAK | 3 | 1 | True | False |
| Pedestrian | crosswalk_distance | NONE | 0 | 0 | False | False |
| Pedestrian | walkway_inside_fraction | PROMISING | 3 | 3 | True | True |
| Pedestrian | crosswalk_intersects | NONE | 0 | 1 | False | False |

## Vehicle error modes, motion groups and high-cost switches

Differences below are raw feature(C Top1)−feature(oracle), among wrong-mode actors with both feature masks valid. Positive distance/heading and negative occupancy differences can be compatible with a map-consistency problem, but reasonable stopping, parking, lane changes and turns can also deviate from lane centerlines.

| Group | Feature | WrongModeActors | PairedMapFeatureActors | Mean | Median | P90 | P95 | MeanOracleGap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vehicle | centerline_mean_distance | 104684 | 84326 | -0.018352 | -0.000567315 | 1.00898 | 2.85618 | 2.07055 |
| Vehicle | centerline_heading_error | 104684 | 84326 | 0.0685667 | 0.00223803 | 1.44108 | 1.94692 | 2.07055 |
| Vehicle | drivable_inside_fraction | 104684 | 90154 | 0.000201507 | 0 | 0 | 0 | 2.07055 |
| MovingVehicle | centerline_mean_distance | 32371 | 31906 | -0.0293437 | -0.0148803 | 3.62966 | 6.32945 | 5.90299 |
| MovingVehicle | centerline_heading_error | 32371 | 31906 | 0.089209 | 0.000339745 | 1.1572 | 2.0379 | 5.90299 |
| MovingVehicle | drivable_inside_fraction | 32371 | 31935 | 0.00124993 | 0 | 0.0833333 | 0.166667 | 5.90299 |
| StoppedVehicle | centerline_mean_distance | 12295 | 11732 | -0.0861665 | -0.000450807 | 0.125096 | 0.221637 | 0.797533 |
| StoppedVehicle | centerline_heading_error | 12295 | 11732 | 0.138519 | 0.095126 | 1.79292 | 2.18479 | 0.797533 |
| StoppedVehicle | drivable_inside_fraction | 12295 | 11865 | 0.00155921 | 0 | 0 | 0 | 0.797533 |
| ParkedVehicle | centerline_mean_distance | 57701 | 38548 | 0.00107525 | -0.000106394 | 0.131077 | 0.21792 | 0.194759 |
| ParkedVehicle | centerline_heading_error | 57701 | 38548 | 0.0271072 | 0.0608789 | 1.40264 | 1.84062 | 0.194759 |
| ParkedVehicle | drivable_inside_fraction | 57701 | 44422 | -0.000309531 | 0 | 0 | 0 | 0.194759 |
| Vehicle>5m | centerline_mean_distance | 30145 | 29865 | -0.0990792 | -0.0486536 | 3.94223 | 6.74703 | 6.33997 |
| Vehicle>5m | centerline_heading_error | 30145 | 29865 | 0.0827193 | 0.000617371 | 0.874801 | 1.89025 | 6.33997 |
| Vehicle>5m | drivable_inside_fraction | 30145 | 29864 | 0.0027123 | 0 | 0.0833333 | 0.166667 | 6.33997 |

MovingVehicle is reported separately from Stopped/ParkedVehicle and Vehicle>5m; it is not averaged away by stationary targets. The motion state/GT displacement labels are used only for this offline stratification. All fourfold scopes and feature quantiles are in the source CSVs.

Focused raw-real subgroup correlations below use each subgroup's real-only eligible actor set. They are exploratory, have no subgroup control-adjusted decision gate, and are not directly comparable to the primary three-way common eligibility denominator:

| Group | Feature | Count | Scenes | SceneMean | SceneMedian | SceneP90 | SceneP95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MovingVehicle | centerline_mean_distance | 41201 | 594 | 0.0627293 | 0.0631447 | 0.480722 | 0.610943 |
| MovingVehicle | centerline_heading_error | 41201 | 594 | 0.0438594 | 0.0548428 | 0.307677 | 0.381619 |
| MovingVehicle | drivable_inside_fraction | 14252 | 544 | 0.175366 | 0.19659 | 0.710727 | 0.768589 |
| Pedestrian5-8m | crosswalk_distance | 2517 | 177 | -0.0220275 | -0.01406 | 0.655649 | 0.843317 |
| Pedestrian5-8m | walkway_inside_fraction | 7834 | 345 | -0.00148369 | -0.00340365 | 0.477788 | 0.652507 |
| Pedestrian5-8m | crosswalk_intersects | 1002 | 123 | 0.111072 | 0.151668 | 0.732003 | 0.825667 |

## Pedestrian error modes and observable history groups

Pedestrian<5m,>5m and5–8m are offline future-displacement groups; those labels never define an inference feature. Drivable entry can be valid crossing behavior, and a nearby crossing does not establish intent. Feature validity/variation and scene denominators are especially important for small displacement subgroups.

| Group | Feature | WrongModeActors | PairedMapFeatureActors | Mean | Median | P90 | P95 | MeanOracleGap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pedestrian | crosswalk_distance | 44449 | 10919 | 0.00720032 | 0 | 0.16771 | 0.363747 | 0.590809 |
| Pedestrian | crosswalk_intersects | 44449 | 10919 | -0.00576976 | 0 | 0 | 0 | 0.590809 |
| Pedestrian | walkway_inside_fraction | 44449 | 34441 | -0.003506 | 0 | 0.0833333 | 0.0833333 | 0.590809 |
| Pedestrian<5m | crosswalk_distance | 13874 | 3336 | -0.0138316 | 0 | 0.261058 | 0.366279 | 0.718547 |
| Pedestrian<5m | crosswalk_intersects | 13874 | 3336 | 0.0263789 | 0 | 0 | 1 | 0.718547 |
| Pedestrian<5m | walkway_inside_fraction | 13874 | 9674 | -0.0282027 | 0 | 0 | 0.166667 | 0.718547 |
| Pedestrian>5m | crosswalk_distance | 30575 | 7583 | 0.0164529 | 0 | 0.0528694 | 0.3548 | 0.532846 |
| Pedestrian>5m | crosswalk_intersects | 30575 | 7583 | -0.019913 | 0 | 0 | 0 | 0.532846 |
| Pedestrian>5m | walkway_inside_fraction | 30575 | 24767 | 0.00614056 | 0 | 0.0833333 | 0.0833333 | 0.532846 |
| Pedestrian5-8m | crosswalk_distance | 15481 | 3522 | 0.00666883 | 0 | 0.118908 | 0.47923 | 0.549045 |
| Pedestrian5-8m | crosswalk_intersects | 15481 | 3522 | -0.0170358 | 0 | 0 | 0 | 0.549045 |
| Pedestrian5-8m | walkway_inside_fraction | 15481 | 12292 | 0.00673202 | 0 | 0.0833333 | 0.166667 | 0.549045 |

Recent speed bins [0,.5,1.5,3,inf), history net-displacement bins [0,.5,2,5,inf), and eight observed heading sectors plus missing are evaluated separately in feature/error, switch and association tables. These use no future trajectory. Pooled observable subgroup associations:

| Group | Feature | Count | Scenes | SceneMean | SceneMedian | SceneP90 | SceneP95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pedestrian_RecentSpeed_0 | crosswalk_distance | 4886 | 142 | -0.151398 | -0.17112 | 0.430833 | 0.654654 |
| Pedestrian_RecentSpeed_1 | crosswalk_distance | 3852 | 227 | -0.0922469 | -0.0785867 | 0.654654 | 0.826003 |
| Pedestrian_RecentSpeed_2 | crosswalk_distance | 1135 | 134 | 0.0214052 | 0.0349156 | 0.70703 | 0.852635 |
| Pedestrian_RecentSpeed_3 | crosswalk_distance | 23 | 7 | 0.266841 | 0.657143 | 0.889698 | 0.944849 |
| Pedestrian_HistoryDisp_0 | crosswalk_distance | 4501 | 138 | -0.154601 | -0.202643 | 0.534968 | 0.773293 |
| Pedestrian_HistoryDisp_1 | crosswalk_distance | 1565 | 167 | -0.143062 | -0.167014 | 0.655649 | 0.839329 |
| Pedestrian_HistoryDisp_2 | crosswalk_distance | 3788 | 212 | -0.0555877 | -0.0384747 | 0.652522 | 0.709816 |
| Pedestrian_HistoryDisp_3 | crosswalk_distance | 42 | 10 | 0.451834 | 0.743071 | 1 | 1 |
| Pedestrian_HistoryHeading_0 | crosswalk_distance | 1160 | 124 | -0.104412 | -0.234391 | 0.794172 | 0.997842 |
| Pedestrian_HistoryHeading_1 | crosswalk_distance | 781 | 107 | -0.0445435 | -0.0146683 | 0.835205 | 0.992653 |
| Pedestrian_HistoryHeading_2 | crosswalk_distance | 914 | 111 | 0.00284067 | -0.0496966 | 0.902871 | 0.984952 |
| Pedestrian_HistoryHeading_3 | crosswalk_distance | 1451 | 135 | -0.0772291 | -0.0988183 | 0.66131 | 0.850832 |
| Pedestrian_HistoryHeading_4 | crosswalk_distance | 1355 | 122 | -0.100172 | -0.154183 | 0.710247 | 0.818999 |
| Pedestrian_HistoryHeading_5 | crosswalk_distance | 1000 | 109 | -0.23572 | -0.392282 | 0.699724 | 0.828571 |
| Pedestrian_HistoryHeading_6 | crosswalk_distance | 935 | 117 | 0.0220728 | 0.0576558 | 0.851608 | 0.981422 |
| Pedestrian_HistoryHeading_7 | crosswalk_distance | 1088 | 129 | -0.0822937 | -0.1 | 0.775729 | 0.894223 |
| Pedestrian_HistoryHeading_Missing | crosswalk_distance | 1212 | 67 | -0.24351 | -0.385714 | 0.656894 | 0.748571 |
| Pedestrian_RecentSpeed_0 | walkway_inside_fraction | 10566 | 303 | 0.330654 | 0.539811 | 0.829375 | 0.848719 |
| Pedestrian_RecentSpeed_1 | walkway_inside_fraction | 11436 | 415 | -0.0246511 | -0.0454065 | 0.601091 | 0.77081 |
| Pedestrian_RecentSpeed_2 | walkway_inside_fraction | 3396 | 301 | 0.0360086 | 0.0392731 | 0.633757 | 0.808543 |
| Pedestrian_RecentSpeed_3 | walkway_inside_fraction | 124 | 25 | -0.175332 | -0.338062 | 0.827818 | 0.943454 |
| Pedestrian_HistoryDisp_0 | walkway_inside_fraction | 9984 | 313 | 0.302143 | 0.550266 | 0.822145 | 0.845154 |
| Pedestrian_HistoryDisp_1 | walkway_inside_fraction | 4126 | 349 | 0.0735351 | 0.0775965 | 0.782099 | 0.892939 |
| Pedestrian_HistoryDisp_2 | walkway_inside_fraction | 11208 | 393 | -0.0155216 | -0.0197891 | 0.48698 | 0.709487 |
| Pedestrian_HistoryDisp_3 | walkway_inside_fraction | 204 | 27 | -0.263314 | -0.552912 | 0.777601 | 0.837199 |
| Pedestrian_HistoryHeading_0 | walkway_inside_fraction | 2964 | 278 | 0.168928 | 0.169273 | 0.841161 | 0.899735 |
| Pedestrian_HistoryHeading_1 | walkway_inside_fraction | 2453 | 234 | 0.224838 | 0.304634 | 0.87536 | 0.935789 |
| Pedestrian_HistoryHeading_2 | walkway_inside_fraction | 2322 | 242 | 0.294373 | 0.434042 | 0.845154 | 0.889137 |
| Pedestrian_HistoryHeading_3 | walkway_inside_fraction | 3487 | 294 | 0.110435 | 0.120759 | 0.790726 | 0.845154 |
| Pedestrian_HistoryHeading_4 | walkway_inside_fraction | 3811 | 289 | 0.11564 | 0.134337 | 0.797889 | 0.845154 |
| Pedestrian_HistoryHeading_5 | walkway_inside_fraction | 2651 | 243 | 0.285517 | 0.470602 | 0.851482 | 0.922697 |
| Pedestrian_HistoryHeading_6 | walkway_inside_fraction | 2367 | 243 | 0.274529 | 0.419514 | 0.845154 | 0.892414 |
| Pedestrian_HistoryHeading_7 | walkway_inside_fraction | 3318 | 298 | 0.114461 | 0.112644 | 0.843856 | 0.877886 |
| Pedestrian_HistoryHeading_Missing | walkway_inside_fraction | 2149 | 148 | 0.426667 | 0.654654 | 0.845154 | 0.845154 |
| Pedestrian_RecentSpeed_0 | crosswalk_intersects | 2352 | 128 | -0.434125 | -0.695309 | 0.476642 | 0.717279 |
| Pedestrian_RecentSpeed_1 | crosswalk_intersects | 1487 | 177 | 0.0308441 | 0.130931 | 0.691723 | 0.794187 |
| Pedestrian_RecentSpeed_2 | crosswalk_intersects | 342 | 86 | 0.0730625 | 0.156938 | 0.747645 | 0.817701 |
| Pedestrian_RecentSpeed_3 | crosswalk_intersects | 14 | 3 | 0.742226 | 0.762153 | 0.78634 | 0.789363 |
| Pedestrian_HistoryDisp_0 | crosswalk_intersects | 2104 | 122 | -0.447048 | -0.741366 | 0.619117 | 0.765854 |
| Pedestrian_HistoryDisp_1 | crosswalk_intersects | 821 | 120 | -0.0542976 | -0.0426727 | 0.75336 | 0.828079 |
| Pedestrian_HistoryDisp_2 | crosswalk_intersects | 1245 | 166 | 0.0599417 | 0.116281 | 0.656974 | 0.741366 |
| Pedestrian_HistoryDisp_3 | crosswalk_intersects | 25 | 5 | 0.7665 | 0.760585 | 0.83657 | 0.851298 |
| Pedestrian_HistoryHeading_0 | crosswalk_intersects | 582 | 77 | -0.180827 | -0.3274 | 0.723507 | 0.828079 |
| Pedestrian_HistoryHeading_1 | crosswalk_intersects | 348 | 69 | -0.275955 | -0.589188 | 0.654654 | 0.687487 |
| Pedestrian_HistoryHeading_2 | crosswalk_intersects | 411 | 72 | -0.253002 | -0.392792 | 0.654654 | 0.779153 |
| Pedestrian_HistoryHeading_3 | crosswalk_intersects | 577 | 92 | -0.159303 | -0.228231 | 0.654654 | 0.747175 |
| Pedestrian_HistoryHeading_4 | crosswalk_intersects | 520 | 90 | -0.136223 | -0.196787 | 0.73177 | 0.758594 |
| Pedestrian_HistoryHeading_5 | crosswalk_intersects | 400 | 71 | -0.295493 | -0.654654 | 0.654654 | 0.678962 |
| Pedestrian_HistoryHeading_6 | crosswalk_intersects | 355 | 72 | -0.135502 | -0.261414 | 0.738013 | 0.828079 |
| Pedestrian_HistoryHeading_7 | crosswalk_intersects | 466 | 83 | -0.243244 | -0.622757 | 0.654654 | 0.741366 |
| Pedestrian_HistoryHeading_Missing | crosswalk_intersects | 536 | 31 | -0.623876 | -0.806401 | -0.280566 | 0.432071 |

## FoldR2 to C Raw switches

Improved/Worsened/Unchanged use the exact frozen ΔFDE sign; high-cost harm is the registered fixed ΔFDE>5m. Counts below retain unchanged and empty-map actors. Gross gain and harm refer to all improvements/worsenings in the named population; repeated outcome rows share those totals. High-cost examples are descriptive development cases.

| Group | Outcome | Count | ModeSwitches | TotalModeSwitches | Mean | Median | P90 | P95 | GrossGain | GrossHarm | HarmP90 | HarmP95 | HarmP99 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vehicle | Improved | 68003 | 68003 | 92261 | -1.19061 | -0.0643697 | -0.0532066 | -0.0276783 | 80965.4 | 44268.8 | 5.88367 | 7.1932 | 13.4654 |
| Vehicle | Worsened | 24258 | 24258 | 92261 | 1.82491 | 0.151831 | 5.88367 | 7.1932 | 80965.4 | 44268.8 | 5.88367 | 7.1932 | 13.4654 |
| Vehicle | Unchanged | 98765 | 0 | 92261 | 0 | 0 | 0 | 0 | 80965.4 | 44268.8 | 5.88367 | 7.1932 | 13.4654 |
| Vehicle | HighCostWorsened | 3599 | 3599 | 92261 | 7.62803 | 6.33059 | 11.9796 | 14.6858 | 80965.4 | 44268.8 | 5.88367 | 7.1932 | 13.4654 |
| Pedestrian | Improved | 9805 | 9805 | 18140 | -0.395686 | -0.335132 | -0.050798 | -0.0239539 | 3879.7 | 2909.48 | 0.599162 | 0.987675 | 2.08956 |
| Pedestrian | Worsened | 8335 | 8335 | 18140 | 0.349068 | 0.268507 | 0.599162 | 0.987675 | 3879.7 | 2909.48 | 0.599162 | 0.987675 | 2.08956 |
| Pedestrian | Unchanged | 48005 | 0 | 18140 | 0 | 0 | 0 | 0 | 3879.7 | 2909.48 | 0.599162 | 0.987675 | 2.08956 |
| Pedestrian | HighCostWorsened | 4 | 4 | 18140 | 5.23801 | 5.20727 | 5.43104 | 5.46036 | 3879.7 | 2909.48 | 0.599162 | 0.987675 | 2.08956 |
| Bicycle | Improved | 0 | 0 | 0 | not estimable | not estimable | not estimable | not estimable | -0 | 0 | not estimable | not estimable | not estimable |
| Bicycle | Worsened | 0 | 0 | 0 | not estimable | not estimable | not estimable | not estimable | -0 | 0 | not estimable | not estimable | not estimable |
| Bicycle | Unchanged | 2980 | 0 | 0 | 0 | 0 | 0 | 0 | -0 | 0 | not estimable | not estimable | not estimable |
| Bicycle | HighCostWorsened | 0 | 0 | 0 | not estimable | not estimable | not estimable | not estimable | -0 | 0 | not estimable | not estimable | not estimable |
| MovingVehicle | Improved | 14113 | 14113 | 23541 | -5.25885 | -4.61312 | -0.491014 | -0.14317 | 74218.2 | 39588.3 | 7.63817 | 10.3234 | 16.0816 |
| MovingVehicle | Worsened | 9428 | 9428 | 23541 | 4.19902 | 3.98304 | 7.63817 | 10.3234 | 74218.2 | 39588.3 | 7.63817 | 10.3234 | 16.0816 |
| MovingVehicle | Unchanged | 18187 | 0 | 23541 | 0 | 0 | 0 | 0 | 74218.2 | 39588.3 | 7.63817 | 10.3234 | 16.0816 |
| MovingVehicle | HighCostWorsened | 3354 | 3354 | 23541 | 7.51218 | 6.29327 | 11.6132 | 14.1804 | 74218.2 | 39588.3 | 7.63817 | 10.3234 | 16.0816 |
| Pedestrian5-8m | Improved | 4157 | 4157 | 7733 | -0.36781 | -0.317637 | -0.0590871 | -0.0270666 | 1528.99 | 1176.75 | 0.549103 | 0.722807 | 1.45356 |
| Pedestrian5-8m | Worsened | 3576 | 3576 | 7733 | 0.329068 | 0.29292 | 0.549103 | 0.722807 | 1528.99 | 1176.75 | 0.549103 | 0.722807 | 1.45356 |
| Pedestrian5-8m | Unchanged | 14750 | 0 | 7733 | 0 | 0 | 0 | 0 | 1528.99 | 1176.75 | 0.549103 | 0.722807 | 1.45356 |
| Pedestrian5-8m | HighCostWorsened | 0 | 0 | 7733 | not estimable | not estimable | not estimable | not estimable | 1528.99 | 1176.75 | 0.549103 | 0.722807 | 1.45356 |

Paired semantic changes for high-cost and ordinary harms, compared identically with improvements:

| Group | Outcome | Feature | OutcomeActors | Count | Mean | Median | P90 | P95 | MeanFDEDeltaPaired |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vehicle | Improved | centerline_mean_distance | 68003 | 52155 | -0.0257929 | -4.97699e-05 | 0.128626 | 1.40385 | -1.51545 |
| Vehicle | Improved | centerline_heading_error | 68003 | 52155 | -0.0558947 | -0.0064922 | 1.24085 | 1.31034 | -1.51545 |
| Vehicle | Improved | drivable_inside_fraction | 68003 | 56563 | 0.00113001 | 0 | 0 | 0 | -1.40655 |
| Vehicle | Improved | crosswalk_distance | 68003 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Vehicle | Improved | crosswalk_intersects | 68003 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Vehicle | Improved | walkway_inside_fraction | 68003 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Vehicle | Worsened | centerline_mean_distance | 24258 | 19847 | -0.0835487 | -0.000882447 | 0.815013 | 2.36494 | 2.16573 |
| Vehicle | Worsened | centerline_heading_error | 24258 | 19847 | 0.00259624 | -0.00105311 | 1.20226 | 1.3961 | 2.16573 |
| Vehicle | Worsened | drivable_inside_fraction | 24258 | 21059 | 0.000387799 | 0 | 0 | 0 | 2.05315 |
| Vehicle | Worsened | crosswalk_distance | 24258 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Vehicle | Worsened | crosswalk_intersects | 24258 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Vehicle | Worsened | walkway_inside_fraction | 24258 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Vehicle | HighCostWorsened | centerline_mean_distance | 3599 | 3550 | 0.04303 | -0.0135337 | 3.61235 | 5.92987 | 7.6037 |
| Vehicle | HighCostWorsened | centerline_heading_error | 3599 | 3550 | -0.0157712 | -0.00280741 | 0.153683 | 1.30744 | 7.6037 |
| Vehicle | HighCostWorsened | drivable_inside_fraction | 3599 | 3551 | -0.00638318 | 0 | 0 | 0 | 7.6003 |
| Vehicle | HighCostWorsened | crosswalk_distance | 3599 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Vehicle | HighCostWorsened | crosswalk_intersects | 3599 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Vehicle | HighCostWorsened | walkway_inside_fraction | 3599 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | Improved | centerline_mean_distance | 9805 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | Improved | centerline_heading_error | 9805 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | Improved | drivable_inside_fraction | 9805 | 6326 | 0.00117241 | 0 | 0 | 0.0833333 | -0.418667 |
| Pedestrian | Improved | crosswalk_distance | 9805 | 2562 | 0.00179613 | 0 | 0.0293977 | 0.230198 | -0.433 |
| Pedestrian | Improved | crosswalk_intersects | 9805 | 2562 | 0.00429352 | 0 | 0 | 0 | -0.433 |
| Pedestrian | Improved | walkway_inside_fraction | 9805 | 7750 | 0.000634409 | 0 | 0.0833333 | 0.0833333 | -0.399435 |
| Pedestrian | Worsened | centerline_mean_distance | 8335 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | Worsened | centerline_heading_error | 8335 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | Worsened | drivable_inside_fraction | 8335 | 5245 | 0.00387671 | 0 | 0 | 0.0833333 | 0.362139 |
| Pedestrian | Worsened | crosswalk_distance | 8335 | 2158 | 9.94977e-05 | 0 | 0.0294705 | 0.0911668 | 0.383102 |
| Pedestrian | Worsened | crosswalk_intersects | 8335 | 2158 | 0.00231696 | 0 | 0 | 0 | 0.383102 |
| Pedestrian | Worsened | walkway_inside_fraction | 8335 | 6713 | -0.00414618 | 0 | 0 | 0.0833333 | 0.349199 |
| Pedestrian | HighCostWorsened | centerline_mean_distance | 4 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | HighCostWorsened | centerline_heading_error | 4 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | HighCostWorsened | drivable_inside_fraction | 4 | 3 | 0.111111 | 0 | 0.266667 | 0.3 | 5.27724 |
| Pedestrian | HighCostWorsened | crosswalk_distance | 4 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | HighCostWorsened | crosswalk_intersects | 4 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian | HighCostWorsened | walkway_inside_fraction | 4 | 3 | -0.138889 | 0 | 0 | 0 | 5.27724 |
| MovingVehicle | Improved | centerline_mean_distance | 14113 | 13964 | -0.0748392 | -0.0176494 | 2.90789 | 5.18318 | -5.26585 |
| MovingVehicle | Improved | centerline_heading_error | 14113 | 13964 | -0.140947 | -0.00465613 | 0.114439 | 0.955374 | -5.26585 |
| MovingVehicle | Improved | drivable_inside_fraction | 14113 | 13979 | 0.00401197 | 0 | 0 | 0.0833333 | -5.27511 |
| MovingVehicle | Improved | crosswalk_distance | 14113 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| MovingVehicle | Improved | crosswalk_intersects | 14113 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| MovingVehicle | Improved | walkway_inside_fraction | 14113 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| MovingVehicle | Worsened | centerline_mean_distance | 9428 | 9325 | -0.198153 | -0.0492184 | 2.35907 | 4.33299 | 4.21277 |
| MovingVehicle | Worsened | centerline_heading_error | 9428 | 9325 | -0.0521809 | -0.00205946 | 0.160297 | 1.1748 | 4.21277 |
| MovingVehicle | Worsened | drivable_inside_fraction | 9428 | 9342 | 0.00332727 | 0 | 0.0833333 | 0.0833333 | 4.20809 |
| MovingVehicle | Worsened | crosswalk_distance | 9428 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| MovingVehicle | Worsened | crosswalk_intersects | 9428 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| MovingVehicle | Worsened | walkway_inside_fraction | 9428 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| MovingVehicle | HighCostWorsened | centerline_mean_distance | 3354 | 3339 | -0.00388512 | -0.0193444 | 3.60593 | 5.95103 | 7.50545 |
| MovingVehicle | HighCostWorsened | centerline_heading_error | 3354 | 3339 | -0.00755058 | -0.0026413 | 0.111438 | 1.24649 | 7.50545 |
| MovingVehicle | HighCostWorsened | drivable_inside_fraction | 3354 | 3336 | -0.00467126 | 0 | 0 | 0 | 7.50069 |
| MovingVehicle | HighCostWorsened | crosswalk_distance | 3354 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| MovingVehicle | HighCostWorsened | crosswalk_intersects | 3354 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| MovingVehicle | HighCostWorsened | walkway_inside_fraction | 3354 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | Improved | centerline_mean_distance | 4157 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | Improved | centerline_heading_error | 4157 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | Improved | drivable_inside_fraction | 4157 | 2625 | 0.00685714 | 0 | 0.0833333 | 0.0833333 | -0.398218 |
| Pedestrian5-8m | Improved | crosswalk_distance | 4157 | 988 | 0.00611173 | 0 | 0.0432308 | 0.419968 | -0.415984 |
| Pedestrian5-8m | Improved | crosswalk_intersects | 4157 | 988 | 0.0172065 | 0 | 0 | 0 | -0.415984 |
| Pedestrian5-8m | Improved | walkway_inside_fraction | 4157 | 3354 | -0.0024846 | 0 | 0.0833333 | 0.0833333 | -0.363835 |
| Pedestrian5-8m | Worsened | centerline_mean_distance | 3576 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | Worsened | centerline_heading_error | 3576 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | Worsened | drivable_inside_fraction | 3576 | 2231 | -0.00272673 | 0 | 0 | 0.0833333 | 0.342202 |
| Pedestrian5-8m | Worsened | crosswalk_distance | 3576 | 832 | 0.017664 | 0 | 0.0272515 | 0.319456 | 0.360832 |
| Pedestrian5-8m | Worsened | crosswalk_intersects | 3576 | 832 | -0.0108173 | 0 | 0 | 0 | 0.360832 |
| Pedestrian5-8m | Worsened | walkway_inside_fraction | 3576 | 2861 | 0.00401957 | 0 | 0.0833333 | 0.0833333 | 0.333743 |
| Pedestrian5-8m | HighCostWorsened | centerline_mean_distance | 0 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | HighCostWorsened | centerline_heading_error | 0 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | HighCostWorsened | drivable_inside_fraction | 0 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | HighCostWorsened | crosswalk_distance | 0 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | HighCostWorsened | crosswalk_intersects | 0 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |
| Pedestrian5-8m | HighCostWorsened | walkway_inside_fraction | 0 | 0 | not estimable | not estimable | not estimable | not estimable | not estimable |

## Semantic versus motion/interaction information

Matching is within the same actor, hence type,region and observed history are identical, and within fixed predicted-displacement bins [0,1,5,10,20,inf). At least3 valid varying modes are needed in a bin. The actor's eligible-bin correlations are averaged before equal-scene aggregation. This is a coarse no-training diagnostic, not full adjustment for endpoint,heading,curvature or interactions. Matching can substantially reduce estimable coverage; its denominators are reported.

| Scope | ActorType | Feature | CommonActors | Scenes | Real_SceneMeanRho | NC1_SceneMeanRho | NC2_SceneMeanRho | DeltaRealMinusNC1 | DeltaNC1_CI95Low | DeltaNC1_CI95High | DeltaRealMinusNC2 | DeltaNC2_CI95Low | DeltaNC2_CI95High |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pooled | Vehicle | centerline_mean_distance | 112939 | 629 | 0.0444208 | -0.00273622 | -0.00119354 | 0.0471571 | 0.0301731 | 0.0642817 | 0.0456144 | 0.0281615 | 0.0639701 |
| Fold1 | Vehicle | centerline_mean_distance | 36848 | 209 | 0.0130172 | -0.00757327 | 0.00376835 | 0.0205904 | -0.00975515 | 0.052364 | 0.00924883 | -0.0234789 | 0.0425876 |
| Fold2 | Vehicle | centerline_mean_distance | 36246 | 210 | 0.046461 | 0.000406072 | -0.00835984 | 0.0460549 | 0.0151953 | 0.0757245 | 0.0548208 | 0.0236211 | 0.0867371 |
| Fold3 | Vehicle | centerline_mean_distance | 39845 | 210 | 0.0736349 | -0.0010645 | 0.00103448 | 0.0746994 | 0.0426579 | 0.103942 | 0.0726004 | 0.0416585 | 0.101665 |
| Pooled | Vehicle | centerline_heading_error | 112939 | 629 | 0.0247179 | -0.000702932 | 0.00354462 | 0.0254208 | 0.012447 | 0.0384694 | 0.0211733 | 0.00825444 | 0.0340223 |
| Fold1 | Vehicle | centerline_heading_error | 36848 | 209 | 0.00918377 | 0.00176112 | -0.00371538 | 0.00742265 | -0.0138204 | 0.0290165 | 0.0128992 | -0.0095887 | 0.0360939 |
| Fold2 | Vehicle | centerline_heading_error | 36246 | 210 | 0.0280263 | -7.00399e-05 | 0.0134872 | 0.0280963 | 0.00837947 | 0.0479626 | 0.0145391 | -0.00535823 | 0.0335989 |
| Fold3 | Vehicle | centerline_heading_error | 39845 | 210 | 0.0368696 | -0.00378814 | 0.000827434 | 0.0406578 | 0.0161426 | 0.0641374 | 0.0360422 | 0.0113215 | 0.0597218 |
| Pooled | Vehicle | drivable_inside_fraction | 6610 | 510 | 0.167752 | 0.00115774 | -0.00254078 | 0.166594 | 0.11707 | 0.215141 | 0.170293 | 0.123848 | 0.216235 |
| Fold1 | Vehicle | drivable_inside_fraction | 2291 | 175 | 0.168044 | 0.0114602 | -0.0125357 | 0.156583 | 0.0655053 | 0.242136 | 0.180579 | 0.0957065 | 0.256202 |
| Fold2 | Vehicle | drivable_inside_fraction | 2190 | 165 | 0.134041 | 0.00413197 | 0.0156623 | 0.129909 | 0.0486034 | 0.215405 | 0.118378 | 0.0405791 | 0.192793 |
| Fold3 | Vehicle | drivable_inside_fraction | 2129 | 170 | 0.200171 | -0.0123345 | -0.00991962 | 0.212506 | 0.132052 | 0.289331 | 0.210091 | 0.122776 | 0.297112 |
| Pooled | Pedestrian | crosswalk_distance | 609 | 145 | -0.0441011 | -0.0251306 | -0.0561019 | -0.0189705 | -0.141169 | 0.095042 | 0.0120008 | -0.11842 | 0.133723 |
| Fold1 | Pedestrian | crosswalk_distance | 177 | 44 | -0.0702965 | 0.0141312 | -0.112317 | -0.0844278 | -0.291807 | 0.116368 | 0.0420207 | -0.174828 | 0.254431 |
| Fold2 | Pedestrian | crosswalk_distance | 206 | 50 | -0.0649148 | 0.00510548 | -0.0460041 | -0.0700203 | -0.280247 | 0.140854 | -0.0189108 | -0.197291 | 0.150843 |
| Fold3 | Pedestrian | crosswalk_distance | 226 | 51 | -0.00109551 | -0.0886469 | -0.0175023 | 0.0875514 | -0.1193 | 0.282765 | 0.0164068 | -0.251349 | 0.273126 |
| Pooled | Pedestrian | walkway_inside_fraction | 11452 | 431 | 0.0885277 | -0.0102704 | 0.0146618 | 0.0987981 | 0.0490676 | 0.146305 | 0.0738659 | 0.0282824 | 0.118997 |
| Fold1 | Pedestrian | walkway_inside_fraction | 3372 | 143 | 0.07162 | 0.000850067 | 0.0144202 | 0.0707699 | -0.0217759 | 0.160346 | 0.0571998 | -0.0270722 | 0.140762 |
| Fold2 | Pedestrian | walkway_inside_fraction | 3523 | 141 | 0.101492 | 0.00687908 | 0.00733452 | 0.0946127 | 0.00627535 | 0.178878 | 0.0941573 | 0.0191687 | 0.170035 |
| Fold3 | Pedestrian | walkway_inside_fraction | 4557 | 147 | 0.0925403 | -0.0375379 | 0.021925 | 0.130078 | 0.0648926 | 0.198859 | 0.0706153 | -0.00335677 | 0.147095 |
| Pooled | Pedestrian | crosswalk_intersects | 87 | 59 | -0.0197847 | -0.0078966 | 0.0481732 | -0.0118881 | -0.22793 | 0.199548 | -0.0679579 | -0.309272 | 0.157171 |
| Fold1 | Pedestrian | crosswalk_intersects | 25 | 17 | -0.0634714 | 0.0913254 | -0.00296753 | -0.154797 | -0.588044 | 0.287061 | -0.0605038 | -0.487488 | 0.334596 |
| Fold2 | Pedestrian | crosswalk_intersects | 31 | 21 | 0.0324329 | 0.0393293 | 0.0547415 | -0.00689642 | -0.338673 | 0.302042 | -0.0223086 | -0.415901 | 0.340137 |
| Fold3 | Pedestrian | crosswalk_intersects | 31 | 21 | -0.036637 | -0.135445 | 0.0830045 | 0.0988081 | -0.275773 | 0.43598 | -0.119642 | -0.590034 | 0.30107 |

Registered SemanticIncrementalValue=SUGGESTED; this label concerns the semantic screening relation after coarse matching. Independent true prediction gain is **UNRESOLVED**. Semantic correlations with frozen endpoint, displacement,heading,length,curvature and interaction summaries show potential redundancy; they do not isolate map information causally. Heading correlations use wrapped angle ranks, a descriptive limitation.

`07_incremental_information/stage12a_region_displacement_associations.csv` additionally reports each type/map-region/predicted-displacement-bin combination in all3folds and pooled, retaining explicit unestimable rows; the region order is the frozen sorted dictionary order. This exploratory table is not used to change the six-feature primary screen.

| ActorType | SemanticFeature | ObservableControl | Count | Scenes | SceneMean |
| --- | --- | --- | --- | --- | --- |
| Vehicle | centerline_mean_distance | predicted_displacement | 149355 | 629 | 0.451396 |
| Vehicle | centerline_mean_distance | endpoint_x | 149355 | 629 | 0.14029 |
| Vehicle | centerline_mean_distance | endpoint_y | 149355 | 629 | 0.000480016 |
| Vehicle | centerline_mean_distance | predicted_heading | 149355 | 629 | 0.118347 |
| Vehicle | centerline_mean_distance | trajectory_length | 149355 | 629 | 0.451141 |
| Vehicle | centerline_mean_distance | trajectory_curvature | 149355 | 629 | -0.328111 |
| Vehicle | centerline_mean_distance | interaction_minimum_distance | 148469 | 625 | 0.0378775 |
| Vehicle | centerline_mean_distance | interaction_mean_distance | 148469 | 625 | 0.268752 |
| Vehicle | centerline_mean_distance | interaction_closing_mean | 148469 | 625 | -0.327933 |
| Vehicle | centerline_heading_error | predicted_displacement | 149355 | 629 | 0.0208347 |
| Vehicle | centerline_heading_error | endpoint_x | 149355 | 629 | -0.0174805 |
| Vehicle | centerline_heading_error | endpoint_y | 149355 | 629 | 0.029161 |
| Vehicle | centerline_heading_error | predicted_heading | 149355 | 629 | 0.0589688 |
| Vehicle | centerline_heading_error | trajectory_length | 149355 | 629 | 0.0210139 |
| Vehicle | centerline_heading_error | trajectory_curvature | 149355 | 629 | 0.0085154 |
| Vehicle | centerline_heading_error | interaction_minimum_distance | 148469 | 625 | 0.0119563 |
| Vehicle | centerline_heading_error | interaction_mean_distance | 148469 | 625 | 0.051436 |
| Vehicle | centerline_heading_error | interaction_closing_mean | 148469 | 625 | -0.0568196 |
| Vehicle | drivable_inside_fraction | predicted_displacement | 74577 | 619 | 0.695094 |
| Vehicle | drivable_inside_fraction | endpoint_x | 74577 | 619 | 0.190353 |
| Vehicle | drivable_inside_fraction | endpoint_y | 74577 | 619 | -0.0666219 |
| Vehicle | drivable_inside_fraction | predicted_heading | 74577 | 619 | 0.0619562 |
| Vehicle | drivable_inside_fraction | trajectory_length | 74577 | 619 | 0.695067 |
| Vehicle | drivable_inside_fraction | trajectory_curvature | 74577 | 619 | -0.577103 |
| Vehicle | drivable_inside_fraction | interaction_minimum_distance | 74028 | 615 | 0.0937081 |
| Vehicle | drivable_inside_fraction | interaction_mean_distance | 74028 | 615 | 0.403998 |
| Vehicle | drivable_inside_fraction | interaction_closing_mean | 74028 | 615 | -0.49485 |
| Pedestrian | crosswalk_distance | predicted_displacement | 9896 | 261 | -0.191967 |
| Pedestrian | crosswalk_distance | endpoint_x | 9896 | 261 | -0.00488812 |
| Pedestrian | crosswalk_distance | endpoint_y | 9896 | 261 | -0.0253818 |
| Pedestrian | crosswalk_distance | predicted_heading | 9896 | 261 | 0.0148685 |
| Pedestrian | crosswalk_distance | trajectory_length | 9896 | 261 | -0.191739 |
| Pedestrian | crosswalk_distance | trajectory_curvature | 9896 | 261 | 0.0847377 |
| Pedestrian | crosswalk_distance | interaction_minimum_distance | 9896 | 261 | 0.0231595 |
| Pedestrian | crosswalk_distance | interaction_mean_distance | 9896 | 261 | 0.0180737 |
| Pedestrian | crosswalk_distance | interaction_closing_mean | 9896 | 261 | 0.00553024 |
| Pedestrian | walkway_inside_fraction | predicted_displacement | 25522 | 459 | 0.3829 |
| Pedestrian | walkway_inside_fraction | endpoint_x | 25522 | 459 | 0.056289 |
| Pedestrian | walkway_inside_fraction | endpoint_y | 25522 | 459 | -0.023146 |
| Pedestrian | walkway_inside_fraction | predicted_heading | 25522 | 459 | -0.203619 |
| Pedestrian | walkway_inside_fraction | trajectory_length | 25522 | 459 | 0.382859 |
| Pedestrian | walkway_inside_fraction | trajectory_curvature | 25522 | 459 | -0.219118 |
| Pedestrian | walkway_inside_fraction | interaction_minimum_distance | 25494 | 458 | 0.0458085 |
| Pedestrian | walkway_inside_fraction | interaction_mean_distance | 25494 | 458 | 0.149001 |
| Pedestrian | walkway_inside_fraction | interaction_closing_mean | 25494 | 458 | -0.186419 |
| Pedestrian | crosswalk_intersects | predicted_displacement | 4195 | 223 | -0.59262 |
| Pedestrian | crosswalk_intersects | endpoint_x | 4195 | 223 | -0.0548371 |
| Pedestrian | crosswalk_intersects | endpoint_y | 4195 | 223 | 0.0278823 |
| Pedestrian | crosswalk_intersects | predicted_heading | 4195 | 223 | 0.24323 |
| Pedestrian | crosswalk_intersects | trajectory_length | 4195 | 223 | -0.592944 |
| Pedestrian | crosswalk_intersects | trajectory_curvature | 4195 | 223 | 0.423977 |
| Pedestrian | crosswalk_intersects | interaction_minimum_distance | 4195 | 223 | -0.023636 |
| Pedestrian | crosswalk_intersects | interaction_mean_distance | 4195 | 223 | -0.111328 |
| Pedestrian | crosswalk_intersects | interaction_closing_mean | 4195 | 223 | 0.184187 |

## Bicycle preservation

All2980 Bicycle targets and17880 candidates participate in coverage, feature distributions and switch diagnostics. No semantic reranking is performed. C Raw logits/probabilities are bitwise FoldR2 for Bicycle; C Calibrated is the existing passthrough, so every Bicycle Top1 remains unchanged. No Vehicle/Pedestrian signal is extrapolated to Bicycle.

## Thirty registered BEV cases

Exactly10 Vehicle errors,10 Pedestrian errors,5 Vehicle successes and5 Pedestrian successes. Errors rank descending C oracle-gap and successes descending FDE gain, actor_id ties; one actor per scene is preferred. Map coverage/semantic values never filter or order cases. Each PNG/SVG/PDF overlays original HD Map,GT,all6 candidates,R2/C/oracle in the same t0 ego axes, includes scene+actor+type+ΔFDE and six-mode feature values with masks. GT and oracle are offline figure references only. Full traceable IDs and original entity tokens are available through case JSON and frozen dictionary.

| Category | Ordinal | Fold | CFDE | R2FDE | OracleFDE | DeltaFDE | CMapValid | Figure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VehicleError | 1 | 1 | 50.9469 | 41.7873 | 11.0517 | 9.15957 | True | stage12a_vehicle_error_01.png |
| VehicleError | 2 | 1 | 48.7533 | 40.557 | 9.27238 | 8.1963 | True | stage12a_vehicle_error_02.png |
| VehicleError | 3 | 2 | 45.0398 | 37.406 | 5.87658 | 7.63374 | True | stage12a_vehicle_error_03.png |
| VehicleError | 4 | 2 | 66.1025 | 57.2055 | 27.0484 | 8.89698 | True | stage12a_vehicle_error_04.png |
| VehicleError | 5 | 2 | 43.1107 | 16.1777 | 4.13803 | 26.933 | True | stage12a_vehicle_error_05.png |
| VehicleError | 6 | 2 | 48.1257 | 40.7903 | 10.9898 | 7.33533 | True | stage12a_vehicle_error_06.png |
| VehicleError | 7 | 3 | 38.0784 | 31.0011 | 1.7982 | 7.07726 | True | stage12a_vehicle_error_07.png |
| VehicleError | 8 | 2 | 41.3086 | 41.3086 | 5.1816 | 0 | True | stage12a_vehicle_error_08.png |
| VehicleError | 9 | 1 | 48.6155 | 42.4189 | 12.9687 | 6.19654 | True | stage12a_vehicle_error_09.png |
| VehicleError | 10 | 3 | 41.8692 | 51.1507 | 7.50746 | -9.28143 | True | stage12a_vehicle_error_10.png |
| VehicleSuccess | 1 | 2 | 19.2767 | 55.2906 | 13.0528 | -36.0139 | True | stage12a_vehicle_success_01.png |
| VehicleSuccess | 2 | 2 | 2.71522 | 34.5364 | 2.71522 | -31.8212 | True | stage12a_vehicle_success_02.png |
| VehicleSuccess | 3 | 3 | 0.864586 | 32.244 | 0.864586 | -31.3794 | True | stage12a_vehicle_success_03.png |
| VehicleSuccess | 4 | 3 | 25.598 | 56.3157 | 9.54562 | -30.7177 | True | stage12a_vehicle_success_04.png |
| VehicleSuccess | 5 | 3 | 4.82737 | 35.5333 | 2.03564 | -30.7059 | True | stage12a_vehicle_success_05.png |
| PedestrianError | 1 | 1 | 20.2572 | 17.343 | 4.21396 | 2.91425 | True | stage12a_pedestrian_error_01.png |
| PedestrianError | 2 | 1 | 8.55385 | 8.55385 | 0.0472738 | 0 | True | stage12a_pedestrian_error_02.png |
| PedestrianError | 3 | 1 | 9.01063 | 9.01063 | 0.54528 | 0 | True | stage12a_pedestrian_error_03.png |
| PedestrianError | 4 | 2 | 8.96408 | 7.86278 | 0.874029 | 1.10129 | True | stage12a_pedestrian_error_04.png |
| PedestrianError | 5 | 3 | 9.44856 | 9.33957 | 1.37743 | 0.108991 | True | stage12a_pedestrian_error_05.png |
| PedestrianError | 6 | 1 | 8.50494 | 8.50494 | 0.606064 | 0 | True | stage12a_pedestrian_error_06.png |
| PedestrianError | 7 | 3 | 8.8349 | 8.8349 | 1.11721 | 0 | True | stage12a_pedestrian_error_07.png |
| PedestrianError | 8 | 3 | 10.0448 | 10.0448 | 2.41832 | 0 | True | stage12a_pedestrian_error_08.png |
| PedestrianError | 9 | 2 | 9.07197 | 9.07197 | 1.54874 | 0 | True | stage12a_pedestrian_error_09.png |
| PedestrianError | 10 | 1 | 8.85942 | 8.85942 | 1.35136 | 0 | True | stage12a_pedestrian_error_10.png |
| PedestrianSuccess | 1 | 1 | 2.2477 | 7.72945 | 2.2477 | -5.48175 | True | stage12a_pedestrian_success_01.png |
| PedestrianSuccess | 2 | 2 | 3.07085 | 8.21101 | 0.769632 | -5.14016 | True | stage12a_pedestrian_success_02.png |
| PedestrianSuccess | 3 | 1 | 3.69196 | 8.41605 | 2.13864 | -4.72408 | True | stage12a_pedestrian_success_03.png |
| PedestrianSuccess | 4 | 1 | 0.180349 | 4.464 | 0.180349 | -4.28365 | True | stage12a_pedestrian_success_04.png |
| PedestrianSuccess | 5 | 1 | 2.93036 | 7.17014 | 2.54546 | -4.23978 | True | stage12a_pedestrian_success_05.png |

## Reproduction and stop

Use the existing `ped_intent` environment and `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`; no package/environment upgrades. Ordered commands are recorded in `00_manifest/stage12a_run_commands.txt`. Registration, preflight, complete semantic freeze, statistics, negative controls, cases, independent verification and report run in that order. Feature extraction supports per-scene resume with SHA256 and source code hashes. Numerical failures abort instead of creating substitute data.

All required small tables,protocols,verification,figures and reports are versioned on the Stage12A branch; raw arrays/full actor CSV remain local. No merge is performed. Final action: **STOP** and wait for 大脑AI review, even when ReadyForStage12B=YES. No Stage12B training, semantic GNN, official VAL/test, HiVT retraining or loss-parameter search was executed.
