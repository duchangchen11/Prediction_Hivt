# Stage8A-0B actor-type-aware semantic entity coverage audit

## 【Why Stage8A-0 Failed】

Stage8A-0 stays **NOT_READY**, with lane-only fallback 25.170734% TRAIN / 26.539193% VAL in current-valid Vehicle modes. This audit introduces a separate actor-type coverage definition and does not rewrite that result. Predictions are the frozen Stage5A candidates; R2 weights and historical rankings are untouched. All six modes have equal diagnostic weight.

| Split | Population | ActorGroup | Candidates | LaneFallbackCandidates | ShareOfVehicleLaneFallback |
| --- | --- | --- | --- | --- | --- |
| train | current_valid | MovingVehicle | 446910 | 9208 | 1.8112% |
| train | current_valid | StoppedVehicle | 195252 | 9302 | 1.8297% |
| train | current_valid | ParkedVehicle | 1330116 | 483687 | 95.1389% |
| train | current_valid | UnknownVehicle | 47532 | 6204 | 1.2203% |
| train | full_horizon_ranking_targets | MovingVehicle | 280770 | 3652 | 1.3634% |
| train | full_horizon_ranking_targets | StoppedVehicle | 155634 | 5667 | 2.1156% |
| train | full_horizon_ranking_targets | ParkedVehicle | 817428 | 256204 | 95.6467% |
| train | full_horizon_ranking_targets | UnknownVehicle | 28638 | 2342 | 0.8743% |
| val | current_valid | MovingVehicle | 102018 | 1922 | 1.7933% |
| val | current_valid | StoppedVehicle | 47940 | 2358 | 2.2001% |
| val | current_valid | ParkedVehicle | 243306 | 101096 | 94.3253% |
| val | current_valid | UnknownVehicle | 10584 | 1802 | 1.6813% |
| val | full_horizon_ranking_targets | MovingVehicle | 62766 | 623 | 1.0067% |
| val | full_horizon_ranking_targets | StoppedVehicle | 33594 | 1131 | 1.8275% |
| val | full_horizon_ranking_targets | ParkedVehicle | 151188 | 58946 | 95.2463% |
| val | full_horizon_ranking_targets | UnknownVehicle | 6444 | 1188 | 1.9196% |

## 【Lane-only Fallback Decomposition】

Complete centerlines, fixed 10 m and top8, with exact token/distance reproduction of the historical nearest-one rule. Every coverage count is **before** nearest fallback. All four observed vehicle motion states are reported. Missing groups have undefined rates, never implicit success. The complete tables include current-valid and full-horizon populations.

| Split | MotionState | Actors | Candidates | LaneOnlyFallbackRate | TypeAwareFallbackRate | RecoveredByDrivableRate | RecoveredByCarparkRate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | all | 213745 | 1282470 | 20.8866% | 11.3058% | 45.8709% | 2.7473% |
| train | vehicle.moving | 46795 | 280770 | 1.3007% | 0.9958% | 23.4392% | 0.0000% |
| train | vehicle.stopped | 25939 | 155634 | 3.6412% | 2.2983% | 36.8802% | 2.6645% |
| train | vehicle.parked | 136238 | 817428 | 31.3427% | 16.8099% | 46.3673% | 2.8114% |
| train | unknown | 4773 | 28638 | 8.1779% | 4.2286% | 48.2921% | 0.2135% |
| val | all | 42332 | 253992 | 24.3661% | 15.3477% | 37.0120% | 16.7011% |
| val | vehicle.moving | 10461 | 62766 | 0.9926% | 0.8030% | 19.1011% | 5.4575% |
| val | vehicle.stopped | 5599 | 33594 | 3.3667% | 2.8457% | 15.4730% | 12.5553% |
| val | vehicle.parked | 25198 | 151188 | 38.9885% | 24.3882% | 37.4478% | 16.8324% |
| val | unknown | 1074 | 6444 | 18.4358% | 10.0869% | 45.2862% | 20.0337% |

Recovery denominators are lane-fallback candidates, with possible drivable/carpark overlap. The CSV also includes union, intersection and unconditional recovery. Candidate displacement is last minus first of the twelve predicted points, with bins [0,1], (1,5], >5 m; it never uses GT. Actor-level all6/any1/all6-failure counts and rates are in `stage8a0b_actor_fallback.csv`.

| Split | PredictedNetDisplacementBin | BinCandidates | LaneOnlyFallbackRate | TypeAwareFallbackRate |
| --- | --- | --- | --- | --- |
| train | 0-1m | 675840 | 27.2643% | 14.8167% |
| train | 1-5m | 126367 | 23.4935% | 12.7066% |
| train | >5m | 480263 | 11.2259% | 5.9965% |
| val | 0-1m | 129108 | 32.5898% | 20.4046% |
| val | 1-5m | 23231 | 29.5553% | 19.2760% |
| val | >5m | 101653 | 12.7355% | 8.0273% |

## 【Map Schema】

All four actual local raw JSON maps and the installed NuScenesMap API were read. All seven requested layers exist. Lane and connector retrieval geometry is the frozen complete LineString; area layers use actual polygon references, including plural `polygon_tokens` for multipart drivable-area records. Stop_line is audited and matched diagnostically but excluded from every primary graph selector.

| MapRegion | Layer | Records | ValidGeometry | InvalidGeometry | UsableGeometry | InvalidReference |
| --- | --- | --- | --- | --- | --- | --- |
| boston-seaport | lane | 1215 | 1215 | 0 | 1215 | 0 |
| boston-seaport | lane_connector | 1631 | 1631 | 0 | 1631 | 0 |
| boston-seaport | drivable_area | 2 | 2 | 0 | 2 | 0 |
| boston-seaport | carpark_area | 275 | 275 | 0 | 275 | 0 |
| boston-seaport | ped_crossing | 340 | 340 | 0 | 340 | 0 |
| boston-seaport | walkway | 301 | 300 | 1 | 300 | 0 |
| boston-seaport | stop_line | 775 | 775 | 0 | 775 | 0 |
| singapore-hollandvillage | lane | 601 | 601 | 0 | 601 | 0 |
| singapore-hollandvillage | lane_connector | 719 | 719 | 0 | 719 | 0 |
| singapore-hollandvillage | drivable_area | 426 | 426 | 0 | 426 | 0 |
| singapore-hollandvillage | carpark_area | 0 | 0 | 0 | 0 | 0 |
| singapore-hollandvillage | ped_crossing | 28 | 28 | 0 | 28 | 0 |
| singapore-hollandvillage | walkway | 498 | 498 | 0 | 498 | 0 |
| singapore-hollandvillage | stop_line | 300 | 300 | 0 | 300 | 0 |
| singapore-onenorth | lane | 936 | 936 | 0 | 936 | 0 |
| singapore-onenorth | lane_connector | 1066 | 1066 | 0 | 1066 | 0 |
| singapore-onenorth | drivable_area | 1 | 1 | 0 | 1 | 0 |
| singapore-onenorth | carpark_area | 39 | 39 | 0 | 39 | 0 |
| singapore-onenorth | ped_crossing | 120 | 120 | 0 | 120 | 0 |
| singapore-onenorth | walkway | 838 | 838 | 0 | 838 | 0 |
| singapore-onenorth | stop_line | 451 | 451 | 0 | 451 | 0 |
| singapore-queenstown | lane | 910 | 910 | 0 | 910 | 0 |
| singapore-queenstown | lane_connector | 1173 | 1173 | 0 | 1173 | 0 |
| singapore-queenstown | drivable_area | 219 | 219 | 0 | 219 | 0 |
| singapore-queenstown | carpark_area | 40 | 40 | 0 | 40 | 0 |
| singapore-queenstown | ped_crossing | 75 | 75 | 0 | 75 | 0 |
| singapore-queenstown | walkway | 457 | 457 | 0 | 457 | 0 |
| singapore-queenstown | stop_line | 437 | 437 | 0 | 437 | 0 |

## 【Polygon Geometry Integrity】

Finite source coordinates and closed API polygon rings were verified. Invalid components: 1; invalid audited references: 0; partially usable multipart records: 0. Boston's one invalid walkway polygon is excluded exactly as preregistered. No buffer, topology repair, invented association or new label was used. Multipart tokens are deduplicated without unioning their geometry.

A polygon matches if a predicted future point lies inside, the future polyline intersects, or its true boundary distance is <=2 m. Interior intersection yields filled-geometry distance zero even when the boundary is far away. Sorting uses filled geometry distance, then type ID/token. The independent brute-force check evaluated 12 candidates across all four maps against every usable entity, and reproduced counts and top8 IDs/distances. Synthetic geometry checks cover segment interiors, crossing with both endpoints outside, interior far from boundary, and the exact fixed 2 m inclusion boundary.

## 【Vehicle Type-Aware Context】

Vehicle relevance is lane/connector/drivable/carpark. Coverage remains uncapped and before any fallback. Full-horizon TRAIN and VAL gates are applied independently.

| Split | Actors | RouteCenterlineCoverage | AreaContextCoverage | CarparkCoverage | TypeAwareAnyCoverage | GenericOnlyRate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 213745 | 79.1134% | 84.7104% | 26.7231% | 88.6942% | 8.6935% |
| val | 42332 | 75.6339% | 80.3124% | 29.8856% | 84.6523% | 4.8498% |

## 【Moving Vehicle】

| Split | Actors | RouteCenterlineCoverage | AreaContextCoverage | TypeAwareAnyCoverage | TypeAwareFallbackRate |
| --- | --- | --- | --- | --- | --- |
| train | 46795 | 98.6993% | 98.8275% | 99.0042% | 0.9958% |
| val | 10461 | 99.0074% | 98.9405% | 99.1970% | 0.8030% |

Moving vehicles retain the independent >=95% route-centerline gate; area coverage cannot replace it.

## 【Stopped Vehicle】

| Split | Actors | RouteCenterlineCoverage | AreaContextCoverage | TypeAwareAnyCoverage | TypeAwareFallbackRate |
| --- | --- | --- | --- | --- | --- |
| train | 25939 | 96.3588% | 97.0469% | 97.7017% | 2.2983% |
| val | 5599 | 96.6333% | 96.2017% | 97.1543% | 2.8457% |

The >=95% any-context gate uses only the four authorized vehicle entity types.

## 【Parked Vehicle】

| Split | Actors | RouteCenterlineCoverage | AreaContextCoverage | TypeAwareAnyCoverage | TypeAwareFallbackRate |
| --- | --- | --- | --- | --- | --- |
| train | 136238 | 68.6573% | 77.5415% | 83.1901% | 16.8099% |
| val | 25198 | 61.0115% | 69.0015% | 75.6118% | 24.3882% |

The >=95% any-context gate uses only the four authorized vehicle entity types.

## 【Pedestrian Semantic Context】

Primary: walkway, ped_crossing or drivable_area. Lane/connector are secondary and cannot rescue pedestrian primary coverage. SemanticSpecific is walkway OR crossing; it has no independent 95% gate. Primary-only groups and generic-only all-six-entity groups have different definitions.

| Split | Candidates | WalkwayCoverage | CrosswalkCoverage | DrivableCoverage | SemanticSpecificCoverage | TypeAwareAnyCoverage | GenericOnlyRate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 439212 | 77.3278% | 24.7744% | 61.0794% | 82.0572% | 90.2009% | 0.4467% |
| val | 72012 | 73.8502% | 24.5265% | 63.5630% | 78.6438% | 90.9932% | 0.9193% |

| Split | WalkwayOnlyRate | CrosswalkOnlyRate | DrivableOnlyPrimaryRate | MultiplePrimaryRate | NoPrimaryContextRate |
| --- | --- | --- | --- | --- | --- |
| train | 28.9266% | 0.0134% | 8.1437% | 53.1172% | 9.7991% |
| val | 26.9927% | 0.0569% | 12.3493% | 51.5942% | 9.0068% |

## 【Bicycle Semantic Context】

Primary: lane, connector or drivable_area. Carpark coverage is descriptive only.

| Split | Candidates | LaneCoverage | ConnectorCoverage | DrivableCoverage | TypeAwareAnyCoverage | GenericOnlyRate |
| --- | --- | --- | --- | --- | --- | --- |
| train | 18828 | 87.7151% | 72.1001% | 63.7083% | 92.5271% | 0.0956% |
| val | 3936 | 86.7886% | 71.4685% | 70.5030% | 88.0843% | 0.0000% |

## 【Generic Drivable Dominance】

GenericOnly requires drivable_area and **no** lane, connector, carpark, crossing or walkway, including secondary context. The denominator is all modes in the stated population. Conditional-on-covered rates are also available in the coverage CSV. The preregistered full-horizon limits are Vehicle <=30%, Pedestrian <=50%, separately in each split. This narrow engineering classification does not demonstrate information gain, forecast accuracy or semantic usefulness.

| Split | ActorGroup | GenericOnlyRate | GenericOnlyAmongCoveredRate |
| --- | --- | --- | --- |
| train | Vehicle | 8.6935% | 9.8017% |
| train | Pedestrian | 0.4467% | 0.4952% |
| train | Bicycle | 0.0956% | 0.1033% |
| val | Vehicle | 4.8498% | 5.7290% |
| val | Pedestrian | 0.9193% | 1.0103% |
| val | Bicycle | 0.0000% | 0.0000% |

## 【Top8 Capacity】

All-entity overflow includes all six authorized types; relevant overflow restricts to actor-type primary context. Neither changes TopM=8. Selection is purely exact geometry distance, type ID, token; no GT, error or semantic priority. Uncapped counts and capped graph counts are reported separately.

| Split | Group | UncappedMeanRelevantEntitiesPerMode | MeanMapEntitiesPerMode | OverflowRateTop8 | RelevantOverflowRateTop8 |
| --- | --- | --- | --- | --- | --- |
| train | Overall | 5.044581760518469 | 3.988776278217304 | 30.9675% | 20.1592% |
| train | Vehicle | 6.113948864300919 | 4.698417116969598 | 29.9984% | 26.9464% |
| train | Pedestrian | 1.8719092374525286 | 1.8719092374525286 | 33.6259% | 0.0000% |
| train | Bicycle | 6.215423836838751 | 5.032982791586998 | 34.9639% | 28.1124% |
| val | Overall | 5.333375765290659 | 4.071816087773534 | 32.3807% | 22.5053% |
| val | Vehicle | 6.320624271630603 | 4.691435163312231 | 31.4852% | 28.9509% |
| val | Pedestrian | 1.8435955118591345 | 1.8435955118591345 | 35.9690% | 0.0000% |
| val | Bicycle | 5.473831300813008 | 4.854420731707317 | 24.5173% | 18.3181% |

The entity statistics CSV gives each type’s mean/median/p95/max and matched-candidate contribution. Stop-line counts are separated and never consume capacity.

| Split | EntityType | Coverage | MeanCount | MedianCount | P95Count | MaxCount |
| --- | --- | --- | --- | --- | --- | --- |
| train | lane | 78.3854% | 2.0518261888756744 | 2.0 | 5.0 | 22 |
| train | lane_connector | 59.3660% | 2.8489666821793613 | 2.0 | 10.0 | 29 |
| train | drivable_area | 78.5200% | 0.8845252253649792 | 1.0 | 2.0 | 8 |
| train | carpark_area | 21.3384% | 0.22782862494326375 | 0.0 | 1.0 | 5 |
| train | ped_crossing | 13.9870% | 0.17290794077597946 | 0.0 | 1.0 | 6 |
| train | walkway | 44.9071% | 0.4817852238711642 | 0.0 | 1.0 | 6 |
| val | lane | 76.2672% | 2.1146269018609445 | 2.0 | 6.0 | 20 |
| val | lane_connector | 60.0024% | 3.002521670606777 | 2.0 | 11.0 | 26 |
| val | drivable_area | 76.5391% | 0.8871673637631085 | 1.0 | 2.0 | 6 |
| val | carpark_area | 24.3184% | 0.25764078317269806 | 0.0 | 1.0 | 5 |
| val | ped_crossing | 13.2448% | 0.1686912772019155 | 0.0 | 1.0 | 5 |
| val | walkway | 38.4140% | 0.4224919682366491 | 0.0 | 1.0 | 4 |

## 【Graph Size】

This is a resource estimate, with no formal new G2/G3 implementation. Six modes each have <=8 relevant map entities, allowing genuinely empty selectors. Thus there are <=48 mode-map edges per target. Future map-node schema is only a proposal: six-type onehot, unchanged nine lane attributes (zero for non-lanes), relative geometry and polygon/centerline attributes. The byte estimate budgets 64 float32 values per map-node record and 64 per map-edge record without defining those as model features.

| Split | Targets | MeanModeMapEdges | P95ModeMapEdges | MaxModeMapEdges | EstimatedBytesPerTarget | Batch128DenseMaxInputBytes | EstimatedPeakCUDAMiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 290085 | 23.932657669303826 | 48.0 | 48 | 35037.18401503008 | 6077440 | 365.4521484375 |
| val | 54990 | 24.430896526641206 | 48.0 | 48 | 35294.77350427351 | 6077440 | 365.4521484375 |

Conservative inference peak budget: 365.452 MiB, derived from fourfold the historical measured non-input peak plus expanded dense inputs and 32 MiB margin. Device capacity: 9.654 GiB. This estimate is unmeasured for any future model; training activation/optimizer memory and future latency are unknown. No training feasibility or predictive performance claim follows.

## 【No-Future-Leakage】

ObservableWindow is an explicit allowlist of history, actor type, frozen candidate/logits/probabilities, identities and observed coordinate frames. Worker jobs carry only predicted candidate geometry, actor type and map location. GT/future/target access is guarded. In the same seed2022 100 TRAIN +100 VAL windows, GT becomes NaN and both masks are inverted. Entity IDs, distances, counts, coverage and relevant/all top8 selectors remain bitwise identical. Offline masks only determine reporting populations after retrieval. Candidate identity inherits the exact frozen predictor replay and revalidates every source cache SHA; predictions are aliased with unchanged data pointers, and all historical lane token IDs/distances are reproduced bitwise. No predictor forward, optimizer, backward, checkpoint, tiny run or test data was used.

## 【Coordinate Integrity】

PASS on 200 windows and 1196 polygon checks. Candidate roundtrip max 2.84217094304e-13 m; polygon roundtrip max 9.09494701773e-13 m; true geometry distance discrepancy max 9.09494701773e-13 m; boundary distance discrepancy max 9.09494701773e-13 m. All are below 1e-5 m. Both actual matched multipart polygons and random source polygons were transformed. Native shared-geometry threading failed during an early attempt; the final complete run used independent GEOS processes and discarded partial results.

## 【Scientific Decision】

| Split | Group | Metric | Value | Threshold | PASS |
| --- | --- | --- | --- | --- | --- |
| train | MovingVehicle | RouteCenterlineCoverage | 0.9869929123481853 | 0.95 | True |
| train | Vehicle | TypeAwareAnyCoverage | 0.886942384617184 | 0.95 | False |
| train | StoppedVehicle | TypeAwareAnyCoverage | 0.9770165902052251 | 0.95 | True |
| train | ParkedVehicle | TypeAwareAnyCoverage | 0.8319007912623497 | 0.95 | False |
| train | Pedestrian | TypeAwareAnyCoverage | 0.9020085972150124 | 0.95 | False |
| train | Bicycle | TypeAwareAnyCoverage | 0.9252708731676227 | 0.95 | False |
| train | Vehicle | GenericOnlyRate | 0.08693536690916746 | 0.3 | True |
| train | Pedestrian | GenericOnlyRate | 0.004467091063085708 | 0.5 | True |
| val | MovingVehicle | RouteCenterlineCoverage | 0.9900742440174617 | 0.95 | True |
| val | Vehicle | TypeAwareAnyCoverage | 0.8465227251252008 | 0.95 | False |
| val | StoppedVehicle | TypeAwareAnyCoverage | 0.9715425373578616 | 0.95 | True |
| val | ParkedVehicle | TypeAwareAnyCoverage | 0.7561182104399821 | 0.95 | False |
| val | Pedestrian | TypeAwareAnyCoverage | 0.9099316780536577 | 0.95 | False |
| val | Bicycle | TypeAwareAnyCoverage | 0.8808434959349594 | 0.95 | False |
| val | Vehicle | GenericOnlyRate | 0.04849759047529056 | 0.3 | True |
| val | Pedestrian | GenericOnlyRate | 0.009192912292395711 | 0.5 | True |

LaneOnlyFailureCause = **MIXED**

TypeAwareSemanticCoverage = **FAIL**

SemanticContextQuality = **STRONG**

Stage8A_0B = **NOT_READY**

The failure-cause label follows the preregistered descriptive evidence rule. Low-motion share evidence: True; residual authorized map-coverage limitation: True. Map limitation here includes geometry/threshold coverage gaps and does not assert that a map layer is missing. No causal training conclusion is inferred. No failed gate is rescued by pooling TRAIN/VAL or adding layers, radius or capacity. Historical Stage8A-0 remains NOT_READY. **STOP. No G1/G2/G3 training. Await scientific review.**
