# Stage7A Semantic Map Enhancement

【Problem】

Lane geometry describes road shape but does not explicitly distinguish connector turn classes, static traffic-control associations or crosswalk intersections. This experiment asks whether a single static lane residual improves the geometry of K=6 predicted candidates. The sole primary comparator is independently trained Stage3B TypeEmbedding. No reranker or future-interaction head is used.

【Frozen Semantic Definition】

The nine float32 lane-segment features are `is_connector, turn_left, turn_straight, turn_right, turn_unknown, traffic_light_controlled, stop_sign_controlled, other_control, crosswalk_intersects`. Ordinary lanes have connector=0 and all four turn slots=0. Connectors use the audited analytic arcline entrance/exit tangent taxonomy: abs heading change<=20° straight, >20° left, <-20° right, abs>=150° unknown. Controls are multi-hot: TRAFFIC_LIGHT, STOP_SIGN, and other(TURN_STOP/PED_CROSSING/YIELD). Control association is whole-centerline intersection with typed stop-line polygons. Crosswalk means whole lane/connector centerline intersects a ped-crossing polygon. No 2m/5m alternative, actor-level50m binary, dynamic signal state or topology repair is used. Earlier audit files retain their historical definitions; the formal vector conversion is independently registered.

【Architecture】

Canonical LocalEncoder → TypeEmbedding → GlobalInteractor → MLPDecoder is preserved. Actor–lane geometry and original all-zero categorical base remain unchanged. Linear(9,32)→ReLU→Linear(32,64) adds a semantic residual immediately before the original K/V projections. Attention, gates, FFN, decoder and losses are unchanged. Stage3B has 646001 parameters; Stage7A has 648433; addition 2432 (0.376470%). The old runtime is read-only.

【Neutral Initialization】

PASS. All shared parameters and buffers match fresh canonical Stage3B at seed2022; max parameter difference=0. The last semantic layer has zero weight and bias. On six real TRAIN graphs with deterministic CUDA aggregation and CuBLAS workspace configured for the audit, raw_prediction, mode_logits and mode_prob differences are exactly0. Unrestricted CUDA aggregation also causes approximately2e-6 differences between repeated forwards of Stage3B itself. Training retains the previous Stage3B kernel policy. Neither baseline nor any later-stage trained weights initialize Stage7A. Both semantic layers and original modules receive finite nonzero gradients within10 updates; the last semantic layer does so on update1.

【Data Integrity】

PASS. Random seed2022 sampled100 TRAIN and100 VAL windows; every original graph field is bitwise equal and the only added field is lane_semantic[L,9]. All850 old scene shards and 1704 frozen source/result/reference files retain SHA256. Five unrelated Stage2C redraw files remain unchanged and unsubmitted. Official split is TRAIN700/VAL150, with16898/3603 supervised windows; final VAL has54990 full and30037 partial targets (85027 total), NaN0/Inf0. Empty-supervision windows remain excluded under the existing definition. No trajectory, GT, mask, identity, map vector or graph edge is rebuilt.

【Training】

One fresh seed2022 run; Th5,Tf12,K6,batch16,embedding64,heads8,global layers3,temporal layers4,dropout0.1,local radius50m,AdamW weight_decay1e-4, natural class frequencies. Tiny is a fixed six-window TRAIN health check:200 fixed-scale updates, no architecture tuning, with semantic turn/control/crosswalk exposure and all three target classes plus moving vehicles. Formal warm-up is5000 fixed-scale updates atLR0.001; every500updates evaluates all150 VAL scenes. The model, optimizer and Torch/CUDA/Python/NumPy RNG restore from Stage7A's own warm-up best at source step3500. NLL begins at global counter5000 withLR0.0001 and reset scene-sampler cursor matching Stage3B. It executes6500 NLLupdates; final counter11500; best global step9000; stop reason `patience_5`. Selection uses strict overall full-horizon minFDE6 improvement, patience5 full VAL checks, hard cap21000. Fresh best reload reproduces selected metrics within1e-5, allowing existing CUDA floating-order variation.

Training source commit: `dd10a55ed6e1cc6a4b52a7b404390ecd6041bed4`. Stage7A checkpoint SHA256: `e30b7dbe7260ef5765e10f95d5eadf58e42f480e79f15b58a3b149a9d271abc1`. Stage3B comparison checkpoint SHA256: `461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547`. Final Stage3B reproduction differs from its archived metrics by at most3.93e-08.

【Main Results】

All following primary metric tables use full-horizon targets. ADE is evaluated on the best-FDE mode; MR uses endpoint error>2m; Top1 is the highest original model probability; NLL uses the original summed-L2 training winner and mean valid coordinate Laplace density. Partial-future results are also retained in the CSV tables. Lower is better.

The primary Stage7A−Stage3B minFDE6 difference is+0.007285m, paired95%CI[+0.000438,+0.013917]. Overall Top1FDE6 difference is+0.061498m, CI[+0.035941,+0.089320]. Both intervals are wholly above0, indicating small deterioration for this fixed-seed, validation-selected comparison. No preregistered difficult vehicle subgroup has a reliable minFDE improvement. This outcome does not support a general claim that static semantics are ineffective; it rejects the intended benefit for this particular feature definition and fusion experiment.

| model | group | count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage3B | overall | 54990 | 0.666406 | 1.343260 | 0.163102 | 1.196715 | 2.687984 | -0.705393 |
| Stage7A | overall | 54990 | 0.673179 | 1.350546 | 0.164830 | 1.235869 | 2.749482 | -0.575358 |
| Stage3B | vehicle | 42332 | 0.708347 | 1.431914 | 0.165761 | 1.343296 | 3.050323 | -0.876745 |
| Stage7A | vehicle | 42332 | 0.716937 | 1.440195 | 0.167651 | 1.397267 | 3.135868 | -0.717729 |
| Stage3B | pedestrian | 12002 | 0.525708 | 1.043875 | 0.154891 | 0.699869 | 1.457706 | -0.099274 |
| Stage7A | pedestrian | 12002 | 0.526554 | 1.048419 | 0.156307 | 0.689869 | 1.440422 | -0.073024 |
| Stage3B | bicycle | 656 | 0.534082 | 1.099908 | 0.141768 | 0.827986 | 1.814865 | -0.737320 |
| Stage7A | bicycle | 656 | 0.532102 | 1.093100 | 0.138720 | 0.810250 | 1.766034 | -0.578700 |
| Stage3B | Vehicle >5m | 9744 | 2.647145 | 5.579887 | 0.676416 | 5.177695 | 12.105361 | 1.718378 |
| Stage7A | Vehicle >5m | 9744 | 2.659870 | 5.586459 | 0.689245 | 5.392634 | 12.457772 | 1.886855 |
| Stage3B | Pedestrian <5m | 4421 | 0.296053 | 0.535423 | 0.067858 | 0.461395 | 0.967858 | -0.707604 |
| Stage7A | Pedestrian <5m | 4421 | 0.303088 | 0.558262 | 0.073513 | 0.502612 | 1.051456 | -0.643276 |
| Stage3B | Pedestrian >5m | 7581 | 0.659635 | 1.340389 | 0.205646 | 0.838939 | 1.743370 | 0.255486 |
| Stage7A | Pedestrian >5m | 7581 | 0.656872 | 1.334262 | 0.204590 | 0.799071 | 1.667255 | 0.259529 |

【Moving Vehicle】

Motion states are the frozen t0 nuScenes attribute labels, not GT-derived inference inputs.

| model | group | count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage3B | vehicle.moving | 10461 | 2.328189 | 4.791532 | 0.597075 | 4.656183 | 10.734859 | 1.336461 |
| Stage7A | vehicle.moving | 10461 | 2.339974 | 4.793599 | 0.605105 | 4.839169 | 11.025390 | 1.472677 |
| Stage3B | vehicle.stopped | 5599 | 0.350035 | 0.868888 | 0.087158 | 0.508323 | 1.313399 | -1.227037 |
| Stage7A | vehicle.stopped | 5599 | 0.360364 | 0.875347 | 0.085551 | 0.516806 | 1.328872 | -1.035525 |
| Stage3B | vehicle.parked | 25198 | 0.109823 | 0.146415 | 0.004088 | 0.155198 | 0.248490 | -1.716679 |
| Stage7A | vehicle.parked | 25198 | 0.116837 | 0.157426 | 0.004127 | 0.167045 | 0.267193 | -1.553256 |
| Stage3B | unknown | 1074 | 0.841131 | 1.803797 | 0.167598 | 1.302886 | 2.992323 | -0.901342 |
| Stage7A | unknown | 1074 | 0.846526 | 1.818046 | 0.171322 | 1.325665 | 3.014775 | -0.793026 |

【Intersection / Turning Vehicle】

IntersectionVehicle20 and NearTurnConnector20 use t0 distance<20m to complete regional connector centerlines, independent of cropped graph geometry. TurningVehicle_GT is strictly offline: full-future endpoint displacement>5m, first/last1s secant displacement>0.5m, abs wrapped heading change>20°. Its frozen VAL count is1663. Turn-context labels use the unique nearest left/right connector within20m; distances tied within1cm remain unassigned. Rules were frozen before training and never adjusted using VAL prediction errors.

TurningVehicle_GT minFDE6 changes by-0.010124m, CI[-0.037896,+0.017706], which includes0. The extreme illustrative improvement case does not establish an aggregate turning benefit.

| model | group | count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage3B | IntersectionVehicle20 | 32085 | 0.858295 | 1.760552 | 0.204021 | 1.639053 | 3.740200 | -0.606982 |
| Stage7A | IntersectionVehicle20 | 32085 | 0.867215 | 1.767967 | 0.205548 | 1.695560 | 3.828672 | -0.464954 |
| Stage3B | NearTurnConnector20 | 24949 | 0.939652 | 1.976263 | 0.218526 | 1.752410 | 4.010570 | -0.573041 |
| Stage7A | NearTurnConnector20 | 24949 | 0.949470 | 1.981576 | 0.219608 | 1.787907 | 4.068386 | -0.420170 |
| Stage3B | TurningVehicle_GT | 1663 | 5.890031 | 14.452465 | 0.971137 | 7.662410 | 18.571997 | 2.456115 |
| Stage7A | TurningVehicle_GT | 1663 | 5.861093 | 14.442341 | 0.971738 | 7.980289 | 19.098375 | 2.542984 |
| Stage3B | Left-context | 8830 | 1.088009 | 2.307863 | 0.255153 | 2.011074 | 4.598699 | -0.473253 |
| Stage7A | Left-context | 8830 | 1.104082 | 2.317802 | 0.259230 | 2.058233 | 4.663983 | -0.302265 |
| Stage3B | Right-context | 11873 | 0.957533 | 2.007125 | 0.218395 | 1.781969 | 4.101210 | -0.566956 |
| Stage7A | Right-context | 11873 | 0.965030 | 2.008652 | 0.217468 | 1.808223 | 4.153504 | -0.420941 |

Vehicle>5m and pedestrian displacement groups are present in the main table and are offline GT-defined analyses.

【Traffic-Control Context】

NearTrafficControl20 uses t0 distance<20m to a control-associated whole lane/connector centerline. Static association does not provide light color, stop compliance or intent.

| model | group | count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage3B | NearTrafficControl20 | 31778 | 0.851623 | 1.748630 | 0.200957 | 1.611249 | 3.686621 | -0.641969 |
| Stage7A | NearTrafficControl20 | 31778 | 0.860763 | 1.755137 | 0.203285 | 1.668290 | 3.774299 | -0.495240 |

【Semantic Zero Ablation】

One full VAL inference per setting on the identical Stage7A best checkpoint; no retraining. ZERO sets all nine inputs to0 but retains the trained MLP biases, so its residual is not necessarily0 and it is not an independently trained Stage3B baseline. Residual at zero input has norm1.884985.

ON overall minFDE6 is1.350546m and ZERO is1.358928m. This point comparison demonstrates that the trained predictions respond to semantic inputs, while ON still fails to outperform the independently trained Stage3B baseline. No additional ablation checkpoint or feature combination was trained.

| model | group | count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic-ON | overall | 54990 | 0.673179 | 1.350546 | 0.164830 | 1.235869 | 2.749482 | -0.575358 |
| Semantic-ZERO | overall | 54990 | 0.675373 | 1.358928 | 0.164430 | 1.218786 | 2.727582 | -0.574599 |
| Semantic-ON | vehicle | 42332 | 0.716937 | 1.440195 | 0.167651 | 1.397267 | 3.135868 | -0.717729 |
| Semantic-ZERO | vehicle | 42332 | 0.719782 | 1.450922 | 0.167131 | 1.375452 | 3.107103 | -0.717384 |
| Semantic-ON | pedestrian | 12002 | 0.526554 | 1.048419 | 0.156307 | 0.689869 | 1.440422 | -0.073024 |
| Semantic-ZERO | pedestrian | 12002 | 0.526644 | 1.049031 | 0.155891 | 0.688875 | 1.442354 | -0.071214 |
| Semantic-ON | vehicle.moving | 10461 | 2.339974 | 4.793599 | 0.605105 | 4.839169 | 11.025390 | 1.472677 |
| Semantic-ZERO | vehicle.moving | 10461 | 2.350959 | 4.833782 | 0.602524 | 4.750318 | 10.903239 | 1.484472 |
| Semantic-ON | TurningVehicle_GT | 1663 | 5.861093 | 14.442341 | 0.971738 | 7.980289 | 19.098375 | 2.542984 |
| Semantic-ZERO | TurningVehicle_GT | 1663 | 5.873834 | 14.457739 | 0.972940 | 7.638684 | 18.547957 | 2.565667 |

【Bootstrap】

1000 paired bootstrap draws over all150 official VAL scene clusters,seed2022. Entire scenes are resampled together; actor-window sums divided by counts determine each estimate. Every actor identity, original future tensor fingerprint, type and mask is paired. Delta=Stage7A−Stage3B. Intervals are percentile95%; overall minFDE6 is primary. Other intervals are exploratory and unadjusted; semantic groups overlap and are correlated.

| group | metric | count | Stage3B | Stage7A | delta | CI_lower | CI_upper |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | minFDE6 | 54990 | 1.343260 | 1.350546 | 0.007285 | 0.000438 | 0.013917 |
| vehicle | minFDE6 | 42332 | 1.431914 | 1.440195 | 0.008281 | -0.000285 | 0.016398 |
| pedestrian | minFDE6 | 12002 | 1.043875 | 1.048419 | 0.004543 | -0.000201 | 0.009119 |
| vehicle.moving | minFDE6 | 10461 | 4.791532 | 4.793599 | 0.002067 | -0.032150 | 0.033304 |
| Vehicle >5m | minFDE6 | 9744 | 5.579887 | 5.586459 | 0.006572 | -0.030823 | 0.043137 |
| IntersectionVehicle20 | minFDE6 | 32085 | 1.760552 | 1.767967 | 0.007415 | -0.003411 | 0.017320 |
| NearTurnConnector20 | minFDE6 | 24949 | 1.976263 | 1.981576 | 0.005313 | -0.006849 | 0.016437 |
| NearTrafficControl20 | minFDE6 | 31778 | 1.748630 | 1.755137 | 0.006507 | -0.004440 | 0.016483 |
| TurningVehicle_GT | minFDE6 | 1663 | 14.452465 | 14.442341 | -0.010124 | -0.037896 | 0.017706 |
| overall | Top1FDE6 | 54990 | 2.687984 | 2.749482 | 0.061498 | 0.035941 | 0.089320 |

【Semantic Residual Analysis】

The learned residual is nonzero. Its magnitude describes a representation, and does not establish physical causality or prove performance benefit. The feature-conditioned mean norm difference from zero input is1.819476, separating the actual input effect from the constant zero-input residual. Statistics weight actual stored VAL lane segment/window occurrences. Repeated appearances of a map token are therefore repeated exposures, not independent physical lane samples. Groups overlap for multi-hot control features.

| group | VAL_segment_occurrences | unique_semantic_patterns | mean_norm | median_norm | std_norm | min_norm | max_norm | mean_norm_difference_from_zero_input | weighting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ordinary_lane | 1212491 | 12 | 2.059737 | 1.884985 | 0.577102 | 0.854566 | 3.220684 | 1.040863 | stored VAL lane segment/window occurrences |
| connector | 1502003 | 38 | 2.504973 | 2.087795 | 0.935771 | 1.104988 | 4.668834 | 2.448011 | stored VAL lane segment/window occurrences |
| left | 331898 | 12 | 2.280432 | 2.419326 | 0.627118 | 1.371918 | 3.822793 | 1.902434 | stored VAL lane segment/window occurrences |
| straight | 859172 | 11 | 2.777359 | 2.087795 | 1.018255 | 1.602233 | 4.668834 | 2.896220 | stored VAL lane segment/window occurrences |
| right | 309481 | 11 | 1.991084 | 2.085141 | 0.649900 | 1.104988 | 3.608798 | 1.792492 | stored VAL lane segment/window occurrences |
| unknown_connector | 1452 | 4 | 2.186393 | 1.652580 | 0.756605 | 1.420474 | 3.572244 | 1.661997 | stored VAL lane segment/window occurrences |
| traffic_light | 804782 | 14 | 3.102355 | 2.739065 | 0.827488 | 1.857721 | 4.668834 | 3.013481 | stored VAL lane segment/window occurrences |
| stop_sign | 188002 | 18 | 2.620796 | 2.364260 | 0.691664 | 1.383717 | 3.822793 | 1.696779 | stored VAL lane segment/window occurrences |
| other_control | 1005063 | 25 | 2.299782 | 2.159922 | 1.081359 | 0.854566 | 4.466718 | 2.497376 | stored VAL lane segment/window occurrences |
| crosswalk | 928522 | 25 | 2.629271 | 2.605332 | 1.148800 | 1.104988 | 4.668834 | 2.951827 | stored VAL lane segment/window occurrences |
| non_crosswalk | 1785972 | 25 | 2.138082 | 2.087795 | 0.518026 | 0.854566 | 3.822793 | 1.230770 | stored VAL lane segment/window occurrences |
| all | 2714494 | 50 | 2.306098 | 2.087795 | 0.826008 | 0.854566 | 4.668834 | 1.819476 | stored VAL lane segment/window occurrences |

【Efficiency】

500 paired forwards, alternating method order, eight identical real VAL batches repeated, batch16, warm-up3pairs/batch, CUDA events with synchronization. Data loading is excluded. Both models remain resident; reported absolute allocated peaks include common co-resident weights and input; incremental forward peak is separately reported. This is one machine/session and excludes preprocessing or cache generation.

| model | parameters | parameter_increase_percent | mean_forward_ms | median_forward_ms | peak_CUDA_allocated_MiB | peak_forward_increment_MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Stage3B | 646001 | 0.000000 | 32.487412 | 27.744768 | 254.079102 | 231.224609 |
| Stage7A | 648433 | 0.376470 | 32.763890 | 28.029536 | 257.064453 | 234.209961 |

【Limitations】

Static map only; no dynamic traffic-light state, camera/VLM, intent, trajectory regeneration, balanced loss or interaction architecture. Controls are geometric associations and turn classes are derived taxonomy, with ambiguous near-U-turn connectors marked unknown. The absent raw trainval mount is not required because byte-frozen official scene shards and the already audited local map cache are used. One seed per independently trained method cannot quantify seed-to-seed variance; validation-selected checkpoints and secondary overlapping subgroup tests limit inference beyond this experiment. GT-turning/displacement groups are offline, never model inputs. ON/ZERO is an input perturbation and may be outside the training feature distribution. Residual norms do not establish causality. Cases are deliberately selected extreme improvements/degradation using the saved rule and are illustrations, not aggregate evidence. No test split, semantic variant retraining or Stage7B was run.

【Scientific Decision】

SemanticMap = NOT_SUPPORTED

PaperUsableSemantic = NO

ReadyStage7B = NO

The exact preregistered numeric interpretation is: marked reliable harm means>5% relative Vehicle/Pedestrian minFDE increase with CI lower>0; basically flat means overall point increase<=1%. SUPPORTED requires overall reliable improvement, no marked harm and at least one difficult-group point improvement. TARGETED_SUPPORTED requires overall point improvement/flat with CI including0, at least two difficult-group reliable improvements, and no marked harm. If paper usability isNO, readiness requires overall increase<=1%, overall CI not wholly positive, and reliable improvements in BOTH moving and GT-turning with no marked harm. Reliable difficult groups: `[]`. Marked reliable harms: `[]`. Rules were fixed before training.

Completed Stage7A only. STOP; Stage7B awaits separate review and authorization.
