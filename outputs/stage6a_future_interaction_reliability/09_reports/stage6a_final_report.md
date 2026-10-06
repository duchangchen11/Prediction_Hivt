# Stage6A: Future Interaction Reliability Re-ranking

【Problem】

The six candidate trajectories already contain better endpoint choices than the probability-selected Top1. This experiment changes only mode logits and probabilities, testing whether inference-computable future interaction features close part of that ranking gap. Stage3B TypeEmbedding remains the stable baseline. Stage5A remains technically PASS and scientifically NOT SUPPORTED under its earlier strict protocol; its geometry is authorized as a candidate backbone here. Those earlier conclusions are unchanged.

Moving-vehicle R0 minFDE is 4.765633 m versus Top1FDE 10.624263 m. minFDE is a GT-selected oracle metric. Top1FDE selects a mode from predicted probabilities and is the ranking metric used for inference.

【Frozen Predictor】

Base commit: `e5cec8f44d27608eddb7820a8f873b53d0b8ddab`. Branch: `stage6a/future-interaction-reliability`. All new files are isolated under `outputs/stage6a_future_interaction_reliability`.

Frozen checkpoint: `outputs/stage5a_motion_aware_decoder/07_checkpoints/stage5a_best_overall_minfde.pt`. SHA256: `88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`. Parameters: 650,403. The predictor is in eval mode, every parameter has requires_grad=False, and no backward or optimization reaches it. The encoders, TypeEmbedding, residual decoder and all six candidate trajectories remain frozen.

TRAIN700 and VAL150 predictions are cached in the original 16-window batch order. Positions use the existing ego_predictions conversion: actor-local predictions times inverse actor rotation, plus the actor current position. Distances between actors use one common t0 ego frame (x forward, y left), in meters. The cache audit replays 120 unique actor instances, 60 TRAIN and 60 VAL; raw predictions, logits, probabilities, rotations and ego predictions have maximum difference zero, below 1e-6.

Runtime determinism was fixed and registered before caching and head optimization: torch deterministic algorithms, deterministic cuDNN and CUBLAS_WORKSPACE_CONFIG=:4096:8. A pre-cache repeat check initially exceeded tolerance (raw 7.629e-6, ego 1.526e-5); no formal cache or head training proceeded under that runtime. The corrected real-batch repeat and neutral probabilities are exactly equal. This changes runtime reproducibility, not checkpoint weights or architecture.

【Reliability Features】

Each actor-mode input has 19 columns: type one-hot (3); log1p historical recent/net/path displacements (3); original logit and probability (2); predicted endpoint displacement, path length, mean step and maximum step from current position through the 12 future points (4); predicted future interaction features (7). History features reuse the existing padding-aware condition function and do not read future observations.

Neighbors must be valid at t0, exclude self, lie within 50 m and comprise at most the nearest eight. Ties retain node order. All neighbor futures and six-mode probabilities come from frozen Stage5A. For each target mode and neighbor mode, minimum and mean distance are computed over all 12 shared timestamps. Soft conflict is mean_t exp(−d²/(2·2²)). Expectation over the six frozen neighbor probabilities is taken before neighbor aggregation. There is no recursive reranking.

The seven aggregates are minimum expected minimum distance; mean expected minimum distance; minimum expected mean distance; maximum expected conflict; mean expected conflict; maximum vehicle-neighbor conflict; maximum pedestrian-neighbor conflict. Radius=50 m, M=8, sigma=2 m and label temperature=1 m are fixed without search.

Normalization uses only HeadTrain630 full-horizon actor-modes: population mean/std and (x−mean)/(std+1e-6) for continuous columns 3–18, including base probability. One-hot columns stay unchanged. No-neighbor raw distance sentinels are excluded from the three distance statistics. After normalization, absent-neighbor distances become one and conflict columns become zero; missing vehicle or pedestrian neighbor conflict becomes zero. R1 sets all seven normalized interaction columns exactly zero; R2 uses all seven. Their first twelve normalized inputs are identical.

Both heads are Linear(19,32) → ReLU → Linear(32,1), applied to each mode. z′=z+Δz and p′=softmax(z′). Both receive identical initial weights; last Linear weight and bias are zero. Initial shared-parameter difference is zero; step0 real-batch logits and probabilities exactly reproduce R0. Each head has 673 parameters.

【No Future Leakage】

The feature API takes only historical positions/padding, actor type, frozen predicted ego trajectories and frozen original logits/probabilities. It has no GT, future mask, target mask, map, future label or refined-neighbor probability argument. Nonneutral-head perturbation tests independently alter target GT, neighbor GT, future masks, target masks and future labels; features and probabilities remain bitwise equal. Positive controls changing neighbor predicted trajectories or base probabilities alter interaction features. Future GT is used only to construct training labels, compute offline metrics/bins and select illustrative cases.

【Head Training Split】

Sorted official TRAIN700 scene tokens are permuted by NumPy default_rng(2022). First70 are HeadDev; the other630 are HeadTrain. No scene overlap exists. HeadTrain has260,151 full-horizon targets (191,026 vehicle;66,145 pedestrian;2,980 bicycle); HeadDev has29,934 (22,719;7,057;158). Partial targets are excluded from normalization and ranking supervision, while current-valid context actors can contribute frozen future predictions.

HeadDev is held out from head optimization only: the previously trained frozen predictor already saw all TRAIN700 scenes. The frozen predictor checkpoint was also selected in the earlier Stage5A experiment on official VAL. Thus this stage uses official VAL only once for final ranking evaluation, but the overall experiment chain is not an untouched test evaluation.

Soft targets are q=softmax(−FDE_k/1 m), using full-horizon endpoint errors. The only loss is −sum q log p′. Both heads use seed2022, the same actor batches/permutations, batch1024, AdamW LR1e-3, weight_decay1e-4, maximum50 epochs. Each epoch evaluates HeadDev Top1FDE; strict improvement and patience5 select the best post-epoch checkpoint. Both stop after7 epochs with best epoch2. R1 best trained HeadDev FDE is slightly worse than the neutral R0; step0 was audited separately and was not an eligible trained checkpoint. No extra head, seed, feature selection or retraining follows.

| Variant | Parameters | Best epoch | Epochs executed | HeadDev Top1FDE | HeadDev step0 Top1FDE |
| --- | --- | --- | --- | --- | --- |
| R1 | 673 | 2 | 7 | 2.492106 | 2.491074 |
| R2 | 673 | 2 | 7 | 2.431732 | 2.491074 |

R1 checkpoint SHA256: `cea83b35fc98bba988e249351f92b44da3b5bcb8d5ed294a4b1429094006a405`.
R2 checkpoint SHA256: `e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314`.

Split SHA256: `55d8a856ce71c75b7af359e8dc66232907ae54b9a14e43c9a5f7f6148ce7d208`. Normalization SHA256: `be045dc0ad0d1198bcfa8f8c270ab77297984597ee6891516e5f84ccc4e9b2d2`. Configuration SHA256: `fe097f01f021e7cbf6503c877baa2ab01b2cf1b59f4a7a7325063cb250c0f31c`.

【Geometry Identity Audit】

PASS. Fresh frozen-predictor VAL outputs are bitwise identical to the registered deterministic cache in all3,603 windows. R0/R1/R2 share exactly the same raw and ego candidate tensors; the two head APIs return only ranking. Actor identity, future mask and GT fingerprints pair exactly with the frozen Stage3B/Stage5A ledger. minADE6/minFDE6/MR6 maximum difference across R0/R1/R2 is zero (<1e-8); predictor state hash is unchanged and no predictor gradient tensors exist.

Overall shared minADE6=0.665786015 m, minFDE6=1.337995889 m, MR6=0.163757047. minADE6 retains the prior ADE-of-best-FDE-mode convention; MR6 means best endpoint error>2 m.

Historical Stage5A CSV reconciliation is a separate check, because its earlier GPU run did not use the deterministic runtime. Maximum group-mean differences are {'minADE6': 8.154792840997516e-09, 'minFDE6': 4.3472249577902744e-08, 'MR6': 0.0, 'Top1ADE6': 2.6725662793936067e-08, 'Top1FDE6': 3.8017055947747735e-08}, all below1e-6. Historical old CSV tensors are not claimed bitwise identical. The exact invariant concerns frozen Stage5A predictions versus their current R1/R2 rerankings; all ranking comparisons use freshly reproduced R0.

【Main Ranking Results】

Official VAL:150 scenes,3,603 supervised windows,54,990 full-horizon +30,037 partial =85,027 supervised actor-windows. Primary tables use only the54,990 full targets, representing4,323 distinct actor instances. Complete identity pairing and finite checks pass (NaN=0, Inf=0). Values are actor-window means, not scene means. Probabilities tie by smallest mode index; best-mode probability ranks use stable descending order. The GT best mode is the first minimum-FDE index.

| Group | Count | Scenes | Instances | Stage3B_Top1FDE | R0_Top1FDE | R1_Top1FDE | R2_Top1FDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 54990 | 150 | 4323 | 2.687984 | 2.680611 | 2.690554 | 2.645563 |
| vehicle | 42332 | 150 | 3144 | 3.050323 | 3.053786 | 3.066375 | 3.010194 |
| pedestrian | 12002 | 121 | 1108 | 1.457706 | 1.424744 | 1.425687 | 1.418253 |
| bicycle | 656 | 47 | 71 | 1.814865 | 1.576419 | 1.580310 | 1.570246 |
| vehicle.moving | 10461 | 137 | 850 | 10.734859 | 10.624263 | 10.671123 | 10.450046 |
| vehicle.stopped | 5599 | 103 | 433 | 1.313399 | 1.338586 | 1.337248 | 1.332262 |
| vehicle.parked | 25198 | 131 | 1892 | 0.248490 | 0.274905 | 0.275516 | 0.276514 |
| unknown | 1074 | 61 | 93 | 2.992323 | 3.454987 | 3.487380 | 3.428921 |
| Vehicle >5m | 9744 | 137 | 853 | 12.105361 | 12.071151 | 12.132410 | 11.882757 |
| Pedestrian <5m | 4421 | 100 | 493 | 0.967858 | 1.000297 | 1.003833 | 0.968377 |
| Pedestrian >5m | 7581 | 110 | 716 | 1.743370 | 1.672268 | 1.671700 | 1.680606 |
| Pedestrian 5-10m | 7321 | 110 | 701 | 1.697247 | 1.629249 | 1.628975 | 1.638200 |
| Heterogeneous-20m | 25113 | 131 | 2399 | 2.603381 | 2.554706 | 2.563890 | 2.530599 |
| Vehicle hetero-20m | 14813 | 131 | 1365 | 3.402311 | 3.350984 | 3.365668 | 3.315746 |
| Pedestrian hetero-20m | 9732 | 120 | 968 | 1.420733 | 1.388372 | 1.389479 | 1.380711 |
| VP-context-20m | 23209 | 125 | 2213 | 2.583959 | 2.544853 | 2.555403 | 2.526395 |

| Variant | Top1ADE | Top1FDE | OracleGap_FDE | Top1HitRate | BestModeRank | MRR_best_mode |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 1.204789 | 2.680611 | 1.342615 | 0.263266 | 2.397763 | 0.551004 |
| R1 | 1.211102 | 2.690554 | 1.352558 | 0.263593 | 2.399818 | 0.550737 |
| R2 | 1.189506 | 2.645563 | 1.307567 | 0.265557 | 2.384306 | 0.553207 |

Source tables: [main ranking results](../06_tables/stage6a_main_ranking_results.csv), [reliability ablation](../06_tables/stage6a_reliability_ablation.csv). The comparison chain is B=Stage3B, E=Stage5A/R0, E+R1 and E+R2. Differences between B and R2 in this table are descriptive; the paired bootstrap comparisons registered for this stage are R1−R0, R2−R0 and R2−R1.

【Oracle Gap】

Overall mean oracle gap is R0=1.342615, R1=1.352558, R2=1.307567 m. R2 closes only part of the gap; it generates no new geometry. The gap delta equals the Top1FDE delta up to float32 subtraction rounding, and is not an independent geometric improvement.

【Hit Rate】

Overall best-FDE-mode hit rate is R0=26.326605%, R1=26.359338%, R2=26.555737%. R2 improves the point estimate, but the paired HitRate CI spans zero; this is not a statistically resolved hit-rate gain. Mean best-mode rank and MRR are reported above. Moving vehicle hit rates are 22.158493%, 22.426154% and 22.971035%.

【R1 vs R2】

R1 and R2 have identical head architecture, parameter budget, initialization, optimization and checkpoint-selection protocol. Their only input difference is the seven future interaction columns. R1 slightly worsens Overall Top1FDE, with a paired CI above zero. R2 improves Overall Top1FDE relative to both R0 and R1. Under this fixed protocol, the improvement is supported by the interaction-enabled head rather than a generic extra MLP alone. This does not isolate individual interaction features or establish causality for observed physical collisions.

| Comparison | Count | Number_changed | Number_improved | Number_worsened | Top1_changed_rate | Improved_rate_among_changed | Worsened_rate_among_changed | Mean_improvement_when_improved_m | Mean_degradation_when_worsened_m | Net_Top1FDE_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1-R0 | 54990 | 1357 | 614 | 743 | 0.024677 | 0.452469 | 0.547531 | 1.684718 | 2.128104 | 0.009943 |
| R2-R0 | 54990 | 5911 | 3039 | 2872 | 0.107492 | 0.514126 | 0.485874 | 2.844499 | 2.338832 | -0.035048 |
| R2-R1 | 54990 | 5548 | 2921 | 2627 | 0.100891 | 0.526496 | 0.473504 | 3.024066 | 2.420718 | -0.044991 |

Changed rates use all full actor-windows as denominator. Improved/worsened conditional rates use only changed Top1s. Improved means strictly smaller endpoint error, worsened strictly larger. The full table also includes rates over all actors and equal changes. Net delta=(worsened count×mean harm−improved count×mean gain)/total count; an independent reconstruction matches within1e-12 m. Improvements and degradations coexist.

【Interaction-sensitive Groups】

The original frozen20 m membership CSV and its SHA are reused without recomputation. These subgroup definitions are distinct from the fixed50 m/nearest8 inference features. Heterogeneous20 m includes25,113 full targets; vehicle heterogeneous14,813; pedestrian heterogeneous9,732; VP context23,209. R2−R1 is the controlled interaction contrast; R2−R0 subgroup intervals may cross zero even where the controlled R2−R1 interval is below zero.

| Group | Count | Scenes | Instances | Stage3B_Top1FDE | R0_Top1FDE | R1_Top1FDE | R2_Top1FDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Heterogeneous-20m | 25113 | 131 | 2399 | 2.603381 | 2.554706 | 2.563890 | 2.530599 |
| Vehicle hetero-20m | 14813 | 131 | 1365 | 3.402311 | 3.350984 | 3.365668 | 3.315746 |
| Pedestrian hetero-20m | 9732 | 120 | 968 | 1.420733 | 1.388372 | 1.389479 | 1.380711 |
| VP-context-20m | 23209 | 125 | 2213 | 2.583959 | 2.544853 | 2.555403 | 2.526395 |

Interaction feature distributions are recorded by candidate mode, actor type and frozen context group in [the distribution table](../06_tables/stage6a_interaction_feature_distributions.csv). No-neighbor raw distance sentinels are excluded from distance distributions; conflict zeros remain included. First-layer column norms and one fixed actor-wise permutation per interaction feature use only HeadDev70 after training. The same actor permutation moves all six mode values together. These are descriptive diagnostics of correlated features; no features are removed and no new model is fit.

| Feature | First_layer_weight_L2_norm | HeadDev_permutation_Top1FDE_delta | HeadDev_permutation_rank_loss_delta | HeadDev_top1_changed_rate |
| --- | --- | --- | --- | --- |
| minimum_expected_min_distance | 1.521655 | 0.044520 | 0.006313 | 0.145754 |
| mean_expected_min_distance | 0.750495 | 0.003176 | 0.000526 | 0.026458 |
| minimum_expected_mean_distance | 0.910209 | 0.013882 | 0.001924 | 0.115888 |
| maximum_expected_conflict | 0.783146 | 0.012978 | 0.000307 | 0.055455 |
| mean_expected_conflict | 0.892128 | 0.010873 | 0.000057 | 0.027594 |
| maximum_vehicle_conflict | 1.191508 | 0.055413 | 0.002328 | 0.074030 |
| maximum_pedestrian_conflict | 0.788164 | 0.000322 | 0.000164 | 0.065611 |

【Bootstrap】

Paired whole-scene percentile bootstrap resamples the same150 official VAL scene clusters with replacement,1,000 replicates, seed2022, pooling actor-window sums/counts in each sampled set. It uses the same resampled weights for all contrasts, groups and metrics. Negative FDE/ADE/gap deltas favor the new model; positive HitRate deltas favor it. HitRate values in the following table are proportions, not percent. Full results cover16 groups×3 contrasts×4 metrics.

| Comparison | Group | Metric | Delta | CI95_lower | CI95_upper | Count | Scenes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1-R0 | overall | Top1FDE | 0.009943 | 0.002108 | 0.018810 | 54990 | 150 |
| R1-R0 | overall | Top1ADE | 0.006313 | 0.002683 | 0.010466 | 54990 | 150 |
| R1-R0 | overall | Top1HitRate | 0.000327 | -0.000589 | 0.001407 | 54990 | 150 |
| R1-R0 | overall | OracleGap_FDE | 0.009943 | 0.002108 | 0.018810 | 54990 | 150 |
| R1-R0 | vehicle | Top1FDE | 0.012588 | 0.002410 | 0.023878 | 42332 | 150 |
| R1-R0 | pedestrian | Top1FDE | 0.000943 | -0.002068 | 0.003462 | 12002 | 121 |
| R1-R0 | bicycle | Top1FDE | 0.003891 | -0.004941 | 0.019556 | 656 | 47 |
| R1-R0 | vehicle.moving | Top1FDE | 0.046860 | 0.005726 | 0.090999 | 10461 | 137 |
| R1-R0 | vehicle.stopped | Top1FDE | -0.001338 | -0.005066 | 0.002372 | 5599 | 103 |
| R1-R0 | vehicle.parked | Top1FDE | 0.000611 | -0.000819 | 0.001955 | 25198 | 131 |
| R1-R0 | unknown | Top1FDE | 0.032393 | -0.025560 | 0.106590 | 1074 | 61 |
| R1-R0 | Vehicle >5m | Top1FDE | 0.061260 | 0.020617 | 0.106385 | 9744 | 137 |
| R1-R0 | Pedestrian <5m | Top1FDE | 0.003536 | -0.000547 | 0.008060 | 4421 | 100 |
| R1-R0 | Pedestrian >5m | Top1FDE | -0.000569 | -0.003482 | 0.001857 | 7581 | 110 |
| R1-R0 | Pedestrian 5-10m | Top1FDE | -0.000275 | -0.003431 | 0.002670 | 7321 | 110 |
| R1-R0 | Heterogeneous-20m | Top1FDE | 0.009184 | -0.000772 | 0.020607 | 25113 | 131 |
| R1-R0 | Vehicle hetero-20m | Top1FDE | 0.014684 | -0.002005 | 0.033342 | 14813 | 131 |
| R1-R0 | Pedestrian hetero-20m | Top1FDE | 0.001107 | -0.002471 | 0.004071 | 9732 | 120 |
| R1-R0 | VP-context-20m | Top1FDE | 0.010550 | 0.000606 | 0.021758 | 23209 | 125 |
| R2-R0 | overall | Top1FDE | -0.035048 | -0.059788 | -0.009494 | 54990 | 150 |
| R2-R0 | overall | Top1ADE | -0.015283 | -0.026361 | -0.004001 | 54990 | 150 |
| R2-R0 | overall | Top1HitRate | 0.002291 | -0.000392 | 0.004912 | 54990 | 150 |
| R2-R0 | overall | OracleGap_FDE | -0.035048 | -0.059788 | -0.009494 | 54990 | 150 |
| R2-R0 | vehicle | Top1FDE | -0.043592 | -0.076216 | -0.009413 | 42332 | 150 |
| R2-R0 | pedestrian | Top1FDE | -0.006491 | -0.013061 | -0.000451 | 12002 | 121 |
| R2-R0 | bicycle | Top1FDE | -0.006173 | -0.031006 | 0.023787 | 656 | 47 |
| R2-R0 | vehicle.moving | Top1FDE | -0.174216 | -0.298948 | -0.038912 | 10461 | 137 |
| R2-R0 | vehicle.stopped | Top1FDE | -0.006323 | -0.019724 | 0.005923 | 5599 | 103 |
| R2-R0 | vehicle.parked | Top1FDE | 0.001609 | -0.002344 | 0.005472 | 25198 | 131 |
| R2-R0 | unknown | Top1FDE | -0.026066 | -0.244813 | 0.203037 | 1074 | 61 |
| R2-R0 | Vehicle >5m | Top1FDE | -0.188394 | -0.324710 | -0.045107 | 9744 | 137 |
| R2-R0 | Pedestrian <5m | Top1FDE | -0.031920 | -0.051817 | -0.013423 | 4421 | 100 |
| R2-R0 | Pedestrian >5m | Top1FDE | 0.008338 | -0.001715 | 0.019892 | 7581 | 110 |
| R2-R0 | Pedestrian 5-10m | Top1FDE | 0.008951 | -0.001123 | 0.020603 | 7321 | 110 |
| R2-R0 | Heterogeneous-20m | Top1FDE | -0.024108 | -0.048256 | 0.000149 | 25113 | 131 |
| R2-R0 | Vehicle hetero-20m | Top1FDE | -0.035237 | -0.075614 | 0.005529 | 14813 | 131 |
| R2-R0 | Pedestrian hetero-20m | Top1FDE | -0.007662 | -0.014879 | -0.000916 | 9732 | 120 |
| R2-R0 | VP-context-20m | Top1FDE | -0.018458 | -0.040917 | 0.003524 | 23209 | 125 |
| R2-R1 | overall | Top1FDE | -0.044991 | -0.069113 | -0.021106 | 54990 | 150 |
| R2-R1 | overall | Top1ADE | -0.021596 | -0.032789 | -0.010756 | 54990 | 150 |
| R2-R1 | overall | Top1HitRate | 0.001964 | -0.000510 | 0.004344 | 54990 | 150 |
| R2-R1 | overall | OracleGap_FDE | -0.044991 | -0.069113 | -0.021106 | 54990 | 150 |
| R2-R1 | vehicle | Top1FDE | -0.056181 | -0.086901 | -0.024857 | 42332 | 150 |
| R2-R1 | pedestrian | Top1FDE | -0.007435 | -0.013597 | -0.001415 | 12002 | 121 |
| R2-R1 | bicycle | Top1FDE | -0.010064 | -0.036768 | 0.021225 | 656 | 47 |
| R2-R1 | vehicle.moving | Top1FDE | -0.221076 | -0.339190 | -0.104091 | 10461 | 137 |
| R2-R1 | vehicle.stopped | Top1FDE | -0.004986 | -0.018515 | 0.007536 | 5599 | 103 |
| R2-R1 | vehicle.parked | Top1FDE | 0.000998 | -0.002795 | 0.004734 | 25198 | 131 |
| R2-R1 | unknown | Top1FDE | -0.058459 | -0.249361 | 0.154368 | 1074 | 61 |
| R2-R1 | Vehicle >5m | Top1FDE | -0.249654 | -0.380422 | -0.116133 | 9744 | 137 |
| R2-R1 | Pedestrian <5m | Top1FDE | -0.035456 | -0.054569 | -0.018352 | 4421 | 100 |
| R2-R1 | Pedestrian >5m | Top1FDE | 0.008906 | 0.000040 | 0.018655 | 7581 | 110 |
| R2-R1 | Pedestrian 5-10m | Top1FDE | 0.009226 | 0.000411 | 0.018660 | 7321 | 110 |
| R2-R1 | Heterogeneous-20m | Top1FDE | -0.033291 | -0.055550 | -0.012426 | 25113 | 131 |
| R2-R1 | Vehicle hetero-20m | Top1FDE | -0.049922 | -0.087138 | -0.014324 | 14813 | 131 |
| R2-R1 | Pedestrian hetero-20m | Top1FDE | -0.008768 | -0.015337 | -0.002626 | 9732 | 120 |
| R2-R1 | VP-context-20m | Top1FDE | -0.029008 | -0.052481 | -0.008159 | 23209 | 125 |

Intervals condition on these fixed HeadDev-selected heads and the previously selected frozen predictor. One training seed is used; intervals do not capture training-seed variability or correct the descriptive final choice among fixed variants. No multiple-comparison correction is applied; secondary subgroup intervals are descriptive. Overlapping windows and nested subgroups are not independent actor observations. Bicycle support is limited to656 full actor-windows,71 instances and47 scenes. No SOTA or broad deployment claim follows from this experiment.

【Case Studies】

Four deterministic purposive examples cover a moving vehicle with improved Top1, a pedestrian recovering its best-FDE mode, a frozen heterogeneous-context actor with R2 better than R1, and a failure where R2 still selects the wrong mode and increases endpoint error. Selection maximizes the specified gain or failure harm with stable identity tie-breaks and distinct instances. GT is used only for this offline selection. No new predictor forward is performed: original prediction and final ranking caches provide all curves/probabilities.

| name | case_kind | agent_type | R0_Top1FDE | R1_Top1FDE | R2_Top1FDE |
| --- | --- | --- | --- | --- | --- |
| stage6a_case_moving_vehicle | Moving vehicle: better Top1 | vehicle | 33.452370 | 33.452370 | 3.654901 |
| stage6a_case_pedestrian | Pedestrian: best mode recovered | pedestrian | 6.347923 | 6.347923 | 0.872210 |
| stage6a_case_heterogeneous | Heterogeneous context: R2 better than R1 | vehicle | 35.230034 | 35.230034 | 6.920377 |
| stage6a_case_failure | Failure: wrong Top1 after reranking | vehicle | 4.788861 | 4.788861 | 36.743221 |

Each case displays the same actor, GT, map, all six original candidates and matched axis limits in three panels, with the R0/R1/R2 selected mode and all six probabilities. Original12 future points are unmodified; no smoothing or interpolation. Float64 endpoint recomputation agrees with the source actor CSV within1e-4 m. Gray candidates can overlap because the frozen geometry is genuinely similar; they are not shifted for display. Cases illustrate possibilities and cannot estimate population benefit.

【Efficiency】

Predictor=650,403 parameters; each complete variant adds one673-parameter head, giving651,076 and0.103474% increase. The two experiment heads are separate ablations, not combined at inference. R0 has no added feature/head stage; its overhead entries below are unmeasured reference dashes.

| Variant | Predictor_params | Head_params | Total_params | Parameter_increase_percent | Mean_feature_extraction_ms | Mean_head_scoring_ms | Mean_ranking_overhead_ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 650403 | 0 | 650403 | 0 | — | — | — |
| R1 | 650403 | 673 | 651076 | 0.103474 | 25.951224 | 4.278055 | 30.229279 |
| R2 | 650403 | 673 | 651076 | 0.103474 | 42.839637 | 4.289775 | 47.129412 |

Timing uses20 preloaded original16-window VAL batches,20 paired warm-up rounds and200 paired measurements, alternating R1/R2 execution order with CUDA events on the existing RTX3080 / torch2.5.1+cu124. Only extra feature extraction, normalization and head scoring are timed; predictor forward, disk I/O and H2D are excluded. R1 skips neighbor distance computation. Feature extraction includes Python loops and validation checks in this implementation; these batch latencies are not deployment-optimized throughput. Whole predictor memory/latency are not remeasured.

【Scientific Interpretation】

ReliabilityHead = **SUPPORTED**. FutureInteractionContribution = **SUPPORTED**. PaperUsableReliability = **YES**. RecommendedFinalVariant = **R2**.

The numerical interpretation rules were operationalized and saved before fitting: ReliabilityHead SUPPORTED requires at least one fixed head with Overall FDE CI upper<0, smaller gap, higher point-estimate HitRate and no severe main-class harm. PROMISING requires favorable point estimates with CI crossing zero. Future interaction SUPPORTED requires R2−R1 Overall FDE CI upper<0 plus improvement in at least one frozen context group and no severe class harm. Severe Vehicle/Pedestrian harm means a≥10% relative FDE increase together with a paired CI lower>0. This effect-size guard is an explicitly disclosed implementation of the qualitative requirement; it was not chosen after seeing results.

R2 satisfies these rules and is the lowest-Overall-FDE safe variant. Its improvement is modest and concentrated in ranking; R1 is NOT SUPPORTED. Neither main class exhibits severe collapse. HitRate improvement is unresolved by the paired CI; interaction subgroup and case findings support limited interpretation. Final R2 recommendation is descriptive selection among prespecified fixed heads on final VAL, not a new checkpoint/feature/threshold search. Stage5A prior NOT SUPPORTED and all prior conclusions remain unchanged.

All new code, configuration, small tables, audits, source-case JSON, reports and eight PNG300dpi/PDF/SVG bundles are versioned in this stage root. Prediction caches, tensor features, raw actor records, logs and head weights remain local; their sizes/counts/schemas/hashes are recorded. Old data/shards and earlier stage files are read-only; the five Stage2C redraw files remain untouched and unsubmitted. The artifact inventory excludes itself, transient Git payloads and the active finalizer log, whose contents are still being appended.

STOP. No joint fine-tuning, decoder changes, third reranker, extra seed, altered radius/M/sigma/temperature or next experiment is executed.
