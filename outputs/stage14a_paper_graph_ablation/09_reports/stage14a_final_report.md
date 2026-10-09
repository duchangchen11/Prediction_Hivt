# Stage14A controlled graph ablation execution report



**Execution complete. Awaiting brain-AI review.**

Registered qualification: GraphIncrementSupported=SUPPORTED; LossIncrementSupported=SUPPORTED.

These labels apply the frozen statistical rules. Paper writing and broader scientific judgment remain with brain-AI.



## Evaluation scope



nuScenes HeadTrain630 internal three-fold ranking OOF; 260151 full-horizon actor/window targets. Each scene belongs to exactly one 210-scene OuterTest fold. Corresponding ranking-head training uses 378 InnerTrain and 42 InnerDev scenes. The frozen Stage5A predictor trained on all 700 official TRAIN scenes, including 630/630 of these ranking scenes and 210/210 of every OuterTest fold. This is not independent end-to-end validation. Historical VAL150 was used for predictor selection and method development. No official VAL/test evaluation was executed in Stage14A.



## Frozen controls and integrity



Base commit: `88b83b7114e72b91544d4566cab26700c1b452dc`. Branch: `stage14a/paper-graph-ablation`. Protocol SHA256: `ae26d06dd83ab9d13773a125676a8bb59d887504fe0d674580f6304ca826e151`.

NoGraph registers only original G1 node_encoder, LayerNorm and scoring head, with the same 64-dimensional embedding and original-logit residual. Interaction message is exactly zero. Full cached node/edge/neighbor inputs are retained unchanged; they are not corrupted or shuffled. NoGraph has 7425 total/trainable parameters; G1 has 24066. The extra G1 interaction parameters and compute are part of the structural contrast, so this is not a parameter-count-matched or causal test.

The original Stage11B split, InnerTrain-fitted normalization, shared initialization, optimizer, learning rate, weight decay, FP32, 128×8 microbatch accumulation, continuous 1024-target carry ordering, maximum 50 epochs and patience 5 are reused. Historical batch-order prefixes and shared initial weights match exactly. Checkpoint selection uses only the original fixed Vehicle/Pedestrian relative InnerDev score. The protocol retains original Stage11B R2 configuration metadata; Stage14A actually reuses frozen fold R2 and does not train R2.

Loss A is original SoftCE; loss C is original normalized expected regret with unchanged 1 m floor. Exact original AST definitions are reused. Future labels remain detached and outside the seven forward inputs. GT poison, zero-message equivalence, shared initialization, frozen predictor gradient/state and candidate coordinate checks passed.

All four models use identical six coordinate arrays. G-A/G-C scores and all 15 actor metrics reproduce original binary arrays bitwise. Published CSV summaries reproduce their original 12-significant-digit serialization precision. Bicycle uses the same predeclared corresponding fold R2 logits/probabilities/choices in all four models, bitwise. Its result is a routing policy, not evidence that Graph improves Bicycle.

Global six-checkpoint freeze file mtime (UTC): `2026-10-09T17:43:51.121190+00:00`. This is the filesystem modification time of the all-frozen manifest, reported with the gate/hash evidence; it is not independent timestamp notarization. Unified OuterTest evaluation is guarded until all 6 checkpoints exist and their SHA256 is verified. No OuterTest-based tuning or reselection occurred.



## Tiny preflight and six formal runs



| Model | Status | Updates | InitialLoss | FinalLoss | Reduction | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| NG-A | PASS | 300 | 1.576090 | 1.212148 | 0.980459 | 0.900000 |
| NG-C | PASS | 300 | 0.412033 | 0.054473 | 0.867795 | 0.800000 |



A reduction is relative to the original entropy floor; C reduction is relative to initial normalized objective. Tiny weights are discarded before formal runs.

| Fold | PaperModel | SelectedEpoch | ExecutedEpochs | CheckpointScore | Seconds | Params |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | NG-A | 1 | 6 | 1.019054 | 177.746379 | 7425 |
| 1 | NG-C | 14 | 19 | 0.956676 | 544.967688 | 7425 |
| 2 | NG-A | 1 | 6 | 1.028842 | 166.237916 | 7425 |
| 2 | NG-C | 15 | 20 | 0.971508 | 568.154546 | 7425 |
| 3 | NG-A | 1 | 6 | 1.013928 | 162.845235 | 7425 |
| 3 | NG-C | 5 | 10 | 0.970271 | 269.887609 | 7425 |

Total six-run measured training wall time: 1889.839 s (0.5250 h), including InnerDev/I/O/checkpoint work; not pure CUDA kernel time.

A pre-optimizer isolation block occurred when a source-hash scan touched evaluation code. It produced zero optimizer updates and zero checkpoints; the trace is preserved in 03_training/stage14a_pre_optimizer_guard_block.txt. The source-hash scope was restricted to training sources without changing model/loss/hyperparameters or weakening the guard. A preliminary label-gradient audit also required enabling gradients within its detached-label test; forward checks remained in eval/no_grad. Neither engineering correction was an experimental search.



## Four-model metrics



FDE/ADE/OracleGap/minFDE6 are meters. Count is full 12-step future actor/window targets, exactly the historical Stage11B denominator/mask. HitRate is the fraction whose top1 mode matches the lowest-index endpoint-FDE oracle. MR6 is the fraction with oracle endpoint FDE>2 m. These are sorting results over the same frozen candidates.

| Group | Model | Count | Top1FDE | Top1ADE | OracleGap | HitRate | minFDE6 | MR6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | NG-A | 260151 | 2.380943 | 1.073364 | 1.122808 | 0.385914 | 1.258135 | 0.143970 |
| Overall | NG-C | 260151 | 2.254734 | 0.994880 | 0.996599 | 0.419872 | 1.258135 | 0.143970 |
| Overall | G-A | 260151 | 2.384061 | 1.089466 | 1.125926 | 0.385069 | 1.258135 | 0.143970 |
| Overall | G-C | 260151 | 2.199512 | 0.973628 | 0.941377 | 0.420121 | 1.258135 | 0.143970 |
| Vehicle | NG-A | 191026 | 2.748780 | 1.225630 | 1.367186 | 0.416299 | 1.381594 | 0.148880 |
| Vehicle | NG-C | 191026 | 2.587120 | 1.123940 | 1.205526 | 0.455598 | 1.381594 | 0.148880 |
| Vehicle | G-A | 191026 | 2.751081 | 1.244684 | 1.369488 | 0.410766 | 1.381594 | 0.148880 |
| Vehicle | G-C | 191026 | 2.516273 | 1.096568 | 1.134680 | 0.451991 | 1.381594 | 0.148880 |
| Pedestrian | NG-A | 66145 | 1.347066 | 0.645696 | 0.439118 | 0.296545 | 0.907948 | 0.129760 |
| Pedestrian | NG-C | 66145 | 1.317552 | 0.630692 | 0.409604 | 0.316607 | 0.907948 | 0.129760 |
| Pedestrian | G-A | 66145 | 1.352684 | 0.653994 | 0.444735 | 0.309199 | 0.907948 | 0.129760 |
| Pedestrian | G-C | 66145 | 1.304968 | 0.626157 | 0.397020 | 0.328007 | 0.907948 | 0.129760 |
| Bicycle | NG-A | 2980 | 1.749867 | 0.805404 | 0.632905 | 0.421812 | 1.116962 | 0.144631 |
| Bicycle | NG-C | 2980 | 1.749867 | 0.805404 | 0.632905 | 0.421812 | 1.116962 | 0.144631 |
| Bicycle | G-A | 2980 | 1.749867 | 0.805404 | 0.632905 | 0.421812 | 1.116962 | 0.144631 |
| Bicycle | G-C | 2980 | 1.749867 | 0.805404 | 0.632905 | 0.421812 | 1.116962 | 0.144631 |
| MovingVehicle | NG-A | 41728 | 10.899423 | 4.781322 | 5.572259 | 0.220859 | 5.327164 | 0.615294 |
| MovingVehicle | NG-C | 41728 | 10.208203 | 4.344736 | 4.881039 | 0.207175 | 5.327164 | 0.615294 |
| MovingVehicle | G-A | 41728 | 10.905267 | 4.860093 | 5.578103 | 0.250216 | 5.327164 | 0.615294 |
| MovingVehicle | G-C | 41728 | 9.906479 | 4.227418 | 4.579315 | 0.224238 | 5.327164 | 0.615294 |
| StoppedVehicle | NG-A | 23044 | 1.255360 | 0.488668 | 0.443941 | 0.458601 | 0.811419 | 0.079327 |
| StoppedVehicle | NG-C | 23044 | 1.252048 | 0.484244 | 0.440629 | 0.476740 | 0.811419 | 0.079327 |
| StoppedVehicle | G-A | 23044 | 1.238519 | 0.484530 | 0.427099 | 0.406613 | 0.811419 | 0.079327 |
| StoppedVehicle | G-C | 23044 | 1.236939 | 0.478716 | 0.425519 | 0.466455 | 0.811419 | 0.079327 |
| ParkedVehicle | NG-A | 121939 | 0.228774 | 0.143606 | 0.104190 | 0.474524 | 0.124584 | 0.001780 |
| ParkedVehicle | NG-C | 121939 | 0.214693 | 0.136337 | 0.090109 | 0.536654 | 0.124584 | 0.001780 |
| ParkedVehicle | G-A | 121939 | 0.235169 | 0.146872 | 0.110584 | 0.466446 | 0.124584 | 0.001780 |
| ParkedVehicle | G-C | 121939 | 0.216743 | 0.137365 | 0.092159 | 0.526804 | 0.124584 | 0.001780 |



## Preregistered comparisons and uncertainty



Delta=first model−second model; negative favors the first. 2000 paired whole-scene bootstrap draws, seed 2022, independently resample 210 scenes within each of three folds. Every actor/window of a drawn scene stays clustered and paired across models. Four co-primary Overall Top1FDE comparisons use Bonferroni family 4: 98.75% individual intervals, giving the preregistered familywise .05 criterion. 95% intervals are also reported descriptively. Type/motion groups and interaction are exploratory; they do not replace the registered comparisons.

| Comparison | Count | DeltaTop1FDE | CI95Lower | CI95Upper | BonferroniCILower | BonferroniCIUpper | Fold1DeltaTop1FDE | Fold2DeltaTop1FDE | Fold3DeltaTop1FDE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NG-C-NG-A | 260151 | -0.126209 | -0.146300 | -0.105474 | -0.153694 | -0.101137 | -0.082031 | -0.183864 | -0.113689 |
| G-A-NG-A | 260151 | 0.003118 | -0.022065 | 0.028358 | -0.030289 | 0.033505 | 0.036315 | -0.021591 | -0.004091 |
| G-C-NG-C | 260151 | -0.055221 | -0.071748 | -0.040932 | -0.078282 | -0.037583 | -0.059371 | -0.044744 | -0.060916 |
| G-C-G-A | 260151 | -0.184549 | -0.209950 | -0.158241 | -0.217195 | -0.152250 | -0.177717 | -0.207016 | -0.170514 |



Graph gate requires comparison C (G-C−NG-C) to satisfy all conditions. Cross-structure Loss gate requires both A (NG-C−NG-A) and D (G-C−G-A) to satisfy them. No post-hoc criterion changes or parameter searches were performed.

The graph comparison under SoftCE (B: G-A−NG-A) does not meet the registered improvement conditions. The supported graph label is specific to the error-aware comparison C; it is not a claim of graph benefit under every loss. Different selected/executed epochs follow the identical original early-stopping rule, rather than an OuterTest-based budget adjustment. Bootstrap intervals condition on the frozen fitted heads and do not resample or refit training runs.

| Comparison | Count | DeltaTop1FDE | CI95Lower | CI95Upper |
| --- | --- | --- | --- | --- |
| (G-C-G-A)-(NG-C-NG-A) | 260151 | -0.058339 | -0.082994 | -0.032678 |

The interaction is descriptive association of controlled contrasts, not causal identification.



## Mode changes, high-cost harm and motion states



| Group | Count | DeltaTop1FDE | CI95Lower | CI95Upper | Fold1DeltaTop1FDE | Fold2DeltaTop1FDE | Fold3DeltaTop1FDE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vehicle | 191026 | -0.070846 | -0.093391 | -0.051169 | -0.075302 | -0.058336 | -0.077830 |
| Pedestrian | 66145 | -0.012584 | -0.018840 | -0.006891 | -0.013487 | -0.009909 | -0.014301 |
| MovingVehicle | 41728 | -0.301724 | -0.384580 | -0.228121 | -0.317476 | -0.241942 | -0.345737 |
| StoppedVehicle | 23044 | -0.015110 | -0.034448 | 0.001375 | -0.016651 | -0.015004 | -0.013905 |
| ParkedVehicle | 121939 | 0.002050 | -0.000186 | 0.004484 | 0.003541 | 0.002692 | 0.000139 |

These type/motion comparisons are exploratory. ParkedVehicle has a positive G-C−NG-C delta (worse), with all three fold directions positive and a descriptive 95% interval spanning zero. StoppedVehicle also has an interval spanning zero. MovingVehicle has a negative delta. Results therefore should be disclosed separately by current motion state.

Gross gain/harm are sums of negative/positive paired target FDE changes in meters. High-cost harm is the original comparison/group-specific largest ceil(0.1×number of positive harms) tail. Different comparisons have different tail memberships; this is not a fixed identical actor subset. Unchanged selected modes have exactly zero FDE difference.

| Comparison | Group | Count | ChangedCount | ImprovedCount | WorsenedCount | GrossGain | GrossHarm | HighCostHarmCount | HighCostHarmSum | NetFDEDelta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NG-C-NG-A | Overall | 260151 | 81298 | 49939 | 31359 | 82368.013726 | 49534.557496 | 3136 | 27042.574254 | -0.126209 |
| G-C-NG-C | Overall | 260151 | 53641 | 27920 | 25721 | 49063.384919 | 34697.528649 | 2573 | 17266.625607 | -0.055221 |
| G-C-G-A | Overall | 260151 | 115886 | 69495 | 46391 | 117556.629527 | 69546.108774 | 4640 | 43387.960014 | -0.184549 |
| NG-C-NG-A | Vehicle | 191026 | 61674 | 37895 | 23779 | 77833.210471 | 46951.985177 | 2378 | 22726.280341 | -0.161660 |
| G-C-NG-C | Vehicle | 191026 | 36517 | 18775 | 17742 | 45562.475588 | 32028.977290 | 1775 | 13002.099051 | -0.070846 |
| G-C-G-A | Vehicle | 191026 | 82713 | 50250 | 32463 | 108348.156057 | 63493.772814 | 3247 | 34891.276703 | -0.234808 |
| NG-C-NG-A | Pedestrian | 66145 | 19624 | 12044 | 7580 | 4534.803255 | 2582.572319 | 758 | 1114.465116 | -0.029514 |
| G-C-NG-C | Pedestrian | 66145 | 17124 | 9145 | 7979 | 3500.909331 | 2668.551358 | 798 | 883.731710 | -0.012584 |
| G-C-G-A | Pedestrian | 66145 | 33173 | 19245 | 13928 | 9208.473470 | 6052.335960 | 1393 | 2594.537409 | -0.047715 |
| NG-C-NG-A | Bicycle | 2980 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 |
| G-C-NG-C | Bicycle | 2980 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 |
| G-C-G-A | Bicycle | 2980 | 0 | 0 | 0 | 0.000000 | 0.000000 | 0 | 0.000000 | 0.000000 |
| NG-C-NG-A | MovingVehicle | 41728 | 22551 | 13225 | 9326 | 71681.224186 | 42838.019986 | 933 | 12457.826150 | -0.691219 |
| G-C-NG-C | MovingVehicle | 41728 | 17524 | 9815 | 7709 | 41563.670082 | 28973.334134 | 771 | 6558.399778 | -0.301724 |
| G-C-G-A | MovingVehicle | 41728 | 26945 | 16019 | 10926 | 98682.481029 | 57005.082007 | 1093 | 16286.249459 | -0.998787 |
| NG-C-NG-A | StoppedVehicle | 23044 | 5438 | 3124 | 2314 | 1123.843998 | 1047.515420 | 232 | 839.134802 | -0.003312 |
| G-C-NG-C | StoppedVehicle | 23044 | 3365 | 1613 | 1752 | 1071.959818 | 723.773817 | 176 | 567.948351 | -0.015110 |
| G-C-G-A | StoppedVehicle | 23044 | 9178 | 5693 | 3485 | 1910.292602 | 1873.881970 | 349 | 1587.404627 | -0.001580 |
| NG-C-NG-A | ParkedVehicle | 121939 | 32346 | 20752 | 11594 | 3471.481536 | 1754.441653 | 1160 | 883.253282 | -0.014081 |
| G-C-NG-C | ParkedVehicle | 121939 | 14664 | 6816 | 7848 | 1404.207395 | 1654.229960 | 785 | 1056.381741 | 0.002050 |
| G-C-G-A | ParkedVehicle | 121939 | 44613 | 27305 | 17308 | 5260.740435 | 3013.983362 | 1731 | 1812.545086 | -0.018425 |



Disjoint MovingVehicle/StoppedVehicle/ParkedVehicle/OtherVehicleState contributions are in 07_diagnostics/stage14a_motion_contribution.csv. These contributions sum to the Vehicle mean delta, preventing a static-target count effect from being silently described as a moving-vehicle improvement. Diagnostics alone do not establish a causal switching mechanism.



## Compute and parameters



| Fold | Model | TotalParameters | OptimizationTrainableParameters | BatchSize | MeanMSPerActor | PeakIncrementalAllocatedGPUMemoryBytes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | NG-A | 7425 | 7425 | 128 | 0.004970 | 2139136 |
| 1 | NG-C | 7425 | 7425 | 128 | 0.004901 | 2139136 |
| 1 | G-A | 24066 | 24066 | 128 | 0.010664 | 52061184 |
| 1 | G-C | 24066 | 24066 | 128 | 0.010535 | 52061184 |
| 2 | NG-A | 7425 | 7425 | 128 | 0.004818 | 2139136 |
| 2 | NG-C | 7425 | 7425 | 128 | 0.004793 | 2139136 |
| 2 | G-A | 24066 | 24066 | 128 | 0.010439 | 52061184 |
| 2 | G-C | 24066 | 24066 | 128 | 0.010383 | 52061184 |
| 3 | NG-A | 7425 | 7425 | 128 | 0.004779 | 2139136 |
| 3 | NG-C | 7425 | 7425 | 128 | 0.004779 | 2139136 |
| 3 | G-A | 24066 | 24066 | 128 | 0.010297 | 52061184 |
| 3 | G-C | 24066 | 24066 | 128 | 0.010293 | 52061184 |

Benchmark uses identical on-device cached seven-argument inputs, FP32, CUDA synchronization, five warmups and 20 rotating-order repeats. Only head forward is timed; frozen HiVT, packing, transfer and R2 routing are excluded. At inference every head has zero trainable parameters. CPU linear FLOP proxies and timing are separately documented in 01_preflight/stage14a_model_integrity.json. Full graph-input construction is retained for control and its removal is not claimed as measured end-to-end acceleration.



## Predictor data provenance and independent evaluation



Detailed scene-token evidence, original sampler replay, selected checkpoint ancestry and 700 training-shard hashes are in 09_reports/stage14a_predictor_data_provenance.json and its four CSV appendices. The selected Stage5A checkpoint saw all 700 training scenes and 16898 supervised windows. Historical VAL150 was checked every 500 steps, selected the predictor checkpoint, and informed Stage8/Stage9 development. No clean independent evaluation scene is verified within the registered local 850-scene corpus. A static official test name list does not establish available labels or a compatible custom three-type evaluator.

Historical RTX3080 timing gives three predictor retraining runs a 6.40–11.69 h planning anchor, depending on 11500 vs 21000 update schedules. Adding historical nine A/C/R2 heads and candidate forward yields 7.48–12.77 h before new NG heads, preprocessing, full I/O and diagnostics. Stage14A measures the six NG runs separately above. These are wall/GPU-reservation estimates, not guarantees or pure CUDA training time. 700 training shards occupy 3.784 GiB; per 378 scene InnerTrain fold approximately 1.96–2.07 GiB. End-to-end retraining was not executed or authorized. Proper outer isolation would address predictor training overlap but would not make previously developed-on scenes pristine research holdouts.



## Figures and review package



Python figures provide editable SVG and PDF, plus PNG previews, source CSV, figure contract, captions and QA. Dataset, internal OOF protocol, sample counts, units and statistical scope are explicit. Cases are descriptive selections of actual improvements/failures, not an unbiased performance estimate.

- [ stage14a_fig1_overall_2x2 ](../08_figures/stage14a_fig1_overall_2x2.svg) / [PDF](../08_figures/stage14a_fig1_overall_2x2.pdf)

- [ stage14a_fig2_vehicle_pedestrian ](../08_figures/stage14a_fig2_vehicle_pedestrian.svg) / [PDF](../08_figures/stage14a_fig2_vehicle_pedestrian.pdf)

- [ stage14a_fig3_moving_vehicle ](../08_figures/stage14a_fig3_moving_vehicle.svg) / [PDF](../08_figures/stage14a_fig3_moving_vehicle.pdf)

- [ stage14a_fig4_ablation_table ](../08_figures/stage14a_fig4_ablation_table.svg) / [PDF](../08_figures/stage14a_fig4_ablation_table.pdf)

- [ stage14a_fig5_high_cost_switch ](../08_figures/stage14a_fig5_high_cost_switch.svg) / [PDF](../08_figures/stage14a_fig5_high_cost_switch.pdf)

- [ stage14a_fig6_improvement_failure_cases ](../08_figures/stage14a_fig6_improvement_failure_cases.svg) / [PDF](../08_figures/stage14a_fig6_improvement_failure_cases.pdf)



All new files remain under outputs/stage14a_paper_graph_ablation/. Local .pt/.npy/.log/cache artifacts are excluded from Git; checkpoint manifests and hashes are included. History, Stage12 failed experiments and five untracked Stage2C redraw files remain unchanged.



## End condition



Publish this branch without merge, then STOP. Await brain-AI review. No Stage14B, new predictor/semantic module, end-to-end retraining, official VAL/test evaluation or Diffusion Planning is started.
