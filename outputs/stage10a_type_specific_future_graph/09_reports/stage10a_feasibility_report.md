【Research Question】

Result: Stage10A=STOP. Both expert types fail the predeclared feasibility targets; Bicycle stays exactly R2. Vehicle DeltaTop1FDE=+0.009103m, descriptive HeadDev 95% CI [-0.089525, +0.105510]; Pedestrian +0.070987m, descriptive HeadDev 95% CI [+0.022260, +0.126405]; Overall +0.023644m, descriptive HeadDev 95% CI [-0.049622, +0.098093]. The research question is whether fully independent Vehicle/Pedestrian G1 reranking experts improve Vehicle ranking while preserving or improving Pedestrian ranking, with Bicycle exactly frozen R2. This is one predeclared low-cost feasibility run on HeadTrain630/HeadDev70. Stage10 uses scene-level multi-actor history and predicted future mode interactions, Th=5, Tf=12, K=6. The new branch starts from `b4dd350e3f9488e284943e0eed9c93fd87adb8d8`. All new files are confined to `outputs/stage10a_type_specific_future_graph/`. Historical conclusions remain Stage8 AgentGraph=SUPPORTED, SemanticGraph=NOT_SUPPORTED, FSCG=NOT_SUPPORTED; Stage9 TAFIG=NOT_SUPPORTED, PedestrianNegativeTransferResolved=NO, VehicleBenefitRetained=NO.

【Frozen Predictor】

Stage5A remains eval(), requires_grad=False, without optimizer membership. Checkpoint SHA256: `88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`. No candidate trajectory training occurred. The fresh HeadDev replay checked 2640 original TRAIN scene-windows against frozen raw/ego prediction hashes; 1703 windows belong to HeadDev. All 29,934 full-horizon HeadDev target identities, candidate coordinates, GT and future masks retain the frozen protocol.

【Type-Specific Architecture】

Two separate complete Stage8 G1 networks, each 24,066 trainable parameters; total 48,132. Independent NodeEncoder, InteractionMessage, InteractionAttention, LayerNorm and RankingHead. No trainable tensor storage is shared. Node15D/edge17D, hidden64, one message layer, nearest eight current-valid non-self actors within <=50m, all six neighbor modes. Vehicle, Pedestrian and Bicycle remain eligible neighbors for both experts. No semantic input, map module, adapter, learned gate or macro loss. The wrapper invokes the unchanged frozen G1 forward with unused map slots set to None; bitwise forward equivalence passed. Vehicle/Pedestrian logits are original Stage5A logits plus their own expert residual. Bicycle logits/probabilities route directly to frozen R2.

|TargetType|NeighborType|DirectedCurrentNeighborOccurrences|
|---|---|---|
|Vehicle|Vehicle|1332907|
|Vehicle|Pedestrian|258810|
|Vehicle|Bicycle|15692|
|Pedestrian|Vehicle|221513|
|Pedestrian|Pedestrian|335649|
|Pedestrian|Bicycle|11041|
|Bicycle|Vehicle|12246|
|Bicycle|Pedestrian|9944|
|Bicycle|Bicycle|1563|

【Vehicle Expert】

Fresh independent seed2022, with G1 interaction submodule seed2123, final score weight/bias zero. Vehicle target loss only; all neighbor types retained. No trained Stage8/Stage9 graph weights loaded.

|Expert|Path|SHA256|SelectedEpoch|HeadDevSoftCE|Params|Seed|Selection|
|---|---|---|---|---|---|---|---|
|vehicle|outputs/stage10a_type_specific_future_graph/02_checkpoints/VehicleExpert_best.pt|e8f672ba1b77dfcf23b091aad65751cf9b27321206c2ad88c5dd0fee9bdd95d5|5|1.479799|24066|2022|minimum own-type HeadDev SoftCE|

【Pedestrian Expert】

Fresh independent seed2123, with G1 interaction submodule seed2224, final score weight/bias zero. Pedestrian target loss only; all neighbor types retained. No shared backbone or optimizer and no trained Stage8/Stage9 weights loaded.

|Expert|Path|SHA256|SelectedEpoch|HeadDevSoftCE|Params|Seed|Selection|
|---|---|---|---|---|---|---|---|
|pedestrian|outputs/stage10a_type_specific_future_graph/02_checkpoints/PedestrianExpert_best.pt|14aefd4d777248cede2a4bdbc144c1734b7bb90e12d51f74930588ca95ab3c99|4|1.692140|24066|2123|minimum own-type HeadDev SoftCE|

【Bicycle Frozen R2】

R2 SHA256: `e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314`. Bicycle keeps full histories, six predictor candidates, graph context, six final probabilities, and contributes to Overall. Bicycle is excluded only from expert training losses. No Bicycle sample was removed. Before training, all 3,138 TRAIN700 Bicycle targets retained bitwise R2 logits/probabilities; 128 actual Bicycle candidate tensors and all 3,138 candidate routing probes retained identity. After training, all 158 HeadDev Bicycle logits, probabilities and actual candidate tensors remained bitwise R2. A real TRAIN all-actor inference audit also routes every current-valid actor, including those without full future supervision, without a future-target selection mask; all three types receive6 candidates/probabilities. Changing GT/future/target masks leaves trained outputs bitwise identical.

【Training Split】

Exact frozen Stage6 seed2022 HeadTrain630/HeadDev70 scene split, SHA256 `55d8a856ce71c75b7af359e8dc66232907ae54b9a14e43c9a5f7f6148ce7d208`. Full target scene/sample/instance/node/window identities, FDE/ADE arrays and original logits join exactly. No altered time windows, horizon masks, candidate geometry or type selection.

|Partition|ActorType|Count|
|---|---|---|
|HeadTrain|Vehicle|191026|
|HeadTrain|Pedestrian|66145|
|HeadTrain|Bicycle|2980|
|HeadDev|Vehicle|22719|
|HeadDev|Pedestrian|7057|
|HeadDev|Bicycle|158|

Normalization directly reuses frozen Stage8 HeadTrain-only values, SHA256 `763d8d4ed7852ef39d87a53194048a6f798d0df32e2e0c1e783361bab1ac2827`; no refitting with HeadDev or VAL. The R2 branch separately reuses its existing frozen 19D normalized inputs. A pre-registration metadata preflight found Stage6 lowercase display labels versus Stage10 uppercase display labels; the new metadata join checks exact lowercase labels and actor_type_id, then changes presentation only. No frozen data/model/protocol was changed and no optimizer had run; this is logged in `00_manifest/stage10a_engineering_preflight.json`.

【No Future Leakage】

The observable-window allowlist exposes only histories, current type, frozen predicted trajectories and original logits/probabilities. Future GT is used solely for q=softmax(-FDE/1m) supervision and offline evaluation/displacement groups. NaN GT, inverted future/target masks and NaN semantic metadata leave graph features bitwise identical on fixed target rows. Live graph and frozen R2 input identity passed for all HeadDev targets. Runtime file-access guards reject official VAL/test cache/shard paths. Only official TRAIN source batches intersecting HeadDev70 are replayed. Official VAL was not opened, evaluated, normalized, used for selection or training.

【Tiny Overfit】

One fixed 128-target same-type HeadTrain set per expert, respective expert seed, 300 AdamW optimizer updates. ExcessLossReduction=1-(finalCE-H(q))/(initialCE-H(q)); mean target entropy is the feasible lower bound. Both exceed the fixed 90% gate. Each graph module has finite nonzero gradients; predictor/R2 gradient counts zero; candidates detached; no NaN/Inf. Formal models were reset to registered initial states, without tiny weight reuse.

|Expert|status|InitialLoss|FinalLoss|EntropyFloor|ExcessLossReduction|Updates|graph_gradient_norm|
|---|---|---|---|---|---|---|---|
|vehicle|PASS|1.524670|1.169770|1.168624|0.996781|300|0.170574|
|pedestrian|PASS|1.704147|1.507317|1.506765|0.997206|300|0.126356|

【Independent Training】

Two fully independent optimizers, AdamW lr1e-3/weight_decay1e-4, FP32, effective batch1024 using 128 microbatches x8. Loss is mean SoftCE over only the expert target type; no type weighting, macro objective, uncertainty, semantic or gate losses. Every epoch residual is prepended to the next shuffled epoch; final residual occurrences are retained in each last checkpoint. Every own-type HeadTrain identity was optimized at least once; zero other-type or HeadDev backprop targets. No short effective batches or target deletion. Max50 epochs, strict improvement, patience5.

|Expert|selected_epoch|executed_epochs|HeadDevSoftCE|parameters|precision|microbatch|effective_batch|accumulation|elapsed_seconds|unique_train_targets|final_unoptimized_carry_occurrences_retained_in_last_checkpoint|
|---|---|---|---|---|---|---|---|---|---|---|---|
|vehicle|5|10|1.479799|24066|FP32|128|1024|8|384.087734|191026|500|
|pedestrian|4|9|1.692140|24066|FP32|128|1024|8|123.163687|66145|361|

【Checkpoint Selection】

Registered before any tiny or formal update: Vehicle selects minimum complete Vehicle HeadDev SoftCE; Pedestrian selects minimum complete Pedestrian HeadDev SoftCE. Top1ADE/Top1FDE are recorded each epoch but never select checkpoints. Earlier epoch wins exact ties. Both checkpoints were frozen together before unified HeadDev evaluation, and further training/overwriting is prohibited by the manifest. Unified own-type selected checkpoint CE/ADE/FDE matches training records within1e-6; frozen G1 HeadDev SoftCE reproduction difference 2.7379032374597045e-10.

|Expert|Path|SHA256|SelectedEpoch|HeadDevSoftCE|Params|Seed|Selection|
|---|---|---|---|---|---|---|---|
|vehicle|outputs/stage10a_type_specific_future_graph/02_checkpoints/VehicleExpert_best.pt|e8f672ba1b77dfcf23b091aad65751cf9b27321206c2ad88c5dd0fee9bdd95d5|5|1.479799|24066|2022|minimum own-type HeadDev SoftCE|
|pedestrian|outputs/stage10a_type_specific_future_graph/02_checkpoints/PedestrianExpert_best.pt|14aefd4d777248cede2a4bdbc144c1734b7bb90e12d51f74930588ca95ab3c99|4|1.692140|24066|2123|minimum own-type HeadDev SoftCE|

【HeadDev Results】

Only 70 HeadDev scenes, 29,934 full-horizon targets. R0 original logits, frozen R2, frozen Stage8 G1 reference, and the combined DualExpert are evaluated on identical candidates. All required groups appear in `06_tables/stage10a_type_results.csv` and `stage10a_motion_results.csv`. Top1FDE is primary; negative DualExpert-R2 difference improves ranking. HitRate/MRR use the lowest-index FDE-best candidate and stable descending probability order; HitRate is a ranking oracle agreement metric, not behavior-prediction accuracy. OracleGap=Top1FDE-minFDE6; MR6 threshold2m. Historic minADE6 is ADE at the FDE-best mode; independent ADE oracle is additionally reported as minADEOracle6.

|Group|Model|Count|Top1FDE|Top1ADE|OracleGap|HitRate|MRR|minADE6|minADEOracle6|minFDE6|MR6|SoftCE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Overall|R0|29934|2.491074|1.120124|1.132238|0.259404|0.550904|0.657352|0.623350|1.358836|0.152502|1.584203|
|Overall|R2|29934|2.431732|1.094487|1.072895|0.263780|0.554720|0.657352|0.623350|1.358836|0.152502|1.549708|
|Overall|G1|29934|2.409881|1.093499|1.051045|0.400314|0.625008|0.657352|0.623350|1.358836|0.152502|1.531178|
|Overall|DualExpert|29934|2.455376|1.118774|1.096539|0.421995|0.635518|0.657352|0.623350|1.358836|0.152502|1.531320|

|Group|Model|Count|Top1FDE|Top1ADE|OracleGap|HitRate|MRR|minADE6|minADEOracle6|minFDE6|MR6|SoftCE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|MovingVehicle|R0|5067|11.245520|4.948338|5.509199|0.204460|0.454263|2.691087|2.550981|5.736320|0.636866|1.766891|
|MovingVehicle|R2|5067|10.890835|4.792633|5.154514|0.227156|0.472140|2.691087|2.550981|5.736320|0.636866|1.768546|
|MovingVehicle|G1|5067|10.590741|4.701892|4.854420|0.297612|0.531481|2.691087|2.550981|5.736320|0.636866|1.681498|
|MovingVehicle|DualExpert|5067|10.847486|4.838971|5.111165|0.276692|0.517548|2.691087|2.550981|5.736320|0.636866|1.685508|
|StoppedVehicle|R0|2895|1.051554|0.416049|0.400113|0.251123|0.558440|0.283862|0.262148|0.651441|0.068048|1.526860|
|StoppedVehicle|R2|2895|1.056705|0.418968|0.405263|0.247668|0.556782|0.283862|0.262148|0.651441|0.068048|1.523661|
|StoppedVehicle|G1|2895|1.064206|0.423241|0.412765|0.394128|0.625124|0.283862|0.262148|0.651441|0.068048|1.507503|
|StoppedVehicle|DualExpert|2895|1.029223|0.411561|0.377781|0.406563|0.631491|0.283862|0.262148|0.651441|0.068048|1.508072|
|ParkedVehicle|R0|14299|0.206892|0.131843|0.096679|0.249178|0.583045|0.086260|0.078847|0.110213|0.000769|1.466970|
|ParkedVehicle|R2|14299|0.204175|0.130454|0.093962|0.248899|0.582670|0.086260|0.078847|0.110213|0.000769|1.401311|
|ParkedVehicle|G1|14299|0.214603|0.133401|0.104390|0.488706|0.698887|0.086260|0.078847|0.110213|0.000769|1.400424|
|ParkedVehicle|DualExpert|14299|0.224338|0.138294|0.114125|0.538499|0.723065|0.086260|0.078847|0.110213|0.000769|1.400783|
|Vehicle>5m|R0|4490|12.953887|5.614342|6.193996|0.175724|0.423712|3.107977|2.951255|6.759892|0.736080|1.834136|
|Vehicle>5m|R2|4490|12.568075|5.445453|5.808183|0.197773|0.442038|3.107977|2.951255|6.759892|0.736080|1.869941|
|Vehicle>5m|G1|4490|12.092468|5.281188|5.332576|0.279955|0.514855|3.107977|2.951255|6.759892|0.736080|1.754090|
|Vehicle>5m|DualExpert|4490|12.427912|5.459042|5.668020|0.256125|0.497120|3.107977|2.951255|6.759892|0.736080|1.769069|
|Pedestrian<5m|R0|2253|0.938316|0.423991|0.444487|0.426542|0.628488|0.265060|0.245787|0.493828|0.066134|1.625686|
|Pedestrian<5m|R2|2253|0.934137|0.421523|0.440309|0.419885|0.623887|0.265060|0.245787|0.493828|0.066134|1.581127|
|Pedestrian<5m|G1|2253|0.940364|0.436516|0.446535|0.374612|0.597189|0.265060|0.245787|0.493828|0.066134|1.587987|
|Pedestrian<5m|DualExpert|2253|0.901894|0.420494|0.408066|0.357745|0.594992|0.265060|0.245787|0.493828|0.066134|1.574284|
|Pedestrian>5m|R0|4804|1.644931|0.774979|0.404462|0.277893|0.519019|0.605348|0.588536|1.240469|0.187552|1.758206|
|Pedestrian>5m|R2|4804|1.657814|0.783497|0.417345|0.281848|0.525066|0.605348|0.588536|1.240469|0.187552|1.760772|
|Pedestrian>5m|G1|4804|1.751573|0.836545|0.511104|0.264571|0.519529|0.605348|0.588536|1.240469|0.187552|1.749594|
|Pedestrian>5m|DualExpert|4804|1.777215|0.846831|0.536746|0.271024|0.524240|0.605348|0.588536|1.240469|0.187552|1.747412|

【Vehicle Results】

Vehicle DeltaTop1FDE=+0.009103m, descriptive HeadDev 95% CI [-0.089525, +0.105510]; this misses the -0.03m GO target and the conditional requirement of any Vehicle improvement. MovingVehicle delta=-0.043349m, descriptive HeadDev 95% CI [-0.464593, +0.362203] and Vehicle>5m delta=-0.140163m, descriptive HeadDev 95% CI [-0.597424, +0.303267]. Those subgroup point improvements coexist with Vehicle aggregate deterioration; both corresponding intervals cross0. ParkedVehicle delta=+0.020163m, descriptive HeadDev 95% CI [-0.006353, +0.056550] and StoppedVehicle delta=-0.027482m, descriptive HeadDev 95% CI [-0.080458, +0.021122]. All groups are retained in the aggregate and reported without selecting only improvements.

|Group|Model|Count|Top1FDE|Top1ADE|OracleGap|HitRate|MRR|minADE6|minADEOracle6|minFDE6|MR6|SoftCE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Vehicle|R0|22719|2.819267|1.259696|1.351911|0.238655|0.549919|0.705699|0.666643|1.467356|0.153088|1.542590|
|Vehicle|R2|22719|2.739791|1.224756|1.272435|0.243189|0.553521|0.705699|0.666643|1.467356|0.153088|1.500523|
|Vehicle|G1|22719|2.687555|1.209498|1.220199|0.431885|0.650376|0.705699|0.666643|1.467356|0.153088|1.478116|
|Vehicle|DualExpert|22719|2.748893|1.243466|1.281538|0.460099|0.663018|0.705699|0.666643|1.467356|0.153088|1.479799|

|Group|New|Baseline|Metric|Count|Delta|CI_lower|CI_upper|Replicates|Seed|EvidenceRole|
|---|---|---|---|---|---|---|---|---|---|---|
|Vehicle|DualExpert|R2|Top1FDE|22719|0.009103|-0.089525|0.105510|1000|2022|HeadDev checkpoint-selected feasibility screening; not independent confirmation|

【Pedestrian Results】

Pedestrian DeltaTop1FDE=+0.070987m, descriptive HeadDev 95% CI [+0.022260, +0.126405]; its descriptive selected-split interval lies above0 and the point delta also exceeds the +0.01m conditional tolerance. Pedestrian<5m delta=-0.032244m, descriptive HeadDev 95% CI [-0.094489, +0.032679] while Pedestrian>5m delta=+0.119401m, descriptive HeadDev 95% CI [+0.055684, +0.192439]. The independent Pedestrian expert has lower own-type SoftCE than frozen R2 and G1, but its aggregate Top1FDE remains worse. The fixed SoftCE checkpoint rule was followed; no FDE-driven reselection or model change follows this result.

|Group|Model|Count|Top1FDE|Top1ADE|OracleGap|HitRate|MRR|minADE6|minADEOracle6|minFDE6|MR6|SoftCE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Pedestrian|R0|7057|1.419339|0.662923|0.417240|0.325351|0.553968|0.496708|0.479111|1.002098|0.148788|1.715898|
|Pedestrian|R2|7057|1.426775|0.667934|0.424677|0.325918|0.556615|0.496708|0.479111|1.002098|0.148788|1.703419|
|Pedestrian|G1|7057|1.492588|0.708833|0.490490|0.299702|0.544322|0.496708|0.479111|1.002098|0.148788|1.697999|
|Pedestrian|DualExpert|7057|1.497762|0.710719|0.495664|0.298711|0.546828|0.496708|0.479111|1.002098|0.148788|1.692140|

|Group|New|Baseline|Metric|Count|Delta|CI_lower|CI_upper|Replicates|Seed|EvidenceRole|
|---|---|---|---|---|---|---|---|---|---|---|
|Pedestrian|DualExpert|R2|Top1FDE|7057|0.070987|0.022260|0.126405|1000|2022|HeadDev checkpoint-selected feasibility screening; not independent confirmation|

【Bicycle Results】

|Group|Model|Count|Top1FDE|Top1ADE|OracleGap|HitRate|MRR|minADE6|minADEOracle6|minFDE6|MR6|SoftCE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Bicycle|R0|158|3.168470|1.471575|1.480202|0.297468|0.555696|0.880476|0.840521|1.688268|0.234177|1.685755|
|Bicycle|R2|158|3.021537|1.414764|1.333269|0.449367|0.642511|0.880476|0.840521|1.688268|0.234177|1.756531|
|Bicycle|G1|158|3.453326|1.594887|1.765058|0.354430|0.581013|0.880476|0.840521|1.688268|0.234177|1.710004|
|Bicycle|DualExpert|158|3.021537|1.414764|1.333269|0.449367|0.642511|0.880476|0.840521|1.688268|0.234177|1.756531|

All Bicycle model outputs match frozen R2 exactly; DualExpert-R2 Top1FDE difference and all bootstrap draws equal0. The 158 target count limits independent claims about Bicycle generalization.

|Group|New|Baseline|Metric|Count|Delta|CI_lower|CI_upper|Replicates|Seed|EvidenceRole|
|---|---|---|---|---|---|---|---|---|---|---|
|Bicycle|DualExpert|R2|Top1FDE|158|0.000000|0.000000|0.000000|1000|2022|HeadDev checkpoint-selected feasibility screening; not independent confirmation|

【Overall Results】

|Group|Model|Count|Top1FDE|Top1ADE|OracleGap|HitRate|MRR|minADE6|minADEOracle6|minFDE6|MR6|SoftCE|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Overall|R0|29934|2.491074|1.120124|1.132238|0.259404|0.550904|0.657352|0.623350|1.358836|0.152502|1.584203|
|Overall|R2|29934|2.431732|1.094487|1.072895|0.263780|0.554720|0.657352|0.623350|1.358836|0.152502|1.549708|
|Overall|G1|29934|2.409881|1.093499|1.051045|0.400314|0.625008|0.657352|0.623350|1.358836|0.152502|1.531178|
|Overall|DualExpert|29934|2.455376|1.118774|1.096539|0.421995|0.635518|0.657352|0.623350|1.358836|0.152502|1.531320|

|Group|New|Baseline|Metric|Count|Delta|CI_lower|CI_upper|Replicates|Seed|EvidenceRole|
|---|---|---|---|---|---|---|---|---|---|---|
|Overall|DualExpert|R2|Top1FDE|29934|0.023644|-0.049622|0.098093|1000|2022|HeadDev checkpoint-selected feasibility screening; not independent confirmation|

【Bootstrap】

Paired whole-scene cluster bootstrap over all70 HeadDev scenes, 1000 draws, seed2022, shared resampling indices for every group/metric. Keep each selected scene intact, including repeated windows/actors, and compute actor-weighted means with duplicated scene multiplicities. Percentile95% intervals on DualExpert-R2, with Top1FDE/Top1ADE/HitRate for all10 requested groups; 30 rows. These intervals screen feasibility on a checkpoint-selection split and do not provide independent confirmation or causal evidence. No interval-based seed/model/hyperparameter choice occurred.

|Group|New|Baseline|Metric|Count|Delta|CI_lower|CI_upper|Replicates|Seed|EvidenceRole|
|---|---|---|---|---|---|---|---|---|---|---|
|Overall|DualExpert|R2|Top1FDE|29934|0.023644|-0.049622|0.098093|1000|2022|HeadDev checkpoint-selected feasibility screening; not independent confirmation|
|Vehicle|DualExpert|R2|Top1FDE|22719|0.009103|-0.089525|0.105510|1000|2022|HeadDev checkpoint-selected feasibility screening; not independent confirmation|
|Pedestrian|DualExpert|R2|Top1FDE|7057|0.070987|0.022260|0.126405|1000|2022|HeadDev checkpoint-selected feasibility screening; not independent confirmation|
|Bicycle|DualExpert|R2|Top1FDE|158|0.000000|0.000000|0.000000|1000|2022|HeadDev checkpoint-selected feasibility screening; not independent confirmation|

【Candidate Identity】

PASS: fresh predictor raw/ego tensors match frozen source bitwise; all model routes share the same detached six candidates. maxdiff candidates/minADE6/minADEOracle6/minFDE6/MR6 =0. Bicycle logits/probability/candidates match frozen R2 bitwise for every158 HeadDev Bicycle target. Predictor/R2 states unchanged, gradients0. Initialization logits/probability maxdiff0 on1024 Vehicle and1024 Pedestrian targets.

|Model|Targets|CandidateMaxDiff|minADE6MaxDiff|minADEOracle6MaxDiff|minFDE6MaxDiff|MR6MaxDiff|BicycleLogitsBitwiseR2|BicycleProbabilityBitwiseR2|Status|
|---|---|---|---|---|---|---|---|---|---|
|R0|29934|0.000000|0.000000|0.000000|0.000000|0.000000|0|0|PASS|
|R2|29934|0.000000|0.000000|0.000000|0.000000|0.000000|1|1|PASS|
|G1|29934|0.000000|0.000000|0.000000|0.000000|0.000000|0|0|PASS|
|DualExpert|29934|0.000000|0.000000|0.000000|0.000000|0.000000|1|1|PASS|

【Negative Transfer Analysis】

The design removes shared trainable parameters across Vehicle/Pedestrian target losses, while preserving heterogeneous interaction neighbors. HeadDev group deltas and mode-change counts characterize whether this single independent-expert run meets the frozen screening goals. Historical Stage8/Stage9 official VAL failures stay unchanged; results on different splits cannot be combined into a causal proof that parameter sharing caused those failures.

|Group|Count|ChangedCount|ChangedRate|ImprovedCount|WorsenedCount|UnchangedCount|DeltaTop1FDE|
|---|---|---|---|---|---|---|---|
|Overall|29934|18881|0.630754|10748|8133|11053|0.023644|
|Vehicle|22719|14489|0.637748|8974|5515|8230|0.009103|
|Pedestrian|7057|4392|0.622361|1774|2618|2665|0.070987|
|Bicycle|158|0|0.000000|0|0|158|0.000000|

The larger Overall HitRate/MRR values do not establish a Top1FDE benefit: the primary Overall delta is +0.023644m, descriptive HeadDev 95% CI [-0.049622, +0.098093]. Complete parameter separation did not meet either aggregate Vehicle improvement or Pedestrian preservation in this run. These observations reject the specified feasibility screen; they do not identify a causal failure mechanism or prove all independent experts ineffective. NegativeTransferMitigated=NO under the predeclared HeadDev rule. YES requires Vehicle point improvement plus nonpositive Pedestrian delta and exact Bicycle preservation; PARTIAL allows Pedestrian<=+0.01m; otherwise NO.

【Limitations】

The Stage10 hypothesis was motivated by prior official VAL outcomes. HeadDev also selects both expert checkpoints, and Stage5A/R2 have historical training/checkpoint-selection exposure. This report therefore supplies HeadDev feasibility evidence, not an independent end-to-end evaluation. One fixed seed per expert, overlapping motion groups and small Bicycle sample; bootstrap is unadjusted descriptive screening. Candidate oracle geometry is frozen and does not improve through reranking. Efficiency below measures only normalized-tensor reranker forward, excluding predictor/graph construction/transfer, not end-to-end latency. No case-based model selection or additional model/seed/hyperparameter search. Official VAL/test and Stage10B were not executed.

|Model|TargetBatch|VehicleTargets|PedestrianTargets|BicycleTargets|Parameters|Stage10TrainableExpertParameters|FrozenParameters|Mean_ms|Median_ms|P95_ms|PeakCUDA_MiB|IncrementalCUDA_MiB|Warmup|MeasuredForwards|Scope|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|R2|128|64|48|16|673|0|673|0.290304|0.284672|0.314370|36.302246|1.093750|50|200|reranker forward on normalized CUDA tensors; DualExpert includes frozen R2 plus type routing; excludes predictor, graph/feature construction and transfer|
|G1|128|64|48|16|24066|0|24066|1.345150|1.332864|1.494803|84.974121|49.765625|50|200|reranker forward on normalized CUDA tensors; DualExpert includes frozen R2 plus type routing; excludes predictor, graph/feature construction and transfer|
|DualExpert|128|64|48|16|48805|48132|673|4.135860|4.075504|4.346914|61.346191|26.137695|50|200|reranker forward on normalized CUDA tensors; DualExpert includes frozen R2 plus type routing; excludes predictor, graph/feature construction and transfer|

【Scientific Decision】

Frozen GO rule: Vehicle delta<=-0.03m, Pedestrian delta<=0m, Bicycle bitwise R2, Overall delta<0m and all engineering gates PASS. Otherwise CONDITIONAL_GO requires Vehicle delta<0m, Pedestrian delta<=+0.01m, Bicycle exact and Overall delta<0m with all gates PASS; otherwise STOP. Auxiliary label definitions are frozen in `00_manifest/stage10a_protocol.json`; bootstrap CI does not substitute for the point-estimate screening rule.

VehicleExpert = NOT_PROMISING
PedestrianExpert = NOT_PROMISING
BicyclePreserved = YES
OverallImproved = NO
NegativeTransferMitigated = NO
Stage10A = STOP
ReadyStage10B = NO

STOP after this phase, regardless of feasibility label. ReadyStage10B is only a recommendation for review. Do not run official VAL, train more models or execute Stage10B automatically. Wait for 大脑AI review.

