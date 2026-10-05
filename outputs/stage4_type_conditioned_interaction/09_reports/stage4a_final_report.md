# Stage4A：Type-Conditioned Interaction

本阶段独立目录：`outputs/stage4_type_conditioned_interaction/`。按用户2026-10-05补充要求，Stage4代码及产物与Stage3分开保存；冻结Stage3输入只读引用。训练与最终评价完成后执行逐SHA一致的文件迁移，原训练源码快照、核心函数AST核验和路径适配记录见 `../00_manifest/stage4a_relocation_audit.json`。未重训、未更换checkpoint、未改指标或科学规则。

【Research Question】

在固定节点级TypeEmbedding之后，有向source-target类型条件关系bias是否进一步改善交互表示和预测误差？本轮仅检验这个变量。

【Model Change】

Type pair definition=pair_id=3*target_type+source_type；PyG edge[0]=source j、edge[1]=target i；V0/P1/B2。

9有向关系=V<-V、V<-P、V<-B、P<-V、P<-P、P<-B、B<-V、B<-P、B<-B，V<-P与P<-V不同。

Pair embedding dim=16；nn.Embedding(9,16)。

Relation feature=[E_pair, target-frame(source−target) position/50m, cos(source_angle−target_angle), sin(source_angle−target_angle)]，20维。

Bias MLP=Linear(20,32)→ReLU→Linear(32,24)；reshape[E,3,8]；最后weight/bias均zero-init。

Insertion position=原agent-agent attention score之后、softmax之前加bias；保留原rel_embed、q/k/v、gate、FFN、LayerNorm、projection及value aggregation。

未直接修改runtime GlobalInteractor。LocalEncoder、TypeEmbedding、Decoder的结构及loss定义不变，Stage4A全部参数从头训练；拓扑、节点、map及数据冻结，无lane-node relation bias。

Stage3B params=646001；Stage4A params=647609；Additional params=1608；relative increase=0.248916%。

【Initialization Audit】

shared diff=0.0；shared state SHA256=bbd3245d253200531ccc2734f31fe1cfd4f50efca1f817787c655f2bad37e90a。

step0 raw prediction diff=8.64267349e-07；logit diff=5.06639481e-07；prob diff=1.49011612e-07，均<1e-6。

canonical Stage3B seed2022 step0逐name/shape完全一致；从头训练，无训练后Stage3B权重加载。12unit tests、future扰动不变性、10update梯度、原六TRAIN-window tiny均PASS，tiny模型丢弃。训练时源码已存为不可变snapshot；独立目录中的执行脚本仅在训练结束后适配路径。

新增relation模块中首反传仅final MLP有非零梯度符合zero-init设计；pair embedding与first MLP在10update内非零。

【Training】

warmup steps=5000；NLL steps=6500；executed global step=11500。

best global step=9000；stop reason=patience_5；converged_by_patience=True；stopped_by_budget=False。

恢复本模型自己的warmup best step=4500的model/AdamW/RNG；仅LR0.001→0.0001并依冻结Stage3B重置NLL sampler cursor。

warmup固定5000，NLL最多16000/global最多21000；每500完整official VAL150；post-update originalNLL overall full-horizon minFDE6 strict improvement；patience5。

只增加五项配置；复制YAML历史预算字段保留，实际执行预算与冻结Stage3B注册/执行一致。seed2022/batch16/Th5/Tf12/K6/embed64/heads8/global3/dropout0.1/radius50/weight_decay1e-4。

official train700/val150，850shard逐SHA核验；VAL3603有监督window，test unused，未重新预处理，ego仅坐标参考。

训练中断后从本模型NLL global5500 checkpoint恢复model、AdamW、CPU/CUDA RNG与sampler cursor，未保存的5600–5700更新重放。原协议未启用确定性算法，GPU重放不保证逐位一致；恢复记录见 ../00_manifest/stage4a_resume_record.json。

训练代码commit=6a3b840d491bbcc02ae34d33fce221ca0a8f0f6e；config SHA256=3e8cd2e6a07046636db6c08a5818b216c293a92d0ea02d2ff2825d025a2a1ea3；best checkpoint SHA256=ff25266ba6a44630cfec01ae1596ef7de05f0d71f06f9676f6ea7ef327775ecb。

【Main Results】

所有正式数字来自最终best checkpoint重新完整VAL推理，NaN/Inf=0。Delta=C(Stage4A)−B(Stage3B)，负值更好；meters。minADE6是best-FDE mode的ADE；MR为endpoint>2m；Top1为argmax概率；原LaplaceNLL定义不变。

首次最终VAL复核中best-FDE等指标差值<1e-6，但stopped vehicle Top1FDE均值差约1.26e-5 m导致过严的统一断言失败。原GPU协议非确定性，近似并列概率的argmax可能切换；仅Top1跨运行复核容差改为1e-4 m，主指标/NLL仍为1e-6。正式结果使用最终重新推理值，checkpoint、训练、科学判定及case逐点数值审计均未改变。逐项差值见 ../04_evaluation/stage4a_fresh_val_reconciliation.json。

| Group | Count | B ADE | C ADE | ΔADE | B FDE | C FDE | ΔFDE | B MR | C MR | B Top1FDE | C Top1FDE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 54990 | 0.666406 | 0.660990 | -0.005415 | 1.343260 | 1.335566 | -0.007694 | 0.163102 | 0.165939 | 2.687984 | 2.667320 |
| Vehicle | 42332 | 0.708347 | 0.700553 | -0.007794 | 1.431914 | 1.420021 | -0.011892 | 0.165761 | 0.168596 | 3.050323 | 3.026712 |
| Pedestrian | 12002 | 0.525708 | 0.527969 | 0.002261 | 1.043875 | 1.050126 | 0.006251 | 0.154891 | 0.157307 | 1.457706 | 1.444188 |
| Bicycle | 656 | 0.534082 | 0.541709 | 0.007628 | 1.099908 | 1.107962 | 0.008054 | 0.141768 | 0.152439 | 1.814865 | 1.853687 |
| vehicle.moving | 10461 | 2.328189 | 2.279427 | -0.048762 | 4.791532 | 4.727113 | -0.064420 | 0.597075 | 0.608164 | 10.734859 | 10.569072 |
| vehicle.stopped | 5599 | 0.350035 | 0.354188 | 0.004152 | 0.868888 | 0.864945 | -0.003943 | 0.087158 | 0.086444 | 1.313399 | 1.344684 |
| vehicle.parked | 25198 | 0.109823 | 0.116820 | 0.006996 | 0.146415 | 0.155541 | 0.009126 | 0.004088 | 0.004207 | 0.248490 | 0.266921 |
| unknown | 1074 | 0.841131 | 0.823100 | -0.018031 | 1.803797 | 1.768961 | -0.034837 | 0.167598 | 0.172253 | 2.992323 | 3.080963 |

【Nontrivial Motion】

严格GT endpoint displacement阈值；Count为重叠actor-window，unique实例/scene另报。

| Group | Count | Instances | Scenes | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vehicle >5m | 9744 | 853 | 137 | 2.594057 | 5.516344 | 0.695402 | 4.961740 | 11.752176 |
| Pedestrian >1m | 8810 | 825 | 113 | 0.672032 | 1.366453 | 0.213394 | 0.863151 | 1.838046 |
| Pedestrian >5m | 7581 | 716 | 110 | 0.660095 | 1.336439 | 0.206041 | 0.809769 | 1.691945 |
| Bicycle >1m | 155 | 15 | 13 | 1.941983 | 4.265745 | 0.645161 | 3.123901 | 7.168146 |
| Bicycle >5m | 132 | 12 | 11 | 2.124955 | 4.637823 | 0.659091 | 3.496546 | 8.001181 |

| Group | B FDE | C FDE | ΔFDE |
| --- | --- | --- | --- |
| Vehicle >5m | 5.579887 | 5.516344 | -0.063543 |
| Pedestrian >1m | 1.361468 | 1.366453 | 0.004984 |
| Pedestrian >5m | 1.340389 | 1.336439 | -0.003950 |
| Bicycle >1m | 4.255760 | 4.265745 | 0.009985 |
| Bicycle >5m | 4.625799 | 4.637823 | 0.012024 |

Bicycle >5m只有132个actor-window、12unique实例、11scene；仅描述性解释，不作强改善或因果论断。

【Interaction-sensitive Results】

20m阈值及membership在任何正式optimizer update之前固定；只用t0 positions/types/current valid distinct actors，无未来GT或预测筛组。Heterogeneous-20m含不同type邻居；VP-context限定vehicle/pedestrian双向。

membership ledger SHA256=4be8dcebb5a5295ab3ff123ab606cf52701d71ab693a1ff9bf7ff1d6b8dd2b7a；全85027actor身份已登记，完整GT/mask未用于membership。

| Group | Count | Instances | Scenes | B FDE | C FDE | ΔFDE |
| --- | --- | --- | --- | --- | --- | --- |
| Heterogeneous-20m | 25113 | 2399 | 131 | 1.335789 | 1.325226 | -0.010563 |
| Vehicle hetero-20m | 14813 | 1365 | 131 | 1.560940 | 1.539630 | -0.021310 |
| Pedestrian hetero-20m | 9732 | 968 | 120 | 0.999131 | 1.004340 | 0.005209 |
| VP-context-20m | 23209 | 2213 | 125 | 1.330427 | 1.318367 | -0.012061 |

【Bootstrap】

严格配对full54990/partial30037/total85027；scene/sample/instance/node/type/motion/全部12步GT哈希及future mask均一致。

paired scene-cluster percentile bootstrap：150official VAL scenes、1000replicates、seed2022；按scene重采样并pool窗口delta，所有组同draws。CI仅衡量固定两模型的scene采样变异，不涵盖训练seed变异；辅助子组未经多重比较校正。

| Group | Metric | Δ C−B (m) | 95%CI (m) |
| --- | --- | --- | --- |
| overall | minADE6 | -0.005415 | [-0.010582, -0.001050] |
| overall | minFDE6 | -0.007694 | [-0.017672, 0.001232] |
| overall | Top1FDE6 | -0.020664 | [-0.055616, 0.013397] |
| vehicle | minADE6 | -0.007794 | [-0.014246, -0.002294] |
| vehicle | minFDE6 | -0.011892 | [-0.025249, -0.000159] |
| pedestrian | minADE6 | 0.002261 | [-0.000324, 0.004810] |
| pedestrian | minFDE6 | 0.006251 | [0.001364, 0.010847] |
| bicycle | minADE6 | 0.007628 | [-0.017214, 0.032169] |
| bicycle | minFDE6 | 0.008054 | [-0.025481, 0.047980] |
| vehicle.moving | minFDE6 | -0.064420 | [-0.111634, -0.019580] |
| Vehicle >5m | minFDE6 | -0.063543 | [-0.113485, -0.015894] |
| Pedestrian >5m | minFDE6 | -0.003950 | [-0.009750, 0.001107] |
| Heterogeneous-20m | minFDE6 | -0.010563 | [-0.023507, 0.001485] |
| Vehicle hetero-20m | minFDE6 | -0.021310 | [-0.044009, -0.000835] |
| Pedestrian hetero-20m | minFDE6 | 0.005209 | [-0.000050, 0.009995] |
| VP-context-20m | minFDE6 | -0.012061 | [-0.025652, 0.000823] |

【Relation Bias Analysis】

9directedpairs分别按3layer×8head记录edge observation count、mean、std、mean absolute bias；heatmap聚合layer/head的mean|bias|并保留source CSV。

详见 ../06_tables/stage4a_relation_bias_statistics.csv 和 ../04_evaluation/stage4a_relation_bias_statistics.json。bias/attention仅模型内部诊断，不代表因果影响或某交通类别物理上更重要。

| Directed pair | Edge observations | Mean bias | Std bias | Mean absolute bias |
| --- | --- | --- | --- | --- |
| V<-V | 1784478 | -3.547069 | 11.423254 | 10.692157 |
| V<-P | 405857 | 3.341585 | 7.893193 | 7.512382 |
| V<-B | 25016 | 0.281055 | 0.709394 | 0.637701 |
| P<-V | 405857 | 3.740420 | 8.894288 | 8.549742 |
| P<-P | 383886 | -2.755580 | 8.524088 | 7.977991 |
| P<-B | 13304 | 0.099881 | 0.331839 | 0.269169 |
| B<-V | 25016 | -0.556171 | 2.480924 | 2.267070 |
| B<-P | 13304 | -0.116975 | 0.861597 | 0.702069 |
| B<-B | 1488 | 0.226998 | 0.989067 | 0.784602 |

【Efficiency】

相同GPU、真实VAL输入、batch16、evalmode与预热；500个配对timed batch observations，另隔离各模型测CUDA peak memory；source timings及统计口径见efficiency audit。

| Model | Params | Mean inference (ms) | Median inference (ms) | Peak CUDA allocated (MiB) |
| --- | --- | --- | --- | --- |
| Stage3B | 646001 | 57.778549 | 53.880831 | 1735.671387 |
| Stage4A | 647609 | 58.314038 | 54.687824 | 1735.678711 |

Inference overhead=0.926794%；additional parameters=1608（0.248916%）。

计时范围、个体window/batch数、warm-up与峰值显存定义均以audit为准，参数量与实际运行开销分别报告。

【Qualitative Cases】

同actor/scene/GT，左右Stage3B/Stage4A同axis/equalaspect；每条future显示12markers；仅绘History/GT/Best-FDE/Top1。案例误差对sourceCSV重新计算<1e-4，inset只按最后6个真实点bbox必要时启用。

vehicle/pedestrian improvement来自预注册heterogeneous20m群组，degradation真实展示局限；V-P neighbor仅淡色history/GT。t0 spatial interaction context不等价于真实因果交互；案例有目的选取不能代表整体频率。

Heterogeneous-20m improvement=stage4a_vehicle_improvement_case_001；source=../04_evaluation/cases/stage4a_vehicle_improvement_case_001.json。

Heterogeneous-20m improvement=stage4a_pedestrian_improvement_case_001；source=../04_evaluation/cases/stage4a_pedestrian_improvement_case_001.json。

Degradation=stage4a_degradation_case_001；source=../04_evaluation/cases/stage4a_degradation_case_001.json。

V-P spatial context=stage4a_vp_context_case_001；source=../04_evaluation/cases/stage4a_vp_context_case_001.json。

【Scientific Decision】

Type-conditioned Interaction=NOT SUPPORTED。

Reason=Preregistered reliable major-class degradation guard triggered: pedestrian

预注册主要类别异常退化守门=任一V/P/B overall FDE配对CI95 lower>0；B仅保守守门，不作强改善结论。

Stage4A=PASS；Ready for Stage4B Reliability=NO。

PASS仅表示实现、初始化、训练、冻结、配对、效率、图与报告流程正确，与效果方向独立。Ready需要SUPPORTED/PARTIAL且无overall/majorclass可靠退化。

本轮STOP，未执行Stage4B Reliability Head、Intent、TTC、future compatibility、mode reranking、校准、新图拓扑、额外seed、超参搜索、class-balanced loss、oversampling或main合并。

【Audit and Artifacts】

frozen protected files=1006；frozen scene shards=850；实际完整VAL点=23。

Stage3A/Stage3B结果与原runtime文件保持SHA一致；旧Stage2C五个未跟踪重绘文件保留且不提交。大checkpoint、actorCSV、membership ledger、frozen数据不上传Git；小代码/报告/真实图表上传stage4a/type-conditioned-interaction。
