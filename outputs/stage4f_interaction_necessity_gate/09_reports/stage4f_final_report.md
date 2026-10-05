# Stage4F：Interaction Necessity-Gated Type-Conditioned Interaction

## Research Motivation

冻结的 Stage4A always-on relation bias 改善 vehicle，却可靠损害 pedestrian；Stage4A-E 将主要损害定位于 pedestrian <5m，global gradient conflict 证据为 WEAK。单一研究问题是：仅根据目标 actor 可观测历史和当前邻域，用 scalar necessity gate 控制额外 relation enhancement，能否保留 moving vehicle 收益并恢复低运动 pedestrian？该假设不是既定事实，下面按预注册规则判定。

## Method

保留 LocalEncoder、TypeEmbedding、decoder、原 HiVT update gate、value/aggregation、FFN 和 LayerNorm。复用 Stage4A 的 directed pair_id=3×target_type+source_type，Embedding(9,16)，20→32→24 relation MLP，输出每条边3layers×8heads。仅新增9→16→1 ReLU MLP，经 sigmoid 得到目标 scalar g_i；所有 incoming edges、layers、heads 共享 g_i。Attention logits = base + g_i×b_ij。新增177参数，总647786；相对 Stage3B 新增1785参数。无 gate label、额外监督、正则损失、类别重权或 oversampling。

九维输入依次为 V/P/B onehot3、log1p recent displacement、log1p history net displacement、log1p history path length、log1p current50m incoming neighbor count、minimum distance/50、heterogeneous fraction。历史路径连接按时间排序的连续有效观测，跳过 padding；不足两点时运动统计为0；无邻居时 count=0、distance/50=1、heterogeneous fraction=0。

**拓扑审计修正**：冻结实现的 GlobalInteractor 实际为当前有效 actor 完整有向图；50m 截断仅位于 LocalEncoder。为了保持 B/C/D 可比，本轮完全保留该 attention 拓扑，门控邻域特征只统计这些已有边中当前距离≤50m 的邻居；未新增、删除或重连 attention edges。这一口径在正式训练前记录。

## No Future Leakage Audit

纯特征函数只接收 positions[:,:5]、padding_mask[:,:5]、agent_type、已有 current edges；不能接收未来 GT、future displacement、未来标签、decoder output 或 prediction error。十二项测试包括手算历史、padding gap、单观测、无邻居、50m边界、真实输入原样保留、未来坐标/标签/mask扰动下输出一致。Pedestrian GT motion bins 仅用于离线评价和案例选择，未进入 gate 或损失。

## Initialization

从 seed2022 canonical Stage3B 和 Stage4A step0 初始化，所有共同参数/缓冲位级一致，max diff=0；relation step0 与 Stage4A 完全一致，未加载任一 trained checkpoint。Relation final weight/bias=0；gate final weight=0、bias=logit(0.1)，初始 g≈0.1。真实 TRAIN batch 在 CPU single-thread eval 上 raw/logit/prob 最大差分别为 0.0, 0.0, 0.0，均小于1e-6。

10次独立优化诊断：第一步 relation final gradient>0，gate gradient=0符合零 relation bias 的链式求导；10步内 relation MLP、pair embedding、gate MLP均获得非零梯度。模型随后丢弃。独立 tiny TRAIN 六完整窗口覆盖11vehicle、12pedestrian、6bicycle targets，历史覆盖moving V、low-motion P、moving P和VP context；1000步后所有三类回归损失下降、关系偏置非零、门控分化、无NaN/Inf。tiny不用于超参数选择，正式训练重新从头初始化。

## Training

Official TRAIN700/VAL150，test未使用；沿用 frozen scene shards，未重新预处理。Th5/Tf12/K6，batch16，embed64，heads8，global3，dropout0.1，LocalEncoder radius50，AdamW weight_decay1e-4。Fixed-scale LR0.001执行5000updates；恢复自己的 warm-up best（global step 5000）model/optimizer/RNG，仅改LR0.0001进入原 learnable-scale Laplace NLL。每500步全VAL，strict overall full-horizon minFDE改善才更新best。NLL执行12500步，最终global 17500，stop reason=patience_5，best global step=15000。

训练源代码 commit：`1bd9485f0c71b95a73bfe4903c5bbc20204c96cc`。Best checkpoint SHA256：`89f2bc886bd5e1cc2a78a05e07c9761d3317f4677237a458ea35450e4f56e110`。Final指标来自重新加载该checkpoint的fresh完整VAL。CUDA原协议scatter非确定性可能改变近似并列mode的Top1 argmax，因此训练/fresh核对对selection指标保持1e-6、Top1报告允许1e-4；阈值在正式训练前注册，未重新选择checkpoint。

## Main Results

| Group | Count | Stage3B_minFDE6 | Stage4A_minFDE6 | Stage4F_minFDE6 | D-B_minFDE6 | D-C_minFDE6 |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 54990 | 1.343260 | 1.335566 | 1.345378 | 0.002118 | 0.009812 |
| vehicle | 42332 | 1.431914 | 1.420021 | 1.430020 | -0.001894 | 0.009999 |
| pedestrian | 12002 | 1.043875 | 1.050126 | 1.061404 | 0.017529 | 0.011278 |
| bicycle | 656 | 1.099908 | 1.107962 | 1.078937 | -0.020972 | -0.029026 |

ADE取 best-FDE mode，不是独立最小ADE；FDE取6mode最小endpoint error；MR为endpoint error>2m；Top1使用最高预测概率；NLL沿用原best-summed-L2 mode。所有窗口等权，full与partial分开。三模型所有85027条identity严格相同，其中54990full、30037partial；scene/sample/instance/node/type/motion/mask/GT trajectory SHA均一致。

| Group | Count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 54990 | 0.669157 | 1.345378 | 0.165303 | 1.269240 | 2.812716 | -0.745558 |
| vehicle | 42332 | 0.709877 | 1.430020 | 0.167864 | 1.435387 | 3.210043 | -0.934765 |
| pedestrian | 12002 | 0.532329 | 1.061404 | 0.157557 | 0.701087 | 1.457921 | -0.082204 |
| bicycle | 656 | 0.544838 | 1.078937 | 0.141768 | 0.942442 | 1.959965 | -0.672463 |
| vehicle.moving | 10461 | 2.329218 | 4.780964 | 0.605105 | 5.003163 | 11.343899 | 1.304375 |
| vehicle.stopped | 5599 | 0.356054 | 0.868827 | 0.085730 | 0.512421 | 1.320898 | -1.267903 |
| vehicle.parked | 25198 | 0.110951 | 0.148961 | 0.004643 | 0.156879 | 0.248900 | -1.788578 |
| unknown | 1074 | 0.833608 | 1.772687 | 0.166667 | 1.492220 | 3.306844 | -0.975763 |

## Vehicle Motion

| Group | Count | Stage3B_minFDE6 | Stage4A_minFDE6 | Stage4F_minFDE6 | D-B_minFDE6 | D-C_minFDE6 |
| --- | --- | --- | --- | --- | --- | --- |
| vehicle.moving | 10461 | 4.791532 | 4.727113 | 4.780964 | -0.010568 | 0.053851 |
| vehicle.stopped | 5599 | 0.868888 | 0.864945 | 0.868827 | -0.000061 | 0.003882 |
| vehicle.parked | 25198 | 0.146415 | 0.155541 | 0.148961 | 0.002546 | -0.006580 |
| unknown | 1074 | 1.803797 | 1.768961 | 1.772687 | -0.031110 | 0.003727 |
| Vehicle >5m | 9744 | 5.579887 | 5.516344 | 5.562417 | -0.017469 | 0.046074 |

## Pedestrian Motion Bins

| Group | Count | Stage3B_minFDE6 | Stage4A_minFDE6 | Stage4F_minFDE6 | D-B_minFDE6 | D-C_minFDE6 |
| --- | --- | --- | --- | --- | --- | --- |
| Pedestrian 0-1m | 3192 | 0.167310 | 0.177058 | 0.173766 | 0.006456 | -0.003291 |
| Pedestrian 1-2m | 313 | 0.975018 | 1.021475 | 1.041790 | 0.066773 | 0.020315 |
| Pedestrian 2-5m | 916 | 1.667979 | 1.732733 | 1.746655 | 0.078677 | 0.013922 |
| Pedestrian 5-10m | 7321 | 1.312738 | 1.308181 | 1.327248 | 0.014511 | 0.019068 |
| Pedestrian 10-20m | 260 | 2.118974 | 2.132128 | 2.082715 | -0.036259 | -0.049413 |

固定为左闭右开区间，不因本轮结果更改。Bicycle >1m/>5m与Pedestrian >1m/>5m见 nontrivial_motion_metrics.csv；稀疏bicycle motion组仅作描述，不宣称强改善。

## Low-Motion Recovery

正式训练前注册 Pedestrian <5m：4421actor-windows、493instances、100scenes；5–10m：7321windows、701instances、110scenes。核心恢复比较是 D−C，同时用D−B约束可靠负迁移。

| Group | Count | Stage3B_minFDE6 | Stage4A_minFDE6 | Stage4F_minFDE6 | D-B_minFDE6 | D-C_minFDE6 |
| --- | --- | --- | --- | --- | --- | --- |
| Pedestrian <5m | 4421 | 0.535423 | 0.559166 | 0.561113 | 0.025690 | 0.001947 |
| Pedestrian 5-10m | 7321 | 1.312738 | 1.308181 | 1.327248 | 0.014511 | 0.019068 |

## Interaction Context

| Group | Count | Stage3B_minFDE6 | Stage4A_minFDE6 | Stage4F_minFDE6 | D-B_minFDE6 | D-C_minFDE6 |
| --- | --- | --- | --- | --- | --- | --- |
| Heterogeneous-20m | 25113 | 1.335789 | 1.325226 | 1.337480 | 0.001691 | 0.012254 |
| Vehicle hetero-20m | 14813 | 1.560940 | 1.539630 | 1.553520 | -0.007420 | 0.013890 |
| Pedestrian hetero-20m | 9732 | 0.999131 | 1.004340 | 1.016575 | 0.017444 | 0.012235 |
| VP-context-20m | 23209 | 1.330427 | 1.318367 | 1.332706 | 0.002278 | 0.014339 |

完全复用冻结的20m interaction ledger：t0当前有效、不同type、非自身、至少一个邻居；VP仅V/P异类context。未使用本轮误差或未来运动改变group定义。空间context不能证明因果交互。

## Bootstrap

Paired scene-cluster percentile bootstrap：150official VAL scenes，1000replicates，seed2022；同一重采样保留整scene的配对actor-window deltas，按window等权池化，95% percentile CI。Primary为Overall D−B minFDE；次要组和bins为预注册/诊断评价，intervals未作多重比较校正。Bootstrap处理scene内窗口相关性，不提供跨训练随机种子的稳定性证据。

| Comparison | Group | Delta | CI95_lower | CI95_upper | Count | Scenes |
| --- | --- | --- | --- | --- | --- | --- |
| D-B | overall | 0.002118 | -0.007173 | 0.009921 | 54990 | 150 |
| D-B | vehicle | -0.001894 | -0.013655 | 0.008315 | 42332 | 150 |
| D-B | pedestrian | 0.017529 | 0.013286 | 0.021809 | 12002 | 121 |
| D-B | vehicle.moving | -0.010568 | -0.057224 | 0.031328 | 10461 | 137 |
| D-B | Vehicle >5m | -0.017469 | -0.066442 | 0.026240 | 9744 | 137 |
| D-B | Pedestrian <5m | 0.025690 | 0.018680 | 0.034034 | 4421 | 100 |
| D-B | Pedestrian 5-10m | 0.014511 | 0.008885 | 0.019157 | 7321 | 110 |
| D-B | Heterogeneous-20m | 0.001691 | -0.011472 | 0.012917 | 25113 | 131 |
| D-B | VP-context-20m | 0.002278 | -0.011362 | 0.014249 | 23209 | 125 |
| D-C | overall | 0.009812 | 0.000276 | 0.019320 | 54990 | 150 |
| D-C | vehicle | 0.009999 | -0.002169 | 0.022509 | 42332 | 150 |
| D-C | pedestrian | 0.011278 | 0.007438 | 0.015322 | 12002 | 121 |
| D-C | vehicle.moving | 0.053851 | 0.009675 | 0.098561 | 10461 | 137 |
| D-C | Vehicle >5m | 0.046074 | 0.003038 | 0.091256 | 9744 | 137 |
| D-C | Pedestrian <5m | 0.001947 | -0.004101 | 0.008808 | 4421 | 100 |
| D-C | Pedestrian 5-10m | 0.019068 | 0.016012 | 0.022533 | 7321 | 110 |
| D-C | Heterogeneous-20m | 0.012254 | 0.003812 | 0.021942 | 25113 | 131 |
| D-C | VP-context-20m | 0.014339 | 0.005447 | 0.024282 | 23209 | 125 |

## Gate Behavior

| Group | Count | mean | std | median | p10 | p25 | p75 | p90 | fraction_lt0p1 | fraction_gt0p5 | fraction_gt0p9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 54990 | 0.207093 | 0.068763 | 0.174144 | 0.153374 | 0.162135 | 0.241972 | 0.323274 | 0.000000 | 0.000000 | 0.000000 |
| vehicle | 42332 | 0.201521 | 0.074429 | 0.168033 | 0.151219 | 0.159726 | 0.191385 | 0.342974 | 0.000000 | 0.000000 | 0.000000 |
| pedestrian | 12002 | 0.227093 | 0.037505 | 0.235685 | 0.173509 | 0.189459 | 0.257792 | 0.271365 | 0.000000 | 0.000000 | 0.000000 |
| bicycle | 656 | 0.200720 | 0.061459 | 0.176656 | 0.159364 | 0.165613 | 0.193074 | 0.317038 | 0.000000 | 0.000000 | 0.000000 |
| vehicle.moving | 10461 | 0.308032 | 0.076928 | 0.323874 | 0.183366 | 0.253684 | 0.369053 | 0.398331 | 0.000000 | 0.000000 | 0.000000 |
| vehicle.stopped | 5599 | 0.166808 | 0.022663 | 0.164226 | 0.147006 | 0.156709 | 0.170350 | 0.187143 | 0.000000 | 0.000000 | 0.000000 |
| vehicle.parked | 25198 | 0.165101 | 0.015094 | 0.164316 | 0.149566 | 0.157183 | 0.172020 | 0.180642 | 0.000000 | 0.000000 | 0.000000 |
| Pedestrian 0-1m | 3192 | 0.181326 | 0.014735 | 0.179497 | 0.165821 | 0.171034 | 0.188453 | 0.198166 | 0.000000 | 0.000000 | 0.000000 |
| Pedestrian 1-2m | 313 | 0.200742 | 0.027393 | 0.191990 | 0.171732 | 0.179925 | 0.222787 | 0.238207 | 0.000000 | 0.000000 | 0.000000 |
| Pedestrian 2-5m | 916 | 0.217466 | 0.029396 | 0.222319 | 0.176711 | 0.193786 | 0.238180 | 0.249886 | 0.000000 | 0.000000 | 0.000000 |
| Pedestrian 5-10m | 7321 | 0.247918 | 0.024317 | 0.251213 | 0.215001 | 0.235066 | 0.265134 | 0.275557 | 0.000000 | 0.000000 | 0.000000 |
| Pedestrian 10-20m | 260 | 0.268230 | 0.029778 | 0.265915 | 0.231004 | 0.247924 | 0.287934 | 0.311539 | 0.000000 | 0.000000 | 0.000000 |
| Heterogeneous-20m | 25113 | 0.218834 | 0.066998 | 0.186514 | 0.161604 | 0.168893 | 0.255709 | 0.321161 | 0.000000 | 0.000000 | 0.000000 |
| non-heterogeneous | 29877 | 0.197223 | 0.068676 | 0.166512 | 0.148708 | 0.157405 | 0.201109 | 0.324478 | 0.000000 | 0.000000 | 0.000000 |

Full-horizon mean gate：moving/stopped/parked vehicle=0.308032/0.166808/0.165101；P0–1/P5–10=0.181326/0.247918。这些真实描述统计不强制符合预期，不构成“门控恢复误差”的因果证明。

Gate collapse=NO；定义为all-current-valid actor-window observations中>95%落在g<0.05或g>0.95。全context、full和partial统计分别记录，未人为改变gate。Spearman对六个可观测历史/邻域量计算，见 gate_feature_correlations.csv；log1p及/50为单调变换，不改变对应raw量的rank关联。仅描述相关，不解释为因果，重叠窗口的nominal p-value不作为独立样本显著性证据。

## Efficiency

| Model | parameters | mean_inference_ms | median_inference_ms | peak_CUDA_memory_MiB |
| --- | --- | --- | --- | --- |
| Stage3B | 646001 | 56.683776 | 53.278864 | 1735.671387 |
| Stage4A | 647609 | 57.431392 | 53.502321 | 1735.678711 |
| Stage4F | 647786 | 59.520856 | 55.425232 | 1735.681152 |

同一GPU相同batch，warm20后执行500paired triplets，循环B/C/D全部六种执行顺序；CUDA events计时forward，排除I/O/H2D和外部input clone，所有模型内部clone均保留。Peak memory由各模型单独在GPU、独立warm/reset、遍历相同226unique VAL batches测量。D相对B mean latency overhead=5.005%，相对C=3.638%。

## Qualitative

| name | case_kind | gate | B_FDE | C_FDE | D_FDE |
| --- | --- | --- | --- | --- | --- |
| moving_vehicle_retention | Moving vehicle: retained gain | 0.208200 | 9.547395 | 2.878692 | 4.070557 |
| low_motion_pedestrian_recovery | Low-motion pedestrian: recovery example | 0.213167 | 2.283807 | 3.627281 | 1.512508 |
| moving_pedestrian | Moving pedestrian: learned interaction gate | 0.307755 | 5.342906 | 5.285436 | 5.328465 |
| degradation | Real degradation relative to Stage3B | 0.187693 | 3.816771 | 7.750455 | 11.967960 |

四个真实例子展示moving vehicle保留收益、low-motion pedestrian恢复、moving pedestrian和真实退化。三列同actor、相同GT/坐标范围；灰圆history、黑方GT、蓝圆Best、橙虚线三角Top1，未来每条12原始marker。每个模式/指标从同一冻结VAL batch replay核对，未插值、平滑、修改轨迹。案例在定量评价锁定后按效果选择，仅作机制说明；总体判断由完整数据bootstrap决定。

## Scientific Decision

Interaction Necessity Gate = **NOT SUPPORTED**。Stage4F = **PASS**（技术完成与科学支持分开）。Ready Reliability Head = **NO**。

判定事实：Overall reliable gain=False；Pedestrian reliable harm=True；major-class harm guard=True；moving vehicle reliable gain=False；P<5相对C恢复=False、可靠恢复=False；P<5相对B无可靠损害=False。

遵循单一seed2022筛选协议，没有超参数搜索或额外seed，没有进入Reliability Head、mode re-ranking或任何第二创新。即使Ready=YES，也须后续研究决策再决定稳定性seed实验；本轮在Stage4F完成后STOP。

## Artifacts

所有新增代码、报告、图表、checkpoint和本地逐actor CSV位于独立Stage4F根目录，Stage3与旧Stage4只读。逐actor误差/gate、checkpoint、每500步完整scene metrics及日志保留本地，Git只存其SHA/规模和适合版本管理的脚本、最终小指标、表图报告。artifact_manifest.json为完整local inventory，self hash除外。
