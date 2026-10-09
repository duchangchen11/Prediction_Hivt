# Stage11C Error-Aware Ranking Probability Calibration

**Stage11C — OOF Development Evidence。CalibrationEngineering=PASS；CalibrationUseful=YES；ReadyForSemanticStudy=YES；Stage11C=GO。**

Vehicle/Pedestrian 的六候选 oracle-mode NLL、Brier 和 ECE 均降低，Top1 选择和几何误差完全不变。温度缩放同时增加 ExpectedRegret，因此这个结论只支持指定 oracle-mode 分类概率指标的改善。Stage11C 科学判定为 GO 时，本次执行仍在本阶段结束后 STOP，等待大脑 AI 审查。

## 冻结与数据边界

基础 commit：`465d3541192303e43809dc553c12e445225ab228`。分支：`stage11c/ranking-probability-calibration`。全部产物单独位于 `outputs/stage11c_probability_calibration/`。

Stage11B 的 12 个 checkpoint 与原 manifest SHA256 全部一致，另外 8 个历史依赖 checkpoint 保持冻结。2551 个历史版本文件及 5 个 Stage2C 未提交文件保持原样。冻结划分、对应折标准化和候选数组；没有训练、fine-tune、修改权重、新 checkpoint、重划分或候选变更。

只访问 HeadTrain630 的原始三折数据：每折 InnerTrain378 / InnerDev42 / OuterTest210；OOF 合并 260,151 个 full-horizon actor/window，其中 Vehicle 191,026、Pedestrian 66,145、Bicycle 2,980。没有访问 HeadDev70、official VAL150 或 test。共享源缓存包含历史 HeadDev 行，但本阶段访问器仅索引已冻结 HeadTrain actor 身份；拟合访问器进一步限制为对应折 InnerDev。

**InnerDev 已用于 Stage11B checkpoint 选择，本阶段再用于温度拟合。Stage11B OOF 结果此前已被科研人员查看；因此以下 OOF 均是新增方案的开发阶段评价，不能声称独立测试。**

## 预注册校准协议与三折参数

仅注册一个正式方案：每折一个全局标量 T，Vehicle/Pedestrian 共用；Bicycle 原始 FoldR2 直通，不参与拟合。固定目标为 `0.5 mean_V NLL + 0.5 mean_P NLL`，oracle 标签为六条冻结候选中 FDE 最小模式，平局取最低 index。未来误差仅作为拟合标签和离线指标，从未进入图网络输入或推理温度选择。

FP64，在 log(T) 空间使用 SciPy 1.14.1 bounded scalar minimization；范围 [1,1000]、`xatol=1e-10`、`maxiter=1000`，显式评价两端点，最小目标精确平局选择较小 T。协议在拟合前登记，SHA256=`e36cfe8de4ca0b8e40bf2a8b3c1df8691b592b2bbdca33cedd94ecd3bbe5a0da`。没有修改区间、type-specific/per-mode 参数、额外 calibrator 或任何 OOF 参数选择。

| Fold | Temperature | FitActors | VehicleCount | PedestrianCount | NLLRawMacro | NLLCalibratedMacro | Boundary |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 93.474591 | 18005 | 12401 | 5604 | 37.680413 | 1.601027 | INTERIOR |
| 2 | 41.893114 | 18782 | 14245 | 4537 | 18.936362 | 1.534788 | INTERIOR |
| 3 | 77.856698 | 17255 | 13644 | 3611 | 32.717983 | 1.545367 | INTERIOR |

三个最优 T 均为合法内部解。冻结 C 的 InnerDev checkpoint 选择分数逐折精确复现；模型 eval、requires_grad=False、inference_mode，无权重梯度或优化器。拟合进程禁止打开 Stage11B OOF cache，访问记录验证 OOF 读取次数为 0。独立 Torch FP64 复算拟合 NLL，并利用 NLL 对 inverse temperature 的凸性检查一阶最优条件；未搜索另一套温度。

## Top1 / 候选 / Bicycle 保持性

全体 260,151 个 actor/window：raw probability、raw logits、calibrated probability 与 calibrated logits 的 argmax 一致，改变模式数为 0；最低 index 平局规则保持。

候选坐标、Top1FDE、Top1ADE、minFDE6、minADEOracle6、MR6 全部逐 actor 严格一致，最大差异为 0。候选始终引用相同冻结坐标源，逐 actor 对照并验证源哈希不变。零几何变化不是新增预测收益。

Bicycle 的 6 candidates、6 logits、6 probabilities、Top1 mode 和 Top1FDE 与 C Raw 及原 FoldR2 逐位一致。V/P 校准结果保留 FP64，Bicycle 序列化为单独的原始 FP32 block，以免 dtype 转换破坏逐位要求。由观测 actor type 路由，未来位移不参与此规则。

## 核心概率指标

NLL 使用稳定的 FP64 log_softmax / shifted logsumexp，无概率裁剪。Brier 为六类 `sum_k (p_k-onehot_k*)²`，不除以 6。ECE 使用预注册 15 个等宽 [0,1] 分箱，左闭右开，最后一箱包括 1，空箱贡献 0，actor/window 数加权。ECE 事件是“Top1 是否等于冻结六候选的 FDE oracle mode”。原始概率直接复用 Stage11B FP32 值，精确提升为 FP64 做统计；稳定 logits NLL 不受原概率下溢影响。

| Group | Model | Count | Top1FDE | OracleBestModeNLL | BrierScore | Top1ModeECE |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | C Raw | 260151 | 2.199512 | 25.198422 | 1.107536 | 0.540848 |
| Vehicle | C Raw | 191026 | 2.516273 | 22.549819 | 1.053766 | 0.517152 |
| Pedestrian | C Raw | 66145 | 1.304968 | 33.915948 | 1.278961 | 0.626255 |
| Bicycle | C Raw | 2980 | 1.749867 | 1.484037 | 0.749308 | 0.165552 |
| Overall | C Calibrated | 260151 | 2.199512 | 1.502695 | 0.734817 | 0.097591 |
| Vehicle | C Calibrated | 191026 | 2.516273 | 1.443814 | 0.716494 | 0.124022 |
| Pedestrian | C Calibrated | 66145 | 1.304968 | 1.673582 | 0.787080 | 0.019813 |
| Bicycle | C Calibrated | 2980 | 1.749867 | 1.484037 | 0.749308 | 0.165552 |

## 全部冻结对照及辅助概率质量

SoftCE 固定 `q=softmax(-FDE/1m)`；ExpectedFDE=`sum p*FDE`；ExpectedRegret=`sum p*(FDE-minFDE)`；NormalizedExpectedRegret 除以 `max(1m,mean six costs)`，复用 Stage11B 定义。原有对照全部保持；此处按相同定义以 FP64 重算概率诊断，与 Stage11B 部分 FP32 辅助指标可能存在舍入级差异。

| Group | Model | Top1ADE | Top1FDE | OracleBestModeNLL | BrierScore | Top1ModeECE | SoftCE | ExpectedFDE | ExpectedRegret | NormalizedExpectedRegret |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | R0 | 1.063395 | 2.362940 | 1.566150 | 0.773482 | 0.027885 | 1.584828 | 3.171695 | 1.913560 | 0.428148 |
| Vehicle | R0 | 1.216739 | 2.733575 | 1.525322 | 0.763915 | 0.008965 | 1.542397 | 3.648897 | 2.267304 | 0.353622 |
| Pedestrian | R0 | 0.632193 | 1.320020 | 1.684493 | 0.801291 | 0.112422 | 1.707686 | 1.820124 | 0.912176 | 0.643200 |
| Bicycle | R0 | 0.804730 | 1.753224 | 1.556546 | 0.769465 | 0.186156 | 1.577778 | 2.581627 | 1.464665 | 0.432096 |
| Overall | FoldR2 | 1.054015 | 2.344301 | 1.507983 | 0.758665 | 0.021971 | 1.551431 | 2.857571 | 1.599436 | 0.326051 |
| Vehicle | FoldR2 | 1.203753 | 2.708376 | 1.462888 | 0.748585 | 0.040337 | 1.504876 | 3.282946 | 1.901352 | 0.252500 |
| Pedestrian | FoldR2 | 0.632774 | 1.319636 | 1.639295 | 0.788200 | 0.090100 | 1.686447 | 1.656075 | 0.748127 | 0.539076 |
| Bicycle | FoldR2 | 0.805404 | 1.749867 | 1.484037 | 0.749308 | 0.165552 | 1.538873 | 2.258680 | 1.141718 | 0.312542 |
| Overall | A SoftCE | 1.089466 | 2.384061 | 1.485327 | 0.752533 | 0.114954 | 1.535392 | 2.754936 | 1.496802 | 0.309241 |
| Vehicle | A SoftCE | 1.244684 | 2.751081 | 1.439409 | 0.741958 | 0.131538 | 1.487497 | 3.159333 | 1.777739 | 0.242058 |
| Pedestrian | A SoftCE | 0.653994 | 1.352684 | 1.617999 | 0.783217 | 0.065726 | 1.673556 | 1.609402 | 0.701454 | 0.503115 |
| Bicycle | A SoftCE | 0.805404 | 1.749867 | 1.484037 | 0.749308 | 0.165552 | 1.538873 | 2.258680 | 1.141718 | 0.312542 |
| Overall | B HardCE | 1.065420 | 2.367404 | 1.340948 | 0.662291 | 0.021832 | 1.736723 | 2.842863 | 1.584728 | 0.314332 |
| Vehicle | B HardCE | 1.219045 | 2.738890 | 1.248393 | 0.624121 | 0.028013 | 1.740455 | 3.274946 | 1.893352 | 0.245952 |
| Pedestrian | B HardCE | 0.633465 | 1.322379 | 1.601800 | 0.768603 | 0.016209 | 1.734861 | 1.621331 | 0.713382 | 0.511893 |
| Bicycle | B HardCE | 0.805404 | 1.749867 | 1.484037 | 0.749308 | 0.165552 | 1.538873 | 2.258680 | 1.141718 | 0.312542 |
| Overall | C Raw | 0.973628 | 2.199512 | 25.198422 | 1.107536 | 0.540848 | 36.790614 | 2.206379 | 0.948244 | 0.161491 |
| Vehicle | C Raw | 1.096568 | 2.516273 | 22.549819 | 1.053766 | 0.517152 | 35.253640 | 2.517360 | 1.135766 | 0.127363 |
| Pedestrian | C Raw | 0.626157 | 1.304968 | 33.915948 | 1.278961 | 0.626255 | 42.817559 | 1.305916 | 0.397968 | 0.253248 |
| Bicycle | C Raw | 0.805404 | 1.749867 | 1.484037 | 0.749308 | 0.165552 | 1.538873 | 2.258680 | 1.141718 | 0.312542 |
| Overall | C Calibrated | 0.973628 | 2.199512 | 1.502695 | 0.734817 | 0.097591 | 1.667322 | 3.154842 | 1.896707 | 0.431724 |
| Vehicle | C Calibrated | 1.096568 | 2.516273 | 1.443814 | 0.716494 | 0.124022 | 1.622968 | 3.691871 | 2.310278 | 0.410851 |
| Pedestrian | C Calibrated | 0.626157 | 1.304968 | 1.673582 | 0.787080 | 0.019813 | 1.801202 | 1.644281 | 0.736332 | 0.497375 |
| Bicycle | C Calibrated | 0.805404 | 1.749867 | 1.484037 | 0.749308 | 0.165552 | 1.538873 | 2.258680 | 1.141718 | 0.312542 |

校准把车辆 ExpectedRegret 从 1.135766 提高到 2.310278 m；行人从 0.397968 提高到 0.736332 m。两类 scene bootstrap 的该指标 delta CI 都大于 0。降低过度自信会增加低排名候选的权重；oracle 分类 NLL 与几何期望损失目标不同。本阶段没有以牺牲 Top1 误差换取 NLL 收益，Top1 始终严格不变；但不能声称所有概率质量指标均改善。

## 概率集中度与可靠性

| Group | Model | PredictionEntropy | Top1Probability | Top1ProbabilityP50 | Top1ProbabilityP90 | Top1ProbabilityP99 | FractionAbove0.9 | FractionAbove0.99 | OracleTop1MatchRate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vehicle | C Raw | 0.074136 | 0.969143 | 0.999987 | 1.000000 | 1.000000 | 0.902422 | 0.793892 | 0.451991 |
| Pedestrian | C Raw | 0.110419 | 0.954262 | 0.999618 | 1.000000 | 1.000000 | 0.854033 | 0.698556 | 0.328007 |
| Vehicle | C Calibrated | 1.559324 | 0.333108 | 0.326708 | 0.396458 | 0.514183 | 0.000000 | 0.000000 | 0.451991 |
| Pedestrian | C Calibrated | 1.567238 | 0.313382 | 0.306354 | 0.376727 | 0.482393 | 0.000000 | 0.000000 | 0.328007 |

两类 mean pmax 和 pmax>0.99 占比均下降。温度缩放降低了原模型 C 的极端自信，但并未消除全部校准偏差：Vehicle 校准后 mean pmax=0.333108，oracle Top1 match rate=0.451991，合并均值表现为偏低自信。全局 macro T 不保证每个类型、fold 或运动子组都达到理想可靠性。15-bin ECE 只是指定分箱下的描述统计，不能替代完整 reliability 分布。

## 2000 次 paired whole-scene bootstrap

seed2022，每次分别在三个 outer210 fold 内有放回抽取 210 个完整场景，合并为 630 场景；同一 actor/window 的 raw/cal 配对。Stage11B 场景顺序和 replicate weights 已逐位重现。按被抽取场景 actor 数计算比值；ECE 每次从 scene/bin confidence 与 correct-count 重新计算非线性绝对差，未对 actor ECE 值简单平均。直接 actor 加权重算检验通过。

| Group | Metric | C_Raw | C_Calibrated | DeltaCalMinusRaw | CI95Low | CI95High |
| --- | --- | --- | --- | --- | --- | --- |
| Vehicle | OracleBestModeNLL | 22.549819 | 1.443814 | -21.106005 | -22.145722 | -20.108106 |
| Vehicle | BrierScore | 1.053766 | 0.716494 | -0.337272 | -0.363783 | -0.308887 |
| Vehicle | ExpectedRegret | 1.135766 | 2.310278 | 1.174512 | 1.151065 | 1.198418 |
| Vehicle | Top1ModeECE | 0.517152 | 0.124022 | -0.393130 | -0.424727 | -0.358684 |
| Pedestrian | OracleBestModeNLL | 33.915948 | 1.673582 | -32.242367 | -33.929571 | -30.629673 |
| Pedestrian | BrierScore | 1.278961 | 0.787080 | -0.491882 | -0.513025 | -0.469702 |
| Pedestrian | ExpectedRegret | 0.397968 | 0.736332 | 0.338364 | 0.318401 | 0.360076 |
| Pedestrian | Top1ModeECE | 0.626255 | 0.019813 | -0.606443 | -0.629288 | -0.578014 |

95% 区间为 2.5/97.5 percentile 描述性区间；固定已选择 checkpoint 和已拟合 T，没有重新训练、重新选择 checkpoint 或 bootstrap refit T，所以不包含模型选择和温度拟合的不确定性。ECE 区间还依赖固定 15-bin 定义，不是独立测试或多重检验控制的确认性结论。Top1ADE/FDE 差严格为零，不包装为新收益。

## 三折与探索性运动子组

| Fold | Group | Model | Count | OracleBestModeNLL | BrierScore | Top1ModeECE | ExpectedRegret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Vehicle | C Raw | 61556 | 25.565838 | 1.059248 | 0.521677 | 1.072411 |
| 2 | Vehicle | C Raw | 60451 | 17.601500 | 1.096559 | 0.534000 | 1.174652 |
| 3 | Vehicle | C Raw | 69019 | 24.193958 | 1.011396 | 0.498360 | 1.158212 |
| 1 | Pedestrian | C Raw | 20250 | 38.094102 | 1.269708 | 0.623761 | 0.380044 |
| 2 | Pedestrian | C Raw | 22108 | 24.705733 | 1.308627 | 0.640546 | 0.438189 |
| 3 | Pedestrian | C Raw | 23787 | 38.919178 | 1.259267 | 0.615097 | 0.375845 |
| 1 | Vehicle | C Calibrated | 61556 | 1.441801 | 0.720269 | 0.138765 | 2.235622 |
| 2 | Vehicle | C Calibrated | 60451 | 1.483686 | 0.731250 | 0.102551 | 2.391359 |
| 3 | Vehicle | C Calibrated | 69019 | 1.410687 | 0.700203 | 0.143941 | 2.305844 |
| 1 | Pedestrian | C Calibrated | 20250 | 1.651687 | 0.781322 | 0.048503 | 0.863803 |
| 2 | Pedestrian | C Calibrated | 22108 | 1.695833 | 0.794214 | 0.031561 | 0.687921 |
| 3 | Pedestrian | C Calibrated | 23787 | 1.671541 | 0.785351 | 0.033202 | 0.672810 |

| Group | Model | Count | OracleBestModeNLL | BrierScore | Top1ModeECE | ExpectedRegret |
| --- | --- | --- | --- | --- | --- | --- |
| MovingVehicle | C Raw | 41728 | 45.180161 | 1.488824 | 0.732839 | 4.584073 |
| StoppedVehicle | C Raw | 23044 | 21.794692 | 1.027969 | 0.503596 | 0.425061 |
| ParkedVehicle | C Raw | 121939 | 15.013817 | 0.910470 | 0.446160 | 0.092352 |
| Vehicle>5m | C Raw | 38241 | 48.876261 | 1.512407 | 0.744000 | 5.002962 |
| Pedestrian<5m | C Raw | 24005 | 32.069901 | 1.123228 | 0.554776 | 0.415491 |
| Pedestrian>5m | C Raw | 42140 | 34.967547 | 1.367675 | 0.666973 | 0.387986 |
| Pedestrian5-8m | C Raw | 22483 | 29.094819 | 1.303741 | 0.635291 | 0.379424 |
| MovingVehicle | C Calibrated | 41728 | 1.837468 | 0.842330 | 0.089319 | 6.262784 |
| StoppedVehicle | C Calibrated | 23044 | 1.429123 | 0.713823 | 0.138469 | 1.379074 |
| ParkedVehicle | C Calibrated | 121939 | 1.312726 | 0.674279 | 0.190143 | 1.134792 |
| Vehicle>5m | C Calibrated | 38241 | 1.891385 | 0.856253 | 0.098618 | 6.702947 |
| Pedestrian<5m | C Calibrated | 24005 | 1.570979 | 0.750859 | 0.093691 | 0.948139 |
| Pedestrian>5m | C Calibrated | 42140 | 1.732030 | 0.807713 | 0.031820 | 0.615677 |
| Pedestrian5-8m | C Calibrated | 22483 | 1.634258 | 0.783380 | 0.038599 | 0.604184 |

Moving/Stopped/Parked 使用原 motion_state 标签；未标记这些状态的 Vehicle 仍计入总体车辆。GT 位移组只做离线探索：Pedestrian<5m 为 <5；>5m 为 >5；5-8m 复用原定义 [5,8)。所有 V/P 在同一 fold 使用相同 T，不根据 GT 位移选择温度。探索性结果不调整正式参数或目标。

## 独立验证与图表

独立 Torch FP64 完整重算全部六模型、260,151 actor 的 14 项逐 actor 指标，并核对全部分组均值与 66 项 group ECE；最大数值差低于 1e-10。所有 logits、概率和诊断值 finite；V/P 校准概率和误差 <1e-12，原始 FP32 概率和误差保留原舍入精度。全历史文件及 checkpoint 的结尾哈希复核通过。

五类图均带 **Stage11C — OOF Development Evidence** 标记，各导出 PNG/SVG/PDF：

1. [Probability distribution](../06_figures/stage11c_probability_distribution.png)：pmax 及所有六候选概率分布。
2. [Vehicle reliability](../06_figures/stage11c_vehicle_reliability.png)：oracle 事件可靠性及分箱样本分布。
3. [Pedestrian reliability](../06_figures/stage11c_pedestrian_reliability.png)：相同分箱。
4. [NLL before/after](../06_figures/stage11c_oracle_mode_nll.png)。
5. [Top1FDE identity](../06_figures/stage11c_top1_fde_identity.png)：全量点与零差审计。

完整对照表在 `07_tables/stage11c_oof_probability_metrics.csv`，type 表在 `stage11c_type_metrics.csv`，bootstrap 表在 `04_bootstrap/stage11c_bootstrap_ci.csv`；分箱原始计数和概率集中度表位于 `05_probability_diagnostics/`。完整 actor 输出、数组和运行日志本地保留，不上传 Git；代码、预注册、小型表格和报告提交到阶段分支。

## 科学决定与执行边界

| Item | Value |
| --- | --- |
| FrozenModelIntegrity | PASS |
| TemperatureFitIsolation | PASS |
| Top1Identity | PASS |
| BicyclePreserved | YES |
| CalibrationEngineering | PASS |
| VehicleNLLImproved | YES |
| PedestrianNLLImproved | YES |
| ProbabilityConcentrationReduced | YES |
| CalibrationUseful | YES |
| ReadyForSemanticStudy | YES |
| Stage11C | GO |

判断严格依照拟合前登记规则：T 合法；V/P NLL 均降低；两类 Brier 没有同时明显恶化，实际两类均改善；ECE 与集中度完整报告；Top1 严格不变。ExpectedRegret 恶化已完整披露，按照预注册规则不单独否决 oracle-event 校准有效性。

这些概率仅对应六个固定候选之间的 FDE-oracle-mode 分类事件，GT 只有单条观测未来，不能视作真实世界轨迹、过街意图、碰撞或风险概率。ReadyForSemanticStudy=YES 不意味着真实风险概率已经校准，也不构成下一阶段授权。

**本阶段结束后 STOP，等待大脑 AI 审查。未启动 Stage12、Semantic Graph、official VAL/test 评价或任何新训练。**
