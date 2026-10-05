# Stage4A-E：Motion-Bin + Class Gradient Conflict Diagnostic

MotionPattern=**LOW_MOTION_DEGRADATION**。GradientConflict=**WEAK**。RecommendedArchitecture=**Interaction Necessity Gate**（决策矩阵 CASE 3，仅建议）。

【Frozen Inputs】

本轮从 `f9619f7a442933cefd9f5bdf9704dff1212b0ab0` 建立 `stage4a-e/motion-gradient-diagnostic`，新增文件均保存在 Stage4 独立目录并使用 `stage4ae_` 前缀。Stage4A、Stage4A-D、Stage3B checkpoint、误差表、membership 及旧 Stage2C 重绘文件在执行前后保持哈希一致；没有更改旧分支、原始数据或环境，没有 merge main。

Stage3B checkpoint SHA256：`461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547`。
Stage4A checkpoint SHA256：`ff25266ba6a44630cfec01ae1596ef7de05f0d71f06f9676f6ea7ef327775ecb`。

Part A 直接读取已经通过 pairing audit 的冻结 Stage3B/Stage4A actor errors，严格核对 54,990 个 full-horizon actor-window 的身份、type、GT SHA、future mask、motion state、endpoint displacement；未做任何新 VAL 推理、target 重定义或 membership 计算。12,002 个 full-horizon pedestrian 的固定计数为 **3192+313+916+7321+260=12002**，区间全部采用 `[lower,upper)`。

【Pedestrian Motion-Bin Results】

| MotionBin | Count | UniqueInstances | UniqueScenes | Stage3B_minFDE6 | Stage4A_minFDE6 | Delta_minFDE6 | CI95Lower | CI95Upper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0-1m | 3192 | 352 | 86 | 0.167310 | 0.177058 | 0.009747 | 0.005835 | 0.013216 |
| 1-2m | 313 | 95 | 49 | 0.975018 | 1.021475 | 0.046457 | 0.019023 | 0.083031 |
| 2-5m | 916 | 176 | 65 | 1.667979 | 1.732733 | 0.064754 | 0.039939 | 0.094398 |
| 5-10m | 7321 | 701 | 110 | 1.312738 | 1.308181 | -0.004557 | -0.009868 | -0.000159 |
| 10-20m | 260 | 41 | 29 | 2.118974 | 2.132128 | 0.013154 | -0.060548 | 0.089392 |

完整 minADE6、minFDE6、MR6、Top1ADE6、Top1FDE6、NLL 及 C−B 差值见 `stage4ae_pedestrian_motion_bin_metrics.csv`；minFDE、Top1FDE、minADE 的 paired scene-bootstrap CI 见对应 bootstrap CSV/JSON。统一从 official VAL150 场景有放回重采样，1000 replicates，seed2022；每个 replicate 对同一场景中的 B/C 差值配对并保持 actor-window pooling。所有 bin 均有 1000 个有效 replicates。

**Q1：YES。** 0–1m、1–2m、2–5m 的 minFDE 差值都为正，三个 scene-bootstrap 区间都在零以上。低于5m 的加权贡献为 **+0.008746 m**，大于净 overall 退化 **+0.006251 m**，因为 ≥5m 的合计贡献为 **-0.002495 m**、抵消了部分低运动退化。这是 count×delta 的严格加权归因，不是独立 actor 因果归因。

**Q2：5–10m 改善；10–20m 无明确方向。** 5–10m 占 7321 个 actor-windows，FDE -0.004557 m，CI 上界仅略低于零，改善幅度小。10–20m 的点估计为 +0.013154 m，但 CI 跨零；260 个 actor-windows 仅来自29个 scene、41个 instance，不能据此声称高位移均改善或可靠退化。1–2m 也只有313个 actor-windows、49个 scene、95个 instance，其正区间作为离线证据保留，避免过强外推。

low-motion degradation evidence=**YES**。本轮主要对 minFDE 作运动模式判断；其他指标不全部同向，例如2–5m minADE 小幅改善、FDE 退化。不得把这一诊断写成所有误差指标均恶化。固定五个 bin 的 CI 未作多重比较校正，结论保持探索性机制诊断范围。

【Motion × Interaction Context】

| MotionBin | Context | Count | UniqueScenes | Stage3B_FDE | Stage4A_FDE | DeltaFDE | Top1FDEDelta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-1m | hetero-20m | 2833 | 80 | 0.169365 | 0.178691 | 0.009326 | 0.018088 |
| 0-1m | non-hetero-20m | 359 | 34 | 0.151097 | 0.164169 | 0.013072 | -0.002595 |
| 1-2m | hetero-20m | 275 | 44 | 0.987241 | 1.038212 | 0.050971 | 0.227083 |
| 1-2m | non-hetero-20m | 38 | 10 | 0.886562 | 0.900353 | 0.013791 | 0.120566 |
| 2-5m | hetero-20m | 804 | 61 | 1.579307 | 1.642216 | 0.062910 | 0.110090 |
| 2-5m | non-hetero-20m | 112 | 22 | 2.304515 | 2.382514 | 0.077998 | 0.194218 |
| 5-10m | hetero-20m | 5620 | 105 | 1.289349 | 1.282098 | -0.007251 | -0.063760 |
| 5-10m | non-hetero-20m | 1701 | 77 | 1.390012 | 1.394354 | 0.004342 | -0.030997 |
| 10-20m | hetero-20m | 200 | 26 | 2.281665 | 2.283813 | 0.002147 | 0.096335 |
| 10-20m | non-hetero-20m | 60 | 10 | 1.576667 | 1.626512 | 0.049845 | 0.032330 |

| Context | Count | ActorShare | MeanDeltaFDE | ContributionToOverallPedestrianDelta | FractionOfNetDelta |
| --- | --- | --- | --- | --- | --- |
| hetero-20m | 9732 | 0.810865 | 0.005209 | 0.004224 | 0.675716 |
| non-hetero-20m | 2270 | 0.189135 | 0.010718 | 0.002027 | 0.324284 |

**Q3：没有 hetero 专属退化模式。** 在五个固定 bin 的交叉表中，三个低于5m 的 bin 均在 hetero 和 non-hetero context 出现 FDE 正差。hetero 占约81.1% 的 pedestrian 样本，贡献约67.6%的加权净退化；non-hetero 虽只有约18.9%的样本，其人均 delta 更大。5–10m hetero 改善而 non-hetero 小幅变差，也表明 context 与 motion 的关系并非统一。所有交叉 cell 只作描述，不在38个或60个 actor-windows、10个 scene 的 cell 上宣称显著性。

【Stage3B Gradient Conflict】

| ParameterGroup | ValidBatches | MeanCosVP | MedianCosVP | P10CosVP | P25CosVP | P75CosVP | P90CosVP | ConflictRate | StrongConflictRate_0.1 | StrongConflictRate_0.25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LocalEncoder | 100 | 0.202180 | 0.218398 | -0.675842 | -0.118719 | 0.692736 | 0.897404 | 0.280000 | 0.260000 | 0.220000 |
| TypeEmbedding | 100 | 0.000598 | 0.000852 | -0.018423 | -0.004536 | 0.005905 | 0.012860 | 0.460000 | 0.000000 | 0.000000 |
| GlobalInteractor | 100 | 0.129798 | 0.098906 | -0.120966 | -0.037703 | 0.277040 | 0.491140 | 0.320000 | 0.150000 | 0.060000 |
| Decoder | 100 | 0.363302 | 0.391357 | -0.054187 | 0.137189 | 0.632236 | 0.773972 | 0.170000 | 0.090000 | 0.040000 |
| ALL SHARED | 100 | 0.334440 | 0.352208 | -0.097458 | 0.088605 | 0.592430 | 0.744928 | 0.140000 | 0.100000 | 0.040000 |

Stage3B ALL SHARED 平均 cosine=0.334440、median=0.352208、ConflictRate=14.00%，说明原模型已经有部分 V/P 方向冲突。cosine 正的总体均值不排除 batch 层面的负向冲突。

【Stage4A Gradient Conflict】

| ParameterGroup | ValidBatches | MeanCosVP | MedianCosVP | P10CosVP | P25CosVP | P75CosVP | P90CosVP | ConflictRate | StrongConflictRate_0.1 | StrongConflictRate_0.25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LocalEncoder | 100 | 0.125913 | 0.256870 | -0.894865 | -0.555028 | 0.774490 | 0.940955 | 0.420000 | 0.370000 | 0.330000 |
| TypeEmbedding | 100 | -0.000602 | 0.001198 | -0.018328 | -0.006589 | 0.008558 | 0.019107 | 0.460000 | 0.000000 | 0.000000 |
| GlobalInteractor | 100 | 0.188783 | 0.091688 | -0.134408 | -0.012182 | 0.332499 | 0.688784 | 0.260000 | 0.140000 | 0.010000 |
| Decoder | 100 | 0.330037 | 0.377527 | -0.319657 | -0.072472 | 0.782250 | 0.926482 | 0.300000 | 0.220000 | 0.150000 |
| Relation Module | 100 | -0.017062 | -0.031781 | -0.260558 | -0.181853 | 0.088270 | 0.240536 | 0.590000 | 0.360000 | 0.110000 |
| ALL SHARED | 100 | 0.287864 | 0.258844 | -0.244481 | -0.096331 | 0.713079 | 0.885063 | 0.330000 | 0.250000 | 0.100000 |

两个 checkpoint 使用完全相同100个符合条件的 TRAIN batches，每批完整16个 scene windows、所有 context actors、同一输入 tensor fingerprint 和 scene/window identity hash。两个模型分别 `eval()`、关闭 dropout，**每模型每批仅 forward 一次**；在同一 output 上调用原 `recovery_loss(original_nll)`，只将 target_mask 设为 V 或 P，再以 `torch.autograd.grad` 提取两类梯度，未使用全局 `no_grad()`。原 Laplace NLL 对该类有效未来坐标取 mean，原 detached soft-target classification 对该类 eligible actor 取 mean，既没有 class sum，也没有删除其他 context actors。

五个 shared 参数组完整覆盖646001个参数，Stage4A GlobalInteractor shared core 明确排除 pair_embedding/relation_mlp；Relation Module 独立1608个参数。所有主要 group 在两模型都有100个非零 V/P gradient batch，None=0、NaN=0、Inf=0。所有参数/buffer state SHA 在每批之后一致；`autograd.grad` 未填充 parameter.grad。没有 optimizer、step 或 backward，未生成新 checkpoint。

【Stage3B vs Stage4A Gradient Shift】

| ParameterGroup | Stage3BMeanCos | Stage4AMeanCos | DeltaCosMean | DeltaCosMedian | DeltaCosCI95Lower | DeltaCosCI95Upper | FractionDeltaCosBelow0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LocalEncoder | 0.202180 | 0.125913 | -0.076267 | -0.023444 | -0.208007 | 0.055223 | 0.510000 |
| TypeEmbedding | 0.000598 | -0.000602 | -0.001201 | 0.001478 | -0.005437 | 0.002598 | 0.480000 |
| GlobalInteractor | 0.129798 | 0.188783 | 0.058985 | 0.028396 | 0.016924 | 0.099505 | 0.460000 |
| Decoder | 0.363302 | 0.330037 | -0.033264 | 0.011839 | -0.124556 | 0.050693 | 0.480000 |
| ALL SHARED | 0.334440 | 0.287864 | -0.046576 | -0.023686 | -0.128315 | 0.024583 | 0.520000 |

ALL SHARED mean/median cosine 下降，ConflictRate 14%→33%，强冲突 cos<−0.1 比例10%→25%，cos<−0.25 比例4%→10%。但是 ALL SHARED 同 batch mean DeltaCos 的 CI **[-0.128315,0.024583]** 跨零，只有52%的 batch DeltaCos<0，不能称为稳定、普遍增强。

GlobalInteractor shared mean cosine **0.129798→0.188783**，mean DeltaCos 的优化诊断 CI **[0.016924,0.099505]** 在零以上，ConflictRate **32%→26%**，方向与“global shared 冲突增强”的假设相反；其 median 略降，分布证据不完全一致。ALL SHARED 的 gradient norm 主要由 Decoder 主导，不能把 ALL SHARED 的变化定位为 GlobalInteractor 独有问题。

1000 paired batch replicates、seed2022、同 batch 索引同时重采样 B/C。这里 batch 不是独立 scene cluster，bootstrap 仅用于优化诊断；不是正式模型显著性结果。

【Dataset Supervision Distribution】

| Class | SupervisedActorWindows | ValidFutureCoordinates | PooledActorShare | PooledCoordinateShare | MeanBatchActorShare | MeanBatchCoordinateShare |
| --- | --- | --- | --- | --- | --- | --- |
| Vehicle | 20459 | 406922 | 0.596698 | 0.605508 | 0.564629 | 0.582451 |
| Pedestrian | 12979 | 250006 | 0.378540 | 0.372014 | 0.406115 | 0.391269 |
| Bicycle | 849 | 15106 | 0.024762 | 0.022478 | 0.029256 | 0.026280 |

从 frozen TRAIN700 index 顺序（shuffle=False、workers=0、seed2022）扫描前108个 loader batch，选取前100个同时有 supervised V/P 的 batch；共1600个独特 windows、**71 个 scene**。未要求 bicycle。valid future coordinates 定义为每个有效未来时间步的两项 x/y scalar coordinates。

实际共同样本的 coordinate shares 为 Vehicle **60.5508%**、Pedestrian **37.2014%**、Bicycle **2.2478%**；没有沿用约73%/25%/1%的先验。pooled share 与 batch share 均值另行报告。这个固定前缀覆盖71/700 scene，不是全 TRAIN700 类别分布的估计，不能推广为整个训练的真实总贡献或历史 optimizer 轨迹。

【Effective Gradient Pressure Proxy】

| Model | ParameterGroup | MeanNormV | MeanNormP | MeanNormRatio | MeanEffectiveV | MeanEffectiveP | MeanEffectiveRatio | MedianEffectiveRatio | RatioOfMeanEffectiveNorms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Stage3B | LocalEncoder | 5.198571 | 4.367608 | 1.477568 | 2.843426 | 1.537732 | 4.094277 | 1.864661 | 1.849104 |
| Stage3B | TypeEmbedding | 0.053955 | 0.063240 | 1.050850 | 0.031186 | 0.022981 | 3.438634 | 1.452640 | 1.357011 |
| Stage3B | GlobalInteractor | 0.539482 | 0.406341 | 1.516985 | 0.320298 | 0.136606 | 4.922477 | 1.955733 | 2.344694 |
| Stage3B | Decoder | 30.321293 | 16.383849 | 2.555359 | 19.451921 | 5.801028 | 10.605629 | 2.872349 | 3.353185 |
| Stage3B | ALL SHARED | 31.358536 | 17.294703 | 2.368648 | 19.969655 | 6.122572 | 9.389423 | 2.697853 | 3.261645 |
| Stage4A | LocalEncoder | 7.691108 | 6.376683 | 2.987550 | 4.117176 | 2.068987 | 6.069168 | 2.347224 | 1.989947 |
| Stage4A | TypeEmbedding | 0.037058 | 0.044538 | 1.334871 | 0.021450 | 0.014774 | 3.681579 | 1.384735 | 1.451840 |
| Stage4A | GlobalInteractor | 0.654483 | 0.464089 | 1.849547 | 0.403206 | 0.151300 | 5.819979 | 1.851164 | 2.664944 |
| Stage4A | Decoder | 45.842216 | 16.240377 | 5.091068 | 29.478877 | 5.466728 | 17.411851 | 4.438055 | 5.392417 |
| Stage4A | Relation Module | 0.008423 | 0.015963 | 0.943907 | 0.004271 | 0.005336 | 1.690819 | 0.832615 | 0.800376 |
| Stage4A | ALL SHARED | 48.050768 | 18.203652 | 4.538306 | 30.509523 | 6.116245 | 14.711140 | 4.408255 | 4.988277 |

定义每批 w_V=C_V/(C_V+C_P+C_B)、w_P 同理；effective_V=w_V||g_V||，effective_P=w_P||g_P||，effective_ratio=effective_V/(effective_P+1e−12)。class gradient 本身来自类内 mean loss，类样本数量不通过 sum 人为放大 norm。

Stage4A ALL SHARED mean class norms V/P 为 **48.050768/18.203652**，mean effective norms 为 **30.509523/6.116245**。per-batch ratio 的均值 **14.711140**、median **4.408255**；mean pressures 的 ratio 为 **4.988277**。81%的 batch effective_V>effective_P，显示这组诊断样本的 shared pressure proxy 偏向 Vehicle；均值比受低 P norm 的 batch 影响，不能只报告均值比。

**这是 optimization-pressure proxy，不是原始 mixed-loss 梯度的严格代数分解。** 官方回归按 valid coordinate mean，分类按 actor mean，两种权重不同；第一批的拆分 reduction 已与原 mixed regression/classification 数值核对，但把同一 coordinate share 乘上组合 class-loss norm 并不等于精确总梯度贡献，更不能据此证明训练中 Vehicle 因果压制 Pedestrian。

【Relation Module Gradients】

Stage4A Relation Module mean cosine=-0.017062、median=-0.031781、ConflictRate=59.00%。mean V norm=0.008423、P norm=0.015963。Relation Module 上 P 的平均 norm 与平均 weighted proxy 反而高于 V；mean proxy ratio=1.690819、median=0.832615、ratio-of-means=0.800376，不能把 shared 的 V dominance 推广到 relation 模块。

| Pair | Batches | ValidBatches | MeanNormV | MeanNormP | MeanCosVP | ConflictRate | ZeroNormVBatches | ZeroNormPBatches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V<-V | 100 | 100 | 0.000337 | 0.000097 | -0.042403 | 0.530000 | 0.000000 | 0.000000 |
| V<-P | 100 | 100 | 0.000380 | 0.000091 | -0.066154 | 0.540000 | 0.000000 | 0.000000 |
| V<-B | 100 | 61 | 0.000071 | 0.000013 | 0.170307 | 0.377049 | 38.000000 | 39.000000 |
| P<-V | 100 | 100 | 0.000031 | 0.000915 | -0.078832 | 0.550000 | 0.000000 | 0.000000 |
| P<-P | 100 | 98 | 0.000029 | 0.000668 | -0.151999 | 0.561224 | 2.000000 | 2.000000 |
| P<-B | 100 | 61 | 0.000006 | 0.000071 | -0.096348 | 0.590164 | 39.000000 | 39.000000 |
| B<-V | 100 | 61 | 0.000017 | 0.000010 | -0.027171 | 0.475410 | 38.000000 | 39.000000 |
| B<-P | 100 | 61 | 0.000019 | 0.000016 | -0.055096 | 0.508197 | 39.000000 | 39.000000 |
| B<-B | 100 | 25 | 0.000009 | 0.000004 | 0.024139 | 0.440000 | 74.000000 | 75.000000 |

pair_embedding 每行的 V/P 梯度通过完整多层 graph 传播；V loss 不只会影响 V-target pair row。稀有 bicycle pair 的零 row gradient 在表中审计，不视为主要 shared group 缺失；这是训练信号诊断，不是某条交互边或 type pair 的因果解释。Stage3B 没有 Relation Module，没有构造不对应的跨模型比较。

【Mechanism Decision】

MotionPattern=**LOW_MOTION_DEGRADATION**。
GradientConflict=**WEAK**。

| Evidence | Finding | details |
| --- | --- | --- |
| ALL SHARED mean/median cosine lower | YES | mean 0.33443994111469744→0.2878638956406628; median 0.35220770484261044→0.25884435672243744 |
| ALL SHARED conflict rate higher | YES | 0.14→0.33 |
| GlobalInteractor shared changes in same conflict direction | NO | mean improves by 0.058985380025356846; CI [0.01692352839735755,0.09950523418070166]; conflict decreases |
| Most paired batch DeltaCos negative | WEAK | ALL SHARED fraction=0.52; mean-change CI crosses zero |
| Vehicle effective shared pressure exceeds Pedestrian | YES | mean-per-batch ratio=14.711140225658095; median=4.408254968761607; ratio-of-means=4.988276583219234 |

判定依据是综合证据，不是事后创造一个阈值或按简单多数投票。ALL SHARED 与 pressure 支持部分担忧，但 GlobalInteractor 出现反向变化、ALL SHARED mean shift 的优化区间跨零、DeltaCos 负值仅略多于一半，尚不支持“Stage4A interaction 后共享梯度冲突明显增强”这一完整机制。WEAK 同时保留了冲突率上升的真实证据，没有将其归为完全缺乏支持。

本轮不重新判 Stage4A PASS/FAIL：Stage3B TypeEmbedding=SUPPORTED、Stage4A Type-conditioned additive relation bias=技术PASS/科学NOT SUPPORTED、Stage4A-D=CASE B 均保持原结论。

【Recommended Architecture】

名称=**Interaction Necessity Gate**（给定矩阵 CASE 3）。

核心公式（仅设计建议）：

`h_i' = h_base,i + g_i · Δh_rel,i`

`g_i = sigmoid(MLP(h_i, type_i, m_i^hist, c_i^relative))`

为什么=保持 HiVT interaction 主路径，通过观测历史的 motion magnitude、relative context 与 type 判断是否需要启用 interaction residual。低运动目标可允许 gate 接近0，运动或交互需求较高的目标可允许更高 gate；现有证据支持需求门控方向，但不支持增加复杂类别解耦。`m_i^hist` 必须从已观测历史轨迹/速度构造，`c_i^relative` 只使用预测时可用上下文，**不能把本轮未来 GT displacement bin 当作 inference input**。

本轮未实现 gate、adapter 或新 loss；未训练、重选 checkpoint、调整 class weights、oversampling、Reliability 或 Intent。该建议等待“大脑 AI”审查。

【Limitations】

- gradient conflict 是 **optimization diagnostic，不代表因果证明**；这里只测两个终态 checkpoint 的局部损失几何，未追踪真实训练过程，不等于训练过程中持续的 gradient contribution。
- motion bins 使用 **GT displacement，只用于离线评价，不能作为推理时直接可用的输入**。
- 固定100 qualifying batches 来自71个 TRAIN scene，受 sequential scene/window clustering 和入选条件影响，不代表整个 TRAIN700；未增加随机种子或另选 batch 来改变结论。
- batch bootstrap 不等于 independent scene-cluster statistics，也不产生正式模型显著性结论；motion 交叉 context 保持描述性。
- 1–2m、10–20m 及小 context cell 的 scene/instance 覆盖有限，不能过强泛化；scene-bootstrap 保留窗口重叠依赖，未做多重比较校正。
- 原 GPU scatter 协议非 bitwise deterministic。dropout 已关闭、种子固定、全部身份/输入哈希配对；无额外重复批次、种子或 hyperparameter search。
- 参数完全共享的 ALL SHARED 向量按梯度幅度合成，norm 较大的 Decoder 会主导总体 cosine；Relation Module 另行报告。

复现入口与命令见 `00_manifest/stage4ae_execution_record.json`。所有六张图使用真实 CSV、附源数据哈希，导出 PNG/PDF/SVG；梯度 norm 与 effective proxy 分图，绝无模拟数据。

**STOP。没有实现或训练下一模型，没有 merge main。**
