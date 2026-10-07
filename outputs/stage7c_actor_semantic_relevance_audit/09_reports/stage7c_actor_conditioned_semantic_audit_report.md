# Stage7C Actor-Conditioned Semantic Lane Relevance Audit

## 【Stage7A Failure Question】

本阶段只做冻结模型的诊断，不训练、不 fine-tune、不实现新的 attention 模型。
源提交 `77e77b3e1f48a5e68e48d5665308232b97d058f4`，独立分支 `stage7c/actor-conditioned-semantic-relevance-audit`。
Stage3B checkpoint SHA256 `461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547`；Stage7A SHA256 `e30b7dbe7260ef5765e10f95d5eadf58e42f480e79f15b58a3b149a9d271abc1`。
原结论保持：Stage3B Overall minFDE6=1.343260461；Stage7A=1.350545814、Top1FDE6=2.749482077。
Stage7A 的 SemanticMap=NOT_SUPPORTED、PaperUsableSemantic=NO、ReadyStage7B=NO 均未修改。
三个问题是 H1 相关性分配不当、H2 K/V value 改变、H3 generic adapter 主导；本报告只评估证据。

## 【Attention Capture Integrity】

使用原 ALEncoder 的只读 forward hooks：观察 `attn_drop` 的输入，即 softmax 后 alpha；所有观察 hook 返回 None。
两个模型各检查相同100个不同的随机 official VAL batch（seed2022，batch16）。
raw_prediction、mode_logits、mode_prob 的最大差值均为 **0.0**；state_dict 完全不变，结果 **PASS**。
完整捕获仍采用原评价 kernel policy；相对冻结逐 actor 误差的最大数值差约3.05e-5m，均值差绝对值<4e-9m。
这与受控 deterministic CUDA 的 instrumentation 等价检查不同；两者均记录，未更改历史指标。
捕获3603 windows、150 scenes、26,807,585条实际保留的 lane-actor edges。
两模型 edge index/edge attributes 完全一致，8-head 每 actor 权重和最大误差 6.41337054e-06。
Stage3B input 显式移除 lane_semantic，解释性 semantic 标签只在 CPU 离线加入。
所有 edge 字段用无损规范化 NPZ 保存，`stage7c_read_edges.py` 可以逐条恢复完整用户字段；大档案只保留本地。

## 【GT Future Lane Relevance】

所有54990 full-horizon targets按 scene/sample/instance、node、GT SHA、mask和类型配对。
GT future 是12个未来点组成的连续 polyline，计算它与当前 `[lane_position,lane_position+lane_vector]` 真正 segment 的最小2D距离。
已验证 interior crossing、stationary GT、degenerate segment和point-to-segment计算；不使用 start-point 近似。
GT-nearest primary在全当前 graph 的 lane segments 中 argmin；相同最小距离选最小本地 lane index。
未进入50m incoming edge的 lane权重为0；attention rank采用相同权重的平均rank。
辅助 incoming-only nearest 和最佳 co-nearest rank（距离容差1e-8m）单独报告，避免几何覆盖与选择混淆。
861个target的全图最近 lane不在其incoming edges；358个target没有任何incoming edge。
相关mass对空edge求和为0；entropy/topK/incoming-rank无定义时留空，明确排除相应均值/CI分母。
Vehicle entropy分母41974，Vehicle mass/rank分母42332；Turning1663全部有incoming edge。
2m/4m仅辅助距离定义，GT、转向和relevance从未加入模型输入。
详细覆盖和tie敏感性见 `06_tables/stage7c_lane_coverage.csv`、`stage7c_attention_relevance.csv`。

## 【Stage3B vs Stage7A Attention】

| Group | Count | B Mass2m | A Mass2m | B Rank | A Rank | B Entropy | A Entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 54990 | 0.045253 | 0.042974 | 116.997127 | 124.043190 | 4.593523 | 4.773848 |
| Overall Vehicle | 42332 | 0.056767 | 0.053629 | 103.634296 | 114.266465 | 4.565671 | 4.717238 |
| vehicle.moving | 10461 | 0.210087 | 0.201039 | 107.609263 | 110.782669 | 4.349927 | 4.666489 |
| Vehicle >5m | 9744 | 0.230394 | 0.222357 | 114.750616 | 110.251437 | 4.323981 | 4.638025 |
| TurningVehicle_GT | 1663 | 0.176139 | 0.171935 | 122.110944 | 103.394768 | 4.623712 | 4.791150 |
| GT-left | 831 | 0.177179 | 0.141940 | 97.544525 | 109.774368 | 4.627615 | 4.821665 |
| GT-right | 832 | 0.175100 | 0.201894 | 146.647837 | 97.022837 | 4.619815 | 4.760672 |
| TurnOptionCount20=0 | 10247 | 0.021533 | 0.020149 | 83.595589 | 82.635015 | 4.235579 | 4.342541 |
| TurnOptionCount20=1 | 7445 | 0.052291 | 0.045817 | 85.536199 | 97.788180 | 4.457357 | 4.626448 |
| TurnOptionCount20=2 | 8735 | 0.053169 | 0.048369 | 111.572066 | 119.533944 | 4.688705 | 4.830141 |
| TurnOptionCount20>=3 | 15905 | 0.083540 | 0.081745 | 120.656649 | 139.465891 | 4.754036 | 4.930700 |
| Pedestrian | 12002 | 0.005967 | 0.006213 | 163.958424 | 160.123813 | 4.684897 | 4.969251 |

B=Stage3B，A=Stage7A。rank越小越靠前；entropy增加只描述分散程度，不预设优劣。

| Group | Metric | Count | Delta | CI_lower | CI_upper |
| --- | --- | --- | --- | --- | --- |
| Overall Vehicle | GTRelevantMass2m | 42332 | -0.003138 | -0.005449 | -0.001266 |
| vehicle.moving | GTRelevantMass2m | 10461 | -0.009048 | -0.017090 | -0.001864 |
| Vehicle >5m | GTRelevantMass2m | 9744 | -0.008037 | -0.016900 | 0.000020 |
| TurningVehicle_GT | GTRelevantMass2m | 1663 | -0.004203 | -0.016492 | 0.008427 |
| TurningVehicle_GT | CorrectTurnMass | 1663 | -0.031739 | -0.040008 | -0.023996 |
| TurningVehicle_GT | OppositeTurnMass | 1663 | -0.033277 | -0.039620 | -0.026647 |
| TurningVehicle_GT | TopConnectorMatchRate | 1663 | -0.177390 | -0.227148 | -0.116680 |
| GT-left | GTRelevantMass2m | 831 | -0.035238 | -0.048022 | -0.022473 |
| GT-left | CorrectTurnMass | 831 | -0.046355 | -0.060239 | -0.035970 |
| GT-left | OppositeTurnMass | 831 | -0.032655 | -0.041493 | -0.024395 |
| GT-left | TopConnectorMatchRate | 831 | -0.269555 | -0.341833 | -0.195268 |
| GT-right | GTRelevantMass2m | 832 | 0.026794 | 0.008592 | 0.045616 |
| GT-right | CorrectTurnMass | 832 | -0.017141 | -0.025925 | -0.008345 |
| GT-right | OppositeTurnMass | 832 | -0.033899 | -0.045064 | -0.023887 |
| GT-right | TopConnectorMatchRate | 832 | -0.085337 | -0.171923 | 0.019508 |

1000 paired scene-cluster bootstrap、seed2022、150 official scenes，actor-window sums/counts pooled，percentile95%CI。重叠次要组不做多重比较调整。全部rank/entropy/turn指标CI见 `stage7c_attention_bootstrap_ci.csv`。

## 【Turning Vehicle】

完全复用 Stage7A membership：vehicle，future endpoint displacement>5m，首尾1秒secant>0.5m，|wrapped heading change|>20°。
Count=1663；GT-left=831，GT-right=832，仅offline。
correct-turn mass下降，**opposite-turn mass也下降**，不是总体转向相反方向。straight mass从0.308012增至0.443093。
24个Turning actor没有incoming connector；主match rate把NO_CONNECTOR记为nonmatch，条件match rate另存。
match是离线解释指标，不能称为预测准确率。
GT-left relevant mass显著下降，GT-right反而改善；Turning全体nearest rank改善。因此不能把所有turning结果归为单一坏attention。

| Group | Count | Stage3B_CorrectTurnMass | Stage7A_CorrectTurnMass | Stage3B_OppositeTurnMass | Stage7A_OppositeTurnMass | Stage3B_TopConnectorMatchRate | Stage7A_TopConnectorMatchRate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TurningVehicle_GT | 1663 | 0.167866 | 0.136127 | 0.138249 | 0.104972 | 0.445580 | 0.268190 |
| GT-left | 831 | 0.178235 | 0.131880 | 0.145078 | 0.112424 | 0.451264 | 0.181709 |
| GT-right | 832 | 0.157510 | 0.140368 | 0.131428 | 0.097530 | 0.439904 | 0.354567 |

| Group | Count | B minFDE | A minFDE | Delta minFDE | B Top1FDE | A Top1FDE |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | 54990 | 1.343260 | 1.350546 | 0.007285 | 2.687984 | 2.749482 |
| Overall Vehicle | 42332 | 1.431914 | 1.440195 | 0.008281 | 3.050323 | 3.135868 |
| vehicle.moving | 10461 | 4.791532 | 4.793599 | 0.002067 | 10.734859 | 11.025390 |
| Vehicle >5m | 9744 | 5.579887 | 5.586459 | 0.006572 | 12.105361 | 12.457772 |
| TurningVehicle_GT | 1663 | 14.452465 | 14.442341 | -0.010124 | 18.571997 | 19.098375 |
| GT-left | 831 | 13.962548 | 13.939678 | -0.022870 | 17.970814 | 18.465645 |
| GT-right | 832 | 14.941793 | 14.944400 | 0.002607 | 19.172457 | 19.730344 |
| TurnOptionCount20=0 | 10247 | 0.402894 | 0.413888 | 0.010994 | 0.890210 | 0.966586 |
| TurnOptionCount20=1 | 7445 | 0.985545 | 0.999404 | 0.013859 | 2.710227 | 2.904737 |
| TurnOptionCount20=2 | 8735 | 1.281224 | 1.290103 | 0.008879 | 3.131176 | 3.207839 |
| TurnOptionCount20>=3 | 15905 | 2.386572 | 2.390166 | 0.003594 | 4.556797 | 4.602120 |
| Pedestrian | 12002 | 1.043875 | 1.048419 | 0.004543 | 1.457706 | 1.440422 |

Turning minFDE点估计略改善，Top1仍恶化；correct-turn mass变化与minFDE误差变化几乎无相关。这些是H1证据的重要边界。

## 【Turn Ambiguity】

| TurnOptionCount20 | Count | Stage3B_minFDE | Stage7A_minFDE | DeltaMinFDE | Stage3B_Top1FDE | Stage7A_Top1FDE | DeltaTop1FDE | RelevantAttentionDelta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 10247 | 0.402894 | 0.413888 | 0.010994 | 0.890210 | 0.966586 | 0.076376 | -0.001383 |
| 1 | 7445 | 0.985545 | 0.999404 | 0.013859 | 2.710227 | 2.904737 | 0.194510 | -0.006474 |
| 2 | 8735 | 1.281224 | 1.290103 | 0.008879 | 3.131176 | 3.207839 | 0.076663 | -0.004800 |
| >=3 | 15905 | 2.386572 | 2.390166 | 0.003594 | 4.556797 | 4.602120 | 0.045323 | -0.001794 |

计数来自所有区域map token的完整centerline到t0 vehicle的严格<20m距离，每个connector token计一次；TurnOptionCount20是非空left/straight/right类别数。
>=3组Count15905、ΔminFDE=+0.003594m。minFDE/Top1FDE的退化没有随着选项数单调增大：1-option组的delta更大。不得把高歧义组较大的绝对误差解释成Stage7A特有退化。

## 【Semantic Residual Decomposition】

| SemanticGroup | Count | mean_norm_r | mean_norm_r0 | mean_norm_delta_r | mean_delta_ratio |
| --- | --- | --- | --- | --- | --- |
| all | 2714494 | 2.306098 | 1.884985 | 1.819476 | 0.774654 |
| ordinary_lane | 1212491 | 2.059737 | 1.884985 | 1.040863 | 0.527513 |
| connector | 1502003 | 2.504973 | 1.884985 | 2.448011 | 0.974159 |
| left | 331898 | 2.280432 | 1.884985 | 1.902434 | 0.861606 |
| straight | 859172 | 2.777359 | 1.884985 | 2.896220 | 1.030706 |
| right | 309481 | 1.991084 | 1.884985 | 1.792492 | 0.938586 |
| unknown_connector | 1452 | 2.186393 | 1.884985 | 1.661997 | 0.823978 |
| traffic_light | 804782 | 3.102355 | 1.884985 | 3.013481 | 0.958406 |
| stop_sign | 188002 | 2.620796 | 1.884985 | 1.696779 | 0.671808 |
| other_control | 1005063 | 2.299782 | 1.884985 | 2.497376 | 1.131915 |
| crosswalk | 928522 | 2.629271 | 1.884985 | 2.951827 | 1.150451 |
| non_crosswalk | 1785972 | 2.138082 | 1.884985 | 1.230770 | 0.579278 |

按实际VAL segment/window出现次数加权，共2714494次、50个已观察9D pattern。r(0)常量较大，但mean||delta_r||=1.819476、mean ratio=0.774654，不属于很小的semantic-specific成分。ratio可>1，因为常量与特异向量可以抵消；这些norm比例不是方差解释比例、能量占比或物理效应。

| Group | Count | Stage3B_MeanValueNorm | Stage7A_MeanValueNorm | MeanValueDeltaNorm | MeanValueCosine | SemanticEffect | GTRelevantSemanticEffect | AttentionWeightedResidualValueNorm | AttentionWeightedSpecificValueNorm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall Vehicle | 42332 | 8.696125 | 8.601664 | 9.975117 | 0.315201 | 1.600831 | 0.106520 | 3.004006 | 2.500829 |
| vehicle.moving | 10461 | 8.893307 | 8.673666 | 10.038997 | 0.333409 | 2.174985 | 0.405190 | 3.496883 | 3.488179 |
| TurningVehicle_GT | 1663 | 9.311752 | 8.977686 | 10.247993 | 0.349306 | 2.289479 | 0.331749 | 3.591097 | 3.665050 |

输入lane semantic residual经Stage7A自身lin_v投影的非零norm可以证明分支确实改变representation。跨Stage3B/Stage7A的V差异还包含独立训练后共享权重变化，不能归因于semantic污染。actor effect是alpha加权delta norm，绝不是因果归因。

## 【ON / ZERO / SHUFFLE / CENTERED】

| Model | Group | Count | minFDE | Top1FDE |
| --- | --- | --- | --- | --- |
| Stage3B | Overall | 54990 | 1.343260 | 2.687984 |
| Stage3B | Vehicle | 42332 | 1.431914 | 3.050323 |
| Stage3B | MovingVehicle | 10461 | 4.791532 | 10.734859 |
| Stage3B | TurningVehicle_GT | 1663 | 14.452465 | 18.571997 |
| Stage7A-ON | Overall | 54990 | 1.350546 | 2.749482 |
| Stage7A-ON | Vehicle | 42332 | 1.440195 | 3.135868 |
| Stage7A-ON | MovingVehicle | 10461 | 4.793599 | 11.025390 |
| Stage7A-ON | TurningVehicle_GT | 1663 | 14.442341 | 19.098375 |
| Stage7A-ZERO | Overall | 54990 | 1.358928 | 2.727582 |
| Stage7A-ZERO | Vehicle | 42332 | 1.450922 | 3.107103 |
| Stage7A-ZERO | MovingVehicle | 10461 | 4.833782 | 10.903239 |
| Stage7A-ZERO | TurningVehicle_GT | 1663 | 14.457739 | 18.547957 |
| Stage7A-SHUFFLE | Overall | 54990 | 1.352316 | 2.750507 |
| Stage7A-SHUFFLE | Vehicle | 42332 | 1.442453 | 3.137778 |
| Stage7A-SHUFFLE | MovingVehicle | 10461 | 4.801725 | 11.020772 |
| Stage7A-SHUFFLE | TurningVehicle_GT | 1663 | 14.447483 | 19.041904 |
| Stage7A-CENTERED | Overall | 54990 | 1.350821 | 2.750369 |
| Stage7A-CENTERED | Vehicle | 42332 | 1.440374 | 3.137583 |
| Stage7A-CENTERED | MovingVehicle | 10461 | 4.796056 | 11.035421 |
| Stage7A-CENTERED | TurningVehicle_GT | 1663 | 14.443619 | 19.189435 |

ON/ZERO及Stage3B均直接复用冻结actor结果并核对完全相同的身份、GT、mask；本轮只新增SHUFFLE/CENTERED各一次完整VAL推理。
SHUFFLE每graph用seed2022和scene+sample哈希派生固定permutation，仅置换9D rows；分布、geometry、lane-actor edges和actor数据均保持。它是一次OOD诊断，不是正式baseline或等效性检验。
CENTERED使用同一checkpoint的evaluation-only输出hook减去r(0)；参数和state_dict不变。
在此ALEncoder，减去同一r0对同actor所有lane的key score加同一常量，softmax理论上抵消该常量，主要留下value/gate路径扰动；不能由此隔离所有semantic-specific K/V效应。

| Model | Metric | Delta | CI_lower | CI_upper |
| --- | --- | --- | --- | --- |
| Stage3B | minFDE6 | -0.007285 | -0.013917 | -0.000438 |
| Stage3B | Top1FDE6 | -0.061498 | -0.089320 | -0.035941 |
| Stage7A-ZERO | minFDE6 | 0.008383 | 0.004291 | 0.012608 |
| Stage7A-ZERO | Top1FDE6 | -0.021900 | -0.044990 | -0.001028 |
| Stage7A-SHUFFLE | minFDE6 | 0.001770 | -0.000440 | 0.003909 |
| Stage7A-SHUFFLE | Top1FDE6 | 0.001025 | -0.008449 | 0.010859 |
| Stage7A-CENTERED | minFDE6 | 0.000275 | -0.000584 | 0.001133 |
| Stage7A-CENTERED | Top1FDE6 | 0.000887 | -0.006129 | 0.008608 |

表中delta为setting−ON。ON比ZERO的candidate minFDE更好，但ZERO的Top1更好。SHUFFLE和CENTERED与ON差值很小且CI跨0；这不证明等效，也不证明semantic无信息。CENTERED没有修复Stage7A相对Stage3B的差距。

## 【Attention-Error Association】

| Group | X | Y | Count | SpearmanR | CI_lower | CI_upper |
| --- | --- | --- | --- | --- | --- | --- |
| Overall Vehicle | DeltaRelevantMass | DeltaMinFDE | 42332 | -0.000307 | -0.024179 | 0.022525 |
| vehicle.moving | DeltaRelevantMass | DeltaMinFDE | 10461 | -0.038535 | -0.077035 | -0.002283 |
| TurningVehicle_GT | DeltaRelevantMass | DeltaMinFDE | 1663 | -0.049901 | -0.101200 | 0.002523 |
| GT-left | DeltaRelevantMass | DeltaMinFDE | 831 | -0.101024 | -0.160604 | -0.032269 |
| GT-right | DeltaRelevantMass | DeltaMinFDE | 832 | -0.045107 | -0.136786 | 0.025964 |
| TurningVehicle_GT | DeltaCorrectTurnMass | DeltaMinFDE | 1663 | -0.006138 | -0.062368 | 0.047467 |

所有要求的Spearman及原始descriptive p-value见 `stage7c_attention_error_correlation.csv`；p-value不能替代scene uncertainty。因主相关幅度很弱，补充1000次whole-scene重采样，每次重新计算Spearman ranks。
Overall Vehicle相关近零。Moving及GT-left的relevant-mass/minFDE相关小幅负向且探索性scene CI仍为负，这提供局部方向一致的H1证据；不能泛化为总体预测退化的因果机制。Turning correct-turn变化与误差基本无相关，GT-right/nearest-rank有相反或不确定模式。

## 【Case Studies】

| Case | Delta minFDE | Delta relevant mass | Delta correct mass | Delta opposite mass |
| --- | --- | --- | --- | --- |
| case1_relevance_shift | 5.016751 | -0.075882 | not turning | not turning |
| case2_opposite_turn | 3.377614 | -0.074381 | 0.030272 | 0.045391 |
| case3_correct_turn | -2.339435 | 0.085025 | 0.008429 | -0.103104 |
| case4_reasonable_attention | 2.539440 | 0.081531 | not turning | not turning |

四个不同actor均按固定条件选取，完整数值source在 `07_cases/`；匹配GT/map/axis和共用attention色标。PNG300dpi/PDF/SVG在 `05_figures/`。
Case1相关mass下降且误差增加；Case2 opposite增加的少数例子不能覆盖全体opposite下降事实；Case3 correct增加且改善；Case4相关mass>0.5并提升，但预测恶化。极端案例不是总体比例证据，也不代替bootstrap。控制/crosswalk用token标记保留multi-hot状态，turn用segment颜色，不覆盖turn颜色。

## 【Interpretation】

**FailurePattern = RELEVANCE_MISALLOCATION**，作为最符合用户CASE A规则的局部、探索性诊断，而非确定的退化原因。
Moving和GT-left同时满足GTRelevantMass2m的paired CI<0及attention/error关联的scene CI<0；correct-turn/connector match的下降补充attention分配改变证据。
必须同时保留反证：整体Vehicle相关为−0.000307，Turning nearest rank改善、GT-right几何相关性改善、opposite mass下降、Turning minFDE不变差、ambiguity退化不单调。局部相关幅度小，不能解释全部总体退化。
CASE B的“relevance未变差”前提不成立，CENTERED也未挽救表现。CASE C的ON≈SHUFFLE前提近似成立，但semantic-specific成分不小，r0大不能单独证明generic dominance。
这些限制并不支持宣称新的semantic方法已有效；只足以推荐一个未来的受控、score-only相关性验证。

**SemanticRoute = CONTINUE**

**RecommendSemanticAttentionModel = YES**

建议仅记录，不实现：
`b_il^sem = MLP(semantic_l, relative_lane_actor_position, actor_type_i)`；
`alpha_il = softmax(q_i k_l / sqrt(d) + b_il^sem)`；geometry-only `V_l`保持不变。
本阶段训练updates=0、新checkpoint=0、新attention model实现=0、Stage7B=未运行、test=未使用、main=未merge。
Stage7A原NOT_SUPPORTED/NO/NO结论保持。CONTINUE只表示建议，后续训练仍需新授权。
**STOP。等待审查，不启动任何新模型训练。**
