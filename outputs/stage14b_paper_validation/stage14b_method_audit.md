# Stage14B 方法实现审计

审计日期：2026-10-10（Asia/Shanghai）。历史基线：`4daa4ae82557270e4f43881fa22c21e3ba6c4068`。本文依据实际源码描述现有方法，区分已完成实现、已有工程证据和新提出的容量控制；不执行新训练或新评价。

## 1. 真实方法边界

现有系统是“冻结的异质参与者六候选生成器 + 候选模式重评分 + 预设 Bicycle R2 路由”。图排序器输出分数和概率，不输出或修改轨迹坐标。统一评价直接复用同一候选数组，并检查候选SHA不变、四模型oracle几何指标逐bit一致。依据：[G1输出](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py:53)、[候选一致性核验](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:185)、[冻结生成器](/home/lrj/Prediction_Hivt/outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_common.py:46)。

因此可研究的是固定候选集合内的Top1模式选择与排序损失。只靠该排序器不能改善共同候选集合的minFDE6/MR6；也不能由内部排序OOF证明整个预测器对从未见过场景的泛化。依据：[几何指标计算](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:66)、[生成器训练重叠审计](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:261)。

## 2. 冻结生成器保留了哪些历史部件

| 部件 | 实际状态与含义 | 源码证据 |
|---|---|---|
| HiVT局部编码 | 历史actor-actor注意力、时序编码、lane-actor注意力均保留。局部模块读取5帧历史，lane radius为50m | [局部forward](/home/lrj/Prediction_Hivt/models/hivt_runtime/local_encoder.py:70)、[配置](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_config.yaml:2) |
| HiVT全局交互 | 原有GlobalInteractor在t0图上使用相对位置与相对运动朝向，产生6个全局模式embedding。NoGraph仅关闭后置排序图消息，生成器内这些交互仍存在 | [全局forward](/home/lrj/Prediction_Hivt/models/hivt_runtime/global_interactor.py:58)、[Stage5A调用链](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_model.py:18) |
| Stage3B类别embedding | 3×64=192参数；在LocalEncoder后加到actor embedding。Vehicle/Pedestrian/Bicycle的t0类型是输入，不是待预测类别 | [类型fusion](/home/lrj/Prediction_Hivt/outputs/stage3_multitype_hivt/00_manifest/stage3b_model.py:9)、[实际启用](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_model.py:19) |
| Stage5A运动条件decoder | 6维条件为类型one-hot和三项历史位移的log1p。两个64→16→64 residual experts由6→16→2 router加权；作用于原aggr embedding后、loc/scale前。两个expert不是两个预测模式，候选仍为6条 | [历史条件](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_decoder.py:11)、[decoder结构及forward](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_decoder.py:43) |
| 原pi/logit头 | Stage5A的pi头读取未加expert residual的local/global embedding；loc/scale分支使用adapted embedding。不能把router概率称为模式概率 | [pi与残差分离](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_decoder.py:68) |
| 基础lane map | centerline几何与lane-actor边保留；当前输入的intersection/turn/control类别值均为0，不能声称真实道路语义已输入生成器 | [lane输入与类别置零](/home/lrj/Prediction_Hivt/outputs/stage3_multitype_hivt/02_preprocessed/stage3_dataset.py:73) |
| Stage4/Stage7增强 | Stage5A继承Stage3B/native HiVT链；这条forward没有Stage4专门type-conditioned global模块，也没有Stage7 semantic branch | [继承链](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_model.py:3)、[Stage3B基类](/home/lrj/Prediction_Hivt/outputs/stage3_multitype_hivt/00_manifest/stage3b_model.py:4)、[native模块构造](/home/lrj/Prediction_Hivt/models/hivt_nuscenes.py:18) |

Stage5A是从相同初始化规则重新训练的增强生成器，并非加载已训练Stage3B再微调；后续排序实验将其650,403参数全部冻结。源码明确初始化时没有加载Stage3B训练权重，严格检查共享随机初始state，随后冻结加载Stage5A checkpoint并验证内容SHA。依据：[初始化规则与参数审计](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_common.py:44)、[冻结权重加载](/home/lrj/Prediction_Hivt/outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_common.py:46)。

## 3. 图的节点和边到底是什么

每个actor的每条预测候选构成一个mode node。每个目标保留至多8个t0距离≤50m的有效邻居；邻居选择与当前观测位置有关，不使用未来GT。目标6个mode与各邻居6个mode之间保留独立关系，未提前把邻居模式聚成单个节点。依据：[邻居选择](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py:62)、[独立6×6关系](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py:74)。

15维节点输入逐字段如下。实现定义及赋值：[字段表](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py:6)、[node_features](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py:40)。

| 索引 | 字段 | 原始数值定义 |
|---|---|---|
| 0–2 | 三类型one-hot | Vehicle、Pedestrian、Bicycle |
| 3 | original_logit | 冻结生成器的对应mode原始logit |
| 4 | original_probability | 冻结生成器对应mode原始softmax概率 |
| 5 | history_recent_displacement | 最近两次有效观测间的距离，m；不是按真实间隔换算的速度 |
| 6 | history_net_displacement | 首末有效历史点距离，m |
| 7 | history_path_length | 有效历史观测之间逐段距离之和，m |
| 8–9 | endpoint_x/y | t0 ego坐标中的预测端点，m |
| 10 | candidate_net_displacement | t0位置至预测端点距离，m |
| 11 | candidate_path_length | t0至12个预测点的路径长，m |
| 12 | candidate_mean_speed | 候选路径长/6s，m/s，使用固定名义预测时域 |
| 13–14 | endpoint_heading_sin/cos | 最末与倒数第三个预测点的secant方向sin/cos |

17维有向关系如下。定义及赋值：[字段表](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py:9)、[interaction_edges](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py:75)。

| 索引 | 字段 | 原始数值定义 |
|---|---|---|
| 0 | future_minimum_distance | 同时刻两候选点距离的最小值，m |
| 1 | closest_timestep_over_Tf | 最接近点索引加1，再除12，无量纲 |
| 2 | future_mean_distance | 12个同时刻距离平均，m |
| 3 | endpoint_distance | 两预测端点距离，m |
| 4 | initial_future_distance | 第一个未来预测点之间距离，m；不是t0距离 |
| 5–6 | relative_endpoint_heading_sin/cos | 两候选末段secant相对方向sin/cos |
| 7 | minimum_relative_step_displacement | 两候选相邻步位移之差的范数最小值，m；不是加速度 |
| 8 | closing_distance_over_6s | (第一个未来距离−末端距离)/6s；保留原定义，不把分母改成5.5s |
| 9 | neighbor_original_probability | 邻居相应候选的原始概率 |
| 10 | target_original_probability | 目标相应候选的原始概率 |
| 11–13 | target三类型one-hot | 目标t0类型 |
| 14–16 | neighbor三类型one-hot | 邻居t0类型 |

节点第3–14列、边第0–10列采用对应fold InnerTrain统计进行标准化；one-hot不标准化，缺失邻居槽在标准化后仍置零。依据：[原标准化代码](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py:50)、[统计仅InnerTrain协议](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/00_manifest/stage11b_protocol.json:73)。

## 4. G1与NoGraph的评分计算

G1 node encoder为15→64→64 ReLU MLP。对目标mode k，message MLP读取目标15维、邻居15维和17维边，输入47维；attention MLP读取两侧64维hidden与17维边，输入145维。masked softmax在最多8×6=48个邻居mode上进行，然后加权求64维message。原始概率是node/edge特征，由MLP学习其作用；这段实现没有额外乘一个邻居概率，也没有预测联合模式分布。依据：[G1构造和forward](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py:30)。

评分形式为：

\[
h_{ik}=\phi_{node}(x_{ik}),\quad
m_{ik}=\sum_{j,l}\alpha_{ik,jl}\phi_{msg}(x_{ik},x_{jl},e_{ik,jl}),\quad
z_{ik}=z^{base}_{ik}+\phi_{score}(\mathrm{LayerNorm}(h_{ik}+m_{ik})).
\]

scoring head为64→32→1 ReLU MLP，最终层在初始化时全零。G1没有map aggregator，map message为零；不能将历史G2/G3语义map分支的设计或失败结果写成G1当前输入。依据：[构造条件与零初始化](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py:35)、[最终评分](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py:51)。

Stage14A NoGraph重用相同node encoder、LayerNorm、head及原logit残差，只把新增的future-interaction message固定为零，并不注册无用交互参数。完整图输入仍由共同pipeline供给，但其forward只读取目标node特征及原logit；候选、actor列表和训练数据不因NoGraph变化。依据：[NoGraph实现](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/02_models/stage14a_nograph.py:26)、[共享初始state逐bit核验](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/01_preflight/stage14a_preflight.py:44)。

**“NoGraph”指后置重评分层没有新增候选交互message，不能解释为整个系统没有参与者交互。** 其输入候选与原始logit已经来自包含局部和全局交互的生成器。依据：[生成器forward](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_model.py:18)、[NoGraph forward](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/02_models/stage14a_nograph.py:41)。

## 5. A/C损失与GT的作用

设固定候选k的GT endpoint error为Eik，pik=softmax(zik)。A为SoftCE，目标qik=softmax(−Eik/1m)，LA=−Σk qik log pik。C为normalized expected regret：

\[
c_{ik}=E_{ik}-\min_l E_{il},\qquad
s_i=\max\left(1\mathrm{m},\frac1{6}\sum_k c_{ik}\right),\qquad
L_C(i)=\sum_k p_{ik}\frac{c_{ik}}{s_i}.
\]

正式训练对actor取均值，没有类别重权。所有E来自detach后的监督标签；1m下限、按actor的mean-regret scale与原代码完全一致。C是一个可微的固定候选选择风险目标，不是新轨迹回归loss，也不是训练时argmax后的Top1FDE。依据：[原objective](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py:82)、[原训练梯度与均值](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/03_training/stage11b_train.py:130)、[闭式值与标签detach核验](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/01_preflight/stage14a_preflight.py:50)。

GT用于监督损失、InnerDev checkpoint选择和离线评价；不进入图feature allowlist或head forward。真实图构建污染检查将GT置NaN、翻转future/target mask，重建输入后逐bit相同。评价入口另将标签与7项forward输入分开；该重复forward检查验证调用不依赖标签对象，图特征隔离的更直接证据来自源window重建审计。依据：[观察allowlist](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00_manifest/stage8a_graph.py:15)、[真实源数据污染核验](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/01_preflight/stage14a_preflight.py:29)、[评价标签分离](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:140)。

原dataset本身保存未来字段，并以未来mask定义监督/完整时域评价资格；这与forward的特征隔离应分开说明。历史OOF是完整12步GT actor子集，不能把其coverage自动推广成所有t0参与者在线预测覆盖率。依据：[dataset监督mask](/home/lrj/Prediction_Hivt/outputs/stage3_multitype_hivt/02_preprocessed/stage3_dataset.py:75)、[OOF评价mask说明](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:277)。

## 6. R2及Bicycle路由

R2使用19维观测/候选特征和19→32→1 residual MLP。其后7维包括原始mode概率加权的邻居候选距离/冲突聚合，因此R2本身也利用未来候选交互摘要，不能称为own-only baseline。其几何距离/exp核是特征，不是经校准的碰撞概率。依据：[19维与概率加权聚合](/home/lrj/Prediction_Hivt/outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_features.py:5)、[R2评分头](/home/lrj/Prediction_Hivt/outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_head.py:5)。

Bicycle采用对应fold冻结R2输出，是预先规定的类别保留/路由控制，使四个受控排序模型的Bicycle输出完全相同。原要求明确以Stage8/9发现的Bicycle负迁移风险为这项策略的动机。Bicycle依然可以作为邻居，也保留在原排序训练目标中；最终覆盖其scores不意味着图网络自身改善了Bicycle，更不能将该覆盖称为验证过的类间泛化。依据：[原要求中的动机及路由](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/00_manifest/stage11b_requirements.txt:508)、[原训练保留Bike目标](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/03_training/stage11b_train.py:167)、[统一评价逐bit覆盖](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:147)。本文没有为该预设策略另行虚构理论保证或新实证。

## 7. 参数量与部署口径

| 组件/系统 | 总参数 | 排序训练时实际可训练参数 | 一fold逻辑模型包总量 |
|---|---:|---:|---:|
| 冻结Stage5A生成器 | 650,403 | 0 | 650,403 |
| R2 | 673 | 673（自身原训练）；作为Bike路由时0 | 651,076（生成器+R2） |
| NG-A / NG-C | 7,425 | 7,425 | 658,501（生成器+NG+Bike R2） |
| G-A / G-C | 24,066 | 24,066 | 675,142（生成器+G+Bike R2） |
| MatchedNG-C结构控制 | 24,001 | 24,001（结构及真实输入preflight已核验） | 675,077（生成器+matched head+Bike R2） |

生成器650,403包含原No-Type骨干645,809、Stage3B类型embedding192、Stage5A experts/router新增4,402；G1与NoGraph差额为16,641，不能隐去这一容量混杂。依据：[生成器参数断言](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_common.py:54)、[类型embedding参数断言](/home/lrj/Prediction_Hivt/outputs/stage3_multitype_hivt/00_manifest/stage3b_common.py:59)、[G1/NoGraph真实参数核验](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/01_preflight/stage14a_preflight.py:46)、[R2参数断言](/home/lrj/Prediction_Hivt/outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_head.py:10)。表内系统总量为这些组件的直接求和，每fold一套，不主张3fold ensemble已部署。

MatchedNG-C为原目标编码/LayerNorm/head加64→128→64 own-feature residual adapter，新增16,576有效参数，合计24,001；较G1少65，即约0.2701%。其forward只读取自身mode hidden，不引入邻居message。结构及真实输入preflight已经PASS，包括共享初始化、邻居/边扰动不改变输出、adapter对非零评分的实际贡献和C损失梯度。该设计控制的是近似参数量，不能保证表达能力、优化难度或计算开销完全等价；这些工程核验不能替代tiny、正式训练冻结及统一OOF效果检验。依据：[新模型实现](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/02_models/stage14b_matched_nograph.py:22)、[结构核验](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/01_preflight/stage14b_model_integrity.json:3)、[真实输入核验](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/01_preflight/stage14b_capacity_preflight.json:2)。

“可训练参数”按参数注册、requires_grad及优化器参与口径统计，不代表每个方向都可从softmax目标辨识。原评分头的共享标量bias只产生六模式共同logit平移，在SoftCE/C中不可辨识；这个相同的遗留维度也存在G1，不是为新模型填充的无效容量。新增adapter参数的实际C梯度已经单独核验。依据：[同维度解释](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/01_preflight/stage14b_capacity_preflight.json:31)、[adapter梯度核验](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/01_preflight/stage14b_model_integrity.json:109)。

Stage14A计时仅覆盖同批设备上缓存输入的head forward，排除了HiVT、特征构建、数据传输与R2路由；不能把该ms/actor数值称为端到端部署延迟。模型包参数总量也不等于缓存评价时实际GPU常驻参数量。依据：[效率scope](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/07_diagnostics/stage14a_efficiency.py:80)。

## 8. 审计结论与论文可用描述边界

方法描述可准确写成：在含类型embedding和历史运动条件decoder的冻结HiVT候选生成器之上，学习基于独立候选对关系的稀疏future-interaction残差排序；采用normalized expected regret训练排序头；Bicycle使用预设fold-R2路由。每项限定都对应上文实际forward和路由代码。

当前证据不能将候选生成器叫“未增强的原始HiVT”，不能把NoGraph叫“完全没有交互”，不能把R2叫“自身候选特征基线”，不能把C叫新轨迹生成loss，不能把原MR6称为官方nuScenes MissRate。后一点原实现明确使用末端oracle FDE>2m，指标与官方协议的完整轨迹最大距离规则须分别命名：[内部MR6代码](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:86)、[官方metrics实现](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/metrics.py)。

内部OOF保持了对应fold排序训练、标准化和InnerDev选择的隔离，但全部630个OOF场景属于Stage5A训练场景；官方VAL还曾用于生成器checkpoint选择和后续研究开发。方法与统计应保留这些边界，不用补训练结果替换或覆盖历史结论。依据：[完整训练来源审计](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:250)、[历史开发复用审计](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:264)。
