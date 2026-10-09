# Stage14B 端到端验证方案与数据来源审计

本文是实验执行方案和数据审计记录，供大脑 AI 审查，不是论文正文。**推荐方案 A；A/B 预测器训练均未获本阶段授权，也未执行。** 当前 Stage14B 获授权的是三个冻结候选 Matched-NG-C 小头实验，应与以下未来端到端设计分别报告。正式选择规则以 [预登记计划](stage14b_preregistered_plan.md) 为准。

## 当前证据的评价范围

Stage14A 三折隔离的是排序器训练、归一化和 checkpoint 选择。冻结 Stage5A 候选生成器实际见过 official TRAIN700 全部场景；HeadTrain630 的每折 OuterTest210 均在它的训练范围内。其 checkpoint 使用 official VAL150 每 500 步的 full-horizon Overall minFDE6 选择；历史 VAL 还参与 Stage8/Stage9 的研究开发和假设形成。HeadDev70 也被重复用于历史排序方法开发。

因此，现有结果可称为“冻结候选条件下的内部排序 OOF”，不能称为独立端到端预测测试。重新训练并隔离 OuterTest 可以解决当前拟合过程的预测器训练重叠；它不能消除这些场景已经参与研究开发的事实，也不能把已使用的 VAL150 恢复为全新的确认集。已注册本地 trainval850 中没有核实到未被预测器训练或方法开发使用的评价场景。

元数据核验完整复现原划分：

| 原集合 | 场景数 | 实际用途及限制 |
| --- | ---: | --- |
| official TRAIN | 700 | 既有 Stage5A 正式优化使用全部 700 场景 |
| HeadTrain | 630 | 原三折排序研究集合；每个 scene 恰好作为一次 OuterTest |
| HeadDev | 70 | 原历史开发集合；推荐新方案明确隔离，不默默加入训练 |
| 每折 InnerTrain | 378 | 新预测器和排序头的拟合集合 |
| 每折 InnerDev | 42 | 新预测器和排序头开发选择，禁止产生模型优化梯度 |
| 每折 OuterTest | 210 | 对应折模型统一冻结后才能评价 |
| official VAL | 150 | 已用于历史选择/开发；A/B 不使用，不能再声称 pristine |

三个原 fold 的 378/42/210 token 列表逐项核对，并独立重算原 SHA256 排序算法；HeadTrain630 与 HeadDev70 互斥且并集严格等于 TRAIN700。每折可写为 **700 = 378 + 42 + 210 + 70**。完整 token 清单和各 scene 角色见 [outer 清单](00_protocol/stage14b_scene_design_outer.csv)，机器设计见 [scene design](00_protocol/stage14b_scene_design.json)。

## 方案 A：最小训练隔离的端到端内部 CV

每折新建一个与实际 Stage5A 相同结构的预测器，从随机初始化训练。只在原 InnerTrain378 优化，在原 InnerDev42 选择，最终用于该折所有方法的共同候选。对应 OuterTest210 和原 HeadDev70 均不参与预测器优化、checkpoint 选择、归一化拟合或方法选择。不得加载任意训练于全 TRAIN700 的历史模型作为初始化。

预测器保留实际的类型 embedding 和运动条件 residual decoder；不能将其称为未经适配的原始 HiVT，也不将这些历史组件包装为本次新的论文贡献。使用预登记的三折 seed 2022/2122/2222，batch16、AdamW、weight_decay=0.0001；fixed-scale warm-up 5000 步、lr=0.001，恢复本实验自己的 warm-up 最佳模型/优化器/RNG 后，原 NLL 最多 16000 步、lr=0.0001，总上限 21000。每 500 步只评价 Dev42，按 full-horizon Overall minFDE6 严格改善选择，NLL patience5；最终仅选 NLL checkpoint。所有预测器冻结后才生成用于统一方法比较的最终缓存。

然后每折新训练 NG-A、NG-C、G-A、G-C、Matched-NG-C 和 R2，合计 **3 次 HiVT 训练 + 18 次排序头训练**。G-A 保留用于完整结构×损失表。新预测器改变了轨迹、原 logits、图/R2 特征，不能把训练于旧候选的 Stage11B/14A 小头权重直接用于公平端到端比较。

每折所有比较方法必须使用相同新候选坐标和模式顺序；预测器原评分是无需额外头训练的参考。排序头归一化只拟合该折 InnerTrain，训练、损失和开发选择遵循原定义。R2 用该折新缓存重新训练；Bicycle 在各方法间均路由到这个同一折 R2，不能据此宣称 Graph 自身改善 Bicycle。每个候选必须保留唯一 scene/sample/instance 键、时间帧、预测器来源 SHA、该折分区以及完整评估 mask；完整 scene 的当前参与者和预测上下文不能按未来 GT 可见性筛除。

方案 A 的局限明确保留：排序器训练候选由见过这些训练 scene 的预测器生成，因此训练样本的预测残差可能比 Dev/Outer 更容易。这是 stacking 的训练/应用分布差异。OuterTest 完全排除于该折预测器和头的拟合仍然成立，但不能据此声称排序器训练数据已经具有 base-learner OOF 性质。

### TRAIN700 中剩余 70 个 scene 如何处理

推荐保留原 378/42/210 分工，原 HeadDev70 全部隔离。备选是每折将这 70 个 scene 加入预测器训练，形成 **448 + 42 + 210 = 700**。这一备选会改变数据量、原 controlled fold 的可比性及开发复用范围，且不会让历史开发 scene 变得全新。其完整 token 列表已经明确记录在设计 JSON 的 `AlternativeA448` 和 outer CSV 中；**它不是默认方案，也尚未授权。** 如未来选择 448，必须在拟合前单独批准、登记新数据角色并统一应用于所有方法，不能在看到 378 方案结果后无记录切换。

## 方案 B：额外的训练候选 OOF 审计

对每个外层折，把原 InnerTrain378 按预登记 `SHA256(2022|inner_oof|foldN|scene_token)` 排序（相同 hash 时按 token 字典序），分为三个 126-scene block。三个内层预测器各仅训练其余 252 场景，给自己排除的 126 场景生成候选；拼接后覆盖原 378 场景且每个 scene 恰好有一次训练 OOF 候选。内层模型使用固定 Dev42 作 checkpoint 开发选择，Dev42 不产生梯度，自己排除的 126、Outer210 和隔离70 均不得用于优化或模型选择。

另为每个外层折训练一个最终预测器，只拟合原 378 场景、只在 Dev42 选择，用于 Dev42 的头选择候选和 Outer210 的评价候选。不能用最终模型在训练 scene 上的输出覆盖已有训练 OOF 行。因此共 **12 次 HiVT 训练 + 18 次排序头训练**，其中 HiVT 是三折×（三个内层模型 + 一个最终模型）。清单见 [inner OOF 清单](00_protocol/stage14b_scene_design_inner_oof.csv)。

B 更严格地排除了排序器训练候选自身 scene 对 base learner 优化/选择的曝光，但存在额外限制：

- 内层模型只训练252场景，最终模型训练378，候选误差、原分数、六模式覆盖、交互距离和 R2 特征分布可能不同；不能预设 B 一定优于 A。
- Dev42 被多个内层/最终模型反复用于选择，它是开发数据，不是独立确认数据。需要在正式实现前明确完整选择与缓存流程，不能偷偷使用排除126的标签选择其预测器。
- 每个完整 window 的目标及全部邻居必须来自同一个内层预测器，不能把不同模型的目标和上下文预测混在一张图里。独立模型间的 mode index 不具有预先固定的物理语义；需审计排序网络是否共享逐模式评分、是否存在隐藏 mode-ID 依赖，以及模式排列/坐标和原分数是否正确配对。
- 需记录每行 OOF 源模型与排除场景、有效 target 缺失、输出有限性和三键身份。预测失败必须保留，不能为满足覆盖率改选另一个已看过该 scene 的模型。
- 内部 OOF 仍取自已参与历史研究的场景；增加模型数不会产生 pristine research holdout，也不会提供训练 seed 的独立重复证据。

## 方案比较与推荐

| 项目 | A：378训练、70隔离 | B：252内层 OOF +378最终模型 |
| --- | --- | --- |
| 新预测器训练 | 3次 | 12次 |
| 新排序头训练 | 18次 | 18次 |
| 对应 Outer210 的预测器训练/选择曝光 | 排除 | 排除 |
| 训练排序样本对自身候选生成器的训练曝光 | 存在，须披露 | 排除自身126，且自己标签不用于模型选择 |
| 候选源与实现复杂度 | 每折1个最终模型 | 每折3个训练 OOF 源 +1个最终源 |
| 主要分布风险 | base learner 训练内残差与外层残差不同 | 252内层与378最终模型输出不同，并反复使用 Dev42 |
| 统计解释 | 训练隔离、开发启发的端到端内部 CV | 同样内部 CV，另外提供 stacking OOF 审计 |
| 全新独立确认 | 不成立 | 不成立 |

**推荐先执行 A，作为最小的预测器训练隔离研究；B 作为额外预算下可选的 stacking 稳健性审计。** 这一推荐基于数据隔离需求、工程复杂度与资源成本，不来自新的 A/B OuterTest 指标。A/B 的选择必须在未来拟合前冻结。A 结果不理想时不得悄悄改用 B、加训练预算或调整损失；新增研究须单独登记和授权。

## GPU 时间、数据量与缓存成本

历史单 Stage5A 运行在 RTX3080/torch2.5.1+cu124：warm-up 3328.944730 秒、NLL 4355.208961 秒，合计 **2.13449小时/11500更新**；按 21000 上限线性延伸为 **3.89776小时/模型**。这是包括开发 forward、I/O 和 checkpoint 的单设备墙钟/GPU占用时间，不是 CUDA kernel 活跃时间。新 Dev42 工作量不同于原 VAL150，缩小训练 scene 集合会改变重复曝光次数和收敛；固定更新预算不能简单按场景数线性缩短。

原三折 G-A/G-C/R2 九头合计3697.624822秒，Stage14A 六个 NG-A/NG-C 合计1889.839373秒，15个已测小头合计 **1.55207小时**。未来18头中的三个 Matched-NG-C 尚未纳入这些实测值。仅为预算说明，历史三个 NG-C 的0.38417小时和三个 G-C 的0.78310小时作为额外三个 matched 运行的两个工作量代理；**它们不是 matched 模型的测量，也不是保证的上下界。** 正式预算应在当前 matched 运行完成后更新成本证据，且不得由新 OuterTest 结果决定训练预算。

| 成本锚点（小时） | A | B |
| --- | ---: | ---: |
| 新预测器：历史11500更新 schedule | 6.40 | 25.61 |
| 新预测器：21000更新上限线性代理 | 11.69 | 46.77 |
| 原15头实测代理 | 1.55 | 1.55 |
| 已知部分加候选 kernel 前向，未含3个matched头 | 8.00–13.29 | 27.24–48.40 |
| 加matched工作量代理后的说明性总量 | 8.39–14.07 | 27.62–49.18 |

以上不是置信区间或实际运行承诺，另需缓存I/O、特征构造、全场景forecast/context保存、开发检查、诊断和额外输出存储；不包括多seed/搜索。B 的12个预测器不能被写成“三次训练”。详见 [成本审计](00_protocol/stage14b_scene_design_cost_audit.json) 和 [比较CSV](00_protocol/stage14b_scene_design_cost_comparison.csv)。

700个已有训练 scene shard 共 **3.784GiB**，推荐每折378训练约 **1.96–2.07GiB**，42开发约 **0.22–0.25GiB**；场景原始文件只引用，不复制。所有378/252/126/42/210/70/448角色的精确窗口、full/partial actor 数与字节量见 [数据体积表](00_protocol/stage14b_scene_design_data_volume.csv)。三折评价 full-horizon actor-window 总数仍是260151，不能把“重复用于多个折的训练 occurrence”或1134条内层scene清单当作新的独立样本量。

仅对旧 TRAIN 缓存作文件大小统计：原700的 dense节点/边/评分/代价/target候选数组约 **6.343GiB**。按同一 dtype/shape 和 full-horizon target 数，三个外层折各一份合并 target cache 约 **17.067GiB**。A/B 可以各自合并成每折630场景的一份目标缓存；B 不必永久保留12份完整dense图缓存。估计未包含所有当前上下文参与者预测、来源记录、临时开发缓存、优化器快照和磁盘开销，实际新存储仍需获授权实施前核验。各方法必须共享候选，不能为四/五种方法复制一套相同缓存。

## 官方 nuScenes prediction 兼容性

当前结论为 **PARTIAL_NEEDS_ADAPTER_AND_PROTOCOL_RESET**，不是可直接提交的官方独立测试。单独核验见 [官方协议审计](00_protocol/stage14b_official_protocol_audit.json)。

官方 prediction helper 使用 challenge train500/train_val200/val150，并通过 `maps/prediction/prediction_scenes.json` 指定预测 target；本项目使用 original TRAIN700 内自定义 HeadTrain630 与完整三类 actor-window，二者不能因共享数据集名称就当作相同评价协议。[官方split实现](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py)

本地安装devkit的split计数已复核，但注册缓存及配置的原始路径没有找到完整 prediction target manifest，因此 `FullTrainValTargetManifestVerified=false`，尚不能核实官方 target universe。当前 hidden TEST 的完整注册数据和可用future标签未核实，不能将静态test场景名单当作可直接执行的本地测试数据。官方README描述 VAL leaderboard 与对 top entries 的 hidden TEST 运行；它不让已用于历史研究的 VAL 变成 pristine confirmation。[官方prediction说明](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md)

2秒历史/6秒12点时间长度可对接，但需要转换为官方 global XY、准确匹配 instance/sample token、按概率取top-k，并明确K6相对官方默认k1/5/10的报告范围；当前模型并没有10个真实候选。官方 Miss 使用概率top-k候选中 `max_t L2 >=2m` 的全时域判定再取最优候选，本项目 MR6 使用所有六候选中的endpoint `minFDE >2m`，两者不能互换。OffRoadRate 也须用其官方实现和地图坐标定义，不能替代成现有自定义道路proxy。[官方metrics](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/metrics.py)，[默认配置](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/configs/predict_2020_icra.json)

恢复完整target manifest、实现官方token/坐标/指标adapter与重新登记使用协议是未来工作。本阶段不执行 official VAL/test评价，不建立新prediction方法或端到端训练任务；未来官方确认需另行授权、明确数据权限和提交/评价流程。

## 执行前必须核验的工程条件

旧 Stage5A 训练入口直接读取 `SceneDataset('train')`/`SceneDataset('val')` 并断言16898/3603窗，不能原样用于378/42方案。未来实施只可在新阶段目录创建明确的fold子集/seed接口，复用原结构和损失、保持旧文件不变，核验每个优化batch、开发batch、归一化统计与checkpoint来源；否则会再次训练全部700或读取旧VAL。来源说明不能替代实际执行时的scene隔离证据。

运行前冻结scene列表、模型实现、原始初始化与seed、loss定义、normalization、batch ordering、warm-up/NLL转阶段、selection/stop规则与所有输入SHA。先训练子集的source/坐标/GT-poison、极小forward及梯度检查，再拟合；未来label只能作监督和离线评价。每个scene始终同fold，B每行训练OOF还须证明其完整window来自没有见过该scene的对应内层模型。

评价保持预登记的三项Overall Top1FDE主比较：G-C−NG-C、G-C−Matched-NG-C、NG-C−NG-A；同scene/actor/window配对，2000次整scene bootstrap、seed2022、每fold重采样210场景。family3 Bonferroni单比较98.333333%覆盖，percentile端点0.833333%/99.166667%；描述性95%CI和各fold方向同时报告。G-A及其他类型/运动组保持透明补充，不能事后升为主要比较。采样区间条件于固定训练结果，不代表重新训练seed变动，不能声称图消息的因果效果。

所有必需模型/配置/SHA冻结后才统一读Outer210；不得用其结果选择训练轮数、A/B、448备选、normalization或model参数。任一非有限、GT泄漏、scene交叉、候选身份变化、规范化越界、SHA异常或训练来源不符均保留失败记录并停止，不用历史模型填补缺失fold。本设计只记录未来路径，下一阶段训练与预算仍等待大脑 AI批准。

## 可复现证据与代码位置

- [原Stage5A训练入口74–80行](https://github.com/duchangchen11/Prediction_Hivt/blob/4daa4ae82557270e4f43881fa22c21e3ba6c4068/outputs/stage5a_motion_aware_decoder/03_training/stage5a_train.py#L74)给出TRAIN/VAL读取、窗口断言及采样器；[104–134行](https://github.com/duchangchen11/Prediction_Hivt/blob/4daa4ae82557270e4f43881fa22c21e3ba6c4068/outputs/stage5a_motion_aware_decoder/03_training/stage5a_train.py#L104)给出VAL选择与保存。
- [原SceneDataset与SceneSampler](https://github.com/duchangchen11/Prediction_Hivt/blob/4daa4ae82557270e4f43881fa22c21e3ba6c4068/outputs/stage3_multitype_hivt/00_manifest/stage3_common.py#L75)定义监督窗口过滤及scene/窗口采样；Stage14A来源审计重放原纯metadata AST，并核验全部700训练shard。
- [Stage6A来源限制](https://github.com/duchangchen11/Prediction_Hivt/blob/4daa4ae82557270e4f43881fa22c21e3ba6c4068/outputs/stage6a_future_interaction_reliability/09_reports/stage6a_final_report.md#L37)明确TRAIN700、HeadTrain/HeadDev与旧预测器曝光；[Stage9A报告](https://github.com/duchangchen11/Prediction_Hivt/blob/4daa4ae82557270e4f43881fa22c21e3ba6c4068/outputs/stage9a_type_adaptive_future_graph/09_reports/stage9a_final_report.md#L5)明确VAL开发复用。
- [Stage14A完整来源JSON](../stage14a_paper_graph_ablation/09_reports/stage14a_predictor_data_provenance.json)及本设计JSON记录输入SHA，scene token清单与成本来源。
- [stage14b_scene_design_audit.py](00_protocol/stage14b_scene_design_audit.py)只用现有Python的标准库读取metadata、原训练日志和文件大小，重算划分和成本表；不导入模型、不拟合、不执行forecast、不读取新的OuterTest结果。运行命令：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage14b_paper_validation/00_protocol/stage14b_scene_design_audit.py`。

推荐结论只涉及后续实验设计：**A先行，B可选，70明确隔离，448须单独授权；A/B均不是未使用过研究数据的独立确认集。** 当前Stage14B已授权小头阶段结束后提交和push，STOP等待大脑 AI审查。
