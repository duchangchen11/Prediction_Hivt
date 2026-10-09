# Stage15A 排序接口兼容性

RankingCompatibility = PASS，涵盖随机初始化及少量NLL更新后候选，使用新随机排序头，无历史排序头权重、无排序头optimizer更新。[stage15a_ranking.py:70](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/05_candidate_interface/stage15a_ranking.py#L70)先对全当前window按原GT-free Stage8特征生成节点/边，再按监督资格选择训练target；未按未来GT筛除邻居。

| 方法 | 参数量 | 输入/固定角色 |
| --- | --- | --- |
| R2 | 673 | [N,6,19]历史/候选/概率加权预测邻居特征 |
| NG-A / NG-C | 7425 | shared目标15维特征，仅目标自身路径 |
| G-A / G-C | 24066 | 节点[N,9,6,15]；边[N,6,8,6,17]；8邻居mask |
| Matched-NG-C | 24001 | 目标自身64→128→64 adapter，原capacity control结构 |

K6、t0≤50m/最多8邻居、独立6×6未来候选关系保持；原logits为独立[N,6] residual基准，节点column3原logit与同mode匹配。G1 map输入为None，实际结构没有语义/map message，不能加入新语义模块。R2/图特征函数与原Stage11B graph_normalize/A/C loss AST直接复用，避免导入历史stage写hook、旧缓存和全700预测器。

[stage15a_ranking.py:96](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/05_candidate_interface/stage15a_ranking.py#L96)只接受该fold InnerTrain，人口std(ddof0)+1e-6，图节点continuous3–14与真实边0–10按原valid mask；one-hot/flags不变，padding归零，原logits residual不归一化。R2缺失邻居distance归一化后设1、冲突设0的约定保留。Dev拟合norm明确拒绝。本阶段统计仅来自少量train窗并标formal_eligible=false，未来必须在新预测器冻结候选上重算整个378场景norm，不能使用这些检查统计或旧norm。

六头zero-init产生与新原logits完全一致的输出；非零随机评分fixture的共同mode置换前后maxdiff<1e-6，证明候选、原分数与边target/neighbor模式配对正确。fixture仅用于检查，没有改变历史模型。训练target上的原A/C或R2 loss有限且头反向有梯度，预测器没有梯度与状态变化；GT/未来mask扰动后全部graph/R2输入maxdiff0。

Bicycle未来统一路由到同fold重新训练R2；post-update接口检查实际Bicycle与固定路由fixture均通过。详见 [Fold1 trained接口](03_checks/stage15a_fold1_trained_interface.json)、[Fold2](03_checks/stage15a_fold2_trained_interface.json)、[Fold3](03_checks/stage15a_fold3_trained_interface.json)。候选payload无GT，标签另存sidecar；绑定scene/sample/instance、fold/seed/role、predictor SHA、历史mask、frame原点/方向、timestamp元数据。只保存少量train/dev候选，未生成Outer候选。

正式方案A每fold必须新训R2、NG-A、NG-C、G-A、G-C、Matched-NG-C，共18头；共同新预测器候选与mode顺序完全相同。不得直接复用旧头作为公平端到端模型，minFDE6仍是冻结共同候选的完整性指标。
