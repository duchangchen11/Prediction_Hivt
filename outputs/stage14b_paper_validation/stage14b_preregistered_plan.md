# Stage14B 正式实验预登记

本文件区分本阶段获授权的冻结候选容量对照与尚未授权执行的端到端实验设计。机器登记 [stage14b_registration.json](00_protocol/stage14b_registration.json) 于新模型 tiny、正式拟合和新 OuterTest 输出之前生成；固定结构与协议 SHA 见其中。历史 G-C/NG-C/NG-A 已知结果不能被包装成新的预登记发现。本阶段不会重新训练 HiVT，不会启动 Stage15。

## 模型与输入冻结

论文只主张 Future Interaction Graph 与 Normalized Regret-Aware Ranking 两项。候选生成器保留实际 Stage5A 网络，论文必须披露其类型 embedding、运动条件 residual decoder 等既有组件；不将其改称未经适配的原始 HiVT。冻结候选实验使用 Stage5A checkpoint SHA256 `88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`，K=6、过去2秒（5个含当前采样）、未来6秒（12点）、50m、最多8邻居、原15维节点/17维边。原轨迹、原 logits、GT标签、分组与三键身份保持字节一致。

Matched-NG-C 是一次固定的有效容量对照：原目标 node_encoder(15→64→64)、LayerNorm64、head(64→32→1)，增加目标自身 residual adapter(64→128→64, ReLU)。24,001参数，G-C为24,066，差65（0.2701%）。adapter固定 foldseed+101；共享模块沿用 G1 初始化；最后评分层零初始化。无真实邻居输入路径、无虚设参数。共同标量 score bias 对 softmax 的不可识别性如实披露。

## 本阶段冻结候选三折实验

每折 InnerTrain378/InnerDev42/OuterTest210，覆盖原 HeadTrain630；原 HeadDev70、official VAL/test 不用于新增拟合、选择或主评价。直接复用原 split、仅 InnerTrain 拟合的 normalization、128个 Fold1 tiny target、R2 checkpoint 与已冻结 G-A/G-C/NG-A/NG-C 输出。新模型只训练 Matched-NG-C 三次。

C损失：令 `c_ik=FDE_ik−min_j FDE_ij`，`s_i=max(1m,mean_k c_ik)`，`L=mean_i sum_k softmax(z_i)_k*c_ik/s_i`。FDE作为 detached监督标签，与模型输入分离。不能将它写为未归一化 expected error，也不加入温度、权重或新门槛。

AdamW，lr=0.001，weight_decay=0.0001，FP32，不使用 AMP；microbatch128×累积8，有效1024；连续epoch流、尾项carry；每epoch `default_rng(foldseed+epoch)` 排序，核对历史相同epoch前缀。最多50epoch，patience5。三折 seed=2022/2122/2222。选择严格最小 `Sdev=0.5*VehicleInnerDevTop1FDE/FoldR2VehicleFDE+0.5*PedestrianInnerDevTop1FDE/FoldR2PedestrianFDE`，第1epoch起可选。Bicycle最终使用原 fold R2，所有模型一致；训练目标仍保留三类。

先结构、有效参数、输入输出、共享初始化、邻居扰动不变、目标扰动有效、有限梯度、GT隔离检查，再原128目标300次更新 tiny：C损失降幅≥80%。任一项失败则保留失败记录并停止正式训练；不搜索结构或调参。tiny权重不用于正式拟合。全部三fold checkpoint、split、normalization、config SHA冻结后才允许统一 OuterTest。任何训练期间读取历史 OOF 数组或新增评价目录由 audit hook 拒绝。

## 未来端到端正式设计（只登记，不执行）

推荐方案A：每折从随机初始化重新训练同一个 Stage5A 结构的预测器，训练378、开发42、外层210、隔离70，总计700。预测器和所有排序头均完全排除 OuterTest；不能使用任何全700场景训练的旧 checkpoint 初始化。只允许训练场景产生梯度；候选 normalization 仅来自该折排序器 InnerTrain。

预测器 seed=2022/2122/2222，AdamW、batch16、weight_decay0.0001，原 fixed-scale warm-up5000步、lr0.001，随后原 NLL最多16000步、lr0.0001，总上限21000；开发评价每500步，NLL patience5。保留原 warm-up最佳模型、优化器与 RNG转阶段的机制；唯一选择数据改为 InnerDev42，指标为 full-horizon Overall minFDE6，严格改善；最终checkpoint只选 NLL。原 official VAL不参与。三折预测器都冻结后，每fold用同一个预测器生成该折所有方法的候选；不能跨方法更换候选或重新选预测器。

随后每折新训练 R2、NG-A、NG-C、G-A、G-C、Matched-NG-C，共18个排序头训练；原始 HiVT评分无需头训练。G-A保留为补充完整结构×损失表。头的原 loss、优化器、选择规则不变（R2保留其原673参数/原软CE/InnerDev Overall Top1FDE规则）。旧头不能复用成端到端公平比较。

增强方案B：每Outer折再在378训练场景按 `SHA256(2022|inner_oof|foldN|scene_token)` 排序均分126×3；每个OOF预测器仅训练其余252，给排除的126生成排序器训练候选；另训练一个378场景最终预测器用于Dev42/Outer210。全部模型仍排除Outer210和HeadDev70，Dev42只作开发，不产生梯度。共12次预测器训练；排序头仍18次。OOF样本量、有效target缺失、候选身份及内外模型分布差异必须报告，不因外层结果选择A/B。若选B，须在正式执行前另行确认计算预算及精确缓存/选择实现，不能从A结果不理想后悄悄升级B。

## 指标、统计及报告

主终点为 actor-window加权 Overall Top1FDE（m）。三项优先配对比较固定为 G-C−NG-C、G-C−Matched-NG-C、NG-C−NG-A，负值表示前者改善。冻结候选阶段后两项已知对照的重复使用明确标记；未来端到端阶段各比较使用新的共同候选。

次要：Overall、Vehicle、Pedestrian、MovingVehicle分别报告 minFDE6、Top1ADE、Top1FDE、HitRate。HitRate是所选mode是否等于endpoint-FDE最优mode（ties最低index），不是距离阈值命中率。MovingVehicle复用 t0 `vehicle.moving`；不按未来运动重新分组。保留Bicycle、Stopped/Parked/OtherVehicleState作为透明补充，不能凭Overall改善隐藏分组退化。排序只改变评分，minFDE6在相同候选下应严格不变，作为完整性检查。

统计单位为整scene，三fold分别抽210scene、配对保留所有actor/window，2000次、seed2022。主比较family3控制 FWER=0.05，Bonferroni单比较覆盖98.333333%，percentile端点[0.833333,99.166667]；同时报告描述性95%CI与各fold方向。支持某项改善须点估计<0、调整CI上界<0、至少2/3fold改善。次要分组和指标95%CI均标探索性，不追加显著性主张；不以actor为独立重抽样，不把固定head条件下bootstrap当作重训seed不确定性。报告统计关联与受控对照，不声称图消息的因果收益或新独立确认。

计算开销在同一 RTX3080/FP32 记录训练wall时间、峰值显存、实际参数、cached-input head延迟；固定128target、同一1024target池、20次warm-up、20次循环/三fold。未来端到端还应记录完整scene forward、特征构造与转移、最终类型路由，并单独列头开销；不能把仅头延迟当作整体FPS。另需离线开发检查官方token、global坐标、top5、全时域miss和OffRoadRate；官方结果作为单独协议，不将旧minFDE6自定义结果与论文榜单混表。

## 失败、重启和停止

非有限输出/梯度、GT输入泄漏、任意scene交叉、候选变更、norm越界、checkpoint身份或SHA不一致、历史文件变更均停止并如实标失败；不继续产生可用于比较的正式结果。基础设施中断只允许从记录的optimizer/RNG/carry/coverage恢复同一运行，不能换seed、重选split、加budget或按Outer调参。没有合格checkpoint则报告该fold缺失，不能以另fold或历史模型代替；不完整的三fold不做主成功判定。全部执行、失败、排除及重启日志保留。

本阶段检查通过后仅完成已授权的三次小头对照，六份报告提交并push，随后STOP，等待审查。方案A/B预测器训练和下一阶段尚未获授权；训练隔离历史开发场景也不构成全新独立确认集。
