# Stage14B 最终报告

Stage14B完成方法审计、700场景来源/隔离设计、基线/官方协议审计和预登记。容量模型全部工程与tiny检查PASS后，执行了获授权的3个冻结候选小头对照并冻结；HiVT重训次数=0，未启动Stage15。历史3124个tracked文件、5个未提交Stage2C文件、35个旧checkpoint及冻结输入SHA全部保持。

## 审查问题的直接回答

1. **两个方法在代码中真实存在：YES。** G1具有候选节点与候选对attention消息、残差模式评分；C实际为归一化expected regret。生成器仍含类型embedding和运动条件decoder，R2含概率加权邻居摘要，必须按实际组件写论文。源码位置见 [method audit](stage14b_method_audit.md)。存在实现不等于已经验证文献新颖性或独立泛化。
2. **参数量增加不足以解释G-C优势：SUPPORTED（近似容量对照范围内）。** NG-C7425/G-C24066原本有容量混杂。Matched-NG-C24001仅用目标自身特征，G-C−Matched Overall Top1FDE=-0.058842m、family3 CI=[-0.078546,-0.041768]、负向fold=3/3；按预登记规则解释，不将非显著写为等价，也不声称因果。完整表见 [capacity control](stage14b_capacity_control.md)。
3. **预测器与排序器场景隔离：** 每fold随机初始化Stage5A结构，训练378、开发42，完全排除对应Outer210，隔离历史HeadDev70；各方法用同fold共同候选，重提特征/仅InnerTrain标准化，全部头重新训练并先冻结再Outer评价。旧全700预测器及旧头不能冒充隔离模型。旧入口硬编码700/VAL150，未来须新目录中的fold/seed入口与逐batch来源检查，不改旧文件。
4. **推荐基础A。** 它解决当前最关键的预测器训练重叠且成本较低。B额外审计训练候选OOF，但存在252/378模型输出分布差异、重复Dev选择与更复杂来源证明；不预设其更优，不根据Outer结果切换。详见 [evaluation protocol](stage14b_evaluation_protocol.md)。
5. **需要重训多少次HiVT：** A=3；B=12（每Outer3个内层+1最终）。两方案完整比较均另需18个小头训练（NG-A/NG-C/G-A/G-C/Matched-NG-C/R2每fold）。共同生成器原分数R0无需头训练；真正OriginalHiVT生成器baseline需额外3次预测器，未包含A/B数字。[命名/seed澄清](00_protocol/stage14b_design_clarifications.md)
6. **当前数据不满足全新独立确认要求。** 700TRAIN全部被旧预测器训练，630池与70开发池有研究复用；150VAL用于历史选择和开发。本地注册850没有已核实pristine确认池。新隔离CV可检验拟合隔离条件下的内部端到端性能，不能把旧开发scene改称新独立数据；hiddenTEST本地目标/标签可用性未证实。
7. **现有基线支持内部机制消融，尚不足支持公平官方/SOTA或独立端到端优越性。** R0、R2、NG/G×A/C及matched有用；当前“original mode scores”属于增强Stage5A，不能冒充原HiVT网络。若主张与外部预测模型公平竞争，须统一split/target/horizon/K/map/metric并重新运行合适基线，禁止直接混入不同协议论文数字。[baseline audit](stage14b_baseline_audit.md)
8. **正式资源预估：** RTX3080单设备历史墙钟锚点，加入本次三matched头实测 0.411 h，A约 8.41–13.70 h，B约 27.65–48.81 h；不是CI或承诺，另需缓存I/O、图构造、诊断和缓冲，也不含真正OriginalHiVT额外3次。当前设备显存实测 10240 MiB（10 GiB），同一设备已完成历史训练；Stage5A历史 1735.690918 MiB 峰值仅是forward测量，不能代表训练峰值或保证未来训练适配，未来fold入口仍须验证训练显存。本阶段小头的CUDA显存也不能替代完整预测器训练测量。见 [历史效率表（报告第175行）](../stage5a_motion_aware_decoder/09_reports/stage5a_final_report.md)。现有shard约 3.784 GiB 只引用；三fold合并目标缓存说明性估计 17.067 GiB，额外上下文/临时/optimizer空间待实施核验。历史代理及更新实测分别见 [设计成本](00_protocol/stage14b_scene_design_cost_audit.json)、[更新预算](00_protocol/stage14b_updated_resource_estimate.json)。
9. **下一阶段正式启动条件：NOT_YET。** 方法/低成本控制和设计可审查；仍需大脑AI审查与下一阶段授权、独立于旧入口的fold训练接口及GT/来源小检查、资源确认。若要求官方或pristine确认，还缺完整官方targets、global导出/官方指标adapter及未开发确认数据/流程。本阶段结束并STOP，不实施这些后续工作。

## 容量对照与冻结事实

新matched三fold选中epoch=[15, 14, 2]，执行epoch=[20, 19, 7]；仅InnerDev选择。tiny降幅83.40%，新增adapter梯度有效，未来GT/raw-mask污染forward差0，邻居扰动差0。全部3checkpoint冻结后评价260151个唯一scene/sample/instance target，630scene；共同oracle几何和Bicycle R2逐bit一致。统计保持2000配对整scenebootstrap、seed2022、family3；历史两contrast已知，新增capacity结果未用于结构/预算/选checkpoint。原Stage14A判定不改。补充如实报告Parked的点退化、OtherVehicleState4315目标，以及HitRate与Top1FDE的差异；不概括为全部分组/指标均改善。

## 官方协议结论

`OfficialProtocolCompatibility=PARTIAL_NEEDS_ADAPTER_AND_PROTOCOL_RESET`。本机devkit确认challenge train500/train_val200/val150，但完整prediction_scenes.json未找到。6秒2Hz12点、2秒历史可对接；还需globalXY、指定token、概率top1/5、官方全时域Miss≥2m和OffRoadRate。项目endpointMR6>2m及自定义full-target池不可替代官方定义。VAL leaderboard与hiddenTEST后续流程不意味着旧VAL独立。依据：[官方prediction README](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md)、[split](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py)、[metrics](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/metrics.py)、[本机核验](00_protocol/stage14b_official_protocol_audit.json)。

## 交付与停止

六文件：[method audit](stage14b_method_audit.md)、[evaluation protocol](stage14b_evaluation_protocol.md)、[capacity control](stage14b_capacity_control.md)、[baseline audit](stage14b_baseline_audit.md)、[preregistered plan](stage14b_preregistered_plan.md)、本报告。原计划SHA保留，方法命名/未来seed澄清单独存档，参数数量与实际有效路径分开审计。

GitBranch=`stage14b/paper-validation-design`。基准commit=`4daa4ae82557270e4f43881fa22c21e3ba6c4068`，预登记本地commit=`461e446`、远端commit=`00998ed1fc12a0273f293781e2837bf37aefbef0`；最终commit由该分支HEAD核验（避免报告自嵌最终SHA循环）。所有新增内容位于 `outputs/stage14b_paper_validation/`，大型候选/模型/日志留本机，不上传原始数据、环境或凭据；不merge main。

Recommendation=`A_FIRST_AFTER_REVIEW_AND_ISOLATION_PREFLIGHT`。当前没有训练HiVT、覆盖历史checkpoint或启动Stage15。提交和push后STOP，等待大脑AI审查。
