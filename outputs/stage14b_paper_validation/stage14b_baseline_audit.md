# Stage14B 基线与可比性审计

审计日期：2026-10-10（Asia/Shanghai）。历史基础commit为`4daa4ae82557270e4f43881fa22c21e3ba6c4068`。本文件只审计基线身份、可复用条件及独立验证所需工作，不提供新性能值，不执行新OuterTest/VAL/test评价。

## 1. 必须区分的两种实验问题

**固定候选的排序控制问题**：所有头共享Stage5A六条候选、原logit、actor身份、mask与fold；只比较排序机制和loss。Stage14A已有这一证据，图A/C输出通过actor三键从历史结果读取，Bike由同fold R2逐bit覆盖，共同几何指标不变。依据：[历史score身份映射](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:39)、[统一四模型评价](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:147)。

**端到端场景泛化问题**：评价场景必须从候选生成器优化及模型选择中隔离，排序头和标准化也必须服从同一隔离。现有固定Stage5A已经训练于官方700 scenes，覆盖所有630 ranking-OOF scenes；只重训排序头无法修复这项生成器重叠。重新划分旧场景可以建立训练隔离实验，但不会消除研究设计已看过这些场景/指标的历史事实。依据：[训练暴露重建](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:124)、[每fold重叠核验](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:144)、[独立验证范围](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:268)。

## 2. 基线身份、当前复用与重新训练要求

| 基线 | 实际计算与能回答的问题 | 固定Stage5A候选是否可复用 | 严格端到端新场景隔离需要什么 |
|---|---|---|---|
| Original HiVT / HiVT-64 nuScenes adaptation | 原局部/全局编码+原MLP decoder。与当前生成器比较时测试生成器增强整体差异，不是只测重评分图 | **不能**把Stage5A候选冒充original HiVT生成输出。Stage5A原logitargmax可作R0，但名称应为Frozen Stage5A original ranking | 按预登记任务/scene split从头训练原HiVT适配器，并在同一隔离评价池生成其自身候选；类别、horizon、map和评价对象需匹配 |
| Frozen Stage5A R0 | 直接以冻结生成器原mode概率argmax选择六候选；没有后置learned ranking head | 可以，原logit/概率已缓存；它是增强生成器基线 | 重新训练隔离场景的Stage5A，并重新生成cache，旧Stage5A权重/候选不能用于从其训练池抽出的“独立”场景 |
| R2 | 原始概率加权的未来候选交互摘要+19→32→1残差头，SoftCE | 可以使用对应fold原R2与标准化/输出；这是包含交互摘要的低参数排序基线 | 用隔离生成器的新候选重新提取R2特征，InnerTrain训练/标准化、InnerDev选择，不把旧全池训练/旧候选直接带入 |
| NG-A | 自身候选15维node encoder+LayerNorm+评分头；zero future-message；SoftCE | 可以复用已冻结Stage14A fold checkpoint与身份匹配结果 | 在新隔离候选上按同预算与初始化训练NG-A，适配同fold R2路由 |
| NG-C | 与NG-A相同结构；normalized expected regret | 可以复用已冻结Stage14A结果，保留完整训练和失败/诊断记录 | 同上，仅loss对应C，禁止用外层结果调整预算 |
| G-A | G1保留独立6×6候选关系，SoftCE | 可以，直接用Stage11B A，禁止重复训练后挑更好版本 | 在新隔离候选上训练同G1/A head；每fold的统计与选择仍只用允许的InnerTrain/Dev |
| G-C | 与G-A同图结构，normalized expected regret | 可以，直接用Stage11B C；是Stage14A四格比较的一格 | 同上，采用C，不改变边、loss scale、类权重或预设路由 |
| MatchedNG-C结构控制 | 目标node路径增加64→128→64 own adapter；24,001参数，对比G-C的24,066 | 候选、原logit、fold与标准化可以复用；新adapter的权重不能从G-C交互网络转移冒充已训练控制。须新初始化并完成指定tiny/训练/评价后才有结果 | 如主张端到端独立性，仍必须使用隔离生成器候选；参数接近并不修复旧生成器scene overlap |

表中模型身份依据：[original HiVT构造](/home/lrj/Prediction_Hivt/baselines/hivt_official/models/hivt.py:61)、[本地基础HiVT适配器](/home/lrj/Prediction_Hivt/models/hivt_nuscenes.py:13)、[Stage5A增强forward](/home/lrj/Prediction_Hivt/outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_model.py:7)、[R2特征](/home/lrj/Prediction_Hivt/outputs/stage6a_future_interaction_reliability/00_manifest/stage6a_features.py:12)、[G1](/home/lrj/Prediction_Hivt/outputs/stage8a_future_scene_compatibility_graph/00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py:30)、[NoGraph](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/02_models/stage14a_nograph.py:26)、[Matched NoGraph新实现](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/02_models/stage14b_matched_nograph.py:22)、[A/C原loss](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py:82)。端到端重新训练要求来自上述scene重叠事实与隔离目标，是本审计的实验设计推论，不是本阶段训练执行授权。

## 3. 哪些控制应保持

Stage14A的NG/G四格控制保持相同原始候选、原logit、fold、InnerTrain标准化、共享模块初始state、actor ordering、AdamW、lr/weight decay、InnerDev选择指标和早停规则；A/C分别原样使用原损失。其不同训练停止epoch是同一早停规则下的结果，不能事后根据OuterTest给某模型追加训练。依据：[共享初始化核验](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/01_preflight/stage14a_preflight.py:44)、[原训练配置/选择](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/03_training/stage11b_train.py:112)、[全部6checkpoint冻结gate](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:24)。

MatchedNG-C针对“G1比NoGraph多16,641参数”的剩余容量混杂。预登记结构只增加目标自身hidden上的有效adapter，不通过空参数padding凑量，也不读取/置乱邻居输入。其24,001参数是近似匹配，不是严格相等，和G1相差65（约0.2701%）；还应分别报告实际可训练数量及计算耗时。实现与计数：[adapter与总量断言](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/02_models/stage14b_matched_nograph.py:25)。当前结构和真实输入preflight已有PASS记录，包含新增adapter活跃性、原始图输入未改变及标签污染核验；这些检查不构成新性能结果，也不替代正式训练与统一评价。[结构审计](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/01_preflight/stage14b_model_integrity.json:3)、[真实输入审计](/home/lrj/Prediction_Hivt/outputs/stage14b_paper_validation/01_preflight/stage14b_capacity_preflight.json:2)。

Bicycle的固定fold-R2路由应在所有适用比较中保持相同，并清楚报告其输出是预设路由结果。另报告Vehicle、Pedestrian、Moving/Stopped/Parked分组；不能用Bike完全相等作为Graph自行车收益，也不能仅靠大量parked actor改善宣称所有运动状态均改善。依据：[原路由登记](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/00_manifest/stage11b_protocol.json:74)、[三种运动状态定义](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py:89)、[路由与完全相同检查](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:147)。

原四项主要比较A=NG-C−NG-A、B=G-A−NG-A、C=G-C−NG-C、D=G-C−G-A有明确family4修正和方向条件；新容量控制的结果不能事后替代其中某项成为历史主要比较，也不能改变过去Stage14A判定。依据：[四主要比较与配对bootstrap](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/06_bootstrap/stage14a_bootstrap.py:12)、[调整区间](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/06_bootstrap/stage14a_bootstrap.py:76)。新阶段的比较、停止规则与数据访问边界须另行在训练前登记。

## 4. 不能混成同一张公平性能榜的协议

| 比较维度 | 本项目历史排序协议 | 官方或外部协议核验要求 |
|---|---|---|
| 评价对象 | 630个开发训练池scene中的完整12步GT actor-window，Vehicle/Pedestrian/Bicycle；同一actor可出现在多个window | 官方prediction split使用`prediction_scenes.json`规定的sample/instance tokens；与本项目“所有完整actor-window”不是同一评价池 |
| 模式数量 | 6条固定候选 | 官方允许≤25条，默认指标含top1/5/10；6条候选可比较实际top1/5，不能因代码把k10截为6就称真正10模式结果 |
| 坐标和时间 | 当前cache使用t0 ego坐标，Th=5、Tf=12；图feature内部分速度使用固定6s | 官方提交为global xy，未来6s、2Hz、12个点，最多2s过去；须验证时间戳、变换和token顺序再导出 |
| Miss | 项目MR6为`min endpoint FDE>2m` | 官方Miss先取每候选全时域最大距离，判断`≥2m`，再在topK候选中取最优；阈值边界和时域聚合均不同 |
| Top1 / Oracle | 主指标是mode概率argmax的Top1FDE；另有共同minFDE6与oracle gap | 不能把外部oracle minFDE数字当作本项目Top1FDE，也不能把相同候选下的几何oracle常数说成排序头的生成能力 |
| 独立性 | 排序训练对应fold隔离，生成器已训练于全部OOF scene；VAL曾用于历史开发 | 官方VAL标签可用于devkit评价/排行榜流程，不代表方法开发未使用；官方top5后续hidden TEST流程存在条件，不应默认本项目直接可提交hidden TEST |

本项目口径源码：[完整actor-window定义](/home/lrj/Prediction_Hivt/outputs/stage3_multitype_hivt/02_preprocessed/stage3_dataset.py:79)、[Top1/Oracle/MR6计算](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:66)、[生成器/VAL来源](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:255)。官方协议primary sources由root核验：[prediction README](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md)、[split实现](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py)、[metrics实现](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/metrics.py)、[默认配置](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/configs/predict_2020_icra.json)。

官方prediction实现还把官方700 train scenes分为500 train和200 train_val，val是150 scenes，并从固定prediction token文件取评价对象；本项目700/630/70池和3fold378/42/210不是该官方prediction split。不能只因场景来自nuScenes、Tf同为12就认定官方可比。依据：[官方split源码](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py)、[原head pool重叠审计](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:141)。具体本机token可用性由Stage14B另行来源审计给出，本文不假设标签和完整正式split已齐全。

因此外部论文的数字只有在dataset版本、prediction tokens、split、horizon、K、概率排序、map/观测输入、评价器和训练暴露全部核对后，才可进入标为公平比较的表格。原始HiVT仓库中的默认20/30步以及只评价`agent_index`的逻辑本身就与当前5/12步、全部完整三类actor-window不同：[原HiVT默认与validation逻辑](/home/lrj/Prediction_Hivt/baselines/hivt_official/models/hivt.py:125)、[原默认history/future](/home/lrj/Prediction_Hivt/baselines/hivt_official/models/hivt.py:183)。不能直接引用其跨任务数字作为本项目baseline，也不将未经核验数字写入本审计。

## 5. 重新训练与复用的具体边界

1. 保留当前冻结候选和所有历史NG/G/R2权重，可以完成“对这套候选，近似匹配容量的own-only排序能否解释G-C收益”的受控补充。这个问题不需要改生成器，但必须保持“开发内部排序OOF”的标签与scene overlap披露。依据：[共同候选核验](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py:218)、[Overlap解释](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:261)。
2. 若采用新预登记scene隔离，将生成器优化限制在train/InnerTrain、checkpoint选择仅dev/InnerDev，随后重新生成该fold train/dev/outer预测，并在此候选上重新提取特征、fit normalization和训练R2/NG/G。旧checkpoint可作历史参照或初始化来源说明，但不能把已暴露评价scene的旧权重当作隔离生成器。依据：[历史三fold再训练假设](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:294)、[原fold数据入口](/home/lrj/Prediction_Hivt/outputs/stage11b_error_aware_ranking/00_manifest/stage11b_common.py:67)。
3. 如果目标是“生成器和方法开发均未使用”的确认性评价，应首先验证一个合规可用、带相应评价条件的未开发holdout或受控官方流程。既有registered trainval850没有已核验的这种池；单有test scene名称清单不能证明其GT/三类评价器可用。依据：[未开发评价池审计](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:268)。旧scene重新划分与bootstrap不会制造此前不存在的未开发数据。
4. 所有新训练、正式官方评价和端到端再训练须服从用户对阶段的明确授权及冻结gate；本基线审计只是设计和证据边界，不启动这些工作。

## 6. 给论文写作者的基线命名与结论限制

建议保留`Frozen Stage5A + original ranking (R0)`、`Frozen Stage5A + R2`、`NG-A`、`NG-C`、`G-A`、`G-C`和完成核验后才可列出的`MatchedNG-C`。名称必须指向真实生成器/排序器组合；“Original HiVT”单列真正原始适配模型，不把R0偷换成该名称。依据为第2节各实际forward。

“Graph在固定候选排序中有增量价值”“C在两种后置排序结构中具备正面证据”和“端到端预测器在独立场景上优于baseline”是三个不同判断。前两项由原登记四格控制约束，第三项需要另行符合场景隔离与协议兼容的证据。近似容量匹配可以减少当前参数数量混杂，但没有逻辑能力替代第三项，也不把统计关联写成碰撞安全性、联合预测一致性或因果保证。依据：[原比较/证据判定](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/06_bootstrap/stage14a_bootstrap.py:123)、[生成器重叠边界](/home/lrj/Prediction_Hivt/outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py:263)。
