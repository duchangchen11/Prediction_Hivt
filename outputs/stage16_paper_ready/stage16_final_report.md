# Stage16 — Paper-Ready Evaluation 最终报告

**阶段交付已完成，等待大脑AI审查。** 本阶段补充分析不改变Stage15B预登记决策，不更改方法结构、不拟合校准、不训练HiVT。正式方法仍为Stage5A候选生成器 + Future Interaction Graph + Normalized Regret-Aware Ranking。

基础commit：`2deef55128730b02c1cf0bd9e7237a23d6865133`。分支：`stage16/paper-ready-evaluation`。全部新增产物位于本目录，历史内容和checkpoint逐SHA核验。

## 交付状态

| 项目 | 状态 |
|---|---|
| Stage15B结果冻结 | PASS：3个预测器、18个排序头保持原checkpoint选择；冻结文件3594个 |
| 外部排序方法 | COMPLETE：TNT评分组件、R-RNet概率组件，两种真实公开代码可行性审计；未拟合外部baseline |
| 初始化稳定性 | COMPLETE：24次新排序头拟合 +12个原头结果；每fold每模型3个初始化 |
| 概率质量 | COMPLETE：固定事件/15-bin ECE/Brier定义；无Outer校准拟合 |
| 科学总结 | COMPLETE：旧固定预测器与新场景隔离CV明确区分，保留改善/退化和协议局限 |
| 论文图表 | COMPLETE：6类主图 +2张补充图，8×SVG/PDF/PNG，源数据和审计齐全 |
| 新HiVT训练 | 0 |
| Stage15B指标/损失/模型改动 | 0 |

## 正式证据保持

| Comparison | Delta | BonferroniCILower | BonferroniCIUpper | NegativeFolds |
| --- | --- | --- | --- | --- |
| G-C-NG-C | -0.045382 | -0.068976 | -0.026489 | 3 |
| G-C-Matched-NG-C | -0.022523 | -0.035387 | -0.011727 | 3 |
| NG-C-NG-A | -0.069239 | -0.104990 | -0.031506 | 3 |

负差值有利于前一个模型，单位m。Stage15B原family3/2000 paired scene bootstrap结论保持原样。主图使用原七模型结果，不用新增seed挑选更好的正式结果。协议为nuScenes TRAIN630的自定义场景隔离3-fold内部CV，260,151个full-horizon actor windows；不是官方nuScenes test，也不是未参与历史方法开发的全新确认集。

## 排序头初始化敏感性

| Model | MeanTop1FDE | SDTop1FDE | MinTop1FDE | MaxTop1FDE |
| --- | --- | --- | --- | --- |
| NG-A | 2.370410 | 0.014471 | 2.354021 | 2.381427 |
| NG-C | 2.294171 | 0.013014 | 2.280598 | 2.306543 |
| G-C | 2.252439 | 0.007577 | 2.247489 | 2.261161 |
| Matched-NG-C | 2.281795 | 0.001855 | 2.279975 | 2.283684 |

| Comparison | FoldInitializationPairs | NegativePairs | PositivePairs | EqualPairs |
| --- | --- | --- | --- | --- |
| G-C-NG-C | 9 | 9 | 0 | 0 |
| G-C-Matched-NG-C | 9 | 8 | 1 | 0 |
| NG-C-NG-A | 9 | 9 | 0 | 0 |

均值/SD对三组预先对应的pooled初始化配置计算，每折仍是固定预测器；不是预测器跨seed稳定性。源表保留每fold和每seed的全部结果以及正向退化，不根据Outer挑seed。SD为ddof=1；补充方向计数复用相同场景和预测器，不能作为9次独立科研重复。详见 [初始化报告](stage16_seed_stability.md)。

## 概率质量

| Model | Top1FDE | Top1Probability | PredictionEntropy | HitRate | ECE15 | BrierScore |
| --- | --- | --- | --- | --- | --- | --- |
| R0 | 2.372741 | 0.239331 | 1.630392 | 0.451830 | 0.213904 | 0.761162 |
| R2 | 2.346574 | 0.277057 | 1.514176 | 0.455017 | 0.178037 | 0.737966 |
| NG-A | 2.375782 | 0.281256 | 1.496853 | 0.439276 | 0.158690 | 0.737091 |
| NG-C | 2.306543 | 0.879821 | 0.269722 | 0.464492 | 0.419706 | 0.935351 |
| G-A | 2.361641 | 0.299848 | 1.459550 | 0.440198 | 0.147497 | 0.731744 |
| G-C | 2.261161 | 0.942559 | 0.132957 | 0.470369 | 0.477292 | 0.988606 |
| Matched-NG-C | 2.283684 | 0.923259 | 0.175243 | 0.467475 | 0.460852 | 0.969263 |

校准事件为“所选mode等于固定六候选的endpoint-FDE oracle”，不是自然动作类别、未来密度或安全事件。Loss C改善选中轨迹误差，同时使softmax更集中、该oracle标签下的ECE/Brier恶化；论文应称其为normalized ranking scores，不能宣称已校准的未来概率。详见 [概率分析](stage16_probability_analysis.md)。

## 图表与源数据

- `fig1_architecture`：按真实代码绘制，包含原HiVT、类型embedding、运动条件decoder、原模式logits、G1重评分和Bicycle R2路由；GT仅为训练/Dev标签。
- `fig2_future_interaction_graph`：15维mode节点、17维候选对关系、最多8邻居/50m、每目标mode最多48个邻居mode的masked attention。
- `fig3_seven_model_performance`：原七模型Top1FDE/Top1ADE，共享candidate geometry，展示各fold与加权pooled mean。
- `fig4_two_by_two_ablation`：Graph×Loss2×2和原family3配对bootstrap区间。
- `fig5_multitype_groups`：Vehicle/Pedestrian/Bicycle/MovingVehicle，明确每组actor-window和scene数量。
- `fig6_improvement_failure_cases`：真实冻结缓存中4个改善/失败极端案例，GT、原六候选、NG-C/G-C选择保持相同axis和米制比例。
- `figS1_probability_quality`：明确oracle事件的可靠性图及Brier对比。
- `figS2_head_initialization`：条件于固定预测器/批次顺序的初始化敏感性。

`05_figures/stage16_figure_manifest.json`逐图列出源文件、导出SHA和解释；`stage16_figure_qa.json`核验vector可编辑文本、PDF单页、标签、真实case源SHA和原始数组到图表数值。`stage16_figure_legends.md`提供包含统计定义、样本量和局限的图注。`06_source_data/`保留aggregate CSV和provenance，`07_cases/`保留scene/sample/instance/node、实际时间与原context SHA。真实案例坐标CSV留在本机，不上传nuScenes原始数据；已明确这一可复现访问边界。SVG/PDF/PNG全部由Python生成。

## 实验资源与限制

| Device | PyTorch | HeadFits | HeadFitWallHours | PeakAllocatedBytes | DiskFreeBytesAtFinalization |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 3080 | 2.5.1+cu124 | 24 | 2.073104 | 155383296 | 25796878336 |

训练时间为小排序头的墙钟/GPU占用估算，包含Dev/I/O/checkpoint，不是纯CUDA kernel耗时；显存为allocated peak，不是完整预测系统部署内存。复用冻结候选、不复制19GB缓存，未新增环境或升级PyTorch/CUDA。外部方法成本仅为代码层面的预算估计，不能写成实测速度。

MovingVehicle剩余绝对误差和长尾较大，Pedestrian的gain小且存在折次退化，概率过度集中和大误差切换均已披露。原推理开销同时区分cached-head benchmark和完整工程路径，不宣称优化后的部署FPS。详见 [科学总结](stage16_scientific_summary.md)。

两个公开外部方法均为**评分组件适配候选**，不是完整论文复现。TNT需补history-only latent context导出；R-RNet需冻结坐标/padding适配并核查代码许可。当前没有公平拟合的外部scorer比较，不能将不同论文协议的数字当公平对照，也不能把现有结构消融称为全面SOTA验证。学术新颖性、真正全新确认数据和目标期刊要求仍需大脑AI决策。

## 审查结论

PaperReadyStatus = **READY_FOR_BRAIN_REVIEW_WITH_LIMITATIONS**。

交付的是完整的内部验证/补充分析/图表包，不是无条件的投稿接受或独立泛化保证。最终校验记录在 `00_manifest/stage16_final_audit.json`；push后以GitHub branch HEAD为准确CommitSHA，报告不嵌入自身commit以避免循环SHA。仅提交本Stage16目录，checkpoint/大型cache/环境/原始数据不上传，不merge main。

**完成提交与push后STOP，等待大脑AI审查；不启动任何下一阶段或新HiVT训练。**
