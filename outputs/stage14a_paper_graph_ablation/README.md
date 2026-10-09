# Stage14A 小论文补充实验

本阶段的全部新增代码、协议、表格、图像和报告集中在这个目录，历史实验保留原位置。研究基础为 Stage13A commit `88b83b7114e72b91544d4566cab26700c1b452dc`，实验分支为 `stage14a/paper-graph-ablation`。

优先查看：

- [完整实验报告](09_reports/stage14a_final_report.md)
- [预登记判定结果](09_reports/stage14a_scientific_decision.json)
- [整体与分组消融指标](05_evaluation/stage14a_ablation_metrics.csv)
- [四项受控比较与场景 bootstrap](06_bootstrap/stage14a_bootstrap_ci.csv)
- [模式切换与高代价损害](07_diagnostics/stage14a_mode_switch.csv)
- [候选预测器的数据来源审计](09_reports/stage14a_predictor_data_provenance.json)
- [论文图像与源数据](08_figures/)

四模型为 NG-A、NG-C、G-A、G-C。只新增 NG-A/NG-C 的三折训练，共六次正式训练；G-A/G-C 直接复用 Stage11B。Stage5A 预测器、六条候选轨迹和原始模式分数冻结。Bicycle 统一采用既定 fold R2 路由。

评价名称是 **nuScenes HeadTrain630 internal three-fold ranking OOF**。这些 OuterTest 场景曾参与冻结预测器的训练，因此不能作为端到端独立泛化验证。没有在本阶段进行 official VAL/test 评价或重新训练 HiVT。

目录职责：

| 目录 | 内容 |
| --- | --- |
| 00_protocol | 用户要求、冻结协议、历史与数据 SHA256、运行定义 |
| 01_preflight | 输入、损失、梯度、初始化与 tiny 完整性检查 |
| 02_models | 真正关闭交互消息的 NoGraph 模型 |
| 03_training | 六个训练配置、曲线、样本顺序和训练汇总 |
| 04_checkpoints | 本地 checkpoint、SHA256 清单及六模型冻结 gate |
| 05_evaluation | 完整消融表、每折指标、身份一致性核验及本地 OOF 缓存 |
| 06_bootstrap | 2000 次场景配对 bootstrap、预登记比较及描述性交互 |
| 07_diagnostics | 模式切换、运动状态贡献与推理开销 |
| 08_figures | 可编辑 SVG、PDF、PNG 预览、源数据和绘图核验 |
| 09_reports | 最终报告、来源审计与完整性总结 |

所有 `.pt`、`.npy`、缓存和大日志留在本地，不上传 Git。SHA256 清单、代码及适合版本管理的报告和图表上传到本阶段分支，不 merge main。

本阶段完成并上传后停止，等待大脑 AI 审查。论文写作与更广泛的科研判断由大脑 AI 负责。
