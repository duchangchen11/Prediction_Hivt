# Stage14B — Paper Method Freeze and Evaluation Design

所有新增代码与报告独立保存于本目录，历史阶段保留原位。起点：Stage14A `4daa4ae82557270e4f43881fa22c21e3ba6c4068`；分支：`stage14b/paper-validation-design`。

论文仅保留 Future Interaction Graph 与 Normalized Regret-Aware Ranking 两项贡献；按实际实现披露既有生成器及类型路由。主要交付如下：

| 文件 | 内容 |
| --- | --- |
| [stage14b_method_audit.md](stage14b_method_audit.md) | 实际网络、图输入/评分、loss、GT和参数的源码证据 |
| [stage14b_evaluation_protocol.md](stage14b_evaluation_protocol.md) | TRAIN700与三折隔离，A/B成本和数据独立性 |
| [stage14b_capacity_control.md](stage14b_capacity_control.md) | 24001参数own-only控制、预检/tiny及三折小头结果 |
| [stage14b_baseline_audit.md](stage14b_baseline_audit.md) | 基线身份、复用/重训边界与官方协议差异 |
| [stage14b_preregistered_plan.md](stage14b_preregistered_plan.md) | 新Outer输出前冻结的选择、统计、失败和停止规则 |
| [stage14b_final_report.md](stage14b_final_report.md) | 九项审查问题的回答、资源估计和启动条件 |

`00_protocol`包含机器登记、历史SHA、精确scene清单、成本及审查记录；`01_preflight`包含真实GT隔离/梯度/tiny；`02_models`为容量模型；`03_training`只训练匹配排序头；`04_checkpoints`记录先冻结门与SHA；`05_evaluation`记录同身份指标与头效率；`06_statistics`为配对scene bootstrap；`07_logs`保留本机执行日志。

历史候选生成器已训练于TRAIN700，VAL曾参与开发。新容量结果属于冻结预测器条件下的内部排序OOF。A/B端到端重训仅设计，真正Original HiVT网络基线不等于Stage5A R0；详见 [设计澄清](00_protocol/stage14b_design_clarifications.md)。本阶段HiVT重训0；不覆盖旧checkpoint、不merge main、不启动Stage15，提交push后STOP等待审查。

大型缓存、checkpoint、完整日志保存在本机并由 `.gitignore` 排除；Git包含代码、协议、来源证明、小表和报告，不包含nuScenes原始数据、环境或凭据。

GitHub审阅源码时可使用 [代码证据索引](00_protocol/stage14b_code_references.md)，其中历史代码链接固定至Stage14A commit，并提供对应行号。
