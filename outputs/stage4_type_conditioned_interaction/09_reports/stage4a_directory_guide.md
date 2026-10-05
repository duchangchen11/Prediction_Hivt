# Stage4 独立目录

本阶段唯一输出根目录为 `outputs/stage4_type_conditioned_interaction/`。
按用户 2026-10-05 补充要求，Stage4 文件从 Stage3 目录逐 SHA256 一致迁移，
Stage3 目录中不再存放 Stage4 文件。

| 目录 | 内容 |
| --- | --- |
| `00_manifest/` | 要求、配置、预注册、冻结与迁移审计、模型和公共接口 |
| `00_manifest/stage4a_training_snapshot/` | 训练时五个源码文件的原始快照，哈希与训练前注册一致 |
| `01_data_audit/` | 保留的阶段数据审计位置；本轮沿用冻结数据 |
| `02_preprocessed/` | 冻结输入引用；本地缓存放 `stage4a_cache/`，不提交 Git |
| `03_type_interaction/` | tiny 和正式训练程序、曲线及阶段摘要 |
| `04_evaluation/` | 最终 VAL、配对、bootstrap、bias 和效率审计；大型 actor CSV 仅本地保留 |
| `04_evaluation/cases/` | 真实案例 JSON，作为正式图的唯一轨迹来源 |
| `05_figures/` | 可复现绘图程序、PNG/PDF/SVG、source CSV 和数值审计 |
| `06_tables/` | 主结果、消融、子组、bootstrap 和效率表 |
| `07_checkpoints/` | 本地 checkpoint 和可提交的小型 checkpoint manifest |
| `08_logs/` | 运行命令、真实日志及恢复记录 |
| `09_reports/` | 最终报告和报告生成程序 |

冻结的 Stage3 模型、配置、850 scene shards、baseline actor CSV 仍在
`outputs/stage3_multitype_hivt/`，由 Stage4 只读引用，不复制、不修改。
Stage4 正式结果见 [stage4a_final_report.md](stage4a_final_report.md)。

原始训练源码用于实验 provenance 核验；当前执行脚本在训练结束后仅适配独立目录的
导入和文件引用。模型源码保持逐字节一致，核心训练与模型计算函数通过 AST 对照。
原始日志中的旧启动路径作为历史记录保留。迁移记录见
[stage4a_relocation_audit.json](../00_manifest/stage4a_relocation_audit.json)。

绘图布局复核时可只读真实 JSON 重画，无需训练或重新推理：

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage4_type_conditioned_interaction/05_figures/stage4a_plot_cases.py --redraw-only
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage4_type_conditioned_interaction/05_figures/stage4a_plot_quantitative.py
/home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage4_type_conditioned_interaction/05_figures/stage4a_check_exports.py
```

Stage4A 已停止；本目录布局调整不授权 Stage4B 或重新训练。
