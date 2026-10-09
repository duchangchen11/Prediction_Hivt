# Stage15A 最终报告

Stage15A工程检查完成。基础commit `7b14f630fa42a4320214e8c6c63ab8794b48d43b`，分支 `stage15a/isolated-predictor-preflight`，新增内容全部位于本目录。本阶段仅工程 preflight；未启动完整三折预测器训练，未运行 OuterTest forward、性能评价或模型选择。30次更新均为带 `PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING` 标记的检查，不可作为正式模型或下一阶段初始化。

| 项目 | 结论 |
| --- | --- |
| SceneIsolation | PASS：逐fold378/42/210/70，原scene token/顺序精确一致 |
| TrainingInterface | PASS：显式fold/seed/list/output；完整预算循环已实现，formal gate关闭 |
| PredictorInitialization | PASS：650403参数fresh Stage5A；fold1原step0完全一致，3seed不同 |
| GTLeakageAudit | PASS：未来GT物理剔除；扰动raw/logits/prob与graph/R2 feature maxdiff0 |
| RankingCompatibility | PASS：六种新随机头，初始与更新后NLL候选K6/15维节点/17维边/19维R2 |
| CheckpointRecovery | PASS：三fold下一步模型、AdamW、batch/cursor及四类RNG逐位一致；NLL strict-load通过 |
| GPUResource | PASS：最大16窗1633actor训练峰值allocated4.520GiB，CUDA可见9.654GiB |
| StorageResource | PASS_WITH_SHARED_STREAMING：free43.13GiB，新增峰值设计22.58GiB＋8GiB余量 |
| Stage15BReady | ENGINEERING_PASS_PENDING_REVIEW_AND_AUTHORIZATION |

资源排程继续参考约8.41–13.70已知组件GPU小时；未来正式方案A需3次新预测器、18次新头，缓存构造与I/O另计。本阶段没有执行正式预测器训练或排序头拟合，不可将30次检查更新描述成已完成3个正式预测器。

保留一次Fold2优化前时间审计失败：本地额外0.15s名义时长门槛误拒原始timestamp jitter；已改为逐点源timestamp精确一致性，无数据/训练/超参数改变。修正登记与原源码/失败记录可复核。5/12帧的名义2/6s并非每条样本严格均匀时间；官方协议仍需单独token/坐标/指标/采样adapter与协议重置，不能提前宣称official兼容或全新独立确认。

全部历史文件/38 checkpoint/5未提交Stage2C文件SHA保持。新小样本CP、候选和标签缓存只保留本地；可版本管理源码、配置、split列表、日志证据及本六份报告提交推送，不merge main。

六份交付：[数据隔离](stage15a_data_isolation_audit.md)、[训练入口](stage15a_training_interface_audit.md)、[模型完整性](stage15a_model_integrity.md)、[排序兼容](stage15a_ranking_compatibility.md)、[资源](stage15a_resource_report.md)、本最终报告。执行/重放说明见 [README](README.md)；机器审核见 [final audit](00_manifest/stage15a_final_audit.json)。

Stage15BReady只表示当前工程检查通过。下一阶段仍等待大脑AI审查、单独正式训练授权及新阶段输出/源码登记；正式模型必须从fresh初始化开始，不能使用本阶段tiny CP/norm或旧TRAIN700/排序头。完成后STOP，未启动完整三折训练。
