# Stage15A 场景隔离审计

SceneIsolation = PASS。方案A严格复用 Stage14B 的 scene token 与原顺序，每折378训练/42开发/210外层排除/70隔离，700场景全部对账；不采用448备选。本阶段仅工程 preflight；未启动完整三折预测器训练，未运行 OuterTest forward、性能评价或模型选择。30次更新均为带 `PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING` 标记的检查，不可作为正式模型或下一阶段初始化。

| Fold | Seed | TrainScenes | DevScenes | OuterExcluded | HeadDevExcluded | TrainWindows | DevWindows | OptimizerChecks | ForwardMean_s | StepMean_s | RecoveryMaxDiff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022 | 378 | 42 | 210 | 70 | 9137 | 1017 | 13 | 0.049511 | 0.343626 | 0.0 |
| 2 | 2122 | 378 | 42 | 210 | 70 | 9103 | 1018 | 9 | 0.063590 | 0.529917 | 0.0 |
| 3 | 2222 | 378 | 42 | 210 | 70 | 9104 | 1011 | 8 | 0.062628 | 0.433024 | 0.0 |

显式列表见 [split integrity](01_data_isolation/stage15a_split_integrity.json) 和 `01_data_isolation/stage15a_foldN_{InnerTrain,InnerDev,OuterTest,QuarantinedHeadDev}.json`。列表 SHA、fold、seed、角色和完整顺序由 [stage15a_common.py:119](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/00_manifest/stage15a_common.py#L119) 校验。只读取原 `stage3_train_index.csv` 的元数据，不调用硬编码 train/val Dataset；训练与开发分别引用原378/42的shard，不复制原数据。

[stage15a_common.py:182](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/00_manifest/stage15a_common.py#L182) 检查每个实际优化batch仅含InnerTrain；开发forward只允许InnerDev。Outer/HeadDev shard打开、Outer Dataset创建、Dev优化、VAL index读取、旧TRAIN700权重、错seed、交换列表、历史输出路径均有明确拒绝探针。失败探针在读入payload前被拒绝；runtime JSON 的 ReadPaths 包含打开尝试，必须结合 BlockedNegativeProbes 和 LoadedPayloads理解，不能当作成功读取。

运行来源与batch证据见 [Fold1](01_data_isolation/stage15a_fold1_runtime_isolation.json)、[Fold2](01_data_isolation/stage15a_fold2_runtime_isolation.json)、[Fold3](01_data_isolation/stage15a_fold3_runtime_isolation.json)。新NLL候选来源另见 `03_checks/stage15a_foldN_trained_interface.json`。所有加载的shard SHA与历史manifest一致。

原始scene/sample/instance、annotation链、17帧时间戳和ego/global坐标对账了每fold4个训练/开发窗口。SQL只查询明确选中的train/dev键；原数据库只读。缓存相对时间与源 timestamp 精确一致，坐标误差均<1e-4m。模型仍使用5历史/12未来关键帧和名义2s/6s定义，不插值或重采样。

Fold2初次审计在优化前被额外0.15s名义时长检查挡住；真实future末帧可为6.248027s且缓存与源完全一致。该门槛未在Stage14B协议登记，已移除并改为精确源时间戳匹配与真实jitter记录。保留 [失败证据](07_logs/stage15a_fold2_timestamp_failure.json)、[原审计源码](07_logs/stage15a_checks_initial_timestamp_audit.py)、[审计修正登记](00_manifest/stage15a_audit_correction_registration.json)。训练源码、超参数、原数据及Stage14B协议未修改，失败尝试优化更新0。

场景训练隔离的工程条件已通过。历史TRAIN/VAL用于方法开发这一事实保持，现有场景仍不能称为全新独立确认集。
