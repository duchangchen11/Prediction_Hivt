# 项目约定

- 本项目固定工作目录为 `/home/lrj/Prediction_Hivt`，不要另建 `traffic_prediction` 工程。
- GitHub 仓库为 https://github.com/duchangchen11/Prediction_Hivt 。本项目代码、文档及适合版本管理的小型报告提交并推送到该仓库。
- 第一阶段要求见 `docs/stage_1_requirements.txt`。其中工程路径以本文件中的路径为准。
- 当前阶段补充要求见 `docs/stage_1_continuation.txt`；工程位置仍遵循用户此前指定的本目录。
- Stage 1 已验收。当前用户已授权 Stage 2，要求见 `docs/stage_2_requirements.txt`；在 `stage2/vehicle-hivt-baseline` 分支开发。保留官方 HiVT 核心架构和损失，仅适配 nuScenes vehicle、基础 lane centerline 与训练兼容接口。完成 tiny overfit、mini 与 trainval 单 batch 验证后停止，不进入 Stage 3。
- moving audit 已完成，记录见 `docs/stage_2_moving_audit_requirements.txt`。当前以 `docs/stage_2b_requirements.txt` 为准：冻结已有结果，依次做 K=6 fixed-scale、warm-up→官方 NLL；只有指定失败条件满足才测 bounded scale。确定协议后可重跑原16个 full tiny windows，Full Tiny PASS 后才可训练 mini。数据定义、网络结构及 K=6 冻结，不进入 Stage 3、不 merge main。
- Stage 2B 已完成；当前用户授权 Stage 2C，以 `outputs/stage2c_trainval_vehicle_baseline/00_manifest/stage2c_requirements.txt` 为准。在 `stage2c/trainval-vehicle-baseline` 开发，官方 train/val split，先10-scene smoke，再可恢复scene shards、batch benchmark、step-based warm-up→原NLL、VAL评价与scene bootstrap。新代码、配置和产物统一放在该阶段唯一根目录，文件名使用stage2c_前缀。保留原始输入与架构、旧实验；不使用test，不执行Stage3，不merge main。
- Stage 2C 已验收。当前授权完成整个 Stage 3A No-Type，要求见 `outputs/stage3_multitype_hivt/00_manifest/stage3_requirements.txt`，分支 `stage3/multitype-hivt`。仅扩展三类 taxonomy 与监督 target，保留完整 scene-window、旧模型架构和 Protocol 1；先数据审计、坐标检查、输入输出和三类 tiny overfit，PASS 后执行 Full Trainval、正式 VAL 评价、vehicle retention、案例图和报告。所有新代码和产物放在该阶段唯一根目录，不加入 type embedding、类别重权或动态交互，完成后停止，不进入 Stage3B，不 merge main；Stage2C 未提交的重绘文件保留原样且不提交。
- 优先检查并复用本机 nuScenes 数据及已有 Python/conda 环境；不下载大型数据、不自行创建新环境、不升级 PyTorch/CUDA。
- 不删除原始数据、不覆盖已有项目、不修改其他论文项目。
- nuScenes 数据只通过本地配置引用，不复制进项目或上传 Git；环境、凭据、缓存和大型生成产物不上传。
- 运行命令与实验结果保留在工程中；第一阶段在数据检查、窗口构造、坐标验证、可视化、scene 划分及 CV baseline 验证后停止，不开始 HiVT 改造。
