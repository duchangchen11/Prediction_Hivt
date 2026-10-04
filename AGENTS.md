# 项目约定

- 本项目固定工作目录为 `/home/lrj/Prediction_Hivt`，不要另建 `traffic_prediction` 工程。
- GitHub 仓库为 https://github.com/duchangchen11/Prediction_Hivt 。本项目代码、文档及适合版本管理的小型报告提交并推送到该仓库。
- 第一阶段要求见 `docs/stage_1_requirements.txt`。其中工程路径以本文件中的路径为准。
- 当前阶段补充要求见 `docs/stage_1_continuation.txt`；工程位置仍遵循用户此前指定的本目录。
- Stage 1 已验收。当前用户已授权 Stage 2，要求见 `docs/stage_2_requirements.txt`；在 `stage2/vehicle-hivt-baseline` 分支开发。保留官方 HiVT 核心架构和损失，仅适配 nuScenes vehicle、基础 lane centerline 与训练兼容接口。完成 tiny overfit、mini 与 trainval 单 batch 验证后停止，不进入 Stage 3。
- 优先检查并复用本机 nuScenes 数据及已有 Python/conda 环境；不下载大型数据、不自行创建新环境、不升级 PyTorch/CUDA。
- 不删除原始数据、不覆盖已有项目、不修改其他论文项目。
- nuScenes 数据只通过本地配置引用，不复制进项目或上传 Git；环境、凭据、缓存和大型生成产物不上传。
- 运行命令与实验结果保留在工程中；第一阶段在数据检查、窗口构造、坐标验证、可视化、scene 划分及 CV baseline 验证后停止，不开始 HiVT 改造。
