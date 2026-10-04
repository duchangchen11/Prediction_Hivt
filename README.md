# Prediction_Hivt

基于 nuScenes 的多交通参与者轨迹预测工程，为后续 HiVT baseline 和改进提供数据基础。

工作目录：`/home/lrj/Prediction_Hivt`

远程仓库：https://github.com/duchangchen11/Prediction_Hivt

第一阶段任务要求保存在 [原始要求](docs/stage_1_requirements.txt) 和 [继续执行要求](docs/stage_1_continuation.txt)。其中原定的 `traffic_prediction` 路径统一替换为本工程路径。

第一阶段已完成：现有环境检查、mini 数据读取、实际 taxonomy 与时间戳检查、5+12 窗口、当前 ego 坐标转换、单样本保存、轨迹图、scene 级划分和常速度 baseline。

原始数据和大型运行产物保留在本机；代码、文档及小型报告通过 Git 管理。

## 本机配置与运行

复用 `ped_intent`：Python 3.10.21、PyTorch 2.5.1+cu124，GPU CUDA 运算正常。仅补装 nuscenes-devkit 1.2.0 及其缺失依赖，现有包版本保持不变。未创建环境。

mini：`/home/lrj/datasets/nuscenes-mini`。trainval：`/media/lrj/54926A1D926A0438/nuscenes-trainval`，本阶段仅确认目录与 metadata 存在，未验证全部 trainval 传感器文件。

本机 `configs/local_paths.yaml` 不提交。其他机器先复制 `configs/local_paths.example.yaml` 并填写真实路径。使用环境中的 Python 运行：

```bash
conda activate ped_intent
bash scripts/run_stage_one.sh
# 查看 outputs/figures/one_window.png 后再运行：
python -m scripts.evaluate_cv
```

单独运行入口：

```bash
python -m preprocessing.inspect_nuscenes
python -m preprocessing.build_one_window
python -m scripts.validate_window
python -m preprocessing.visualize_window
python -m unittest discover -s tests -v
python -m scripts.evaluate_cv --max-per-scene 3
```

## 样本约定

- 所有位置使用当前 t0 自车平面坐标：原点为 LIDAR_TOP 关键帧对应 ego 位置，+x 为当前 ego heading，+y 为左侧。heading 为弧度。
- 交通参与者在 t0 选取，用 `instance_token` 和 annotation 的 prev/next 链关联；无最近邻关联、无轨迹插值、无未来信息估计历史速度。
- bicycle 单独映射 `vehicle.bicycle`；vehicle 为其余 `vehicle.*`，含 motorcycle；pedestrian 为 `human.pedestrian.*`。包含静止目标。完整 mapping 与零计数类别见报告。
- 使用实际时间戳：Th=5 大约 2 秒、Tf=12 大约 6 秒，并不假设帧间隔严格相同。
- 无效位置、heading、速度用有限零填充，必须结合 mask 使用。速度另有 `velocity_mask`；首个有效历史点无法估速。历史少于两个有效点的 actor 从 CV 指标中排除。
- `history_times`、`future_times` 是相对 t0 的实际秒数。样本同时保存原始时间戳、source annotation tokens、ego 原点和 yaw，便于反查。

## 划分与指标

先按 scene 分为 train/val/test，再在各自 scene 内构造窗口。mini 用 seed=42 的 6/2/2 自定义划分，仅检查工程流程；正式实验需要下一阶段采用官方划分。官方 devkit 的 split 定义见 [nuscenes/utils/splits.py](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py)。

每个 scene 只取最多 3 个均匀间隔的可用 t0，合计 30 个诊断窗口；未生成完整训练集。窗口可能重叠，actor-window 数量不代表独立目标数量。

ADE 先对单个 actor-window 的有效未来点求平均，再对 actor-window 等权平均。FDE 使用最后一个有效未来点；同时单独报告在最终请求帧有效的 `FDE_fixed_horizon`。汇总包含 train/val/test，只用于工程诊断；报告另列各 split 指标。

## 产物与停止位置

- [最终检查报告](outputs/reports/stage_one_report.md)
- [数据统计](outputs/reports/category_statistics.csv)、[时间检查](outputs/reports/sample_timing.txt)
- [样本摘要](outputs/reports/one_window_summary.json)、[坐标检查](outputs/reports/coordinate_checks.json)、[原始 metadata 回查](outputs/reports/serialized_window_validation.json)
- [可视化](outputs/figures/one_window.png)
- [CV baseline](outputs/reports/cv_baseline.md)、[JSON](outputs/reports/cv_baseline.json)
- [测试记录](outputs/reports/unit_tests.txt)、[命令记录](docs/execution_commands.md)
- `outputs/debug/one_window.pt` 留在本机，通过构造脚本可复现。

当前停止在 CV baseline。后续进入 vehicle-only HiVT 时仍需检查 PyG/Lightning 等依赖兼容性，当前尚未安装，未修改完整 HiVT。
