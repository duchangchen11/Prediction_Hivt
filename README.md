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

## Stage 2：vehicle-only HiVT

本分支 `stage2/vehicle-hivt-baseline` 已推进到 tiny-set 验收，**OVERFIT=FAIL**；按用户要求停止后续训练，mini baseline 尚未运行，Stage 3 不可进入。[Stage 2 报告](outputs/reports/stage_two_report.md)。上面的环境/停止位置描述保留为 Stage 1 记录。

冻结作者官方 HiVT 提交 `6876656ce7671982ebdc29113aaaa028c2931518`，保留 Apache-2.0 源码及文件哈希。运行组件仅调整包导入、PyG batching 签名、PyTorch causal keyword；核心 encoder/global interactor/MLP decoder 和 loss 保持。使用原生 PyTorch wrapper，PyG 2.6.1 及 scatter/sparse 的 CUDA12.4 wheel 已安装，无需 Lightning。

vehicle adapter 和 lane/connector centerline 输入已完成：当前 ego 坐标、5/12 帧、K=6、HiVT-64、每个 actor 50m 地图半径、2m centerline 间距。mini 为 146/48/50 个 train/val/test windows，沿用 Stage 1 scene split。全窗口和随机100窗口检查通过。trainval 流式抽样5个完整scene、四个地图读取、batch size1/2前后向及optimizer step均通过，未读任何原始点云。

```bash
conda activate ped_intent
python -m scripts.audit_trainval
python -m scripts.prepare_hivt_mini
python -m scripts.hivt_smoke_test
python -m unittest discover -s tests -p test_hivt_stage_two.py -v
python -m scripts.train_hivt_stage2 --mode tiny
# 本次 tiny 未通过；mini 命令内部也检查 OVERFIT gate，并拒绝启动。
python -m scripts.diagnose_hivt_overfit
```

Tiny 使用16个训练窗口、240 epochs。确定性eval loss从10.8056降至1.5769，但full-horizon minADE_6/minFDE_6仍为5.5717/11.2742m。静止目标拟合良好，移动目标仍严重低估未来位移；未把loss下降当作成功overfit。配置、曲线、初始/最终预测、5个失败案例、诊断报告均保留；checkpoint和processed图数据留在本机。

`minADE_K`按官方HiVT/Argoverse规则使用最低FDE的mode计算ADE，另外单独记录独立minimum-ADE。MR沿用上游末端误差>2m；full-horizon（12个未来关键帧均有效，约6s）与partial-future分开报告。本阶段不是nuScenes官方leaderboard实验。

## 移动车辆 overfit 诊断

原 tiny 的失败结果和 checkpoint 已冻结，原文件 SHA256 保持相同。本轮只做 train split 的单目标/规模/平衡诊断，全部保留场景 vehicle 与 lane context，不修改原 HiVT 架构、baseline loss 或数据尺度。

单目标 K=6 官方 loss（两档固定 LR）和 K=1 官方 loss 仍 FAIL；同一 K=1 模型固定 uncertainty 后，在 epoch 610 达到 ADE=0.0728m、FDE=0.3386m。相同初始权重与同一 epoch 的 free-scale 官方 K=1 仍为 ADE=16.3463m、FDE=37.5303m，支持 **Case C：当前设置下 uncertainty/NLL 优化抑制 location 学习**。结论限于该诊断，K=1/fixed-scale 不能替代官方 K=6 baseline。

按用户条件继续测试 4/8/16 moving targets 与 balanced 的结果见 [诊断报告](outputs/reports/moving_overfit_audit_report.md)，量纲、source target、梯度和各目标真实 attribute 都可追溯。原完整 tiny、mini 完整训练和 Stage 3 均未运行，仍需先解决官方 K=6 moving overfit。

[运行命令](docs/moving_overfit_audit_commands.md) · [Vehicle 运动分布](outputs/reports/vehicle_motion_distribution.md) · [Target audit](outputs/reports/single_moving_target_audit.json) · [梯度 CSV](outputs/reports/moving_gradient_audit.csv)

## Stage 2B：K=6 loss recovery

当前进度：K=6 recovery PASS、Full Tiny PASS、mini baseline 完成。采用 **Protocol 1：fixed-scale warm-up → 原始 learnable-scale Laplace NLL**；HiVT架构、K=6、5/12帧、vehicle定义及地图输入不变，仅适配约6s预测的优化预热。上文保留早期失败实验记录。

单 moving actor：K6 fixed-scale在580 epochs达到ADE/FDE=0.1231/0.9506m；fresh warm-up后恢复原NLL，300 epochs后为0.0140/0.0130m。B PASS，因此未运行bounded-scale C。

原16 tiny windows上的真实 moving 子组（63 actor-windows）ADE/FDE从19.1113/38.5269m下降至2.3165/5.8465m，优于同窗口CV 4.5293/10.3200m。全部236个full-horizon与146个partial targets、stopped/parked均保留。

Mini fresh初始化，64 epochs warm-up + 64 epochs 原NLL；仅按validation选择最终NLL阶段epoch 1。沿用既有6/2/2 scene split，不使用test调参。完整未来validation共583 actor-windows：

| Method | Overall ADE/FDE | Moving ADE/FDE | Stopped ADE/FDE | Parked ADE/FDE |
|---|---|---|---|---|
| CV | 1.2461/2.9148 | 3.0771/7.2688 | 0.2260/0.4329 | 0.1297/0.2540 |
| Recovered HiVT | 1.8617/3.3863 | 4.7853/8.7412 | 0.0711/0.0728 | 0.0679/0.0674 |

Mini validation的moving与overall仍弱于CV；本轮完成baseline工程验收，尚未证明泛化优势。14张validation图包含5个成功moving、5个失败moving、2个stopped及2个parked目标。8个单测通过；175个原实验文件和19个官方源码文件哈希不变。Checkpoint及大型actor日志留本机，小型结果与代码上传本分支。

[完整报告](outputs/reports/k6_loss_recovery_report.md) · [Mini分组CSV](outputs/reports/k6_mini_metrics.csv) · [运行命令](docs/stage2b_execution_commands.md) · [可视化审计](outputs/stage2/k6_loss_recovery/mini_visualization_audit.json)

Stage2 baseline=PASS（本轮约定的recovery/tiny/mini完成验收）；允许讨论Stage3，本轮未执行Stage3，不merge main。
