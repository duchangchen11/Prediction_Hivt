# 第一阶段验收报告

日期：2026-10-04（Asia/Shanghai）。工程：/home/lrj/Prediction_Hivt。

## 1. 环境

SELECTED_ENV=ped_intent

Python=3.10.21；PyTorch=2.5.1+cu124；CUDA available=True；PyTorch CUDA runtime=12.4。

GPU=NVIDIA GeForce RTX 3080（10 GB）；Driver=570.133.07；CUDA Driver=12.8。nuscenes-devkit=1.2.0。

选用原因：已有环境通过 GPU 运算且核心科学计算依赖齐备；只补装 devkit 及缺失依赖，现有包版本未改变。

其他实测环境：navsim（Python3.9/PyTorch2.0.1+cu117，无devkit）；stp3（Python3.7/PyTorch1.10.2/CUDA11.3/devkit1.1.0）；e2e（Python3.6，无torch）。详细版本见环境 JSON。

## 2. nuScenes

ROOT=/home/lrj/datasets/nuscenes-mini

VERSION=v1.0-mini；mini 存在；trainval 存在于 /media/lrj/54926A1D926A0438/nuscenes-trainval。

mini：10 scenes / 404 samples / 18,538 annotations。31,206 个 metadata 引用传感器文件和 4 个 map 文件均存在。trainval 只检查目录及 metadata 文件存在，未做全量完整性验收。

## 3. taxonomy

vehicle=vehicle.*，排除 vehicle.bicycle，包含 motorcycle、bus、truck、trailer 等。

pedestrian=human.pedestrian.*。bicycle=vehicle.bicycle。实际 loaded taxonomy 生成 mapping，静止目标保留。

类别统计：outputs/reports/category_statistics.csv。

## 4. 时间

首场景平均 sample dt=0.503935947 s，范围0.399810–0.599937 s；整个mini平均0.498851322 s。

单窗口平均dt=0.500051500 s；Th=5覆盖2.001162 s；Tf=12覆盖5.999662 s。

速度与CV用实际 timestamp，不固定0.5 s。

## 5. 单样本

scene=scene-0757；t0_index=17；N=13；types={'vehicle': 11, 'pedestrian': 1, 'bicycle': 1}。

agent_pos=[13,5,2]；agent_velocity=[13,5,2]；agent_heading=[13,5]；agent_type=[13]；history_mask=[13,5]；future_pos=[13,12,2]；future_mask=[13,12]；ego_history=[5,2]；ego_future=[12,2]。

所有tensor NaN=0，Inf=0。无效点零填充+mask；velocity_mask区分无法估速的历史点。

样本：outputs/debug/one_window.pt（本机）；完整shape/dtype/min/max/NaN/Inf见tensor_statistics.json。

## 6. 坐标

ego +x：PASS；+y左侧：PASS；轨迹连续：PASS；instance_token：PASS。

回查原始 metadata 的有效 annotation 数=194；存储位置最大误差=3.11535683e-06 m；heading最大误差=1.16969658e-07 rad。

独立90°旋转和左右手性测试通过；原始 ego history/future 和 heading 回查通过。30个CV窗口也通过时间链、annotation链与坐标断言。

## 7. 可视化

outputs/figures/one_window.png。已查看，两幅图为完整场景与近ego区域；ego +x箭头沿水平方向；历史/未来分段连续；未见身份跳变、镜像或明显旋转错误。

## 8. CV baseline

30个诊断窗口，1,147个当前actor-window，1,087个满足历史/未来条件进入指标；45个历史不足，15个无有效未来。按scene先划分6/2/2，断言无交集。

此处汇总包含train/val/test的诊断窗口，不能作为正式泛化性能；逐split结果已保存。ADE为每个actor-window有效点平均后等权汇总；FDE为最后有效未来点，另列最终帧FDE。

| Type | ADE (m) | FDE last-valid (m) | FDE fixed-horizon (m) |
|---|---:|---:|---:|
| vehicle | 1.156408 | 2.570848 | 2.793824 |
| pedestrian | 0.511357 | 1.053188 | 1.304765 |
| bicycle | 0.424552 | 0.852467 | 0.978731 |
| overall | 0.909887 | 1.990897 | 2.217906 |

## 9. Git

工程路径=/home/lrj/Prediction_Hivt；branch=main；remote=https://github.com/duchangchen11/Prediction_Hivt.git。最终commit和工作区状态见聊天最终汇报及outputs/debug/git_final_status.txt。

## 10. 下一阶段条件

vehicle-only HiVT：YES（第一阶段数据基础及CV验证已满足，可以开始下一阶段的baseline适配工作）。当前完整HiVT尚不能直接运行：PyG/Lightning未安装，下一阶段需要核对并补齐兼容依赖；本阶段未安装它们。

真实问题/限制：

- 当前工程验收只使用mini；自定义6/2/2划分仅供流程验证，正式评估应另用官方train/val/test规则。
- trainval samples目录目前只见六个camera目录，未见LIDAR_TOP；原始传感器文件未验收，完整目录du在55秒内未完成。基于metadata的轨迹处理与需要原始LiDAR的任务应分别核对数据要求。
- 部分actor历史或未来缺失，按mask排除；不能将最后有效点FDE称为固定6秒FDE。
- 自行车只有18个有效诊断actor-window，且窗口可能重叠，不能据此推断泛化性能。

已停止在CV baseline，等待下一条指令。
