# Moving overfit diagnostic commands

本轮要求保存在 `docs/stage_2_moving_audit_requirements.txt`。工作目录固定为 `/home/lrj/Prediction_Hivt`，分支 `stage2/vehicle-hivt-baseline`。环境复用 `/home/lrj/anaconda3/envs/ped_intent/bin/python`，没有创建环境或安装/升级依赖。

先把原 `outputs/stage2/tiny_overfit`、`outputs/figures/stage2/tiny_failure` 完整复制到本机诊断目录下的 `frozen_*`。24 个原文件（含 checkpoint）的大小与 SHA256、原 commit `2bddcd8d7b166d60120e4fb3b347d196f5214a58` 写入 `frozen_failure_manifest.json`。所有诊断都写入独立目录，原 tiny 文件没有覆盖。

以下命令均在项目根目录执行，完整标准输出/错误保存在同名 run.txt 中：

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.audit_vehicle_motion > outputs/stage2/moving_overfit_audit/motion_audit_run.txt 2>&1
OMP_NUM_THREADS=4 /home/lrj/anaconda3/envs/ped_intent/bin/python -m unittest tests.test_moving_overfit_audit tests.test_hivt_stage_two > outputs/stage2/moving_overfit_audit/unit_tests.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.moving_overfit_audit --phase single > outputs/stage2/moving_overfit_audit/single_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.moving_overfit_audit --phase followups > outputs/stage2/moving_overfit_audit/followups_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.moving_overfit_audit --phase budget-check > outputs/stage2/moving_overfit_audit/budget_check_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.moving_overfit_audit --phase scaleups > outputs/stage2/moving_overfit_audit/scaleups_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.inspect_moving_decoder > outputs/stage2/moving_overfit_audit/decoder_inspection_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.report_moving_overfit_audit > outputs/stage2/moving_overfit_audit/report_run.txt 2>&1
```

执行顺序与门槛：

1. `single`：官方 K=6 + 官方 loss，固定 LR=5e-4，最多 500 epoch。
2. `followups`：只有上一步失败才运行 fresh seed、固定 LR=1e-3 的 K=6（其余相同、最多 500 epoch）。仍失败才运行 K=1 官方 loss；K=1 再失败才运行 K=1、单位 scale Laplace。后两者初次上限均 500。
3. `budget-check`：K=1 两个 500-epoch 诊断都失败、fixed scale 的轨迹误差却显著改善后，预注册相同最大 1000 epoch 的 K=1 对照。从各自 seed2022 相同初始权重重训，只改变预算。官方 K=6 没有延长。比较 fixed scale 首次通过时的同一 epoch，避免比较不等训练步数。
4. `scaleups`：单 moving 诊断通过后，沿用已通过设置测试 4、8、16 个 distinct moving targets。4/8 有指定门槛，16 只看趋势。4/8 都通过才运行 8 moving + 4 真实 stopped + 4 真实 parked 的 balanced 诊断。这些 K=1/fixed-scale 结果不能替代官方 K=6 baseline。
5. `inspect_moving_decoder`：CPU eval-mode backward，仅检查梯度和参数，无 optimizer.step；代数输出范围 probe 只使用临时模型副本，显式注入 GT，不作为训练 PASS 证据。
6. `report_moving_overfit_audit`：等步数比较、分布/梯度/量纲/后续门槛汇总，最终再次核对原 tiny SHA256。

每个实验保存独立 config、target tokens、initial/final metrics、evaluation curve、每 epoch train-mode 梯度、逐评估 epoch 的 t=1/3/6/12 量纲数据、图及本机 checkpoint。`target_mask` 是唯一改变的 graph 字段；其他原 nodes、edges、lanes、positions/masks/metadata 完全一致，非 target raw output 不参与 loss。所有 samples 均来自已划分的 mini **train** scenes。

首次 single 启动时，严格零误差的 GPU timestep probe 遇到 scatter 浮点归约波动，尚未创建 optimizer/训练；该失败日志和无训练配置保留在 `startup_failure_*`。修正为同权重 CPU probe，之后完整实验独立运行。

这些脚本拒绝覆盖已存在的实验目录。若要另做实验，必须选择新的诊断目录/实验名。不要直接重复上面带输出重定向的命令覆盖已有日志。本轮未运行原完整 tiny 重训、mini 完整训练、trainval 训练或 Stage 3。

适合 Git 的代码、配置、派生 CSV/JSON/Markdown 与小型 PNG 提交到同一分支；checkpoint、processed graphs、原始 nuScenes、local config、完整冻结副本及缓存不上传。
