# Stage 2B commands and gates

本轮要求：`docs/stage_2b_requirements.txt`。工作目录固定为 `/home/lrj/Prediction_Hivt`，分支 `stage2/vehicle-hivt-baseline`，环境复用 `/home/lrj/anaconda3/envs/ped_intent/bin/python`。没有安装/升级依赖。

首先复制原 tiny、moving audit、原 failure figures 到本机 `outputs/stage2/k6_loss_recovery/frozen_previous/`，并将原目录及 `outputs/reports/moving_*` 共175个文件的 SHA256、大小与原提交 d8b93ae 写入 `frozen_previous_manifest.json`。各阶段结束都核对原文件哈希；完整副本与所有 checkpoint 不上传 Git。

```bash
OMP_NUM_THREADS=4 /home/lrj/anaconda3/envs/ped_intent/bin/python -m unittest tests.test_k6_loss_recovery tests.test_hivt_stage_two > outputs/stage2/k6_loss_recovery/unit_tests.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.k6_loss_recovery --experiment A > outputs/stage2/k6_loss_recovery/A_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.k6_loss_recovery --experiment B > outputs/stage2/k6_loss_recovery/B_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.k6_baseline --stage reference > outputs/stage2/k6_loss_recovery/tiny_reference_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.k6_baseline --stage tiny > outputs/stage2/k6_loss_recovery/full_tiny_run.txt 2>&1
# 以下仅在 full_tiny_result.json 的 status=PASS 后运行。
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.k6_baseline --stage mini > outputs/stage2/k6_loss_recovery/mini_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.report_k6_loss_recovery > outputs/stage2/k6_loss_recovery/report_run.txt 2>&1
```

A 保留 K=6 best-mode selection 与原 classification，只使用 mean(abs(y−μ)) regression；b=1，不加入 log(2) 常数。lr=.001 固定，最多1000，每10 epoch评估，严格 ADE<.5/FDE<1 后停止。只有 A PASS 才允许运行 B 并做 mode audit。

所有 hyperparameters/gates 都在 `configs/stage2b_recovery.yaml` 中预先记录。B 从 fresh seed2022 开始，warm-up 上限700、gate ADE<1/FDE<2；随后保存并读取 warmup checkpoint，恢复 AdamW state 和 Torch RNG，LR降至1e-4，原 NLL 运行300。将‘无明显退化’量化为最终 ADE 增量≤.5m/FDE增量≤1m，同时仍须 ADE<1/FDE<2。

仅 B FAIL 时才可测 C 的 b_max=2/4/8。其余配置相同，每个最多700，单移动 gate 沿用 A。优先 warm-up→原 NLL，再选择能通过的最大 b_max；最后才允许明确命名的 permanent fixed regression fallback。

Full Tiny gate 在任何恢复训练之前保存于 configs/stage2b_baseline.yaml：overall ADE/FDE 至少比原失败模型好30%；真实 moving 子组至少好50%，且 ADE<5m/FDE<10m。接近 CV 是优先目标，不作为额外强制 gate。固定 scale warm-up 最多700，若 overall<1m/2m 且 moving≤1.5×CV 可提前进入原 NLL；达到700也进入原 NLL（未达到这个优先目标不等于最终 Full Tiny FAIL）。原 NLL 最多300，最终按新 gate 验收。原16 tiny windows 不重新抽样；所有原 eligible vehicle 都参与，mask不改。

原失败模型只在 CPU 上重新推理并按真实 attributes 分组；与旧报告 pooled metrics 的差异检查<1e-3m。CV 和旧 HiVT 使用相同236个 full-horizon actor-window（moving63/stopped73/parked90/unknown10），另有146个 partial targets 单独报告。

mini 只在 Full Tiny PASS 后允许运行，沿用 train146/val48 个原窗口；fresh seed2022，warm-up64 epochs + 原 NLL64 epochs，固定LR .001/1e-4。只在最后 probabilistic phase 内按 validation overall full-horizon minFDE 选择 checkpoint。test 不用于调参、checkpoint selection 或可视化选择。数据、K、宽度、层数和地图变量不变。

Mini 成功/失败可视化按真实 t0 vehicle.moving、完整未来、GT位移≥5m 选择，成功 minFDE≤2m、失败>2m，优先不同 instance。另取2个 stopped 和2个 parked；14张图的窗口、误差、全部6条轨迹与概率记录在 mini_visualization_audit.json。图轴覆盖所有 mode，避免截掉非最佳轨迹。Mini 完成状态表示预定训练/评估/可视化完成，不预设其泛化优于 CV。

实验目录拒绝覆盖。不要直接重复带重定向的命令覆盖已有日志；新实验需使用独立名称。首次新 loss 单测捕获了二维 boolean mask 与坐标切片的索引顺序错误，训练前已修正；失败日志保存为 unit_tests_first_attempt.txt，随后7个测试全部通过。

所有原 HiVT encoder/global/decoder 参数、forward、K=6、5/12、地图半径/采样和 vehicle 定义均冻结；新模块只显式管理已授权的 regression phase。没有添加论文创新，没有进入 Stage 3，不 merge main。
