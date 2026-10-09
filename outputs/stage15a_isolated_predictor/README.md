# Stage15A isolated predictor preflight

六份审计报告位于本目录根部。工程检查仅使用InnerTrain/InnerDev；本阶段formal入口关闭，所有checkpoint/norm/candidate标记preflight，不能复用为正式训练产物。

- [数据隔离](stage15a_data_isolation_audit.md)
- [训练入口](stage15a_training_interface_audit.md)
- [模型完整性](stage15a_model_integrity.md)
- [排序兼容](stage15a_ranking_compatibility.md)
- [资源报告](stage15a_resource_report.md)
- [最终报告](stage15a_final_report.md)

使用既有 `/home/lrj/anaconda3/envs/ped_intent/bin/python`，环境 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8`。不要创建新环境、升级依赖或重新预处理数据。

可只读复核并再生成报告：`python3 outputs/stage15a_isolated_predictor/08_reports/stage15a_finalize.py`。`stage15a_prepare.py`首次登记只运行一次，保留登记与审计修正记录。数据引用原Stage3 shard，SQLite仅查询少量明确train/dev键，原数据不复制。

入口示例（fold1）：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8   /home/lrj/anaconda3/envs/ped_intent/bin/python   outputs/stage15a_isolated_predictor/02_training/stage15a_train.py   --fold 1 --seed 2022   --training-scenes outputs/stage15a_isolated_predictor/01_data_isolation/stage15a_fold1_InnerTrain.json   --development-scenes outputs/stage15a_isolated_predictor/01_data_isolation/stage15a_fold1_InnerDev.json   --output-dir outputs/stage15a_isolated_predictor/02_training/new_description/fold1   --mode describe
```

全部small检查执行`03_checks/stage15a_checks.py`并将mode改为preflight；fold2/3的seed为2122/2222。为复现使用新的输出目录，不能覆盖已有04_checkpoints/foldN；候选缓存/最终汇总也应在独立副本中重放以保留当前冻结证据。正式fold训练仍被gate拒绝，不能因有训练代码就擅自启动。

诊断来源与命令见`07_logs/stage15a_corrected_process_receipts.json`、`stage15a_trained_interface_receipts.json`及`02_training/stage15a_cli_smoke.json`。Fold1首次执行通过，在shell工具运行，无subprocess计时receipt；其训练/forward/恢复时序与来源完整保存在fold1 JSON。Fold2初次timestamp审计失败及修正登记均保留。large checkpoint/array/log/cache不上传，small机器JSON和源码可复核。
