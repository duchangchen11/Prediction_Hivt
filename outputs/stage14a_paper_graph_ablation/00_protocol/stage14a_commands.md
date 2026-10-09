# Stage14A 执行命令与停止条件

固定工作目录 `/home/lrj/Prediction_Hivt`，复用 `/home/lrj/anaconda3/envs/ped_intent/bin/python`。没有安装或升级环境，没有下载数据。

运行脚本时设置 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1`。正式训练前必须依次通过登记、模型核验、输入/损失核验及 tiny 检查；任何关键科学完整性检查失败即停止，不调参。

```bash
cd /home/lrj/Prediction_Hivt
git ls-remote origin refs/heads/stage13a/prediction-to-ego-planning-audit
git switch -c stage14a/paper-graph-ablation 88b83b7114e72b91544d4566cab26700c1b452dc

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/00_protocol/stage14a_register.py

PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/01_preflight/stage14a_model_audit.py

STAGE14A_PHASE=preflight PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/01_preflight/stage14a_preflight.py

STAGE14A_PHASE=tiny PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/01_preflight/stage14a_tiny.py

STAGE14A_PHASE=train PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/03_training/stage14a_train.py
```

`stage14a_register.py` 是本次登记脚本，要求仍在历史基础 commit 且不存在冻结协议。已登记的本次目录不能直接覆盖重跑。tiny 脚本禁止覆盖已有审计，训练脚本禁止覆盖已冻结模型或隐式重启带 `last.pt` 的未完成训练。

模型审计只执行合成 CPU forward/gradient 检查；没有 optimizer 更新或 checkpoint。输入审计最初的标签梯度检查错误地处于 `no_grad` 中，已将这一检查局部置于 `enable_grad`，随后科学完整性检查全部通过。第一次训练启动的 source SHA 扫描被预冻结读取保护阻断，发生在 optimizer 创建前，零更新、零 checkpoint；保留完整 trace：`03_training/stage14a_pre_optimizer_guard_block.txt`。随后只锁训练有关 source，读取保护未放宽。

所有六个 checkpoint 冻结后才允许执行：

```bash
STAGE14A_PHASE=evaluate PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/05_evaluation/stage14a_evaluate.py

STAGE14A_PHASE=analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/06_bootstrap/stage14a_bootstrap.py

STAGE14A_PHASE=analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/07_diagnostics/stage14a_mode_switch.py

STAGE14A_PHASE=benchmark PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/07_diagnostics/stage14a_efficiency.py

STAGE14A_PHASE=figures PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/08_figures/stage14a_figures.py
```

来源审计可以独立执行，不进行模型训练或推理：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/09_reports/stage14a_provenance_audit.py
```

六张图经过 PNG 视觉检查、SVG/PDF 格式核验并保存审计后，执行最终完整性复核：

```bash
STAGE14A_PHASE=finalize PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /home/lrj/anaconda3/envs/ped_intent/bin/python \
  outputs/stage14a_paper_graph_ablation/09_reports/stage14a_finalize.py
```

本地日志：`00_protocol/stage14a_registration.log`、`01_preflight/stage14a_preflight.log`、`01_preflight/stage14a_tiny.log`、`03_training/stage14a_training.log`，以及各后续阶段同名执行日志。CSV/JSON 结果和 SHA256 清单参与版本管理，`.log/.pt/.npy/cache/` 留在本地。发布时核对每个 Git blob/tree 与本地提交一致，发布到本阶段分支，不 merge main，随后 STOP 等待大脑 AI。
