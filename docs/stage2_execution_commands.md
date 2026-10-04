# Stage 2 执行记录

工作目录 `/home/lrj/Prediction_Hivt`。没有重新执行 Stage 1 验收，没有修改其他论文项目或原始数据。

## 分支与官方来源

```bash
git status
git log -5 --oneline
git remote -v
git checkout -b stage2/vehicle-hivt-baseline
find /home/lrj -maxdepth 4 -type d -iname '*hivt*'
git clone --filter=blob:none --no-checkout https://github.com/ZikangZhou/HiVT.git outputs/debug/hivt_upstream_checkout
git -C outputs/debug/hivt_upstream_checkout sparse-checkout init --no-cone
git -C outputs/debug/hivt_upstream_checkout sparse-checkout set '/models/' '/losses/' '/metrics/' '/datasets/' '/utils.py' '/train.py' '/README.md' '/LICENSE'
git -C outputs/debug/hivt_upstream_checkout checkout
git -C outputs/debug/hivt_upstream_checkout rev-parse HEAD
git -C outputs/debug/hivt_upstream_checkout archive 6876656ce7671982ebdc29113aaaa028c2931518 models losses metrics datasets utils.py train.py README.md LICENSE | tar -x -C baselines/hivt_official
```

取得20个源文件，没有下载模型权重。SOURCE.json的SHA256及runtime差异记录已经核对；loss文件逐字节一致。

## 环境安装

Python路径固定为 `/home/lrj/anaconda3/envs/ped_intent/bin/python`。每次安装前冻结环境，各次记录都已保留。

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip freeze > outputs/reports/stage2/environment_before.txt
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip install --dry-run -c outputs/reports/stage2/environment_before.txt torch-geometric==2.6.1 ijson > outputs/reports/stage2/dependency_plan.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip install -c outputs/reports/stage2/environment_before.txt torch-geometric==2.6.1 ijson > outputs/reports/stage2/dependency_install.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip freeze > outputs/reports/stage2/environment_before_extensions.txt
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip install --no-index -c outputs/reports/stage2/environment_before_extensions.txt torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html > outputs/reports/stage2/extension_install.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip freeze > outputs/reports/stage2/environment_after.txt
```

前后现有包版本比较通过。PyTorch CUDA、PyG、scatter CUDA kernel及sparse import通过。原生PyTorch训练，不安装Lightning/TorchMetrics/Argoverse数据API。

## 只读数据审计与运行

以下命令均已执行；原始stdout/stderr保留在对应文件中。

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.audit_trainval > outputs/reports/stage2/trainval_audit_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.prepare_hivt_mini > outputs/reports/stage2/prepare_mini_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m unittest discover -s tests -p test_hivt_stage_two.py -v > outputs/reports/stage2/model_unit_tests.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.hivt_smoke_test > outputs/reports/stage2/trainval_smoke_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.train_hivt_stage2 --mode tiny > outputs/stage2/tiny_overfit_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.diagnose_hivt_overfit > outputs/reports/stage2/tiny_diagnosis_run.txt 2>&1
```

trainval首次map读取因为expansion在根目录而devkit要求maps/expansion失败；初始错误日志保留。通过工程缓存中的软链接视图修复布局，不改动原始目录。第二次审计通过。

附加执行：生成并查看hivt_map_sample.png；首个真实mini双窗口forward/loss/backward；核对frozen源码哈希和白名单runtime diff；查看tiny失败预测与训练曲线。具体修改由Git提交记录保留。

## 停止与产物

首轮tiny overfit明确失败后停止训练。之后的diagnose_hivt_overfit只加载checkpoint并推理，没有optimizer或backward，也没有运行mini train/val实验。

本机产物：`outputs/stage2/tiny_overfit/checkpoint.pt`、mini processed graphs、trainval metadata cache。Git仅保留configs/logs/CSV/JSON/报告/小型图像和源码，忽略所有checkpoint、processed dataset、cache与raw data。未开始完整trainval训练或Stage3。
