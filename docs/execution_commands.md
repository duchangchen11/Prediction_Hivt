# 第一阶段执行记录

日期：2026-10-04（用户时区 Asia/Shanghai）。工作目录 `/home/lrj/Prediction_Hivt`。

本次修改以 Git 提交记录为准；配置、原始数据和 `.pt` 留在本机。检查脚本、测试、stdout/stderr 和结构化报告均已保留。此前初始化使用 `git init -b main` 时发现本机 Git 不支持该选项，改用 `git init` 和 `git symbolic-ref HEAD refs/heads/main`。最初未配置身份导致提交失败；本次通过已连接 GitHub 账户确认身份后，仅配置本仓库 user.name / user.email。

## 资产检查

```bash
bash scripts/asset_audit.sh > outputs/reports/asset_audit.txt
conda env list > outputs/reports/conda_envs.txt
```

脚本包含系统、GPU、Python、conda、深度受限的数据与 repo 搜索。额外检查了 `/media/lrj` 深度 6，找到同一 trainval 根目录。对 `/home/lrj/ped_intent_project` 和 `/home/lrj/nuscenes_mini_stage1` 仅执行 git status / branch / remote / log，结果在 `related_repositories.txt`。后者的 `.git` 目录未被 Git 识别为有效仓库，未修复或修改。

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python scripts/probe_environment.py > outputs/reports/environment_ped_intent.json
/home/lrj/anaconda3/envs/navsim/bin/python scripts/probe_environment.py > outputs/reports/environment_navsim.json
timeout 45 /home/lrj/anaconda3/envs/stp3/bin/python scripts/probe_environment.py > outputs/reports/environment_stp3_compatible_probe.json 2> outputs/reports/environment_stp3_compatible_probe_stderr.txt
timeout 45 /home/lrj/anaconda3/envs/e2e/bin/python scripts/probe_environment.py > outputs/reports/environment_e2e_compatible_probe.json 2> outputs/reports/environment_e2e_compatible_probe_stderr.txt
```

stp3/e2e 的第一次 probe 因 Python<3.8 不支持 importlib.metadata 失败，原始 stderr 保留；probe 加入旧版 Python fallback 后已完成检查。

对 trainval 的 `du -sh` 加了 55 秒上限，未在上限内完成，未得到有效磁盘总量；未扩大扫描。目录检查发现 trainval 的 samples 中仅有六个相机目录，未见 LIDAR_TOP。mini 的磁盘占用为 5.6 GB，31,206 个 metadata 引用的传感器文件及 4 个 map 文件存在。

## 仅补装缺失依赖

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip install --dry-run nuscenes-devkit > outputs/reports/nuscenes_install_plan.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip freeze > outputs/reports/environment_before.txt
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip install -c outputs/reports/environment_before.txt nuscenes-devkit==1.2.0 > outputs/reports/nuscenes_install.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python scripts/probe_environment.py > outputs/reports/environment_selected.json
/home/lrj/anaconda3/envs/ped_intent/bin/python -m pip freeze > outputs/reports/environment_after.txt
```

前后包清单已逐项比较：existing_packages_changed={}，详见 `dependency_changes.json`。未升级现有包、未创建新环境、未下载任何数据。

## 数据窗口、坐标、可视化与 CV

以下命令均已执行，其对应输出日志在 reports 中：

```bash
/home/lrj/anaconda3/envs/ped_intent/bin/python -m preprocessing.inspect_nuscenes > outputs/reports/inspect_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m preprocessing.build_one_window > outputs/reports/window_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m preprocessing.visualize_window > outputs/reports/visualization_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m unittest discover -s tests -v > outputs/reports/unit_tests.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.evaluate_cv > outputs/reports/cv_run.txt 2>&1
/home/lrj/anaconda3/envs/ped_intent/bin/python -m scripts.validate_window > outputs/reports/serialized_validation_run.txt 2>&1
```

图像已查看；绘图修正了方向箭头，使其严格沿 +x，再次生成并查看。CV 指标只使用有效点。坐标、heading、identity、mask 通过原始 metadata 回查。

一键复现使用 `bash scripts/run_stage_one.sh`。脚本在生成图后暂停后续步骤，便于查看图像；查看通过后运行 `python -m scripts.evaluate_cv`。本阶段停止在 CV，不执行 HiVT。

## Git

Git remote 为用户指定仓库。通过已连接 GitHub 账户读取到的用户名与邮箱仅用于本仓库 Git 配置。提交前执行 `git diff --check`；上传方式和最终提交见 `outputs/reports/git_upload.txt`。本机最终 Git 状态保留在忽略目录 `outputs/debug/git_final_status.txt`，避免状态文件自身造成工作区改动。
