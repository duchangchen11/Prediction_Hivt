# Stage14B 本地执行命令记录

工作目录固定 `/home/lrj/Prediction_Hivt`，现有环境 `/home/lrj/anaconda3/envs/ped_intent/bin/python`；未创建环境、安装/升级torch/CUDA或下载大型数据。下列是实际工作流记录，已有输出的命令带防覆盖断言，不能把本文件当作覆盖旧结果的一键入口。

1. `git switch -c stage14b/paper-validation-design 4daa4ae82557270e4f43881fa22c21e3ba6c4068`
2. `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage14b_paper_validation/00_protocol/stage14b_register.py`
3. 只读设计与官方审计：`stage14b_scene_design_audit.py`、`stage14b_official_protocol_audit.py`（均在本目录00_protocol）
4. 本目录01_preflight依次执行 `stage14b_model_audit.py`、`stage14b_capacity_preflight.py`、`stage14b_tiny.py`，使用上述python及相同环境变量，tiny添加 `STAGE14B_PHASE=tiny`。
5. 工程/tiny PASS后执行：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 STAGE14B_PHASE=train /home/lrj/anaconda3/envs/ped_intent/bin/python -u outputs/stage14b_paper_validation/03_training/stage14b_train.py > outputs/stage14b_paper_validation/07_logs/stage14b_train.log 2>&1`
6. 全3checkpoint冻结后，用上述python执行 `05_evaluation/stage14b_evaluate.py`，然后 `06_statistics/stage14b_statistics.py` 与 `05_evaluation/stage14b_efficiency.py`；阶段路径均位于本根目录。
7. 执行 `00_protocol/stage14b_finalize.py`，核验报告SHA、历史文件/旧checkpoint/输入不变，新增文件仅commit/push本分支。

三头checkpoint、optimizer/carry/coverage快照和大数组留本机，SHA、训练曲线、配置、小表和报告版本管理。GitHub认证提交使用已连接GitHub工具写blob/tree/commit并建立新分支，远端tree/parent/branch SHA及本地状态一致性单独核验；不merge/force覆盖历史分支。

预登记本地commit `461e446e2db8ff3ff766efc1c535277a552b8213`，对应远端同tree commit `00998ed1fc12a0273f293781e2837bf37aefbef0`；这两者同内容不同提交元数据。最终远端commit以完成后的branch HEAD为准。

HiVT训练命令没有执行，方案A/B仅设计。全部Stage14B工作完成并上传后STOP，不启动Stage15。
