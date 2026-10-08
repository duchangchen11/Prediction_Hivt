# Stage11A frozen mode-selection audit

本阶段独立目录。只使用 TRAIN700 内既有 HeadTrain630 / HeadDev70，禁止训练、模型/loss/checkpoint变更和 official VAL/test。最终报告位于 [09_reports/stage11a_root_cause_report.md](09_reports/stage11a_root_cause_report.md)。

规则先登记于 [stage11a_protocol.json](01_identity_audit/stage11a_protocol.json)，来源与历史文件哈希见 frozen_history/source_identity。所有脚本仅向本目录写诊断产物；复用历史模型代码和本机 `ped_intent` 环境，未升级依赖。

实际执行顺序（环境变量每次都设置；日志保留本机）：

```bash
export PYTHONDONTWRITEBYTECODE=1
export STAGE11_PY=/home/lrj/anaconda3/envs/ped_intent/bin/python
export STAGE11_ROOT=outputs/stage11a_mode_selection_audit
# prepare/infer 为一次性缓存构造器；已有完成标记时不重复覆盖。
"$STAGE11_PY" -u "$STAGE11_ROOT/01_identity_audit/stage11a_prepare.py" > "$STAGE11_ROOT/01_identity_audit/stage11a_prepare.log" 2>&1
"$STAGE11_PY" -u "$STAGE11_ROOT/01_identity_audit/stage11a_infer.py" > "$STAGE11_ROOT/01_identity_audit/stage11a_infer.log" 2>&1
"$STAGE11_PY" -u "$STAGE11_ROOT/stage11a_analyze.py" > "$STAGE11_ROOT/01_identity_audit/stage11a_analyze.log" 2>&1
"$STAGE11_PY" -u "$STAGE11_ROOT/stage11a_cases.py" > "$STAGE11_ROOT/08_cases/stage11a_cases.log" 2>&1
"$STAGE11_PY" -u "$STAGE11_ROOT/stage11a_verify.py" > "$STAGE11_ROOT/01_identity_audit/stage11a_verify.log" 2>&1
"$STAGE11_PY" -u "$STAGE11_ROOT/stage11a_report.py" > "$STAGE11_ROOT/09_reports/stage11a_report.log" 2>&1
```

统计和作图曾在补充 unknown vehicle state、改善图例布局后重新执行；未重新训练。prepare 后新增的显式 raw SHA 比较通过 verify 对全部 TRAIN window 独立复核。各步骤的完成标记和独立核验构成最终证据。

大缓存与逐actor全量CSV只留本机，不上传Git。`01_identity_audit/cache/`：

- identities 顺序沿用 Stage10 `01_training/cache/identities.csv`，通过 source_index 与 scene/sample/instance 对齐。
- candidates `[N,6,12,2]`、GT `[N,12,2]` 与 observed_motion `[N,5]`。
- model_logits / model_probabilities `[N,9,6]`；模型顺序 R0,R2,G1,G3,T2,T3,VehicleExpert,PedestrianExpert,DualExpert；专家域外为NaN。
- actor_metrics `[N,9,8]`：SoftCE,Top1FDE,Top1ADE,Regret,HitRate,LogitGradL1,LogitGradL2,LogitGradSquaredL2。
- soft_label_actor、candidate_geometry_actor、endpoint_pair_distance 的字段分别见 soft_label_schema/geometry_schema；endpoint为全部15个无序pair。
- `02_switch_regret/stage11a_{G1,G3,DualExpert}_actor_records.csv`：逐actor的三种R2切换比较。

逐actor ADE/FDE 与损失记录在以上数组，较小的分组表、固定30案例分数/终点差异及PNG、报告和文献核验适合版本管理。新checkpoint=0，模型参数backward=0，isolated logits的梯度核验不涉及冻结参数。

完成后 STOP，等待大脑AI审查；建议不等于启动Stage11B授权。
