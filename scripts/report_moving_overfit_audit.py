"""Evidence-based report, including matched-epoch uncertainty comparison."""
import csv
import json

from preprocessing.common import PROJECT_ROOT, write_json
from scripts.audit_vehicle_motion import AUDIT
from scripts.moving_overfit_audit import verify_freeze


def main():
    results=json.loads((AUDIT/"experiment_summary.json").read_text())
    experiments={r["experiment"]:r for r in results}
    distribution=json.loads((PROJECT_ROOT/"outputs/reports/vehicle_motion_distribution.json").read_text())
    sanity=json.loads((PROJECT_ROOT/"outputs/reports/single_moving_target_audit.json").read_text())
    decoder=json.loads((PROJECT_ROOT/"outputs/reports/moving_decoder_audit.json").read_text())
    selection=json.loads((AUDIT/"target_selection.json").read_text())
    k6=experiments["single_moving_actor_overfit_lr1e3"]
    k1=experiments["single_moving_actor_k1_budget1000"]
    fixed=experiments["single_moving_actor_fixed_scale_budget1000"]
    moving=[experiments.get(f"moving_only_{n}") for n in (4,8,16)]
    balanced=experiments.get("balanced_8_moving_8_stationary")
    per_actor=json.loads((PROJECT_ROOT/"outputs/reports/moving_final_actor_metrics.json").read_text())
    assert k1["initial_state_sha256"]==fixed["initial_state_sha256"]
    assert k1["configuration"]["learning_rate"]==fixed["configuration"]["learning_rate"]==.001
    # Same target, context, architecture, seed, optimizer and learning rate.
    # Read the official comparison at the EXACT fixed-scale passing epoch.
    curves=list(csv.DictReader((AUDIT/k1["experiment"]/"training_curve.csv").open()))
    matched=next(row for row in curves if int(row["epoch"])==fixed["epochs"])
    matched={k:float(v) for k,v in matched.items()}
    if fixed["status"]=="PASS" and k1["status"]=="FAIL" and (matched["minADE"]>=.5 or matched["minFDE"]>=1):
        case="Case C"
        finding="同一 moving actor、同一 K=1 模型及相同训练步数下，free-scale 官方 Laplace 失败，固定 b=1 通过；uncertainty/NLL 优化抑制 location 学习是本例中得到实验支持的原因。"
    elif fixed["status"]=="FAIL":
        case="Case A"
        finding="单 moving target 的官方 loss 与 fixed-scale 都未通过；当前模型/target/loss/decoder adaptation 的实现或优化问题尚未隔离，继续 Stage 2。"
    else:
        raise RuntimeError("Observed results do not satisfy the preregistered case comparison")
    excluded=["单目标 loss 已消除 parked/static 占比因素，不能将失败仅归因于静止车分布。",
              "K=1 去除了多模态竞争和 classification 的影响后仍失败，Case D 不成立。",
              "target 原始标注、旋转往返、输出米制和 decoder range 均检查通过；本次未发现坐标或统一 timestep multiplier 错误。",
              "规模/平衡实验若使用 fixed-scale 诊断 loss，即使成功，也不能形成 Case B 的官方 loss/采样对照；不能据此决定删除 parked 或 stopped 车辆。"]
    final={"root_cause_case":case,"finding":finding,"scope":"one selected previously failing moving car; does not prove all original tiny failures share a sole cause",
           "matched_epoch_comparison":{"epoch":fixed["epochs"],"same_initial_weights_sha256":k1["initial_state_sha256"],
               "same_target_count":1,"same_learning_rate":.001,"same_context_agents":sanity["context_agents"],
               "same_context_lane_segments":sanity["context_lane_segments"],"official_K1_at_same_epoch":matched,
               "fixed_scale_K1_at_same_epoch":{k:v for k,v in fixed["final"].items() if k!="actors"}},
           "official_K6_single_moving_overfit":"FAIL","diagnostic_location_memorization":fixed["status"],
           "excluded_or_unproved":excluded,
           "moving_only_4_8_16":{str(n):{"status":r["status"],"epochs":r["epochs"],"minADE":r["final"]["minADE"],"minFDE":r["final"]["minFDE"],"loss":r["final"]["loss"]} if r else {"status":"NOT_RUN"} for n,r in zip((4,8,16),moving)},
           "balanced":{"status":balanced["status"],"moving":8,"stopped":4,"parked":4,
                        "minADE":balanced["final"]["minADE"],"minFDE":balanced["final"]["minFDE"],"loss":balanced["final"]["loss"],
                        "by_real_attribute":per_actor[balanced["experiment"]]["by_real_attribute"],
                        "gate_scope":"pooled actor mean only; moving subgroup not automatically PASS"} if balanced else {"status":"NOT_RUN","reason":"4/8 diagnostic scaleup gates not both passed"},
           "next_stage":{"complete_original_tiny_rerun":"NO","mini_training":"NO","Stage3":"NO"},
           "next_work":"保持 K=6、原 horizon 和 context，继续诊断官方 loss 的 scale/loc 优化；先使官方 single moving PASS，再在官方 loss 下验收 4/8/16 和 balanced。本次 fixed-scale/K=1 诊断不能替代最终 baseline 验收。"}
    write_json(PROJECT_ROOT/"outputs/reports/moving_overfit_audit.json",final)
    lines=["# Moving vehicle overfit audit", "", f"根因判定：**{case}**。{finding}", "",
           "结论限于本次原失败 tiny 中选出的 moving car 和这些优化设置。官方 K=6 baseline 仍 FAIL；完整 tiny、mini、Stage 3 均未运行。", "",
           "## 冻结与范围", "", "原 commit：`2bddcd8d7b166d60120e4fb3b347d196f5214a58`。原 tiny checkpoint/metrics/curve/figures 及 failure figures 共 24 个文件 SHA256 前后相同。",
           "完整快照只保存在本机 moving_overfit_audit/frozen_*，checkpoint 和缓存不上传；可追溯 manifest 与报告提交 Git。复用 ped_intent 环境，没有安装/升级包。", "",
           "## 【Vehicle distribution】", "", f"总 vehicle actor-window={distribution['total_vehicle_actor_windows']}；loss eligible={distribution['eligible_prediction_actor_windows']}；full horizon={distribution['full_horizon_actor_windows']}。",
           "真实 t0 attribute："+"，".join(f"{k}={v}" for k,v in sorted(distribution["attribute_state_counts"].items()))+"。", "",
           "先读取实际 attribute_name："+"、".join(distribution["actual_attribute_names"])+"。没有把低速车辆直接标为 parked。", "",
           "| Endpoint bin | 全部有 future 的 actor-window | 占 3529 的比例 | full 12-step actor-window | 占 2158 的比例 |", "|---|---:|---:|---:|---:|"]
    all_bins=distribution["endpoint_bins_all_actor_windows"]
    full_bins=distribution["endpoint_bins_full_horizon"]
    for a,b in zip(all_bins["bins"],full_bins["bins"]):
        lines.append(f"| {a['bin']} | {a['count']} | {a['fraction_of_valid_future']:.2%} | {b['count']} | {b['fraction_of_valid_future']:.2%} |")
    lines += ["", "另外 94 个 actor-window 没有 future，未计入 endpoint bins。partial endpoint 是最后可用位置，不能都称为 6s 位移。仅 full horizon 一栏可用于约 6s 比较。",
              "历史速度、当前 past-only 速度、GT 平均位移/path length 和 future valid length 的分布及 attribute 交叉表见 vehicle_motion_distribution.json；每个 actor-window 见 vehicle_actor_windows.csv。", "",
              "## 单目标与 target sanity", "", f"scene={selection['single']['scene_name']}；scene_token=`{selection['single']['scene_token']}`；sample_token=`{selection['single']['sample_token']}`；instance_token=`{selection['single']['instance_token']}`。",
              f"GT endpoint displacement={selection['single']['gt_endpoint_displacement_m']:.6f}m，future duration={selection['single']['last_valid_future_s']:.6f}s；真实 `vehicle.moving`；5/12 全有效；17 个 annotation 的 prev/next 连续。",
              f"当前 past-only speed={selection['single']['current_past_only_speed_mps']:.3f}m/s，未来最大区间速度={selection['single']['future_max_interval_speed_mps']:.3f}m/s，最大区间加速度={selection['single']['future_max_interval_acceleration_mps2']:.3f}m/s²，无明显跳变。",
              f"保留 {sanity['context_agents']} 个 vehicle nodes、{sanity['context_lane_segments']} 个 lane segments，以及所有原 actor/lane edges。仅该车 target_mask=True；每步 backward 验证其他车辆 raw prediction gradient 恰为零。context 的 encoder 仍可参与该车计算。",
              f"source annotation → cached ego 最大误差={sanity['source_to_cached_ego_max_abs_error_m']:.9g}m；rotated_y → inverse rotation → +current 最大误差={sanity['roundtrip_max_abs_error_m']:.9g}m，均 <1e-4m。",
              "history/current/future/y/rotation/rotated_y/source annotation tokens 均保存在 single_moving_target_audit.json。只改变 timestamps 的 CPU forward 输出差为零，未发现隐式时间倍率。", "",
              "## 【Single moving K=6】 / 【Single moving K=1】 / 【Fixed scale diagnostic】", "",
              "所有模型均为原 HiVT-64 encoder+MLP decoder，原 dropout=0.1、AdamW 和 weight decay，fresh seed2022，固定 LR，不使用 scheduler。K=6 保留原 loss 和 K=6；K=1 与 fixed scale 仅为诊断。",
              "| Experiment | LR | Epochs | Initial ADE | Final ADE | Initial FDE | Final FDE | Status |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in results:
        a,b=r["initial"],r["final"]
        lines.append(f"| {r['experiment']} | {r['configuration']['learning_rate']} | {r['epochs']} | {a['minADE']:.6f} | {b['minADE']:.6f} | {a['minFDE']:.6f} | {b['minFDE']:.6f} | {r['status']} |")
    lines += ["", "ADE 使用原 HiVT 的 best-FDE mode，另记录 independent minADE；所有诊断 target 均 full horizon。严格单目标阈值：ADE<0.5m 且 FDE<1.0m。", "",
              "K=1 classification 对单一 mode 恒为 0。fixed-scale 使用相同逐坐标 mean 的单位 scale Laplace：mean(|y−μ|)+log(2)，只训练 location，scale/pi 分支仍保留但不影响 loss。它不是最终 baseline。", "",
              "500 epoch 的 fixed-scale 已明显改善，但还失败，因此另预注册等预算 K=1 对照（见 budget_check_plan.json），各自从同一个初始权重重新训练，上限 1000；只改变 epoch budget。没有延长官方 K=6 的 500 epoch 上限。", "",
              f"**等训练步数比较：epoch {fixed['epochs']}**，free-scale K=1 ADE={matched['minADE']:.6f}m、FDE={matched['minFDE']:.6f}m；fixed-scale K=1 ADE={fixed['final']['minADE']:.6f}m、FDE={fixed['final']['minFDE']:.6f}m。初始模型 SHA256 相同，target/context/optimizer/LR/dropout 均相同，loss 中的 uncertainty 是对照变量。", "",
              "官方 K=1 最终预测（1000 epoch，仍 FAIL）：", "",
              "![Official K1](../stage2/moving_overfit_audit/single_moving_actor_k1_budget1000/target_prediction_01.png)", "",
              "同一车 fixed-scale 预测（610 epoch，PASS）：", "",
              "![Fixed-scale K1](../stage2/moving_overfit_audit/single_moving_actor_fixed_scale_budget1000/target_prediction_01.png)", "",
              "## loc / scale / pi 梯度", "", "下表为 train-mode backward、optimizer step 前的 L2 norm；每个 epoch 都记录，至少包含 1/10/50/100/final。GPU 归约可能产生微小浮点差异；新预算实验的相同 seed 不保证训练轨迹逐位相同。", "",
              "| Experiment | Epoch | loc norm | scale norm | pi norm | Pred max displacement (m) | scale mean/max (m) |", "|---|---:|---:|---:|---:|---:|---|"]
    for r in results:
        rows=list(csv.DictReader((AUDIT/r["experiment"]/"gradient_audit.csv").open()))
        for row in rows:
            if int(row["epoch"]) not in (1,10,50,100,r["epochs"]): continue
            lines.append(f"| {r['experiment']} | {row['epoch']} | {float(row['loc_grad_norm']):.6f} | {float(row['scale_grad_norm']):.6f} | {float(row['pi_grad_norm']):.6f} | {float(row['pred_max_displacement_m']):.6f} | {float(row['scale_mean']):.6f}/{float(row['scale_max']):.6f} |")
    lines += ["", "汇总文件：moving_gradient_audit.csv。预测位移和 scale 只统计 supervised target，mean/max 默认覆盖全部 mode。fixed scale 的 effective scale 恒为 1，原未使用 scale head 的输出单独标记，不能当作本实验 uncertainty。", "",
              "post-training CPU eval backward 没有 optimizer.step，额外验证每个参数进入 optimizer 一次、梯度 finite、loc 最后一层权重确实改变、最后 Linear 输出等于 raw location。",
              "| Experiment (post-training eval) | Mean |μ−y| x/y (m) | Chosen scale mean x/y (m) | Endpoint scale x/y (m) | Endpoint |dL/dμ| x/y |", "|---|---|---|---|---|"]
    for name,d in decoder["experiments"].items():
        endpoint=d["stepwise_location_scale_gradient"][-1]
        fmt=lambda values:"/".join(f"{v:.6g}" for v in values)
        lines.append(f"| {name} | {fmt(d['residual_mean_abs_xy_m'])} | {fmt(d['chosen_scale_mean_xy_m'])} | {fmt(endpoint['effective_scale_xy_m'])} | {fmt(endpoint['location_output_gradient_abs_xy'])} |")
    lines += ["", "官方 NLL 的 location 导数为 sign(μ−y)/(24b)。逐时刻 autograd 与解析梯度核对通过。一个方向大误差被较大 b 降低梯度，接近拟合的另一个方向可以通过缩小 b 降低 NLL 并产生大梯度，因此总 loc norm 非零并不表示远期主运动方向已拟合。必须结合 stepwise residual/scale/gradient 判断。", "",
              "## 输出量纲", "", "下表的 prediction 使用各实验 best-FDE mode 的 agent-local displacement，单位米。没有修改原数据尺度。", "",
              "| Experiment | Future step | GT x/y (m) | Pred x/y (m) | Pred/GT norm ratio |", "|---|---:|---|---|---:|"]
    for r in results:
        if r["experiment"] not in (k6["experiment"],k1["experiment"],fixed["experiment"]): continue
        for point in r["final"]["actors"][0]["output_units"]:
            fmt=lambda values:"/".join(f"{v:.4f}" for v in values)
            lines.append(f"| {r['experiment']} | {point['future_step']} | {fmt(point['GT_local_xy_m'])} | {fmt(point['prediction_local_xy_m'])} | {point['prediction_to_GT_norm_ratio']:.6f} |")
    lines += ["", "各时刻及训练阶段的比例不一致，没有观测到统一 5×、10× 或 0.5× 因子。保留 2Hz/5-history/12-future 数据，位置与输出始终使用米。",
              "额外的代数 range probe 在临时模型副本上，将 GT 放入最终 Linear 的 bias、weight 置零，可还原同一条 49.53m GT（误差 <1e-4m）。这是明确使用 GT 的输出范围检查，**不是训练结果、预测指标或 PASS 证据**，原 checkpoint 未改变。", "",
              "## 【Moving-only】 / 【Balanced】", "", "单目标 fixed-scale 通过后，按用户条件扩展 target 数量；沿用已通过的 K=1、固定 scale、LR=.001 诊断设置。这些不是官方 K=6 baseline 的结果。16 个 moving 来自 distinct instances；stationary 有 4 个真实 stopped、4 个真实 parked，全部完整历史与未来，各自场景 context 保留。", "",
              "| Targets | Scene windows | Epochs | Optimizer steps | ADE (m) | FDE (m) | Loss | Status |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for n,r in zip((4,8,16),moving):
        if r:
            config=r["configuration"]
            steps=r["epochs"]*((config["scene_windows"]+config["batch_size"]-1)//config["batch_size"])
            lines.append(f"| {n} moving | {config['scene_windows']} | {r['epochs']} | {steps} | {r['final']['minADE']:.6f} | {r['final']['minFDE']:.6f} | {r['final']['loss']:.6f} | {r['status']} |")
    if balanced:
        c=balanced["configuration"]; steps=balanced["epochs"]*((c["scene_windows"]+3)//4)
        lines.append(f"| 8 moving + 4 stopped + 4 parked | {c['scene_windows']} | {balanced['epochs']} | {steps} | {balanced['final']['minADE']:.6f} | {balanced['final']['minFDE']:.6f} | {balanced['final']['loss']:.6f} | {balanced['status']} |")
    else:
        lines.append("| Balanced | — | — | — | — | — | — | NOT_RUN: moving scaleup gates not both passed |")
    lines += ["", "4 targets 阈值 ADE<.75/FDE<1.5；8 targets 阈值 ADE<1/FDE<2；16 targets 为 trend only、不强制阈值。balanced 预注册诊断阈值 ADE<1/FDE<2。它在 4/8 gate 都通过后运行。每个 epoch 的更新步数随 scene-window 数变化，比较 target 数量时必须同时看 optimizer steps；不把此诊断当作严格等步数的容量曲线。",
              "以上门槛是 actor 平均指标；不能称为每辆车都已完全记忆。逐 actor identities/attributes/metrics 保存在 moving_final_actor_metrics.json。", "",
              "| Experiment | MR (>2m endpoint) | Worst actor FDE (m) |", "|---|---:|---:|"]
    for r in [x for x in moving if x]+([balanced] if balanced else []):
        measured=per_actor[r["experiment"]]
        lines.append(f"| {r['experiment']} | {measured['MR']:.4f} | {max(a['minFDE_K'] for a in measured['actors']):.6f} |")
    if balanced:
        lines += ["", "Balanced 真实 attribute 子组：", "", "| Attribute | Count | ADE (m) | FDE (m) |", "|---|---:|---:|---:|"]
        for state,row in per_actor[balanced["experiment"]]["by_real_attribute"].items():
            lines.append(f"| {state} | {row['count']} | {row['minADE']:.6f} | {row['minFDE']:.6f} |")
        subgroup=per_actor[balanced["experiment"]]["by_real_attribute"]["moving"]
        if subgroup["minADE"]>=1 or subgroup["minFDE"]>=2:
            lines += ["", "**Balanced 仅 pooled mean gate PASS；moving 子组仍未达到 8-moving 的 ADE<1/FDE<2 门槛。不能把 balanced 总体达标当作移动车全部过拟合，也不能宣称静止比例问题已经解决。**"]
    lines += ["",
              "## 【Root cause judgment】", "", finding, "", *[f"- {x}" for x in excluded], "",
              "此结论不将官方 HiVT loss 说成实现 bug；在当前 nuScenes 6s 位移、初始化和优化设置下，free-scale NLL 可以下降而 location 长距离误差仍很大。static 分布可能另有作用，但本轮没有验证其独立贡献。", "",
              "## 【Next Stage】", "", "重新运行完整 tiny：**NO**。mini 完整训练：**NO**。Stage 3：**NO**。", "", final["next_work"], "",
              "## 验证与命令", "", "7 个单元测试通过，包括原 baseline 4 个和新诊断 3 个；额外 source/roundtrip/gradient/optimizer/output range 检查通过，19 个 upstream 文件 SHA256 保持原值。",
              "启动阶段最初的 GPU timestamps probe 使用严格零误差断言，受 scatter 归约浮点变化影响失败；当时尚未创建 optimizer/训练。已改用相同权重的 CPU probe，完整保留 startup_failure_run.txt 与无训练配置，避免将这种误差误判为时间尺度依赖。",
              "完整执行命令见 docs/moving_overfit_audit_commands.md。原 tiny 冻结文件再次验证 24/24 相同。"]
    (PROJECT_ROOT/"outputs/reports/moving_overfit_audit_report.md").write_text("\n".join(lines)+"\n")
    verify_freeze()
    print(json.dumps(final,indent=2,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
