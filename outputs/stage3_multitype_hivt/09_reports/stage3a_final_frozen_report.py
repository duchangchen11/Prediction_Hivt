"""Audit and document the scientifically budgeted, final frozen No-Type baseline."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage3_common import CONFIG, atomic_json, config, read_json, sha256
sys.path.insert(0, str(ROOT / "04_evaluation"))
from stage3a_final_frozen_evaluate import (ACTORS, MANIFEST, METRICS_PATH, FIELDS,
                                         checkpoint_from_manifest, read_actors,
                                         verify_actor_aggregation, actor_summary)
import numpy as np


def yes(value):
    return "YES" if value else "NO"


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def verify_training(final):
    training = read_json(ROOT / "07_checkpoints/stage3a_plusplus_training_manifest.json")
    assert training["status"] == final["status"] == "COMPLETE"
    assert training["start_global_step"] == final["start_global_step"] == 18000
    executed = training["executed_extra_steps"]
    assert executed == final["executed_extra_steps"] and 0 < executed <= 3000 and executed % 500 == 0
    assert final["final_executed_global_step"] == 18000 + executed <= 21000
    assert final["converged_by_patience"] == (training["consecutive_nonimprovements"] >= 5)
    assert final["stopped_by_budget"] == (final["final_executed_global_step"] == 21000)
    assert final["converged_by_patience"] or final["stopped_by_budget"]
    assert final["stop_reason"] == ("patience_5" if final["converged_by_patience"] else "global_step_21000_budget")
    assert final["LR"] == 1e-4 and final["batch_size"] == 16
    assert final["validation_interval"] == 500 and final["patience"] == 5
    assert final["config_sha256"] == sha256(CONFIG)
    assert not final["Type_Embedding"] and not final["test_used"]
    curve = read_csv(ROOT / "03_no_type_baseline/stage3a_plusplus_nll_curve.csv")
    assert [int(float(r["global_step"])) for r in curve] == list(range(18500, 18000 + executed + 1, 500))
    source = read_json(ROOT / "04_evaluation/stage3a_plus_best_val_metrics.json")["metrics"]["full_horizon"]["overall"]["minFDE6"]
    best = source; best_step = 18000; bad = 0
    for row in curve:
        current = float(row["VAL_overall_FDE"])
        improvement = current < best
        assert bool(int(float(row["improved"]))) == improvement
        bad = 0 if improvement else bad + 1
        if improvement:
            best = current; best_step = int(float(row["global_step"]))
        assert float(row["learning_rate"]) == 1e-4
        assert int(float(row["consecutive_nonimprovements"])) == bad
        assert abs(float(row["best_overall_FDE"]) - best) <= 1e-12
        assert int(float(row["best_step"])) == best_step
        # All five requested groups have metrics from the same complete VAL pass.
        for group in ("overall", "vehicle", "pedestrian", "bicycle", "vehicle.moving"):
            assert all(np.isfinite(float(row["VAL_" + group + "_" + field]))
                       for field in ("ADE", "FDE", "MR", "Top1ADE", "Top1FDE", "NLL", "count"))
    assert best_step == final["final_best_global_step"]
    assert abs(best - final["overall_minFDE6"]) <= 1e-6
    assert bad == training["consecutive_nonimprovements"]
    return training, curve, source


def verify_subsets(actors, motion):
    assert len(motion) == 5
    for row in motion:
        selected = [r for r in actors if r["horizon"] == "full_horizon" and r["agent_type"] == row["AgentType"]
                    and float(r["GT_endpoint_displacement_m"]) > row["GT_endpoint_displacement_gt_m"]]
        actual = actor_summary(selected, FIELDS[:-1])
        assert actual.pop("count") == row["Count"]
        assert len({r["instance_token"] for r in selected}) == row["UniqueInstances"]
        assert len({r["scene_token"] for r in selected}) == row["UniqueScenes"]
        assert all(abs(actual[k] - row[k]) <= 1e-12 for k in actual)


def main():
    final = read_json(MANIFEST); checkpoint = checkpoint_from_manifest(final)
    training, curve, source_fde = verify_training(final)
    measured = read_json(METRICS_PATH)
    evaluation = read_json(ROOT / "04_evaluation/stage3a_final_frozen_evaluation_audit.json")
    assert measured["status"] == evaluation["status"] == "PASS"
    assert measured["fresh_full_official_VAL_inference"] and not measured["old_actor_CSV_concatenated"]
    assert evaluation["official_VAL_complete"] and evaluation["checkpoint_valid"]
    assert measured["checkpoint_sha256"] == evaluation["checkpoint_sha256"] == sha256(checkpoint)
    assert measured["actor_errors_sha256"] == evaluation["actor_errors_sha256"] == sha256(ACTORS)
    assert measured["NaN"] == measured["Inf"] == evaluation["NaN"] == evaluation["Inf"] == 0
    actors, finite = read_actors(ACTORS); verify_actor_aggregation(actors, measured)
    motion_record = read_json(ROOT / "04_evaluation/stage3a_final_frozen_nontrivial_motion_metrics.json")
    assert motion_record["checkpoint_sha256"] == sha256(checkpoint)
    motion = motion_record["metrics"]; verify_subsets(actors, motion)
    retention = read_json(ROOT / "04_evaluation/stage3a_final_vehicle_retention_audit.json")
    assert retention["status"] == "PASS" and retention["exactly_paired_vehicle_actor_windows"] == 62981
    assert retention["final_checkpoint_sha256"] == sha256(checkpoint)
    assert retention["Stage3A_final_actor_errors_sha256"] == sha256(ACTORS)
    interaction_path = ROOT / "01_data_audit/stage3_interaction_density.json"
    interaction = read_json(interaction_path); assert interaction["status"] == "PASS"
    interaction_sources = evaluation["interaction_files_reused_without_scan"]
    assert all(sha256(ROOT / name) == digest for name, digest in interaction_sources.items())
    means = {}; pairs = {}
    for cls in ("vehicle", "pedestrian", "bicycle"):
        means[cls] = next(r for r in interaction["neighbor_statistics"] if r["Split"] == "all"
                          and r["TargetType"] == cls and r["NeighborType"] == "all" and r["Radius_m"] == 20)
    for name in ("vehicle-vehicle", "vehicle-pedestrian", "vehicle-bicycle"):
        pairs[name] = next(r for r in interaction["pair_counts"] if r["Split"] == "all"
                          and r["Pair"] == name and r["Radius_m"] == 20)
    ready = (finite["NaN"] == finite["Inf"] == 0 and evaluation["checkpoint_valid"]
             and evaluation["official_VAL_complete"] and retention["vehicle_no_abnormal_collapse"])
    decision = {"Stage3A_frozen": "YES", "Ready_for_Stage3B_Type_Embedding": yes(ready),
                "converged_by_patience": bool(final["converged_by_patience"]),
                "stopped_by_budget": bool(final["stopped_by_budget"]),
                "scientific_freeze_rule": "Freeze at patience 5 or global step 21000; no further No-Type extension",
                "patience_is_not_a_readiness_prerequisite": True,
                "final_checkpoint_relative_path": str(checkpoint.relative_to(ROOT)),
                "final_checkpoint_sha256": sha256(checkpoint),
                "final_best_global_step": final["final_best_global_step"],
                "NaN": 0, "Inf": 0, "checkpoint_valid": True, "official_VAL_complete": True,
                "vehicle_no_abnormal_collapse": retention["vehicle_no_abnormal_collapse"],
                "vehicle_retention_FDE_ratios": retention["vehicle_retention_FDE_ratios"],
                "Stage3B_executed": False, "further_No_Type_training_allowed": False,
                "future_ablation_protocol": final["future_ablation_protocol"]}
    atomic_json(ROOT / "09_reports/stage3a_final_frozen_decision.json", decision)
    # Enhance only this task's new manifest. Earlier manifests and checkpoints remain untouched.
    final.update(official_VAL_fresh_evaluation_complete=True,
                 final_actor_errors_relative_path=str(ACTORS.relative_to(ROOT)), final_actor_errors_sha256=sha256(ACTORS),
                 final_evaluation_metrics_relative_path=str(METRICS_PATH.relative_to(ROOT)),
                 final_evaluation_metrics_sha256=sha256(METRICS_PATH),
                 final_fresh_full_horizon_metrics=measured["metrics"]["full_horizon"],
                 stage3a_frozen="YES", Ready_Stage3B=yes(ready),
                 Ready_for_Stage3B_Type_Embedding=yes(ready), decision=decision)
    atomic_json(MANIFEST, final)
    full = measured["metrics"]["full_horizon"]; overall = full["overall"]
    relative = 1 - overall["minFDE6"] / source_fde
    update_cases = relative >= .005
    lines = ["# Stage3A 最终冻结报告", "",
             "本轮从 18000-step 正式 best 恢复模型、AdamW、CPU/CUDA 随机状态和 scene sampler 游标，仅继续原始 learnable-scale Laplace NLL。数据、完整 scene-window、HiVT64、LR=1e-4、batch=16、K=6、Th=5、Tf=12 均沿用冻结协议。", "",
             "【Training】", "", "Start step=18000",
             f"Final executed step={final['final_executed_global_step']}",
             f"Final best step={final['final_best_global_step']}", f"Stop reason={final['stop_reason']}",
             f"converged_by_patience={yes(final['converged_by_patience'])}",
             f"stopped_by_budget={yes(final['stopped_by_budget'])}",
             f"Executed extra optimizer steps={final['executed_extra_steps']}; full official VAL evaluations={len(curve)}.", "",
             "每 500 步运行完整 official VAL150；只用 full-horizon overall minFDE6 严格改善选择 checkpoint。连续 5 次不改善可停止，global step 21000 为不可再延长的预算上限。预算停止与 patience 收敛分开报告，固定预算并不证明全局最优。", "",
             f"Training code commit: `{final['training_code_git_commit']}`; config SHA256: `{final['config_sha256']}`.", "",
             "![Final convergence](../05_figures/stage3a_final_convergence.png)", "",
             "【Final Overall】", "", f"ADE={overall['minADE6']:.9f} m",
             f"FDE={overall['minFDE6']:.9f} m", f"MR={overall['MR6']:.9f}",
             f"Top1 ADE={overall['Top1ADE6']:.9f} m", f"Top1 FDE={overall['Top1FDE6']:.9f} m", "",
             f"最终 checkpoint 独立重新推理所有 3603 个有监督 VAL windows，覆盖 150 scenes；得到 85027 actor-windows，其中完整 12 步 54990，partial future 30037。NaN=0，Inf=0；最终 actor CSV SHA256: `{sha256(ACTORS)}`。该 CSV 来自新推理，不由旧结果拼接。报告及表格采用最终 fresh evaluation；manifest 的 selection 字段保留训练中选中 VAL 值，并另存 final_fresh_full_horizon_metrics。两次评价的已核对指标误差 <=1e-6。", "",
             "minADE6 为 best-FDE mode 的 ADE；MR6 为 endpoint error >2 m；Top1 使用模型最大概率 mode。NLL 沿用原始 best-summed-L2 mode 的 Laplace valid-time 坐标密度均值，允许负值。所有主指标对 actor-window 等权。", "",
             "【Per Class】", "", "| Group | Count | minADE6 (m) | minFDE6 (m) | MR6 | Top1ADE6 (m) | Top1FDE6 (m) | NLL |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("vehicle", "pedestrian", "bicycle"):
        values = full[name]
        lines.append(f"| {name.capitalize()} | {values['count']} | " + " | ".join(f"{values[k]:.6f}" for k in FIELDS) + " |")
    lines += ["", "[Full main results](../06_tables/stage3a_final_frozen_main_results.csv)", "", "【Motion Groups】", "",
              "只统计完整 12 步未来，按 norm(GT endpoint − t0 position) 严格大于阈值分组。Count 为 actor-window；同一 instance 的重叠窗口有相关性，两个阈值组嵌套。运动 bicycle 单独报告，并列出独立 instance/scene 覆盖。", "",
              "| Group | Count | UniqueInstances | UniqueScenes | minADE6 (m) | minFDE6 (m) | MR6 | Top1ADE6 (m) | Top1FDE6 (m) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in motion:
        lines.append(f"| {row['Group']} | {row['Count']} | {row['UniqueInstances']} | {row['UniqueScenes']} | "
                     + " | ".join(f"{row[k]:.6f}" for k in FIELDS[:-1]) + " |")
    lines += ["", "[Nontrivial motion table](../06_tables/stage3a_final_frozen_nontrivial_motion_metrics.csv)", "", "【Vehicle Retention】", "",
              "Stage2C vehicle-only 与最终 No-Type 逐 actor 配对：scene/sample/instance/horizon 完全一致，62981 个 full+partial vehicle actor-windows；future mask length、motion state、GT endpoint displacement 一致。下表仅完整未来，未删减 actor。", "",
              "| Group | Count | Stage2C ADE/FDE/MR | Stage3A final ADE/FDE/MR | Stage3A/Stage2C FDE ratio | relative change |",
              "|---|---:|---|---|---:|---:|"]
    for row in retention["metrics"]:
        old = "/".join(f"{row['Stage2C_' + k]:.6f}" for k in FIELDS[:3])
        new = "/".join(f"{row['Stage3A_' + k]:.6f}" for k in FIELDS[:3])
        lines.append(f"| {row['Group']} | {row['Count']} | {old} | {new} | {row['Stage3A_over_Stage2C_FDE_ratio']:.6f} | {row['FDE_relative_change_percent']:+.3f}% |")
    lines += ["", "[Vehicle retention table](../06_tables/stage3a_final_vehicle_retention.csv)", "",
              "Vehicle overall/moving FDE ratio <=1.25 为既有异常退化防护阈值；它不选择 checkpoint，也不改变训练。", "",
              "【Interaction Dataset】", "",
              "本轮未为交互密度重新扫描 graph；直接引用既有冻结 Stage3A+ 统计。以下为 train700 + val150 pooled、20 m、self excluded，邻居包括所有当前三类 context actors，目标含完整和部分未来。", "",
              "20m mean neighbors:",
              *[f"{cls.capitalize()}={means[cls]['MeanNeighborCount']:.6f} (targets={means[cls]['TargetCount']})" for cls in means], ""]
    for name, short in (("vehicle-vehicle", "V-V"), ("vehicle-pedestrian", "V-P"), ("vehicle-bicycle", "V-B")):
        row = pairs[name]
        lines.append(f"{short} pairs={row['UniqueTargetIncidentPairs']}; windows={row['WindowsWithTargetIncidentPair']}")
    lines += ["", "Pairs 在每个 window 内无序去重且至少一端为监督 target；同一物理 pair 可在不同 window 重复。空间邻近描述可用 context，不证明因果交互。", "",
              "[Frozen interaction CSV](../01_data_audit/stage3_interaction_density.csv), [JSON](../01_data_audit/stage3_interaction_density.json).", "",
              f"Frozen interaction JSON SHA256: `{sha256(interaction_path)}`.", "",
              "【Motion Case Policy】", "", f"Overall FDE improvement relative to step18000={relative * 100:.6f}%."]
    if not update_cases:
        lines += ["改善小于 0.5%，按本轮要求保留 Stage3A+ 的三个 motion case 图及其真实 source JSON，未重新生成。这些旧图的预测来源为 step18000 checkpoint，不用于声称最终 checkpoint 的案例性能。",
                  "[Existing Stage3A+ case manifest](../04_evaluation/stage3a_plus_motion_case_manifest.json)."]
    else:
        case_path = ROOT / "04_evaluation/stage3a_final_motion_case_manifest.json"
        assert case_path.is_file(), "Improvement >=0.5% requires updated final motion cases before report"
        case_record = read_json(case_path)
        assert case_record["source_checkpoint_sha256"] == sha256(checkpoint)
        assert case_record["case_count"] == 6 and len(case_record["figures"]) == 3
        lines += ["改善达到 0.5%，三个 motion case 已由最终 checkpoint 的真实预测更新，并保留原 Stage3A+ 图。未来每条轨迹显示全部 12 个 marker，局部放大来自真实最后 6 点。",
                  "[Updated final case manifest](../04_evaluation/stage3a_final_motion_case_manifest.json)."]
    lines += ["", "【Scientific Decision】", "", "Stage3A frozen = YES",
              f"Final baseline checkpoint=`{checkpoint.relative_to(ROOT)}`",
              f"Final checkpoint SHA256=`{sha256(checkpoint)}`", f"Ready for Stage3B Type Embedding = {yes(ready)}", "",
              "按预先限定的 patience/21000-step 预算正式冻结 No-Type baseline；本轮结束后不再追加 No-Type 训练。Stage3B 后续消融必须采用相同最大 global step=21000、validation interval=500、full-horizon official VAL overall minFDE6 严格改善 selection，保证比较的预算一致。", "",
              "本轮未执行 Type Embedding。checkpoint、shards 和大型 actor CSV 留在本地；代码及小型可追溯结果提交 stage3/multitype-hivt，不 merge main。", ""]
    (ROOT / "09_reports/stage3a_final_frozen_report.md").write_text("\n".join(lines))
    artifact_paths = [METRICS_PATH, MANIFEST, ROOT / "09_reports/stage3a_final_frozen_report.md",
                      ROOT / "06_tables/stage3a_final_frozen_main_results.csv",
                      ROOT / "06_tables/stage3a_final_frozen_nontrivial_motion_metrics.csv",
                      ROOT / "06_tables/stage3a_final_vehicle_retention.csv"]
    atomic_json(ROOT / "00_manifest/stage3a_final_frozen_report_audit.json", {
        "status": "PASS", "training_curve_strict_selection_verified": True,
        "executed_extra_steps": final["executed_extra_steps"], "validation_count": len(curve),
        "actor_aggregations_independently_verified": True, "motion_group_counts_verified": 5,
        "exact_vehicle_actor_pairing_verified": True, "NaN": 0, "Inf": 0,
        "interaction_files_reused_without_scan": interaction_sources,
        "motion_cases_update_required": update_cases, "relative_FDE_improvement_vs_18000": relative,
        "artifact_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in artifact_paths},
        "figure_render_and_export_verification_delegated_to_final_artifact_audit": True,
        "decision": decision, "Stage3B_executed": False, "test_used": False})
    print("FINAL_FROZEN_REPORT_AUDIT=PASS", decision, flush=True)


if __name__ == "__main__":
    main()
