"""CPU-only, independent evidence audit after final No-Type evaluation/report.

This script performs no forward pass, optimizer update, preprocessing, interaction
scan or Git mutation. Only its new audit JSON is written. Run after the final
manifest, fresh actor CSV, tables and report have been finalized.
"""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

# Keep the audit unable to initialize or claim GPU execution.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
sys.path.insert(0, str(ROOT / "03_no_type_baseline"))
from stage3a_plus_finalize_checkpoint import model_digest, state_equal
from stage3_common import CONFIG, MODEL_KEYS, config
from models.hivt_loss_recovery import HiVTLossRecovery
import numpy as np
import torch

OUT = ROOT / "00_manifest/stage3a_final_frozen_independent_audit.json"
METRICS = ("minADE6", "minFDE6", "MR6", "Top1ADE6", "Top1FDE6", "NLL")
GROUPS = ("overall", "vehicle", "pedestrian", "bicycle", "vehicle.moving",
          "vehicle.stopped", "vehicle.parked", "unknown")
FULL_COUNTS = dict(zip(GROUPS, (54990, 42332, 12002, 656, 10461, 5599, 25198, 1074)))
PARTIAL_COUNTS = dict(zip(GROUPS, (30037, 20649, 8738, 650, 5478, 1997, 12589, 585)))
MOTION_GROUPS = (("vehicle", 5, 9744, 853, 137),
                 ("pedestrian", 1, 8810, 825, 113),
                 ("pedestrian", 5, 7581, 716, 110),
                 ("bicycle", 1, 155, 15, 13),
                 ("bicycle", 5, 132, 12, 11))


def read_json(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def close(a, b, tolerance=1e-10):
    a, b = float(a), float(b)
    assert math.isfinite(a) and math.isfinite(b)
    assert abs(a - b) <= tolerance, (a, b, tolerance)
    return abs(a - b)


def actor_key(row):
    return tuple(row[key] for key in ("scene_token", "sample_token", "instance_token", "horizon"))


def select(actors, horizon, group):
    return [r for r in actors if r["horizon"] == horizon and
            (group == "overall" or r["agent_type"] == group or
             (r["agent_type"] == "vehicle" and r["motion_state"] == group))]


def summary(actors, fields=METRICS):
    assert actors
    values = np.array([[float(r[k]) for k in fields] for r in actors], dtype=np.float64)
    assert np.isfinite(values).all()
    return {"count": len(actors), **dict(zip(fields, map(float, values.mean(axis=0))))}


def optimizer_steps(saved):
    state = saved["optimizer_state_dict"]["state"]
    assert len(state) == 275
    steps = []
    for entry in state.values():
        assert all(torch.isfinite(v).all() for v in entry.values() if torch.is_tensor(v))
        steps.append(float(entry["step"]))
    assert len(set(steps)) == 1
    return steps[0]


def groups_without_params(saved):
    return [{k: v for k, v in group.items() if k != "params"}
            for group in saved["optimizer_state_dict"]["param_groups"]]


def validate_checkpoint(saved, source, global_step, steps_per_epoch, commit, structural_mask):
    delta = global_step - 18000
    meta = saved["metadata"]
    assert meta["global_step"] == global_step and meta["phase"] == "original_nll"
    assert meta["phase_step"] == source["metadata"]["phase_step"] + delta
    assert saved["config"] == source["config"] == config()
    assert meta["config_sha256"] == sha(CONFIG)
    assert meta["selection_metric"] == "overall minFDE6"
    assert groups_without_params(saved) == groups_without_params(source)
    assert all(group["lr"] == 1e-4 for group in groups_without_params(saved))
    assert optimizer_steps(saved) == 15500 + delta
    iterator = saved["iterator"]; origin = source["iterator"]
    assert 0 <= iterator["next_batch"] <= steps_per_epoch and iterator["epoch"] >= origin["epoch"]
    assert (iterator["epoch"] * steps_per_epoch + iterator["next_batch"] ==
            origin["epoch"] * steps_per_epoch + origin["next_batch"] + delta)
    close(meta["epoch"], iterator["epoch"] + iterator["next_batch"] / steps_per_epoch, 1e-12)
    assert saved["torch_rng"].dtype == torch.uint8 and saved["torch_rng"].numel() == 5056
    assert len(saved["cuda_rng"]) == 1
    assert all(t.dtype == torch.uint8 and tuple(t.shape) == (16,) for t in saved["cuda_rng"])
    assert torch.equal(saved["state_dict"]["local_encoder.temporal_encoder.attn_mask"], structural_mask)
    if delta:
        assert meta["plusplus_step"] == saved["plusplus_state"]["extension_step"] == delta
        assert meta["git_commit_SHA"] == commit
    return {"global_step": global_step, "phase_step": meta["phase_step"],
            "AdamW_state_count": 275, "AdamW_step": optimizer_steps(saved),
            "sampler_cursor": iterator, "cursor_update_count_verified": delta,
            "CPU_and_CUDA_RNG_payload_valid": True}


def main():
    torch.set_num_threads(1)
    final_path = ROOT / "07_checkpoints/stage3a_final_frozen_checkpoint_manifest.json"
    final = read_json(final_path)
    training = read_json(ROOT / "07_checkpoints/stage3a_plusplus_training_manifest.json")
    assert final["status"] == training["status"] == "COMPLETE"
    assert final["official_VAL_fresh_evaluation_complete"]
    assert final["stage3a_frozen"] == training["stage3a_frozen"] == "YES"
    assert final["further_No_Type_training_allowed"] is False
    prereg = read_json(ROOT / "00_manifest/stage3a_plusplus_preregistration.json")
    frozen = read_json(ROOT / "00_manifest/stage3a_plusplus_frozen_previous.json")
    # Hash existing protected files; do not open/recompute graph interactions or shards.
    for name, digest in frozen["files"].items():
        assert sha(PROJECT / name) == digest, "Protected previous artifact changed: " + name
    for name, digest in prereg["source_sha256"].items():
        assert sha(ROOT / name) == digest, "Registered optimization source changed: " + name
        blob = subprocess.check_output(["git", "show", training["training_code_git_commit"] + ":" +
                                        str((ROOT / name).relative_to(PROJECT))], cwd=PROJECT)
        assert hashlib.sha256(blob).hexdigest() == digest
    source_path = ROOT / "07_checkpoints/stage3a_plus_best_overall_minfde.pt"
    assert sha(source_path) == prereg["source_checkpoint_sha256"] == final["source_checkpoint_sha256"]
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    assert source["metadata"]["global_step"] == 18000
    assert optimizer_steps(source) == 15500
    assert source["iterator"] == {"epoch": 12, "next_batch": 316}
    assert source["config"] == config() and sha(CONFIG) == final["config_sha256"]
    assert config()["batch_size"] == 16 and config()["num_modes"] == 6
    assert not config()["type_embedding"] and not config()["class_balanced_loss"] and not config()["oversampling"]

    extra = int(final["executed_extra_steps"])
    last_global = int(final["final_executed_global_step"])
    assert 0 < extra <= 3000 and extra % 500 == 0 and last_global == 18000 + extra <= 21000
    assert training["executed_extra_steps"] == extra and training["final_executed_global_step"] == last_global
    curve = rows(ROOT / "03_no_type_baseline/stage3a_plusplus_nll_curve.csv")
    assert [int(float(r["global_step"])) for r in curve] == list(range(18500, last_global + 1, 500))
    best_fde = float(source["metadata"]["validation_FDE"]); best_global = 18000; bad = 0
    curve_json_deltas = []; val_checks = []
    aliases = {"count": "count", "minADE6": "ADE", "minFDE6": "FDE", "MR6": "MR",
               "Top1ADE6": "Top1ADE", "Top1FDE6": "Top1FDE", "NLL": "NLL"}
    for row in curve:
        step = int(float(row["global_step"]))
        measured = read_json(ROOT / f"04_evaluation/stage3a_plusplus_val_step_{step:05d}.json")
        assert (measured["windows"], measured["candidate_windows"], measured["empty_supervision_windows"]) == (3603, 3619, 16)
        assert measured["split"] == "val" and measured["K"] == 6 and not measured["test_used"]
        assert len(measured["scenes"]) == 150
        assert measured["plusplus_step"] == measured["checkpoint_metadata"]["plusplus_step"] == step - 18000
        assert measured["checkpoint_metadata"]["global_step"] == step
        full = measured["metrics"]["full_horizon"]
        for group in ("overall", "vehicle", "pedestrian", "bicycle", "vehicle.moving"):
            assert full[group]["count"] == FULL_COUNTS[group]
            for metric, alias in aliases.items():
                curve_json_deltas.append(close(row["VAL_" + group + "_" + alias], full[group][metric]))
        current = float(full["overall"]["minFDE6"])
        improved = current < best_fde
        assert bool(int(float(row["improved"]))) == improved
        assert int(float(row["extension_step"])) == step - 18000
        close(row["learning_rate"], 1e-4, 0)
        bad = 0 if improved else bad + 1
        if improved: best_fde, best_global = current, step
        close(row["best_overall_FDE"], best_fde, 1e-12)
        assert int(float(row["best_step"])) == best_global
        assert int(float(row["consecutive_nonimprovements"])) == bad
        assert bad < 5 or step == last_global, "Training continued after patience exhausted"
        val_checks.append({"global_step": step, "overall_minFDE6": current, "strict_improvement": improved,
                           "consecutive_nonimprovements": bad, "full_VAL_scenes": 150})
    assert final["final_best_global_step"] == training["final_best_global_step"] == best_global
    close(final["overall_minFDE6"], best_fde, 1e-12)
    assert final["converged_by_patience"] == training["converged_by_patience"] == (bad >= 5)
    assert final["stopped_by_budget"] == training["stopped_by_budget"] == (last_global == 21000)
    assert final["converged_by_patience"] or final["stopped_by_budget"]
    expected_reason = "patience_5" if bad >= 5 else "global_step_21000_budget"
    assert final["stop_reason"] == training["stop_reason"] == expected_reason
    assert training["consecutive_nonimprovements"] == bad

    last_path = ROOT / "07_checkpoints/stage3a_plusplus_last_checkpoint.pt"
    assert sha(last_path) == training["last_checkpoint_sha256"]
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    selected_path = ROOT / final["checkpoint_relative_path"]
    assert sha(selected_path) == final["checkpoint_sha256"]
    assert selected_path.name == ("stage3a_final_frozen_best_overall_minfde.pt" if best_global > 18000
                                  else "stage3a_plus_best_overall_minfde.pt")
    selected = torch.load(selected_path, map_location="cpu", weights_only=False)
    index = rows(ROOT / "02_preprocessed/stage3_train_index.csv")
    windows = sum(int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) > 0 for r in index)
    assert windows == 16898; steps_per_epoch = math.ceil(windows / 16); assert steps_per_epoch == 1057
    model = HiVTLossRecovery(**{k: config()[k] for k in MODEL_KEYS})
    mask = model.state_dict()["local_encoder.temporal_encoder.attn_mask"].clone()
    cp_audits = {"last": validate_checkpoint(last, source, last_global, steps_per_epoch,
                                              training["training_code_git_commit"], mask),
                 "selected": validate_checkpoint(selected, source, best_global, steps_per_epoch,
                                                  training["training_code_git_commit"], mask)}
    assert last["plusplus_state"]["completed"] and last["plusplus_state"]["bad_validations"] == bad
    assert last["plusplus_state"]["best_step"] == best_global
    assert last["iterator"] == training["final_iterator"]
    model.load_state_dict(selected["state_dict"], strict=True)
    assert sum(p.numel() for p in model.parameters()) == 645809
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert sum(int(torch.isnan(t).sum()) for t in model.state_dict().values()) == 0
    nonfinite_buffers = [name for name, value in model.named_buffers() if not torch.isfinite(value).all()]
    assert nonfinite_buffers == ["local_encoder.temporal_encoder.attn_mask"]
    assert int(torch.isneginf(mask).sum()) == 15
    opt = model.optimizer(1e-4, config()["weight_decay"])
    opt.load_state_dict(selected["optimizer_state_dict"])
    assert state_equal(opt.state_dict(), selected["optimizer_state_dict"])
    selected_model_sha = model_digest(selected["state_dict"])
    assert selected_model_sha == final["model_state_content_sha256"]

    actors_path = ROOT / "04_evaluation/stage3a_final_frozen_actor_errors.csv"
    actors = rows(actors_path)
    assert len(actors) == len({actor_key(r) for r in actors}) == 85027
    assert {r["horizon"] for r in actors} == {"full_horizon", "partial_future"}
    assert {r["agent_type"] for r in actors} == {"vehicle", "pedestrian", "bicycle"}
    assert len({r["scene_token"] for r in actors}) == 150
    numeric_fields = METRICS + ("independent_minADE6", "GT_endpoint_displacement_m")
    numbers = np.array([[float(r[k]) for k in numeric_fields] for r in actors])
    assert np.isfinite(numbers).all()
    for row in actors:
        valid = int(row["valid_future_steps"])
        assert valid == 12 if row["horizon"] == "full_horizon" else 1 <= valid < 12
        assert float(row["MR6"]) == float(float(row["minFDE6"]) > 2)
        assert 0 <= int(row["best_mode"]) < 6 and 0 <= int(row["top1_mode"]) < 6
    measured_path = ROOT / "04_evaluation/stage3a_final_frozen_metrics.json"
    measured = read_json(measured_path)
    assert measured["split"] == "val" and measured["K"] == 6 and not measured["test_used"]
    assert measured["fresh_full_official_VAL_inference"] and not measured["old_actor_CSV_concatenated"]
    assert measured["NaN"] == measured["Inf"] == 0 and len(measured["scenes"]) == 150
    assert (measured["windows"], measured["candidate_windows"], measured["empty_supervision_windows"]) == (3603, 3619, 16)
    assert measured["checkpoint_sha256"] == final["checkpoint_sha256"]
    assert measured["actor_errors_sha256"] == final["final_actor_errors_sha256"] == sha(actors_path)
    actor_deltas = []
    for horizon in ("full_horizon", "partial_future"):
        for group in GROUPS:
            values = summary(select(actors, horizon, group))
            expected = measured["metrics"][horizon][group]
            assert values["count"] == expected["count"]
            assert values["count"] == (FULL_COUNTS if horizon == "full_horizon" else PARTIAL_COUNTS)[group]
            actor_deltas.extend(close(values[k], expected[k]) for k in METRICS)
    full = measured["metrics"]["full_horizon"]
    selected_full = final["selected_full_horizon_metrics"]
    selected_fresh_deltas = {group: {k: close(full[group][k], selected_full[group][k], 1e-6)
                                   for k in METRICS} for group in GROUPS}
    main_table = rows(ROOT / "06_tables/stage3a_final_frozen_main_results.csv")
    labels = {"Overall": "overall", "Vehicle": "vehicle", "Pedestrian": "pedestrian", "Bicycle": "bicycle"}
    assert len(main_table) == 8
    assert set(main_table[0]) == {"Group", "Count", *METRICS}
    assert {labels.get(r["Group"], r["Group"]) for r in main_table} == set(GROUPS)
    for row in main_table:
        values = full[labels.get(row["Group"], row["Group"])]
        assert int(float(row["Count"])) == values["count"]
        for k in METRICS: close(row[k], values[k])
    motion = rows(ROOT / "06_tables/stage3a_final_frozen_nontrivial_motion_metrics.csv")
    assert len(motion) == 5
    motion_audits = []
    for cls, threshold, count, instances, scenes in MOTION_GROUPS:
        eligible = [r for r in actors if r["horizon"] == "full_horizon" and r["agent_type"] == cls
                    and float(r["GT_endpoint_displacement_m"]) > threshold]
        found = [r for r in motion if r["AgentType"] == cls and float(r["GT_endpoint_displacement_gt_m"]) == threshold]
        assert len(found) == 1; table = found[0]
        actual = summary(eligible, METRICS[:-1])
        assert actual["count"] == int(table["Count"]) == count
        assert len({r["instance_token"] for r in eligible}) == int(table["UniqueInstances"]) == instances
        assert len({r["scene_token"] for r in eligible}) == int(table["UniqueScenes"]) == scenes
        for k in METRICS[:-1]: close(actual[k], table[k])
        motion_audits.append({"AgentType": cls, "strict_GT_displacement_threshold_m": threshold,
                             "Count": count, "UniqueInstances": instances, "UniqueScenes": scenes,
                             **{k: actual[k] for k in METRICS[:-1]}})

    previous = PROJECT / "outputs/stage2c_trainval_vehicle_baseline"
    old_actors = rows(previous / "04_evaluation/stage2c_val_actor_errors.csv")
    old_by_key = {actor_key(r): r for r in old_actors}
    vehicle_by_key = {actor_key(r): r for r in actors if r["agent_type"] == "vehicle"}
    assert len(old_actors) == len(old_by_key) == len(vehicle_by_key) == 62981
    assert old_by_key.keys() == vehicle_by_key.keys()
    displacement_deltas = []
    for key, old in old_by_key.items():
        new = vehicle_by_key[key]
        assert old["valid_future_steps"] == new["valid_future_steps"] and old["motion_state"] == new["motion_state"]
        displacement_deltas.append(close(old["GT_endpoint_displacement_m"], new["GT_endpoint_displacement_m"], 1e-6))
    retention = rows(ROOT / "06_tables/stage3a_final_vehicle_retention.csv")
    ratios = {}
    for group in ("vehicle", "vehicle.moving", "vehicle.stopped", "vehicle.parked", "unknown"):
        label = "Vehicle overall" if group == "vehicle" else group
        table = next(r for r in retention if r["Group"] == label)
        old_subset = [r for r in old_actors if r["horizon"] == "full_horizon" and
                      (group == "vehicle" or r["motion_state"] == group)]
        old = summary(old_subset, METRICS[:3]); new = summary(select(actors, "full_horizon", group), METRICS[:3])
        assert old["count"] == new["count"] == int(table["Count"])
        for k in METRICS[:3]: close(old[k], table["Stage2C_" + k]); close(new[k], table["Stage3A_" + k])
        ratio = new["minFDE6"] / old["minFDE6"]
        close(ratio, table["Stage3A_over_Stage2C_FDE_ratio"])
        close(100 * (ratio - 1), table["FDE_relative_change_percent"])
        ratios[group] = ratio
    ready = ratios["vehicle"] <= 1.25 and ratios["vehicle.moving"] <= 1.25
    decision = read_json(ROOT / "09_reports/stage3a_final_frozen_decision.json")
    assert decision["Stage3A_frozen"] == "YES"
    assert decision["Ready_for_Stage3B_Type_Embedding"] == final["Ready_for_Stage3B_Type_Embedding"] == ("YES" if ready else "NO")
    assert decision["converged_by_patience"] == final["converged_by_patience"]
    assert decision["stopped_by_budget"] == final["stopped_by_budget"]
    assert not decision["Stage3B_executed"] and not decision["further_No_Type_training_allowed"]
    assert final["future_ablation_protocol"]["maximum_global_step"] == 21000
    assert final["future_ablation_protocol"]["validation_interval"] == 500
    report_path = ROOT / "09_reports/stage3a_final_frozen_report.md"
    report = report_path.read_text()
    assert "Stage3A frozen = YES" in report
    assert f"Ready for Stage3B Type Embedding = {'YES' if ready else 'NO'}" in report
    assert f"converged_by_patience={'YES' if bad >= 5 else 'NO'}" in report
    assert f"stopped_by_budget={'YES' if last_global == 21000 else 'NO'}" in report
    assert final["checkpoint_sha256"] in report
    checked_artifacts = [final_path, measured_path, actors_path, report_path, CONFIG,
                         ROOT / "03_no_type_baseline/stage3a_plusplus_nll_curve.csv",
                         ROOT / "07_checkpoints/stage3a_plusplus_training_manifest.json",
                         ROOT / "06_tables/stage3a_final_frozen_main_results.csv",
                         ROOT / "06_tables/stage3a_final_frozen_nontrivial_motion_metrics.csv",
                         ROOT / "06_tables/stage3a_final_vehicle_retention.csv",
                         ROOT / "09_reports/stage3a_final_frozen_decision.json"]
    checked_artifacts += [ROOT / f"04_evaluation/stage3a_plusplus_val_step_{r['global_step']:05d}.json"
                          for r in val_checks]
    result = {
        "status": "PASS", "audit_scope": "Independent CPU-only checkpoint/optimizer/cursor/strict-selection/fresh-actor/table/decision audit",
        "GPU_initialization_or_forward_pass": False, "optimizer_update_performed": False,
        "previous_protected_file_hashes_verified": len(frozen["files"]),
        "registered_training_source_hashes_and_commit_verified": len(prereg["source_sha256"]),
        "scene_shards_rescanned": False, "interaction_density_recalculated": False,
        "source_checkpoint_sha256": sha(source_path), "source_global_step": 18000,
        "source_AdamW_step": 15500, "source_model_state_content_sha256": model_digest(source["state_dict"]),
        "final_best_global_step": best_global, "final_executed_global_step": last_global,
        "executed_extra_optimizer_steps": extra, "checkpoint_audits": cp_audits,
        "fixed_causal_mask_expected_negative_infinity_count": 15,
        "learned_parameters_NaN": 0, "learned_parameters_Inf": 0,
        "final_checkpoint_sha256": sha(selected_path), "final_model_state_content_sha256": selected_model_sha,
        "last_checkpoint_sha256": sha(last_path),
        "strict_selection_recomputed_from_18000_and_every_new_full_VAL": val_checks,
        "curve_vs_VAL_JSON_max_absolute_difference": max(curve_json_deltas),
        "fresh_actor_count": len(actors), "fresh_numeric_values_checked": int(numbers.size),
        "fresh_actor_errors_sha256": sha(actors_path), "actor_aggregation_max_absolute_difference": max(actor_deltas),
        "selected_vs_fresh_full_horizon_absolute_differences": selected_fresh_deltas,
        "motion_group_metrics_independently_recomputed": motion_audits,
        "paired_vehicle_actor_windows": 62981, "future_valid_lengths_and_motion_states_identical": True,
        "maximum_paired_GT_displacement_difference_m": max(displacement_deltas),
        "vehicle_retention_FDE_ratios_independently_recomputed": ratios,
        "converged_by_patience": bad >= 5, "stopped_by_budget": last_global == 21000,
        "stop_reason": expected_reason, "Stage3A_frozen": "YES",
        "Ready_for_Stage3B_Type_Embedding": "YES" if ready else "NO",
        "freeze_by_budget_independent_of_patience": True, "Stage3B_executed": False,
        "artifact_sha256": {str(p.relative_to(ROOT)): sha(p) for p in checked_artifacts},
        "NaN": 0, "Inf": 0,
    }
    temp = OUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, OUT)
    print("INDEPENDENT_FINAL_AUDIT=PASS", "best", best_global, "executed", last_global,
          "AdamW", cp_audits["selected"]["AdamW_step"], cp_audits["last"]["AdamW_step"], flush=True)


if __name__ == "__main__":
    main()
