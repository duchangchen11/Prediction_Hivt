"""Fresh official-VAL inference and actor-paired final No-Type baseline tables.

This entry point never trains, preprocesses or rewrites earlier stage artifacts.
The large actor CSV is a local-only output; small tables carry its SHA256.
"""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage3_common import (CONFIG, PREVIOUS, GROUPS, HORIZONS, SceneDataset,
                           atomic_json, config, evaluate, model_new, read_json,
                           sha256, write_csv)
import numpy as np
import torch

MANIFEST = ROOT / "07_checkpoints/stage3a_final_frozen_checkpoint_manifest.json"
ACTORS = ROOT / "04_evaluation/stage3a_final_frozen_actor_errors.csv"
METRICS_PATH = ROOT / "04_evaluation/stage3a_final_frozen_metrics.json"
FIELDS = ("minADE6", "minFDE6", "MR6", "Top1ADE6", "Top1FDE6", "NLL")
MOTION_FIELDS = FIELDS[:-1]
LABELS = {"overall": "Overall", "vehicle": "Vehicle",
          "pedestrian": "Pedestrian", "bicycle": "Bicycle"}
EXPECTED_FULL = {"overall": 54990, "vehicle": 42332, "pedestrian": 12002,
                 "bicycle": 656, "vehicle.moving": 10461,
                 "vehicle.stopped": 5599, "vehicle.parked": 25198, "unknown": 1074}


def actor_key(row):
    return tuple(row[k] for k in ("scene_token", "sample_token", "instance_token", "horizon"))


def checkpoint_from_manifest(manifest):
    name = manifest.get("checkpoint_relative_path")
    if name is None:
        name = manifest["final_best"]["relative_path"]
    checkpoint = ROOT / name
    assert checkpoint.resolve().is_relative_to(ROOT.resolve())
    assert checkpoint.is_file()
    assert sha256(checkpoint) == manifest["checkpoint_sha256"]
    return checkpoint


def read_actors(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    numeric = FIELDS + ("independent_minADE6", "GT_endpoint_displacement_m")
    numbers = np.array([[float(r[k]) for k in numeric] for r in rows], dtype=np.float64)
    nan = int(np.isnan(numbers).sum()); inf = int(np.isinf(numbers).sum())
    assert nan == inf == 0
    assert len(rows) == 85027
    assert len({actor_key(r) for r in rows}) == len(rows)
    assert {r["horizon"] for r in rows} == set(HORIZONS)
    for row in rows:
        valid = int(row["valid_future_steps"])
        assert valid == 12 if row["horizon"] == "full_horizon" else 1 <= valid < 12
        assert float(row["MR6"]) == float(float(row["minFDE6"]) > 2)
        assert 0 <= int(row["best_mode"]) < 6 and 0 <= int(row["top1_mode"]) < 6
    return rows, {"NaN": nan, "Inf": inf,
                  "checked_actor_values": int(numbers.size), "checked_fields": list(numeric)}


def group_rows(rows, horizon, group):
    return [r for r in rows if r["horizon"] == horizon
            and (group == "overall" or r["agent_type"] == group
                 or (r["agent_type"] == "vehicle" and r["motion_state"] == group))]


def actor_summary(rows, fields=FIELDS):
    assert rows
    values = np.array([[float(r[k]) for k in fields] for r in rows], dtype=np.float64)
    assert np.isfinite(values).all()
    return {"count": len(rows), **dict(zip(fields, map(float, values.mean(axis=0))))}


def verify_actor_aggregation(rows, measured):
    differences = {}
    for horizon in HORIZONS:
        for group in GROUPS:
            selected = group_rows(rows, horizon, group)
            actual = actor_summary(selected)
            saved = measured["metrics"][horizon][group]
            assert actual["count"] == saved["count"]
            if horizon == "full_horizon":
                assert actual["count"] == EXPECTED_FULL[group]
            delta = {k: abs(actual[k] - saved[k]) for k in FIELDS}
            assert max(delta.values()) <= 1e-10, (horizon, group, delta)
            differences[horizon + "/" + group] = delta
    return differences


def save_main_and_motion(rows, measured, checkpoint):
    full = measured["metrics"]["full_horizon"]
    main = [{"Group": LABELS.get(group, group), "Count": values["count"],
             **{key: values[key] for key in FIELDS}} for group, values in full.items()]
    write_csv(ROOT / "06_tables/stage3a_final_frozen_main_results.csv", main)
    motion = []
    for cls, thresholds in (("vehicle", (5,)), ("pedestrian", (1, 5)), ("bicycle", (1, 5))):
        for threshold in thresholds:
            selected = [r for r in rows if r["horizon"] == "full_horizon"
                        and r["agent_type"] == cls and float(r["GT_endpoint_displacement_m"]) > threshold]
            actual = actor_summary(selected, MOTION_FIELDS)
            motion.append({"Group": cls.capitalize() + f" >{threshold}m", "AgentType": cls,
                           "GT_endpoint_displacement_gt_m": threshold,
                           "Count": actual.pop("count"),
                           "UniqueInstances": len({r["instance_token"] for r in selected}),
                           "UniqueScenes": len({r["scene_token"] for r in selected}), **actual})
    expected_motion_counts = (9744, 8810, 7581, 155, 132)
    assert tuple(r["Count"] for r in motion) == expected_motion_counts
    write_csv(ROOT / "06_tables/stage3a_final_frozen_nontrivial_motion_metrics.csv", motion)
    audit = {"status": "PASS", "checkpoint_relative_path": str(checkpoint.relative_to(ROOT)),
             "checkpoint_sha256": sha256(checkpoint), "source_actor_errors_relative_path": str(ACTORS.relative_to(ROOT)),
             "source_actor_errors_sha256": sha256(ACTORS), "fresh_full_VAL_inference": True,
             "subgroups_recomputed_from_fresh_final_predictions": True, "metrics": motion,
             "split": "official val150", "horizon": "full 12-step future only",
             "GT_displacement": "norm(GT[t12]-position[t0]); strict > threshold",
             "metric_definition": measured["metric_definition"], "full_horizon_target_count": 54990,
             "excluded_partial_future_count": 30037, "count_unit": "actor-window",
             "overlapping_windows_are_not_independent_replicates": True, "test_used": False}
    atomic_json(ROOT / "04_evaluation/stage3a_final_frozen_nontrivial_motion_metrics.json", audit)
    return motion


def save_retention(rows, measured, checkpoint):
    old_path = PREVIOUS / "04_evaluation/stage2c_val_actor_errors.csv"
    with old_path.open() as handle:
        old_rows = list(csv.DictReader(handle))
    vehicles = {actor_key(r): r for r in rows if r["agent_type"] == "vehicle"}
    old_by_key = {actor_key(r): r for r in old_rows}
    assert len(old_rows) == len(old_by_key) == len(vehicles) == 62981
    assert vehicles.keys() == old_by_key.keys()
    max_displacement_delta = 0.
    for key, old in old_by_key.items():
        new = vehicles[key]
        assert old["motion_state"] == new["motion_state"]
        assert old["valid_future_steps"] == new["valid_future_steps"]
        delta = abs(float(old["GT_endpoint_displacement_m"]) - float(new["GT_endpoint_displacement_m"]))
        max_displacement_delta = max(max_displacement_delta, delta)
    assert max_displacement_delta <= 1e-6
    reference = read_json(PREVIOUS / "04_evaluation/stage2c_val_metrics.json")
    results = []
    for group, old_group in (("vehicle", "overall"), ("vehicle.moving", "vehicle.moving"),
                             ("vehicle.stopped", "vehicle.stopped"), ("vehicle.parked", "vehicle.parked"),
                             ("unknown", "unknown")):
        selected_old = [r for r in old_rows if r["horizon"] == "full_horizon"
                        and (group == "vehicle" or r["motion_state"] == group)]
        previous = actor_summary(selected_old, FIELDS[:3])
        current = actor_summary(group_rows(rows, "full_horizon", group), FIELDS[:3])
        saved_old = reference["metrics"]["full_horizon"][old_group]
        assert previous["count"] == current["count"] == saved_old["count"]
        for key in FIELDS[:3]:
            assert abs(previous[key] - saved_old[key]) <= 1e-10
            assert abs(current[key] - measured["metrics"]["full_horizon"][group][key]) <= 1e-10
        ratio = current["minFDE6"] / previous["minFDE6"]
        results.append({"Group": "Vehicle overall" if group == "vehicle" else group,
                        "Count": previous["count"],
                        **{"Stage2C_" + k: previous[k] for k in FIELDS[:3]},
                        **{"Stage3A_" + k: current[k] for k in FIELDS[:3]},
                        "Stage3A_over_Stage2C_FDE_ratio": ratio,
                        "FDE_relative_change_percent": 100 * (ratio - 1)})
    write_csv(ROOT / "06_tables/stage3a_final_vehicle_retention.csv", results)
    ratios = {"vehicle_overall_FDE": results[0]["Stage3A_over_Stage2C_FDE_ratio"],
              "vehicle_moving_FDE": results[1]["Stage3A_over_Stage2C_FDE_ratio"]}
    c = config()["readiness_judgement"]
    passed = (ratios["vehicle_overall_FDE"] <= c["maximum_vehicle_overall_FDE_ratio_to_Stage2C"]
              and ratios["vehicle_moving_FDE"] <= c["maximum_vehicle_moving_FDE_ratio_to_Stage2C"])
    audit = {"status": "PASS", "exactly_paired_vehicle_actor_windows": len(old_rows),
             "full_horizon_vehicle_actor_windows": 42332,
             "actor_identity_horizon_masks_motion_states_identical": True,
             "maximum_GT_displacement_difference_m": max_displacement_delta,
             "Stage2C_actor_errors_sha256": sha256(old_path),
             "Stage3A_final_actor_errors_sha256": sha256(ACTORS),
             "reference_checkpoint_sha256": reference["checkpoint_sha256"],
             "final_checkpoint_sha256": sha256(checkpoint), "metrics": results,
             "vehicle_retention_FDE_ratios": ratios,
             "existing_vehicle_collapse_guard": "overall and moving FDE ratio <=1.25; does not select checkpoints",
             "vehicle_no_abnormal_collapse": passed, "test_used": False}
    atomic_json(ROOT / "04_evaluation/stage3a_final_vehicle_retention_audit.json", audit)
    return audit


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(); torch.set_num_threads(args.threads)
    manifest = read_json(MANIFEST); checkpoint = checkpoint_from_manifest(manifest)
    assert manifest["final_best_global_step"] >= 18000
    assert manifest["final_best_global_step"] <= manifest["final_executed_global_step"] <= 21000
    assert sha256(CONFIG) == manifest["config_sha256"]
    interaction_paths = [ROOT / "01_data_audit/stage3_interaction_density.json",
                         ROOT / "01_data_audit/stage3_interaction_density.csv"]
    interaction_before = {str(p.relative_to(ROOT)): sha256(p) for p in interaction_paths}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["config"] == config()
    assert saved["metadata"]["global_step"] == manifest["final_best_global_step"]
    assert saved["metadata"]["phase"] == "original_nll"
    assert saved["metadata"]["selection_metric"] == "overall minFDE6"
    model = model_new()
    parameter_names = set(dict(model.named_parameters()))
    constructed = model.state_dict(); structural_negative_infinity = {}
    for name, tensor in saved["state_dict"].items():
        if not (tensor.is_floating_point() or tensor.is_complex()): continue
        if name in parameter_names:
            assert torch.isfinite(tensor).all(), "Nonfinite learned parameter: " + name
        elif not torch.isfinite(tensor).all():
            # The official causal attention mask contains fixed -Inf entries by design.
            # They must match the unchanged model constructor exactly; they are not predictions.
            assert name == "local_encoder.temporal_encoder.attn_mask", name
            assert not torch.isnan(tensor).any() and not torch.isposinf(tensor).any()
            assert torch.equal(tensor.cpu(), constructed[name].cpu())
            structural_negative_infinity[name] = int(torch.isneginf(tensor).sum())
    model.load_state_dict(saved["state_dict"], strict=True); model.eval()
    # A new inference over all VAL windows is mandatory, even when step 18000 remains best.
    measured = evaluate(SceneDataset("val"), model, actor_path=ACTORS, progress=True)
    assert measured["split"] == "val" and not measured["test_used"]
    assert (measured["windows"], measured["candidate_windows"], measured["empty_supervision_windows"]) == (3603, 3619, 16)
    assert len(measured["scenes"]) == 150
    full = measured["metrics"]["full_horizon"]
    expected = {"overall_minADE6": full["overall"]["minADE6"],
                "overall_minFDE6": full["overall"]["minFDE6"], "overall_MR6": full["overall"]["MR6"],
                "vehicle_minFDE6": full["vehicle"]["minFDE6"],
                "pedestrian_minFDE6": full["pedestrian"]["minFDE6"],
                "bicycle_minFDE6": full["bicycle"]["minFDE6"],
                "vehicle_moving_minFDE6": full["vehicle.moving"]["minFDE6"]}
    selected_delta = {k: abs(v - manifest[k]) for k, v in expected.items()}
    for group, values in full.items():
        selected = manifest["selected_full_horizon_metrics"][group]
        assert values["count"] == selected["count"]
        for key in FIELDS:
            selected_delta[group + "/" + key] = abs(values[key] - selected[key])
    assert max(selected_delta.values()) <= 1e-6, selected_delta
    actors, finite_audit = read_actors(ACTORS)
    differences = verify_actor_aggregation(actors, measured)
    measured.update(status="PASS", checkpoint_relative_path=str(checkpoint.relative_to(ROOT)),
                    checkpoint_sha256=sha256(checkpoint), checkpoint_metadata=saved["metadata"],
                    final_best_global_step=manifest["final_best_global_step"],
                    fresh_full_official_VAL_inference=True, old_actor_CSV_concatenated=False,
                    actor_errors_relative_path=str(ACTORS.relative_to(ROOT)), actor_errors_sha256=sha256(ACTORS),
                    actor_count=len(actors), NaN=finite_audit["NaN"], Inf=finite_audit["Inf"],
                    selected_validation_absolute_differences=selected_delta,
                    independently_recomputed_actor_aggregation_absolute_differences=differences,
                    inference_batch_size=config()["batch_size"], inference_shuffle=False, type_embedding=False)
    atomic_json(METRICS_PATH, measured)
    motion = save_main_and_motion(actors, measured, checkpoint)
    retention = save_retention(actors, measured, checkpoint)
    assert sha256(checkpoint) == manifest["checkpoint_sha256"]
    assert sha256(CONFIG) == manifest["config_sha256"]
    assert interaction_before == {str(p.relative_to(ROOT)): sha256(p) for p in interaction_paths}
    atomic_json(ROOT / "04_evaluation/stage3a_final_frozen_evaluation_audit.json", {
        "status": "PASS", "checkpoint_sha256": sha256(checkpoint), "checkpoint_valid": True,
        "official_VAL_complete": True, "fresh_inference": True, "old_actor_CSV_concatenated": False,
        "official_VAL_scenes": 150, "supervised_windows": 3603, "actor_windows": len(actors),
        "full_horizon_actor_windows": 54990, "partial_horizon_actor_windows": 30037,
        "NaN": finite_audit["NaN"], "Inf": finite_audit["Inf"], "finite_actor_audit": finite_audit,
        "all_learned_model_parameters_finite": True,
        "fixed_causal_mask_negative_infinity_entries_match_constructor": structural_negative_infinity,
        "selected_validation_absolute_differences": selected_delta,
        "actor_errors_sha256": sha256(ACTORS), "motion_groups": 5,
        "motion_group_counts": {r["Group"]: r["Count"] for r in motion},
        "vehicle_no_abnormal_collapse": retention["vehicle_no_abnormal_collapse"],
        "vehicle_retention_FDE_ratios": retention["vehicle_retention_FDE_ratios"],
        "interaction_files_reused_without_scan": interaction_before,
        "no_training": True, "no_preprocessing": True, "test_used": False, "Stage3B_executed": False})
    print("FINAL_FROZEN_VAL_EVALUATION=PASS", expected, flush=True)
    for row in motion: print("FINAL_NONTRIVIAL_MOTION", row, flush=True)
    print("VEHICLE_RETENTION", retention["vehicle_retention_FDE_ratios"], flush=True)


if __name__ == "__main__":
    main()
