"""Independently check completed experiment artifacts, counts and source arrays."""
import csv
from collections import defaultdict
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage2c_common import CONFIG, GROUPS, HORIZONS, atomic_json, config, git, read_json, sha256, update_manifest, verify_previous
import numpy as np
import torch
from PIL import Image


def main():
    prep = read_json(ROOT / "02_preprocessed/stage2c_preprocess_manifest.json")
    split = read_json(ROOT / "01_data_audit/stage2c_nuscenes_split_audit.json")
    assert prep["status"] == "COMPLETE" and not prep["failed_shards"] and prep["NaN"] == prep["Inf"] == 0
    index_totals = {}
    for name in ("train", "val"):
        with open(ROOT / f"02_preprocessed/stage2c_{name}_index.csv") as f: rows = list(csv.DictReader(f))
        shards = {r["file_path"] for r in rows}; scenes = {r["scene_token"] for r in rows}
        assert len(shards) == len(scenes) == split[name + "_scenes"]
        assert all((ROOT / p).is_file() and (ROOT / p).stat().st_size > 0 for p in shards)
        grouped = defaultdict(list)
        for row in rows: grouped[row["file_path"]].append(row)
        for filename, scene_rows in grouped.items():
            saved = torch.load(ROOT / filename, weights_only=False, map_location="cpu")
            assert len(saved["graphs"]) == len(scene_rows)
            for row in scene_rows:
                graph = saved["graphs"][int(row["window_index"])]
                if graph is None:
                    assert int(row["vehicle_count"]) == int(row["full_horizon_target_count"]) == int(row["partial_target_count"]) == 0
                    continue
                assert graph.scene_token == row["scene_token"] and graph.sample_token == row["sample_token"]
                assert graph.num_nodes == int(row["vehicle_count"])
                full = graph.target_mask & graph.future_mask.all(dim=-1)
                partial = graph.target_mask & graph.future_mask.any(dim=-1) & ~full
                assert int(full.sum()) == int(row["full_horizon_target_count"]) and int(partial.sum()) == int(row["partial_target_count"])
                assert all(c.startswith("vehicle.") and c != "vehicle.bicycle" for c in graph.category_names)
                assert all(torch.isfinite(v).all() for _, v in graph if isinstance(v, torch.Tensor) and v.is_floating_point())
            del saved
        counts = {"candidate_windows": len(rows), "full_horizon_targets": sum(int(r["full_horizon_target_count"]) for r in rows),
                  "partial_targets": sum(int(r["partial_target_count"]) for r in rows)}
        counts["windows"] = sum(int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) > 0 for r in rows)
        counts["empty_supervision_windows"] = counts["candidate_windows"] - counts["windows"]
        assert all(prep["splits"][name][k] == v for k, v in counts.items())
        index_totals[name] = {**counts, "scene_tokens": scenes, "shards": len(shards)}
    assert not index_totals["train"]["scene_tokens"] & index_totals["val"]["scene_tokens"]
    curves = {}
    for phase in ("warmup", "nll"):
        summary = read_json(ROOT / f"03_training/stage2c_{phase}_summary.json")
        assert summary["status"] == "COMPLETE" and summary["NaN"] == summary["Inf"] == 0 and not summary["test_used"]
        with open(ROOT / f"03_training/stage2c_{phase}_curve.csv") as f: rows = list(csv.DictReader(f))
        steps = [int(float(r["phase_step"])) for r in rows]
        assert steps == list(range(500, summary["phase_steps"] + 1, 500))
        assert all(math.isfinite(float(v)) for r in rows for v in r.values() if v)
        fde = min(float(r["VAL_overall_FDE"]) for r in rows)
        assert abs(fde - summary["best_overall_FDE"]) < 1e-8
        curves[phase] = {"phase_steps": summary["phase_steps"], "validations": len(rows), "best_overall_FDE": fde}
    assert 2500 <= curves["warmup"]["phase_steps"] <= 10000 and 0 < curves["nll"]["phase_steps"] <= 10000
    checkpoints = read_json(ROOT / "00_manifest/stage2c_checkpoint_manifest.json")["checkpoints"]
    required = {"epoch", "global_step", "phase", "LR", "validation_ADE", "validation_FDE", "validation_MR", "moving_ADE", "moving_FDE", "git_commit_SHA", "config_sha256", "sha256"}
    for p, metadata in checkpoints.items():
        assert required <= metadata.keys() and metadata["git_tracked"] is False
        assert metadata["config_sha256"] == sha256(CONFIG) and metadata["sha256"] == sha256(ROOT / p)
        assert len(metadata["git_commit_SHA"]) == 40
    primary = checkpoints["07_checkpoints/stage2c_best_overall_minfde.pt"]
    assert primary["phase"] == "original_nll" and primary["selection_metric"] == "overall minFDE6"
    assert abs(primary["validation_FDE"] - curves["nll"]["best_overall_FDE"]) < 1e-8
    totals = {}
    for method, filename in (("HiVT", "stage2c_val_metrics.json"), ("CV", "stage2c_cv_metrics.json")):
        result = read_json(ROOT / "04_evaluation" / filename)
        assert result["split"] == "val" and not result["test_used"]
        assert set(result["scenes"]) == index_totals["val"]["scene_tokens"]
        assert result["windows"] == index_totals["val"]["windows"]
        for horizon, count_key in (("full_horizon", "full_horizon_targets"), ("partial_future", "partial_targets")):
            groups = result["metrics"][horizon]
            assert groups["overall"]["count"] == index_totals["val"][count_key]
            assert sum(groups[g]["count"] for g in GROUPS if g != "overall") == groups["overall"]["count"]
            assert all(math.isfinite(v) for values in groups.values() for v in values.values() if v is not None)
        totals[method] = {h: result["metrics"][h]["overall"]["count"] for h in HORIZONS}
    assert totals["HiVT"] == totals["CV"]
    bootstrap = read_json(ROOT / "04_evaluation/stage2c_bootstrap_ci.json")
    assert bootstrap["replicates"] == 1000 and bootstrap["scene_count"] == split["val_scenes"]
    for group in bootstrap["groups"].values():
        for k, v in group.items():
            if k.startswith("delta_"): assert v["valid_replicates"] == 1000 and v["CI95_m"][0] <= v["CI95_m"][1]
    cases = read_json(ROOT / "04_evaluation/stage2c_figure_case_manifest.json")
    assert cases["main_case_found"] and all(v["produced"] >= v["requested"] for v in cases["coverage"].values())
    assert all(c["visible_lane_segments_in_case_viewport"] >= 5 for c in cases["cases"])
    assert cases["checkpoint_sha256"] == primary["sha256"]
    source = read_json(ROOT / "04_evaluation/stage2c_qualitative_main_case.json")
    assert source["primary_checkpoint_sha256"] == primary["sha256"] and source["motion_state"] == "vehicle.moving"
    pred, gt, cv = (np.array(source[k]) for k in ("HiVT_trajectories_m", "GT_trajectory_m", "CV_trajectory_m"))
    prob = np.array(source["mode_probabilities"])
    assert pred.shape == (6, 12, 2) and gt.shape == cv.shape == (12, 2) and prob.shape == (6,)
    assert all(np.isfinite(x).all() for x in (pred, gt, cv, prob)) and np.isclose(prob.sum(), 1, atol=1e-5)
    errors = np.linalg.norm(pred - gt, axis=-1); best = errors[:, -1].argmin()
    assert best == source["best_FDE_mode_zero_based"]
    assert abs(errors[best].mean() - source["minADE6"]) < 1e-4 and abs(errors[best, -1] - source["minFDE6"]) < 1e-4
    cv_errors = np.linalg.norm(cv - gt, axis=-1)
    assert abs(cv_errors.mean() - source["CV_ADE_m"]) < 1e-5 and abs(cv_errors[-1] - source["CV_FDE_m"]) < 1e-5
    history = np.array(source["history_trajectory_m"]); times = np.array(source["history_times_seconds"])
    first, last = gt[2] - history[-1], gt[-1] - gt[-4]
    assert min(np.linalg.norm(first), np.linalg.norm(last)) >= 1
    turn = np.degrees(np.arccos(np.clip(np.dot(first, last) / (np.linalg.norm(first) * np.linalg.norm(last)), -1, 1)))
    relative = gt - history[-1]
    lateral = np.abs(relative[:, 0] * first[1] - relative[:, 1] * first[0]).max() / np.linalg.norm(first)
    assert abs(turn - source["turn_degrees"]) < 1e-3 and turn >= 60 and lateral >= 3
    assert source["minADE6"] <= .75 * source["CV_ADE_m"] and source["minFDE6"] <= .75 * source["CV_FDE_m"]
    observed = np.flatnonzero(source["history_mask"]); previous, current = observed[-2:]
    velocity = (history[current] - history[previous]) / (times[current] - times[previous])
    expected_cv = history[current] + (np.array(source["future_times_seconds"]) - times[current])[:, None] * velocity
    assert np.allclose(cv, expected_cv, rtol=1e-6, atol=1e-4)
    image_sizes = {}
    for p in sorted((ROOT / "05_figures").glob("stage2c_*.png")):
        with Image.open(p) as image:
            image.verify(); image_sizes[p.name] = image.size
    assert len(image_sizes) >= 17
    assert (ROOT / "05_figures/stage2c_qualitative_main_case.svg").stat().st_size > 0
    assert (ROOT / "05_figures/stage2c_qualitative_main_case.pdf").stat().st_size > 0
    verify_previous(); update_manifest()
    manifest = read_json(ROOT / "00_manifest/stage2c_artifact_manifest.json")
    for artifact in manifest["artifacts"]:
        if artifact["relative_path"].endswith((".pt", ".sqlite")): assert not artifact["git_tracked"]
    for name, counts in index_totals.items(): counts["scene_tokens"] = len(counts["scene_tokens"])
    audit = {"status": "PASS", "index_totals": index_totals, "training": curves, "paired_actor_counts": totals,
             "checkpoint_files_verified": len(checkpoints), "primary_checkpoint_sha256": primary["sha256"],
             "bootstrap_replicates": 1000, "images_verified": image_sizes, "main_case_numerical_arrays": "PASS",
             "previous_sources_frozen": True, "test_used": False, "Stage3_executed": False,
             "git_commit_at_audit": git("rev-parse", "HEAD")}
    atomic_json(ROOT / "00_manifest/stage2c_final_audit.json", audit); update_manifest(); print(audit, flush=True)


if __name__ == "__main__": torch.set_num_threads(4); main()
