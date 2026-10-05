"""Conditionally regenerate final predictions for the six prior motion actors."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage3_common import CLASSES, SceneDataset, config, model_new, model_input, errors_with_top1
from stage3a_plus_common import read_json, atomic_json, sha256
from stage3a_plusplus_plot_convergence import selected_fde, selected_step
from stage3a_plus_plot_motion import draw_panel
import numpy as np
import torch
from torch_geometric.data import Batch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

FIELDS = ("minADE6", "minFDE6", "MR6", "Top1ADE6", "Top1FDE6")


def actor_key(row):
    return tuple(row[k] for k in ("scene_token", "sample_token", "instance_token", "horizon"))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=ROOT / "07_checkpoints/stage3a_final_frozen_checkpoint_manifest.json")
    parser.add_argument("--actor-csv", type=Path, default=ROOT / "04_evaluation/stage3a_final_frozen_actor_errors.csv")
    args = parser.parse_args()
    final_manifest = read_json(args.manifest)
    plus_manifest = read_json(ROOT / "07_checkpoints/stage3a_plus_best_checkpoint_manifest.json")
    old_fde = float(plus_manifest["final_best"]["overall_minFDE6"])
    final_fde = selected_fde(final_manifest)
    gain = (old_fde - final_fde) / old_fde
    old_case_manifest = ROOT / "04_evaluation/stage3a_plus_motion_case_manifest.json"
    previous = read_json(old_case_manifest)
    previous_hash = sha256(old_case_manifest)
    assert previous["case_count"] == 6
    old_hashes = {item["source_json"]: item["source_sha256"] for item in previous["figures"]}
    assert all(sha256(ROOT / name) == digest for name, digest in old_hashes.items())
    output_manifest = ROOT / "04_evaluation/stage3a_final_motion_case_manifest.json"
    if gain < .005:
        atomic_json(output_manifest, {
            "status": "RETAINED_PREVIOUS_CASES", "regeneration_threshold_relative_overall_gain": .005,
            "actual_relative_overall_gain": gain, "regenerated": False,
            "source_case_manifest": str(old_case_manifest.relative_to(ROOT)), "source_case_manifest_sha256": previous_hash,
            "case_checkpoint_global_step": 18000, "case_checkpoint_sha256": plus_manifest["final_best"]["sha256"],
            "final_metrics_checkpoint_relative_path": str(args.checkpoint.resolve().relative_to(ROOT)),
            "final_metrics_checkpoint_sha256": sha256(args.checkpoint),
            "reason": "Overall minFDE6 gain below 0.5%; user requires preserving prior case files unchanged.",
            "figures": previous["figures"]})
        print("FINAL_MOTION_CASES=RETAINED", "relative_overall_gain=", gain, flush=True)
        return
    checkpoint = args.checkpoint.resolve()
    checkpoint_sha = sha256(checkpoint)
    actor_sha = sha256(args.actor_csv)
    with args.actor_csv.open(newline="") as f:
        actors = list(csv.DictReader(f))
    assert len(actors) == 85027
    lookup_actor = {actor_key(row): row for row in actors}
    assert len(lookup_actor) == len(actors)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["config"] == config()
    assert int(saved["metadata"]["global_step"]) == selected_step(final_manifest)
    expected_checkpoint_sha = final_manifest.get("checkpoint_sha256", final_manifest.get("sha256"))
    if expected_checkpoint_sha is None and "final_best" in final_manifest:
        expected_checkpoint_sha = final_manifest["final_best"].get("sha256")
    if expected_checkpoint_sha is not None:
        assert checkpoint_sha == expected_checkpoint_sha
    model = model_new()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    ds = SceneDataset("val")
    assert len(ds) == 3603
    graph_lookup = {(row["scene_token"], row["sample_token"]): i for i, row in enumerate(ds.rows)}
    bs = int(config()["batch_size"])
    assert bs == 16
    figures = []
    for item in previous["figures"]:
        cls = item["agent_type"]
        old_source = read_json(ROOT / item["source_json"])
        assert [panel["kind"] for panel in old_source["panels"]] == ["success", "failure"]
        panels = []
        for old_panel in old_source["panels"]:
            row = lookup_actor[actor_key(old_panel)]
            assert row["agent_type"] == cls and row["horizon"] == "full_horizon" and int(row["valid_future_steps"]) == 12
            assert float(row["GT_endpoint_displacement_m"]) > 5
            if cls == "vehicle":
                assert row["motion_state"] == "vehicle.moving"
            index = graph_lookup[row["scene_token"], row["sample_token"]]
            graph = ds[index]
            node = int(row["node_in_graph"])
            assert graph.instance_tokens[node] == row["instance_token"] and graph.future_mask[node].all()
            assert int(graph.agent_type[node]) == CLASSES.index(cls)
            start = index // bs * bs
            graphs = [ds[i] for i in range(start, min(start + bs, len(ds)))]
            data = Batch.from_data_list(graphs).cuda()
            output_node = int(data.ptr[index - start]) + node
            out = model(model_input(data))
            predictions, errors = errors_with_top1(model, out, data)
            trajectory = predictions[output_node].cpu().numpy()
            gt = graph.positions[node, 5:].numpy()
            origin = graph.positions[node, 4].numpy()
            best, top = int(errors["best_mode"][output_node]), int(errors["top1_mode"][output_node])
            best_error = np.linalg.norm(trajectory[best].astype(np.float64) - gt, axis=1)
            top_error = np.linalg.norm(trajectory[top].astype(np.float64) - gt, axis=1)
            actual = dict(zip(("minADE6", "minFDE6", "Top1ADE6", "Top1FDE6"),
                              map(float, (best_error.mean(), best_error[-1], top_error.mean(), top_error[-1]))))
            delta = {key: abs(value - float(row[key])) for key, value in actual.items()}
            assert max(delta.values()) < 1e-4, (cls, old_panel["kind"], delta)
            displacement = float(np.linalg.norm(gt[-1].astype(np.float64) - origin))
            assert abs(displacement - float(row["GT_endpoint_displacement_m"])) < 1e-4 and displacement > 5
            assert graph.positions[node, :5].tolist() == old_panel["history_trajectory_m"]
            assert gt.tolist() == old_panel["GT_trajectory_m"]
            panels.append({**row, "kind": old_panel["kind"],
                "history_trajectory_m": graph.positions[node, :5].tolist(), "history_mask": graph.history_mask[node].tolist(),
                "GT_trajectory_m": gt.tolist(), "future_mask": graph.future_mask[node].tolist(),
                "HiVT_trajectories_m": trajectory.tolist(), "mode_probabilities": out["mode_prob"][output_node].cpu().tolist(),
                "best_FDE_mode_zero_based": best, "top1_mode_zero_based": top,
                "numeric_metrics": {key: float(row[key]) for key in FIELDS}, "metric_absolute_differences_m": delta,
                "lane_positions_m": graph.lane_positions.tolist(), "lane_vectors_m": graph.lane_vectors.tolist(),
                "origin_global_m": graph.origin.tolist(), "ego_yaw_global_rad": float(graph.ego_yaw),
                "coordinate_frame": "t0 ego +x forward, +y left; meters",
                "history_times_seconds": graph.history_times.tolist(), "future_times_seconds": graph.future_times.tolist(),
                "reproduced_VAL_batch_start_index": start, "reproduced_VAL_batch_size": len(graphs),
                "checkpoint_sha256": checkpoint_sha, "GT_endpoint_displacement_recomputed_m": displacement,
                "prior_case_source_json": item["source_json"], "prior_case_source_sha256": item["source_sha256"],
                "actor_identity_reused": True, "prior_kind_label_reused": True})
        name = f"stage3a_final_{cls}_motion_case"
        source_path = ROOT / "04_evaluation/cases" / (name + ".json")
        atomic_json(source_path, {"agent_type": cls, "panels": panels, "checkpoint_sha256": checkpoint_sha,
            "selection": "Reuse the same Stage3A+ lowest/highest-FDE moving actor identities and prior success/failure labels; no new ranking.",
            "source_actor_errors_relative_path": str(args.actor_csv.relative_to(ROOT)), "source_actor_errors_sha256": actor_sha,
            "prior_source_json": item["source_json"], "prior_source_sha256": item["source_sha256"],
            "trajectory_edits": False, "smoothing": False, "interpolation": False})
        source_sha = sha256(source_path)
        fig, axes = plt.subplots(2, 1, figsize=(10, 15))
        fig.subplots_adjust(left=.12, right=.89, bottom=.08, top=.86, hspace=.58)
        fig.suptitle(f"Frozen No-Type | {cls} motion >5 m | same prior case actors", fontsize=14, y=.975)
        audits = [draw_panel(fig, ax, panel) for ax, panel in zip(axes, panels)]
        fig.text(.5, .025, "12 original future observations; best-FDE uses GT; Top-1 uses highest saved probability.",
                 ha="center", fontsize=9, color="#5C646D")
        fig.canvas.draw()
        renderer, canvas = fig.canvas.get_renderer(), fig.bbox
        for audit in audits:
            boxes = [text.get_bbox_patch().get_window_extent(renderer) for text in audit.pop("annotations")]
            objects = audit.pop("occupied")
            occupied = [obj.get_tightbbox(renderer) if hasattr(obj, "get_tightbbox") else obj.get_window_extent(renderer) for obj in objects]
            for i, box in enumerate(boxes):
                assert box.x0 >= 12 and box.y0 >= 12 and box.x1 <= canvas.x1 - 12 and box.y1 <= canvas.y1 - 12
                assert not any(box.overlaps(other) for other in occupied + boxes[:i]), "Endpoint text obscured"
            audit["endpoint_layout_checks"] = "PASS"
        stem = ROOT / "05_figures" / name
        exports = {}
        for ext in ("png", "pdf", "svg"):
            export_path = Path(str(stem) + "." + ext)
            fig.savefig(export_path, dpi=300)
            if ext == "svg":
                export_path.write_text("\n".join(line.rstrip() for line in export_path.read_text().splitlines()) + "\n")
            exports[ext] = {"relative_path": str(export_path.relative_to(ROOT)), "sha256": sha256(export_path)}
        plt.close(fig)
        with Image.open(str(stem) + ".png") as image:
            image.verify()
        assert sha256(source_path) == source_sha
        atomic_json(Path(str(stem) + "_audit.json"), {"status": "PASS", "source_json": str(source_path.relative_to(ROOT)),
            "source_sha256": source_sha, "panels": audits, "actor_identities_changed": False,
            "trajectory_coordinates_edited": False, "smoothing": False, "interpolation": False, "exports": exports})
        figures.append({"name": name, "agent_type": cls, "source_json": str(source_path.relative_to(ROOT)), "source_sha256": source_sha})
        print("FINAL_MOTION_FIGURE=PASS", name, flush=True)
    assert sha256(old_case_manifest) == previous_hash
    assert all(sha256(ROOT / name) == digest for name, digest in old_hashes.items())
    assert sha256(args.actor_csv) == actor_sha and sha256(checkpoint) == checkpoint_sha
    atomic_json(output_manifest, {"status": "PASS", "regenerated": True, "figures": figures, "case_count": 6,
        "regeneration_threshold_relative_overall_gain": .005, "actual_relative_overall_gain": gain,
        "source_checkpoint_relative_path": str(checkpoint.relative_to(ROOT)), "source_checkpoint_sha256": checkpoint_sha,
        "source_actor_csv_sha256": actor_sha, "same_six_prior_actor_identities": True,
        "previous_case_files_unchanged": True})
    print("FINAL_MOTION_CASES=PASS", "relative_overall_gain=", gain, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "path.simplify": False})
    main()
