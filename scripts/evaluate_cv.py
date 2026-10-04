"""Evaluate a small, scene-first diagnostic sample; never generate a full set."""
import argparse
from collections import defaultdict
import json

import torch

from datasets.scene_split import split_scenes, validate_scene_split, window_anchors
from metrics.trajectory import displacement_errors
from models.constant_velocity import constant_velocity
from preprocessing.build_one_window import build_window, tensor_statistics
from preprocessing.common import PROJECT_ROOT, TYPE_NAMES, load_config, load_nuscenes, write_json


def summarize(records):
    result = {}
    for type_index, name in [*enumerate(TYPE_NAMES), (None, "overall")]:
        selected = [record for record in records if type_index is None or record["type"] == type_index]
        eligible = [r for r in selected if r["eligible"]]
        full = [r for r in eligible if r["final_valid"]]
        result[name] = {
            "ADE": sum(r["ade"] for r in eligible)/len(eligible) if eligible else None,
            "FDE": sum(r["fde"] for r in eligible)/len(eligible) if eligible else None,
            "FDE_fixed_horizon": sum(r["fixed_fde"] for r in full)/len(full) if full else None,
            "actor_window_count": len(selected), "evaluated_actor_windows": len(eligible),
            "fixed_horizon_actor_windows": len(full),
            "insufficient_history_count": sum(not r["history_valid"] for r in selected),
            "no_valid_future_count": sum(not r["future_valid"] for r in selected),
            "valid_future_points": sum(r["valid_points"] for r in eligible),
            "mean_last_valid_horizon_s": sum(r["last_horizon_s"] for r in eligible)/len(eligible) if eligible else None,
        }
    return result


def evaluate(nusc, per_scene=3):
    split = split_scenes([s["token"] for s in nusc.scene])
    validate_scene_split(split, [s["token"] for s in nusc.scene])
    write_json(PROJECT_ROOT / "outputs/reports/scene_split.json", {
        "seed": 42, "ratios": [0.6, 0.2, 0.2], "official_benchmark_split": False,
        "purpose": "mini scene-level pipeline smoke test only", "scenes": split,
        "scene_names": {name: [nusc.get("scene", t)["name"] for t in tokens] for name, tokens in split.items()},
        "pairwise_disjoint": True,
    })
    records, window_summaries, by_split = [], [], defaultdict(list)
    for partition, tokens in split.items():
        for scene, anchor, samples in window_anchors(nusc, tokens, max_per_scene=per_scene):
            window = build_window(nusc, scene, anchor, samples=samples)
            assert window["scene_token"] in tokens
            tensor_statistics(window)
            prediction, prediction_mask, velocity = constant_velocity(
                window["agent_pos"], window["history_mask"], window["history_times"], window["future_times"])
            errors = displacement_errors(prediction, window["future_pos"], window["future_mask"], prediction_mask)
            for i, token in enumerate(window["instance_tokens"]):
                eligible = bool(errors["valid"][i])
                record = {"split": partition, "scene_token": scene["token"],
                          "sample_token": window["sample_token"], "instance_token": token,
                          "type": int(window["agent_type"][i]), "eligible": eligible,
                          "history_valid": bool(prediction_mask[i]), "future_valid": bool(window["future_mask"][i].any()),
                          "valid_points": int(errors["valid_points"][i]),
                          "ade": float(errors["ade"][i]) if eligible else None,
                          "fde": float(errors["fde"][i]) if eligible else None,
                          "final_valid": bool(errors["fixed_horizon_valid"][i]),
                          "fixed_fde": float(errors["fixed_horizon_fde"][i]) if bool(errors["fixed_horizon_valid"][i]) else None,
                          "last_horizon_s": float(window["future_times"][errors["fde_index"][i]]) if eligible else None}
                records.append(record)
                by_split[partition].append(record)
            window_summaries.append({"split": partition, "scene_token": scene["token"], "scene_name": scene["name"],
                                     "sample_token": window["sample_token"], "t0_index": anchor,
                                     "actor_count": len(window["instance_tokens"]), "audit": window["audit"]})
    result = {
        "dataset_version": nusc.version, "window_count": len(window_summaries),
        "max_windows_per_scene": per_scene, "th": 5, "tf": 12, "units": "metres",
        "selection": "Up to 3 uniformly spaced eligible anchors per scene, after scene split",
        "purpose": "Pipeline diagnostic; pooled scores are not held-out benchmark results",
        "metric_definition": {"ADE": "Mean valid displacement per actor-window, then equally across eligible actor-windows",
                              "FDE": "Displacement at each actor's last valid future annotation",
                              "FDE_fixed_horizon": "Final requested frame only; exclude actors missing that frame",
                              "history": "Last two valid history observations, actual timestamp interval; fewer than two excluded",
                              "pooling": "Includes train/val/test diagnostic windows; per-split scores below",
                              "overlap": "Anchors within each scene may overlap; they are not independent samples"},
        "metrics": summarize(records), "by_split": {p: summarize(by_split[p]) for p in split},
        "window_summaries": window_summaries,
    }
    reports = PROJECT_ROOT / "outputs/reports"
    write_json(reports / "cv_baseline.json", result)
    write_json(reports / "cv_actor_errors.json", records)
    lines = ["# Constant velocity diagnostic", "", f"{len(window_summaries)} windows; Th=5, Tf=12; actual timestamps; distances in metres.", "",
             "Mini scene split: 6 train / 2 val / 2 test, seed 42. This is a custom smoke-test split.", "",
             "Pooled values include all three partitions and are diagnostic, not benchmark performance. Actor-windows are averaged equally; overlapping windows are not independent.", "",
             "ADE uses valid future points. FDE uses the last valid future point. Fixed-horizon FDE separately requires the final requested frame. Histories with fewer than two observations are excluded.", ""]
    for label, metrics in [("Pooled diagnostic", result["metrics"]), *result["by_split"].items()]:
        lines += [f"## {label}", "", "| Type | ADE (m) | FDE last valid (m) | FDE final frame (m) | Evaluated / current actors |", "|---|---:|---:|---:|---:|"]
        for name, values in metrics.items():
            fmt = lambda value: "N/A" if value is None else f"{value:.6f}"
            lines.append(f"| {name} | {fmt(values['ADE'])} | {fmt(values['FDE'])} | {fmt(values['FDE_fixed_horizon'])} | {values['evaluated_actor_windows']} / {values['actor_window_count']} |")
        lines.append("")
    (reports / "cv_baseline.md").write_text("\n".join(lines).rstrip()+"\n")
    print(json.dumps(result["metrics"], indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-per-scene", type=int, default=3)
    args = parser.parse_args()
    evaluate(load_nuscenes(load_config(args.config)), args.max_per_scene)


if __name__ == "__main__":
    main()
