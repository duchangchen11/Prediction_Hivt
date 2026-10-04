"""Audit all train actor-windows and select diagnostic targets, never drop context."""
import csv
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from preprocessing.common import PROJECT_ROOT, load_config, load_nuscenes, scene_samples, write_json

AUDIT = PROJECT_ROOT / "outputs/stage2/moving_overfit_audit"
BINS = ((0., 1., "0–1m"), (1., 2., "1–2m"), (2., 5., "2–5m"),
        (5., 10., "5–10m"), (10., 20., "10–20m"), (20., 40., "20–40m"),
        (40., float("inf"), ">40m"))


def motion_stats(graph, node):
    history = graph.positions[node, :5].double()
    current = history[-1]
    future = graph.positions[node, 5:].double()[graph.future_mask[node]]
    h = torch.where(graph.history_mask[node])[0]
    ht = graph.history_times.double()[h]
    hp = history[h]
    history_speed = float(torch.linalg.vector_norm(hp[1:]-hp[:-1], dim=-1).sum() / (ht[-1]-ht[0])) if len(h)>1 else None
    current_speed = float(torch.linalg.vector_norm(hp[-1]-hp[-2])/(ht[-1]-ht[-2])) if len(h)>1 else None
    points = torch.cat([current[None], future])
    time = torch.cat([torch.zeros(1, dtype=torch.float64), graph.future_times.double()[graph.future_mask[node]]])
    velocity = (points[1:]-points[:-1])/(time[1:]-time[:-1])[:, None]
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    acceleration = torch.linalg.vector_norm(velocity[1:]-velocity[:-1], dim=-1) / ((time[2:]-time[:-2])/2) if len(velocity)>1 else torch.zeros(0)
    displacement = torch.linalg.vector_norm(future-current, dim=-1)
    return {"gt_endpoint_displacement_m": float(displacement[-1]) if len(future) else None,
            "gt_mean_future_displacement_m": float(displacement.mean()) if len(future) else None,
            "gt_path_length_m": float(torch.linalg.vector_norm(points[1:]-points[:-1], dim=-1).sum()) if len(future) else None,
            "history_mean_speed_mps": history_speed, "current_past_only_speed_mps": current_speed,
            "future_valid_length": len(future), "history_valid_length": len(h),
            "last_valid_future_s": float(time[-1]) if len(future) else None,
            "future_max_interval_speed_mps": float(speed.max()) if len(speed) else None,
            "future_max_interval_acceleration_mps2": float(acceleration.max()) if len(acceleration) else None}


def bin_counts(rows):
    valid = [r for r in rows if r["gt_endpoint_displacement_m"] is not None]
    result = []
    for low, high, label in BINS:
        # Right-closed bins: [0,1], (1,2], ..., >40. No boundary overlap.
        count = sum((r["gt_endpoint_displacement_m"] >= low if low == 0 else r["gt_endpoint_displacement_m"] > low)
                    and r["gt_endpoint_displacement_m"] <= high for r in valid)
        result.append({"bin": label, "count": count, "fraction_of_valid_future": count/len(valid) if valid else None,
                       "fraction_of_all_actor_windows": count/len(rows) if rows else None})
    return {"denominator_all_actor_windows": len(rows), "denominator_with_future": len(valid),
            "no_future_count": len(rows)-len(valid), "bins": result}


def describe(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "quantiles_0_25_50_75_90_100": np.quantile(values, [0,.25,.5,.75,.9,1]).tolist() if values else []}


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    nusc = load_nuscenes(load_config())
    names = sorted(a["name"] for a in nusc.attribute)
    print("ACTUAL ATTRIBUTE_NAMES", json.dumps(names), flush=True)
    attributes = {a["token"]: a["name"] for a in nusc.attribute}
    graphs = torch.load(PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt", weights_only=False)["train"]
    old_errors = json.loads((PROJECT_ROOT/"outputs/reports/stage2/tiny_actor_errors.json").read_text())
    previous_error = {(r["sample_token"], r["instance_token"]): r["minFDE_K"] for r in old_errors}
    chain = {s["token"]: scene_samples(nusc, s) for s in nusc.scene}
    ann_index = {(a["sample_token"], a["instance_token"]): a for a in nusc.sample_annotation}
    rows = []
    for index, g in enumerate(graphs):
        for node, instance in enumerate(g.instance_tokens):
            ann = ann_index[(g.sample_token, instance)]
            observed = sorted(attributes[token] for token in ann["attribute_tokens"])
            actual_state = next((a.split(".")[-1] for a in observed if a in ("vehicle.moving", "vehicle.stopped", "vehicle.parked")), "unknown")
            rows.append({"graph_index": index, "node": node, "scene_token": g.scene_token, "scene_name": g.scene_name,
                         "sample_token": g.sample_token, "instance_token": instance, "category_name": g.category_names[node],
                         "annotation_token": ann["token"], "attribute_names": observed, "attribute_state": actual_state,
                         "prediction_eligible": bool(g.target_mask[node]), "full_horizon": bool(g.full_horizon_mask[node]),
                         "previous_tiny_minFDE_m": previous_error.get((g.sample_token, instance)), **motion_stats(g, node)})
    full = [r for r in rows if r["full_horizon"]]
    eligible = [r for r in rows if r["prediction_eligible"]]
    cross = Counter((r["attribute_state"], "endpoint>2m" if r["gt_endpoint_displacement_m"] is not None and r["gt_endpoint_displacement_m"]>2 else "endpoint<=2m" if r["gt_endpoint_displacement_m"] is not None else "no_future") for r in rows)
    summary = {"dataset": "nuScenes v1.0-mini", "split": "train only, existing scene split", "train_windows": len(graphs),
               "actual_attribute_names": names, "t0_vehicle_attribute_counts": dict(Counter(a for r in rows for a in r["attribute_names"])),
               "total_vehicle_actor_windows": len(rows), "eligible_prediction_actor_windows": len(eligible),
               "full_horizon_actor_windows": len(full), "attribute_state_counts": dict(Counter(r["attribute_state"] for r in rows)),
               "attribute_vs_motion": [{"attribute": a, "kinematic_group": m, "count": n} for (a,m),n in sorted(cross.items())],
               "kinematic_counts_with_future": dict(Counter("moving_endpoint>2m" if r["gt_endpoint_displacement_m"]>2 else "small_endpoint<=2m" for r in rows if r["gt_endpoint_displacement_m"] is not None)),
               "endpoint_bins_all_actor_windows": bin_counts(rows), "endpoint_bins_eligible_targets": bin_counts(eligible),
               "endpoint_bins_full_horizon": bin_counts(full),
               "definitions": {"state": "t0 real annotation attribute; absent/other is unknown; low speed never implies parked",
                   "endpoint": "distance from t0 to last valid future, approximately 6s only when full 12 steps",
                   "GT_ADE_displacement": "mean norm of valid future minus t0; descriptive motion quantity, not prediction ADE",
                   "path_length": "sum segments from t0 through available future points; gaps bridged and valid length retained",
                   "history_speed": "observed history path length / elapsed time; current speed uses last two past observations",
                   "bins": "[0,1], (1,2], (2,5], (5,10], (10,20], (20,40], (40,inf) meters"},
               "statistics": {key: describe(rows, key) for key in ("gt_endpoint_displacement_m", "gt_mean_future_displacement_m", "gt_path_length_m", "history_mean_speed_mps", "current_past_only_speed_mps", "future_valid_length")}}
    write_json(PROJECT_ROOT/"outputs/reports/vehicle_motion_distribution.json", summary)
    with open(AUDIT/"vehicle_actor_windows.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n"); writer.writeheader()
        writer.writerows({**r, "attribute_names": "|".join(r["attribute_names"])} for r in rows)
    lines = ["# Vehicle motion distribution", "", "Source: nuScenes mini, existing training scenes only. Counts are actor-windows, not unique vehicles.", "",
             f"All current vehicles: {len(rows)}; loss-eligible: {len(eligible)}; full 12-step future: {len(full)}.", "",
             "Actual attribute names: " + ", ".join(names), "", "## Real t0 attribute", "", "| State | Count | Fraction |", "|---|---:|---:|"]
    lines += [f"| {s} | {c} | {c/len(rows):.2%} |" for s,c in sorted(summary["attribute_state_counts"].items())]
    for title, population in [("All actor-windows", rows), ("Loss-eligible targets", eligible), ("Full horizon only", full)]:
        result = bin_counts(population)
        lines += ["", f"## {title}", "", f"Valid future denominator={result['denominator_with_future']}; no future={result['no_future_count']}.", "", "| Endpoint bin | Count | Fraction of valid future |", "|---|---:|---:|"]
        lines += [f"| {b['bin']} | {b['count']} | {b['fraction_of_valid_future']:.2%} |" for b in result["bins"]]
    lines += ["", "## Definitions and limits", "", *[f"- {k}: {v}" for k,v in summary["definitions"].items()], "",
              "Real stopped and parked labels remain separate. Full-horizon bins provide the approximate 6s comparison; partial-horizon endpoints must not be described as 6s displacement.", "",
              "Detailed descriptive speeds, path lengths and valid lengths are in the JSON and vehicle_actor_windows.csv."]
    (PROJECT_ROOT/"outputs/reports/vehicle_motion_distribution.md").write_text("\n".join(lines)+"\n")

    def source_continuity(r):
        samples = chain[r["scene_token"]]
        t0 = next(i for i,s in enumerate(samples) if s["token"] == r["sample_token"])
        tokens = [ann_index.get((s["token"], r["instance_token"])) for s in samples[t0-4:t0+13]]
        return len(tokens) == 17 and all(tokens) and all(a["next"] == b["token"] and b["prev"] == a["token"] for a,b in zip(tokens,tokens[1:]))

    # Smooth, complete trajectories; no coordinate/timestep rescaling.
    candidates = [r for r in full if r["history_valid_length"] == 5 and 20<=r["gt_endpoint_displacement_m"]<=50
                  and r["future_max_interval_speed_mps"]<35 and r["future_max_interval_acceleration_mps2"]<12
                  and r["attribute_state"] == "moving" and source_continuity(r)]
    failed = [r for r in candidates if r["previous_tiny_minFDE_m"] is not None]
    if not failed:
        raise RuntimeError("No prior failing full-history moving actor satisfies selection criteria")
    single = max(failed, key=lambda r:r["previous_tiny_minFDE_m"])
    moving = [single]
    seen = {single["instance_token"]}
    # Prefer different instances. Ordering fixed before training, no outcome-based changes.
    for r in sorted(candidates, key=lambda r:(r["scene_token"], r["instance_token"], r["sample_token"])):
        if r["instance_token"] not in seen:
            moving.append(r); seen.add(r["instance_token"])
        if len(moving) == 16: break
    if len(moving)<16:
        raise RuntimeError(f"Only {len(moving)} distinct eligible moving vehicles; report before relaxing selection")
    stationary = []
    for state in ("stopped", "parked"):
        selected = [r for r in full if r["history_valid_length"] == 5 and r["attribute_state"] == state
                    and r["gt_endpoint_displacement_m"]<=1 and r["gt_path_length_m"]<=2
                    and r["current_past_only_speed_mps"]<=.5 and source_continuity(r)]
        chosen = 0
        for r in sorted(selected, key=lambda r:(r["scene_token"], r["instance_token"], r["sample_token"])):
            if r["instance_token"] in seen: continue
            stationary.append(r); seen.add(r["instance_token"]); chosen += 1
            if chosen == 4: break
        if chosen<4: raise RuntimeError(f"Insufficient distinct {state} actors for balanced set")
    selection = {"selection_rule": "complete 5+12, source prev/next continuous, true moving attribute, endpoint 20–50m, future speed<35m/s and acceleration<12m/s²; distinct instances",
                 "single": single, "moving": moving, "stationary": stationary,
                 "balanced_rule": "first 8 moving + 4 real stopped + 4 real parked, all complete; low displacement is not interpreted as parked"}
    write_json(AUDIT/"target_selection.json", selection)
    print("DISTRIBUTION", json.dumps({k:summary[k] for k in ("total_vehicle_actor_windows", "eligible_prediction_actor_windows", "full_horizon_actor_windows", "attribute_state_counts", "kinematic_counts_with_future")}), flush=True)
    print("SELECTED_SINGLE", json.dumps(single), flush=True)
    print("SELECTION_COUNTS", len(moving), len(stationary), flush=True)


if __name__ == "__main__":
    main()
