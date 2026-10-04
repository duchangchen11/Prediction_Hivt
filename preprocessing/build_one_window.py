"""Construct 5 history + 12 future keyframes by exact instance identity."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .common import PROJECT_ROOT, category_mapping, load_config, load_nuscenes, scene_samples, write_json
from .coordinates import (ego_heading_to_global, ego_to_global, global_heading_to_ego,
                          global_to_ego, quaternion_yaw, wrap_angle)


def annotation_index(nusc, sample):
    records = {}
    for token in sample["anns"]:
        ann = nusc.get("sample_annotation", token)
        instance = ann["instance_token"]
        if instance in records:
            raise ValueError("Duplicate instance in a sample")
        if ann["sample_token"] != sample["token"]:
            raise ValueError("Annotation / sample mismatch")
        records[instance] = ann
    return records


def build_window(nusc, scene, t0_index, th=5, tf=12, samples=None):
    samples = samples if samples is not None else scene_samples(nusc, scene)
    if th < 2 or tf < 1 or t0_index < th - 1 or t0_index + tf >= len(samples):
        raise ValueError("Insufficient history/future for requested window")
    frames = samples[t0_index-th+1:t0_index+tf+1]
    indices = [annotation_index(nusc, sample) for sample in frames]
    mapping = category_mapping(nusc)
    current = indices[th-1]
    tokens = sorted(token for token, ann in current.items() if ann["category_name"] in mapping)
    if not tokens:
        raise ValueError("No supported actors at t0")
    n = len(tokens)
    pos = np.zeros((n, th+tf, 2), dtype=np.float64)
    heading = np.zeros((n, th+tf), dtype=np.float64)
    mask = np.zeros((n, th+tf), dtype=bool)
    annotation_tokens = [[""] * (th+tf) for _ in tokens]
    types, categories = [], []
    chain_edges = 0
    for i, token in enumerate(tokens):
        category = current[token]["category_name"]
        types.append(mapping[category])
        categories.append(category)
        previous = None
        for j, index in enumerate(indices):
            ann = index.get(token)
            if ann is None:
                continue
            if ann["category_name"] != category:
                raise ValueError("Instance category changed inside window")
            if previous is not None:
                if previous["next"] != ann["token"] or ann["prev"] != previous["token"]:
                    raise ValueError("Non-continuous annotation instance chain")
                chain_edges += 1
            previous = ann
            pos[i, j] = ann["translation"][:2]
            heading[i, j] = quaternion_yaw(ann["rotation"])
            mask[i, j] = True
            annotation_tokens[i][j] = ann["token"]
    timestamps = np.array([s["timestamp"] for s in frames], dtype=np.int64)
    times = (timestamps - timestamps[th-1]) / 1e6
    ego_records, ego_timestamps = [], []
    for sample in frames:
        sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        if not sd["is_key_frame"] or sd["sample_token"] != sample["token"]:
            raise ValueError("Ego pose is not from the matching LIDAR_TOP keyframe")
        ego_records.append(nusc.get("ego_pose", sd["ego_pose_token"]))
        ego_timestamps.append(sd["timestamp"])
    ego_pos = np.array([r["translation"][:2] for r in ego_records], dtype=np.float64)
    ego_heading = np.array([quaternion_yaw(r["rotation"]) for r in ego_records])
    origin, yaw = ego_pos[th-1], ego_heading[th-1]
    local_pos = np.zeros_like(pos)
    local_heading = np.zeros_like(heading)
    local_pos[mask] = global_to_ego(pos[mask], origin, yaw)
    local_heading[mask] = global_heading_to_ego(heading[mask], yaw)
    local_ego = global_to_ego(ego_pos, origin, yaw)
    local_ego_heading = global_heading_to_ego(ego_heading, yaw)
    # Velocity uses only past/current observations and actual elapsed seconds.
    velocity = np.zeros((n, th, 2), dtype=np.float64)
    velocity_mask = np.zeros((n, th), dtype=bool)
    for i in range(n):
        valid = np.flatnonzero(mask[i, :th])
        for previous, latest in zip(valid[:-1], valid[1:]):
            velocity[i, latest] = (local_pos[i, latest] - local_pos[i, previous]) / (times[latest] - times[previous])
            velocity_mask[i, latest] = True
    # Independently test direction and handedness in addition to round trips.
    forward = origin + np.array([np.cos(yaw), np.sin(yaw)])
    left = origin + np.array([-np.sin(yaw), np.cos(yaw)])
    forward_local = global_to_ego(forward, origin, yaw)
    left_local = global_to_ego(left, origin, yaw)
    audit = {
        "agent_roundtrip_max_error_m": float(np.abs(ego_to_global(local_pos[mask], origin, yaw) - pos[mask]).max()),
        "ego_history_roundtrip_max_error_m": float(np.abs(ego_to_global(local_ego[:th], origin, yaw) - ego_pos[:th]).max()),
        "ego_future_roundtrip_max_error_m": float(np.abs(ego_to_global(local_ego[th:], origin, yaw) - ego_pos[th:]).max()),
        "heading_roundtrip_max_error_rad": float(np.abs(wrap_angle(ego_heading_to_global(local_heading[mask], yaw) - heading[mask])).max()),
        "ego_heading_roundtrip_max_error_rad": float(np.abs(wrap_angle(ego_heading_to_global(local_ego_heading, yaw) - ego_heading)).max()),
        "ego_forward_local": forward_local.tolist(), "ego_left_local": left_local.tolist(),
        "history_duration_s": float(-times[0]), "future_duration_s": float(times[-1]),
        "dt_mean_s": float(np.diff(times).mean()), "dt_min_s": float(np.diff(times).min()),
        "dt_max_s": float(np.diff(times).max()), "instance_chain_edges_checked": chain_edges,
        "all_current_actors_present": bool(mask[:, th-1].all()),
        "lidar_sample_timestamp_max_offset_us": int(np.max(np.abs(np.array(ego_timestamps)-timestamps))),
    }
    if any(audit[k] > 1e-9 for k in audit if "roundtrip" in k):
        raise ValueError("Coordinate round-trip validation failed")
    if not np.allclose(forward_local, [1, 0], atol=1e-9) or not np.allclose(left_local, [0, 1], atol=1e-9):
        raise ValueError("Coordinate orientation validation failed")
    audit["max_observed_displacement_m"] = 0.0
    audit["max_observed_speed_mps"] = 0.0
    for i in range(n):
        valid = np.flatnonzero(mask[i])
        displacement = np.linalg.norm(np.diff(local_pos[i, valid], axis=0), axis=1)
        if len(displacement):
            audit["max_observed_displacement_m"] = max(audit["max_observed_displacement_m"], float(displacement.max()))
            audit["max_observed_speed_mps"] = max(audit["max_observed_speed_mps"], float((displacement/np.diff(times[valid])).max()))
    tensor = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    window = {
        "agent_pos": tensor(local_pos[:, :th]), "agent_velocity": tensor(velocity),
        "agent_heading": tensor(local_heading[:, :th]), "agent_type": torch.tensor(types, dtype=torch.int64),
        "history_mask": torch.from_numpy(mask[:, :th]), "velocity_mask": torch.from_numpy(velocity_mask),
        "future_pos": tensor(local_pos[:, th:]), "future_mask": torch.from_numpy(mask[:, th:]),
        "future_heading": tensor(local_heading[:, th:]),
        "ego_history": tensor(local_ego[:th]), "ego_future": tensor(local_ego[th:]),
        "ego_history_heading": tensor(local_ego_heading[:th]), "ego_future_heading": tensor(local_ego_heading[th:]),
        "history_times": torch.from_numpy(times[:th]), "future_times": torch.from_numpy(times[th:]),
        "sample_timestamps_us": torch.from_numpy(timestamps),
        "ego_pose_timestamps_us": torch.tensor(ego_timestamps, dtype=torch.int64),
        "ego_origin_global": torch.from_numpy(origin.copy()), "ego_yaw_global": torch.tensor(yaw, dtype=torch.float64),
        "scene_token": scene["token"], "scene_name": scene["name"], "sample_token": frames[th-1]["token"],
        "sample_tokens": [s["token"] for s in frames], "instance_tokens": tokens,
        "annotation_tokens": annotation_tokens, "category_names": categories,
        "dataset_version": nusc.version, "t0_index": t0_index, "th": th, "tf": tf,
        "coordinate_frame": "t0 ego planar: +x forward, +y left", "audit": audit,
    }
    # Quantify the loss introduced by float32 storage against original globals.
    stored_positions = np.concatenate([window["agent_pos"].numpy(), window["future_pos"].numpy()], axis=1)
    audit["stored_float32_roundtrip_max_error_m"] = float(np.abs(ego_to_global(stored_positions[mask], origin, yaw)-pos[mask]).max())
    if audit["stored_float32_roundtrip_max_error_m"] > 1e-4:
        raise ValueError("Float32 coordinate storage error exceeds 0.1 mm")
    return window


def tensor_statistics(window):
    stats = {}
    for key, value in window.items():
        if not isinstance(value, torch.Tensor):
            continue
        nan = int(torch.isnan(value).sum()) if value.is_floating_point() else 0
        inf = int(torch.isinf(value).sum()) if value.is_floating_point() else 0
        stats[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                      "min": value.min().item() if value.numel() else None,
                      "max": value.max().item() if value.numel() else None,
                      "nan_count": nan, "inf_count": inf}
        if nan or inf:
            raise ValueError(f"Non-finite values in {key}")
    return stats


def select_debug_window(nusc):
    """Select a readable all-three-type example; this is a diagnostic only."""
    mapping, candidates = category_mapping(nusc), []
    for scene in nusc.scene:
        samples = scene_samples(nusc, scene)
        for t0 in range(4, len(samples)-12):
            annotations = annotation_index(nusc, samples[t0])
            types = {mapping[a["category_name"]] for a in annotations.values() if a["category_name"] in mapping}
            count = sum(a["category_name"] in mapping for a in annotations.values())
            # Prefer coverage, then fewer actors, then a central anchor.
            candidates.append((-len(types), count, abs(t0-len(samples)//2), scene["name"], t0, scene, samples))
    if not candidates:
        raise ValueError("No scene supports a 5+12 window")
    chosen = min(candidates, key=lambda record: record[:5])
    return build_window(nusc, chosen[5], chosen[4], samples=chosen[6])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/debug/one_window.pt"))
    args = parser.parse_args()
    nusc = load_nuscenes(load_config(args.config))
    window = select_debug_window(nusc)
    stats = tensor_statistics(window)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(window, args.output)
    reports = PROJECT_ROOT / "outputs/reports"
    write_json(reports / "tensor_statistics.json", stats)
    write_json(reports / "coordinate_checks.json", window["audit"])
    summary = {key: window[key] for key in ("scene_token", "scene_name", "sample_token", "t0_index", "th", "tf")}
    summary["agent_count"] = len(window["instance_tokens"])
    summary["type_counts"] = {name: int((window["agent_type"] == i).sum()) for i, name in enumerate(("vehicle", "pedestrian", "bicycle"))}
    summary["tensor_statistics"] = stats
    summary["audit"] = window["audit"]
    write_json(reports / "one_window_summary.json", summary)
    for name, record in stats.items():
        print(name, record)
    print(summary["type_counts"])
    print(window["audit"])


if __name__ == "__main__":
    main()
