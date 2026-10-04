"""Audit official three-class trainval windows and create resumable scene shards."""
import csv
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage3_common import (PROJECT, PREVIOUS, CLASSES, HORIZONS, atomic_json, read_json,
                           write_csv, sha256, verify_frozen, update_manifest)
sys.path.insert(0, str(PREVIOUS / "02_preprocessed"))
from stage2c_prepare import SceneMetadata, raw_root
from stage3_dataset import Stage3MultiTypeDataset
from preprocessing.common import category_mapping, scene_samples
from preprocessing.coordinates import ego_to_global, quaternion_yaw, global_heading_to_ego, wrap_angle
from preprocessing.extract_lane_polylines import LaneExtractor
from nuscenes.utils.splits import create_splits_scenes
import numpy as np
import torch

FIELDS = ["split", "scene_name", "scene_token", "sample_token", "window_index", "actor_count",
          "vehicle_count", "pedestrian_count", "bicycle_count", "full_horizon_target_count", "partial_target_count",
          "full_vehicle", "full_pedestrian", "full_bicycle", "file_path", "window_status"]
BINS = [0, 1, 2, 5, 10, 20, 40, float("inf")]
BIN_NAMES = ["0–1m", "1–2m", "2–5m", "5–10m", "10–20m", "20–40m", ">40m"]


def validate(g):
    n = g.num_nodes
    assert g.positions.shape == (n, 17, 2) and g.x.shape == (n, 5, 2) and g.y.shape == (n, 12, 2)
    assert g.agent_type.shape == (n,) and g.agent_type.dtype == torch.long
    assert len(g.instance_tokens) == len(set(g.instance_tokens)) == n
    assert g.history_mask[:, -1].all()
    assert torch.equal(g.padding_mask, ~torch.cat([g.history_mask, g.future_mask], 1))
    assert torch.equal(g.target_mask, (g.history_mask.sum(1) >= 2) & g.future_mask.any(1))
    assert torch.equal(g.full_horizon_mask, g.target_mask & g.future_mask.all(1))
    for name, t in zip(g.category_names, g.agent_type.tolist()):
        assert t == (2 if name == "vehicle.bicycle" else 0 if name.startswith("vehicle.") else 1 if name.startswith("human.pedestrian.") else -1)
    for key, value in g:
        if torch.is_tensor(value) and value.is_floating_point(): assert torch.isfinite(value).all(), key
    if g.lane_actor_index.numel():
        lanes, actors = g.lane_actor_index
        torch.testing.assert_close(g.lane_actor_vectors, g.lane_positions[lanes] - g.positions[actors, 4])
        assert (torch.linalg.vector_norm(g.lane_actor_vectors, dim=1) < 50.0001).all()
    audit = json.loads(g.coordinate_audit_json)
    assert audit["stored_float32_roundtrip_max_error_m"] < 1e-4


def case_source(g, node, nusc, lane):
    positions = g.positions[node].numpy(); mask = np.r_[g.history_mask[node].numpy(), g.future_mask[node].numpy()]
    original_global = np.zeros_like(positions, dtype=np.float64)
    original_heading = np.zeros(17)
    previous = None; chain_edges = 0
    for j, token in enumerate(g.annotation_tokens[node]):
        assert bool(token) == bool(mask[j])
        if not token: continue
        ann = nusc.get("sample_annotation", token)
        assert ann["instance_token"] == g.instance_tokens[node] and ann["category_name"] == g.category_names[node]
        if previous:
            assert previous["next"] == token and ann["prev"] == previous["token"]
            chain_edges += 1
        previous = ann
        original_global[j] = ann["translation"][:2]; original_heading[j] = quaternion_yaw(ann["rotation"])
    restored = ego_to_global(positions[mask], g.origin.numpy(), float(g.ego_yaw))
    coordinate_error = float(np.abs(restored - original_global[mask]).max())
    heading_mask = g.history_mask[node].numpy()
    heading_expected = global_heading_to_ego(original_heading[:5][heading_mask], float(g.ego_yaw))
    heading_error = float(np.abs(wrap_angle(g.agent_heading[node].numpy()[heading_mask] - heading_expected)).max())
    assert coordinate_error < 1e-4 and heading_error < 1e-5
    visible = positions[mask]; lo, hi = visible.min(0) - 10, visible.max(0) + 10
    endpoints = torch.stack([g.lane_positions, g.lane_positions + g.lane_vectors], 1).numpy()
    near = ((endpoints.max(1) >= lo) & (endpoints.min(1) <= hi)).all(1)
    # Compare the saved map segment starts to source API discretizations in global coordinates.
    maximum_lane_error = 0.; checked = 0
    for index in np.flatnonzero(near):
        token = g.lane_tokens[index]
        source_polyline = lane.polyline_cache.get((g.map_location, token))
        if source_polyline is None:
            source_polyline = np.asarray(lane.api(g.map_location).discretize_lanes([token], 2.)[token])[:, :2]
        restored_start = ego_to_global(g.lane_positions[index:index+1].numpy(), g.origin.numpy(), float(g.ego_yaw))[0]
        distance = float(np.linalg.norm(source_polyline - restored_start, axis=1).min())
        maximum_lane_error = max(maximum_lane_error, distance); checked += 1
    assert maximum_lane_error < 1e-4
    return {"scene_name": g.scene_name, "scene_token": g.scene_token, "sample_token": g.sample_token,
            "instance_token": g.instance_tokens[node], "agent_type": CLASSES[int(g.agent_type[node])],
            "category_name": g.category_names[node], "annotation_tokens": g.annotation_tokens[node],
            "history_trajectory_m": positions[:5].tolist(), "GT_trajectory_m": positions[5:].tolist(),
            "history_mask": g.history_mask[node].tolist(), "future_mask": g.future_mask[node].tolist(),
            "history_heading_rad": g.agent_heading[node].tolist(), "history_times_seconds": g.history_times.tolist(),
            "future_times_seconds": g.future_times.tolist(), "global_positions_m": original_global.tolist(),
            "origin_global_m": g.origin.tolist(), "ego_yaw_global_rad": float(g.ego_yaw),
            "ego_history_m": g.ego_history.tolist(), "lane_positions_m": g.lane_positions[near].tolist(),
            "lane_vectors_m": g.lane_vectors[near].tolist(), "coordinate_frame": "t0 ego +x forward, +y left, meters",
            "audit": {"instance_chain_edges_checked": chain_edges, "coordinate_max_error_m": coordinate_error,
                      "heading_max_error_rad": heading_error, "source_map_segments_checked": checked,
                      "map_alignment_max_error_m": maximum_lane_error, "status": "PASS"}}


def main():
    verify_frozen(); started = time.monotonic()
    definitions = create_splits_scenes(); train = set(definitions["train"]); val = set(definitions["val"])
    assert not train & val
    source_scenes = json.loads((raw_root() / "v1.0-trainval/scene.json").read_text())
    by_name = {s["name"]: s for s in source_scenes}
    prior = read_json(PREVIOUS / "01_data_audit/stage2c_nuscenes_split_audit.json")
    assert train == {s["name"] for s in prior["scenes"]["train"]}
    assert val == {s["name"] for s in prior["scenes"]["val"]}
    atomic_json(ROOT / "01_data_audit/stage3_official_split.json",
                {"train_scenes": len(train), "val_scenes": len(val), "overlap": 0, "test_used": False,
                 "source": "nuscenes.utils.splits.create_splits_scenes()", "scenes": {s: sorted(ns) for s, ns in (("train", train), ("val", val))}})
    print("OFFICIAL_SPLIT", len(train), len(val), "overlap=0; test unused", flush=True)
    db = PREVIOUS / "02_preprocessed/stage2c_metadata_cache/stage2c_trajectory_metadata.sqlite"
    source_signature = sha256(PREVIOUS / "02_preprocessed/stage2c_metadata_cache/stage2c_metadata_index_manifest.json")
    input_signature = sha256(ROOT / "02_preprocessed/stage3_dataset.py")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cache = ROOT / "02_preprocessed/stage3_cache"; map_root = cache / "stage3_map_view"
    (map_root / "maps").mkdir(parents=True, exist_ok=True)
    expansion = raw_root() / "maps/expansion" if (raw_root() / "maps/expansion").is_dir() else raw_root() / "expansion"
    link = map_root / "maps/expansion"
    if not link.exists(): link.symlink_to(expansion, target_is_directory=True)
    lane = LaneExtractor(map_root, 50., 2.)
    stats = defaultdict(lambda: np.zeros(4)); context = Counter(); distribution = Counter(); motion = Counter()
    reservoirs = {name: [] for name in CLASSES}; seen = Counter(); rng = random.Random(2022)
    all_rows = {"train": [], "val": []}; shards = []; resumed = 0; vehicle_paired = 0
    for split, names in (("train", train), ("val", val)):
        folder = ROOT / "02_preprocessed" / split; folder.mkdir(exist_ok=True)
        for name in sorted(names):
            scene = by_name[name]; path = folder / f"stage3_{name}.pt"
            nusc = SceneMetadata(scene, con)
            chain = scene_samples(nusc, scene); anchors = [(scene["token"], i) for i in range(4, len(chain)-12)]
            if path.exists():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                assert saved["input_signature"] == input_signature and saved["source_signature"] == source_signature
                assert saved["scene_token"] == scene["token"]
                graphs = saved["graphs"]; resumed += 1
            else:
                ds = Stage3MultiTypeDataset.__new__(Stage3MultiTypeDataset)
                ds.nusc = nusc; ds.scene_tokens = frozenset([scene["token"]]); ds.extractor = lane
                ds.chains = {scene["token"]: chain}; ds.anchors = anchors; ds.cache = {}
                graphs = []
                for i in range(len(ds)):
                    try: graph = ds[i]
                    except ValueError as e:
                        if str(e) != "No supported actors at t0": raise
                        graph = None
                    if graph is not None: validate(graph)
                    graphs.append(graph); ds.cache.clear()
                saved = {"scene_token": scene["token"], "input_signature": input_signature,
                         "source_signature": source_signature, "graphs": graphs,
                         "sample_tokens": [chain[t0]["token"] for _, t0 in anchors]}
                temp = path.with_name(path.name + ".tmp"); torch.save(saved, temp); os.replace(temp, path)
            # Exact original vehicle identities, histories, masks and targets are retained.
            previous_graphs = torch.load(PREVIOUS / f"02_preprocessed/{split}/stage2c_{name}.pt", map_location="cpu", weights_only=False)["graphs"]
            assert len(previous_graphs) == len(graphs)
            for i, g in enumerate(graphs):
                old = previous_graphs[i]
                if g is None:
                    assert old is None
                    values = [split, name, scene["token"], saved["sample_tokens"][i], i, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              str(path.relative_to(ROOT)), "no supported current actors"]
                    all_rows[split].append(dict(zip(FIELDS, values))); continue
                validate(g); vehicle_nodes = torch.where(g.agent_type == 0)[0]
                if old is None:
                    assert len(vehicle_nodes) == 0
                else:
                    assert [g.instance_tokens[j] for j in vehicle_nodes.tolist()] == old.instance_tokens
                    for field in ("positions", "x", "y", "history_mask", "future_mask", "target_mask", "rotate_angles", "agent_heading"):
                        assert torch.equal(getattr(g, field)[vehicle_nodes], getattr(old, field)), (name, i, field)
                    vehicle_paired += int(old.target_mask.sum())
                full = g.full_horizon_mask; partial = g.target_mask & ~full
                counts = [int((g.target_mask & (g.agent_type == t)).sum()) for t in range(3)]
                full_counts = [int((full & (g.agent_type == t)).sum()) for t in range(3)]
                values = [split, name, scene["token"], g.sample_token, i, g.num_nodes, *counts,
                          int(full.sum()), int(partial.sum()), *full_counts, str(path.relative_to(ROOT)),
                          "supervised" if g.target_mask.any() else "zero eligible targets"]
                all_rows[split].append(dict(zip(FIELDS, values)))
                for t, class_name in enumerate(CLASSES):
                    context[split, class_name] += int((g.agent_type == t).sum())
                    for horizon, mask in (("full_horizon", full), ("partial_future", partial)):
                        for node in torch.where(mask & (g.agent_type == t))[0].tolist():
                            hcount = int(g.history_mask[node].sum()); fcount = int(g.future_mask[node].sum())
                            last = int(torch.where(g.future_mask[node])[0][-1])
                            displacement = float(torch.linalg.vector_norm(g.y[node, last]))
                            b = min(int(np.searchsorted(BINS, displacement, side="right"))-1, 6)
                            stats[split, class_name, horizon] += (1, hcount, fcount, displacement)
                            distribution[split, class_name, horizon, BIN_NAMES[b]] += 1
                            motion[split, class_name, horizon, g.t0_motion_state[node]] += 1
                            seen[class_name] += 1; k = seen[class_name]
                            slot = len(reservoirs[class_name]) if k <= 10 else rng.randrange(k)
                            if slot < 10:
                                source = case_source(g, node, nusc, lane)
                                source["split"] = split
                                if k <= 10: reservoirs[class_name].append(source)
                                else: reservoirs[class_name][slot] = source
            shards.append({"split": split, "scene_name": name, "relative_path": str(path.relative_to(ROOT)), "windows": len(graphs)})
            lane.patch_cache.clear()
            if len(shards) % 5 == 0:
                print("SCENE_SHARDS", len(shards), "/", len(train)+len(val), "resumed", resumed,
                      f"elapsed={time.monotonic()-started:.1f}s", flush=True)
            del graphs, saved, previous_graphs, nusc
    con.close()
    rows = []
    for split in all_rows:
        write_csv(ROOT / f"02_preprocessed/stage3_{split}_index.csv", all_rows[split], FIELDS)
        for name in CLASSES:
            total = sum(stats[split, name, h][0] for h in HORIZONS)
            hsum = sum(stats[split, name, h][1] for h in HORIZONS)
            fsum = sum(stats[split, name, h][2] for h in HORIZONS)
            rows.append({"split": split, "class": name, "total_target_actor_windows": int(total),
                         "context_actor_windows": context[split, name],
                         "full_horizon": int(stats[split, name, "full_horizon"][0]),
                         "partial_future": int(stats[split, name, "partial_future"][0]),
                         "history_valid_count_sum": int(hsum), "future_valid_count_sum": int(fsum),
                         "mean_history_valid_length": float(hsum / total), "mean_future_valid_length": float(fsum / total)})
    assert vehicle_paired == 315257 + 62981
    distribution_rows = [{"split": s, "class": c, "horizon": h, "bin": b, "count": distribution[s,c,h,b]}
                         for s in all_rows for c in CLASSES for h in HORIZONS for b in BIN_NAMES]
    motion_rows = [{"split": s, "class": c, "horizon": h, "motion_state": m, "count": n}
                   for (s,c,h,m), n in sorted(motion.items())]
    write_csv(ROOT / "01_data_audit/stage3_class_statistics.csv", rows)
    write_csv(ROOT / "01_data_audit/stage3_displacement_distribution.csv", distribution_rows)
    write_csv(ROOT / "01_data_audit/stage3_motion_distribution.csv", motion_rows)
    ratios = {s: [int(sum(stats[s,c,h][0] for h in HORIZONS)) for c in CLASSES] for s in all_rows}
    atomic_json(ROOT / "01_data_audit/stage3_class_statistics.json",
                {"statistics": rows, "endpoint_displacement_bins": distribution_rows, "class_ratio_counts": ratios,
                 "class_ratio_normalized_to_bicycle": {s: [n/ratios[s][2] for n in ratios[s]] for s in ratios},
                 "oversampling": False, "class_balanced_loss": False, "target_definition": ">=2 observed history, >=1 future; current t0 actors"})
    summary = {"status": "COMPLETE", "scope": "full official trainval", "scene_shards": len(shards), "resumed_shards": resumed,
               "failed_shards": 0, "NaN": 0, "Inf": 0, "test_used": False, "shards": shards,
               "source_signature": source_signature, "input_signature": input_signature,
               "previous_vehicle_actor_windows_exactly_paired": vehicle_paired,
               "splits": {s: {"scenes": len({r['scene_token'] for r in rs}), "candidate_windows": len(rs),
                              "supervised_windows": sum(r['window_status']=='supervised' for r in rs)} for s,rs in all_rows.items()}}
    atomic_json(ROOT / "02_preprocessed/stage3_preprocess_manifest.json", summary)
    for name, cases in reservoirs.items():
        assert len(cases) == 10
        atomic_json(ROOT / f"01_data_audit/stage3_data_{name}_examples.json", {"selection": "seed2022 reservoir sampling of target actor-windows over official train+val", "cases": cases})
    atomic_json(ROOT / "01_data_audit/stage3_coordinate_checks.json",
                {"status": "PASS", "samples_per_class": 10, "checks": {n: [c['audit'] for c in cases] for n,cases in reservoirs.items()},
                 "all_windows_checked": True, "previous_vehicle_inputs_unchanged": True})
    text = "# Stage3A official multi-type data audit\n\n700 train / 150 val scenes; overlap0; test unused. Natural class distribution; no oversampling or class-balanced loss.\n\n"
    for split in ratios:
        text += f"{split}: vehicle/pedestrian/bicycle target actor-windows={ratios[split]}.\n\n"
    text += f"Scene shards={len(shards)}; failed0; NaN0; Inf0. All anchors retained, zero-supervision anchors indexed separately. Exact old vehicle input/target pairing={vehicle_paired}.\n\n"
    text += "All generated windows use the existing instance-chain and ego-coordinate checks. Ten random examples per class also compare stored positions/headings/map coordinates directly against source annotations and map discretizations.\n"
    (ROOT / "01_data_audit/stage3_dataset_audit.md").write_text(text)
    verify_frozen(); update_manifest()
    print("CLASS_TARGET_COUNTS", ratios, flush=True)
    print("DATA_AUDIT=PASS", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    try: main()
    except Exception as error:
        atomic_json(ROOT / "01_data_audit/stage3_preprocessing_failure.json", {"status": "FAIL", "error": repr(error), "training_started": False})
        raise
