"""Official splits, streamed trajectory metadata and atomically completed scene shards."""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage2c_common import (PROJECT, CONFIG, atomic_json, config, git, read_json, sha256, update_manifest,
                           verify_previous, write_csv, model_new, SceneShardDataset)
import ijson
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from nuscenes.utils.splits import create_splits_scenes
from preprocessing.common import load_config
from datasets.nuscenes_hivt_vehicle_dataset import NuScenesHiVTVehicleDataset
from preprocessing.extract_lane_polylines import LaneExtractor
from scripts.prepare_hivt_mini import validate_graph

CACHE = ROOT / "02_preprocessed/stage2c_metadata_cache"
DB = CACHE / "stage2c_trajectory_metadata.sqlite"
INDEX_FIELDS = ["split", "scene_name", "scene_token", "sample_token", "window_index", "vehicle_count",
                "full_horizon_target_count", "partial_target_count", "file_path"]
GROUPS = ("vehicle.moving", "vehicle.stopped", "vehicle.parked", "unknown")
BINS = [0., 1., 2., 5., 10., 20., 40., float("inf")]
BIN_NAMES = ["0–1m", "1–2m", "2–5m", "5–10m", "10–20m", "20–40m", ">40m"]


def raw_root():
    p = Path(load_config()["nuscenes_trainval_root"])
    if not (p / "v1.0-trainval/scene.json").exists(): raise FileNotFoundError(f"Mount official trainval data first: {p}")
    return p


def official_split():
    root = raw_root(); scenes = json.loads((root / "v1.0-trainval/scene.json").read_text())
    definitions = create_splits_scenes(); train = set(definitions["train"]); val = set(definitions["val"])
    assert not train & val
    available = {s["name"]: s for s in scenes}
    assert train <= available.keys() and val <= available.keys(), "Official train/val scenes missing locally"
    result = {"source": "nuscenes.utils.splits.create_splits_scenes()", "version": "v1.0-trainval", "root": str(root),
              "train_scenes": len(train), "val_scenes": len(val), "overlap": 0, "test_used": False,
              "scenes": {split: [available[n] for n in sorted(names)] for split, names in (("train", train), ("val", val))}}
    atomic_json(ROOT / "01_data_audit/stage2c_nuscenes_split_audit.json", result)
    (ROOT / "01_data_audit/stage2c_nuscenes_split_audit.md").write_text(
        f"# Official nuScenes split audit\n\ntrain scenes={len(train)}\n\nval scenes={len(val)}\n\noverlap=0\n\n"
        "Names obtained from the installed official devkit. All requested scenes exist locally. Test is not used. No random split.\n")
    print("OFFICIAL_SPLIT", len(train), "train scenes", len(val), "val scenes, overlap=0", flush=True)
    return result


def initialize():
    split = official_split()
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "torch": torch.__version__, "CUDA_runtime": torch.version.cuda,
                   "GPU": torch.cuda.get_device_name(0), "GPU_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "nuscenes_devkit": importlib.metadata.version("nuscenes-devkit"), "pyg": importlib.metadata.version("torch-geometric"),
                   "raw_root": str(raw_root()), "source_mount": "read-only /dev/sda2", "dependencies_changed": False,
                   "CPU_RAM_GiB": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30,
                   "git_base": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current")}
    atomic_json(ROOT / "00_manifest/stage2c_environment.json", environment)
    (ROOT / "00_manifest/stage2c_git_state.txt").write_text(git("status", "--short", "--branch") + "\n" + git("log", "-1", "--oneline") + "\n")
    verify_previous(); update_manifest()


def stream_rows(path):
    with open(path, "rb") as f: yield from ijson.items(f, "item", use_float=True)


def build_metadata():
    """Persist only trajectory-required fields; no sensor files or raw JSON copies."""
    source = raw_root() / "v1.0-trainval"; CACHE.mkdir(parents=True, exist_ok=True)
    tables = ["scene", "sample", "instance", "category", "attribute", "log", "sample_annotation", "sample_data", "ego_pose", "sensor", "calibrated_sensor"]
    signature = {name: {"bytes": (source / f"{name}.json").stat().st_size,
                        "mtime_ns": (source / f"{name}.json").stat().st_mtime_ns} for name in tables}
    cache_manifest = CACHE / "stage2c_metadata_index_manifest.json"
    if DB.exists() and cache_manifest.exists():
        assert read_json(cache_manifest)["source_signature"] == signature, "Source metadata changed"
        print("METADATA_INDEX_RESUME", DB, flush=True); return
    temp = DB.with_name(DB.name + ".tmp")
    if temp.exists(): temp.unlink()  # Only this incomplete generated cache.
    con = sqlite3.connect(temp); con.execute("PRAGMA journal_mode=OFF"); con.execute("PRAGMA synchronous=OFF")
    con.executescript("""
        CREATE TABLE scenes(token TEXT PRIMARY KEY, payload TEXT);
        CREATE TABLE samples(token TEXT PRIMARY KEY, scene TEXT, payload TEXT);
        CREATE TABLE annotations(token TEXT PRIMARY KEY, sample TEXT, payload TEXT);
        CREATE TABLE lidar(token TEXT PRIMARY KEY, sample TEXT UNIQUE, pose TEXT, payload TEXT);
        CREATE TABLE poses(token TEXT PRIMARY KEY, payload TEXT);
    """)
    counts = {}; started = time.monotonic()
    for name, sql in (("scene", "INSERT INTO scenes VALUES (?,?)"), ("sample", "INSERT INTO samples VALUES (?,?,?)")):
        rows = list(stream_rows(source / f"{name}.json")); counts[name] = len(rows)
        con.executemany(sql, [(r["token"], json.dumps(r, separators=(",", ":"))) if name == "scene" else
                            (r["token"], r["scene_token"], json.dumps(r, separators=(",", ":"))) for r in rows]); con.commit()
    cats = {r["token"]: r["name"] for r in stream_rows(source / "category.json")}
    instance_category = {r["token"]: cats[r["category_token"]] for r in stream_rows(source / "instance.json")}
    count = 0; pending = []
    for row in stream_rows(source / "sample_annotation.json"):
        keep = {k: row[k] for k in ("token", "sample_token", "instance_token", "translation", "rotation", "prev", "next", "attribute_tokens")}
        keep["category_name"] = instance_category[row["instance_token"]]
        pending.append((row["token"], row["sample_token"], json.dumps(keep, separators=(",", ":")))); count += 1
        if len(pending) == 4000:
            con.executemany("INSERT INTO annotations VALUES (?,?,?)", pending); con.commit(); pending.clear()
        if count % 200000 == 0: print("METADATA_ANNOTATIONS", count, f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    con.executemany("INSERT INTO annotations VALUES (?,?,?)", pending); con.commit(); counts["sample_annotation"] = count
    sensors = {r["token"]: r["channel"] for r in stream_rows(source / "sensor.json")}
    lidar_calibrations = {r["token"] for r in stream_rows(source / "calibrated_sensor.json") if sensors[r["sensor_token"]] == "LIDAR_TOP"}
    needed_poses = set(); pending = []; scanned = 0; kept = 0
    for row in stream_rows(source / "sample_data.json"):
        scanned += 1
        if not row["is_key_frame"] or row["calibrated_sensor_token"] not in lidar_calibrations: continue
        keep = {k: row[k] for k in ("token", "sample_token", "ego_pose_token", "is_key_frame", "timestamp", "calibrated_sensor_token")}
        pending.append((row["token"], row["sample_token"], row["ego_pose_token"], json.dumps(keep, separators=(",", ":"))))
        needed_poses.add(row["ego_pose_token"]); kept += 1
        if len(pending) == 4000: con.executemany("INSERT INTO lidar VALUES (?,?,?,?)", pending); con.commit(); pending.clear()
    con.executemany("INSERT INTO lidar VALUES (?,?,?,?)", pending); con.commit(); counts["sample_data_scanned"] = scanned; counts["lidar_keyframes"] = kept
    pending = []; scanned = 0; kept = 0
    for row in stream_rows(source / "ego_pose.json"):
        scanned += 1
        if row["token"] not in needed_poses: continue
        keep = {k: row[k] for k in ("token", "translation", "rotation")}
        pending.append((row["token"], json.dumps(keep, separators=(",", ":")))); kept += 1
        if len(pending) == 4000: con.executemany("INSERT INTO poses VALUES (?,?)", pending); con.commit(); pending.clear()
    con.executemany("INSERT INTO poses VALUES (?,?)", pending); con.commit(); counts["ego_pose_scanned"] = scanned; counts["keyframe_poses"] = kept
    assert counts["lidar_keyframes"] == counts["sample"], "Every sample must have exactly one LIDAR_TOP keyframe"
    con.executescript("CREATE INDEX samples_scene ON samples(scene); CREATE INDEX annotations_sample ON annotations(sample);")
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close(); os.replace(temp, DB)
    atomic_json(cache_manifest, {"status": "COMPLETE", "source_signature": signature, "counts": counts,
                               "elapsed_seconds": time.monotonic()-started, "scope": "derived minimal trajectory metadata; no raw sensor content", "schema": 1})
    print("METADATA_INDEX_COMPLETE", counts, flush=True)


class SceneMetadata:
    """A bounded scene facade implements the get interface used by the frozen adapter."""
    def __init__(self, scene, con):
        source = raw_root() / "v1.0-trainval"; self.version = "v1.0-trainval"; self.dataroot = str(raw_root())
        self.category = json.loads((source / "category.json").read_text())
        self.attribute = json.loads((source / "attribute.json").read_text())
        self.log = json.loads((source / "log.json").read_text()); self.scene = [scene]
        self.tables = {"scene": {scene["token"]: scene}, "log": {r["token"]: r for r in self.log},
                       "sample": {}, "sample_annotation": {}, "sample_data": {}, "ego_pose": {}}
        self.tables["sample"] = {t: json.loads(p) for t, p in con.execute("SELECT token,payload FROM samples WHERE scene=?", (scene["token"],))}
        for sample in self.tables["sample"].values(): sample["anns"] = []; sample["data"] = {}
        query = "SELECT a.token,a.payload FROM annotations a JOIN samples s ON a.sample=s.token WHERE s.scene=?"
        for t, payload in con.execute(query, (scene["token"],)):
            ann = json.loads(payload); self.tables["sample_annotation"][t] = ann; self.tables["sample"][ann["sample_token"]]["anns"].append(t)
        query = "SELECT l.token,l.payload,p.token,p.payload FROM lidar l JOIN samples s ON l.sample=s.token JOIN poses p ON l.pose=p.token WHERE s.scene=?"
        for t, payload, pose_token, pose_payload in con.execute(query, (scene["token"],)):
            sd = json.loads(payload); self.tables["sample_data"][t] = sd; self.tables["ego_pose"][pose_token] = json.loads(pose_payload)
            self.tables["sample"][sd["sample_token"]]["data"]["LIDAR_TOP"] = t
        self.sample_annotation = list(self.tables["sample_annotation"].values())
    def get(self, table, token): return self.tables[table][token]


def preprocessing(smoke=False):
    split = read_json(ROOT / "01_data_audit/stage2c_nuscenes_split_audit.json")
    if not smoke:
        assert read_json(ROOT / "00_manifest/stage2c_preprocess_smoke.json")["status"] == "PASS", "Ten-scene smoke required"
    build_metadata(); con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    # Normalize maps below the single Stage 2C root, then reuse exact existing geometry extraction.
    map_root = CACHE / "stage2c_map_view"; (map_root / "maps").mkdir(parents=True, exist_ok=True)
    raw = raw_root(); expansion = raw / "maps/expansion" if (raw / "maps/expansion").is_dir() else raw / "expansion"
    link = map_root / "maps/expansion"
    if not link.exists(): link.symlink_to(expansion, target_is_directory=True)
    lane = LaneExtractor(map_root, 50., 2.)
    source_signature = sha256(CACHE / "stage2c_metadata_index_manifest.json")
    input_signature = sha256(PROJECT / "datasets/nuscenes_hivt_vehicle_dataset.py")
    all_rows = {"train": [], "val": []}; stats = Counter(); bins = Counter(); failed = []
    completed = resumed = 0; started = time.monotonic(); shard_records = []
    for name in ("train", "val"):
        scenes = split["scenes"][name][:5] if smoke else split["scenes"][name]
        folder = ROOT / "02_preprocessed" / name; folder.mkdir(exist_ok=True)
        for scene in scenes:
            path = folder / f"stage2c_{scene['name']}.pt"
            try:
                if path.exists():
                    saved = torch.load(path, weights_only=False, map_location="cpu")
                    assert saved["source_signature"] == source_signature and saved["input_signature"] == input_signature
                    assert saved["scene_token"] == scene["token"]
                    graphs = saved["graphs"]; resumed += 1
                else:
                    nusc = SceneMetadata(scene, con)
                    # The constructor is bypassed only to avoid its old map-cache destination.
                    ds = NuScenesHiVTVehicleDataset.__new__(NuScenesHiVTVehicleDataset)
                    ds.nusc = nusc; ds.scene_tokens = frozenset([scene["token"]]); ds.extractor = lane
                    from preprocessing.common import scene_samples
                    chain = scene_samples(nusc, scene); ds.chains = {scene["token"]: chain}
                    ds.anchors = [(scene["token"], i) for i in range(4, len(chain)-12)]; ds.cache = {}
                    attributes = {a["token"]: a["name"] for a in nusc.attribute}; graphs = []
                    for i in range(len(ds)):
                        g = ds[i]; validate_graph(g)
                        current = {nusc.get("sample_annotation", t)["instance_token"]: nusc.get("sample_annotation", t)
                                   for t in nusc.get("sample", g.sample_token)["anns"]}
                        actual = [[attributes[t] for t in current[token]["attribute_tokens"]] for token in g.instance_tokens]
                        g.t0_attribute_names = actual
                        g.t0_motion_state = [next((a for a in names if a in GROUPS), "unknown") for names in actual]
                        assert g.target_mask.any(), "No eligible targets; must audit explicitly before training"
                        graphs.append(g); ds.cache.clear()
                    saved = {"scene_token": scene["token"], "source_signature": source_signature, "input_signature": input_signature, "graphs": graphs}
                    temp = path.with_name(path.name + ".tmp"); torch.save(saved, temp); os.replace(temp, path)
                for i, g in enumerate(graphs):
                    validate_graph(g)
                    full = g.target_mask & g.future_mask.all(-1); partial = g.target_mask & ~full
                    all_rows[name].append(dict(zip(INDEX_FIELDS, [name, g.scene_name, g.scene_token, g.sample_token, i, g.num_nodes,
                        int(full.sum()), int(partial.sum()), str(path.relative_to(ROOT))])))
                    stats[(name, "windows")] += 1; stats[(name, "vehicle_actor_windows_context_included")] += g.num_nodes
                    for horizon, mask in (("full_horizon", full), ("partial_future", partial)):
                        for node in torch.where(mask)[0].tolist():
                            group = g.t0_motion_state[node]; stats[(name, horizon, group)] += 1
                            last = int(torch.where(g.future_mask[node])[0][-1]); displacement = float(torch.linalg.vector_norm(g.y[node, last]))
                            b = min(int(np.searchsorted(BINS, displacement, side="right"))-1, len(BIN_NAMES)-1)
                            bins[(name, horizon, group, BIN_NAMES[b])] += 1
                completed += 1
                shard_records.append({"split": name, "scene_name": scene["name"], "relative_path": str(path.relative_to(ROOT)), "windows": len(graphs)})
                if completed % 5 == 0: print("SCENE_SHARDS", completed, "resumed", resumed, f"elapsed={time.monotonic()-started:.1f}s", flush=True)
                del graphs, saved
                # Lane discretizations are reused; enclosing patch keys are bounded to one scene.
                lane.patch_cache.clear()
            except Exception as error:
                failed.append({"split": name, "scene_name": scene["name"], "error": repr(error)}); print("SHARD_FAIL", failed[-1], flush=True)
                if smoke: raise
    con.close()
    for name, rows in all_rows.items(): write_csv(ROOT / f"02_preprocessed/stage2c_{name}_index.csv", rows, INDEX_FIELDS)
    motion_rows = [{"split": s, "horizon": h, "motion_state": g, "actor_windows": stats[(s,h,g)]}
                   for s in ("train","val") for h in ("full_horizon","partial_future") for g in GROUPS]
    displacement_rows = [{"split": s, "horizon": h, "motion_state": g, "endpoint_displacement_bin": b, "actor_windows": bins[(s,h,g,b)]}
                         for s in ("train","val") for h in ("full_horizon","partial_future") for g in GROUPS for b in BIN_NAMES]
    write_csv(ROOT / "01_data_audit/stage2c_motion_state_statistics.csv", motion_rows)
    write_csv(ROOT / "01_data_audit/stage2c_vehicle_statistics.csv", displacement_rows)
    summary = {"scope": "10-scene smoke" if smoke else "full official trainval", "status": "FAIL" if failed else "COMPLETE",
               "scene_shards": completed, "resumed_shards": resumed, "failed_shards": failed, "NaN": 0, "Inf": 0,
               "splits": {s: {"scenes": len({r['scene_token'] for r in rows}), "windows": len(rows),
                                "context_vehicle_actor_windows": stats[(s,'vehicle_actor_windows_context_included')],
                                "full_horizon_targets": sum(int(r['full_horizon_target_count']) for r in rows),
                                "partial_targets": sum(int(r['partial_target_count']) for r in rows)} for s,rows in all_rows.items()},
               "shards": shard_records, "source_signature": source_signature, "input_signature": input_signature,
               "elapsed_seconds": time.monotonic()-started, "definition": "unchanged >=2 valid history and >=1 future; real t0 attribute; no parked/stopped removal"}
    atomic_json(ROOT / "02_preprocessed/stage2c_preprocess_manifest.json", summary)
    atomic_json(ROOT / "01_data_audit/stage2c_vehicle_statistics.json", {"summary": summary, "motion_states": motion_rows, "displacements": displacement_rows})
    verify_previous(); update_manifest()
    assert not failed, "Incomplete preprocessing; inspect failed_shards before training"
    if smoke:
        ds = SceneShardDataset("train"); val = SceneShardDataset("val"); model = model_new()
        optimizer = model.optimizer(.001, config()["weight_decay"]); tested = []
        for dataset in (ds, val):
            # One window per scene exercises all 10 scene shards through PyG DataLoader.
            subset = torch.utils.data.Subset(dataset, [indices[0] for indices in dataset.scene_indices.values()])
            for batch in DataLoader(subset, batch_size=4, shuffle=False):
                data = batch.cuda(); optimizer.zero_grad(set_to_none=True); output = model(data)
                loss = model.recovery_loss(output, data, "fixed_scale")["loss"]
                assert torch.isfinite(loss); loss.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                if dataset.split == "train": optimizer.step()
                tested.extend(data.scene_token)
        assert len(set(tested)) == len(split["scenes"]["train"][:5])+len(split["scenes"]["val"][:5])
        atomic_json(ROOT / "00_manifest/stage2c_preprocess_smoke.json", {"status": "PASS", "train_scenes": 5, "val_scenes": 5,
                    "scene_tokens_tested": tested, "NaN": 0, "Inf": 0, "val_optimizer_steps": 0,
                    "model_scope": "temporary smoke model discarded; fresh formal initialization later", "source": "official trainval"})
        print("PREPROCESS_SMOKE=PASS", flush=True); update_manifest()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--stage", choices=("init", "smoke", "full"), required=True); args=p.parse_args()
    torch.set_num_threads(4)
    if args.stage == "init": initialize()
    else: preprocessing(smoke=args.stage == "smoke")
