"""Stream 5 complete scenes into a metadata-only devkit cache on a 16GB host.

No sensors are copied/read. Source tables are read-only. Cached JSON is a subset,
not a substitute for a full trainval integrity/benchmark audit.
"""
import argparse
import json
from pathlib import Path
import random
import shutil
import time

import ijson
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from preprocessing.common import PROJECT_ROOT, load_config, scene_samples, write_json
from preprocessing.map_paths import map_root_view


def stream_select(source, destination, predicate):
    records = []
    scanned = 0
    started = time.monotonic()
    with open(source, "rb") as handle:
        for record in ijson.items(handle, "item", use_float=True):
            scanned += 1
            if predicate(record):
                records.append(record)
    destination.write_text(json.dumps(records))
    print(f"{source.name}: scanned={scanned}, selected={len(records)}, elapsed={time.monotonic()-started:.1f}s", flush=True)
    return records, {"scanned": scanned, "selected": len(records)}


def audit(root, cache):
    root, cache = Path(root), Path(cache)
    source = root / "v1.0-trainval"
    destination = cache / "v1.0-trainval"
    destination.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((source / "scene.json").read_text())
    chosen = random.Random(42).sample(sorted(scenes, key=lambda s: s["token"]), 5)
    selected_scenes = {s["token"] for s in chosen}
    (destination / "scene.json").write_text(json.dumps(chosen))
    counts = {"scene": {"scanned": len(scenes), "selected": len(chosen)}}
    samples, counts["sample"] = stream_select(source/"sample.json", destination/"sample.json", lambda r: r["scene_token"] in selected_scenes)
    selected_samples = {s["token"] for s in samples}
    annotations, counts["sample_annotation"] = stream_select(source/"sample_annotation.json", destination/"sample_annotation.json", lambda r: r["sample_token"] in selected_samples)
    instances = {a["instance_token"] for a in annotations}
    stream_select(source/"instance.json", destination/"instance.json", lambda r: r["token"] in instances)
    sd, counts["sample_data"] = stream_select(source/"sample_data.json", destination/"sample_data.json", lambda r: r["sample_token"] in selected_samples)
    poses = {s["ego_pose_token"] for s in sd}
    _, counts["ego_pose"] = stream_select(source/"ego_pose.json", destination/"ego_pose.json", lambda r: r["token"] in poses)
    for name in ("category", "attribute", "visibility", "sensor", "calibrated_sensor", "log", "map"):
        shutil.copyfile(source/f"{name}.json", destination/f"{name}.json")
    map_view = map_root_view(root)
    for name, source_path in (("maps", map_view/"maps"),):
        link = cache/name
        if link.is_symlink() and link.resolve() != source_path.resolve():
            link.unlink()  # Only our project-cache symlink, never raw data.
        if not link.exists():
            link.symlink_to(source_path, target_is_directory=True)
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(cache), verbose=False)
    checks = []
    for scene in nusc.scene:
        chain = scene_samples(nusc, scene)
        location = nusc.get("log", scene["log_token"])["location"]
        annotation_edges = 0
        for sample in chain:
            lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", lidar["ego_pose_token"])
            assert np.isfinite(ego["translation"]).all()
            assert np.isfinite(ego["rotation"]).all()
            nusc.get("calibrated_sensor", lidar["calibrated_sensor_token"])
            for token in sample["anns"]:
                ann = nusc.get("sample_annotation", token)
                nusc.get("instance", ann["instance_token"])
                assert ann["sample_token"] == sample["token"]
                for direction, back in (("next", "prev"), ("prev", "next")):
                    if ann[direction]:
                        neighbour = nusc.get("sample_annotation", ann[direction])
                        assert neighbour["instance_token"] == ann["instance_token"]
                        assert neighbour[back] == token
                        annotation_edges += 1
        checks.append({"scene_name": scene["name"], "scene_token": scene["token"], "samples": len(chain),
                       "map_location": location, "annotation_links_checked": annotation_edges, "status": "PASS"})
    maps = []
    for location in ("boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"):
        api = NuScenesMap(dataroot=str(map_view), map_name=location)
        lanes = api.lane[:3]
        tokens = [lane["token"] for lane in lanes]
        polylines = api.discretize_lanes(tokens, resolution_meters=2)
        assert all(np.isfinite(polylines[t]).all() and len(polylines[t]) > 1 for t in tokens)
        maps.append({"location": location, "lane_count": len(api.lane), "connector_count": len(api.lane_connector), "status": "PASS"})
    result = {"TRAJECTORY_METADATA_READY": "YES", "HD_MAP_READY": "YES", "LIDAR_REQUIRED_FOR_STAGE2": "NO",
              "root": str(root), "cache_root": str(cache), "map_view_root": str(map_view), "sampling_seed": 42,
              "scope": "5 complete randomly sampled scenes via streaming source metadata and official NuScenes constructor; all four HD Map APIs",
              "source_scene_count": len(scenes), "table_counts": counts, "scene_checks": checks, "maps": maps}
    write_json(PROJECT_ROOT/"outputs/reports/trainval_readiness.json", result)
    lines = ["# Trainval readiness", "", "TRAJECTORY_METADATA_READY=YES", "HD_MAP_READY=YES", "LIDAR_REQUIRED_FOR_STAGE2=NO", "",
             "Readiness=PASS for the trajectory/map task. No raw LiDAR file was opened or downloaded.", "",
             f"Source: {root}; metadata cache: {cache}.", "",
             "This 16GB host uses streaming JSON to extract five complete scenes (seed 42), then loads them through the official devkit. All source rows in the large tables were scanned; only selected records are materialized. This is a sampled task audit, not a full sensor-data or full trainval integrity claim.", "",
             "| Scene | Samples | Location | Annotation links checked |", "|---|---:|---|---:|"]
    for check in checks:
        lines.append(f"| {check['scene_name']} | {check['samples']} | {check['map_location']} | {check['annotation_links_checked']} |")
    lines += ["", "All four HD maps loaded and sample lane centerlines discretized successfully.", ""]
    (PROJECT_ROOT/"outputs/reports/trainval_readiness.md").write_text("\n".join(lines))
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=load_config()["nuscenes_trainval_root"])
    parser.add_argument("--cache", default=str(PROJECT_ROOT/"outputs/stage2/trainval/cache"))
    args = parser.parse_args()
    audit(args.root, args.cache)
