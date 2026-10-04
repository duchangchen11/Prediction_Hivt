"""Run with python -m preprocessing.inspect_nuscenes."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

from .common import PROJECT_ROOT, TYPE_NAMES, category_mapping, load_config, load_nuscenes, scene_samples, write_json


def inspect(nusc, reports):
    reports = Path(reports)
    reports.mkdir(parents=True, exist_ok=True)
    scene = nusc.scene[0]
    samples = scene_samples(nusc, scene)
    first = samples[0]
    mapping = category_mapping(nusc)
    summary = {"version": nusc.version, "root": nusc.dataroot,
               "scene_count": len(nusc.scene), "sample_count": len(nusc.sample),
               "annotation_count": len(nusc.sample_annotation),
               "first_scene": {k: scene[k] for k in ("name", "token", "first_sample_token", "last_sample_token")},
               "first_sample_categories": dict(Counter(nusc.get("sample_annotation", t)["category_name"] for t in first["anns"])),
               "taxonomy": {name: TYPE_NAMES[index] for name, index in mapping.items()}}
    # Inspect actual files referenced by mini metadata, without reading raw sensors.
    missing = [sd["filename"] for sd in nusc.sample_data if not (Path(nusc.dataroot) / sd["filename"]).is_file()]
    missing_maps = [m["filename"] for m in nusc.map if not (Path(nusc.dataroot) / m["filename"]).is_file()]
    summary["sensor_file_check"] = {"checked": len(nusc.sample_data), "missing_count": len(missing), "examples": missing[:10]}
    summary["map_file_check"] = {"checked": len(nusc.map), "missing": missing_maps}
    write_json(reports / "dataset_inspection.json", summary)
    text = __import__("json").dumps(summary, indent=2, ensure_ascii=False)
    (reports / "dataset_inspection.txt").write_text(text + "\n")
    (reports / "stage_a_dataset_inspection.txt").write_text(text + "\n")
    counts, instances = Counter(), defaultdict(set)
    for ann in nusc.sample_annotation:
        counts[ann["category_name"]] += 1
        instances[ann["category_name"]].add(ann["instance_token"])
    with open(reports / "category_statistics.csv", "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["category_name", "annotation_count", "instance_count", "prediction_type"])
        for name in sorted(c["name"] for c in nusc.category):
            writer.writerow([name, counts[name], len(instances[name]), TYPE_NAMES[mapping[name]] if name in mapping else "excluded"])
    dts = np.diff([s["timestamp"] for s in samples]) / 1e6
    timing = {"scene_token": scene["token"], "mean_dt_s": float(dts.mean()),
              "min_dt_s": float(dts.min()), "max_dt_s": float(dts.max()),
              "all_scene_chains_valid": True}
    all_dts = []
    for record in nusc.scene:
        chain = scene_samples(nusc, record)
        all_dts.extend(np.diff([s["timestamp"] for s in chain]) / 1e6)
    timing["dataset_mean_dt_s"] = float(np.mean(all_dts))
    lines = ["timestamp_us\tsample_token\tdt_s"]
    for i, sample in enumerate(samples):
        dt = "-" if i == 0 else f"{dts[i-1]:.6f}"
        lines.append(f"{sample['timestamp']}\t{sample['token']}\t{dt}")
    lines.append(str(timing))
    (reports / "sample_timing.txt").write_text("\n".join(lines) + "\n")
    write_json(reports / "sample_timing.json", timing)
    print(text)
    print(timing)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--reports", default=str(PROJECT_ROOT / "outputs/reports"))
    args = parser.parse_args()
    inspect(load_nuscenes(load_config(args.config)), args.reports)


if __name__ == "__main__":
    main()
