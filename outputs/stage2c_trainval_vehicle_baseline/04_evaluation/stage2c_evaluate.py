"""Paired official-VAL CV comparison and scene-cluster bootstrap."""
import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "00_manifest"))
from stage2c_common import (GROUPS, HORIZONS, SceneShardDataset, atomic_json, config, evaluate, read_json,
                           model_new, sha256, update_manifest, verify_previous, write_csv)
import numpy as np
import torch


def paired_scene_bootstrap(hivt, cv, replicates=1000, seed=2022):
    scenes = sorted(hivt["scenes"]); assert scenes == sorted(cv["scenes"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), size=(replicates, len(scenes)))
    multiplicities = np.stack([np.bincount(row, minlength=len(scenes)) for row in draws])
    result = {"method": "paired scene-cluster percentile bootstrap", "replicates": replicates, "seed": seed,
              "resampling_unit": "validation scene", "scene_count": len(scenes), "confidence": .95,
              "weighting": "actor-window equal weights inside pooled resampled scenes; repeated scenes contribute their entire clusters",
              "difference": "HiVT - CV; negative values favor HiVT", "groups": {}}
    for group in ("overall", "vehicle.moving"):
        counts = []; sums = {"ADE": [], "FDE": []}
        for scene in scenes:
            h = hivt["scenes"][scene]["full_horizon"][group]; c = cv["scenes"][scene]["full_horizon"][group]
            assert h["count"] == c["count"]; n = h["count"]; counts.append(n)
            for metric in sums: sums[metric].append(n*(h["min"+metric+"6"]-c["min"+metric+"6"]) if n else 0.)
        counts = np.array(counts); denominator = multiplicities @ counts
        assert (denominator > 0).all(), "Bootstrap draw without target actors needs explicit handling"
        result["groups"][group] = {"actor_windows": int(counts.sum())}
        for metric, values in sums.items():
            values = np.array(values); boot = (multiplicities @ values)/denominator
            lo, hi = np.quantile(boot, [.025, .975]); point = values.sum()/counts.sum()
            result["groups"][group]["delta_"+metric] = {"estimate_m": float(point), "CI95_m": [float(lo), float(hi)],
                                                      "bootstrap_std_m": float(boot.std(ddof=1)), "valid_replicates": len(boot)}
    return result


def table_rows(methods, horizons=HORIZONS):
    rows = []
    for method, result in methods.items():
        for horizon in horizons:
            for group, values in result["metrics"][horizon].items():
                rows.append({"Method": method, "Horizon": horizon, "Group": group, **values})
    return rows


def audit_paired_actors(hivt_path, cv_path):
    count = 0
    with open(hivt_path) as h, open(cv_path) as c:
        hi = csv.DictReader(h); ci = csv.DictReader(c)
        for hr, cr in zip(hi, ci, strict=True):
            for key in ("scene_token", "sample_token", "instance_token", "horizon", "motion_state", "valid_future_steps"):
                assert hr[key] == cr[key], (count, key)
            count += 1
    return count


def main():
    c = config(); prep = read_json(ROOT / "02_preprocessed/stage2c_preprocess_manifest.json")
    assert prep["status"] == "COMPLETE" and prep["scope"] == "full official trainval"
    assert read_json(ROOT / "03_training/stage2c_nll_summary.json")["status"] == "COMPLETE"
    checkpoint_path = ROOT / "07_checkpoints/stage2c_best_overall_minfde.pt"
    saved = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    assert saved["metadata"]["phase"] == "original_nll" and saved["metadata"]["selection_metric"] == "overall minFDE6"
    assert saved["metadata"]["config_sha256"] == sha256(ROOT / "00_manifest/stage2c_config.yaml")
    model = model_new(); model.load_state_dict(saved["state_dict"])
    ds = SceneShardDataset("val"); hpath=ROOT / "04_evaluation/stage2c_val_actor_errors.csv"; cpath=ROOT / "04_evaluation/stage2c_cv_actor_errors.csv"
    hivt = evaluate(ds, model, actor_path=hpath, progress=True)
    assert abs(hivt["metrics"]["full_horizon"]["overall"]["minFDE6"]-saved["metadata"]["validation_FDE"]) < 1e-4
    cv = evaluate(ds, actor_path=cpath, progress=True)
    paired = audit_paired_actors(hpath, cpath)
    assert paired == sum(hivt["metrics"][h]["overall"]["count"] for h in HORIZONS)
    for horizon in HORIZONS:
        for group in GROUPS: assert hivt["metrics"][horizon][group]["count"] == cv["metrics"][horizon][group]["count"]
    hivt.update(primary_checkpoint=str(checkpoint_path.relative_to(ROOT)), checkpoint_sha256=sha256(checkpoint_path),
                checkpoint_metadata=saved["metadata"], paired_CV_actor_rows=paired)
    atomic_json(ROOT / "04_evaluation/stage2c_val_metrics.json", hivt)
    atomic_json(ROOT / "04_evaluation/stage2c_cv_metrics.json", cv)
    methods = {"CV": cv, "HiVT": hivt}
    rows = table_rows(methods)
    write_csv(ROOT / "06_tables/stage2c_motion_state_metrics.csv", rows)
    main_rows = [r for r in rows if r["Horizon"] == "full_horizon"]
    for filename in ("stage2c_cv_vs_hivt.csv", "stage2c_main_results.csv"): write_csv(ROOT / "06_tables" / filename, main_rows)
    scene_rows = []
    for method, measured in methods.items():
        for scene, summaries in measured["scenes"].items():
            for horizon, groups in summaries.items():
                for group, values in groups.items(): scene_rows.append({"Method": method, "scene_token": scene, "Horizon": horizon, "Group": group, **values})
    write_csv(ROOT / "04_evaluation/stage2c_val_scene_metrics.csv", scene_rows)
    bootstrap = paired_scene_bootstrap(hivt, cv, **{k: c["bootstrap"][k] for k in ("replicates", "seed")})
    atomic_json(ROOT / "04_evaluation/stage2c_bootstrap_ci.json", bootstrap)
    write_csv(ROOT / "06_tables/stage2c_bootstrap_ci.csv", [{"Group": group, "Metric": metric, "Delta_HiVT_minus_CV_m": values["estimate_m"],
             "CI95_lower_m": values["CI95_m"][0], "CI95_upper_m": values["CI95_m"][1], "Resampling_unit": "scene", "Replicates": bootstrap["replicates"]}
             for group, fields in bootstrap["groups"].items() for metric, values in fields.items() if metric.startswith("delta_")])
    verify_previous(); update_manifest()
    print("FORMAL_VAL", {method: values["metrics"]["full_horizon"] for method,values in methods.items()}, flush=True)
    print("SCENE_BOOTSTRAP", bootstrap, flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4); main()
