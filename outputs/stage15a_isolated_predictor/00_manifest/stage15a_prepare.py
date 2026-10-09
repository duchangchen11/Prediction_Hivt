"""Register Stage15A without loading graphs, predictions, or outer labels."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
BASE = "7b14f630fa42a4320214e8c6c63ab8794b48d43b"
BRANCH = "stage15a/isolated-predictor-preflight"
S14 = PROJECT / "outputs/stage14b_paper_validation"
S3 = PROJECT / "outputs/stage3_multitype_hivt"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT, text=True).strip() == BRANCH
    destination = ROOT / "00_manifest/stage15a_protocol.json"
    assert not destination.exists(), "Registration is immutable; use the existing registration on replay"
    design_path = S14 / "00_protocol/stage14b_scene_design.json"
    design = json.loads(design_path.read_text())
    index_path = S3 / "02_preprocessed/stage3_train_index.csv"
    with index_path.open() as f:
        rows = list(csv.DictReader(f))
    train700 = set(design["OfficialTrain700"])
    assert {r["scene_token"] for r in rows} == train700 and len(train700) == 700
    folds = []
    for fold in design["A"]["Folds"]:
        n = fold["OuterFold"]
        s = json.loads((PROJECT / f"outputs/stage11b_error_aware_ranking/02_splits/stage11b_fold{n}_split.json").read_text())
        parts = {"InnerTrain": fold["Train378"], "InnerDev": fold["Development42"],
                 "OuterTest": fold["Evaluation210"], "QuarantinedHeadDev": fold["Quarantined70"]}
        assert [len(v) for v in parts.values()] == [378, 42, 210, 70]
        assert all(parts[k] == s[k] for k in ("InnerTrain", "InnerDev", "OuterTest"))
        assert set.union(*(set(v) for v in parts.values())) == train700
        assert sum(map(len, parts.values())) == len(set.union(*(set(v) for v in parts.values())))
        record = {"fold": n, "seed": 2022 + 100 * (n - 1), "parts": parts, "files": {}, "metadata": {}}
        for role, tokens in parts.items():
            p = ROOT / f"01_data_isolation/stage15a_fold{n}_{role}.json"
            write(p, {"fold": n, "seed": record["seed"], "role": role, "scene_tokens": tokens,
                      "stage14b_design_sha256": sha(design_path), "original_order_retained": True})
            record["files"][role] = {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
            rr = [r for r in rows if r["scene_token"] in set(tokens)]
            record["metadata"][role] = {"scenes": len(tokens), "windows": len(rr),
                "supervised_windows": sum(int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) > 0 for r in rr),
                "full_targets": sum(int(r["full_horizon_target_count"]) for r in rr),
                "partial_targets": sum(int(r["partial_target_count"]) for r in rr)}
        folds.append(record)
    historical = subprocess.check_output(["git", "ls-files", "-z"], cwd=PROJECT).decode().split("\0")
    historical = {p: sha(PROJECT / p) for p in historical if p}
    old = json.loads((S14 / "00_protocol/stage14b_frozen_history.json").read_text())
    checkpoints = dict(old["checkpoints"])
    checkpoints.update(json.loads((S14 / "00_protocol/stage14b_final_audit.json").read_text())["CheckpointSHA256"])
    assert all(sha(PROJECT / p) == h for p, h in checkpoints.items())
    write(ROOT / "00_manifest/stage15a_frozen_history.json", {
        "BaseCommit": BASE, "historical_files": historical,
        "preserved_untracked": old["preserved_untracked"], "checkpoints": checkpoints})
    sources = [design_path, index_path, S14 / "stage14b_preregistered_plan.md",
               PROJECT / "outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_config.yaml"]
    sources += list((PROJECT / "models/hivt_runtime").glob("*.py"))
    sources += [PROJECT / "models/hivt_loss_recovery.py", PROJECT / "models/hivt_nuscenes.py",
                S3 / "00_manifest/stage3b_model.py", S3 / "00_manifest/stage3b_common.py",
                S3 / "00_manifest/stage3_common.py", S3 / "02_preprocessed/stage3_dataset.py",
                PROJECT / "outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_model.py",
                PROJECT / "outputs/stage5a_motion_aware_decoder/00_manifest/stage5a_decoder.py",
                PROJECT / "outputs/stage5a_motion_aware_decoder/03_training/stage5a_train.py"]
    protocol = {
        "Stage": "Stage15A", "BaseCommit": BASE, "Branch": BRANCH,
        "Scheme": "A_Recommended378", "FullTrainingAuthorized": False, "OuterInferenceAuthorized": False,
        "model": {"historical_steps": 5, "future_steps": 12, "num_modes": 6, "embed_dim": 64,
                  "num_heads": 8, "dropout": .1, "num_temporal_layers": 4, "num_global_layers": 3, "local_radius": 50.},
        "parameter_count": 650403, "batch_size": 16, "optimizer": "AdamW", "weight_decay": .0001,
        "warmup": {"phase": "fixed_scale", "steps": 5000, "lr": .001},
        "nll": {"phase": "original_nll", "maximum_steps": 16000, "lr": .0001, "patience": 5},
        "maximum_global_steps": 21000, "validation_interval": 500,
        "selection": "post-update InnerDev42 full-horizon Overall minFDE6 strict improvement; final NLL only",
        "warmup_transition": "own best model/optimizer/CPU+CUDA+Python+NumPy RNG; LR only change; sampler reset; NLL epoch offset100000",
        "normalization": "new per-fold InnerTrain only, population std +1e-6; preflight subset moments never formal",
        "precision": "FP32", "AMP": False, "cpu_threads": 4, "workers": 0, "cache_scenes": 2,
        "preflight": {"warm_updates_per_fold": 4, "nll_updates_per_fold": 2, "train_batch_windows": 16,
                      "dev_subset_windows": 4, "resume_probe_extra_steps_per_fold": 2,
                      "stress_optimizer_updates": 1, "maximum_optimizer_updates_all_checks": 32,
                      "recovery_tolerance": 0., "forward_integrity_tolerance": 1e-6,
                      "checkpoint_scope": "PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING",
                      "formal_validation_schedule_not_accelerated": True},
        "source_sha256": {str(p.relative_to(PROJECT)): sha(p) for p in sources}, "folds": folds,
        "official_VAL_read": False, "HeadDev70_fitted": False, "448_alternative": "FORBIDDEN",
        "after_stage": "commit/push Stage15A, STOP brainAI review; no full3fold training"}
    write(destination, protocol)
    write(ROOT / "01_data_isolation/stage15a_split_integrity.json", {
        "Status": "PASS", "Scheme": protocol["Scheme"], "CountsPerFold": [378, 42, 210, 70],
        "ExactStage14BSplitOrder": True, "All700AccountedPerFold": True,
        "OuterUnionsExactly630": len(set.union(*(set(f["parts"]["OuterTest"]) for f in folds))) == 630,
        "SourceSHA256": {str(p.relative_to(PROJECT)): sha(p) for p in sources[:4]},
        "MetadataOnly": True, "OuterPredictionsRead": False, "OuterPerformanceComputed": False,
        "folds": [{"fold": f["fold"], "seed": f["seed"], "metadata": f["metadata"]} for f in folds]})
    print("STAGE15A_REGISTERED", "folds3", "from_scratch", "no_full_training", flush=True)


if __name__ == "__main__":
    main()
