"""Shared Stage 2C paths, artifact integrity, lazy scene shards and metrics."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, Sampler
from torch_geometric.loader import DataLoader
from models.hivt_loss_recovery import HiVTLossRecovery
from scripts.train_hivt_stage2 import MODEL_KEYS
from metrics.hivt_forecasting import multimodal_errors
from models.constant_velocity import constant_velocity

GROUPS = ("overall", "vehicle.moving", "vehicle.stopped", "vehicle.parked", "unknown")
HORIZONS = ("full_horizon", "partial_future")
CONFIG = ROOT / "00_manifest/stage2c_config.yaml"


def config():
    return yaml.safe_load(CONFIG.read_text())


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_csv(path, rows, fields=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None: fields = list(rows[0])
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "w", newline="") as f:
        w = csv.DictWriter(f, fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    os.replace(temp, path)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=PROJECT).decode().strip()


def seed_all(seed=2022):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def model_new():
    c = config(); seed_all(c["seed"])
    return HiVTLossRecovery(**{k: c[k] for k in MODEL_KEYS}).cuda()


def update_manifest():
    path = ROOT / "00_manifest/stage2c_artifact_manifest.json"
    old = read_json(path) if path.exists() else {"artifacts": []}
    cached = {r["relative_path"]: r for r in old["artifacts"]}
    tracked = set(git("ls-files", "--cached").splitlines())
    cp = ROOT / "00_manifest/stage2c_checkpoint_manifest.json"
    checkpoints = read_json(cp).get("checkpoints", {}) if cp.exists() else {}
    descriptions = {"00_manifest": "configuration, provenance or reproducibility manifest",
                    "01_data_audit": "official train/val vehicle audit",
                    "02_preprocessed": "scene-level graph or preprocessing index",
                    "03_training": "step-based optimization configuration or curve",
                    "04_evaluation": "validation evaluation, paired actor errors or bootstrap",
                    "05_figures": "validation qualitative figure with six predicted modes",
                    "06_tables": "validation comparison table",
                    "07_checkpoints": "local model and optimizer checkpoint",
                    "08_logs": "reproducible execution log",
                    "09_reports": "Stage 2C report or execution documentation"}
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p == path or p.name.endswith((".tmp", ".pyc")) or "__pycache__" in p.parts: continue
        rel = str(p.relative_to(ROOT)); st = p.stat(); previous = cached.get(rel)
        digest = previous["sha256"] if previous and previous.get("mtime_ns") == st.st_mtime_ns and previous["file_size"] == st.st_size else sha256(p)
        row = {"relative_path": rel, "description": descriptions.get(rel.split("/")[0], "Stage 2C artifact"),
               "file_size": st.st_size, "sha256": digest, "git_tracked": str(p.relative_to(PROJECT)) in tracked,
               "created_by_stage": "Stage 2C", "mtime_ns": st.st_mtime_ns}
        if rel in checkpoints: row.update(checkpoint_metadata=checkpoints[rel])
        rows.append(row)
    atomic_json(path, {"stage": "Stage 2C", "root": str(ROOT), "self_hash_excluded": True, "artifacts": rows})


def verify_previous():
    frozen = read_json(ROOT / "00_manifest/stage2c_previous_artifacts_sha256.json")
    assert all(sha256(PROJECT / p) == h for p, h in frozen["files"].items())
    official = read_json(PROJECT / "baselines/hivt_official/SOURCE.json")
    assert all(sha256(PROJECT / "baselines/hivt_official" / p) == h for p, h in official["sha256"].items())
    atomic_json(ROOT / "00_manifest/stage2c_freeze_verification.json",
                {"status": "PASS", "previous_versioned_files": len(frozen["files"]), "official_sources": len(official["sha256"]),
                 "base_commit": frozen["base_commit"], "architecture_and_input_unchanged": True})


class SceneShardDataset(Dataset):
    """At most a configurable number of scenes resident, never all trainval graphs."""
    def __init__(self, split, cache_scenes=2):
        self.split = split; self.cache_limit = cache_scenes; self.cache = OrderedDict()
        with open(ROOT / f"02_preprocessed/stage2c_{split}_index.csv") as f: self.rows = list(csv.DictReader(f))
        self.training_rows = [i for i, r in enumerate(self.rows) if int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) > 0]
        assert len(self.training_rows) == len(self.rows), "Zero-supervision windows require explicit audit before training"
        self.scene_indices = OrderedDict()
        for i, row in enumerate(self.rows): self.scene_indices.setdefault(row["scene_token"], []).append(i)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]; filename = row["file_path"]
        if filename not in self.cache:
            self.cache[filename] = torch.load(ROOT / filename, weights_only=False, map_location="cpu")["graphs"]
            if len(self.cache) > self.cache_limit: self.cache.popitem(last=False)
        self.cache.move_to_end(filename)
        return self.cache[filename][int(row["window_index"])].clone()
    def clear(self): self.cache.clear()


class SceneShuffleSampler(Sampler):
    """Shuffle scenes and windows each epoch while bounding shard I/O and memory."""
    def __init__(self, dataset, seed=2022): self.dataset = dataset; self.seed = seed; self.epoch = 0
    def __len__(self): return len(self.dataset)
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); scenes = list(self.dataset.scene_indices); rng.shuffle(scenes)
        for scene in scenes:
            indices = list(self.dataset.scene_indices[scene]); rng.shuffle(indices); yield from indices


class MetricAccumulator:
    def __init__(self):
        self.sums = {h: {g: np.zeros(4, dtype=np.float64) for g in GROUPS} for h in HORIZONS}
    def add(self, horizon, group, ade, fde, miss):
        for name in ("overall", group): self.sums[horizon][name] += (1, ade, fde, miss)
    def summary(self):
        return {h: {g: {"count": int(s[0]), "minADE6": float(s[1] / s[0]) if s[0] else None,
                         "minFDE6": float(s[2] / s[0]) if s[0] else None, "MR6": float(s[3] / s[0]) if s[0] else None}
                    for g, s in groups.items()} for h, groups in self.sums.items()}


@torch.no_grad()
def evaluate(dataset, model=None, phase="original_nll", actor_path=None, progress=False):
    c = config(); acc = MetricAccumulator(); scene_acc = {}; seen = 0
    if model is not None: model.eval()
    fields = ["scene_name", "scene_token", "sample_token", "instance_token", "node_in_graph", "horizon", "motion_state",
              "valid_future_steps", "GT_endpoint_displacement_m", "minADE6", "minFDE6", "MR6", "independent_minADE6"]
    handle = open(actor_path, "w", newline="") if actor_path else None
    writer = csv.DictWriter(handle, fields, lineterminator="\n") if handle else None
    if writer: writer.writeheader()
    try:
        loader = DataLoader(dataset, batch_size=c["batch_size"], shuffle=False, num_workers=0)
        for batch in loader:
            if model is None:
                preds = []; offset = 0
                for graph in batch.to_data_list():
                    pred, valid, _ = constant_velocity(graph.positions[:, :5], graph.history_mask, graph.history_times, graph.future_times)
                    assert torch.equal(valid & graph.target_mask, graph.target_mask)
                    preds.append(pred[:, None]); offset += graph.num_nodes
                prediction = torch.cat(preds); data = batch
            else:
                data = batch.cuda(); output = model(data); prediction = model.ego_predictions(output, data)
                assert torch.isfinite(prediction).all() and torch.isfinite(output["raw_prediction"]).all()
            errors = multimodal_errors(prediction, data.positions[:, 5:], data.future_mask, data.target_mask)
            for horizon in HORIZONS:
                for node in torch.where(errors[horizon])[0].tolist():
                    graph = int(data.batch[node]); local = node - int(data.ptr[graph]); group = data.t0_motion_state[graph][local]
                    ade = float(errors["minADE_K"][node]); fde = float(errors["minFDE_K"][node]); miss = float(errors["MR_K"][node])
                    assert math.isfinite(ade) and math.isfinite(fde)
                    acc.add(horizon, group, ade, fde, miss)
                    token = data.scene_token[graph]; scene_acc.setdefault(token, MetricAccumulator()).add(horizon, group, ade, fde, miss)
                    if writer:
                        last = int(errors["last_valid_index"][node])
                        writer.writerow(dict(zip(fields, [data.scene_name[graph], token, data.sample_token[graph], data.instance_tokens[graph][local], local,
                            horizon, group, int(errors["valid_steps"][node]), float(torch.linalg.vector_norm(data.y[node, last])),
                            ade, fde, miss, float(errors["independent_minADE_K"][node])])))
            seen += batch.num_graphs
            if progress and seen % 200 < c["batch_size"]: print("VAL_EVALUATE", seen, "/", len(dataset), flush=True)
    finally:
        if handle: handle.close()
        dataset.clear()
    assert seen == len(dataset)
    return {"metrics": acc.summary(), "scenes": {t: a.summary() for t, a in scene_acc.items()}, "windows": seen,
            "metric_definition": "ADE of lowest-FDE mode; independent minimum ADE separately; actor-window equal weights; MR endpoint >2m",
            "K": 1 if model is None else 6, "split": dataset.split, "test_used": False}


if __name__ == "__main__":
    verify_previous(); update_manifest(); print("Stage 2C freeze and artifact manifest refreshed", flush=True)
