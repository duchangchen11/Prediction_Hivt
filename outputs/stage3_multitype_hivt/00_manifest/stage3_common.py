"""Stage3 paths, frozen No-Type model, bounded shards and class diagnostics."""
import csv
from collections import OrderedDict
import json
import os
import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
PREVIOUS = PROJECT / "outputs/stage2c_trainval_vehicle_baseline"
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PREVIOUS / "00_manifest"))
sys.path.insert(0, str(ROOT / "02_preprocessed"))
from stage2c_common import atomic_json, read_json, write_csv, sha256, git, seed_all
from scripts.train_hivt_stage2 import MODEL_KEYS
from models.hivt_loss_recovery import HiVTLossRecovery
from metrics.hivt_forecasting import multimodal_errors
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import yaml

CLASSES = ("vehicle", "pedestrian", "bicycle")
GROUPS = ("overall",) + CLASSES + ("vehicle.moving", "vehicle.stopped", "vehicle.parked", "unknown")
HORIZONS = ("full_horizon", "partial_future")
CONFIG = ROOT / "00_manifest/stage3_config.yaml"


def config():
    return yaml.safe_load(CONFIG.read_text())


def verify_frozen():
    prior = read_json(ROOT / "00_manifest/stage3_frozen_previous.json")
    assert all(sha256(PROJECT / name) == value for name, value in prior["files"].items())
    atomic_json(ROOT / "00_manifest/stage3_freeze_verification.json",
                {"status": "PASS", "previous_files": len(prior["files"]), "base_commit": prior["base_commit"]})


def update_manifest():
    path = ROOT / "00_manifest/stage3_artifact_manifest.json"
    old = read_json(path) if path.exists() else {"artifacts": []}
    cached = {r["relative_path"]: r for r in old["artifacts"]}
    tracked = set(git("ls-files").splitlines())
    cp = ROOT / "00_manifest/stage3_checkpoint_manifest.json"
    checkpoints = read_json(cp).get("checkpoints", {}) if cp.exists() else {}
    rows = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p == path or p.name.endswith((".tmp", ".pyc")) or "__pycache__" in p.parts:
            continue
        rel = str(p.relative_to(ROOT)); stat = p.stat(); prior = cached.get(rel)
        digest = prior["sha256"] if prior and prior.get("mtime_ns") == stat.st_mtime_ns and prior["file_size"] == stat.st_size else sha256(p)
        row = {"relative_path": rel, "description": "Stage3A " + rel.split("/")[0],
               "file_size": stat.st_size, "sha256": digest, "git_tracked": str(p.relative_to(PROJECT)) in tracked,
               "created_by_stage": "Stage 3A", "mtime_ns": stat.st_mtime_ns}
        if rel in checkpoints: row["checkpoint_metadata"] = checkpoints[rel]
        rows.append(row)
    atomic_json(path, {"stage": "Stage 3A", "root": str(ROOT), "self_hash_excluded": True, "artifacts": rows})


def model_new():
    c = config(); seed_all(c["seed"])
    return HiVTLossRecovery(**{k: c[k] for k in MODEL_KEYS}).cuda()


def model_input(data):
    """The encoder receives no participant type field; targets retain it for auditing."""
    working = data.clone()
    if "agent_type" in working: del working.agent_type
    return working


class SceneDataset(Dataset):
    def __init__(self, split, cache_scenes=2):
        self.split = split; self.cache_limit = cache_scenes; self.cache = OrderedDict()
        with open(ROOT / f"02_preprocessed/stage3_{split}_index.csv") as f:
            self.all_rows = list(csv.DictReader(f))
        self.rows = [r for r in self.all_rows if int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) > 0]
        self.empty_rows = [r for r in self.all_rows if int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) == 0]
        self.scene_indices = OrderedDict()
        for i, row in enumerate(self.rows): self.scene_indices.setdefault(row["scene_token"], []).append(i)

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]; name = row["file_path"]
        if name not in self.cache:
            self.cache[name] = torch.load(ROOT / name, map_location="cpu", weights_only=False)["graphs"]
            if len(self.cache) > self.cache_limit: self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name][int(row["window_index"])].clone()

    def clear(self): self.cache.clear()


class SceneSampler(Sampler):
    def __init__(self, ds, seed=2022): self.dataset = ds; self.seed = seed; self.epoch = 0
    def __len__(self): return len(self.dataset)
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); scenes = list(self.dataset.scene_indices); rng.shuffle(scenes)
        for scene in scenes:
            indices = list(self.dataset.scene_indices[scene]); rng.shuffle(indices); yield from indices


def loss_diagnostics(model, output, data, phase):
    values = model.recovery_loss(output, data, phase)
    for t, name in enumerate(CLASSES):
        mask = data.target_mask & (data.agent_type == t)
        if mask.any():
            working = data.clone(); working.target_mask = mask
            diagnostic = model.recovery_loss(output, working, phase)
            values[f"{name}_regression_loss"] = diagnostic["regression_loss"].detach()
            values[f"{name}_NLL"] = diagnostic["NLL"].detach()
    return values


def errors_with_top1(model, output, data):
    prediction = model.ego_predictions(output, data)
    errors = multimodal_errors(prediction, data.positions[:, 5:], data.future_mask, data.target_mask)
    rows = torch.arange(data.num_nodes, device=prediction.device)
    top = output["mode_prob"].argmax(-1)
    top_error = torch.linalg.vector_norm(prediction[rows, top] - data.positions[:, 5:], dim=-1)
    errors["Top1ADE6"] = (top_error * data.future_mask).sum(-1) / errors["valid_steps"].clamp(min=1)
    errors["Top1FDE6"] = top_error[rows, errors["last_valid_index"].clamp(min=0)]
    target = torch.bmm(data.y, output["rotation"])
    raw = output["raw_prediction"]
    mask = data.future_mask & data.target_mask[:, None]
    best_loss = (torch.linalg.vector_norm(raw[..., :2] - target[None], dim=-1) * mask[None]).sum(-1).argmin(0)
    chosen = raw[best_loss, rows]
    density = (torch.log(2 * chosen[..., 2:]) + (chosen[..., :2] - target).abs() / chosen[..., 2:]).mean(-1)
    errors["NLL"] = (density * mask).sum(-1) / mask.sum(-1).clamp(min=1)
    errors["top1_mode"] = top
    return prediction, errors


def evaluate_graphs(model, graphs):
    from torch_geometric.data import Batch
    model.eval(); sums = {name: np.zeros(6) for name in ("overall",) + CLASSES}
    diagnostic = {name: [] for name in CLASSES}
    with torch.no_grad():
        for graph in graphs:
            data = Batch.from_data_list([graph]).cuda(); out = model(model_input(data))
            _, errors = errors_with_top1(model, out, data)
            values = loss_diagnostics(model, out, data, "fixed_scale")
            for t, name in enumerate(CLASSES):
                mask = data.target_mask & (data.agent_type == t)
                if mask.any():
                    diagnostic[name].append(float(values[name + "_regression_loss"]))
                for i in torch.where(mask)[0].tolist():
                    row = np.array([1, float(errors["minADE_K"][i]), float(errors["minFDE_K"][i]),
                                    float(errors["Top1ADE6"][i]), float(errors["Top1FDE6"][i]), float(errors["NLL"][i])])
                    assert np.isfinite(row).all()
                    sums[name] += row; sums["overall"] += row
    return {name: {"count": int(s[0]), **{k: float(s[j]/s[0]) if s[0] else None for j, k in enumerate(
        ("ADE", "FDE", "Top1ADE", "Top1FDE", "NLL"), start=1)},
        "fixed_scale_regression_loss": float(np.mean(diagnostic[name])) if name in diagnostic and diagnostic[name] else None}
        for name, s in sums.items()}


METRICS = ("minADE6", "minFDE6", "MR6", "Top1ADE6", "Top1FDE6", "NLL", "independent_minADE6")


class Accumulator:
    def __init__(self):
        self.sums = {h: {g: np.zeros(8) for g in GROUPS} for h in HORIZONS}
    def add(self, horizon, group, values):
        self.sums[horizon][group] += np.r_[1, values]
    def summary(self):
        return {h: {g: {"count": int(s[0]), **{k: float(s[i+1]/s[0]) if s[0] else None for i,k in enumerate(METRICS)}}
                    for g,s in groups.items()} for h,groups in self.sums.items()}


@torch.no_grad()
def evaluate(dataset, model, phase="original_nll", actor_path=None, progress=False):
    from torch_geometric.loader import DataLoader
    model.eval(); acc=Accumulator(); scenes={r["scene_token"]:Accumulator() for r in dataset.all_rows}
    seen=0; fields=["scene_name","scene_token","sample_token","instance_token","node_in_graph","horizon",
                    "agent_type","motion_state","valid_future_steps","GT_endpoint_displacement_m",*METRICS,"best_mode","top1_mode"]
    path=Path(actor_path) if actor_path else None
    temp=path.with_name(path.name+".tmp") if path else None
    handle=open(temp,"w",newline="") if temp else None
    writer=csv.DictWriter(handle,fields,lineterminator="\n") if handle else None
    if writer:writer.writeheader()
    try:
        for batch in DataLoader(dataset,batch_size=config()["batch_size"],shuffle=False,num_workers=0):
            data=batch.cuda();out=model(model_input(data));_,errors=errors_with_top1(model,out,data)
            assert torch.isfinite(out["raw_prediction"]).all() and torch.isfinite(out["mode_prob"]).all()
            keys=("minADE_K","minFDE_K","MR_K","Top1ADE6","Top1FDE6","NLL","independent_minADE_K")
            values=torch.stack([errors[k] for k in keys],-1).cpu().numpy()
            types=data.agent_type.cpu().numpy();graph_index=data.batch.cpu().numpy();ptr=data.ptr.cpu().numpy()
            valid=errors["valid_steps"].cpu().numpy();last=errors["last_valid_index"].cpu().numpy()
            displacement=torch.linalg.vector_norm(data.y[torch.arange(data.num_nodes,device=data.y.device),errors["last_valid_index"].clamp(min=0)],dim=-1).cpu().numpy()
            best=errors["best_mode"].cpu().numpy();top=errors["top1_mode"].cpu().numpy()
            for horizon in HORIZONS:
                selected=errors[horizon].cpu().numpy()
                assert np.isfinite(values[selected]).all()
                for node in np.flatnonzero(selected):
                    graph=int(graph_index[node]);local=int(node-ptr[graph]);cls=CLASSES[int(types[node])]
                    motion=data.t0_motion_state[graph][local];groups=["overall",cls]
                    if cls=="vehicle":groups.append(motion)
                    token=data.scene_token[graph]
                    for group in groups:
                        acc.add(horizon,group,values[node]);scenes[token].add(horizon,group,values[node])
                    if writer:
                        writer.writerow(dict(zip(fields,[data.scene_name[graph],token,data.sample_token[graph],data.instance_tokens[graph][local],local,horizon,
                            cls,motion,int(valid[node]),float(displacement[node]),*values[node].tolist(),int(best[node]),int(top[node])])))
            seen+=batch.num_graphs
            if progress and seen%200<config()["batch_size"]:print("VAL_EVALUATE",seen,"/",len(dataset),flush=True)
    finally:
        if handle:handle.close()
        dataset.clear()
    assert seen==len(dataset)
    if path:os.replace(temp,path)
    return {"metrics":acc.summary(),"scenes":{token:a.summary() for token,a in scenes.items()},"windows":seen,
            "candidate_windows":len(dataset.all_rows),"empty_supervision_windows":len(dataset.empty_rows),"test_used":False,
            "metric_definition":"ADE of best-FDE mode; MR endpoint>2m; Top1 argmax probability; NLL original best-summed-L2 mode, per-actor valid-time coordinate density mean",
            "K":6,"split":dataset.split}


# Training mechanics reused from Stage2C, with Stage3-bound paths and metadata.
SceneShardDataset=SceneDataset
SceneShuffleSampler=SceneSampler
verify_previous=verify_frozen
