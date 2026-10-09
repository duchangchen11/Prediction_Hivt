"""Fold-bound sources and fresh Stage5A construction; never import its trainer."""
import csv
import hashlib
import json
import os
import random
import sys
from collections import OrderedDict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
S3 = PROJECT / "outputs/stage3_multitype_hivt"
S5 = PROJECT / "outputs/stage5a_motion_aware_decoder"
PROTOCOL = ROOT / "00_manifest/stage15a_protocol.json"
sys.path[:0] = [str(PROJECT), str(S3 / "00_manifest"), str(S3 / "02_preprocessed"), str(S5 / "00_manifest")]
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from stage3_dataset import VehicleGraph  # read-only class required by existing shards
from stage3b_model import HiVTTypeEmbedding
from stage5a_model import HiVTMotionAwareDecoder
from models.hivt_loss_recovery import HiVTLossRecovery
from metrics.hivt_forecasting import multimodal_errors

READS = set()
BLOCKED = []
LOADED = []
ACTIVE = None


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, t in sorted(state.items()):
        h.update(name.encode())
        h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT), "All new output must be Stage15A-local"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def model_new(seed, device="cuda"):
    """Exact Stage3B→Stage5A fresh-constructor sequence, now parameterized by seed."""
    c = read_json(PROTOCOL)
    kwargs = c["model"]
    seed_all(seed)
    no_type = HiVTLossRecovery(**kwargs)
    shared = {n: t.detach().clone() for n, t in no_type.state_dict().items()}
    canonical_rng = torch.get_rng_state().clone()
    typed = HiVTTypeEmbedding(**kwargs)
    missing, unexpected = typed.load_state_dict(shared, strict=False)
    assert missing == ["type_embedding.weight"] and not unexpected
    torch.set_rng_state(canonical_rng)
    shared = {n: t.detach().clone() for n, t in typed.state_dict().items()}
    model = HiVTMotionAwareDecoder(**kwargs)
    missing, unexpected = model.load_state_dict(shared, strict=False)
    assert not unexpected and len(missing) == 12
    assert all(n.startswith(("decoder.experts.", "decoder.router.")) for n in missing)
    assert all(torch.equal(v, model.state_dict()[n]) for n, v in shared.items())
    assert sum(p.numel() for p in model.parameters()) == c["parameter_count"] == 650403
    assert all(torch.count_nonzero(e[-1].weight) == 0 and torch.count_nonzero(e[-1].bias) == 0 for e in model.decoder.experts)
    assert torch.count_nonzero(model.decoder.router[-1].weight) == 0
    assert torch.count_nonzero(model.decoder.router[-1].bias) == 0
    torch.set_rng_state(canonical_rng)
    return model.to(device)


INPUT_KEYS = ("x", "positions", "edge_index", "padding_mask", "bos_mask", "rotate_angles",
              "lane_vectors", "is_intersections", "turn_directions", "traffic_controls",
              "lane_actor_index", "lane_actor_vectors", "agent_type")


def model_input(data):
    """Physically remove supervision and truncate history before predictor.forward."""
    values = {k: data[k].clone() for k in INPUT_KEYS}
    values["positions"] = values["positions"][:, :5].clone()
    values["padding_mask"] = values["padding_mask"][:, :5].clone()
    for k in ("batch", "ptr"):
        if k in data:
            values[k] = data[k].clone()
    working = Data(num_nodes=data.num_nodes, **values)
    assert set(working.keys()) <= set(INPUT_KEYS) | {"num_nodes", "batch", "ptr"}
    assert working.positions.shape == (data.num_nodes, 5, 2)
    assert working.padding_mask.shape == (data.num_nodes, 5)
    HiVTTypeEmbedding.validate_actor_types(working.agent_type, working.num_nodes)
    return working


class FoldContext:
    def __init__(self, fold, seed, training_scenes, development_scenes, output_dir, install=True):
        self.protocol = read_json(PROTOCOL)
        self.fold = fold
        assert fold in (1, 2, 3)
        self.record = self.protocol["folds"][fold - 1]
        assert self.record["fold"] == fold and seed == self.record["seed"], "Frozen fold seed mismatch"
        self.seed = seed
        self.parts = {k: frozenset(v) for k, v in self.record["parts"].items()}
        for role, path in (("InnerTrain", training_scenes), ("InnerDev", development_scenes)):
            value = read_json(path)
            assert value["fold"] == fold and value["seed"] == seed and value["role"] == role
            assert value["scene_tokens"] == self.record["parts"][role], "Original scene token order/list mismatch"
            assert sha256(path) == self.record["files"][role]["sha256"], "Scene list differs from registration"
        assert [len(self.parts[k]) for k in ("InnerTrain", "InnerDev", "OuterTest", "QuarantinedHeadDev")] == [378, 42, 210, 70]
        assert len(set.union(*(set(v) for v in self.parts.values()))) == 700
        self.output = Path(output_dir).resolve()
        assert self.output.is_relative_to(ROOT) and self.output != ROOT
        for path, digest in self.protocol["source_sha256"].items():
            assert sha256(PROJECT / path) == digest, "Frozen dependency changed: " + path
        with (S3 / "02_preprocessed/stage3_train_index.csv").open() as f:
            self.index = list(csv.DictReader(f))
        self.path_to_scene = {}
        for r in self.index:
            p = (S3 / r["file_path"]).resolve()
            assert not p in self.path_to_scene or self.path_to_scene[p] == r["scene_token"]
            self.path_to_scene[p] = r["scene_token"]
        self.allowed_files = {p for p, s in self.path_to_scene.items() if s in self.parts["InnerTrain"] | self.parts["InnerDev"]}
        with (PROJECT / "outputs/stage14a_paper_graph_ablation/09_reports/stage14a_predictor_train_shard_manifest.csv").open() as f:
            self.shard_hashes = {str((PROJECT / r["Path"]).resolve()): r["SHA256"] for r in csv.DictReader(f)}
        self.batch_records = []
        self.loaded_shards = {}
        if install:
            self.install_guard()
        self.output.mkdir(parents=True, exist_ok=True)

    def install_guard(self):
        global ACTIVE
        assert ACTIVE is None, "One fold per process; isolated contexts cannot be mixed"
        ACTIVE = self
        def guard(event, args):
            if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
                return
            p = Path(os.path.abspath(os.fsdecode(args[0]))).resolve()
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            write = isinstance(mode, str) and any(c in mode for c in "wax+") or isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR))
            reason = None
            if write and p.is_relative_to(PROJECT) and not p.is_relative_to(ROOT):
                reason = "HISTORICAL_WRITE"
            if not write and p.is_relative_to(PROJECT):
                READS.add(str(p.relative_to(PROJECT)))
                if p.suffix in (".pt", ".pth", ".npy", ".npz") and not p.is_relative_to(ROOT) and p not in self.allowed_files:
                    reason = "NONFOLD_ARRAY_OR_HISTORICAL_CHECKPOINT"
                if "/02_preprocessed/val/" in str(p) or "/02_preprocessed/test/" in str(p) or p.name == "stage3_val_index.csv":
                    reason = "OFFICIAL_VAL_OR_TEST"
                if p in self.path_to_scene and p not in self.allowed_files:
                    reason = "OUTER_OR_HEADDEV_SHARD"
            if reason:
                BLOCKED.append({"path": str(p), "reason": reason})
                raise RuntimeError("Stage15A forbidden read/write: " + reason + " " + str(p))
        sys.addaudithook(guard)

    def check_batch(self, data, role, purpose):
        assert role in ("InnerTrain", "InnerDev"), "OuterTest cannot be evaluated in Stage15A"
        scenes = list(data.scene_token)
        assert scenes and set(scenes) <= self.parts[role], "Batch scene role mismatch"
        assert not set(scenes) & (self.parts["OuterTest"] | self.parts["QuarantinedHeadDev"])
        assert role == "InnerTrain" or purpose != "optimization", "InnerDev must not generate optimizer updates"
        self.batch_records.append({"fold": self.fold, "role": role, "purpose": purpose,
                                   "scene_tokens": scenes, "sample_tokens": list(data.sample_token), "windows": data.num_graphs})

    def load(self, path, scope):
        p = Path(path).resolve()
        if scope == "graph":
            assert p in self.allowed_files, "OuterTest/HeadDev70/old weights cannot be loaded"
        else:
            assert scope == "preflight_checkpoint" and p.is_relative_to(self.output), "Checkpoint must belong to this new fold/run"
        LOADED.append({"path": str(p.relative_to(PROJECT)), "scope": scope})
        return torch.load(p, map_location="cpu", weights_only=False)


class FoldDataset(Dataset):
    def __init__(self, context, role):
        assert role in ("InnerTrain", "InnerDev"), "Only train/dev datasets exist in this entry"
        self.context, self.role = context, role
        self.all_rows = [r for r in context.index if r["scene_token"] in context.parts[role]]
        self.rows = [r for r in self.all_rows if int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) > 0]
        self.empty_rows = [r for r in self.all_rows if int(r["full_horizon_target_count"]) + int(r["partial_target_count"]) == 0]
        self.scene_indices = OrderedDict()
        for i, row in enumerate(self.rows):
            self.scene_indices.setdefault(row["scene_token"], []).append(i)
        assert set(self.scene_indices) == context.parts[role]
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        assert isinstance(i, int) and 0 <= i < len(self.rows)
        row = self.rows[i]
        p = (S3 / row["file_path"]).resolve()
        assert p in self.context.allowed_files and row["scene_token"] in self.context.parts[self.role]
        if p not in self.cache:
            digest = sha256(p)
            assert digest == self.context.shard_hashes[str(p)], "Original shard hash mismatch"
            self.context.loaded_shards[str(p.relative_to(PROJECT))] = digest
            payload = self.context.load(p, "graph")
            assert payload["scene_token"] == row["scene_token"]
            self.cache[p] = payload["graphs"]
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(p)
        graph = self.cache[p][int(row["window_index"])].clone()
        assert (graph.scene_token, graph.scene_name, graph.sample_token) == (row["scene_token"], row["scene_name"], row["sample_token"])
        assert graph.num_nodes == int(row["actor_count"])
        return graph

    def clear(self):
        self.cache.clear()


def scene_order(dataset, seed, epoch):
    """Exact historical SceneSampler algorithm, with caller-provided fold seed."""
    rng = random.Random(seed + epoch)
    scenes = list(dataset.scene_indices)
    rng.shuffle(scenes)
    order = []
    for s in scenes:
        ids = list(dataset.scene_indices[s])
        rng.shuffle(ids)
        order.extend(ids)
    return order


def next_training_batch(dataset, iterator, phase):
    offset = 0 if phase == "fixed_scale" else 100000
    order = scene_order(dataset, dataset.context.seed, iterator["epoch"] + offset)
    batches = [order[i:i + 16] for i in range(0, len(order), 16)]
    if iterator["next_batch"] >= len(batches):
        iterator["epoch"] += 1
        iterator["next_batch"] = 0
        return next_training_batch(dataset, iterator, phase)
    ids = batches[iterator["next_batch"]]
    # Match the historical DataLoader base-seed draw exactly once per new epoch.
    # Rebuilding one small loader per step must not introduce extra RNG draws;
    # a restored mid-epoch cursor therefore preserves the saved RNG streams.
    if iterator["next_batch"] == 0:
        torch.empty((), dtype=torch.int64).random_()
    loader = DataLoader(dataset, batch_sampler=[ids], num_workers=0, generator=torch.Generator().manual_seed(0))
    data = next(iter(loader))
    dataset.context.check_batch(data, "InnerTrain", "optimization")
    return data, ids


@torch.no_grad()
def evaluate_dev(model, dataset, ids=None):
    assert dataset.role == "InnerDev"
    model.eval()
    batches = [list(range(i, min(i + 16, len(dataset)))) for i in range(0, len(dataset), 16)] if ids is None else [list(ids)]
    count, total = 0, 0.
    for batch in DataLoader(dataset, batch_sampler=batches, num_workers=0):
        dataset.context.check_batch(batch, "InnerDev", "development_evaluation")
        data = batch.to(next(model.parameters()).device)
        out = model(model_input(data))
        pred = model.ego_predictions(out, model_input(data))
        errors = multimodal_errors(pred, data.positions[:, 5:], data.future_mask, data.target_mask)
        mask = errors["full_horizon"]
        values = errors["minFDE_K"][mask]
        assert torch.isfinite(values).all()
        count += int(mask.sum())
        total += float(values.double().sum())
    assert count > 0
    return {"full_horizon_Overall_minFDE6": total / count, "Count": count,
            "selection_role": "InnerDev", "subset_only": ids is not None}
