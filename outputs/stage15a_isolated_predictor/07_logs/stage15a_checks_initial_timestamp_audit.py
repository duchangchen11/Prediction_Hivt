"""Reproducible small InnerTrain/InnerDev checks, with no outer model evaluation."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "00_manifest"), str(ROOT / "02_training"), str(ROOT / "05_candidate_interface")]
import stage15a_common as c
from stage15a_train import (parse_args, config_for, checkpoint_save, restore, optimize_step,
    update_selection, new_phase_state, train_registered_phase, run_preflight_phases)
from stage15a_ranking import (candidate_windows, pack_window, fit_normalization, normalized,
    fresh_heads, head_forward, objective, ranking_loss)
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from preprocessing.coordinates import ego_to_global, quaternion_yaw


def difference(a, b):
    return float((a - b).abs().max()) if a.numel() else 0.


def must_reject(action, name):
    try:
        action()
    except (AssertionError, RuntimeError) as e:
        return {"Probe": name, "Rejected": True, "Reason": str(e)}
    raise AssertionError("Isolation violation was not rejected: " + name)


def manual_loss(out, data, phase):
    raw = out["raw_prediction"]
    gt = torch.bmm(data.y, out["rotation"])
    mask = data.future_mask & data.target_mask[:, None]
    d = (raw[..., :2] - gt[None]).norm(dim=-1)
    cost = (d * mask[None]).sum(-1)
    mode = cost.argmin(0)
    chosen = raw[mode, torch.arange(data.num_nodes, device=raw.device)]
    steps = mask.sum(-1)
    eligible = steps > 0
    q = (-cost[:, eligible] / steps[eligible]).softmax(0).t().detach()
    cls = -(q * out["mode_logits"][eligible].log_softmax(-1)).sum(-1).mean()
    err = (chosen[..., :2][mask] - gt[mask]).abs()
    reg = err.mean() if phase == "fixed_scale" else (torch.log(2 * chosen[..., 2:][mask]) + err / chosen[..., 2:][mask]).mean()
    return reg + cls


def input_and_loss_audit(model, data):
    model.eval()
    with torch.no_grad():
        normal = model(data)
        stripped = c.model_input(data)
        out = model(stripped)
        diff = {k: difference(normal[k], out[k]) for k in ("raw_prediction", "mode_logits", "mode_prob")}
        assert max(diff.values()) < 1e-6
        poison = data.clone()
        poison.positions[:, 5:] = float("nan")
        poison.y.fill_(float("nan"))
        poison.future_mask = ~poison.future_mask
        poison.target_mask = ~poison.target_mask
        poison.padding_mask[:, 5:] = ~poison.padding_mask[:, 5:]
        poison.ego_future.fill_(float("nan"))
        poison.future_times.fill_(float("nan"))
        mutated = model(c.model_input(poison))
        pdiff = {k: difference(out[k], mutated[k]) for k in diff}
        assert max(pdiff.values()) == 0.
        # Physically absent labels, plus independent manual original losses.
        assert not set(stripped.keys()) & {"y", "future_mask", "target_mask", "full_horizon_mask", "ego_future"}
        ld = {}
        for phase in ("fixed_scale", "original_nll"):
            actual = model.recovery_loss(out, data, phase)["loss"]
            expected = manual_loss(out, data, phase)
            ld[phase] = difference(actual, expected)
            assert ld[phase] < 1e-6
        pred = model.ego_predictions(out, stripped)
        explicit = out["raw_prediction"][..., :2].permute(1, 0, 2, 3) @ out["rotation"].transpose(-1, -2)[:, None]
        explicit += data.positions[:, 4, None, None]
        assert difference(pred, explicit) == 0.
        assert pred.shape == (data.num_nodes, 6, 12, 2)
        assert out["mode_logits"].shape == out["mode_prob"].shape == (data.num_nodes, 6)
        assert torch.allclose(out["mode_prob"].sum(-1), torch.ones(data.num_nodes, device=data.y.device), atol=1e-6)
    return {"Status": "PASS", "OrdinaryVsStrippedMaxDiff": diff, "GTPoisonMaxDiff": pdiff,
            "ForwardKeys": sorted(stripped.keys()), "PositionsShape": list(stripped.positions.shape),
            "FutureFieldsPhysicallyAbsent": True, "ManualLossMaxDiff": ld, "CoordinateConversionMaxDiff": 0.}


def compare_rng(saved):
    assert torch.equal(torch.get_rng_state(), saved["torch_rng"])
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), saved["cuda_rng"]))
    assert random.getstate() == saved["python_rng"]
    a, b = np.random.get_state(), saved["numpy_rng"]
    assert a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


def recovery_audit(context, train):
    warm = context.output / "stage15a_preflight_warm_best.pt"
    model, optimizer, saved = restore(warm, context)
    compare_rng(saved)
    cursor = dict(saved["iterator"])
    reference = optimize_step(model, optimizer, train, cursor, "fixed_scale")
    expected_model = {n: t.cpu().clone() for n, t in model.state_dict().items()}
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_rng = {"torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
    del model, optimizer
    torch.cuda.empty_cache()
    # Scramble all RNG streams before the independent reload.
    c.seed_all(context.seed + 999)
    torch.rand(17, device="cuda")
    model, optimizer, resumed = restore(warm, context)
    compare_rng(resumed)
    resumed_cursor = dict(resumed["iterator"])
    replay = optimize_step(model, optimizer, train, resumed_cursor, "fixed_scale")
    maximum = max(difference(v, model.state_dict()[n].cpu()) for n, v in expected_model.items())
    assert maximum == 0. and reference == replay and cursor == resumed_cursor
    after = optimizer.state_dict()
    assert expected_optimizer["param_groups"] == after["param_groups"]
    moments = 0.
    for i, row in expected_optimizer["state"].items():
        for k, v in row.items():
            if torch.is_tensor(v):
                moments = max(moments, difference(v.cpu(), after["state"][i][k].cpu()))
            else:
                assert v == after["state"][i][k]
    assert moments == 0.
    compare_rng(expected_rng)
    reloaded = context.output / "stage15a_preflight_resume_probe.pt"
    state = dict(saved["phase_state"])
    state["phase_step"] += 1
    checkpoint_save(reloaded, model, optimizer, context, state, resumed_cursor)
    result = {"Status": "PASS", "ModelMaxDiff": maximum, "OptimizerMomentsMaxDiff": moments,
              "NextBatchIdentitiesExact": True, "LossAndGradientsExact": True, "AllRNGStreamsExact": True,
              "CursorExact": True, "AdditionalProbeUpdates": 2,
              "CheckpointSHA256": c.sha256(reloaded), "CheckpointBytes": reloaded.stat().st_size}
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def stopping_rule_audit():
    s = new_phase_state()
    s["phase_step"] = 500
    assert update_selection(s, 3., 5500, "original_nll", 16000)
    for j, value in enumerate((3., 3.1, 3., 3.4, 3.2), start=2):
        s["phase_step"] = 500 * j
        assert not update_selection(s, value, 5000 + s["phase_step"], "original_nll", 16000)
        assert s["completed"] == (j == 6)
    assert s["bad_validations"] == 5 and s["best_step"] == 5500
    w = new_phase_state()
    w["phase_step"] = 4500
    w["bad_validations"] = 8
    assert update_selection(w, 2., 4500, "fixed_scale", 5000) and not w["completed"]
    w["phase_step"] = 5000
    update_selection(w, 2., 5000, "fixed_scale", 5000)
    assert w["completed"]
    n = new_phase_state()
    n["phase_step"] = 16000
    update_selection(n, 1., 21000, "original_nll", 16000)
    assert n["completed"]
    return {"Status": "PASS", "StrictTiesDoNotImprove": True, "NLLPatience5": True,
            "WarmupFixed5000": True, "NLLMaximum16000": True, "GlobalMaximum21000": True,
            "SimulationOnlyNoOptimizer": True}


def raw_identity_audit(graphs):
    db = c.PROJECT / "outputs/stage2c_trainval_vehicle_baseline/02_preprocessed/stage2c_metadata_cache/stage2c_trajectory_metadata.sqlite"
    con = sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True)
    def get(table, token):
        row = con.execute(f"SELECT payload FROM {table} WHERE token=?", (token,)).fetchone()
        assert row is not None
        return json.loads(row[0])
    results = []
    for g in graphs:
        anchor = get("samples", g.sample_token)
        assert anchor["scene_token"] == g.scene_token
        before, cur = [], anchor
        for j in range(4):
            cur = get("samples", cur["prev"])
            assert cur["scene_token"] == g.scene_token
            before.append(cur)
        after, cur = [], anchor
        for j in range(12):
            cur = get("samples", cur["next"])
            assert cur["scene_token"] == g.scene_token
            after.append(cur)
        samples = before[::-1] + [anchor] + after
        timestamps = np.array([s["timestamp"] for s in samples], dtype=np.int64)
        times = (timestamps - timestamps[4]) / 1e6
        assert np.all(np.diff(timestamps) > 0)
        assert np.array_equal(times[:5], g.history_times.numpy()) and np.array_equal(times[5:], g.future_times.numpy())
        assert abs(times[0] + 2.) < .15 and abs(times[-1] - 6.) < .15
        maximum, annotations = 0., 0
        for node, tokens in enumerate(g.annotation_tokens):
            visible = torch.cat((g.history_mask[node], g.future_mask[node])).numpy()
            assert len(g.instance_tokens) == len(set(g.instance_tokens)) == g.num_nodes
            for j, token in enumerate(tokens):
                assert bool(token) == bool(visible[j])
                if not token:
                    continue
                ann = get("annotations", token)
                assert ann["instance_token"] == g.instance_tokens[node] and ann["sample_token"] == samples[j]["token"]
                global_xy = ego_to_global(g.positions[node, j:j + 1].numpy(), g.origin.numpy(), float(g.ego_yaw))[0]
                maximum = max(maximum, float(np.abs(global_xy - np.array(ann["translation"][:2])).max()))
                annotations += 1
        assert maximum < 1e-4
        assert torch.equal(g.padding_mask, ~torch.cat((g.history_mask, g.future_mask), 1))
        assert torch.equal(g.target_mask, (g.history_mask.sum(-1) >= 2) & g.future_mask.any(-1))
        if g.lane_actor_index.numel():
            lane, actor = g.lane_actor_index
            assert torch.allclose(g.lane_actor_vectors, g.lane_positions[lane] - g.positions[actor, 4], atol=1e-5)
        results.append({"scene_token": g.scene_token, "sample_token": g.sample_token, "actors": g.num_nodes,
                        "annotation_identity_checks": annotations, "coordinate_max_error_m": maximum,
                        "timestamp_sequence_sha256": hashlib.sha256(timestamps.tobytes()).hexdigest(),
                        "history_times": times[:5].tolist(), "future_times": times[5:].tolist()})
    con.close()
    return {"Status": "PASS", "Source": "read-only original metadata SQLite exact selected train/dev keys",
            "Windows": results, "OuterQueries": 0, "SourceSHA256": c.sha256(db)}


def ranking_audit(model, context, train, dev, graphs, batch):
    windows = candidate_windows(model, batch, graphs, context, "InnerTrain")
    raw = [pack_window(w) for w in windows]
    # Fit only full-horizon train targets, after constructing the full GT-free
    # graph; inference probes default to past-only eligible actors.
    fit = [pack_window(w, torch.where(g.target_mask & g.future_mask.all(-1))[0]) for w, g in zip(windows, graphs)]
    fit = [e for e in fit if len(e["targets"])]
    norm = fit_normalization(fit, context)
    dg = [dev[i] for i in range(4)]
    db = Batch.from_data_list(dg)
    dw = candidate_windows(model, db, dg, context, "InnerDev")
    dr = [pack_window(w) for w in dw]
    rejected = must_reject(lambda: fit_normalization(dr, context), "development_normalization_fit")
    heads = fresh_heads(context.seed)
    assert {k: sum(p.numel() for p in m.parameters()) for k, m in heads.items()} == {
        "R2": 673, "NG-A": 7425, "NG-C": 7425, "G-A": 24066, "G-C": 24066, "Matched-NG-C": 24001}
    results = []
    before = c.state_digest(model.state_dict())
    model.zero_grad(set_to_none=True)
    for e in raw + dr:
        args, r2 = normalized(e, norm)
        for name, head in heads.items():
            out = head_forward(head, name, args, r2)
            assert torch.equal(out["mode_logits"], args[6]) and torch.count_nonzero(out["delta_logits"]) == 0
            assert torch.allclose(out["mode_prob"].sum(-1), torch.ones(len(r2)), atol=1e-6)
        # A nonzero random fixture scoring layer exercises shared-mode alignment
        # beyond the trivial zero-initialized head. No head optimizer is created.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(context.seed + 901)
            for name, head in heads.items():
                score = head.net[-1] if name == "R2" else head.head[-1]
                torch.nn.init.normal_(score.weight, std=.01)
        perm = torch.tensor([2, 5, 0, 4, 1, 3])
        pa = (args[0][:, :, perm], args[1][:, perm][:, :, :, perm], args[2], None, None, None, args[6][:, perm])
        pr = r2[:, perm]
        for name, head in heads.items():
            out = head_forward(head, name, args, r2)
            reordered = head_forward(head, name, pa, pr)
            pdiff = difference(out["mode_logits"][:, perm], reordered["mode_logits"])
            assert pdiff < 1e-6
            results.append({"Model": name, "Role": e["source_role"], "scene_token": e["scene_token"],
                            "Targets": len(e["targets"]), "ModePermutationMaxDiff": pdiff})
        # Reset fixture score layers for the next original-init check.
        for name, head in heads.items():
            score = head.net[-1] if name == "R2" else head.head[-1]
            torch.nn.init.zeros_(score.weight)
    poison_diffs = []
    for w, e in zip(windows, raw):
        poisoned = dict(w)
        poisoned.update(GT=torch.full_like(w["ego_prediction"][:, 0], float("nan")),
                        future_mask=torch.zeros(w["ego_prediction"].shape[::2], dtype=torch.bool),
                        target_mask=torch.zeros(len(w["history"]), dtype=torch.bool))
        ee = pack_window(poisoned)
        assert e["identities"] == ee["identities"]
        for a, b in zip(e["args"], ee["args"]):
            if a is not None:
                assert torch.equal(a, b)
        assert torch.equal(e["r2"], ee["r2"]) and torch.equal(e["flags"], ee["flags"])
        poison_diffs.append(0.)
    # Detached training labels are used only by each ranking loss after forward.
    e = fit[0]
    args, r2 = normalized(e, norm)
    wi = next(i for i, w in enumerate(windows) if w["sample_token"] == e["sample_token"])
    gt = graphs[wi].positions[e["targets"], 5:]
    fde = (windows[wi]["ego_prediction"][e["targets"], :, -1] - gt[:, None, -1]).norm(dim=-1)
    for name, head in heads.items():
        out = head_forward(head, name, args, r2)
        loss = ranking_loss(out["mode_logits"], fde) if name == "R2" else objective(out["mode_logits"], fde, "A" if name.endswith("-A") else "C").mean()
        head.zero_grad(set_to_none=True)
        loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert all(p.grad is None for p in model.parameters()) and c.state_digest(model.state_dict()) == before
    dest = ROOT / f"05_candidate_interface/cache/fold{context.fold}"
    dest.mkdir(parents=True, exist_ok=True)
    candidate_path, label_path = dest / "stage15a_candidates.pt", dest / "stage15a_labels.pt"
    torch.save({"windows": windows, "fold": context.fold, "scope": "PREFLIGHT_ONLY", "GT_in_payload": False}, candidate_path)
    torch.save({"labels": [{"GT": g.positions[:, 5:], "future_mask": g.future_mask, "target_mask": g.target_mask,
                            "scene_token": g.scene_token, "sample_token": g.sample_token} for g in graphs]}, label_path)
    c.atomic_json(ROOT / f"05_candidate_interface/stage15a_fold{context.fold}_normalization.json", norm)
    result = {"Status": "PASS", "ParameterCounts": {k: sum(p.numel() for p in m.parameters()) for k, m in heads.items()},
        "RawFeatureShapes": {"node": list(raw[0]["args"][0].shape), "edge": list(raw[0]["args"][1].shape),
                             "neighbor_mask": list(raw[0]["args"][2].shape), "R2": list(raw[0]["r2"].shape)},
        "K": 6, "NodeDim": 15, "EdgeDim": 17, "R2Dim": 19, "MapInputs": "None; frozen G1 has no semantic/map module",
        "ModesAndLogitsAligned": True, "GTPoisonFeatureMaxDiff": max(poison_diffs),
        "Normalization": norm, "RejectedDevNormalization": rejected, "HeadBackwardFinite": True,
        "HistoricalHeadWeightsLoaded": False, "HeadOptimizerUpdates": 0, "BicycleFutureRoute": "new same-fold R2 after future authorized fitting",
        "PredictorGradientsZeroAndStateUnchanged": True, "ProbeResults": results,
        "CandidateCacheBytes": candidate_path.stat().st_size, "LabelSidecarBytes": label_path.stat().st_size,
        "CandidateSHA256": c.sha256(candidate_path), "LabelSHA256": c.sha256(label_path)}
    c.atomic_json(ROOT / f"05_candidate_interface/stage15a_fold{context.fold}_ranking_audit.json", result)
    return result


def forward_timing(model, batch, context, repeats=5):
    context.check_batch(batch, "InnerTrain", "forward_benchmark")
    data = c.model_input(batch.cuda())
    model.eval()
    with torch.no_grad():
        model(data)
        rows = []
        for i in range(repeats):
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall = time.perf_counter()
            start.record()
            model(data)
            end.record()
            torch.cuda.synchronize()
            rows.append({"cuda_seconds": start.elapsed_time(end) / 1000., "wall_seconds": time.perf_counter() - wall,
                         "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
    return {"BatchWindows": batch.num_graphs, "BatchActors": batch.num_nodes, "Repeats": repeats, "Rows": rows,
            "Scope": "predictor eval forward with pre-transferred observed graph; excludes shard loading and graph/R2 construction"}


def run(context):
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train, dev = c.FoldDataset(context, "InnerTrain"), c.FoldDataset(context, "InnerDev")
    assert not (context.output / "stage15a_training_probe.json").exists(), "Use a new directory for a reproducibility replay"
    c.atomic_json(context.output / "stage15a_entry_description.json", {"Status": "PASS", "Fold": context.fold, "Seed": context.seed,
        "Config": config_for(context), "TrainWindows": len(train), "DevWindows": len(dev),
        "TrainScenes": len(train.scene_indices), "DevScenes": len(dev.scene_indices), "FullTrainingAuthorized": False})
    negative = []
    tp = ROOT / context.record["files"]["InnerTrain"]["path"]
    dp = ROOT / context.record["files"]["InnerDev"]["path"]
    negative.append(must_reject(lambda: c.FoldContext(context.fold, context.seed + 1, tp, dp, context.output, install=False), "wrong_fold_seed"))
    negative.append(must_reject(lambda: c.FoldContext(context.fold, context.seed, dp, tp, context.output, install=False), "swapped_train_dev_lists"))
    negative.append(must_reject(lambda: c.FoldContext(context.fold, context.seed, tp, dp, c.S5 / "07_checkpoints", install=False), "historical_output_directory"))
    for role in ("OuterTest", "QuarantinedHeadDev"):
        token = next(iter(context.parts[role]))
        path = next(p for p, s in context.path_to_scene.items() if s == token)
        negative.append(must_reject(lambda p=path: context.load(p, "graph"), role + "_graph_load"))
        negative.append(must_reject(lambda p=path: p.open("rb"), role + "_raw_shard_open"))
    negative.append(must_reject(lambda: c.FoldDataset(context, "OuterTest"), "outer_dataset_creation"))
    negative.append(must_reject(lambda: train_registered_phase(context, "fixed_scale"), "formal_training_gate"))
    negative.append(must_reject(lambda: c.PROJECT.joinpath("outputs/stage3_multitype_hivt/02_preprocessed/stage3_val_index.csv").read_text(), "official_VAL_index"))
    oldcp = c.S5 / "07_checkpoints/stage5a_best_overall_minfde.pt"
    negative.append(must_reject(lambda: oldcp.open("rb"), "old_TRAIN700_checkpoint"))
    model = c.model_new(context.seed)
    digest = c.state_digest(model.state_dict())
    model2 = c.model_new(context.seed)
    assert digest == c.state_digest(model2.state_dict())
    del model2
    if context.fold == 1:
        # Construct the literal historical fresh step0 reference, never its weights.
        from stage5a_common import model_new as historical_fresh
        reference = historical_fresh(device="cpu", audit=False)
        assert digest == c.state_digest(reference.state_dict())
        del reference
    assert not c.LOADED, "No checkpoint/shard loader may have been invoked at initialization"
    init = {"Status": "PASS", "Fold": context.fold, "Seed": context.seed, "StateSHA256": digest,
            "Parameters": sum(p.numel() for p in model.parameters()), "TrainableParameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "FreshRepeatExact": True, "HistoricalSeed2022FreshReferenceExact": context.fold == 1,
            "TypeEmbedding": [3, 64], "MotionExperts": 2, "NeutralRouter": [.5, .5],
            "TRAIN700WeightsLoaded": False, "LoadsAtInitialization": list(c.LOADED)}
    ids = c.scene_order(train, context.seed, 0)[:16]
    graphs = [train[int(i)] for i in ids]
    batch = Batch.from_data_list(graphs)
    context.check_batch(batch, "InnerTrain", "model_integrity")
    inp = input_and_loss_audit(model, batch.cuda())
    identity = raw_identity_audit(graphs[:2] + [dev[i] for i in range(2)])
    timing = forward_timing(model, batch, context)
    rank = ranking_audit(model, context, train, dev, graphs[:4], Batch.from_data_list(graphs[:4]))
    del model
    torch.cuda.empty_cache()
    training = run_preflight_phases(context)
    recovery = recovery_audit(context, train)
    resource = {"forward": timing, "training_rows": training["Rows"], "StressBatch": None, "StepBenchmark": []}
    extra_updates = 0
    if context.fold == 2:
        # Largest16 actor-count windows are a metadata-only conservative stress
        # check; selection uses neither GT trajectories nor error metrics.
        stress_ids = sorted(range(len(train.rows)), key=lambda i: (-int(train.rows[i]["actor_count"]), i))[:16]
        sg = [train[i] for i in stress_ids]
        sb = Batch.from_data_list(sg)
        context.check_batch(sb, "InnerTrain", "optimization")
        model = c.model_new(context.seed)
        optimizer = model.optimizer(.001, .0001)
        model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        data = sb.cuda()
        optimizer.zero_grad(set_to_none=True)
        loss = model.recovery_loss(model(c.model_input(data)), data, "fixed_scale")["loss"]
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        torch.cuda.synchronize()
        resource["StressBatch"] = {"Status": "PASS", "Windows": 16, "Actors": sb.num_nodes,
            "ActorCounts": [g.num_nodes for g in sg], "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "selection": "largest t0 actor_count metadata within Fold2 InnerTrain; stress fixture never formal checkpoint"}
        extra_updates += 1
        del model, optimizer, loss, data
        torch.cuda.empty_cache()
    if context.fold == 1:
        model = c.model_new(context.seed)
        optimizer = model.optimizer(.001, .0001)
        cursor = {"epoch": 0, "next_batch": 0}
        for j in range(5):
            torch.cuda.synchronize()
            started = time.perf_counter()
            row = optimize_step(model, optimizer, train, cursor, "fixed_scale")
            torch.cuda.synchronize()
            resource["StepBenchmark"].append({"seconds": time.perf_counter() - started, "batch_windows": len(row["dataset_indices"]),
                "scene_tokens": row["scene_tokens"], "scope": "full training step with shard fetch/collation/H2D/backward/AdamW"})
            extra_updates += 1
        del model, optimizer
        torch.cuda.empty_cache()
    dev_batch = Batch.from_data_list([dev[i] for i in range(2)])
    negative.append(must_reject(lambda: context.check_batch(dev_batch, "InnerDev", "optimization"), "development_optimizer_use"))
    rejection_count = len(c.BLOCKED)
    # All blocked attempts are deliberate negative probes, with no payload read.
    reads = {"Status": "PASS", "Loaded": c.LOADED, "LoadedShardSHA256": context.loaded_shards,
             "BatchRecords": context.batch_records, "BlockedNegativeProbes": c.BLOCKED,
             "NegativeTests": negative, "ReadPaths": sorted(c.READS), "OuterPerformanceComputed": False,
             "OfficialVALLoaded": False, "HeadDevLoaded": False, "OldWeightsLoaded": False}
    updates = 6 + 2 + extra_updates
    resource.update(Device=torch.cuda.get_device_name(), DeviceTotalBytes=torch.cuda.get_device_properties(0).total_memory,
        TorchVersion=torch.__version__, CUDAVersion=torch.version.cuda, DiskFreeBytes=shutil.disk_usage(ROOT).free,
        ActualOptimizerUpdates=updates, CandidateCacheBytes=rank["CandidateCacheBytes"], LabelSidecarBytes=rank["LabelSidecarBytes"],
        CheckpointBytes={p.name: p.stat().st_size for p in context.output.glob("*.pt")})
    summary = {"Status": "PASS", "Fold": context.fold, "Seed": context.seed, "Initialization": init,
               "ModelInputAndLoss": inp, "Identity": identity, "Recovery": recovery, "StoppingRules": stopping_rule_audit(),
               "RankingStatus": rank["Status"], "ActualOptimizerUpdates": updates, "FullPredictorRuns": 0,
               "OuterInferenceRuns": 0, "Stage15BStarted": False}
    c.atomic_json(ROOT / f"03_checks/stage15a_fold{context.fold}_checks.json", summary)
    c.atomic_json(ROOT / f"01_data_isolation/stage15a_fold{context.fold}_runtime_isolation.json", reads)
    c.atomic_json(ROOT / f"06_resources/stage15a_fold{context.fold}_resources.json", resource)
    print("STAGE15A_FOLD_PREFLIGHT_PASS", context.fold, "updates", updates, "outer_metrics0", flush=True)


if __name__ == "__main__":
    args = parse_args()
    assert args.mode == "preflight", "This check entry accepts preflight only"
    ctx = c.FoldContext(args.fold, args.seed, args.training_scenes, args.development_scenes, args.output_dir)
    run(ctx)
