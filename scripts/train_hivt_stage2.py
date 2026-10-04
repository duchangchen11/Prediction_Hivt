"""Native PyTorch HiVT baseline: gated tiny overfit, then mini validation."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import yaml

from metrics.hivt_forecasting import aggregate, metric_records, multimodal_errors
from models.constant_velocity import constant_velocity
from models.hivt_nuscenes import HiVTNuScenesVehicle
from preprocessing.common import PROJECT_ROOT, write_json
from scripts.plot_hivt_predictions import plot_case

MODEL_KEYS = ("historical_steps", "future_steps", "num_modes", "embed_dim", "num_heads", "dropout", "num_temporal_layers", "num_global_layers", "local_radius")


@torch.no_grad()
def evaluate(model, graphs, batch_size=4):
    model.eval()
    records, losses = [], []
    for data in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        data = data.to("cuda")
        output = model(data)
        values = model.loss(output, data)
        assert all(torch.isfinite(value) for value in values.values())
        losses.append({k: float(v) for k, v in values.items()})
        pred = model.ego_predictions(output, data)
        records.extend(metric_records(multimodal_errors(pred, data.positions[:, 5:], data.future_mask, data.target_mask)))
    return {"metrics": aggregate(records), **{k: sum(v[k] for v in losses)/len(losses) for k in losses[0]}}


def overfit_pass(initial, final, config):
    gate = config["overfit_gate"]
    a, b = initial["metrics"]["full_horizon"], final["metrics"]["full_horizon"]
    if not a["count"] or not b["count"]:
        return False
    return (final["loss"] < initial["loss"]*gate["max_final_to_initial_loss"]
            and b["minADE_K"] < a["minADE_K"]*gate["max_final_to_initial_minADE"]
            and b["minFDE_K"] < a["minFDE_K"]*gate["max_final_to_initial_minFDE"]
            and b["minADE_K"] < gate["max_train_minADE_m"]
            and b["minFDE_K"] < gate["max_train_minFDE_m"])


@torch.no_grad()
def save_cases(model, graphs, directory, kind, count=5):
    model.eval()
    candidates = []
    for index, graph in enumerate(graphs):
        data = graph.clone().to("cuda")
        output = model(data)
        prediction = model.ego_predictions(output, data)
        errors = multimodal_errors(prediction, data.positions[:, 5:], data.future_mask, data.target_mask)
        # Prefer moving targets to avoid presenting parked-car trivial successes.
        motion = torch.linalg.vector_norm(data.positions[:, -1]-data.positions[:, 4], dim=1)
        eligible = torch.where(errors["full_horizon"] & (motion > 2))[0]
        for node in eligible.tolist():
            candidates.append((float(errors["minFDE_K"][node]), index, node))
    if not candidates:
        raise RuntimeError("No moving full-horizon vehicle available for figures")
    candidates.sort(reverse=(kind == "failure"))
    chosen, seen = [], set()
    for error, index, node in candidates:
        graph = graphs[index]
        token = graph.instance_tokens[node]
        if token in seen:
            continue
        seen.add(token)
        data = graph.clone().to("cuda")
        output = model(data)
        prediction = model.ego_predictions(output, data)
        path = Path(directory)/f"{kind}_{len(chosen)+1:02d}.png"
        plot_case(graph, prediction.cpu(), output["mode_prob"].cpu(), node, path, f"{kind}: minFDE_6={error:.3f}m")
        chosen.append({"figure": str(path.relative_to(PROJECT_ROOT)), "sample_token": graph.sample_token,
                       "scene_token": graph.scene_token, "instance_token": token, "node": node, "minFDE_K": error})
        if len(chosen) == count:
            break
    write_json(Path(directory)/f"{kind}_cases.json", chosen)
    return chosen


def plot_curves(rows, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    axes[0].plot([r["epoch"] for r in rows], [r["loss"] for r in rows], label="Train loss")
    axes[0].plot([r["epoch"] for r in rows], [r["regression_loss"] for r in rows], label="Regression")
    axes[0].plot([r["epoch"] for r in rows], [r["classification_loss"] for r in rows], label="Mode loss")
    valid = [r for r in rows if r["train_minADE"] is not None]
    axes[1].plot([r["epoch"] for r in valid], [r["train_minADE"] for r in valid], label="Train minADE_6")
    axes[1].plot([r["epoch"] for r in valid], [r["train_minFDE"] for r in valid], label="Train minFDE_6")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Full-horizon distance (m)")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run(mode, config):
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    # Original TF32 defaults are off on this torch build; retain FP32 behavior.
    corpus = torch.load(PROJECT_ROOT/"outputs/stage2/mini/processed/graphs.pt", weights_only=False, map_location="cpu")
    train = corpus["train"]
    if mode == "tiny":
        # Only assigned training scenes, deterministic uniformly spaced anchors.
        selected = np.linspace(0, len(train)-1, config["windows"], dtype=int)
        train = [train[i] for i in selected]
        val = []
        directory = PROJECT_ROOT/"outputs/stage2/tiny_overfit"
    else:
        gate = json.loads((PROJECT_ROOT/"outputs/stage2/tiny_overfit/metrics.json").read_text())
        if not gate["OVERFIT"]:
            raise RuntimeError("Tiny overfit failed; mini training prohibited")
        val = corpus["val"]
        directory = PROJECT_ROOT/"outputs/stage2/mini"
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(config, train_windows=len(train), val_windows=len(val),
                  metric_selection="best FDE mode as upstream; independent minimum ADE also reported", MR_threshold_m=2.,
                  processed_source="outputs/stage2/mini/processed/graphs.pt")
    (directory/"config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    write_json(directory/"training_windows.json", [{"scene_token": g.scene_token, "sample_token": g.sample_token} for g in train])
    model = HiVTNuScenesVehicle(**{key: config[key] for key in MODEL_KEYS}).cuda()
    optimizer = model.optimizer(config["learning_rate"], config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=0.)
    initial = evaluate(model, train, config["batch_size"])
    if mode == "tiny":
        save_cases(model, train, directory/"initial_predictions", "initial", count=5)
    print("INITIAL", json.dumps(initial), flush=True)
    loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    rows = [{"epoch": 0, **{k: initial[k] for k in ("loss", "regression_loss", "classification_loss")},
             "train_minADE": initial["metrics"]["full_horizon"]["minADE_K"], "train_minFDE": initial["metrics"]["full_horizon"]["minFDE_K"],
             "val_minADE": None, "val_minFDE": None, "learning_rate": config["learning_rate"], "epoch_seconds": 0.}]
    best_val = float("inf")
    for epoch in range(1, config["epochs"]+1):
        started = time.monotonic()
        model.train()
        losses = []
        for data in loader:
            data = data.to("cuda")
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            values = model.loss(output, data)
            if not torch.isfinite(values["loss"]):
                raise RuntimeError("Non-finite training loss")
            values["loss"].backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError("Non-finite gradient")
            optimizer.step()
            losses.append({k: float(v) for k, v in values.items()})
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {"epoch": epoch, **{key: sum(v[key] for v in losses)/len(losses) for key in losses[0]},
               "train_minADE": None, "train_minFDE": None, "val_minADE": None, "val_minFDE": None,
               "learning_rate": lr, "epoch_seconds": time.monotonic()-started}
        should_eval = epoch % config["evaluation_interval"] == 0 or epoch == config["epochs"]
        success = False
        if should_eval:
            final = evaluate(model, train, config["batch_size"])
            row["train_minADE"] = final["metrics"]["full_horizon"]["minADE_K"]
            row["train_minFDE"] = final["metrics"]["full_horizon"]["minFDE_K"]
            if val:
                validation = evaluate(model, val, config["batch_size"])
                row["val_minADE"] = validation["metrics"]["full_horizon"]["minADE_K"]
                row["val_minFDE"] = validation["metrics"]["full_horizon"]["minFDE_K"]
                if row["val_minFDE"] < best_val:
                    best_val = row["val_minFDE"]
                    torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch}, directory/"best_checkpoint.pt")
            else:
                success = epoch >= config["minimum_overfit_epochs"] and overfit_pass(initial, final, config)
        rows.append(row)
        with open(directory/"training_curve.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        if should_eval:
            print("EPOCH", json.dumps(row), flush=True)
        if success:
            break
    final = evaluate(model, train, config["batch_size"])
    torch.save({"state_dict": model.state_dict(), "config": config, "epoch": epoch}, directory/"checkpoint.pt")
    result = {"mode": mode, "epochs": epoch, "train_windows": len(train), "val_windows": len(val),
              "initial": initial, "final": final, "K": 6, "model": "HiVT-NuScenes-Vehicle-Baseline"}
    if mode == "tiny":
        result["OVERFIT"] = overfit_pass(initial, final, config)
        result["gate"] = config["overfit_gate"]
        save_cases(model, train, directory/"prediction_figures", "overfit", count=5)
        save_cases(model, train, PROJECT_ROOT/"outputs/figures/stage2/tiny_overfit", "overfit", count=5)
    else:
        result["validation_final_checkpoint"] = evaluate(model, val, config["batch_size"])
        selected = torch.load(directory/"best_checkpoint.pt", weights_only=False, map_location="cpu")
        model.load_state_dict(selected["state_dict"])
        result["best_validation_epoch"] = selected["epoch"]
        result["validation_best_checkpoint"] = evaluate(model, val, config["batch_size"])
        save_cases(model, val, PROJECT_ROOT/"outputs/figures/stage2/mini_success", "success", count=5)
        save_cases(model, val, PROJECT_ROOT/"outputs/figures/stage2/mini_failure", "failure", count=5)
    write_json(directory/"metrics.json", result)
    plot_curves(rows, directory/"training_curve.png")
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tiny", "mini"), required=True)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    path = args.config or PROJECT_ROOT/f"configs/stage2_{args.mode}.yaml"
    run(args.mode, yaml.safe_load(Path(path).read_text()))
