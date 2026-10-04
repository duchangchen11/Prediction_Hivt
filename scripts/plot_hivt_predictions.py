"""Lane overlays, six mode probabilities, best-FDE trajectory and GT."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import torch


def plot_case(data, prediction, probabilities, node, path, label, fit_all_modes=False):
    lanes = torch.stack([data.lane_positions, data.lane_positions+data.lane_vectors], dim=1).cpu().numpy()
    history = data.positions[node, :5].cpu().numpy().copy()
    history[~data.history_mask[node].cpu().numpy()] = np.nan
    future = data.positions[node, 5:].cpu().numpy().copy()
    fmask = data.future_mask[node].cpu().numpy()
    future[~fmask] = np.nan
    pred, prob = prediction[node].cpu().numpy(), probabilities[node].cpu().numpy()
    last = np.flatnonzero(fmask)[-1]
    best = np.argmin(np.linalg.norm(pred[:, last]-future[last], axis=1))
    fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
    ax.add_collection(LineCollection(lanes, colors="#9da5aa", linewidths=.8, alpha=.65, label="Lane centerlines"))
    ax.plot(history[:, 0], history[:, 1], "ko-", ms=4, lw=2, label="History")
    target = np.vstack([history[-1], future])
    ax.plot(target[:, 0], target[:, 1], color="#159750", marker="o", ms=3, lw=2.2, label="GT future")
    colors = plt.get_cmap("tab10")
    for mode in range(len(pred)):
        trajectory = np.vstack([history[-1], pred[mode]])
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors(mode), lw=1.3, ls="--", alpha=.7,
                label=f"Mode {mode+1}: p={prob[mode]:.2f}")
    trajectory = np.vstack([history[-1], pred[best]])
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=colors(best), lw=3, label=f"Best FDE: mode {best+1}")
    shown = pred.reshape(-1, 2) if fit_all_modes else pred[best]
    valid = np.concatenate([history[np.isfinite(history).all(axis=1)], future[fmask], shown])
    lower, upper = valid.min(axis=0)-10, valid.max(axis=0)+10
    center, half = (lower+upper)/2, max((upper-lower).max()/2, 15)
    ax.set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half),
           xlabel="t0 ego forward x (m)", ylabel="t0 ego left y (m)", title=f"{label}\n{data.scene_name} | {data.instance_tokens[node][:12]} | full horizon={bool(data.full_horizon_mask[node])}")
    ax.set_aspect("equal")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8, loc="best")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map(data, path):
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    lanes = torch.stack([data.lane_positions, data.lane_positions+data.lane_vectors], dim=1).numpy()
    ax.add_collection(LineCollection(lanes, colors="#a1a8ad", linewidths=1., label="Lane/connector centerlines"))
    for i in range(data.num_nodes):
        h, f = data.positions[i, :5].numpy().copy(), data.positions[i, 5:].numpy().copy()
        h[~data.history_mask[i].numpy()] = np.nan
        f[~data.future_mask[i].numpy()] = np.nan
        ax.plot(h[:, 0], h[:, 1], color="#2464a7", lw=1.5)
        curve = np.vstack([h[-1], f])
        ax.plot(curve[:, 0], curve[:, 1], color="#d36f27", ls="--", lw=1.)
    ax.scatter([0], [0], color="black", marker="*", s=120, label="Ego t0")
    ax.annotate("", (15, 0), (0, 0), arrowprops={"arrowstyle": "->", "color": "black"})
    ax.text(1, -5, "Ego +x")
    ax.set(xlim=(-70, 100), ylim=(-80, 80), xlabel="Ego forward x (m)", ylabel="Ego left y (m)", title=f"{data.scene_name}: lanes + vehicle history / GT future")
    ax.set_aspect("equal")
    ax.legend()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
