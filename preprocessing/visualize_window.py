"""Plot valid trajectory segments only in the common t0 ego frame."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from .common import PROJECT_ROOT

STYLES = [("Vehicle", "#2464a7", "s"), ("Pedestrian", "#d75a2a", "o"), ("Bicycle", "#7c4ba1", "^")]


def plot_valid(ax, points, mask, **kwargs):
    # NaN is used only in the plotting array to break missing observations.
    shown = np.array(points, dtype=float, copy=True)
    shown[~np.asarray(mask, dtype=bool)] = np.nan
    ax.plot(shown[:, 0], shown[:, 1], **kwargs)


def visualize(window, output):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), layout="constrained")
    hist, future = window["agent_pos"].numpy(), window["future_pos"].numpy()
    hmask, fmask = window["history_mask"].numpy(), window["future_mask"].numpy()
    ego_h, ego_f = window["ego_history"].numpy(), window["ego_future"].numpy()
    types = window["agent_type"].numpy()
    for ax in axes:
        for i, kind in enumerate(types):
            name, color, marker = STYLES[kind]
            plot_valid(ax, hist[i], hmask[i], color=color, marker=marker, ms=2.6, lw=1, alpha=.65)
            # Include t0 in future line so the history/future junction is visible.
            plot_valid(ax, np.vstack([hist[i, -1], future[i]]), np.r_[hmask[i, -1], fmask[i]],
                       color=color, marker=marker, ms=2, ls="--", lw=1, alpha=.65)
            current = hist[i, -1]
            direction = np.array([np.cos(window["agent_heading"][i, -1]), np.sin(window["agent_heading"][i, -1])])
            ax.quiver(*current, *direction, color=color, angles="xy", scale_units="xy", scale=.5, width=.0025)
        ax.plot(ego_h[:, 0], ego_h[:, 1], "k.-", lw=2, ms=5, zorder=5)
        ef = np.vstack([ego_h[-1], ego_f])
        ax.plot(ef[:, 0], ef[:, 1], "k.--", lw=2, ms=4, zorder=5)
        ax.scatter([0], [0], marker="*", s=130, color="black", zorder=6)
        ax.annotate("", xy=(15, 0), xytext=(0, 0),
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.7}, zorder=7)
        ax.text(2, -6, "Ego forward +x", fontsize=10, zorder=7)
        ax.axhline(0, color="gray", alpha=.2)
        ax.axvline(0, color="gray", alpha=.2)
        ax.set(xlabel="Ego forward x (m)", ylabel="Ego left y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.15)
    all_valid = np.concatenate([hist[hmask], future[fmask], ego_h, ego_f])
    lower, upper = all_valid.min(axis=0)-10, all_valid.max(axis=0)+10
    center = (lower+upper)/2
    half = max((upper-lower).max()/2, 30)
    axes[0].set(xlim=(center[0]-half, center[0]+half), ylim=(center[1]-half, center[1]+half), title="All current actors")
    axes[1].set(xlim=(-30, 70), ylim=(-45, 45), title="Near ego (same coordinates)")
    legend = [Line2D([0], [0], color=c, marker=m, label=f"{name} (N={int((types == i).sum())})")
              for i, (name, c, m) in enumerate(STYLES)]
    legend += [Line2D([0], [0], color="k", lw=2, label="Ego"),
               Line2D([0], [0], color="gray", ls="-", label="History"),
               Line2D([0], [0], color="gray", ls="--", label="GT future")]
    axes[0].legend(handles=legend, fontsize=9, loc="best")
    audit = window["audit"]
    fig.suptitle(f"{window['scene_name']} | t0={window['t0_index']} | history {audit['history_duration_s']:.3f}s, future {audit['future_duration_s']:.3f}s\n"
                 "Origin: current ego | +x forward | +y left | Exact instance-token association", fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(PROJECT_ROOT / "outputs/debug/one_window.pt"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs/figures/one_window.png"))
    args = parser.parse_args()
    visualize(torch.load(args.input, map_location="cpu", weights_only=True), args.output)


if __name__ == "__main__":
    main()
