"""Plot only actual official VAL observations, preserving prior figure files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("overall", "vehicle", "pedestrian", "bicycle")
LABELS = ("Overall", "Vehicle", "Pedestrian", "Bicycle")
COLORS = ("#246C9E", "#626A73", "#558E80", "#C18740")
MARKERS = ("o", "^", "s", "D")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_curve(path: Path, expected_steps):
    with path.open(newline="") as f:
        source = list(csv.DictReader(f))
    rows = []
    for row in source:
        step = int(float(row["global_step"]))
        fields = {name: float(row[f"VAL_{name}_FDE"]) for name in GROUPS}
        assert all(math.isfinite(v) and v >= 0 for v in fields.values()), row
        rows.append({"global_step": step, **fields})
    assert [r["global_step"] for r in rows] == list(expected_steps), (path.name, expected_steps, rows)
    return rows


def selected_step(manifest):
    if "final_best_global_step" in manifest:
        return int(manifest["final_best_global_step"])
    return int(manifest["final_best"]["global_step"])


def selected_fde(manifest):
    for key in ("overall_minFDE6", "overall_FDE", "overall_fde"):
        if key in manifest:
            return float(manifest[key])
    if "final_best" in manifest and "overall_minFDE6" in manifest["final_best"]:
        return float(manifest["final_best"]["overall_minFDE6"])
    for key in ("metrics", "final_metrics", "full_horizon_metrics"):
        if key in manifest:
            value = manifest[key]
            if "full_horizon" in value:
                value = value["full_horizon"]
            if "overall" in value:
                value = value["overall"]
            if "minFDE6" in value:
                return float(value["minFDE6"])
    raise KeyError("Final manifest must expose overall_minFDE6 or nested overall minFDE6.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "07_checkpoints/stage3a_final_frozen_checkpoint_manifest.json")
    p.add_argument("--curve", type=Path, default=ROOT / "03_no_type_baseline/stage3a_plusplus_nll_curve.csv")
    args = p.parse_args()
    plus_path = ROOT / "03_no_type_baseline/stage3a_plus_nll_extension_curve.csv"
    old_manifest_path = ROOT / "07_checkpoints/stage3a_plus_best_checkpoint_manifest.json"
    contract_path = ROOT / "00_manifest/stage3a_plusplus_convergence_figure_contract.json"
    old_manifest = read_json(old_manifest_path)
    final_manifest = read_json(args.manifest)
    assert read_json(contract_path)["backend"] == "python"
    checkpoint = ROOT / old_manifest["original_best"]["relative_path"]
    assert sha256(checkpoint) == old_manifest["original_best"]["sha256"]
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)["metadata"]
    initial = {
        "global_step": 15000,
        "overall": float(old_manifest["original_best"]["overall_minFDE6"]),
        **{name: float(original["per_class_full_horizon_metrics"][name]["minFDE6"]) for name in GROUPS[1:]},
    }
    plus = read_curve(plus_path, range(15500, 18001, 500))
    with args.curve.open(newline="") as f:
        raw_new = list(csv.DictReader(f))
    assert raw_new, "Wait for completed Stage3A++ validation observations."
    last = int(float(raw_new[-1]["global_step"]))
    assert 18500 <= last <= 21000 and (last - 18000) % 500 == 0
    new = read_curve(args.curve, range(18500, last + 1, 500))
    observations = [initial] + plus + new
    candidates = [plus[-1]] + new
    selected = min(candidates, key=lambda r: r["overall"])
    best_step, best_fde = selected_step(final_manifest), selected_fde(final_manifest)
    assert best_step == selected["global_step"], (best_step, selected)
    assert abs(best_fde - selected["overall"]) < 1e-6, (best_fde, selected)
    source_paths = [plus_path, args.curve, old_manifest_path, args.manifest, contract_path]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    source_table = ROOT / "05_figures/stage3a_final_convergence_source.csv"
    source_table_tmp = source_table.with_suffix(".csv.tmp")
    with source_table_tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["global_step", *[f"{name}_minFDE6_m" for name in GROUPS], "source_stage"], lineterminator="\n")
        writer.writeheader()
        for row in observations:
            step = row["global_step"]
            writer.writerow({"global_step": step, **{f"{name}_minFDE6_m": row[name] for name in GROUPS},
                             "source_stage": "Stage3A" if step == 15000 else "Stage3A+" if step <= 18000 else "Stage3A++"})
    source_table_tmp.replace(source_table)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8, "path.simplify": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig = plt.figure(figsize=(8.4, 4.3))
    ax = fig.add_axes((.10, .18, .64, .68))
    artists = []
    x = [row["global_step"] for row in observations]
    for name, label, color, marker in zip(GROUPS, LABELS, COLORS, MARKERS):
        artist, = ax.plot(x, [row[name] for row in observations], linestyle="None", marker=marker,
                          color=color, markerfacecolor=color, markeredgecolor="white", markeredgewidth=.6,
                          markersize=5.5, label=label, zorder=4)
        assert artist.get_linestyle() == "None"
        artists.append(artist)
    refs = [(15000, "Original best", "#B8BDC2", ":"), (18000, "Stage3A+ best", "#9FA9B4", "--")]
    if best_step != 18000:
        refs.append((best_step, "Frozen best", "#246C9E", "--"))
    for step, label, color, style in refs:
        ax.axvline(step, color=color, linestyle=style, linewidth=.8, zorder=1)
    values = [row[name] for row in observations for name in GROUPS]
    margin = max((max(values) - min(values)) * .10, .02)
    ax.set_ylim(min(values) - margin, max(values) + margin)
    ax.set_xlim(14880, last + 160)
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.xaxis.set_minor_locator(MultipleLocator(500))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.grid(axis="y", color="#E6E8EB", linewidth=.6)
    ax.set_xlabel("Executed global optimizer step", labelpad=8)
    ax.set_ylabel("Full-horizon minFDE6 (m)", labelpad=8)
    ax.set_title("Multi-Type No-Type baseline: observed validation", loc="left", fontsize=10, pad=11)
    legend = fig.legend(artists, LABELS, loc="upper left", bbox_to_anchor=(.765, .86), frameon=False,
                        borderaxespad=0, handlelength=1.0, labelspacing=.75)
    annotation = fig.text(.775, .48, f"Frozen best\nStep {best_step}\nOverall FDE =\n{selected['overall']:.6f} m",
                          fontsize=10, ha="left", va="top", linespacing=1.45, color="#246C9E")
    stage_text = fig.text(.775, .22, "Reference steps\nOriginal best: 15000\nStage3A+ best: 18000", fontsize=8,
                          ha="left", va="top", linespacing=1.4, color="#59626C")
    footer = fig.text(.10, .045, "Markers show actual official VAL observations; no fitted, smoothed or interpolated data.",
                     fontsize=8, color="#59626C", ha="left")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labelled = [legend, annotation, stage_text, footer, ax.xaxis.label, ax.yaxis.label, ax._left_title]
    boxes = [item.get_window_extent(renderer) for item in labelled]
    for box in boxes:
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1, "Text outside canvas"
    assert not boxes[0].overlaps(boxes[1]) and not boxes[1].overlaps(boxes[2]), "Side annotation overlap"
    stems = [(ROOT / "03_no_type_baseline/stage3a_plusplus_nll_curve", ("png",)),
             (ROOT / "05_figures/stage3a_final_convergence", ("png", "pdf", "svg"))]
    exports = {}
    for stem, extensions in stems:
        for ext in extensions:
            output = Path(str(stem) + "." + ext)
            fig.savefig(output, dpi=300, facecolor="white")
            if ext == "svg":
                output.write_text("\n".join(s.rstrip() for s in output.read_text().splitlines()) + "\n")
                assert "<text" in output.read_text(), "SVG text must remain editable"
            if ext == "png":
                with Image.open(output) as image:
                    assert image.size == (2520, 1290)
                    image.verify()
            exports[str(output.relative_to(ROOT))] = sha256(output)
    plt.close(fig)
    assert all(sha256(ROOT / name) == digest for name, digest in source_hashes.items()), "Source changed during plotting"
    atomic_json(ROOT / "05_figures/stage3a_final_convergence_audit.json", {
        "status": "PASS", "backend": "python", "sources_sha256": source_hashes,
        "original_15000_checkpoint_sha256": old_manifest["original_best"]["sha256"],
        "source_data_relative_path": str(source_table.relative_to(ROOT)), "source_data_sha256": sha256(source_table),
        "observed_global_steps": x, "observations_per_series": len(observations),
        "final_executed_global_step": last, "frozen_best_global_step": best_step,
        "frozen_best_selection_time_overall_minFDE6_m": selected["overall"],
        "reference_steps": [step for step, _, _, _ in refs],
        "marker_only_data": True, "connecting_data_lines": False,
        "smoothing": False, "fitting": False, "interpolation": False, "fabricated_points": False,
        "source_data_changed": False, "text_inside_canvas": True, "side_text_overlap": False,
        "white_background": True, "exports_sha256": exports})
    print("FINAL_CONVERGENCE_FIGURE=PASS", "steps=", x, "frozen_best=", best_step, flush=True)


if __name__ == "__main__":
    main()
