"""Paper figure exports from the completed, frozen Stage14A internal OOF audit.

No model fitting, new inference, scientific decision, or checkpoint selection.
The figure contract is defined before any evaluation-array read. SVG/PDF exports
retain editable text; PNGs are previews of the same matplotlib figures.
"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "00_protocol"))
from stage14a_common import *

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


FIGURES = ROOT / "08_figures"
DATASET = "nuScenes HeadTrain630 | 3-fold internal ranking OOF"
LIMIT = "Frozen predictor trained on overlapping scenes; not an independent official test."
COMPARISONS = ("NG-C-NG-A", "G-A-NG-A", "G-C-NG-C", "G-C-G-A")
SWITCH_COMPARISONS = ("NG-C-NG-A", "G-C-NG-C", "G-C-G-A")
COLORS = {"NG-A": "#8DA9C4", "NG-C": "#436F98", "G-A": "#B4A2C8", "G-C": "#75538D"}
DISPLAY = {"NG-A": "NoGraph + SoftCE", "NG-C": "NoGraph + Error-Aware",
           "G-A": "Graph + SoftCE", "G-C": "Graph + Error-Aware"}
SOURCES = (
    "05_evaluation/stage14a_ablation_metrics.csv",
    "05_evaluation/stage14a_fold_metrics.csv",
    "06_bootstrap/stage14a_bootstrap_ci.csv",
    "06_bootstrap/stage14a_model_mean_ci.csv",
    "07_diagnostics/stage14a_mode_switch.csv",
)


def gate():
    assert os.environ.get("STAGE14A_PHASE") not in ("tiny", "train")
    frozen = read_json(ROOT / "04_checkpoints/stage14a_all_frozen.json")
    assert frozen["Status"] == "FROZEN_ALL_COMPLETE"
    assert frozen["OuterTestEvaluationPermitted"] and len(frozen["Checkpoints"]) == 6
    expected = {str(cp_path(fold, variant).relative_to(PROJECT))
                for fold in (1, 2, 3) for variant in VARIANTS}
    assert {row["Path"] for row in frozen["Checkpoints"]} == expected
    for row in frozen["Checkpoints"]:
        assert sha256(PROJECT / row["Path"]) == row["SHA256"]
    audit = read_json(ROOT / "05_evaluation/stage14a_identity_audit.json")
    assert audit["Status"] == "PASS" and audit["Models"] == list(MODELS)
    assert audit["AllSixNewCheckpointsFrozenBeforeEvaluation"]
    assert audit["GeometryAllModelsBitwiseEqual"] and audit["BicyclePreserved"] == "YES"
    for name, digest in audit["cache_files"].items():
        assert sha256(ROOT / "05_evaluation/cache" / name) == digest
    for name in ("06_bootstrap/stage14a_bootstrap_audit.json",
                 "07_diagnostics/stage14a_mode_switch_audit.json"):
        assert read_json(ROOT / name)["Status"] == "PASS"
    assert read_json(FIGURES / "stage14a_figure_contract.json")["Backend"] == "Python / matplotlib only"
    return audit


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.frameon": False, "legend.fontsize": 7,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def one(table, **criteria):
    keep = np.ones(len(table), bool)
    for field, value in criteria.items():
        keep &= np.asarray(table[field] == value)
    result = table.loc[keep]
    assert len(result) == 1, criteria
    return result.iloc[0]


def footer(fig, text, count):
    fig.text(.02, .062, f"{DATASET} | n={count:,} actor-windows", fontsize=7)
    fig.text(.02, .035, text, fontsize=6.7)
    fig.text(.02, .012, LIMIT, fontsize=6.4, color="#555555")


def export(fig, stem, family, output, extra=None):
    paths = []
    fig.canvas.draw()
    for extension in ("svg", "pdf", "png"):
        path = FIGURES / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=.08,
                    dpi=180 if extension == "png" else 300)
        assert path.exists() and path.stat().st_size > 500
        if extension == "svg":
            svg = path.read_text()
            assert "<text" in svg, "SVG text must remain editable"
        paths.append({"Path": str(path.relative_to(ROOT)), "Bytes": path.stat().st_size,
                      "SHA256": sha256(path), "Format": extension})
    plt.close(fig)
    output.append({"Family": family, "Files": paths, **(extra or {})})


def model_mean_axis(ax, means, group):
    for index, model in enumerate(MODELS):
        row = one(means, Group=group, Model=model)
        mean = row.Top1FDE
        ax.plot([row.CI95Lower, row.CI95Upper], [index, index],
                color=COLORS[model], lw=1.8)
        ax.scatter([mean], [index], s=30, color=COLORS[model], zorder=3,
                   marker="o" if model.endswith("A") else "s")
        ax.annotate(f"{mean:.4f}", (row.CI95Upper, index), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=7)
    ax.set_yticks(range(4), [DISPLAY[model] for model in MODELS])
    ax.invert_yaxis()
    upper = max(one(means, Group=group, Model=model).CI95Upper for model in MODELS)
    ax.set_xlim(0, upper * 1.2)
    ax.set_xlabel("Top1FDE (m)")
    ax.grid(axis="x", color="#EEEEEE", lw=.6)


def contrast_axis(ax, intervals, group, comparisons=COMPARISONS, adjusted=False):
    lows, highs = [], []
    for index, comparison in enumerate(comparisons):
        row = one(intervals, Group=group, Comparison=comparison)
        lo = row.BonferroniCILower if adjusted else row.CI95Lower
        hi = row.BonferroniCIUpper if adjusted else row.CI95Upper
        color = "#75538D" if comparison.startswith("G-") else "#436F98"
        ax.plot([lo, hi], [index, index], lw=1.7, color=color)
        ax.scatter([row.DeltaTop1FDE], [index], color=color, s=26, zorder=3)
        lows.append(lo); highs.append(hi)
    ax.axvline(0, color="#777777", ls=":", lw=.8)
    ax.set_yticks(range(len(comparisons)), [name.replace("-NG-", " − NG-")
                                          .replace("-G-", " − G-") for name in comparisons])
    ax.invert_yaxis()
    lo, hi = min(0., min(lows)), max(0., max(highs))
    padding = max((hi - lo) * .18, .003)
    ax.set_xlim(lo - padding, hi + padding)
    ax.set_xlabel("Paired Δ Top1FDE (m)\nNegative favors first model", fontsize=7)
    ax.grid(axis="x", color="#EEEEEE", lw=.6)


def overall_figure(means, intervals, output):
    count = int(one(means, Group="Overall", Model="NG-A").Count)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.35),
                             gridspec_kw={"width_ratios": [1, 1.4]})
    fig.subplots_adjust(left=.08, right=.98, bottom=.23, top=.86, wspace=.38)
    ax = axes[0]
    for row, family in enumerate(("NG", "G")):
        for column, loss in enumerate(("A", "C")):
            model = f"{family}-{loss}"
            value = one(means, Group="Overall", Model=model)
            ax.add_patch(Rectangle((column, row), 1, 1, facecolor=COLORS[model], alpha=.14,
                                   edgecolor="white", lw=2))
            ax.text(column + .5, row + .34, f"{value.Top1FDE:.4f} m", ha="center", va="center",
                    fontsize=12, color=COLORS[model], weight="bold")
            ax.text(column + .5, row + .63,
                    f"95% CI [{value.CI95Lower:.4f}, {value.CI95Upper:.4f}]",
                    ha="center", fontsize=6.2)
            ax.text(column + .5, row + .86, model, ha="center", fontsize=7)
    ax.set_xlim(0, 2); ax.set_ylim(2, 0)
    ax.set_xticks([.5, 1.5], ["SoftCE", "Error-Aware"])
    ax.set_yticks([.5, 1.5], ["NoGraph", "Graph"])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("a  Overall: graph × loss", loc="left")
    contrast_axis(axes[1], intervals, "Overall", adjusted=True)
    axes[1].set_title("b  Four registered paired contrasts", loc="left")
    footer(fig, "Means: descriptive 95% scene-cluster CI. Contrasts: family4-adjusted 98.75% CI. Top1FDE in m.", count)
    dump("08_figures/stage14a_fig1_source_data.csv", pd.concat([
        means.loc[means.Group == "Overall"].assign(Panel="model means"),
        intervals.loc[intervals.Group == "Overall"].assign(Panel="paired contrasts")
    ], ignore_index=True).to_dict("records"))
    export(fig, "stage14a_fig1_overall_2x2", "Overall2x2", output)


def type_figure(means, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    fig.subplots_adjust(left=.22, right=.98, bottom=.24, top=.87, wspace=.72)
    for letter, ax, group in zip("ab", axes, ("Vehicle", "Pedestrian")):
        model_mean_axis(ax, means, group)
        ax.set_title(f"{letter}  {group} | n={int(one(means, Group=group, Model=MODELS[0]).Count):,}", loc="left")
    count = int(one(means, Group="Vehicle", Model="NG-A").Count + one(means, Group="Pedestrian", Model="NG-A").Count)
    footer(fig, "Group means and descriptive 95% paired scene-cluster intervals; units m. Bicycle uses identical R2 route.", count)
    dump("08_figures/stage14a_fig2_source_data.csv", means.loc[means.Group.isin(["Vehicle", "Pedestrian"])].to_dict("records"))
    export(fig, "stage14a_fig2_vehicle_pedestrian", "VehiclePedestrian", output)


def moving_figure(means, intervals, output):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.3))
    fig.subplots_adjust(left=.22, right=.98, bottom=.24, top=.87, wspace=.72)
    model_mean_axis(axes[0], means, "MovingVehicle")
    axes[0].set_title(f"a  MovingVehicle | n={int(one(means, Group='MovingVehicle', Model='NG-A').Count):,}", loc="left")
    contrast_axis(axes[1], intervals, "MovingVehicle", SWITCH_COMPARISONS)
    axes[1].set_title("b  Exploratory paired contrasts", loc="left")
    count = int(one(means, Group="MovingVehicle", Model="NG-A").Count)
    footer(fig, "Current-state vehicle.moving group; descriptive/exploratory 95% scene-cluster CI. Units m.", count)
    dump("08_figures/stage14a_fig3_source_data.csv", pd.concat([
        means.loc[means.Group == "MovingVehicle"].assign(Panel="model means"),
        intervals.loc[intervals.Group == "MovingVehicle"].assign(Panel="paired contrasts")
    ], ignore_index=True).to_dict("records"))
    export(fig, "stage14a_fig3_moving_vehicle", "MovingVehicle", output)


def table_axis(ax, cells, headers, title, widths):
    ax.axis("off")
    ax.set_title(title, loc="left", pad=9)
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="upper center", colWidths=widths)
    table.auto_set_font_size(False); table.set_fontsize(7)
    table.scale(1, 1.8)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#DCDCDC"); cell.set_linewidth(.5)
        if row == 0:
            cell.set_facecolor("#EEF1F4"); cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#FAFAFA")
    return table


def ablation_figure(metrics, intervals, output):
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 6.8))
    fig.subplots_adjust(left=.03, right=.98, bottom=.16, top=.93, hspace=.45)
    cells = []
    for model in MODELS:
        cells.append([model, "Yes" if model.startswith("G-") else "No", "SoftCE" if model.endswith("A") else "Error-Aware",
                      *[f"{one(metrics, Group=group, Model=model).Top1FDE:.4f}"
                        for group in ("Overall", "Vehicle", "Pedestrian", "Bicycle")]])
    table_axis(axes[0], cells,
               ["Model", "Graph", "Loss", "Overall", "Vehicle", "Pedestrian", "Bicycle"],
               "a  Controlled graph / loss ablation: Top1FDE (m)", [.1, .08, .15, .15, .15, .15, .2])
    contrast_cells = []
    for comparison in COMPARISONS:
        row = one(intervals, Group="Overall", Comparison=comparison)
        contrast_cells.append([comparison, f"{row.DeltaTop1FDE:+.5f}",
                               f"[{row.BonferroniCILower:+.5f}, {row.BonferroniCIUpper:+.5f}]",
                               *[f"{row[f'Fold{fold}DeltaTop1FDE']:+.5f}" for fold in (1, 2, 3)]])
    table_axis(axes[1], contrast_cells,
               ["Paired contrast", "Δ (m)", "Adjusted 98.75% CI", "Fold1 Δ", "Fold2 Δ", "Fold3 Δ"],
               "b  Four registered Overall contrasts: first model minus second", [.2, .11, .3, .13, .13, .13])
    motion_groups = ("MovingVehicle", "StoppedVehicle", "ParkedVehicle")
    motion_cells = [[model, *[f"{one(metrics, Group=group, Model=model).Top1FDE:.5f}"
                            for group in motion_groups]] for model in MODELS]
    motion_headers = ["Model", *[f"{group}\n(n={int(one(metrics, Group=group, Model='NG-A').Count):,})"
                                  for group in motion_groups]]
    table_axis(axes[2], motion_cells, motion_headers,
               "c  All prespecified current vehicle states: Top1FDE (m)", [.13, .29, .29, .29])
    count = int(one(metrics, Group="Overall", Model="NG-A").Count)
    sizes = "; ".join(f"{group} n={int(one(metrics, Group=group, Model='NG-A').Count):,}"
                      for group in ("Vehicle", "Pedestrian", "MovingVehicle"))
    parked = one(intervals, Group="ParkedVehicle", Comparison="G-C-NG-C")
    fig.text(.02, .105, f"ParkedVehicle G-C − NG-C={parked.DeltaTop1FDE:+.5f} m; exploratory 95% CI [{parked.CI95Lower:+.5f}, {parked.CI95Upper:+.5f}].", fontsize=6.7)
    footer(fig, f"{sizes}. Bicycle: identical frozen fold-R2 route. Paired Overall intervals: family4 Bonferroni.", count)
    dump("08_figures/stage14a_fig4_source_data.csv", pd.concat([
        metrics.loc[metrics.Group.isin(["Overall", "Vehicle", "Pedestrian", "Bicycle", "MovingVehicle", "StoppedVehicle", "ParkedVehicle"])].assign(Panel="model means"),
        intervals.loc[intervals.Group.isin(["Overall", "ParkedVehicle"])].assign(Panel="paired contrasts")
    ], ignore_index=True).to_dict("records"))
    export(fig, "stage14a_fig4_ablation_table", "AblationTable", output)


def switch_figure(switches, output):
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.3))
    fig.subplots_adjust(left=.09, right=.98, bottom=.18, top=.91, wspace=.34, hspace=.53)
    labels = ("NG-C − NG-A", "G-C − NG-C", "G-C − G-A")
    x = np.arange(3)
    for row_index, group in enumerate(("Overall", "MovingVehicle")):
        rows = [one(switches, Comparison=comparison, Group=group) for comparison in SWITCH_COMPARISONS]
        left, right = axes[row_index]
        left.bar(x - .18, [row.GrossGain for row in rows], .34, color="#719E92", label="Gross gain")
        left.bar(x + .18, [row.GrossHarm for row in rows], .34, color="#C3978B", label="Gross harm")
        for index, row in enumerate(rows):
            y = max(row.GrossGain, row.GrossHarm)
            left.annotate(f"{int(row.ImprovedCount):,} / {int(row.WorsenedCount):,}",
                          (index, y), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6)
        left.set_ylim(0, max(max(row.GrossGain, row.GrossHarm) for row in rows) * 1.24)
        left.set_ylabel("Sum of FDE gains / harms (m)")
        left.set_title(f"{'ac'[row_index]}  {group}: gain / harm", loc="left")
        left.legend(loc="upper right", ncol=2, fontsize=6)
        right.bar(x, [row.HighCostHarmSum for row in rows], .58, color="#A97E73")
        for index, row in enumerate(rows):
            right.annotate(f"tail n={int(row.HighCostHarmCount):,}", (index, row.HighCostHarmSum),
                           xytext=(0, 4), textcoords="offset points", ha="center", fontsize=6)
        right.set_ylim(0, max(row.HighCostHarmSum for row in rows) * 1.24)
        right.set_ylabel("High-cost FDE harm sum (m)")
        right.set_title(f"{'bd'[row_index]}  {group}: high-cost harms", loc="left")
        for ax in (left, right):
            ax.set_xticks(x, labels, fontsize=6.5)
            ax.grid(axis="y", color="#EEEEEE", lw=.6); ax.set_axisbelow(True)
    overall = int(one(switches, Group="Overall", Comparison=SWITCH_COMPARISONS[0]).Count)
    moving = int(one(switches, Group="MovingVehicle", Comparison=SWITCH_COMPARISONS[0]).Count)
    footer(fig, f"MovingVehicle n={moving:,}. Labels: improved / worsened counts. Tail=largest ceil(10% × positive harms); tails differ.", overall)
    dump("08_figures/stage14a_fig5_source_data.csv", switches.loc[
        switches.Group.isin(["Overall", "MovingVehicle"]) & switches.Comparison.isin(SWITCH_COMPARISONS)].to_dict("records"))
    export(fig, "stage14a_fig5_high_cost_switch", "HighCostSwitch", output)


def choose_cases(frame, metrics, probabilities):
    delta = np.asarray(metrics[:, 3, 0] - metrics[:, 1, 0])
    changed = probabilities[:, 3].argmax(-1) != probabilities[:, 1].argmax(-1)
    keys = frame.scene_token + "|" + frame.sample_token + "|" + frame.instance_token
    definitions = (("Vehicle improvement", "Vehicle", "median improvement"),
                   ("Vehicle failure", "Vehicle", "median harm"),
                   ("Pedestrian improvement", "Pedestrian", "median improvement"),
                   ("Pedestrian failure", "Pedestrian", "median harm"),
                   ("MovingVehicle large improvement", "MovingVehicle", "largest improvement"),
                   ("MovingVehicle high-cost failure", "MovingVehicle", "largest harm"))
    result, seen_scenes, seen_actors = [], set(), set()
    for title, group, selection in definitions:
        keep = changed & (np.asarray(frame.agent_type == "Vehicle") & np.asarray(frame.motion_state == "vehicle.moving")
                          if group == "MovingVehicle" else np.asarray(frame.agent_type == group))
        keep &= delta < 0 if "improvement" in selection else delta > 0
        ids = np.flatnonzero(keep)
        if len(ids) == 0:
            result.append({"Title": title, "Available": False, "Reason": "No changed-mode actor in the specified direction"})
            continue
        distinct = np.array([index for index in ids if int(index) not in seen_actors], dtype=np.int64)
        if len(distinct): ids = distinct
        target = float(np.median(delta[ids])) if selection.startswith("median") else float(np.min(delta[ids]) if "improvement" in selection else np.max(delta[ids]))
        ordered = sorted(ids, key=lambda index: (abs(float(delta[index]) - target), keys.iloc[index]))
        # Representative cases favor a different displayed scene, after defining
        # the median target. Tail cases preserve the exact largest-gain/harm rule.
        if selection.startswith("median"):
            different = [index for index in ordered if frame.iloc[index].scene_token not in seen_scenes]
            if different: ordered = different
        index = int(ordered[0]); row = frame.iloc[index]
        seen_actors.add(index); seen_scenes.add(row.scene_token)
        result.append({"Title": title, "Available": True, "Selection": selection,
                       "RowIndex": index, "SceneToken": row.scene_token, "SampleToken": row.sample_token,
                       "InstanceToken": row.instance_token, "Fold": int(row.Fold),
                       "SourceIndex": int(row.source_index), "DatasetIndex": int(row.dataset_index),
                       "NodeInGraph": int(row.node_in_graph), "Group": group,
                       "Comparison": "G-C-NG-C", "DeltaTop1FDE": float(delta[index]),
                       "GroupDirectionalMedianOrExtreme": target,
                       "GroupDirectionalChangedCount": int(keep.sum())})
    return result


def case_windows(cases):
    needed = {(case["SceneToken"], case["SampleToken"]) for case in cases if case["Available"]}
    selected = {}
    if not needed: return selected
    manifest = read_json(S6 / "01_cache/stage6a_cache_manifest.json")
    for record in manifest["batches"]:
        if record["split"] != "train": continue
        wanted_scenes = {key[0] for key in needed - selected.keys()}
        if "scene_tokens" in record and not wanted_scenes.intersection(record["scene_tokens"]): continue
        path = S6 / record["path"]
        assert sha256(path) == record["sha256"]
        block = torch.load(path, map_location="cpu", weights_only=False)
        for window in block["windows"]:
            key = (window["scene_token"], window["sample_token"])
            if key in needed:
                assert key not in selected
                selected[key] = window
        if len(selected) == len(needed): break
    assert set(selected) == needed
    return selected


def case_figure(audit, output):
    cache = ROOT / "05_evaluation/cache"
    actors = pd.read_csv(cache / "stage14a_oof_actor_records.csv", dtype={"future_mask_bits": str})
    metrics = np.load(cache / "stage14a_oof_metrics.npy", mmap_mode="r")
    probabilities = np.load(cache / "stage14a_oof_probabilities.npy", mmap_mode="r")
    candidates_path = S11A / "01_identity_audit/cache/candidates.npy"
    assert sha256(candidates_path) == audit["CandidateSHA256"]
    candidates = np.load(candidates_path, mmap_mode="r")
    gt = np.load(S11A / "01_identity_audit/cache/GT.npy", mmap_mode="r")
    cases = choose_cases(actors, metrics, probabilities)
    windows = case_windows(cases)
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 10.2))
    fig.subplots_adjust(left=.09, right=.98, bottom=.2, top=.94, wspace=.28, hspace=.5)
    for index, (case, ax) in enumerate(zip(cases, axes.flat)):
        if not case["Available"]:
            ax.axis("off"); ax.text(.5, .5, f"{case['Title']}\nUnavailable: {case['Reason']}",
                                      ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue
        row = actors.iloc[case["RowIndex"]]
        source = case["SourceIndex"]
        window = windows[(case["SceneToken"], case["SampleToken"])]
        node = case["NodeInGraph"]
        assert int(window["dataset_index"]) == case["DatasetIndex"]
        assert window["instance_tokens"][node] == case["InstanceToken"]
        assert np.array_equal(candidates[source], window["ego_prediction"][node].numpy())
        assert np.array_equal(gt[source], window["GT"][node].numpy())
        current = window["current_position"][node].numpy()
        history_mask = ~window["history_padding"][node].numpy()
        history = window["history"][node].numpy()[history_mask]
        candidate = np.asarray(candidates[source])
        groundtruth = np.asarray(gt[source])
        assert np.isfinite(candidate).all() and np.isfinite(groundtruth).all() and len(history) >= 2
        top_modes = probabilities[case["RowIndex"]].argmax(-1)
        endpoints = torch.from_numpy(np.array(candidate[:, -1], copy=True))
        endpoint_gt = window["GT"][node, -1]
        endpoint_fde = (endpoints - endpoint_gt).norm(dim=-1).numpy()
        for model_index in range(4):
            assert float(endpoint_fde[top_modes[model_index]]) == float(metrics[case["RowIndex"], model_index, 0])
        centered = np.concatenate([candidate.reshape(-1, 2), groundtruth, history]) - current
        lower, upper = centered.min(0), centered.max(0)
        padding = max(float((upper - lower).max()) * .1, 1.5)
        position = window["current_position"].numpy() - current
        valid_current = ~window["history_padding"][:, 4].numpy()
        visible = valid_current & (position[:, 0] >= lower[0] - padding) & (position[:, 0] <= upper[0] + padding)
        visible &= (position[:, 1] >= lower[1] - padding) & (position[:, 1] <= upper[1] + padding)
        ax.scatter(position[visible, 0], position[visible, 1], s=9, color="#D1D1D1", zorder=1)
        for mode in range(6):
            path = np.concatenate([current[None], candidate[mode]]) - current
            ax.plot(path[:, 0], path[:, 1], color="#BDBDBD", lw=.8, zorder=1, alpha=.7)
            ax.scatter(path[-1, 0], path[-1, 1], s=10, facecolor="white", edgecolor="#A0A0A0", linewidth=.5)
        for model_index, model in enumerate(MODELS):
            path = np.concatenate([current[None], candidate[top_modes[model_index]]]) - current
            ax.plot(path[:, 0], path[:, 1], color=COLORS[model], ls="--" if model.endswith("A") else "-",
                    lw=1.5, alpha=.9, zorder=2)
        truth = np.concatenate([current[None], groundtruth]) - current
        history_centered = history - current
        ax.plot(truth[:, 0], truth[:, 1], color="#222222", lw=1.6, zorder=3)
        ax.plot(history_centered[:, 0], history_centered[:, 1], color="#222222", ls=":", lw=1.3, zorder=3)
        ax.scatter(0, 0, s=20, color="#222222", zorder=4)
        display_center = (lower + upper) / 2
        half_span = float((upper - lower).max()) / 2 + padding
        ax.set_xlim(display_center[0] - half_span, display_center[0] + half_span)
        ax.set_ylim(display_center[1] - half_span, display_center[1] + half_span)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Ego-frame x, centered at target t0 (m)", fontsize=6.5)
        ax.set_ylabel("Ego-frame y (m)", fontsize=6.5)
        ax.set_title(f"{'abcdef'[index]}  {case['Title']}\nFold{case['Fold']} | Δ(G-C − NG-C)={case['DeltaTop1FDE']:+.4f} m",
                     fontsize=7.5, loc="left")
        annotation = "\n".join(f"{model}: mode{int(top_modes[j]) + 1}, {metrics[case['RowIndex'], j, 0]:.3f} m"
                                for j, model in enumerate(MODELS))
        normalized = (centered - (display_center - half_span)) / (2 * half_span)
        left_density = int(((normalized[:, 0] < .52) & (normalized[:, 1] > .75)).sum())
        right_density = int(((normalized[:, 0] > .48) & (normalized[:, 1] > .75)).sum())
        annotation_right = right_density < left_density
        ax.text(.985 if annotation_right else .015, .985, annotation, transform=ax.transAxes,
                ha="right" if annotation_right else "left", va="top", fontsize=6,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none", "pad": 1.8})
        case.update(CandidateSHA256=array_sha(candidate), GTSHA256=array_sha(groundtruth),
                    HistorySHA256=array_sha(history), Top1ModesOneBased=[int(value) + 1 for value in top_modes],
                    Top1FDEByModel={model: float(metrics[case["RowIndex"], j, 0]) for j, model in enumerate(MODELS)},
                    FigureDisplayTranslation=current.tolist(), CoordinateFrame="original t0 ego orientation, target-centered display only",
                    CandidateGeometryUnchanged=True, GTUsedForEvaluationAndCaseSelectionOnly=True)
    legend = [Line2D([], [], color=COLORS[model], ls="--" if model.endswith("A") else "-", lw=1.5, label=model)
              for model in MODELS]
    legend += [Line2D([], [], color="#222222", label="GT (evaluation only)"),
               Line2D([], [], color="#222222", ls=":", label="Observed history"),
               Line2D([], [], color="#BDBDBD", label="All six frozen candidates")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .12), ncol=4, fontsize=6.6)
    count = len(actors)
    fig.text(.02, .093, f"{DATASET} | n={count:,}; {sum(case['Available'] for case in cases)} displayed actor cases", fontsize=6.8)
    fig.text(.02, .072, "Representative median-direction cases plus moving-vehicle extremes, selected after freezing; examples do not establish aggregate evidence.", fontsize=6.2)
    fig.text(.02, .051, "5 observed / 12 future frames; positions and FDE in m. Same frozen six candidates for every model.", fontsize=6.1)
    fig.text(.02, .030, LIMIT, fontsize=6.1)
    atomic_json(FIGURES / "stage14a_case_manifest.json", {
        "Status": "PASS", "Comparison": "G-C-NG-C", "Cases": cases,
        "SelectionContract": read_json(FIGURES / "stage14a_figure_contract.json")["CaseSelection"],
        "AllModelsUseUnchangedGeometry": True, "RerankerOrPredictorInferencePerformed": False,
        "NoTrainingOrModelSelectionUse": True,
    })
    export(fig, "stage14a_fig6_improvement_failure_cases", "ImprovementAndFailureCases", output,
           {"AvailableCases": sum(case["Available"] for case in cases)})


def main():
    verify(history=True)
    audit = gate()
    inputs_before = {path: sha256(ROOT / path) for path in SOURCES}
    metrics, folds, intervals, means, switches = [pd.read_csv(ROOT / path) for path in SOURCES]
    for table in (metrics, folds, means):
        assert set(table.Model) == set(MODELS)
    assert set(intervals.Comparison) == set(COMPARISONS)
    style()
    output = []
    overall_figure(means, intervals, output)
    type_figure(means, output)
    moving_figure(means, intervals, output)
    ablation_figure(metrics, intervals, output)
    switch_figure(switches, output)
    case_figure(audit, output)
    assert len(output) == 6
    assert inputs_before == {path: sha256(ROOT / path) for path in SOURCES}
    atomic_json(FIGURES / "stage14a_figure_manifest.json", {
        "Status": "PASS", "Backend": "Python / matplotlib", "Families": output,
        "FamilyCount": 6, "SVGEditableText": True, "PDFFontType": 42,
        "UnitsAndSampleCountsShown": True, "InternalOOFLabelShown": True,
        "IndependentOfficialTestClaim": False, "BicycleGraphImprovementClaim": False,
        "FigureContractSHA256": sha256(FIGURES / "stage14a_figure_contract.json"),
        "SourceDataSHA256": inputs_before,
        "FrozenGateSHA256": sha256(ROOT / "04_checkpoints/stage14a_all_frozen.json"),
        "IdentityAuditSHA256": sha256(ROOT / "05_evaluation/stage14a_identity_audit.json"),
        "NoFitOrNewInference": True, "ScientificDecisionMadeByPlottingCode": False,
        "VisualReview": "Review the six PNG previews before final delivery; source/format checks do not substitute for visual QA.",
    })
    verify(history=True)
    print("STAGE14A_SIX_FIGURE_FAMILIES_EXPORTED", flush=True)


if __name__ == "__main__":
    main()
