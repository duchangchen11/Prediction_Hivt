"""Unsmoothed Stage4A scientific figures from locked real CSV/JSON sources."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE3_ROOT = ROOT.parent / 'stage3_multitype_hivt'
sys.path.insert(0, str(STAGE3_ROOT / '00_manifest'))
sys.path.insert(0, str(STAGE3_ROOT / '04_evaluation'))
sys.path.insert(0, str(ROOT / '00_manifest'))
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3b_common import atomic_json, read_json, sha256, write_csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = ('#73808A', '#2478A5')
METHODS = ('Stage3B Type Embedding', 'Stage4A Type Interaction')
CURVES = (STAGE3_ROOT / '03_type_embedding/stage3b_training_curve.csv',
          ROOT / '03_type_interaction/stage4a_training_curve.csv')
SUMMARIES = (STAGE3_ROOT / '03_type_embedding/stage3b_training_summary.json',
             ROOT / '03_type_interaction/stage4a_training_summary.json')
ABLATION = ROOT / '06_tables/stage4a_type_interaction_ablation.csv'


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def export(fig, stem, sources, extra=None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = list(fig.texts)
    for ax in fig.axes:
        artists.extend(list(ax.texts) + [ax.xaxis.label, ax.yaxis.label, ax.title])
    for artist in artists:
        if not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, 'Text clipped'
    hashes = {}
    for suffix in ('png', 'pdf', 'svg'):
        path = Path(str(stem) + '.' + suffix)
        fig.savefig(path, dpi=300, facecolor='white')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        hashes[str(path.relative_to(ROOT))] = sha256(path)
    plt.close(fig)
    atomic_json(Path(str(stem) + '_audit.json'), {
        'status': 'PASS', 'source_SHA256': {__import__('os').path.relpath(p, ROOT): sha256(p) for p in sources},
        'exports_SHA256': hashes, 'smoothing': False, 'fabricated_data': False,
        'text_inside_canvas': True, 'backend': 'python', **(extra or {})})


def checked_curve(path, summary_path):
    values = rows(path)
    summary = read_json(summary_path)
    steps = [int(float(row['global_step'])) for row in values]
    assert steps == list(range(500, int(summary['final_executed_global_step']) + 1, 500))
    assert all(int(float(row['VAL_overall_count'])) == 54990 for row in values)
    assert all(np.isfinite(float(row['VAL_' + group + '_FDE']))
               for row in values for group in ('overall', 'vehicle', 'pedestrian', 'bicycle', 'vehicle.moving'))
    best = int(summary['best_global_step'])
    point = next(row for row in values if int(float(row['global_step'])) == best)
    fde = float(summary['selected_full_horizon_metrics']['overall']['minFDE6'])
    assert abs(float(point['VAL_overall_FDE']) - fde) < 1e-10
    return values, summary, best, fde


def plot_training():
    histories = [checked_curve(path, summary) for path, summary in zip(CURVES, SUMMARIES)]
    source = ROOT / '05_figures/stage4a_training_comparison_source.csv'
    write_csv(source, [{'Method': method, 'global_step': int(float(row['global_step'])),
                        'overall_minFDE6': float(row['VAL_overall_FDE']), 'phase': row['phase']}
                       for method, (values, _, _, _) in zip(METHODS, histories) for row in values])
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout='constrained')
    for method, color, (values, _, best, fde) in zip(METHODS, COLORS, histories):
        ax.plot([int(float(row['global_step'])) for row in values],
                [float(row['VAL_overall_FDE']) for row in values],
                marker='o', markersize=3, lw=1.1, color=color, label=method)
        ax.scatter([best], [fde], s=80, marker='*', color=color, edgecolor='white', linewidth=.35, zorder=6)
    ax.axvline(5000, color='#A4ABB0', lw=.7, ls=':')
    ax.set(xlabel='Global optimizer step', ylabel='Full-horizon overall minFDE6 (m)',
           xlim=(0, max(int(float(h[0][-1]['global_step'])) for h in histories) + 500))
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, .80), fontsize=8)
    ax.grid(axis='y', alpha=.15)
    ax.text(.98, .95, '\n'.join(f'{method.split()[0]} best: {best} / {fde:.6f} m'
            for method, (_, _, best, fde) in zip(METHODS, histories)), transform=ax.transAxes,
            ha='right', va='top', fontsize=8, bbox={'facecolor': 'white', 'alpha': .94, 'edgecolor': 'none'})
    export(fig, ROOT / '05_figures/stage4a_training_comparison', list(CURVES) + list(SUMMARIES) + [source],
           {'actual_markers_every500_steps': True, 'official_VAL_scenes': 150,
            'best_markers': [{'method': method, 'global_step': best, 'overall_minFDE6': fde}
                             for method, (_, _, best, fde) in zip(METHODS, histories)]})
    values = histories[1][0]
    steps = [int(float(row['global_step'])) for row in values]
    fig, ax = plt.subplots(figsize=(7.4, 3.8), layout='constrained')
    for key, label, color in zip(('train_loss', 'regression_loss', 'classification_loss'),
                                 ('Total', 'Regression', 'Classification'), ('#2478A5', '#62717B', '#BA8549')):
        y = [float(row[key]) for row in values]
        assert np.isfinite(y).all()
        ax.plot(steps, y, marker='o', ms=3, lw=1, color=color, label=label)
    ax.axvline(5000, color='#A4ABB0', ls=':', lw=.8)
    ax.set(xlabel='Global optimizer step', ylabel='Mean batch loss over preceding500 updates')
    ax.legend(frameon=False)
    export(fig, ROOT / '03_type_interaction/stage4a_loss_curve', [CURVES[1]],
           {'point_definition': 'Mean of preceding500 actual optimizer losses; no extra smoothing.'})
    fig, ax = plt.subplots(figsize=(7.4, 3.8), layout='constrained')
    for group, color in zip(('overall', 'vehicle', 'pedestrian', 'bicycle', 'vehicle.moving'),
                            ('#2478A5', '#62717B', '#569580', '#BA8549', '#8B6B94')):
        ax.plot(steps, [float(row['VAL_' + group + '_FDE']) for row in values],
                marker='o', ms=3, lw=1, color=color, label=group)
    ax.axvline(5000, color='#A4ABB0', ls=':', lw=.8)
    ax.set(xlabel='Global optimizer step', ylabel='Full-horizon minFDE6 (m)')
    ax.legend(frameon=False, ncol=2)
    export(fig, ROOT / '03_type_interaction/stage4a_val_fde_curve', [CURVES[1]],
           {'actual_markers_every500_steps': True, 'official_VAL_scenes': 150})


def grouped_bars(groups, stem, display_labels=None):
    data = {row['Group']: row for row in rows(ABLATION)}
    fig, ax = plt.subplots(figsize=(7.4, 4.2), layout='constrained')
    x = np.arange(len(groups)); maximum = 0
    source_rows = []
    for i, (method, color, prefix) in enumerate(zip(METHODS, COLORS, ('B', 'C'))):
        values = [float(data[group][prefix + '_minFDE']) for group in groups]
        assert np.isfinite(values).all()
        maximum = max(maximum, max(values))
        bars = ax.bar(x + (i - .5) * .40, values, .29, color=color, label=method)
        ax.bar_label(bars, labels=[f'{value:.3f}' for value in values], padding=3, fontsize=7.5)
        source_rows.extend({'Group': group, 'Method': method, 'Count': int(data[group]['Count']), 'minFDE6': value}
                           for group, value in zip(groups, values))
    source = ROOT / '05_figures' / (stem + '_source.csv'); write_csv(source, source_rows)
    ax.set_xticks(x, display_labels or [group.replace(' >', '\n>') for group in groups])
    ax.set_ylabel('Full-horizon minFDE6 (m)'); ax.set_ylim(0, maximum * 1.22)
    ax.legend(loc='upper left', frameon=False, ncol=2, fontsize=8)
    export(fig, ROOT / '05_figures' / stem, [ABLATION, source], {
        'center': 'Actor-window arithmetic mean', 'bar_values_exact_source': True,
        'formal_inference': 'Separate paired150-scene cluster bootstrap;1000 replicates seed2022.',
        'groups': list(groups)})


def bias_heatmap():
    path = ROOT / '06_tables/stage4a_relation_bias_statistics.csv'
    stats = rows(path); aggregate = []
    values = np.zeros((3, 3), dtype=float)
    names = ('Vehicle', 'Pedestrian', 'Bicycle')
    for pair_id in range(9):
        selected = [row for row in stats if int(row['pair_id']) == pair_id]
        assert len(selected) == 24
        assert {(int(row['layer']), int(row['head'])) for row in selected} == {(layer, head) for layer in range(3) for head in range(8)}
        counts = [int(row['count']) for row in selected]
        assert len(set(counts)) == 1 and counts[0] > 0
        mean_abs = float(np.mean([float(row['mean_abs_bias']) for row in selected]))
        target, source = divmod(pair_id, 3); values[target, source] = mean_abs
        aggregate.append({'pair_id': pair_id, 'target_type': names[target].lower(),
                          'source_type': names[source].lower(), 'directed_pair': f'{names[target][0]}<-{names[source][0]}',
                          'edge_observation_count': counts[0], 'layer_head_count': 24, 'mean_abs_bias': mean_abs,
                          'aggregation': 'Unweighted mean across24 layer/head means; each observes identical edges.'})
    assert np.isfinite(values).all()
    source = ROOT / '05_figures/stage4a_type_pair_bias_heatmap_source.csv'; write_csv(source, aggregate)
    fig, ax = plt.subplots(figsize=(5.0, 4.2), layout='constrained')
    image = ax.imshow(values, vmin=0, vmax=max(float(values.max()), 1e-8), cmap='Blues')
    ax.set_xticks(range(3), names); ax.set_yticks(range(3), names)
    ax.set(xlabel='Source type (j)', ylabel='Target type (i)', title='Directed pair mean absolute bias')
    for target in range(3):
        for src in range(3):
            ax.text(src, target, f'{values[target, src]:.4f}', ha='center', va='center',
                    color='white' if values[target, src] > .6 * values.max() else '#252B30')
    fig.colorbar(image, ax=ax, label='Mean |bias| over layers / heads', shrink=.82)
    export(fig, ROOT / '05_figures/stage4a_type_pair_bias_heatmap', [path, source], {
        'rows': 'target type', 'columns': 'source type', 'pair_id_definition': '3*target_type+source_type',
        'values': values.tolist(), 'diagnostic_only_not_causal_influence': True,
        'aggregation': 'Unweighted mean of24 layer/head mean absolute bias entries for each directed pair.'})


def main():
    assert read_json(ROOT / '00_manifest/stage4a_figure_contract.json')['backend'] == 'python'
    assert read_json(ROOT / '04_evaluation/stage4a_pairing_audit.json')['status'] == 'PASS'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white', 'path.simplify': False})
    plot_training()
    grouped_bars(('Overall', 'Vehicle', 'Pedestrian', 'Bicycle'), 'stage4a_fde_ablation')
    grouped_bars(('Overall', 'Heterogeneous-20m', 'VP-context-20m', 'Vehicle >5m', 'Pedestrian >5m'),
                 'stage4a_interaction_context_fde',
                 ('All', 'Heterogeneous\n20m', 'VP-context\n20m', 'Vehicle\n>5m', 'Pedestrian\n>5m'))
    bias_heatmap()
    print('STAGE4A_QUANTITATIVE_FIGURES=PASS', flush=True)


if __name__ == '__main__':
    main()
