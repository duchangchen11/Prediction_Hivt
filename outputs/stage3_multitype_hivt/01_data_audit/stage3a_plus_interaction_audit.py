"""Exact t0 proximity statistics from immutable scene graphs; no model or trajectory changes."""
from collections import Counter
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3a_plus_common import CLASSES, FREEZE, read_json, atomic_json, write_csv, sha256, verify_freeze
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RADII = (10, 20, 30, 50)
PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))


def median_from_histogram(hist):
    n = sum(hist.values()); positions = ((n - 1) // 2, n // 2)
    values = []; cumulative = 0
    for value, count in sorted(hist.items()):
        values.extend(value for p in positions if cumulative <= p < cumulative + count)
        cumulative += count
    assert len(values) == 2
    return sum(values) / 2


def main():
    freeze = verify_freeze(); prep = read_json(ROOT / '02_preprocessed/stage3_preprocess_manifest.json')
    hist = {(s, t, n, r): Counter() for s in ('train', 'val') for t in range(3) for n in range(4) for r in RADII}
    counts = {(s, r, p): np.zeros(4, dtype=np.int64) for s in ('train', 'val') for r in RADII for p in PAIRS}
    distance_hist = {s: np.zeros(50, dtype=np.int64) for s in ('train', 'val')}
    graph_counts = {s: 0 for s in ('train', 'val')}; empty_counts = graph_counts.copy()
    no_target_graph_counts = graph_counts.copy()
    for si, shard in enumerate(prep['shards'], start=1):
        path = ROOT / shard['relative_path']; split = shard['split']
        assert sha256(path) == freeze['scene_shards'][shard['relative_path']]
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['input_signature'] == prep['input_signature']
        for graph in saved['graphs']:
            if graph is None:
                empty_counts[split] += 1; continue
            graph_counts[split] += 1
            assert graph.history_mask[:, -1].all()
            xy = graph.positions[:, 4].numpy().astype(np.float64)
            types = graph.agent_type.numpy(); targets = graph.target_mask.numpy()
            no_target_graph_counts[split] += int(not targets.any())
            assert np.isfinite(xy).all() and set(np.unique(types)).issubset({0, 1, 2})
            distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
            np.fill_diagonal(distance, np.inf)
            directed = distance[targets]
            distance_hist[split] += np.histogram(directed[directed <= 50], bins=np.arange(51))[0]
            for radius in RADII:
                near = distance <= radius
                for target_type in range(3):
                    selected = targets & (types == target_type)
                    for neighbor_type in range(4):
                        neighbor_mask = types == neighbor_type if neighbor_type < 3 else np.ones(len(types), dtype=bool)
                        n = near[selected][:, neighbor_mask].sum(1)
                        vals, frequency = np.unique(n, return_counts=True)
                        hist[split, target_type, neighbor_type, radius].update(dict(zip(vals.tolist(), frequency.tolist())))
                i, j = np.triu_indices(len(types), k=1)
                inside = distance[i, j] <= radius
                incident = targets[i] | targets[j]
                low, high = np.minimum(types[i], types[j]), np.maximum(types[i], types[j])
                for pair in PAIRS:
                    context_pairs = inside & (low == pair[0]) & (high == pair[1])
                    target_pairs = context_pairs & incident
                    counts[split, radius, pair] += (target_pairs.sum(), int(target_pairs.any()), context_pairs.sum(), int(context_pairs.any()))
        del saved
        if si % 50 == 0: print('INTERACTION_SHARDS', si, '/ 850', flush=True)
    for split in ('train', 'val'):
        assert graph_counts[split] + empty_counts[split] == prep['splits'][split]['candidate_windows']
        assert graph_counts[split] - no_target_graph_counts[split] == prep['splits'][split]['supervised_windows']
    for t in range(3):
        for n in range(4):
            for r in RADII:
                hist['all', t, n, r] = hist['train', t, n, r] + hist['val', t, n, r]
    for r in RADII:
        for p in PAIRS: counts['all', r, p] = counts['train', r, p] + counts['val', r, p]
    statistics = []
    for split in ('train', 'val', 'all'):
        for t in range(3):
            for n in range(4):
                for r in RADII:
                    h = hist[split, t, n, r]; size = sum(h.values()); total = sum(value * count for value, count in h.items())
                    assert size > 0
                    statistics.append({'Split': split, 'TargetType': CLASSES[t], 'NeighborType': CLASSES[n] if n < 3 else 'all',
                                       'Radius_m': r, 'TargetCount': size, 'TotalNeighborCount': total,
                                       'MeanNeighborCount': total / size, 'MedianNeighborCount': median_from_histogram(h),
                                       'P_target_has_ge1_neighbor': (size - h[0]) / size,
                                       'P_target_has_ge2_neighbors': (size - h[0] - h[1]) / size,
                                       'NeighborCountHistogram': {str(k): v for k, v in sorted(h.items())}})
    pair_rows = []
    for split in ('train', 'val', 'all'):
        for radius in RADII:
            for pair in PAIRS:
                c = counts[split, radius, pair]
                pair_rows.append({'Split': split, 'Radius_m': radius, 'Pair': '-'.join(CLASSES[i] for i in pair),
                                  'UniqueTargetIncidentPairs': int(c[0]), 'WindowsWithTargetIncidentPair': int(c[1]),
                                  'UniqueAllContextPairs': int(c[2]), 'WindowsWithAllContextPair': int(c[3])})
    source = {'status': 'PASS', 'scope': 'official train700/val150 all frozen candidate scene-windows; full and partial supervised targets',
              'radii_m': RADII, 'coordinate_frame': 't0 ego meters; Euclidean distance<=radius; self excluded',
              'target_definition': 'unchanged target_mask (history>=2 and future>=1)',
              'neighbor_definition': 'all current three-type context actors, including non-targets; ego is separate coordinate context',
              'pair_definition': 'unique unordered pairs within each graph; target-incident means at least one endpoint is a supervised target',
              'window_definition': 'each sample scene-window counted once for a pair type and radius; the same physical instance can occur in multiple windows',
              'interpretation': 'spatial proximity and available context, not attention weights or proof of causal interaction',
              'graph_window_counts': graph_counts, 'anchors_without_graph': empty_counts,
              'windows_with_graph_but_no_supervised_targets': no_target_graph_counts,
              'scene_shards': 850, 'all_shard_hashes_verified': True, 'neighbor_statistics': statistics, 'pair_counts': pair_rows,
              'distance_histogram': {'bin_edges_m': list(range(51)), 'directed_target_neighbor_pairs_up_to_50m': True,
                                     'train_counts': distance_hist['train'].tolist(), 'val_counts': distance_hist['val'].tolist()},
              'frozen_source_manifest_sha256': sha256(FREEZE)}
    write_csv(ROOT / '01_data_audit/stage3_interaction_density.csv', [{k: v for k, v in row.items() if k != 'NeighborCountHistogram'} for row in statistics])
    write_csv(ROOT / '01_data_audit/stage3_interaction_density_pair_counts.csv', pair_rows)
    atomic_json(ROOT / '01_data_audit/stage3_interaction_density.json', source)
    plot(source)
    verify_freeze(); print('INTERACTION_DENSITY=PASS', graph_counts, flush=True)


def plot(source):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    total = np.array(source['distance_histogram']['train_counts']) + np.array(source['distance_histogram']['val_counts'])
    axes[0].stairs(total / total.sum() * 100, np.arange(51), fill=True, color='#8D969F', alpha=.7)
    axes[0].set(xlabel='t0 neighbor distance (m)', ylabel='Directed target-neighbor pairs (%)', title='a  Neighbor distance distribution', xlim=(0, 50))
    pair_labels = ['V-V', 'V-P', 'V-B', 'P-P', 'P-B', 'B-B']; x = np.arange(6)
    for offset, radius, color in ((-.18, 20, '#8D969F'), (.18, 50, '#246C9E')):
        selected = [r for r in source['pair_counts'] if r['Split'] == 'all' and r['Radius_m'] == radius]
        values = [r['UniqueTargetIncidentPairs'] for r in selected]
        bars = axes[1].bar(x + offset, values, width=.34, color=color, label=f'{radius} m')
        axes[1].bar_label(bars, labels=[f'{v:,}' for v in values], rotation=90, padding=3, fontsize=7)
    axes[1].set_xticks(x, pair_labels); axes[1].set_yscale('log'); axes[1].set_ylim(top=max(r['UniqueTargetIncidentPairs'] for r in source['pair_counts']) * 25)
    axes[1].set(ylabel='Unique target-incident pairs (log scale)', title='b  Interaction pair counts'); axes[1].legend(frameon=False)
    for name, color in zip(CLASSES, ('#737B84', '#4A8A77', '#C87B24')):
        rows = [r for r in source['neighbor_statistics'] if r['Split'] == 'all' and r['TargetType'] == name and r['NeighborType'] == 'all']
        axes[2].plot([r['Radius_m'] for r in rows], [r['MeanNeighborCount'] for r in rows], 'o-', color=color, label=name)
    axes[2].set(xlabel='Radius (m)', ylabel='Mean context neighbors per target', title='c  Mean neighbors by target type')
    axes[2].set_xticks(RADII); axes[2].legend(frameon=False)
    for ax in axes: ax.grid(axis='y', alpha=.15)
    fig.suptitle('Frozen train + VAL scene-windows: spatial proximity at t0', fontsize=12)
    stem = ROOT / '05_figures/stage3_interaction_density'
    for ext in ('png', 'pdf', 'svg'): fig.savefig(str(stem) + '.' + ext, dpi=300)
    svg = Path(str(stem) + '.svg'); svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    torch.set_num_threads(2)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    main()
