"""Matched real Stage3B/Stage4A cases with fixed t0 heterogeneous membership."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE3_ROOT = ROOT.parent / 'stage3_multitype_hivt'
sys.path.insert(0, str(STAGE3_ROOT / '00_manifest'))
sys.path.insert(0, str(STAGE3_ROOT / '04_evaluation'))
sys.path.insert(0, str(ROOT / '00_manifest'))
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3b_common import (CLASSES, SceneDataset, errors_with_top1, GT_fingerprint,
    atomic_json, read_json, sha256, model_new as baseline_model_new)
import numpy as np
import torch
from torch_geometric.data import Batch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from PIL import Image

BASE_CHECKPOINT = STAGE3_ROOT / '07_checkpoints/stage3b_best_overall_minfde.pt'
CHECKPOINT = ROOT / '07_checkpoints/stage4a_best_overall_minfde.pt'
BASE_ACTORS = STAGE3_ROOT / '04_evaluation/stage3b_type_embedding_actor_errors.csv'
ACTORS = ROOT / '04_evaluation/stage4a_actor_errors.csv'
MEMBERSHIP = ROOT / '04_evaluation/stage4a_interaction_membership.csv'
SUMMARY = ROOT / '03_type_interaction/stage4a_training_summary.json'
FIELDS = ('minADE6', 'minFDE6', 'Top1ADE6', 'Top1FDE6')
FROZEN_BASE_SHA = '461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547'


def actor_key(row):
    return tuple(row[key] for key in ('scene_token', 'sample_token', 'instance_token', 'horizon'))


def read_full(path):
    with path.open() as handle:
        rows = [row for row in csv.DictReader(handle) if row['horizon'] == 'full_horizon']
    values = {actor_key(row): row for row in rows}
    assert len(values) == len(rows) == 54990
    return values


def enabled(row, group):
    value = row[group]
    assert value in ('0', '1', '0.0', '1.0', 'False', 'True')
    return value in ('1', '1.0', 'True')


def case_candidates(base, interaction, membership):
    assert base.keys() == interaction.keys() == membership.keys()
    all_cases = []
    for key, row in interaction.items():
        old = base[key]; context = membership[key]
        for field in ('node_in_graph', 'agent_type', 'motion_state', 'future_mask_bits', 'GT_trajectory_sha256'):
            assert row[field] == old[field]
        assert row['node_in_graph'] == context['node_in_graph'] and row['agent_type'] == context['agent_type']
        all_cases.append((float(row['minFDE6']) - float(old['minFDE6']), key, row, old,
                          float(row['GT_endpoint_displacement_m']), context))
    selected = []; used = set(); omitted = []
    for cls in ('vehicle', 'pedestrian'):
        candidates = [item for item in all_cases if item[2]['agent_type'] == cls and item[0] < 0
                      and enabled(item[5], 'Heterogeneous-20m')]
        moving = [item for item in candidates if item[4] > 5]
        if not candidates:
            omitted.append({'kind': cls + '_heterogeneous_improvement',
                            'reason': 'No negative-delta full-horizon target in fixed Heterogeneous-20m; no case fabricated.'})
            continue
        choice = min(moving or candidates, key=lambda item: (item[0], item[1]))
        selected.append((f'stage4a_{cls}_improvement_case_001', choice, 'Heterogeneous-20m improvement'))
        used.add(choice[1])
    candidates = [item for item in all_cases if item[0] > 0 and item[1] not in used]
    moving = [item for item in candidates if item[4] > 5]
    if candidates:
        choice = max(moving or candidates, key=lambda item: (item[0], item[1]))
        selected.append(('stage4a_degradation_case_001', choice, 'Degradation')); used.add(choice[1])
    else:
        omitted.append({'kind': 'degradation', 'reason': 'No positive-delta actor; no case fabricated.'})
    candidates = [item for item in all_cases if enabled(item[5], 'VP-context-20m') and item[1] not in used]
    improved = [item for item in candidates if item[0] < 0]
    moving_improved = [item for item in improved if item[4] > 5]
    if candidates:
        choice = min(moving_improved or improved or candidates, key=lambda item: (item[0], item[1]))
        selected.append(('stage4a_vp_context_case_001', choice, 'V-P spatial context'))
    else:
        omitted.append({'kind': 'VP-context-20m', 'reason': 'No distinct eligible full-horizon V/P actor.'})
    assert len(selected) >= 3, 'Required heterogeneous improvement/degradation examples absent; report actual absence.'
    return selected, omitted


def nearest_context(graph, node, vp_only=False):
    """Context eligibility uses t0 and type only, never future trajectories."""
    own_type = int(graph.agent_type[node])
    assert not bool(graph.padding_mask[node, 4])
    candidates = ~graph.padding_mask[:, 4] & (graph.agent_type != own_type)
    if vp_only:
        assert own_type in (0, 1)
        candidates &= graph.agent_type == 1 - own_type
    indices = torch.where(candidates)[0].tolist(); eligible = []
    for other in indices:
        distance = float(torch.linalg.vector_norm(graph.positions[node, 4].double() - graph.positions[other, 4].double()))
        if distance <= 20:
            eligible.append((distance, other))
    if not eligible:
        return None
    distance, other = min(eligible)
    return {'neighbor_node': other, 'neighbor_instance_token': graph.instance_tokens[other],
            'neighbor_agent_type': CLASSES[int(graph.agent_type[other])], 't0_distance_m': distance,
            'neighbor_history_trajectory_m': graph.positions[other, :5].tolist(),
            'neighbor_history_mask': graph.history_mask[other].tolist(),
            'neighbor_GT_trajectory_m': graph.positions[other, 5:].tolist(),
            'neighbor_future_mask': graph.future_mask[other].tolist(),
            'eligibility': 'Different type and t0 euclidean distance<=20m; no future/GT/prediction used.',
            'interpretation': 't0 spatial interaction context; no causal interaction inferred.'}


@torch.no_grad()
def replay(ds, index, node, model, batch_size):
    start = index // batch_size * batch_size
    graphs = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
    graph = graphs[index - start]
    data = Batch.from_data_list(graphs).cuda(); output_node = int(data.ptr[index - start]) + node
    out = model(data.clone()); prediction, errors = errors_with_top1(model, out, data)
    modes = prediction[output_node].cpu().numpy(); gt = graph.positions[node, 5:].numpy()
    best = int(errors['best_mode'][output_node]); top = int(errors['top1_mode'][output_node])
    best_error = np.linalg.norm(modes[best].astype(np.float64) - gt, axis=-1)
    top_error = np.linalg.norm(modes[top].astype(np.float64) - gt, axis=-1)
    metrics = dict(zip(FIELDS, map(float, (best_error.mean(), best_error[-1], top_error.mean(), top_error[-1]))))
    assert modes.shape == (6, 12, 2) and np.isfinite(modes).all()
    assert torch.isfinite(out['mode_prob']).all() and torch.isfinite(out['raw_prediction']).all()
    return {'all_mode_trajectories_m': modes.tolist(), 'mode_probabilities': out['mode_prob'][output_node].cpu().tolist(),
            'best_FDE_mode_zero_based': best, 'top1_mode_zero_based': top, 'recomputed_metrics': metrics,
            'reproduced_VAL_batch_start_index': start, 'reproduced_VAL_batch_size': len(graphs)}


def checked_replay(ds, index, row, base_row, baseline, interaction, batch_size):
    node = int(row['node_in_graph']); graph = ds[index]
    assert graph.instance_tokens[node] == row['instance_token'] and graph.future_mask[node].all()
    assert int(graph.agent_type[node]) == CLASSES.index(row['agent_type'])
    fingerprint = GT_fingerprint(graph.positions[node, 5:], graph.future_mask[node], graph.agent_type[node])
    assert row['GT_trajectory_sha256'] == base_row['GT_trajectory_sha256'] == fingerprint['GT_trajectory_sha256']
    assert row['future_mask_bits'] == base_row['future_mask_bits'] == fingerprint['future_mask_bits'] == '111111111111'
    panels = []
    for method, model, source in (('Stage3B', baseline, base_row), ('Stage4A', interaction, row)):
        panel = replay(ds, index, node, model, batch_size)
        differences = {key: abs(value - float(source[key])) for key, value in panel['recomputed_metrics'].items()}
        assert max(differences.values()) < 1e-4, (method, differences)
        assert panel['best_FDE_mode_zero_based'] == int(source['best_mode'])
        assert panel['top1_mode_zero_based'] == int(source['top1_mode'])
        panel.update({'method': method, 'source_metrics': {key: float(source[key]) for key in FIELDS},
                      'metric_absolute_differences_m': differences, 'metric_tolerance_m': 1e-4})
        panels.append(panel)
    return graph, node, panels, fingerprint


def geometry(source):
    history = np.array(source['history_trajectory_m']); mask = np.array(source['history_mask'], dtype=bool)
    gt = np.array(source['GT_trajectory_m']); visible = [history[mask], gt]; late = [gt[-6:]]
    overlap = False
    for panel in source['panels']:
        modes = np.array(panel['all_mode_trajectories_m'])
        best = modes[panel['best_FDE_mode_zero_based']]; top = modes[panel['top1_mode_zero_based']]
        visible.extend((best, top)); late.extend((best[-6:], top[-6:]))
        overlap |= any(np.mean(np.linalg.norm(a - b, axis=-1)) < 1
                       for a, b in ((gt, best), (gt, top), (best, top)))
    context = source['t0_spatial_context']
    if context:
        visible.extend((np.array(context['neighbor_history_trajectory_m'])[np.array(context['neighbor_history_mask'], dtype=bool)],
                        np.array(context['neighbor_GT_trajectory_m'])[np.array(context['neighbor_future_mask'], dtype=bool)]))
    vectors = np.diff(gt, axis=0); valid = np.linalg.norm(vectors, axis=-1) > .1
    directions = np.unwrap(np.arctan2(vectors[valid, 1], vectors[valid, 0]))
    turn = len(directions) > 1 and float(np.ptp(directions)) > np.pi / 6
    visible = np.concatenate(visible); lower = visible.min(0); upper = visible.max(0)
    margin = max(1., float(np.ptp(visible, axis=0).max()) * .06)
    width = max(float(upper[0] - lower[0]) + 2 * margin, 8.)
    height = max(float(upper[1] - lower[1]) + 2 * margin, width * .26, 3.)
    center = (lower + upper) / 2; low = center - np.array([width, height]) / 2; high = center + np.array([width, height]) / 2
    late = np.concatenate(late); zoom_margin = max(.5, float(np.ptp(late, axis=0).max()) * .08)
    return low, high, late.min(0) - zoom_margin, late.max(0) + zoom_margin, {
        'GT_turn_over30deg': bool(turn), 'hard_overlap_mean_distance_under1m': bool(overlap),
        'add_inset': bool(turn or overlap), 'bbox_uses_actual_last6_target_future_points_only': True}


def trajectories(ax, source, panel, segments, history=True, late_only=False):
    ax.add_collection(LineCollection(segments, colors='#AAB3B9', linewidths=.5, alpha=.22, zorder=1))
    if history:
        h = np.array(source['history_trajectory_m']); mask = np.array(source['history_mask'], dtype=bool); h[~mask] = np.nan
        ax.plot(h[:, 0], h[:, 1], 'o-', color='#7D858B', lw=1.5, ms=4, zorder=3, label='History')
        context = source['t0_spatial_context']
        if context:
            nh = np.array(context['neighbor_history_trajectory_m']); nh[~np.array(context['neighbor_history_mask'], dtype=bool)] = np.nan
            ng = np.array(context['neighbor_GT_trajectory_m']); ng[~np.array(context['neighbor_future_mask'], dtype=bool)] = np.nan
            ax.plot(nh[:, 0], nh[:, 1], 'o-', color='#ABBFAE', lw=.9, ms=2.5, alpha=.65, zorder=2, label='Neighbor history')
            ax.plot(ng[:, 0], ng[:, 1], 's-', color='#91AF98', lw=1., ms=2.5, alpha=.55, zorder=2, label='Neighbor GT')
    gt = np.array(source['GT_trajectory_m']); modes = np.array(panel['all_mode_trajectories_m'])
    best = modes[panel['best_FDE_mode_zero_based']]; top = modes[panel['top1_mode_zero_based']]; records = []
    for values, label, color, marker, style, linewidth, zorder, size in (
        (gt, 'GT', '#111111', 's', '-', 2.5, 10, 5),
        (top, 'Top1', '#C87B24', '^', '--', 2., 9, 6),
        (best, 'Best-FDE', '#2478A5', 'o', '-', 2.5, 11, 4.5)):
        plotted = values[-6:] if late_only else values
        line, = ax.plot(plotted[:, 0], plotted[:, 1], color=color, marker=marker, ls=style,
                        lw=linewidth, zorder=zorder, ms=size, markerfacecolor='white' if label != 'Best-FDE' else color,
                        markeredgewidth=.9, label=label)
        records.append({'label': label, 'marker_count': len(line.get_xdata()), 'marker': marker,
                        'linewidth': linewidth, 'zorder': zorder})
    ax.set_aspect('equal', adjustable='box'); ax.grid(alpha=.12, lw=.5)
    return records


def draw(source, stem):
    gt = np.array(source['GT_trajectory_m'], dtype=float)
    for panel in source['panels']:
        modes = np.array(panel['all_mode_trajectories_m'], dtype=float)
        best_error = np.linalg.norm(modes[panel['best_FDE_mode_zero_based']] - gt, axis=-1)
        top_error = np.linalg.norm(modes[panel['top1_mode_zero_based']] - gt, axis=-1)
        actual = dict(zip(FIELDS, map(float, (best_error.mean(), best_error[-1], top_error.mean(), top_error[-1]))))
        assert max(abs(actual[key] - panel['source_metrics'][key]) for key in FIELDS) < 1e-4
    low, high, zoom_low, zoom_high, inset_rule = geometry(source)
    # Expand the data-derived zoom box to the physical inset aspect ratio.
    # A nearly vertical trajectory otherwise collapses equal-aspect axes to
    # a thin strip with overlapping tick labels. Coordinates stay unchanged.
    zoom_center = (zoom_low + zoom_high) / 2
    zoom_span = zoom_high - zoom_low
    zoom_aspect = .17 * 9.5 / .63
    zoom_width = max(float(zoom_span[0]), float(zoom_span[1]) * zoom_aspect)
    zoom_height = max(float(zoom_span[1]), float(zoom_span[0]) / zoom_aspect)
    zoom_low = zoom_center - np.array([zoom_width, zoom_height]) / 2
    zoom_high = zoom_center + np.array([zoom_width, zoom_height]) / 2
    inset_rule['data_bbox_expanded_to_inset_display_aspect'] = True
    starts = np.array(source['lane_positions_m']).reshape(-1, 2); vectors = np.array(source['lane_vectors_m']).reshape(-1, 2)
    segments = np.stack((starts, starts + vectors), axis=1)
    segments = segments[((segments.max(1) >= low) & (segments.min(1) <= high)).all(1)]
    ratio = float((high[1] - low[1]) / (high[0] - low[0]))
    body_height = min(4.6, max(1.05, 3.8 * ratio)); body_width = body_height / ratio
    header_height = 1.8 if inset_rule['add_inset'] else 1.1; bottom = 1.1
    figure_height = body_height + bottom + header_height; fig = plt.figure(figsize=(9.5, figure_height))
    panel_starts = (.075, .57)
    axes = [fig.add_axes([left + (.4 - body_width / 9.5) / 2, bottom / figure_height,
                         body_width / 9.5, body_height / figure_height]) for left in panel_starts]
    fig.suptitle(f"{source['case_kind']} | {source['agent_type']} | GT displacement {source['GT_endpoint_displacement_m']:.2f} m",
                 fontsize=11, y=1 - .12 / figure_height)
    fig.text(.5, 1 - .40 / figure_height,
             f"{source['scene_name']} / sample {source['sample_token'][:8]} / actor {source['instance_token'][:8]}",
             ha='center', fontsize=8, color='#53616B')
    audits = []; annotations = []; body_top = bottom + body_height
    for left, ax, panel in zip(panel_starts, axes, source['panels']):
        styles = trajectories(ax, source, panel, segments); assert all(record['marker_count'] == 12 for record in styles)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), xlabel='t0 ego forward x (m)', ylabel='t0 ego left y (m)')
        ax.tick_params(labelsize=8)
        label_y = (body_top + (1.05 if inset_rule['add_inset'] else .40)) / figure_height
        fig.text(left, label_y, panel['method'], ha='left', va='top', fontsize=10)
        text = fig.text(left + .4, label_y,
                        f"Best FDE = {panel['recomputed_metrics']['minFDE6']:.2f} m\nTop1 FDE = {panel['recomputed_metrics']['Top1FDE6']:.2f} m",
                        ha='right', va='top', fontsize=8,
                        bbox={'facecolor': 'white', 'edgecolor': '#DEE3E6', 'linewidth': .5, 'pad': 3}, zorder=25)
        annotations.append((ax, text))
        if inset_rule['add_inset']:
            zoom = fig.add_axes([left + .015, (body_top + .13) / figure_height, .17, .63 / figure_height], zorder=20)
            trajectories(zoom, source, panel, segments, history=False, late_only=True)
            zoom.set(xlim=(zoom_low[0], zoom_high[0]), ylim=(zoom_low[1], zoom_high[1])); zoom.tick_params(labelsize=6)
            zoom.set_title('Last 6 future points', fontsize=7, pad=3)
            for axis in (zoom.xaxis, zoom.yaxis):
                axis.set_major_locator(MaxNLocator(nbins=2))
            for spine in zoom.spines.values():
                spine.set_visible(True); spine.set_color('#7D858B')
        audits.append({'method': panel['method'], 'future_trajectory_styles': styles,
                       'metrics_recomputed': panel['recomputed_metrics'],
                       'metric_absolute_differences_m': panel['metric_absolute_differences_m']})
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(name) for name in ('History', 'GT', 'Best-FDE', 'Top1')]
    if source['t0_spatial_context']:
        order += [labels.index(name) for name in ('Neighbor history', 'Neighbor GT')]
    fig.legend([handles[index] for index in order], [labels[index] for index in order], loc='lower center',
               bbox_to_anchor=(.5, .42 / figure_height), ncol=len(order), frameon=False, fontsize=8)
    fig.text(.5, .27 / figure_height, '12 original future observations; Best-FDE uses GT; Top1 uses highest predicted probability.',
             ha='center', fontsize=8, color='#53616B')
    if source['t0_spatial_context']:
        context = source['t0_spatial_context']
        fig.text(.5, .13 / figure_height,
                 f"Neighbor: {context['neighbor_agent_type']}, t0 distance {context['t0_distance_m']:.2f} m. t0 spatial interaction context; no causal interaction inferred.",
                 ha='center', fontsize=7.5, color='#53616B')
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    for ax, text in annotations:
        box = text.get_bbox_patch().get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.x1 and box.y1 <= fig.bbox.y1
        for line in ax.lines:
            points = ax.transData.transform(np.c_[line.get_xdata(), line.get_ydata()]); points = points[np.isfinite(points).all(1)]
            assert not ((points[:, 0] >= box.x0) & (points[:, 0] <= box.x1)
                        & (points[:, 1] >= box.y0) & (points[:, 1] <= box.y1)).any(), 'Endpoint box hides trajectory'
    exports = {}
    for suffix in ('png', 'pdf', 'svg'):
        path = Path(str(stem) + '.' + suffix); fig.savefig(path, dpi=300, facecolor='white')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        exports[str(path.relative_to(ROOT))] = sha256(path)
    plt.close(fig)
    with Image.open(str(stem) + '.png') as im:
        im.verify()
    return {'status': 'PASS', 'panels': audits, 'same_xy_limits': True, 'xlim_m': [low[0], high[0]], 'ylim_m': [low[1], high[1]],
            'equal_aspect': True, 'compact_equal_aspect_rectangle': True, 'separate_header_for_endpoint_text_and_zoom': True,
            'inset_rule': inset_rule, 'inset_bbox_m': [zoom_low.tolist(), zoom_high.tolist()] if inset_rule['add_inset'] else None,
            'neighbor_only_history_and_observed_GT': True, 'neighbor_eligibility_uses_t0_only': True,
            'lane_alpha': .22, 'lane_linewidth': .5, 'lanes_do_not_set_axes_limits': True, 'endpoint_annotation_layout': 'PASS',
            'trajectory_edits': False, 'smoothing': False, 'interpolation': False, 'exports_SHA256': exports}


@torch.no_grad()
def main():
    from stage4a_common import config, model_new
    summary = read_json(SUMMARY)
    assert summary['status'] == 'COMPLETE'
    assert read_json(ROOT / '04_evaluation/stage4a_pairing_audit.json')['status'] == 'PASS'
    base = read_full(BASE_ACTORS); interaction = read_full(ACTORS); membership = read_full(MEMBERSHIP)
    selected, omitted = case_candidates(base, interaction, membership)
    ds = SceneDataset('val'); lookup = {(row['scene_token'], row['sample_token']): index for index, row in enumerate(ds.rows)}
    assert len(lookup) == len(ds) == 3603 and config()['batch_size'] == 16
    baseline = baseline_model_new(); baseline.load_state_dict(torch.load(BASE_CHECKPOINT, map_location='cpu', weights_only=False)['state_dict']); baseline.eval()
    model = model_new(); model.load_state_dict(torch.load(CHECKPOINT, map_location='cpu', weights_only=False)['state_dict']); model.eval()
    checkpoint_sha = {'Stage3B': sha256(BASE_CHECKPOINT), 'Stage4A': sha256(CHECKPOINT)}
    assert checkpoint_sha['Stage3B'] == FROZEN_BASE_SHA and checkpoint_sha['Stage4A'] == summary['checkpoint_sha256']
    csv_sha = {'Stage3B': sha256(BASE_ACTORS), 'Stage4A': sha256(ACTORS), 'membership': sha256(MEMBERSHIP)}
    figures = []
    for name, choice, kind in selected:
        delta, key, row, base_row, displacement, member = choice; index = lookup[row['scene_token'], row['sample_token']]
        graph, node, panels, fingerprint = checked_replay(ds, index, row, base_row, baseline, model, config()['batch_size'])
        hetero = nearest_context(graph, node)
        assert bool(hetero) == enabled(member, 'Heterogeneous-20m')
        vp = nearest_context(graph, node, vp_only=True) if int(graph.agent_type[node]) in (0, 1) else None
        assert bool(vp) == enabled(member, 'VP-context-20m')
        for label, neighbor in (('heterogeneous', hetero), ('VP', vp)):
            if neighbor:
                assert neighbor['neighbor_instance_token'] == member[f'nearest_{label}_instance_token']
                assert neighbor['neighbor_node'] == int(member[f'nearest_{label}_node_in_graph'])
                assert abs(neighbor['t0_distance_m'] - float(member[f'nearest_{label}_distance_m'])) < 1e-6
        context = vp if name == 'stage4a_vp_context_case_001' else hetero
        source = {'name': name, 'case_kind': kind,
                  **{field: row[field] for field in ('scene_name', 'scene_token', 'sample_token', 'instance_token', 'agent_type', 'motion_state')},
                  'node_in_graph': node, 'GT_endpoint_displacement_m': displacement, 'delta_FDE_Stage4A_minus_Stage3B_m': delta,
                  'history_trajectory_m': graph.positions[node, :5].tolist(), 'history_mask': graph.history_mask[node].tolist(),
                  'GT_trajectory_m': graph.positions[node, 5:].tolist(), 'future_mask': graph.future_mask[node].tolist(),
                  'history_times_seconds': graph.history_times.tolist(), 'future_times_seconds': graph.future_times.tolist(),
                  'lane_positions_m': graph.lane_positions.tolist(), 'lane_vectors_m': graph.lane_vectors.tolist(),
                  'origin_global_m': graph.origin.tolist(), 'ego_yaw_global_rad': float(graph.ego_yaw),
                  'coordinate_frame': 't0 ego +x forward +y left; meters', 'panels': panels, 'GT_fingerprint': fingerprint,
                  't0_spatial_context': context, 'fixed_interaction_membership': member,
                  'checkpoint_SHA256': checkpoint_sha, 'actor_CSV_SHA256': csv_sha,
                  'selection': 'Deterministic largest gain/degradation, GT>5m prioritized; heterogeneous/VP eligibility fixed solely by t0 geometry and type.',
                  'representativeness': 'Purposive illustrative examples, not unbiased population-effect estimates.',
                  'trajectory_edits': False, 'smoothing': False, 'interpolation': False}
        source_path = ROOT / '04_evaluation/cases' / (name + '.json'); atomic_json(source_path, source)
        audit = draw(read_json(source_path), ROOT / '05_figures' / name)
        audit.update({'source_json': str(source_path.relative_to(ROOT)), 'source_sha256': sha256(source_path),
                      'checkpoint_SHA256': checkpoint_sha, 'same_actor_and_GT': True})
        atomic_json(ROOT / '05_figures' / (name + '_audit.json'), audit)
        figures.append({'name': name, 'source_json': str(source_path.relative_to(ROOT)), 'source_sha256': sha256(source_path),
                        'case_kind': kind, 'delta_FDE_m': delta, 'agent_type': row['agent_type'], 't0_spatial_context': context,
                        'actor_key': list(key), 'GT_endpoint_displacement_m': displacement, 'metric_audit': 'PASS'})
        print('STAGE4A_MATCHED_CASE=PASS', name, 'delta_FDE', delta, flush=True)
    ds.clear()
    atomic_json(ROOT / '04_evaluation/stage4a_qualitative_case_manifest.json', {
        'status': 'PASS', 'figures': figures, 'omitted_cases': omitted, 'case_count': len(figures),
        'checkpoint_SHA256': checkpoint_sha, 'actor_CSV_SHA256': csv_sha,
        'selected_after_locked_quantitative_evaluation': True, 'no_training': True,
        'spatial_interaction_context_does_not_prove_causality': True})
    print('STAGE4A_QUALITATIVE=PASS', flush=True)


def redraw_only():
    manifest = read_json(ROOT / '04_evaluation/stage4a_qualitative_case_manifest.json')
    assert manifest['status'] == 'PASS'
    for item in manifest['figures']:
        source_path = ROOT / item['source_json']; assert sha256(source_path) == item['source_sha256']
        source = read_json(source_path); audit = draw(source, ROOT / '05_figures' / item['name'])
        audit.update({'source_json': item['source_json'], 'source_sha256': item['source_sha256'],
                      'checkpoint_SHA256': source['checkpoint_SHA256'], 'same_actor_and_GT': True,
                      'redrawn_from_existing_true_JSON_without_inference_or_training': True})
        atomic_json(ROOT / '05_figures' / (item['name'] + '_audit.json'), audit)
        print('STAGE4A_CASE_LAYOUT=PASS', item['name'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'path.simplify': False})
    if '--redraw-only' in sys.argv:
        redraw_only()
    else:
        main()
