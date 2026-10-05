"""Three paired motion figures, each with one genuine success and one failure."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3a_plus_common import read_json, atomic_json, sha256, verify_freeze
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from PIL import Image


def draw_panel(fig, ax, d):
    history = np.array(d['history_trajectory_m'], dtype=float); hm = np.array(d['history_mask'], dtype=bool)
    gt = np.array(d['GT_trajectory_m'], dtype=float); modes = np.array(d['HiVT_trajectories_m'], dtype=float)
    prob = np.array(d['mode_probabilities']); b = d['best_FDE_mode_zero_based']; t = d['top1_mode_zero_based']
    assert gt.shape == (12, 2) and modes.shape == (6, 12, 2) and all(d['future_mask'])
    assert b == int(np.linalg.norm(modes[:, -1] - gt[-1], axis=1).argmin()) and t == int(prob.argmax())
    best, top = modes[b], modes[t]
    be = np.linalg.norm(best - gt, axis=1); te = np.linalg.norm(top - gt, axis=1)
    actual = dict(zip(('minADE6', 'minFDE6', 'Top1ADE6', 'Top1FDE6'), map(float, (be.mean(), be[-1], te.mean(), te[-1]))))
    differences = {k: abs(v - d['numeric_metrics'][k]) for k, v in actual.items()}
    assert max(differences.values()) < 1e-4, differences
    visible = np.r_[history[hm], gt, best, top]; lo, hi = visible.min(0) - 7, visible.max(0) + 7
    center = (lo + hi) / 2; half = (hi - lo).max() / 2
    starts = np.array(d['lane_positions_m']).reshape(-1, 2); vectors = np.array(d['lane_vectors_m']).reshape(-1, 2)
    segments = np.stack((starts, starts + vectors), axis=1)
    near = ((segments.max(1) >= lo) & (segments.min(1) <= hi)).all(1); segments = segments[near]

    def trajectories(target, labels=False, with_history=False):
        target.add_collection(LineCollection(segments, colors='#B9C1C9', linewidths=.5, alpha=.3, zorder=1))
        if with_history:
            h = history.copy(); h[~hm] = np.nan
            target.plot(h[:, 0], h[:, 1], 'o-', color='#777D86', lw=1.5, ms=5, label='History' if labels else None, zorder=3)
        target.plot(gt[:, 0], gt[:, 1], 's-', color='#161616', lw=2.5, ms=7, markerfacecolor='white',
                    markeredgewidth=1.2, label='GT' if labels else None, zorder=10)
        target.plot(top[:, 0], top[:, 1], '^--', color='#C87B24', lw=2, ms=8, markerfacecolor='white',
                    markeredgewidth=1.1, label=f'Top-1 (Mode {t + 1})' if labels else None, zorder=9)
        target.plot(best[:, 0], best[:, 1], 'o-', color='#246C9E', lw=2.5, ms=5.5, markeredgecolor='white',
                    markeredgewidth=.6, label=f'Best-FDE (Mode {b + 1})' if labels else None, zorder=11)
        for xy, color, marker in ((gt[-1], '#161616', 's'), (top[-1], '#C87B24', '^'), (best[-1], '#246C9E', 'o')):
            target.plot(*xy, marker=marker, ms=11, color=color, markerfacecolor='white' if marker == 's' else color,
                        markeredgecolor=color if marker == 's' else 'white', markeredgewidth=1.3, zorder=12)
        target.set_aspect('equal', adjustable='box'); target.grid(alpha=.1, linewidth=.5)

    trajectories(ax, labels=True, with_history=True)
    ax.set_xlim(center[0] - half, center[0] + half); ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_xlabel('t0 ego forward x (m)'); ax.set_ylabel('t0 ego left y (m)')
    legend = ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.015), ncol=2, fontsize=9, frameon=False)
    title = ax.set_title(f"{d['kind'].capitalize()} | {d['scene_name']} | {d['agent_type']}\n"
                         f"sample {d['sample_token'][:8]} | actor {d['instance_token'][:8]} | GT displacement {d['GT_endpoint_displacement_recomputed_m']:.2f} m\n"
                         f"Best ADE/FDE {actual['minADE6']:.3f}/{actual['minFDE6']:.3f} m   Top-1 ADE/FDE {actual['Top1ADE6']:.3f}/{actual['Top1FDE6']:.3f} m",
                         fontsize=10, pad=62)
    annotations = []
    for xy, text, color, position, align in (
            (gt[-1], 'GT endpoint', '#161616', (-.06, .20), 'right'),
            (best[-1], f"Best endpoint\nFDE = {actual['minFDE6']:.3f} m", '#246C9E', (1.06, .68), 'left'),
            (top[-1], f"Top-1 endpoint\nFDE = {actual['Top1FDE6']:.3f} m", '#C87B24', (1.06, .36), 'left')):
        annotations.append(ax.annotate(text, xy=xy, xytext=position, textcoords='axes fraction', ha=align, va='center',
                            annotation_clip=False, fontsize=9, color=color, zorder=16,
                            bbox={'boxstyle': 'square,pad=.15', 'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9},
                            arrowprops={'arrowstyle': '-', 'color': color, 'lw': .75, 'shrinkB': 7}))
    late = np.r_[gt[-6:], best[-6:], top[-6:]]; zl, zh = late.min(0) - 2.5, late.max(0) + 2.5
    normalized = (visible - (center - half)) / (2 * half)
    corners = ((.03, .59, .42, .34), (.55, .59, .42, .34), (.03, .04, .42, .34), (.55, .04, .42, .34))
    occupied = [int(((normalized[:, 0] >= x) & (normalized[:, 0] <= x + w) & (normalized[:, 1] >= y) & (normalized[:, 1] <= y + h)).sum()) for x, y, w, h in corners]
    inset = ax.inset_axes(corners[int(np.argmin(occupied))], zorder=20)
    trajectories(inset); inset.set_xlim(zl[0], zh[0]); inset.set_ylim(zl[1], zh[1])
    inset.set_title('Zoom: last 6 future points', fontsize=8, pad=4); inset.tick_params(labelsize=7)
    for axis in (inset.xaxis, inset.yaxis): axis.set_major_locator(MaxNLocator(nbins=2, steps=(1, 2, 5, 10), min_n_ticks=1))
    for spine in inset.spines.values(): spine.set_visible(True); spine.set_color('#8D969F')
    ax.indicate_inset_zoom(inset, edgecolor='#77818B', alpha=.5, zorder=2)
    return {'kind': d['kind'], 'scene_token': d['scene_token'], 'sample_token': d['sample_token'], 'instance_token': d['instance_token'],
            'recomputed_metrics': actual, 'metric_absolute_differences_m': differences, 'metric_tolerance_m': 1e-4,
            'future_marker_count_per_trajectory': 12, 'zoom_min_xy_m': zl.tolist(), 'zoom_max_xy_m': zh.tolist(),
            'zoom_rule': 'last6 GT/best/Top1+2.5m margin', 'annotations': annotations, 'occupied': (legend, title, inset)}


def main():
    verify_freeze(); manifest = read_json(ROOT / '04_evaluation/stage3a_plus_motion_case_manifest.json')
    for item in manifest['figures']:
        path = ROOT / item['source_json']; source_hash = sha256(path); assert source_hash == item['source_sha256']
        source = read_json(path); assert [p['kind'] for p in source['panels']] == ['success', 'failure']
        fig, axes = plt.subplots(2, 1, figsize=(10, 15))
        fig.subplots_adjust(left=.12, right=.89, bottom=.08, top=.86, hspace=.58)
        fig.suptitle(f"Stage3A+ | {item['agent_type']} motion >5 m | final best checkpoint", fontsize=14, y=.975)
        audits = [draw_panel(fig, ax, d) for ax, d in zip(axes, source['panels'])]
        fig.text(.5, .025, '12 original future observations; best-FDE uses GT; Top-1 uses highest saved probability.', ha='center', fontsize=9, color='#5C646D')
        fig.canvas.draw(); renderer = fig.canvas.get_renderer(); canvas = fig.bbox
        for audit in audits:
            boxes = [a.get_bbox_patch().get_window_extent(renderer) for a in audit.pop('annotations')]
            objects = audit.pop('occupied'); occupied = [o.get_tightbbox(renderer) if hasattr(o, 'get_tightbbox') else o.get_window_extent(renderer) for o in objects]
            for i, box in enumerate(boxes):
                assert box.x0 >= 12 and box.y0 >= 12 and box.x1 <= canvas.x1 - 12 and box.y1 <= canvas.y1 - 12, 'Text outside canvas'
                assert not any(box.overlaps(other) for other in occupied + boxes[:i]), 'Endpoint text obscured'
            audit['endpoint_layout_checks'] = 'PASS'
        stem = ROOT / '05_figures' / item['name']
        for ext in ('png', 'pdf', 'svg'): fig.savefig(str(stem) + '.' + ext, dpi=300)
        svg = Path(str(stem) + '.svg'); svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines()) + '\n')
        plt.close(fig)
        assert sha256(path) == source_hash
        with Image.open(str(stem) + '.png') as image: image.verify()
        atomic_json(Path(str(stem) + '_audit.json'), {'status': 'PASS', 'source_json': item['source_json'], 'source_sha256': source_hash,
                    'panels': audits, 'trajectory_coordinates_changed': False, 'smoothing': False, 'interpolation': False,
                    'exports': {ext: {'relative_path': str(Path(str(stem) + '.' + ext).relative_to(ROOT)),
                                      'sha256': sha256(Path(str(stem) + '.' + ext))} for ext in ('png', 'pdf', 'svg')}})
        print('MOTION_FIGURE=PASS', item['name'], flush=True)
    verify_freeze()


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'path.simplify': False})
    main()
