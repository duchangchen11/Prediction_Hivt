"""A readable final view of the exact extension VAL observations; no optimization."""
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3a_plus_common import BEST_MANIFEST, CLASSES, read_json, atomic_json, sha256
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    m = read_json(BEST_MANIFEST); assert m['status'] == 'COMPLETE'
    curve = ROOT / '03_no_type_baseline/stage3a_plus_nll_extension_curve.csv'
    with curve.open() as f: rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    source_sha = sha256(curve); original = m['original_best']
    cp = torch.load(ROOT / original['relative_path'], map_location='cpu', weights_only=False)
    assert sha256(ROOT / original['relative_path']) == original['sha256']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained', gridspec_kw={'width_ratios': [1.25, 1]})
    x = [original['global_step']] + [r['global_step'] for r in rows]
    values = [original['overall_minFDE6']] + [r['VAL_overall_FDE'] for r in rows]
    axes[0].plot(x, values, 'o-', color='#246C9E', ms=5, label='Overall full-horizon VAL')
    axes[0].axhline(original['overall_minFDE6'], color='#737B84', ls='--', lw=1, label='Original best')
    axes[0].set_title('a  Overall convergence check'); axes[0].set_ylabel('Overall minFDE@6 (m; expanded range)')
    axes[0].legend(frameon=False, fontsize=9)
    for name, color in zip(CLASSES, ('#737B84', '#4A8A77', '#C87B24')):
        initial = cp['metadata']['per_class_full_horizon_metrics'][name]['minFDE6']
        axes[1].plot(x, [initial] + [r['VAL_' + name + '_FDE'] for r in rows], 'o-', ms=4, color=color, label=name)
    axes[1].set_title('b  Per-type validation'); axes[1].set_ylabel('Type-wise minFDE@6 (m)'); axes[1].legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.set_xlabel('Executed optimizer step'); ax.grid(alpha=.15); ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    stop_label = '3000-step limit reached' if m['stop_reason'] == 'extra_steps_3000_limit' else '5 nonimproving validations'
    fig.suptitle(f"Stage3A+ NLL continuation | {m['executed_extra_steps']} extra updates | {stop_label}", fontsize=12)
    stem = curve.with_suffix('')
    for ext in ('png', 'pdf', 'svg'): fig.savefig(str(stem) + '.' + ext, dpi=300)
    svg = Path(str(stem) + '.svg'); svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines()) + '\n')
    plt.close(fig); assert sha256(curve) == source_sha
    atomic_json(ROOT / '03_no_type_baseline/stage3a_plus_nll_extension_curve_audit.json', {
        'status': 'PASS', 'source_CSV_sha256': source_sha, 'original_checkpoint_sha256': original['sha256'],
        'validation_observations': len(rows), 'original_reference_included': True,
        'overall_y_axis_expanded_for_small_observed_differences': True,
        'source_values_modified': False, 'fitted_or_smoothed_curve': False,
        'exports': {ext: sha256(Path(str(stem) + '.' + ext)) for ext in ('png', 'pdf', 'svg')}})
    print('FINAL_EXTENSION_CURVE=PASS', flush=True)


if __name__ == '__main__':
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    main()
