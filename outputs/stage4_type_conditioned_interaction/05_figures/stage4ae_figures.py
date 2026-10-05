"""Real-source motion and gradient figures for the offline mechanism diagnostic."""
import subprocess
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage4ae_common import read_csv, read_json, atomic_json, sha256
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = ROOT / '05_figures'
B_COLOR, C_COLOR = '#6E8694', '#B86F4A'
GROUPS = ('LocalEncoder', 'TypeEmbedding', 'GlobalInteractor', 'Decoder', 'ALL SHARED')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .8,
    'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'path.simplify': False, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def save(fig, name, sources, note, values):
    fig.tight_layout(pad=1.2)
    exports = {}
    for extension in ('png', 'pdf', 'svg'):
        path = OUT / f'{name}.{extension}'
        fig.savefig(path, dpi=300)
        if extension == 'svg':
            # Remove renderer formatting whitespace without changing SVG tokens.
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        exports[extension] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'bytes': path.stat().st_size}
    with Image.open(OUT / f'{name}.png') as im:
        exports['png']['pixels'] = list(im.size)
        assert min(im.size) >= 900
    pdf = OUT / f'{name}.pdf'
    text = subprocess.check_output(['pdftotext', str(pdf), '-']).decode()
    fonts = subprocess.check_output(['pdffonts', str(pdf)]).decode()
    assert len(text.strip()) > 20 and 'TrueType' in fonts
    svg = ET.parse(OUT / f'{name}.svg')
    assert len(svg.findall('.//{http://www.w3.org/2000/svg}text')) > 4
    atomic_json(OUT / f'{name}_audit.json', {'status': 'PASS', 'backend': 'python',
        'contract_sha256': sha256(ROOT / '00_manifest/stage4ae_figure_contract.json'),
        'sources': {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        'plotted_values': values, 'statistics_and_interpretation': note,
        'exports': exports, 'editable_PDF_SVG_text_verified': True,
        'smoothing': False, 'mock_data': False})
    plt.close(fig)


def main():
    read_json(ROOT / '00_manifest/stage4ae_figure_contract.json')
    source = ROOT / '06_tables/stage4ae_pedestrian_motion_bin_metrics.csv'
    boot_source = ROOT / '06_tables/stage4ae_pedestrian_motion_bin_bootstrap.csv'
    motion = read_csv(source)
    boots = [r for r in read_csv(boot_source) if r['Metric'] == 'minFDE6']
    assert [r['MotionBin'] for r in motion] == [r['MotionBin'] for r in boots]
    labels = [r['MotionBin'].rstrip('m').replace('-', '–') for r in motion]
    x = np.arange(5)
    b, c = [float(r['Stage3B_minFDE6']) for r in motion], [float(r['Stage4A_minFDE6']) for r in motion]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x-.19, b, width=.36, color=B_COLOR, label='Stage3B')
    ax.bar(x+.19, c, width=.36, color=C_COLOR, label='Stage4A')
    for positions, values in ((x-.19, b), (x+.19, c)):
        for pos, v in zip(positions, values):
            ax.text(pos, v+.035, f'{v:.3f}', ha='center', fontsize=8)
    ax.set(xticks=x, xticklabels=labels, xlabel='GT endpoint displacement (m)',
        ylabel='Pedestrian minFDE6 (m)', ylim=(0, max(b+c)*1.19))
    ax.legend(loc='upper left')
    save(fig, 'stage4ae_pedestrian_motion_bin_fde', [source],
        'Frozen full-horizon pedestrian actor-window means; bins [lower,upper). Counts: 3192,313,916,7321,260. No inference.',
        {'bins': labels, 'Stage3B': b, 'Stage4A': c})

    delta = np.array([float(r['Delta_C_minus_B']) for r in boots])
    lo, hi = np.array([float(r['CI95Lower']) for r in boots]), np.array([float(r['CI95Upper']) for r in boots])
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(x, delta, color=[C_COLOR if v >= 0 else B_COLOR for v in delta], width=.55)
    ax.errorbar(x, delta, yerr=np.vstack((delta-lo, hi-delta)), fmt='none', ecolor='#27323B', capsize=4, lw=1)
    ax.axhline(0, color='#53616C', lw=.9)
    ax.set(xticks=x, xticklabels=labels, xlabel='GT endpoint displacement (m)',
        ylabel='Δ minFDE6: Stage4A − Stage3B (m)')
    ax.set_ylim(min(lo.min(), 0)-.025, max(hi.max(), 0)+.022)
    for i, v in enumerate(delta):
        ax.text(i, hi[i]+.006, f'{v:+.4f}', ha='center', fontsize=8)
    save(fig, 'stage4ae_pedestrian_motion_delta', [source, boot_source],
        'Paired official VAL scene-cluster percentile bootstrap CI95, 1000 replicates seed2022; positive is worse. Sparse bins require unique-scene caution; no multiple-comparison correction.',
        {'bins': labels, 'delta': delta.tolist(), 'CI95Lower': lo.tolist(), 'CI95Upper': hi.tolist()})

    summary_source = ROOT / '06_tables/stage4ae_gradient_conflict_summary.csv'
    shift_source = ROOT / '06_tables/stage4ae_gradient_shift_bootstrap.csv'
    summary = read_csv(summary_source)
    shift = {r['ParameterGroup']: r for r in read_csv(shift_source)}
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    plotted = {}
    for model, offset, color in (('Stage3B', -.19, B_COLOR), ('Stage4A', .19, C_COLOR)):
        vals = np.array([float(shift[g][model+'MeanCos']) for g in GROUPS])
        lows = np.array([float(shift[g][model+'MeanCI95Lower']) for g in GROUPS])
        highs = np.array([float(shift[g][model+'MeanCI95Upper']) for g in GROUPS])
        ax.bar(x+offset, vals, width=.36, color=color, label=model)
        ax.errorbar(x+offset, vals, yerr=np.vstack((vals-lows, highs-vals)), fmt='none',
            ecolor='#27323B', capsize=3, lw=.9)
        plotted[model] = {'mean': vals.tolist(), 'CI95Lower': lows.tolist(), 'CI95Upper': highs.tolist()}
    ax.axhline(0, color='#53616C', lw=.8)
    ax.set(xticks=x, xticklabels=('Local\nEncoder', 'Type\nEmbedding', 'Global\nInteractor', 'Decoder', 'All\nShared'),
        ylabel='Mean Vehicle/Pedestrian gradient cosine')
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(loc='upper left')
    save(fig, 'stage4ae_gradient_cosine_comparison', [summary_source, shift_source],
        'Same 100 TRAIN batches; original class-mean NLL+classification; model-mean 95% intervals from paired batch resampling (1000, seed2022). Not independent scene-cluster model significance. Relation Module excluded.', plotted)

    batch_source = ROOT / '04_evaluation/diagnostics/stage4ae_gradient_conflict_batches.csv'
    batches = read_csv(batch_source)
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    plotted = {}
    hist_bins = np.linspace(-1, 1, 21)
    for model, color in (('Stage3B', B_COLOR), ('Stage4A', C_COLOR)):
        values = [float(r['CosVP']) for r in batches if r['Model'] == model and r['ParameterGroup'] == 'ALL SHARED']
        assert len(values) == 100
        ax.hist(values, bins=hist_bins, histtype='step', linewidth=1.6, color=color, label=model)
        plotted[model] = values
    ax.axvline(0, color='#333E46', linestyle='--', lw=1, label='Conflict boundary')
    ax.set(xlabel='ALL SHARED Vehicle/Pedestrian gradient cosine', ylabel='Batch count', xlim=(-1, 1))
    ax.legend(fontsize=8, loc='upper left')
    save(fig, 'stage4ae_global_gradient_conflict_distribution', [batch_source],
        'Observed batch-level cosine distribution, same 100 TRAIN batches each; x<0 marks conflicting class-mean directions. Fixed 0.1-wide histogram bins; dependent batches.',
        {'histogram_edges': hist_bins.tolist(), **plotted})

    all_shared = next(r for r in summary if r['Model'] == 'Stage4A' and r['ParameterGroup'] == 'ALL SHARED')
    for name, keys, ylabel, note in (
        ('stage4ae_class_gradient_norm', ('MeanNormV', 'MeanNormP'), 'Mean class gradient norm',
         'Stage4A ALL SHARED means of per-batch class-mean-loss gradient norms; no class-frequency multiplication.'),
        ('stage4ae_gradient_pressure', ('MeanEffectiveV', 'MeanEffectiveP'), 'Mean effective pressure proxy',
         'Stage4A ALL SHARED means of w_class × class-mean gradient norm, w from valid supervised coordinate share. Optimization-pressure proxy, not exact mixed-gradient contribution.')):
        vals = [float(all_shared[k]) for k in keys]
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        ax.bar(('Vehicle', 'Pedestrian'), vals, color=(B_COLOR, C_COLOR), width=.5)
        ax.set(ylabel=ylabel, ylim=(0, max(vals)*1.22))
        for i, value in enumerate(vals):
            ax.text(i, value+max(vals)*.035, f'{value:.3f}', ha='center', fontsize=9)
        save(fig, name, [summary_source], note, {'Vehicle': vals[0], 'Pedestrian': vals[1]})
    print('STAGE4AE_FIGURES=PASS', flush=True)


if __name__ == '__main__':
    main()
