"""Observed Stage4A-D diagnostic figures, with source hashes and export QA."""
import csv
from pathlib import Path
import sys
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '04_evaluation/diagnostics'))
from stage4ad_diagnostic import atomic_json, sha256, read_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image

OUT = ROOT / '05_figures'
TYPES = ('Vehicle', 'Pedestrian', 'Bicycle')
COLORS = ('#426E86', '#B66C45', '#8A819B')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': .8,
    'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'path.simplify': False, 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def save(fig, name, source, note, plotted):
    fig.tight_layout(pad=1.2)
    exports = {}
    for extension in ('png', 'pdf', 'svg'):
        path = OUT / (name + '.' + extension)
        fig.savefig(path, dpi=300)
        exports[extension] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'bytes': path.stat().st_size}
    with Image.open(OUT / (name + '.png')) as im:
        exports['png']['pixels'] = list(im.size)
        assert min(im.size) >= 900
    svg = ET.parse(OUT / (name + '.svg'))
    texts = svg.findall('.//{http://www.w3.org/2000/svg}text')
    assert len(texts) > 4
    pdf_text = subprocess.check_output(['pdftotext', str(OUT / (name + '.pdf')), '-']).decode()
    fonts = subprocess.check_output(['pdffonts', str(OUT / (name + '.pdf'))]).decode()
    assert len(pdf_text.strip()) > 20 and ('TrueType' in fonts or 'CID TrueType' in fonts)
    exports['pdf']['editable_text_verified'] = True
    atomic_json(OUT / (name + '_audit.json'), {'status': 'PASS', 'backend': 'python',
        'contract_sha256': sha256(ROOT / '00_manifest/stage4ad_figure_contract.json'),
        'source': str(source.relative_to(ROOT)), 'source_sha256': sha256(source),
        'plotted_values': plotted, 'interpretation_and_statistics': note,
        'exports': exports, 'SVG_text_elements': len(texts), 'smoothing': False,
        'mock_or_generated_data': False})
    plt.close(fig)


def main():
    read_json(ROOT / '00_manifest/stage4ad_figure_contract.json')
    summary = ROOT / '06_tables/stage4ad_attention_scale_summary.csv'
    rows = [r for r in read_csv(summary) if r['Layer'] == 'All']
    assert [r['TargetType'] for r in rows] == list(TYPES)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    x = np.arange(3)
    base = np.array([float(r['MeanAbsBase']) for r in rows])
    bias = np.array([float(r['MeanAbsBias']) for r in rows])
    ax.bar(x-.19, base, width=.36, color='#A0ABB2', label='Base logit')
    ax.bar(x+.19, bias, width=.36, color=COLORS[0], label='Relation bias')
    for xs, values in ((x-.19, base), (x+.19, bias)):
        for pos, value in zip(xs, values):
            ax.text(pos, value+.025*max(base.max(), bias.max()), f'{value:.2f}', ha='center', fontsize=8)
    ax.set(xticks=x, xticklabels=TYPES, ylabel='Mean absolute logit', ylim=(0, 1.26*max(base.max(), bias.max())))
    ax.legend(loc='upper right', fontsize=8)
    save(fig, 'stage4ad_logit_scale_by_target', summary,
        'Equal weight for actual edge-layer-head observations; descriptive means, no uncertainty bars.',
        {r['TargetType']: {'base': float(r['MeanAbsBase']), 'bias': float(r['MeanAbsBias'])} for r in rows})

    pair_source = ROOT / '06_tables/stage4ad_pair_scale_summary.csv'
    pairs = read_csv(pair_source)
    matrix = np.array([float(r['BiasBaseRatio']) for r in pairs]).reshape(3, 3)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=matrix.max(), aspect='equal')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=TYPES, yticklabels=TYPES,
        xlabel='Source type', ylabel='Target type')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                color='white' if matrix[i,j] > matrix.max()*.55 else '#172330')
    cb = fig.colorbar(im, ax=ax, shrink=.9, pad=.035)
    cb.set_label('Mean |bias| / mean |base|', fontsize=8)
    save(fig, 'stage4ad_bias_base_ratio_heatmap', pair_source,
        'Ratio of pooled absolute means, not average of per-edge ratios; target rows and source columns.', matrix.tolist())

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    delta = [float(r['DeltaEntropy']) for r in rows]
    ax.bar(TYPES, delta, color=COLORS, width=.56)
    ax.axhline(0, color='#60686E', lw=.8)
    ax.set_ylabel('Mean entropy change (nats)')
    lower, upper = min(delta + [0]), max(delta + [0])
    span = max(upper-lower, .01)
    ax.set_ylim(lower-.22*span, upper+.20*span)
    for i, value in enumerate(delta):
        ax.text(i, value-.055*span if value < 0 else value+.045*span, f'{value:+.3f}',
            ha='center', va='top' if value < 0 else 'bottom', fontsize=8)
    save(fig, 'stage4ad_attention_entropy_shift', summary,
        'H_final - H_base on whole actual neighborhoods, equal target-layer-head weight; negative means concentration, not improvement.',
        dict(zip(TYPES, delta)))

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    switch = [100*float(r['TopNeighborSwitchRate']) for r in rows]
    ax.bar(TYPES, switch, color=COLORS, width=.56)
    ax.set(ylabel='Top neighbor switch rate (%)', ylim=(0, max(switch)*1.25))
    for i, value in enumerate(switch):
        ax.text(i, value+max(switch)*.035, f'{value:.1f}%', ha='center', fontsize=8)
    save(fig, 'stage4ad_top_neighbor_switch', summary,
        'Different source-node identity for argmax attention, same edge order; equal target-layer-head weight.', dict(zip(TYPES, switch)))

    sensitivity = ROOT / '04_evaluation/diagnostics/stage4ad_bias_scale_sensitivity.csv'
    series = read_csv(sensitivity)
    lambdas = [float(r['Lambda']) for r in series]
    assert lambdas == [0, .25, .5, .75, 1]
    for name, groups, colors in (
        ('stage4ad_lambda_sensitivity', ('Overall', 'Vehicle', 'Pedestrian'), ('#707980', COLORS[0], COLORS[1])),
        ('stage4ad_lambda_motion_sensitivity', ('Vehicle >5m', 'Pedestrian >5m'), (COLORS[0], COLORS[1]))):
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        plotted = {}
        for group, color, marker in zip(groups, colors, ('o', 's', '^')):
            values = [float(r[group+'_minFDE6']) for r in series]
            plotted[group] = values
            ax.plot(lambdas, values, color=color, marker=marker, ms=4, lw=1.3, label=group)
        ax.set(xlabel=r'Inference bias scale $\lambda$', ylabel='minFDE (m)', xticks=lambdas,
            xticklabels=('0', '0.25', '0.50', '0.75', '1.00'))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.legend(fontsize=8, loc='best')
        save(fig, name, sensitivity,
            'Full official VAL150 and 54990 full-horizon actor windows per lambda, one frozen Stage4A checkpoint; no new model selection or independent-sample CIs.',
            {'lambda': lambdas, **plotted})
    print('STAGE4AD_FIGURES=PASS', flush=True)


if __name__ == '__main__':
    main()
