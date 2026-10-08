"""Diagnostic type diversity and natural zero-map rates, without error metrics."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0c_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    compare=list(csv.DictReader((ROOT/'06_tables/stage8a0c_selector_comparison.csv').open()))
    coverage=list(csv.DictReader((ROOT/'06_tables/stage8a0c_sparse_coverage.csv').open()))
    names=('Vehicle','MovingVehicle','Pedestrian','Bicycle')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
    val={(r['Group'],r['Selector']):r for r in compare if r['Split']=='val' and r['Population']=='full_horizon_ranking_targets'}
    x=np.arange(len(names))
    for offset,selector,color in [(-.18,'GlobalTop8','#7B8C9A'),(.18,'TypeAwareQuota','#167D9A')]:
        axes[0].bar(x+offset,[float(val[(n,selector)]['MeanDistinctEntityTypes']) for n in names],width=.36,label=selector,color=color)
    axes[0].set_xticks(x,['Vehicle','Moving','Pedestrian','Bicycle']);axes[0].set_ylim(0,4)
    axes[0].set_ylabel('Mean authorized entity types / mode');axes[0].set_title('VAL full-horizon: fixed primary entity pool')
    axes[0].legend(frameon=False,fontsize=8)
    groups_=('Vehicle','ParkedVehicle','Pedestrian','Bicycle');by={(r['Split'],r['Group']):r for r in coverage if r['Population']=='full_horizon_ranking_targets'}
    x=np.arange(len(groups_))
    for offset,split,color in [(-.18,'train','#7B8C9A'),(.18,'val','#167D9A')]:
        axes[1].bar(x+offset,[100*float(by[(split,n)]['ZeroMapRate']) for n in groups_],width=.36,label=split.upper(),color=color)
    axes[1].set_xticks(x,['Vehicle','Parked','Pedestrian','Bicycle']);axes[1].set_ylim(0,30)
    axes[1].set_ylabel('Zero-map candidate rate (%)');axes[1].set_title('Zero-map samples retained');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ('svg','pdf','png'):fig.savefig(ROOT/'07_figures'/f'stage8a0c_selector_comparison.{ext}',dpi=180)
    svg=ROOT/'07_figures/stage8a0c_selector_comparison.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
if __name__=='__main__':main()
