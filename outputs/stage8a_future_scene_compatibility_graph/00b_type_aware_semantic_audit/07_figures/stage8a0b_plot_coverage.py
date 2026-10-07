"""Static diagnostic coverage figure from complete, fixed-protocol audit tables."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0b_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    rows=list(csv.DictReader((ROOT/'06_tables/stage8a0b_coverage.csv').open()))
    groups_=('Vehicle','MovingVehicle','StoppedVehicle','ParkedVehicle','Pedestrian','Bicycle')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),sharey=True,layout='constrained')
    for ax,split in zip(axes,('train','val')):
        by={r['ActorGroup']:r for r in rows if r['Split']==split and r['Population']=='full_horizon_ranking_targets'}
        x=np.arange(len(groups_));lane=[100*float(by[g]['RouteCenterlineCoverage']) for g in groups_]
        aware=[100*float(by[g]['TypeAwareAnyCoverage']) for g in groups_]
        ax.bar(x-.18,lane,width=.36,color='#7B8C9A',label='Lane/connector within 10 m')
        ax.bar(x+.18,aware,width=.36,color='#167D9A',label='Type-aware context')
        ax.axhline(95,color='#B64040',ls='--',lw=1,label='95% coverage gate')
        ax.set_xticks(x,['Vehicle','Moving','Stopped','Parked','Pedestrian','Bicycle'],rotation=35,ha='right')
        ax.set_ylim(0,105);ax.set_title(split.upper()+' full-horizon candidates');ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_ylabel('Candidate coverage (%)');axes[1].legend(frameon=False,loc='lower left',fontsize=8)
    fig.suptitle('Fixed 10 m lane radius / 2 m polygon boundary rule; coverage before fallback',fontsize=11)
    for ext in ('svg','pdf','png'):fig.savefig(ROOT/'07_figures'/f'stage8a0b_coverage.{ext}',dpi=180)
    svg=ROOT/'07_figures/stage8a0b_coverage.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
if __name__=='__main__':main()
