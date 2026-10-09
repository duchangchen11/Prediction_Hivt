"""Standalone diagnostic figures, all explicitly development evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11c_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RAW='#b34a36';CAL='#236a96';LABEL='Stage11C — OOF Development Evidence'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'figure.dpi':120})
def save(fig,name):
    fig.suptitle(LABEL,fontsize=13,y=.99);fig.tight_layout(rect=(0,0,1,.95))
    for ext in ('png','svg','pdf'):
        p=ROOT/f'06_figures/{name}.{ext}';fig.savefig(p,dpi=200,bbox_inches='tight')
        if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
    plt.close(fig)
def main():
    verify();table=pd.read_csv(ROOT/'07_tables/stage11c_oof_probability_metrics.csv')
    rel=pd.read_csv(ROOT/'05_probability_diagnostics/stage11c_reliability_bins.csv')
    hist=pd.read_csv(ROOT/'05_probability_diagnostics/stage11c_probability_histograms.csv')
    rawp=np.load(OOF/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    vp=np.load(ROOT/'03_oof_evaluation/cache/stage11c_calibrated_vehicle_pedestrian.npz')
    f=frame();vptypes=f.iloc[vp['headtrain_indices']].agent_type.to_numpy()
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for c,group in enumerate(TYPES[:2]):
        ax=axes[0,c]
        for model,color in [('C Raw',RAW),('C Calibrated',CAL)]:
            h=hist[(hist.Group==group)&(hist.Model==model)]
            ax.stairs(h.Count/h.Count.sum(),np.r_[h.Lower.to_numpy(),h.Upper.iloc[-1]],label=model,color=color,lw=2)
        ax.set(xlabel='Highest mode probability',ylabel='Fraction of actors/windows',title=group,xlim=(0,1))
        ax.legend(frameon=False)
        ax=axes[1,c];mask=vptypes==group;orig=rawp[vp['headtrain_indices'][mask],4];cal=vp['probabilities'][mask]
        for data,label,color in [(orig,'C Raw',RAW),(cal,'C Calibrated',CAL)]:
            counts,edges=np.histogram(data.ravel(),bins=np.linspace(0,1,51))
            ax.stairs(counts/counts.sum(),edges,label=label,color=color,lw=2)
        ax.set(xlabel='Probability across all six modes',ylabel='Fraction of candidate probabilities',xlim=(0,1))
    save(fig,'stage11c_probability_distribution')
    for group in TYPES[:2]:
        fig,axes=plt.subplots(2,1,figsize=(6.5,7),gridspec_kw={'height_ratios':[3,1]},sharex=True)
        axes[0].plot([0,1],[0,1],color='.5',ls='--',lw=1,label='Identity reference')
        for model,color in [('C Raw',RAW),('C Calibrated',CAL)]:
            r=rel[(rel.Group==group)&(rel.Model==model)];nonempty=r[r.Count>0]
            axes[0].plot(nonempty.MeanConfidence,nonempty.OracleTop1MatchRate,'o-',color=color,label=model,lw=1.5)
            centers=(r.Lower+r.Upper)/2
            axes[1].step(centers,r.Count/r.Count.sum(),where='mid',label=model,color=color)
        axes[0].set(title=f'{group}: oracle-mode classification',ylabel='Fraction Top1 equals oracle-best mode',xlim=(0,1),ylim=(0,1))
        axes[0].legend(frameon=False);axes[1].set(xlabel='Highest mode probability (15 uniform bins)',ylabel='Bin fraction')
        save(fig,'stage11c_'+group.lower()+'_reliability')
    fig,ax=plt.subplots(figsize=(8,5));groupnames=['Overall','Vehicle','Pedestrian'];x=np.arange(3)
    for offset,model,color in [(-.18,'C Raw',RAW),(.18,'C Calibrated',CAL)]:
        vals=[float(table[(table.Group==g)&(table.Model==model)].iloc[0].OracleBestModeNLL) for g in groupnames]
        bars=ax.bar(x+offset,vals,width=.34,label=model,color=color)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3)
    ax.set(xticks=x,xticklabels=groupnames,ylabel='Oracle-best-mode NLL (natural log)',title='Fixed temperatures: no change in Top1 selection')
    ax.legend(frameon=False);save(fig,'stage11c_oracle_mode_nll')
    vals=np.load(ROOT/'03_oof_evaluation/cache/stage11c_actor_probability_metrics.npy',mmap_mode='r')
    fields=read_json(ROOT/'03_oof_evaluation/cache/stage11c_evaluation_complete.json')['fields'];k=fields.index('Top1FDE')
    raw=vals[:,4,k];cal=vals[:,5,k];assert np.array_equal(raw,cal)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8));maxfd=float(raw.max())
    image=axes[0].hexbin(raw,cal,gridsize=70,bins='log',mincnt=1,cmap='Blues')
    axes[0].plot([0,maxfd],[0,maxfd],color='.4',ls='--',lw=1)
    axes[0].set(xlabel='C Raw Top1FDE (m)',ylabel='C Calibrated Top1FDE (m)',title=f'All {len(raw):,} actors/windows',xlim=(0,maxfd),ylim=(0,maxfd))
    fig.colorbar(image,ax=axes[0],label='Actor/window count (log scale)')
    axes[1].bar(['Overall','Vehicle','Pedestrian','Bicycle'],[0,0,0,0],color=CAL)
    axes[1].axhline(0,color='.4',lw=1);axes[1].set(ylabel='Calibrated − raw Top1FDE (m)',ylim=(-.01,.01),title='Exact identity, not a new geometry gain')
    axes[1].text(.5,.8,'Changed Top1 modes: 0\nMax |Δ Top1FDE|: 0\nBicycle: bitwise FoldR2',ha='center',va='center',transform=axes[1].transAxes)
    save(fig,'stage11c_top1_fde_identity')
    files={p.name:sha256(p) for p in (ROOT/'06_figures').glob('stage11c_*') if p.suffix in ('.png','.svg','.pdf')}
    assert len(files)==15
    atomic_json(ROOT/'06_figures/stage11c_figure_manifest.json',{'Status':'PASS','CaptionLabel':LABEL,'Files':files,
        'ReliabilityEvent':'Top1 mode equals frozen-six-candidate FDE oracle, not real-world risk probability'})
    print('FIVE_DEVELOPMENT_FIGURES_EXPORTED_PNG_SVG_PDF',flush=True)
if __name__=='__main__':main()
