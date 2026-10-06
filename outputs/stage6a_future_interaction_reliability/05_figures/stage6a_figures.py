"""Four source-checked ranking figures; paired intervals and shared geometry."""
from pathlib import Path
import sys,csv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage6a_common import read_json,atomic_json,sha256
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
COLORS={'R0':'#818A92','R1':'#9C83A9','R2':'#3488AD'}
MAIN=ROOT/'06_tables/stage6a_main_ranking_results.csv'
CI=ROOT/'06_tables/stage6a_bootstrap_ci.csv'
GROUPS=('overall','vehicle','pedestrian','vehicle.moving')
LABELS=('Overall','Vehicle','Pedestrian','Moving vehicle')

def tables():
    with MAIN.open() as f:main={r['Group']:r for r in csv.DictReader(f)}
    with CI.open() as f:ci={(r['Comparison'],r['Group'],r['Metric']):r for r in csv.DictReader(f)}
    return main,ci

def export(fig,name,records,caption):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=list(fig.texts)+[t for legend in fig.legends for t in legend.get_texts()]
    for ax in fig.axes:
        texts.extend([ax.xaxis.label,ax.yaxis.label,ax.title,ax._left_title,ax._right_title]+list(ax.texts))
        for ticks,labels,limits in ((ax.get_xticks(),ax.get_xticklabels(),ax.get_xlim()),(ax.get_yticks(),ax.get_yticklabels(),ax.get_ylim())):
            texts.extend(t for x,t in zip(ticks,labels) if min(limits)<=x<=max(limits))
    for text in texts:
        if not text.get_visible() or not text.get_text():continue
        b=text.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1,(text.get_text(),b.bounds)
    exports={}
    for ext in ('png','pdf','svg'):
        p=ROOT/'05_figures'/(name+'.'+ext);fig.savefig(p,dpi=300,facecolor='white')
        if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
        exports[str(p.relative_to(ROOT))]=sha256(p)
    plt.close(fig)
    atomic_json(ROOT/'05_figures'/(name+'_audit.json'),{'status':'PASS','source_csv':str(MAIN.relative_to(ROOT)),
        'source_csv_sha256':sha256(MAIN),'bootstrap_csv':str(CI.relative_to(ROOT)),
        'bootstrap_csv_sha256':sha256(CI),'artist_max_abs_diff':0.,'plotted_values':records,
        'caption':caption,'exports_SHA256':exports,'backend':'python','training_seeds':1,
        'n_definition':'full-horizon actor-windows; whole scenes are bootstrap clusters',
        'multiple_comparison_correction':'none; subgroup intervals descriptive'})

def comparison(name,metric,groups,labels,title,scale=1.):
    main,ci=tables();fig=plt.figure(figsize=(7.2,4.6))
    gs=fig.add_gridspec(1,2,width_ratios=(1.05,1),left=.22,right=.96,bottom=.27,top=.78,wspace=.22)
    a,b=fig.add_subplot(gs[0]),fig.add_subplot(gs[1]);y=np.arange(len(groups));records=[]
    for j,v in enumerate(('R0','R1','R2')):
        yy=y+(j-1)*.21;values=np.array([float(main[g][v+'_'+metric])*scale for g in groups])
        dots,=a.plot(values,yy,'o',ms=5,color=COLORS[v],label=v)
        assert np.array_equal(dots.get_xdata(),values)
        for x,z in zip(values,yy):a.annotate(f'{x:.3f}' if scale==1 else f'{x:.2f}',(x,z),xytext=(7,0),textcoords='offset points',va='center',fontsize=7)
        records.append({'variant':v,'groups':groups,'mean':values.tolist()})
    maxv=max(max(r['mean']) for r in records);a.set_xlim(0,maxv*1.24)
    a.set_yticks(y,labels);a.set_ylim(len(groups)-.55,-.45);a.set_xlabel('Top1 FDE (m)' if scale==1 else 'Best-mode hit rate (%)')
    a.set_title('a   Full-horizon means',loc='left',fontweight='bold',fontsize=9)
    fig.legend(*a.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.22,.90),ncol=3,fontsize=7,frameon=False);a.grid(axis='x',alpha=.15)
    for j,comp in enumerate(('R1-R0','R2-R0','R2-R1')):
        variant='R1' if comp=='R1-R0' else 'R2';yy=y+(j-1)*.21
        rows=[ci[comp,g,metric] for g in groups]
        d=np.array([float(r['Delta'])*scale for r in rows]);lo=np.array([float(r['CI95_lower'])*scale for r in rows]);hi=np.array([float(r['CI95_upper'])*scale for r in rows])
        dots,=b.plot(d,yy,('s' if comp=='R2-R1' else 'o'),ms=4,color=COLORS[variant],markerfacecolor='white' if comp=='R2-R1' else COLORS[variant],label=comp)
        interval=b.hlines(yy,lo,hi,color=COLORS[variant],lw=1.2)
        assert np.array_equal(dots.get_xdata(),d)
        assert np.array_equal(np.array(interval.get_segments())[:,:,0],np.c_[lo,hi])
        records.append({'comparison':comp,'groups':groups,'delta':d.tolist(),'CI95_lower':lo.tolist(),'CI95_upper':hi.tolist()})
    b.axvline(0,color='#888888',ls=':',lw=.8);b.set_yticks(y,[]);b.set_ylim(len(groups)-.55,-.45);b.grid(axis='x',alpha=.15)
    b.set_title('b   Paired scene differences',loc='left',fontweight='bold',fontsize=9)
    b.set_xlabel('Paired difference (m)' if scale==1 else 'Paired difference (percentage points)')
    fig.legend(*b.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.60,.90),ncol=3,fontsize=6.5,frameon=False,handletextpad=.3,columnspacing=.7)
    fig.suptitle(title,y=.96,fontsize=11)
    counts='; '.join(f'{l.replace(chr(10)," ")}: n={int(main[g]["Count"]):,}' for g,l in zip(groups,labels))
    fig.text(.5,.15,counts,ha='center',fontsize=7)
    direction='Negative differences favor the new variant.' if scale==1 else 'Positive differences favor the new variant.'
    caption='Paired whole-scene percentile 95% CI; 150 VAL scenes, 1,000 resamples, seed 2022.\n'+direction+' One seed; subgroup intervals descriptive.\nR0/R1/R2 share all six trajectories.'
    fig.text(.5,.065,caption,ha='center',fontsize=7,color='#53616B')
    export(fig,name,records,caption)

def oracle():
    main,_=tables();fig,ax=plt.subplots(figsize=(7.2,3.6));fig.subplots_adjust(left=.29,right=.97,top=.79,bottom=.30)
    r=main['overall'];floor=float(r['R0_minFDE6']);records=[]
    for j,v in enumerate(('R0','R1','R2')):
        top=float(r[v+'_Top1FDE']);gap=float(r[v+'_OracleGap_FDE'])
        line,=ax.plot([floor,top],[j,j],lw=5,color=COLORS[v],solid_capstyle='butt')
        assert np.array_equal(line.get_xdata(),[floor,top]);assert abs(top-floor-gap)<2e-7
        ax.plot(top,j,'o',color=COLORS[v]);ax.plot(floor,j,'s',color='#313A42')
        ax.text((floor+top)/2,j-.13,f'Oracle gap {gap:.4f} m',ha='center',fontsize=8)
        ax.text(top+.07,j,f'Top1 {top:.4f}',va='center',fontsize=8)
        records.append({'variant':v,'minFDE6':floor,'Top1FDE':top,'OracleGap_FDE':gap})
    ax.axvline(floor,color='#313A42',ls=':',lw=1);ax.set_xlim(0,3.4);ax.set_ylim(2.5,-.6)
    ax.set_yticks(range(3),['R0 original','R1 no interaction','R2 future interaction']);ax.set_xlabel('Overall FDE (m)')
    fig.suptitle('Ranking closes part of the shared oracle gap',y=.96,fontsize=11)
    fig.text(.5,.84,f'Shared minFDE = {floor:.6f} m; geometry is bitwise identical',ha='center',fontsize=8)
    caption='n = 54,990 full-horizon actor-windows in 150 VAL scenes.\nminFDE uses GT to choose a candidate; Top1 uses predicted probability.\nSegments show mean gaps, not uncertainty intervals; geometry does not improve.'
    fig.text(.5,.08,caption,ha='center',fontsize=8,color='#53616B')
    export(fig,'stage6a_oracle_gap',records,caption)

def main():
    assert read_json(ROOT/'04_evaluation/stage6a_evaluation_complete.json')['status']=='PASS'
    assert read_json(ROOT/'00_manifest/stage6a_figure_contract.json')['backend']=='python'
    comparison('stage6a_top1fde_comparison','Top1FDE',GROUPS,LABELS,'Frozen candidates, different mode ranking')
    oracle()
    comparison('stage6a_hit_rate','Top1HitRate',GROUPS,LABELS,'How often does Top1 select the best-FDE candidate?',100.)
    comparison('stage6a_interaction_ablation','Top1FDE',('Heterogeneous-20m','VP-context-20m'),
        ('Heterogeneous\n20 m','VP context\n20 m'),'Future interaction ablation on frozen context groups')
    print('STAGE6A_QUANTITATIVE_FIGURES_PASS')

if __name__=='__main__':main()
