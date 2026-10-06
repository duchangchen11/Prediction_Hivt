"""CSV-sourced comparison and routing figures; editable vectors and faithful data."""
from pathlib import Path
import sys
import csv
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage5a_common import atomic_json, read_json, sha256
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False,
    'path.simplify':False})
COLORS=('#8F9AA3','#3F83A5')


def load_csv(name):
    path=ROOT/'06_tables'/name
    with path.open() as f:rows=list(csv.DictReader(f))
    return path,rows


def save(fig,name,audit):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for text in fig.texts:
        box=text.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1
    exports={}
    for ext in ('png','pdf','svg'):
        path=ROOT/'05_figures'/(name+'.'+ext)
        fig.savefig(path,dpi=300,facecolor='white')
        if ext=='svg':
            path.write_text('\n'.join(x.rstrip() for x in path.read_text().splitlines())+'\n')
        exports[str(path.relative_to(ROOT))]=sha256(path)
    plt.close(fig)
    audit.update(status='PASS',exports_SHA256=exports,annotation_rounding_only=True,
                 editable_SVG_text=True,PDF_fonttype=42)
    atomic_json(ROOT/'05_figures'/(name+'_audit.json'),audit)
    return audit


def compare(name,groups,labels,title,source_name):
    source,rows=load_csv(source_name);lookup={r['Group']:r for r in rows}
    bootstrap=ROOT/'04_evaluation/stage5a_bootstrap_ci.json'
    ci=read_json(bootstrap)['groups']
    data=np.array([[float(lookup[g]['Stage3B_minFDE6']),float(lookup[g]['Stage5A_minFDE6'])] for g in groups])
    count=[int(lookup[g]['Count']) for g in groups]
    delta=np.array([ci[g]['minFDE6']['delta'] for g in groups])
    interval=np.array([ci[g]['minFDE6']['CI95'] for g in groups])
    assert np.max(np.abs(data[:,1]-data[:,0]-delta))<1e-12
    fig,(left,right)=plt.subplots(1,2,figsize=(7.2,3.4),gridspec_kw={'width_ratios':[1.15,1]})
    x=np.arange(len(groups));width=.35;artist_error=0.
    for j,label in enumerate(('Stage3B','Stage5A')):
        bars=left.bar(x+(j-.5)*width,data[:,j],width,color=COLORS[j],label=label)
        artist_error=max(artist_error,float(np.max(np.abs(np.array([p.get_height() for p in bars])-data[:,j]))))
        for rect,value in zip(bars,data[:,j]):
            left.annotate(f'{value:.3f}',(rect.get_x()+rect.get_width()/2,rect.get_height()),
                          xytext=(0,3+10*j),textcoords='offset points',ha='center',va='bottom',fontsize=6.5)
    assert artist_error==0.
    left.set_xticks(x,labels);left.tick_params(axis='x',labelsize=7)
    left.set_ylabel('Full-horizon minFDE6 (m)');left.set_ylim(0,float(data.max())*1.2)
    left.legend(loc='upper left',bbox_to_anchor=(0,1.15),ncol=2,fontsize=7)
    left.set_title('a  Frozen baseline comparison',loc='left',fontsize=8,pad=25)
    y=np.arange(len(groups))[::-1]
    for yy,d,(lo,hi) in zip(y,delta,interval):
        right.plot((lo,hi),(yy,yy),color=COLORS[1],lw=1.5)
        right.plot(d,yy,'o',color=COLORS[1],ms=4)
    right.axvline(0,color='#7B858D',lw=.8,ls='--')
    right.set_yticks(y,labels);right.tick_params(axis='y',labelsize=7)
    right.set_ylim(-.65,len(groups)-.35)
    right.set_xlabel('Stage5A − Stage3B minFDE6 (m)')
    right.set_title('b  Paired scene-bootstrap 95% CI',loc='left',fontsize=8,pad=25)
    for ax in (left,right):ax.grid(axis='y' if ax==left else 'x',alpha=.12,lw=.5);ax.set_axisbelow(True)
    fig.suptitle(title,y=.98,fontsize=10)
    fig.subplots_adjust(left=.09,right=.98,bottom=.25,top=.77,wspace=.52)
    fig.text(.5,.08,'n (actor-windows): '+', '.join(f'{l.replace(chr(10)," ")}={n:,}' for l,n in zip(labels,count)),ha='center',fontsize=6.5)
    fig.text(.5,.025,'Official VAL150 · 1 training seed · 1,000 paired scene resamples · negative Δ favors Stage5A',ha='center',fontsize=6.5)
    return save(fig,name,{'source_csv':str(source.relative_to(ROOT)),'source_csv_sha256':sha256(source),
        'bootstrap_json_sha256':sha256(bootstrap),'groups':groups,'counts':count,'bar_artist_max_abs_diff':artist_error,
        'data_values':data.tolist(),'paired_delta':delta.tolist(),'CI95':interval.tolist(),'bar_annotation_decimals':3})


def router():
    source,rows=load_csv('stage5a_router_distributions.csv')
    stats_source,stats=load_csv('stage5a_router_statistics.csv')
    groups=('vehicle.moving','vehicle.parked','Pedestrian 0-1m','Pedestrian 2-5m','Pedestrian 5-10m')
    labels=('Moving\nvehicle','Parked\nvehicle','Pedestrian\n0–1 m','Pedestrian\n2–5 m','Pedestrian\n5–10 m')
    fig,axes=plt.subplots(1,5,figsize=(7.2,2.9),sharey=True)
    checks=[]
    for ax,group,label in zip(axes,groups,labels):
        rr=next(r for r in stats if r['Group']==group and r['Population']=='full_horizon')
        for expert,color in ((1,'#3F83A5'),(2,'#CA9957')):
            selected=[r for r in rows if r['Group']==group and r['Population']=='full_horizon' and int(r['Expert'])==expert]
            assert len(selected)==20
            values=np.array([float(r['Density']) for r in selected])
            edges=np.array([float(r['BinLower']) for r in selected]+[float(selected[-1]['BinUpper'])])
            assert abs(np.sum(values*np.diff(edges))-1)<1e-12
            artist=ax.stairs(values,edges,color=color,lw=1.3,label=f'Expert {expert}')
            checks.append({'Group':group,'Expert':expert,'density_integral':float(np.sum(values*np.diff(edges))),
                           'artist_max_abs_diff':float(np.max(np.abs(artist.get_data().values-values)))})
        ax.set_title(label,fontsize=8,pad=9);ax.set_xlim(0,1);ax.set_xticks([0,.5,1])
        ax.tick_params(labelsize=7);ax.grid(axis='y',alpha=.12,lw=.5)
        ax.text(.5,-.24,f"n={int(rr['Count']):,}\nmean r₁={float(rr['r1_mean']):.3f}\nmean r₂={float(rr['r2_mean']):.3f}",
                transform=ax.transAxes,ha='center',va='top',fontsize=6.5)
    axes[0].set_ylabel('Probability density');handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.9),ncol=2,fontsize=7)
    fig.suptitle('Actor-level probabilities of two learned experts',fontsize=10,y=.98)
    fig.subplots_adjust(left=.07,right=.99,top=.69,bottom=.33,wspace=.3)
    fig.text(.5,.015,'Full-horizon actor-windows · bin width 0.05 · expert names carry no preassigned motion semantics',ha='center',fontsize=6.5)
    assert all(x['artist_max_abs_diff']==0 for x in checks)
    return save(fig,'stage5a_router_distribution',{'source_csv':str(source.relative_to(ROOT)),
        'source_csv_sha256':sha256(source),'statistics_csv_sha256':sha256(stats_source),'groups':list(groups),
        'density_checks':checks,'annotation_decimals':3})


def main():
    assert read_json(ROOT/'00_manifest/stage5a_figure_contract.json')['backend']=='python'
    assert read_json(ROOT/'04_evaluation/stage5a_pairing_audit.json')['status']=='PASS'
    compare('stage5a_main_fde_comparison',('overall','vehicle','pedestrian'),('Overall','Vehicle','Pedestrian'),
            'Motion-aware residual decoder: main prediction comparison','stage5a_comparison_all_groups.csv')
    compare('stage5a_motion_group_comparison',('vehicle.moving','vehicle.parked','Pedestrian <5m','Pedestrian 5-10m'),
            ('Moving\nvehicle','Parked\nvehicle','Pedestrian\n<5 m','Pedestrian\n5–10 m'),
            'Motion-sensitive vehicle and pedestrian groups','stage5a_comparison_all_groups.csv')
    router()
    compare('stage5a_pedestrian_motion_bins',tuple(f'Pedestrian {lo}-{hi}m' for lo,hi in ((0,1),(1,2),(2,5),(5,10),(10,20))),
            ('0–1 m','1–2 m','2–5 m','5–10 m','10–20 m'),
            'Pedestrian future-displacement bins (offline analysis)','stage5a_pedestrian_motion_bins.csv')
    print('STAGE5A_QUANTITATIVE_FIGURES_PASS')


if __name__=='__main__':main()
