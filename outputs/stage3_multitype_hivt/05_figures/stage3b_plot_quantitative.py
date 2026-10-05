"""Source-only, unsmoothed quantitative Type Embedding ablation figures."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'));sys.path.insert(0,str(ROOT/'04_evaluation'))
from stage3b_common import read_json,atomic_json,sha256,write_csv,SUMMARY,PREREG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS=('#73808A','#2478A5');LABELS=('No-Type','Type Embedding')

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def export(fig,stem,sources,extra=None):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        for artist in list(ax.texts)+[ax.xaxis.label,ax.yaxis.label,ax.title]:
            if not artist.get_text():continue
            box=artist.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1,'Text clipped'
    out={}
    for ext in ('png','pdf','svg'):
        path=Path(str(stem)+'.'+ext);fig.savefig(path,dpi=300,facecolor='white')
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        out[str(path.relative_to(ROOT))]=sha256(path)
    plt.close(fig)
    atomic_json(Path(str(stem)+'_audit.json'),{'status':'PASS','source_SHA256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        'exports_SHA256':out,'smoothing':False,'fabricated_data':False,'text_inside_canvas':True,**(extra or {})})

def curve_rows():
    base_paths=[ROOT/'03_no_type_baseline/stage3_no_type_train_curve.csv',
        ROOT/'03_no_type_baseline/stage3a_plus_nll_extension_curve.csv',
        ROOT/'03_no_type_baseline/stage3a_plusplus_nll_curve.csv']
    base=[]
    for p in base_paths:
        for row in rows(p):base.append((int(float(row['global_step'])),float(row['VAL_overall_FDE'])))
    base.sort();assert [s for s,v in base]==list(range(500,21001,500))
    type_path=ROOT/'03_type_embedding/stage3b_training_curve.csv';typed=rows(type_path)
    assert [int(float(r['global_step'])) for r in typed]==list(range(500,int(read_json(SUMMARY)['final_executed_global_step'])+1,500))
    return base,typed,base_paths+[type_path]

def plot_training():
    base,typed,sources=curve_rows();summary=read_json(SUMMARY)
    source=ROOT/'05_figures/stage3b_training_comparison_source.csv'
    write_csv(source,[{'Method':'No-Type','global_step':s,'overall_minFDE6':v} for s,v in base]+[
        {'Method':'Type Embedding','global_step':r['global_step'],'overall_minFDE6':r['VAL_overall_FDE']} for r in typed])
    fig,ax=plt.subplots(figsize=(7.4,4.2),layout='constrained')
    for label,color,points in zip(LABELS,COLORS,(base,[(int(float(r['global_step'])),float(r['VAL_overall_FDE'])) for r in typed])):
        ax.plot([p[0] for p in points],[p[1] for p in points],marker='o',markersize=3,lw=1.1,color=color,label=label)
    no_type_fde=read_json(ROOT/'07_checkpoints/stage3a_final_frozen_checkpoint_manifest.json')['overall_minFDE6']
    best=summary['best_global_step'];fde=summary['selected_full_horizon_metrics']['overall']['minFDE6']
    ax.scatter([21000,best],[no_type_fde,fde],s=75,marker='*',color=COLORS,zorder=6)
    ax.axvline(5000,color='#A4ABB0',lw=.7,ls=':')
    ax.set(xlabel='Global optimizer step',ylabel='Full-horizon overall minFDE6 (m)',xlim=(0,21500))
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    ax.text(.98,.95,f'No-Type frozen best: 21000 / {no_type_fde:.6f} m\nType best: {best} / {fde:.6f} m',
        transform=ax.transAxes,ha='right',va='top',fontsize=8,bbox={'facecolor':'white','alpha':.9,'edgecolor':'none'})
    export(fig,ROOT/'05_figures/stage3b_training_comparison',sources+[source],{'actual_markers_every500_steps':True})
    # Loss points are means of the preceding500 actual optimizer steps, saved at real VAL points.
    fig,ax=plt.subplots(figsize=(7.4,3.8),layout='constrained')
    for key,label,color in zip(('train_loss','regression_loss','classification_loss'),('Total','Regression','Classification'),('#2478A5','#62717B','#BA8549')):
        ax.plot([int(float(r['global_step'])) for r in typed],[float(r[key]) for r in typed],marker='o',ms=3,lw=1,color=color,label=label)
    ax.axvline(5000,color='#A4ABB0',ls=':',lw=.8);ax.set(xlabel='Global optimizer step',ylabel='Mean batch loss over preceding500 updates')
    ax.legend(frameon=False);export(fig,ROOT/'03_type_embedding/stage3b_loss_curve',[sources[-1]])
    fig,ax=plt.subplots(figsize=(7.4,3.8),layout='constrained')
    for group,color in zip(('overall','vehicle','pedestrian','bicycle','vehicle.moving'),('#2478A5','#62717B','#569580','#BA8549','#8B6B94')):
        ax.plot([int(float(r['global_step'])) for r in typed],[float(r['VAL_'+group+'_FDE']) for r in typed],marker='o',ms=3,lw=1,color=color,label=group)
    ax.axvline(5000,color='#A4ABB0',ls=':',lw=.8);ax.set(xlabel='Global optimizer step',ylabel='Full-horizon minFDE6 (m)');ax.legend(frameon=False,ncol=2)
    export(fig,ROOT/'03_type_embedding/stage3b_val_fde_curve',[sources[-1]])

def grouped_bars(path,metric,groups,stem,ylabel):
    data={r['Group']:r for r in rows(path)};fig,ax=plt.subplots(figsize=(7.4,4),layout='constrained');x=np.arange(len(groups));width=.29
    maximum=0
    for i,(method,color) in enumerate(zip(('NoType','Type'),COLORS)):
        values=[float(data[g][method+'_'+metric]) for g in groups];maximum=max(maximum,max(values))
        bars=ax.bar(x+(i-.5)*.40,values,width,color=color,label=LABELS[i])
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3,fontsize=7.5)
    ax.set_xticks(x,[g.replace(' >','\n>') for g in groups]);ax.set_ylabel(ylabel);ax.set_ylim(0,maximum*1.18)
    ax.legend(loc='upper left',frameon=False,ncol=2);export(fig,ROOT/'05_figures'/stem,[path])

def distributions():
    from stage3b_common import ACTORS,BASE_ACTORS
    from stage3b_pairing_ledger import actor_key
    baseline={actor_key(r):r for r in rows(BASE_ACTORS)};typed=rows(ACTORS);values=[]
    source=[]
    for cls in ('vehicle','pedestrian','bicycle'):
        selected=[r for r in typed if r['horizon']=='full_horizon' and r['agent_type']==cls]
        delta=np.array([float(r['minFDE6'])-float(baseline[actor_key(r)]['minFDE6']) for r in selected]);values.append(delta)
        source.append({'AgentType':cls,'Count':len(delta),'Mean_delta_m':float(delta.mean()),
                       **{'Quantile_'+str(q):float(np.quantile(delta,q)) for q in (0,.01,.05,.25,.5,.75,.95,.99,1)}})
    p=ROOT/'05_figures/stage3b_paired_delta_fde_source.csv';write_csv(p,source)
    fig,ax=plt.subplots(figsize=(6.5,4),layout='constrained')
    ax.boxplot(values,tick_labels=['Vehicle','Pedestrian','Bicycle'],showfliers=False,widths=.5,
        medianprops={'color':'#2478A5','linewidth':1.7},boxprops={'color':'#677780'},whiskerprops={'color':'#677780'})
    ax.axhline(0,color='#252B30',lw=1.2);ax.set(ylabel='Actor-window delta minFDE6 (Type - No-Type; m)')
    ax.set_title('Descriptive distribution; formal CIs cluster by scene',fontsize=9)
    export(fig,ROOT/'05_figures/stage3b_paired_delta_fde',[ACTORS,BASE_ACTORS,p],{
        'outliers_hidden_in_boxplot_only':True,'all_data_used_in_quantiles_and_bootstrap':True,
        'actor_windows_are_correlated':True,'center':'median','boxes':'25th to75th percentile',
        'whiskers':'most extreme observed points within1.5 IQR','descriptive_no_actor_level_significance_test':True,
        'counts_by_class':{r['AgentType']:r['Count'] for r in source},
        'formal_interval':'paired150-scene cluster percentile bootstrap;1000 replicates seed2022'})

def embedding_plot():
    p=ROOT/'04_evaluation/stage3b_type_embedding_vectors.json';data=read_json(p)
    values=np.array(data['cosine_similarity_matrix']);names=[name.capitalize() for name in data['class_order']]
    fig,ax=plt.subplots(figsize=(4.4,4),layout='constrained');im=ax.imshow(values,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(3),names);ax.set_yticks(range(3),[f'{name}\nL2={data["L2_norm"][name.lower()]:.3f}' for name in names])
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{values[i,j]:.3f}',ha='center',va='center',color='white' if abs(values[i,j])>.6 else '#252B30')
    fig.colorbar(im,ax=ax,label='Cosine similarity',shrink=.8);ax.set_title('Type vector cosine similarity',fontsize=8)
    export(fig,ROOT/'05_figures/stage3b_type_embedding_similarity',[p])

def main():
    assert read_json(ROOT/'00_manifest/stage3b_figure_contract.json')['backend']=='python'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','path.simplify':False})
    plot_training();path=ROOT/'06_tables/stage3b_type_embedding_ablation.csv'
    grouped_bars(path,'minFDE6',('Overall','Vehicle','Pedestrian','Bicycle'),'stage3b_per_class_fde_comparison','minFDE6 (m)')
    grouped_bars(path,'Top1FDE6',('Overall','Vehicle','Pedestrian','Bicycle','Vehicle >5m','Pedestrian >5m'),'stage3b_top1_fde_comparison','Top1FDE6 (m)')
    distributions();embedding_plot();print('STAGE3B_QUANTITATIVE_FIGURES=PASS',flush=True)

if __name__=='__main__':main()
