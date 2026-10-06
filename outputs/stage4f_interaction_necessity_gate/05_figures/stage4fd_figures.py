"""CSV-grounded amplitude, compensation, attention and complete-head figures."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'04_evaluation/diagnostics'))
from stage4fd_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image

FOCUS=('vehicle.moving','vehicle.parked','Pedestrian 0-1m','Pedestrian 2-5m','Pedestrian 5-10m')
LABELS=('V moving','V parked','P 0–1m','P 2–5m','P 5–10m')
RAW_COLOR='#9ABACB';EFFECTIVE_COLOR='#5789AC';A_COLOR='#D59A69'

def csv_rows(p):
    with p.open() as f:return list(csv.DictReader(f))

def export(fig,name,sources,checks):
    assert checks and max(x['absolute_difference'] for x in checks)<1e-8
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        labels=[ax.title,ax.xaxis.label,ax.yaxis.label,*ax.texts]
        labels.extend(l for v,l in zip(ax.get_xticks(),ax.get_xticklabels()) if min(ax.get_xlim())<=v<=max(ax.get_xlim()))
        labels.extend(l for v,l in zip(ax.get_yticks(),ax.get_yticklabels()) if min(ax.get_ylim())<=v<=max(ax.get_ylim()))
        for label in labels:
            if label.get_visible() and label.get_text():
                box=label.get_window_extent(renderer);assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.x1+1 and box.y1<=fig.bbox.y1+1,(name,label.get_text())
        for legend in fig.legends:
            box=legend.get_window_extent(renderer)
            assert not any(box.overlaps(t.get_window_extent(renderer)) for t in ax.texts if t.get_visible()),'legend/data label overlap'
    paths={}
    for ext in ('png','pdf','svg'):
        p=ROOT/'05_figures'/f'{name}.{ext}';fig.savefig(p,dpi=300,facecolor='white')
        if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text().splitlines())+'\n')
        paths[str(p.relative_to(ROOT))]=sha256(p)
    plt.close(fig)
    with Image.open(ROOT/'05_figures'/f'{name}.png') as im:size=im.size;im.verify()
    assert ET.parse(ROOT/'05_figures'/f'{name}.svg').findall('.//{http://www.w3.org/2000/svg}text')
    atomic_json(ROOT/'05_figures'/f'{name}_audit.json',{'status':'PASS','CSV_source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
        'plot_vs_CSV_checks':checks,'max_absolute_difference':max(x['absolute_difference'] for x in checks),'exports_sha256':paths,'PNG_dimensions':size,
        'Weighting':'EdgeWeighted','reference1_is_descriptive_not_significance':True,'no_smoothing_or_interpolation':True,'labels_inside_canvas':True,
        'numeric_audit':'unrounded plotted artist arrays vs full-precision CSV; textual bar labels round to3 decimals and heatmap labels to2 for readability'})

def comparison(table,name,title,fields,colors,labels,ylabel):
    fig,ax=plt.subplots(figsize=(7.2,3.6));x=np.arange(len(FOCUS));checks=[];width=.23 if len(fields)==3 else .34
    for j,(field,color,label) in enumerate(zip(fields,colors,labels)):
        values=np.array([float(table[g][field]) for g in FOCUS]);centers=x+(j-(len(fields)-1)/2)*width
        bars=ax.bar(centers,values,width*.95,color=color,label=label)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=7,padding=3)
        checks.extend({'Group':g,'Metric':field,'CSV':float(v),'plotted':float(b.get_height()),'absolute_difference':abs(v-b.get_height())} for g,v,b in zip(FOCUS,values,bars))
    ax.set_xticks(x);ax.set_xticklabels([f'{label}\nn={table[g]["TargetCount"]}' for label,g in zip(LABELS,FOCUS)])
    ax.set_ylabel(ylabel);ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    fig.suptitle(title,y=.96,fontsize=11);fig.legend(*ax.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.54,.86),ncol=len(fields),fontsize=8)
    fig.text(.5,.025,'Full-horizon targets; EdgeWeighted; complete incoming neighborhoods. GT bins offline only.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.11,right=.98,bottom=.24,top=.77)
    export(fig,name,[ROOT/'06_tables/stage4fd_effective_strength_by_group.csv'],checks)

def compensation(table):
    fig,ax=plt.subplots(figsize=(7.2,3.3));values=np.array([float(table[g]['CompensationIndex']) for g in FOCUS]);bars=ax.bar(np.arange(5),values,.56,color=EFFECTIVE_COLOR)
    ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8,padding=4)
    ax.axhline(1,color='#647782',ls='--',lw=1);ax.text(4.48,1.035,'Raw-unchanged reference',ha='right',fontsize=7,color='#59646C')
    ax.set_xticks(np.arange(5));ax.set_xticklabels(LABELS);ax.set(ylim=(0,1.15),ylabel='Compensation index',title='Compensation index across motion groups')
    fig.text(.5,.035,'EdgeWeighted. Reference 1 denotes unchanged raw amplitude; it is not a significance threshold.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.12,right=.98,bottom=.21,top=.87)
    checks=[{'Group':g,'Metric':'CompensationIndex','CSV':float(v),'plotted':float(b.get_height()),'absolute_difference':abs(v-b.get_height())} for g,v,b in zip(FOCUS,values,bars)]
    export(fig,'stage4fd_compensation_index',[ROOT/'06_tables/stage4fd_effective_strength_by_group.csv'],checks)

def layer_figure():
    lp=ROOT/'06_tables/stage4fd_layer_statistics.csv';hp=ROOT/'06_tables/stage4fd_layer_head_statistics.csv'
    layers=[r for r in csv_rows(lp) if r['Group']=='Pedestrian' and r['Weighting']=='EdgeWeighted'];heads=[r for r in csv_rows(hp) if r['Group']=='Pedestrian' and r['Weighting']=='EdgeWeighted']
    assert len(layers)==3 and len(heads)==24
    fig=plt.figure(figsize=(7.2,3.3));left=fig.add_axes([.09,.25,.29,.48]);right=fig.add_axes([.50,.25,.41,.48]);checks=[]
    for field,color,marker,label in (('RawAmplification',RAW_COLOR,'s','Raw amplification'),('EffectiveRetention',EFFECTIVE_COLOR,'o','Effective retention')):
        values=np.array([float(r[field]) for r in layers]);line,=left.plot(np.arange(1,4),values,color=color,marker=marker,lw=1.2,ms=4,label=label)
        checks.extend({'Layer':i+1,'Metric':field,'CSV':float(v),'plotted':float(p),'absolute_difference':abs(v-p)} for i,(v,p) in enumerate(zip(values,line.get_ydata())))
    left.set_xticks([1,2,3]);left.set_xticklabels(['Layer1','Layer2','Layer3']);left.set(ylabel='Ratio relative to Stage4A',ylim=(0,.21))
    matrix=np.empty((3,8))
    for row in heads:matrix[int(row['Layer'])-1,int(row['Head'])-1]=float(row['CompensationIndex'])
    im=right.imshow(matrix,cmap='Blues',vmin=0,vmax=max(2.2,float(matrix.max())),aspect='auto',interpolation='nearest')
    for row in heads:
        li,h=int(row['Layer'])-1,int(row['Head'])-1;value=float(im.get_array()[li,h]);expected=float(row['CompensationIndex'])
        checks.append({'Layer':li+1,'Head':h+1,'Metric':'CompensationIndex','CSV':expected,'plotted':value,'absolute_difference':abs(value-expected)})
        right.text(h,li,f'{value:.2f}',ha='center',va='center',fontsize=6.5,color='white' if value>1.2 else '#293D4A')
    right.set_xticks(np.arange(8));right.set_xticklabels(np.arange(1,9));right.set_yticks([0,1,2]);right.set_yticklabels(['L1','L2','L3']);right.set_xlabel('Head')
    cax=fig.add_axes([.93,.25,.018,.48]);fig.colorbar(im,cax=cax);cax.tick_params(labelsize=7)
    fig.suptitle('Pedestrian: pooled layer attenuation and a localized head exception',fontsize=10,y=.96)
    fig.legend(*left.get_legend_handles_labels(),loc='center',bbox_to_anchor=(.24,.84),fontsize=7,ncol=1)
    fig.text(.70,.84,'Compensation index: all 24 heads',ha='center',fontsize=8)
    fig.text(.5,.045,'EdgeWeighted; original unsmoothed measurements. Head exception does not establish a causal failure mechanism.',ha='center',fontsize=7)
    export(fig,'stage4fd_layer_compensation',[lp,hp],checks)

def main():
    verify();rows=csv_rows(ROOT/'06_tables/stage4fd_effective_strength_by_group.csv');table={r['Group']:r for r in rows if r['Weighting']=='EdgeWeighted'}
    comparison(table,'stage4fd_raw_effective_bias_by_group','Raw relation amplitude and effective injected bias',
        ('A_MeanAbsRawBias','F_MeanAbsRawBias','F_MeanAbsEffectiveBias'),(A_COLOR,RAW_COLOR,EFFECTIVE_COLOR),('Stage4A raw','Stage4F raw','Stage4F effective'),'Mean absolute bias (logit units)')
    compensation(table)
    comparison(table,'stage4fd_attention_shift_comparison','Actual attention change from relation modulation',
        ('A_AttentionL1Shift','F_AttentionL1Shift'),(A_COLOR,EFFECTIVE_COLOR),('Stage4A','Stage4F'),'Incoming attention L1 shift')
    layer_figure();print('STAGE4FD_FIGURES_PASS',flush=True)

if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False,'axes.spines.right':False,'axes.spines.top':False,'path.simplify':False})
    main()
