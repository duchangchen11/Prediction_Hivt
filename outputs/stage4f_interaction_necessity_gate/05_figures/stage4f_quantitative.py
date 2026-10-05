"""Actual three-model measurements; paired intervals and unsmoothed500points."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage4f_common import read_json, atomic_json, sha256, OLD, STAGE3_ROOT, CURVE
from stage4f_evaluate import paired, select_keys, GATES, BINS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

COLORS=('#89959E','#D59A69','#5789AC')
METHODS=('Stage3B','Stage4A','Stage4F')


def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def export(fig,name,sources):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        xticks=[label for value,label in zip(ax.get_xticks(),ax.get_xticklabels()) if min(ax.get_xlim())<=value<=max(ax.get_xlim())]
        yticks=[label for value,label in zip(ax.get_yticks(),ax.get_yticklabels()) if min(ax.get_ylim())<=value<=max(ax.get_ylim())]
        for text in (ax.title,ax.xaxis.label,ax.yaxis.label,*xticks,*yticks,*ax.texts):
            if not text.get_visible() or not text.get_text():continue
            bb=text.get_window_extent(renderer)
            assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.x1+1 and bb.y1<=fig.bbox.y1+1,(name,text.get_text(),bb)
        for legend in fig.legends:
            box=legend.get_window_extent(renderer)
            assert not any(box.overlaps(label.get_window_extent(renderer)) for label in ax.texts if label.get_visible()),(name,'legend overlaps data labels')
    exports={}
    for ext in ('png','pdf','svg'):
        p=ROOT/'05_figures'/f'{name}.{ext}';fig.savefig(p,dpi=300,facecolor='white')
        if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        exports[str(p.relative_to(ROOT))]=sha256(p)
    plt.close(fig)
    with Image.open(ROOT/'05_figures'/f'{name}.png') as im:
        size=im.size;im.verify()
    svg=ET.parse(ROOT/'05_figures'/f'{name}.svg');assert svg.findall('.//{http://www.w3.org/2000/svg}text')
    atomic_json(ROOT/'05_figures'/f'{name}_audit.json',{'status':'PASS','sources_sha256':{str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha256(p) for p in sources},
        'exports_sha256':exports,'PNG_dimensions':size,'editable_SVG_text':True,'all_labels_inside_canvas':True,'no_smoothing':True,'actual_data_only':True})


def grouped_bar(table,groups,name,title):
    lookup={r['Group']:r for r in table};x=np.arange(len(groups));fig,ax=plt.subplots(figsize=(7.2,3.6))
    for j,method in enumerate(METHODS):
        heights=[float(lookup[g][method+'_minFDE6']) for g in groups]
        bars=ax.bar(x+(j-1)*.24,heights,.23,color=COLORS[j],label=method)
        ax.bar_label(bars,labels=[f'{h:.3f}' for h in heights],fontsize=7,padding=3,rotation=0)
    ax.set_xticks(x);ax.set_xticklabels([g.replace('overall','Overall').replace('vehicle','Vehicle').replace('pedestrian','Pedestrian').replace(' hetero-20m','\nhetero-20m').replace('VP-context-20m','V–P context\n≤20m') for g in groups])
    ax.set_ylabel('minFDE@6 (m)');ax.set_ylim(0,ax.get_ylim()[1]*1.12)
    fig.suptitle(title,y=.95,fontsize=11)
    fig.legend(*ax.get_legend_handles_labels(),ncol=3,loc='center',bbox_to_anchor=(.55,.85))
    fig.text(.5,.025,'Full-horizon actor-window means; official VAL 150; seed 2022.',ha='center',fontsize=8,color='#59646C')
    fig.subplots_adjust(left=.11,right=.98,bottom=.23,top=.77)
    export(fig,name,[ROOT/'06_tables/stage4f_three_model_ablation.csv'])


def pedestrian_bins(table,boot):
    groups=[f'Pedestrian {lo}-{hi}m' for lo,hi in BINS];lookup={r['Group']:r for r in table};x=np.arange(5)
    fig,axes=plt.subplots(2,1,figsize=(7.2,6.2),sharex=True,gridspec_kw={'height_ratios':[1.3,1]})
    for j,method in enumerate(METHODS):
        axes[0].plot(x,[float(lookup[g][method+'_minFDE6']) for g in groups],color=COLORS[j],marker=('o','s','^')[j],ms=5,lw=1.3,label=method)
    axes[0].set_ylabel('minFDE@6 (m)');axes[0].set_ylim(bottom=0);axes[0].legend(ncol=3,loc='upper left');axes[0].set_title('Pedestrian motion regimes',pad=13)
    old=csv_rows(OLD/'06_tables/stage4ae_pedestrian_motion_bin_bootstrap.csv')
    old={r['MotionBin']:r for r in old if r['Metric']=='minFDE6'}
    for j,(label,color) in enumerate((('Stage4A − Stage3B',COLORS[1]),('Stage4F − Stage3B',COLORS[2]))):
        delta=[];lower=[];upper=[]
        for lo,hi in BINS:
            if j==0:r=old[f'{lo}-{hi}m'];d=float(r['Delta_C_minus_B']);l=float(r['CI95Lower']);u=float(r['CI95Upper'])
            else:r=boot['comparisons']['D-B'][f'Pedestrian {lo}-{hi}m']['minFDE6'];d=r['delta'];l,u=r['CI95']
            delta.append(d);lower.append(d-l);upper.append(u-d)
        axes[1].errorbar(x+(j-.5)*.08,delta,yerr=[lower,upper],color=color,marker=('s','^')[j],capsize=3,lw=1,label=label)
    axes[1].axhline(0,color='#66747D',lw=.7,ls='--');axes[1].set_ylabel('Paired Δ minFDE@6 (m)');axes[1].legend(fontsize=8,loc='upper left')
    axes[1].set_xticks(x);axes[1].set_xticklabels([f'{lo}–{hi}m\nn={lookup[g]["Count"]}' for (lo,hi),g in zip(BINS,groups)])
    axes[1].set_xlabel('GT endpoint displacement (offline grouping only)')
    fig.text(.5,.015,'Intervals: 95% paired scene bootstrap, 150 scenes, 1000 replicates. Secondary analyses unadjusted.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.15,right=.98,bottom=.17,top=.93,hspace=.20)
    export(fig,'stage4f_pedestrian_motion_bin_comparison',[ROOT/'06_tables/stage4f_pedestrian_motion_bin_ablation.csv',ROOT/'04_evaluation/stage4f_bootstrap_ci.json',OLD/'06_tables/stage4ae_pedestrian_motion_bin_bootstrap.csv'])


def gate_distributions():
    maps,membership=paired();gates={tuple(r[k] for k in ('scene_token','sample_token','instance_token','horizon')):float(r['necessity_gate']) for r in csv_rows(GATES) if r['horizon']=='full_horizon'}
    groups=('vehicle.moving','vehicle.stopped','vehicle.parked','Pedestrian 0-1m','Pedestrian 2-5m','Pedestrian 5-10m')
    values=[[gates[k] for k in select_keys(maps['Stage3B'],g,membership)] for g in groups]
    fig,ax=plt.subplots(figsize=(7.2,3.8));parts=ax.boxplot(values,patch_artist=True,showfliers=False,widths=.55,whis=(10,90),medianprops={'color':'#293D4A','linewidth':1.3})
    for j,box in enumerate(parts['boxes']):box.set_facecolor('#C1D4DF' if j<3 else '#E9D1BC');box.set_edgecolor('#647782')
    ax.set_ylim(0,1);ax.set_ylabel('Interaction necessity gate');ax.set_xticks(np.arange(1,7))
    ax.set_xticklabels([f'{g.replace("vehicle.","V ").replace("Pedestrian ","P ")}\nn={len(v)}' for g,v in zip(groups,values)],fontsize=8)
    ax.set_title('Learned gate distributions',pad=14)
    fig.text(.5,.045,'Median, IQR box, p10–p90 whiskers; outliers omitted visually. Full raw values retained locally.',ha='center',fontsize=7)
    fig.text(.5,.013,'GT motion bin only used for offline analysis; observational distributions do not establish causality.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.11,right=.98,bottom=.22,top=.86)
    export(fig,'stage4f_gate_distribution_by_group',[GATES,ROOT/'06_tables/stage4f_gate_statistics.csv'])


def training():
    sources=[STAGE3_ROOT/'03_type_embedding/stage3b_training_curve.csv',OLD/'03_type_interaction/stage4a_training_curve.csv',CURVE]
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    for j,(method,path) in enumerate(zip(METHODS,sources)):
        rows=csv_rows(path);steps=[int(float(r['global_step'])) for r in rows];assert all(s%500==0 for s in steps)
        ax.plot(steps,[float(r['VAL_overall_FDE']) for r in rows],label=method,color=COLORS[j],marker=('o','s','^')[j],ms=3,lw=1.2)
    ax.axvline(5000,color='#697780',ls='--',lw=.8);ax.set(xlabel='Optimizer update',ylabel='VAL overall minFDE@6 (m)',title='Training comparison: original 500-step measurements')
    ax.legend(ncol=3);ax.set_ylim(bottom=0);fig.text(.5,.02,'Fixed-scale warm-up: 5,000 updates; restore each model’s best, then original NLL. No smoothing.',ha='center',fontsize=7)
    fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.87)
    export(fig,'stage4f_training_comparison',sources)


def main():
    table=csv_rows(ROOT/'06_tables/stage4f_three_model_ablation.csv');boot=read_json(ROOT/'04_evaluation/stage4f_bootstrap_ci.json')
    grouped_bar(table,('overall','vehicle','pedestrian'),'stage4f_three_model_fde_comparison','Three-model comparison')
    pedestrian_bins(table,boot);gate_distributions()
    grouped_bar(table,('Vehicle hetero-20m','Pedestrian hetero-20m','VP-context-20m'),'stage4f_interaction_context_comparison','Frozen interaction-context groups')
    training();print('QUANTITATIVE_FIGURES_PASS',flush=True)


if __name__=='__main__':
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'path.simplify':False})
    main()
