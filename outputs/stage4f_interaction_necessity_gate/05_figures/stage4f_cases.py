"""Four purposive real actor triptychs; fixed replay batches and original points."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage4f_common import (SceneDataset, atomic_json, read_json, sha256, OLD,
    STAGE3_ROOT, BEST, errors_with_top1, CLASSES)
from stage4f_evaluate import paired, GATES
from stage4f_efficiency import networks
sys.path.insert(0,str(OLD/'05_figures'))
from stage4a_plot_cases import replay, nearest_context, geometry, trajectories
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import MaxNLocator


def choices(maps,membership,gates):
    b,c,d=(maps[n] for n in ('Stage3B','Stage4A','Stage4F'))
    full=[k for k in b if k[-1]=='full_horizon'];used=set();selected=[]
    fde=lambda m,k:float(m[k]['minFDE6'])
    motion=lambda k:float(b[k]['GT_endpoint_displacement_m'])
    groups=[('moving_vehicle_retention','Moving vehicle: retained gain',
        [k for k in full if b[k]['agent_type']=='vehicle' and b[k]['motion_state']=='vehicle.moving' and motion(k)>5 and fde(c,k)<fde(b,k) and fde(d,k)<fde(b,k)],
        lambda k:min(fde(b,k)-fde(c,k),fde(b,k)-fde(d,k))),
        ('low_motion_pedestrian_recovery','Low-motion pedestrian: recovery example',
        [k for k in full if b[k]['agent_type']=='pedestrian' and motion(k)<5 and fde(c,k)>fde(b,k) and fde(d,k)<fde(c,k)],
        lambda k:fde(c,k)-fde(d,k)),
        ('moving_pedestrian','Moving pedestrian: learned interaction gate',
        [k for k in full if b[k]['agent_type']=='pedestrian' and 5<=motion(k)<10 and fde(d,k)<=fde(b,k) and membership[k]['Heterogeneous-20m']=='1'],
        lambda k:float(gates[k]['necessity_gate'])),
        ('degradation','Real degradation relative to Stage3B',
        [k for k in full if b[k]['agent_type'] in ('vehicle','pedestrian') and motion(k)>1 and fde(d,k)>fde(b,k)],
        lambda k:fde(d,k)-fde(b,k))]
    for name,label,eligible,score in groups:
        candidates=[k for k in eligible if k not in used]
        assert candidates,'Requested illustrative case unavailable: '+name
        # Prefer frozen heterogeneous20m context, then the specified effect.
        k=max(candidates,key=lambda k:(membership[k]['Heterogeneous-20m']=='1',score(k),k))
        used.add(k);selected.append((name,label,k))
    return selected


def draw(source,stem):
    lo,hi,zoomlo,zoomhi,inset_rule=geometry(source)
    # Data-derived aspect gives compact equal-aspect rectangles for straight
    # trajectories; a separate header keeps annotations off the original points.
    ratio=float((hi[1]-lo[1])/(hi[0]-lo[0]))
    body_width=min(3.0,3.2/ratio);body_height=body_width*ratio
    add_inset=inset_rule['add_inset'];header=2.15 if add_inset else 1.25
    bottom=1.05;figure_height=body_height+header+bottom
    starts=np.array(source['lane_positions_m']).reshape(-1,2);vectors=np.array(source['lane_vectors_m']).reshape(-1,2)
    segments=np.stack((starts,starts+vectors),axis=1);segments=segments[((segments.max(1)>=lo)&(segments.min(1)<=hi)).all(1)]
    fig=plt.figure(figsize=(10.8,figure_height));lefts=(.06,.375,.69)
    axes=[fig.add_axes([left+(.278-body_width/10.8)/2,bottom/figure_height,body_width/10.8,body_height/figure_height]) for left in lefts]
    fig.suptitle(source['case_kind'],fontsize=12,y=1-.12/figure_height)
    fig.text(.5,1-.40/figure_height,f"{source['scene_name']} | {source['agent_type']} | GT endpoint displacement {source['GT_endpoint_displacement_m']:.2f}m",ha='center',fontsize=9)
    audits=[]
    for left,ax,panel in zip(lefts,axes,source['panels']):
        styles=trajectories(ax,source,panel,segments);assert all(r['marker_count']==12 for r in styles)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='t0 ego x (m)');ax.tick_params(labelsize=7)
        label_y=(bottom+body_height+(1.32 if add_inset else .43))/figure_height
        fig.text(left,label_y,panel['method'],fontsize=10,va='top')
        label=f"Best FDE {panel['recomputed_metrics']['minFDE6']:.3f}m\nTop1 FDE {panel['recomputed_metrics']['Top1FDE6']:.3f}m"
        if panel['method']=='Stage4F':label+=f"\nNecessity gate {source['necessity_gate']:.3f}"
        fig.text(left+.278,label_y,label,ha='right',va='top',fontsize=8)
        if add_inset:
            zoom=fig.add_axes([left+.065,(bottom+body_height+.14)/figure_height,.15,.70/figure_height])
            trajectories(zoom,source,panel,segments,history=False,late_only=True)
            center=(zoomlo+zoomhi)/2;span=zoomhi-zoomlo;aspect=.15*10.8/.70
            w=max(float(span[0]),float(span[1])*aspect);h=max(float(span[1]),float(span[0])/aspect)
            low=center-np.array([w,h])/2;high=center+np.array([w,h])/2
            zoom.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]));zoom.tick_params(labelsize=6)
            zoom.set_title('Last 6 future points',fontsize=7,pad=3)
            zoom.xaxis.set_major_locator(MaxNLocator(2));zoom.yaxis.set_major_locator(MaxNLocator(2))
        audits.append({'method':panel['method'],'styles':styles,'recomputed_metrics':panel['recomputed_metrics'],'metric_differences':panel['metric_absolute_differences_m']})
    axes[0].set_ylabel('t0 ego y (m)')
    handles,labels=axes[0].get_legend_handles_labels();ordered=[labels.index(n) for n in ('History','GT','Best-FDE','Top1')]
    fig.legend([handles[i] for i in ordered],[labels[i] for i in ordered],loc='lower center',bbox_to_anchor=(.5,.42/figure_height),ncol=4,fontsize=9,frameon=False)
    fig.text(.5,.27/figure_height,'12 original future markers; Best-FDE selected with GT; Top1 selected by predicted probability.',ha='center',fontsize=8)
    fig.text(.5,.10/figure_height,'GT motion bin only used for offline analysis. Purposive examples; no causal interaction claim.',ha='center',fontsize=8,color='#56656D')
    exports={}
    for suffix in ('png','pdf','svg'):
        p=Path(str(stem)+'.'+suffix);fig.savefig(p,dpi=300,facecolor='white')
        if suffix=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        exports[str(p.relative_to(ROOT))]=sha256(p)
    plt.close(fig)
    with Image.open(str(stem)+'.png') as im:im.verify()
    return {'status':'PASS','panels':audits,'same_actor_and_GT':True,'same_xy_limits':True,'xlim':[float(lo[0]),float(hi[0])],'ylim':[float(lo[1]),float(hi[1])],
        '12_future_markers_each_trace':True,'equal_aspect':True,'compact_data_derived_rectangle':True,'inset_last6_points':bool(add_inset),'no_smoothing_or_interpolation':True,'gate_from_history_only':True,'exports_sha256':exports}


@torch.no_grad()
def main():
    maps,membership=paired()
    with GATES.open() as f:gates={tuple(r[k] for k in ('scene_token','sample_token','instance_token','horizon')):r for r in csv.DictReader(f) if r['horizon']=='full_horizon'}
    selected=choices(maps,membership,gates);ds=SceneDataset('val');lookup={(r['scene_token'],r['sample_token']):i for i,r in enumerate(ds.rows)}
    models=networks();figures=[]
    for name,label,key in selected:
        row=maps['Stage4F'][key];index=lookup[key[:2]];graph=ds[index];node=int(row['node_in_graph'])
        assert graph.instance_tokens[node]==key[2] and graph.future_mask[node].all()
        panels=[]
        for method,model in models.items():
            panel=replay(ds,index,node,model,16);source=maps[method][key]
            differences={f:abs(v-float(source[f])) for f,v in panel['recomputed_metrics'].items()}
            assert max(differences.values())<1e-4,(method,name,differences)
            assert panel['best_FDE_mode_zero_based']==int(source['best_mode']) and panel['top1_mode_zero_based']==int(source['top1_mode'])
            panel.update(method=method,source_metrics={f:float(source[f]) for f in differences},metric_absolute_differences_m=differences)
            panels.append(panel)
        source={'name':name,'case_kind':label,**{f:row[f] for f in ('scene_name','scene_token','sample_token','instance_token','agent_type','motion_state')},
            'node_in_graph':node,'GT_endpoint_displacement_m':float(row['GT_endpoint_displacement_m']),
            'necessity_gate':float(gates[key]['necessity_gate']),'gate_features':{f:gates[key][f] for f in gates[key] if f.startswith(('log1p_','type_','min_','heterogeneous_'))},
            'history_trajectory_m':graph.positions[node,:5].tolist(),'history_mask':graph.history_mask[node].tolist(),
            'GT_trajectory_m':graph.positions[node,5:].tolist(),'future_mask':graph.future_mask[node].tolist(),
            'lane_positions_m':graph.lane_positions.tolist(),'lane_vectors_m':graph.lane_vectors.tolist(),
            't0_spatial_context':nearest_context(graph,node),'panels':panels,'frozen_membership':membership[key],
            'GT_fingerprint':row['GT_trajectory_sha256'],'coordinate_frame':'t0 ego meters',
            'selection':'posthoc deterministic illustrative effects, prefer frozen hetero20m context; not representative population evidence',
            'GT_motion_bin_only_offline':True,'gate_future_inputs':False,'trajectory_edits':False}
        p=ROOT/'04_evaluation'/f'stage4f_case_{name}.json';atomic_json(p,source)
        audit=draw(read_json(p),ROOT/'05_figures'/f'stage4f_case_{name}');audit.update(source_json_sha256=sha256(p))
        atomic_json(ROOT/'05_figures'/f'stage4f_case_{name}_audit.json',audit)
        figures.append({'name':name,'case_kind':label,'actor_key':key,'source_json':str(p.relative_to(ROOT)),'source_sha256':sha256(p),
            'gate':source['necessity_gate'],'B_FDE':panels[0]['recomputed_metrics']['minFDE6'],'C_FDE':panels[1]['recomputed_metrics']['minFDE6'],'D_FDE':panels[2]['recomputed_metrics']['minFDE6']})
        print('CASE_PASS',name,figures[-1],flush=True)
    ds.clear();atomic_json(ROOT/'04_evaluation/stage4f_qualitative_case_manifest.json',{'status':'PASS','cases':figures,'count':len(figures),'matched_triptychs':True,'selected_after_quantitative_lock':True,'purposive_examples_not_causal_or_representative':True})


if __name__=='__main__':
    torch.set_num_threads(4)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
    main()
