"""Matched frozen-VAL examples; replay the original 16-window inference batches."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'));sys.path.insert(0,str(ROOT/'04_evaluation'))
from stage3b_common import (CLASSES,MODEL_KEYS,SceneDataset,HiVTLossRecovery,config,
    model_new,model_input,errors_with_top1,GT_fingerprint,atomic_json,read_json,
    sha256,BEST,ACTORS,BASE_ACTORS,SUMMARY,PREREG)
from stage3b_pairing_ledger import actor_key
import numpy as np
import torch
from torch_geometric.data import Batch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from PIL import Image

BASE_CHECKPOINT=ROOT/'07_checkpoints/stage3a_final_frozen_best_overall_minfde.pt'
FIELDS=('minADE6','minFDE6','Top1ADE6','Top1FDE6')

def read_actors(path):
    with path.open() as f:return {actor_key(r):r for r in csv.DictReader(f) if r['horizon']=='full_horizon'}

def case_candidates(base,typed):
    assert base.keys()==typed.keys() and len(typed)==54990
    full=[]
    for key,row in typed.items():
        original=base[key];displacement=float(row['GT_endpoint_displacement_m'])
        full.append((float(row['minFDE6'])-float(original['minFDE6']),key,row,original,displacement))
    selected=[];omitted=[]
    for cls,threshold in (('vehicle',-2.),('pedestrian',-1.)):
        items=[x for x in full if x[2]['agent_type']==cls and x[4]>5 and x[0]<0]
        preferred=[x for x in items if float(x[3]['minFDE6'])>2 and x[0]<=threshold]
        if preferred or items:
            choice=min(preferred or items,key=lambda x:(x[0],x[1]))
            selected.append((f'stage3b_{cls}_improvement_case_001',choice,'improvement',bool(preferred)))
        else:omitted.append({'kind':cls+'_improvement','reason':'No >5m actor with negative delta FDE; no case fabricated.'})
    items=[x for x in full if x[0]>0];preferred=[x for x in items if x[4]>5 and x[0]>=2]
    if preferred or items:
        choice=max(preferred or items,key=lambda x:(x[0],x[1]))
        selected.append(('stage3b_degradation_case_001',choice,'degradation',bool(preferred)))
    else:omitted.append({'kind':'degradation','reason':'No positive delta FDE; no case fabricated.'})
    return full,selected,omitted

def vp_context(graph,node):
    own=int(graph.agent_type[node]);opposite=1-own
    if own not in (0,1):return None
    full=graph.future_mask.all(-1)&graph.target_mask&(graph.agent_type==opposite)
    candidates=torch.where(full)[0].tolist();near=[]
    for other in candidates:
        d0=float(torch.linalg.vector_norm(graph.positions[node,4]-graph.positions[other,4]))
        distances=torch.linalg.vector_norm(graph.positions[node,5:]-graph.positions[other,5:],dim=-1)
        closest=float(distances.min())
        if d0<=20 and closest<=5:
            near.append({'neighbor_node':other,'neighbor_instance_token':graph.instance_tokens[other],
                'neighbor_agent_type':CLASSES[opposite],'t0_distance_m':d0,
                'min_aligned_GT_future_distance_m':closest,'closest_future_timestep':int(distances.argmin())+1,
                'description':'GT proximity / interaction candidate; no causal interaction inferred.'})
    return min(near,key=lambda x:x['min_aligned_GT_future_distance_m']) if near else None

@torch.no_grad()
def replay(ds,index,node,model,typed):
    bs=config()['batch_size'];start=index//bs*bs
    graphs=[ds[i] for i in range(start,min(start+bs,len(ds)))];graph=graphs[index-start]
    data=Batch.from_data_list(graphs).cuda();output_node=int(data.ptr[index-start])+node
    inputs=model_input(data)
    if not typed:del inputs.agent_type
    out=model(inputs);prediction,errors=errors_with_top1(model,out,data)
    modes=prediction[output_node].cpu().numpy();gt=graph.positions[node,5:].numpy()
    best=int(errors['best_mode'][output_node]);top=int(errors['top1_mode'][output_node])
    best_error=np.linalg.norm(modes[best].astype(np.float64)-gt,axis=-1)
    top_error=np.linalg.norm(modes[top].astype(np.float64)-gt,axis=-1)
    metrics=dict(zip(FIELDS,map(float,(best_error.mean(),best_error[-1],top_error.mean(),top_error[-1]))))
    assert modes.shape==(6,12,2) and np.isfinite(modes).all()
    return {'all_mode_trajectories_m':modes.tolist(),'mode_probabilities':out['mode_prob'][output_node].cpu().tolist(),
        'best_FDE_mode_zero_based':best,'top1_mode_zero_based':top,'recomputed_metrics':metrics,
        'reproduced_VAL_batch_start_index':start,'reproduced_VAL_batch_size':len(graphs)}

def checked_replay(ds,index,row,base_row,no_type,type_model):
    node=int(row['node_in_graph']);graph=ds[index]
    assert graph.instance_tokens[node]==row['instance_token'] and graph.future_mask[node].all()
    assert int(graph.agent_type[node])==CLASSES.index(row['agent_type'])
    fingerprint=GT_fingerprint(graph.positions[node,5:],graph.future_mask[node],graph.agent_type[node])
    assert row['GT_trajectory_sha256']==fingerprint['GT_trajectory_sha256']
    assert row['future_mask_bits']==fingerprint['future_mask_bits']=='111111111111'
    panels=[]
    for name,model,source,is_type in (('No-Type',no_type,base_row,False),('Type Embedding',type_model,row,True)):
        panel=replay(ds,index,node,model,is_type)
        difference={key:abs(value-float(source[key])) for key,value in panel['recomputed_metrics'].items()}
        assert max(difference.values())<1e-4,(name,difference)
        assert panel['best_FDE_mode_zero_based']==int(source['best_mode'])
        assert panel['top1_mode_zero_based']==int(source['top1_mode'])
        panel.update({'method':name,'source_metrics':{key:float(source[key]) for key in FIELDS},
            'metric_absolute_differences_m':difference,'metric_tolerance_m':1e-4})
        panels.append(panel)
    return graph,node,panels,fingerprint

def geometry(source):
    history=np.array(source['history_trajectory_m']);mask=np.array(source['history_mask'],dtype=bool)
    gt=np.array(source['GT_trajectory_m']);visible=[history[mask],gt];late=[gt[-6:]]
    overlap=False;turn=False
    for panel in source['panels']:
        modes=np.array(panel['all_mode_trajectories_m']);best=modes[panel['best_FDE_mode_zero_based']];top=modes[panel['top1_mode_zero_based']]
        visible.extend((best,top));late.extend((best[-6:],top[-6:]))
        overlap=overlap or any(np.mean(np.linalg.norm(a-b,axis=-1))<1 for a,b in ((gt,best),(gt,top),(best,top)))
    vectors=np.diff(gt,axis=0);valid=np.linalg.norm(vectors,axis=-1)>.1
    directions=np.unwrap(np.arctan2(vectors[valid,1],vectors[valid,0]))
    if len(directions)>1:turn=float(np.ptp(directions))>np.pi/6
    visible=np.concatenate(visible);lower=visible.min(0);upper=visible.max(0)
    margin=max(1.,float(np.ptp(visible,axis=0).max())*.06)
    # Equal physical aspect with a compact rectangle; text gets a separate header.
    width=max(float(upper[0]-lower[0])+2*margin,8.)
    height=max(float(upper[1]-lower[1])+2*margin,width*.22,3.)
    center=(lower+upper)/2;lo=center-np.array([width,height])/2;hi=center+np.array([width,height])/2
    late=np.concatenate(late);zoom_margin=max(.5,float(np.ptp(late,axis=0).max())*.08)
    crossing=source['VP_context'] is not None
    return lo,hi,late.min(0)-zoom_margin,late.max(0)+zoom_margin,{
        'turn_over30deg':turn,'hard_overlap_mean_distance_under1m':overlap,
        'VP_GT_proximity_candidate':crossing,'add_inset':bool(turn or overlap or crossing)}

def trajectories(ax,source,panel,segments,history=True,late_only=False):
    ax.add_collection(LineCollection(segments,colors='#AAB3B9',linewidths=.5,alpha=.22,zorder=1))
    if history:
        h=np.array(source['history_trajectory_m']);hm=np.array(source['history_mask'],dtype=bool);h[~hm]=np.nan
        ax.plot(h[:,0],h[:,1],'o-',color='#7D858B',lw=1.5,ms=4,zorder=3,label='History')
    gt=np.array(source['GT_trajectory_m']);modes=np.array(panel['all_mode_trajectories_m'])
    best=modes[panel['best_FDE_mode_zero_based']];top=modes[panel['top1_mode_zero_based']]
    records=[]
    for values,label,color,marker,style,lw,z,ms in (
        (gt,'GT','#111111','s','-',2.5,10,5),
        (top,'Top1','#C87B24','^','--',2.,9,6),
        (best,'Best-FDE','#2478A5','o','-',2.5,11,4.5)):
        plotted=values[-6:] if late_only else values
        line,=ax.plot(plotted[:,0],plotted[:,1],color=color,marker=marker,ls=style,
            lw=lw,zorder=z,ms=ms,markerfacecolor='white' if label!='Best-FDE' else color,
            markeredgewidth=.9,label=label)
        records.append({'label':label,'marker_count':len(line.get_xdata()),'linewidth':lw,'zorder':z})
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.12,lw=.5)
    return records

def draw(source,stem):
    gt=np.array(source['GT_trajectory_m'],dtype=float)
    for panel in source['panels']:
        modes=np.array(panel['all_mode_trajectories_m'],dtype=float)
        best_error=np.linalg.norm(modes[panel['best_FDE_mode_zero_based']]-gt,axis=-1)
        top_error=np.linalg.norm(modes[panel['top1_mode_zero_based']]-gt,axis=-1)
        actual=dict(zip(FIELDS,map(float,(best_error.mean(),best_error[-1],top_error.mean(),top_error[-1]))))
        assert max(abs(actual[k]-panel['source_metrics'][k]) for k in FIELDS)<1e-4
    lo,hi,zlo,zhi,inset_rule=geometry(source)
    starts=np.array(source['lane_positions_m']).reshape(-1,2);vectors=np.array(source['lane_vectors_m']).reshape(-1,2)
    segments=np.stack((starts,starts+vectors),axis=1)
    nearby=((segments.max(1)>=lo)&(segments.min(1)<=hi)).all(1);segments=segments[nearby]
    body_height=3.8*float((hi[1]-lo[1])/(hi[0]-lo[0]));bottom=.9
    header_height=1.55 if inset_rule['add_inset'] else 1.25
    figure_height=max(3.2,body_height+bottom+header_height);fig=plt.figure(figsize=(9.5,figure_height))
    axes=[fig.add_axes([left,bottom/figure_height,.4,body_height/figure_height]) for left in (.085,.58)]
    fig.suptitle(f"{source['case_kind']} | {source['agent_type']} | GT displacement {source['GT_endpoint_displacement_m']:.2f} m",fontsize=11,y=1-.12/figure_height)
    fig.text(.5,1-.40/figure_height,f"{source['scene_name']} / sample {source['sample_token'][:8]} / actor {source['instance_token'][:8]}",ha='center',fontsize=8,color='#53616B')
    panel_audits=[];annotations=[]
    for panel_index,(ax,panel) in enumerate(zip(axes,source['panels'])):
        records=trajectories(ax,source,panel,segments);assert all(r['marker_count']==12 for r in records)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),
            xlabel='t0 ego forward x (m)',ylabel='t0 ego left y (m)')
        left=(.085,.58)[panel_index];body_top=bottom+body_height
        label_y=(body_top+(.78 if inset_rule['add_inset'] else .48))/figure_height
        fig.text(left,label_y,panel['method'],ha='left',va='top',fontsize=10)
        text=fig.text(left+.4,label_y,f"Best FDE = {panel['recomputed_metrics']['minFDE6']:.2f} m\nTop1 FDE = {panel['recomputed_metrics']['Top1FDE6']:.2f} m",
            ha='right',va='top',fontsize=8,
            bbox={'facecolor':'white','edgecolor':'#DEE3E6','linewidth':.5,'pad':3},zorder=25)
        annotations.append((ax,text,panel))
        if inset_rule['add_inset']:
            zoom=fig.add_axes([left+.005,(body_top+.10)/figure_height,.16,.44/figure_height],zorder=20)
            trajectories(zoom,source,panel,segments,history=False,late_only=True)
            zoom.set(xlim=(zlo[0],zhi[0]),ylim=(zlo[1],zhi[1]));zoom.tick_params(labelsize=6)
            zoom.set_title('Last 6 future points',fontsize=7,pad=3)
            for axis in (zoom.xaxis,zoom.yaxis):axis.set_major_locator(MaxNLocator(nbins=2))
            for spine in zoom.spines.values():spine.set_visible(True);spine.set_color('#7D858B')
        panel_audits.append({'method':panel['method'],'future_trajectory_styles':records,
            'metrics_recomputed':panel['recomputed_metrics'],'metric_absolute_differences_m':panel['metric_absolute_differences_m']})
    handles,labels=axes[0].get_legend_handles_labels();order=[labels.index(x) for x in ('History','GT','Best-FDE','Top1')]
    fig.legend([handles[i] for i in order],[labels[i] for i in order],loc='lower center',bbox_to_anchor=(.5,.27/figure_height),ncol=4,frameon=False,fontsize=9)
    fig.text(.5,.13/figure_height,'12 original future observations; Best-FDE uses GT; Top1 uses highest predicted probability.',ha='center',fontsize=8,color='#53616B')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax,text,panel in annotations:
        box=text.get_bbox_patch().get_window_extent(renderer)
        assert box.x0>=0 and box.y0>=0 and box.x1<=fig.bbox.x1 and box.y1<=fig.bbox.y1
        for line in ax.lines:
            points=ax.transData.transform(np.c_[line.get_xdata(),line.get_ydata()])
            points=points[np.isfinite(points).all(1)]
            covered=(points[:,0]>=box.x0)&(points[:,0]<=box.x1)&(points[:,1]>=box.y0)&(points[:,1]<=box.y1)
            assert not covered.any(),'Endpoint box hides a trajectory marker'
    exports={}
    for ext in ('png','pdf','svg'):
        path=Path(str(stem)+'.'+ext);fig.savefig(path,dpi=300,facecolor='white')
        if ext=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        exports[str(path.relative_to(ROOT))]=sha256(path)
    plt.close(fig)
    with Image.open(str(stem)+'.png') as im:im.verify()
    return {'status':'PASS','panels':panel_audits,'same_xy_limits':True,'xlim_m':[lo[0],hi[0]],'ylim_m':[lo[1],hi[1]],
        'equal_aspect':True,'compact_equal_aspect_rectangle':True,'separate_header_for_endpoint_text_and_zoom':True,
        'inset_rule':inset_rule,'inset_bbox_m':[zlo.tolist(),zhi.tolist()] if inset_rule['add_inset'] else None,
        'lane_alpha':.22,'lane_linewidth':.5,'lanes_do_not_set_axes_limits':True,'endpoint_annotation_layout':'PASS',
        'trajectory_edits':False,'smoothing':False,'interpolation':False,'exports_SHA256':exports}

@torch.no_grad()
def main():
    assert read_json(SUMMARY)['status']=='COMPLETE'
    assert read_json(ROOT/'04_evaluation/stage3b_pairing_audit.json')['status']=='PASS'
    base=read_actors(BASE_ACTORS);typed=read_actors(ACTORS);full,selected,omitted=case_candidates(base,typed)
    ds=SceneDataset('val');lookup={(r['scene_token'],r['sample_token']):i for i,r in enumerate(ds.rows)}
    assert len(lookup)==len(ds)==3603
    # Search improvement-ranked moving V/P targets for an auditable GT-proximity case.
    ordered=sorted((x for x in full if x[2]['agent_type'] in ('vehicle','pedestrian') and x[4]>5),key=lambda x:(x[0],x[1]))
    vp=None
    for choice in ordered:
        row=choice[2];index=lookup[row['scene_token'],row['sample_token']]
        context=vp_context(ds[index],int(row['node_in_graph']))
        if context:
            vp=(choice,context);selected.append(('stage3b_vp_interaction_case_001',choice,'V-P GT proximity',None));break
    if vp is None:omitted.append({'kind':'V-P interaction','reason':'No moving full-horizon V/P actor satisfies preregistered <=20m t0 and <=5m aligned future GT proximity.'})
    c=config();no_type=HiVTLossRecovery(**{k:c[k] for k in MODEL_KEYS}).cuda()
    baseline_state=torch.load(BASE_CHECKPOINT,map_location='cpu',weights_only=False)
    no_type.load_state_dict(baseline_state['state_dict']);no_type.eval();del baseline_state
    type_model=model_new();type_state=torch.load(BEST,map_location='cpu',weights_only=False)
    type_model.load_state_dict(type_state['state_dict']);type_model.eval();del type_state
    checkpoint_sha={'No-Type':sha256(BASE_CHECKPOINT),'Type Embedding':sha256(BEST)}
    assert checkpoint_sha['No-Type']==read_json(PREREG)['NoType_checkpoint_sha256']
    assert checkpoint_sha['Type Embedding']==read_json(SUMMARY)['checkpoint_sha256']
    actor_sha={'No-Type':sha256(BASE_ACTORS),'Type Embedding':sha256(ACTORS)};figures=[]
    for name,choice,kind,preferred in selected:
        delta,key,row,base_row,displacement=choice;index=lookup[row['scene_token'],row['sample_token']]
        graph,node,panels,fingerprint=checked_replay(ds,index,row,base_row,no_type,type_model)
        context=vp_context(graph,node) if name=='stage3b_vp_interaction_case_001' else None
        source={'name':name,'case_kind':kind,'preferred_threshold_satisfied':preferred,
            **{k:row[k] for k in ('scene_name','scene_token','sample_token','instance_token','agent_type','motion_state')},
            'node_in_graph':node,'GT_endpoint_displacement_m':displacement,'delta_FDE_Type_minus_NoType_m':delta,
            'history_trajectory_m':graph.positions[node,:5].tolist(),'history_mask':graph.history_mask[node].tolist(),
            'GT_trajectory_m':graph.positions[node,5:].tolist(),'future_mask':graph.future_mask[node].tolist(),
            'history_times_seconds':graph.history_times.tolist(),'future_times_seconds':graph.future_times.tolist(),
            'lane_positions_m':graph.lane_positions.tolist(),'lane_vectors_m':graph.lane_vectors.tolist(),
            'origin_global_m':graph.origin.tolist(),'ego_yaw_global_rad':float(graph.ego_yaw),
            'coordinate_frame':'t0 ego +x forward +y left; meters','panels':panels,'GT_fingerprint':fingerprint,
            'VP_context':context,'checkpoint_SHA256':checkpoint_sha,'actor_CSV_SHA256':actor_sha,
            'selection':'Preregistered deterministic largest FDE gain / degradation; VP proximity among improvement-ranked >5m targets.',
            'representativeness':'Purposive illustrative examples, not unbiased population-effect estimates.',
            'trajectory_edits':False,'smoothing':False,'interpolation':False}
        source_path=ROOT/'04_evaluation/cases'/(name+'.json');atomic_json(source_path,source)
        audit=draw(read_json(source_path),ROOT/'05_figures'/name)
        audit.update({'source_json':str(source_path.relative_to(ROOT)),'source_sha256':sha256(source_path),
            'checkpoint_SHA256':checkpoint_sha,'same_actor_and_GT':True})
        atomic_json(ROOT/'05_figures'/(name+'_audit.json'),audit)
        figures.append({'name':name,'source_json':str(source_path.relative_to(ROOT)),'source_sha256':sha256(source_path),
            'case_kind':kind,'delta_FDE_m':delta,'agent_type':row['agent_type'],'VP_context':context,
            'actor_key':list(key),'metric_audit':'PASS'})
        print('STAGE3B_MATCHED_CASE=PASS',name,'delta_FDE',delta,flush=True)
    ds.clear()
    assert len(figures)>=3,'Required improvement/degradation cases absent; report honest absence before delivery.'
    atomic_json(ROOT/'04_evaluation/stage3b_qualitative_case_manifest.json',{'status':'PASS','figures':figures,
        'omitted_cases':omitted,'case_count':len(figures),'checkpoint_SHA256':checkpoint_sha,
        'actor_CSV_SHA256':actor_sha,'selected_after_locked_quantitative_evaluation':True,
        'no_training':True,'VP_GT_proximity_does_not_prove_causal_interaction':True})
    print('STAGE3B_QUALITATIVE=PASS',flush=True)

def redraw_only():
    manifest=read_json(ROOT/'04_evaluation/stage3b_qualitative_case_manifest.json')
    assert manifest['status']=='PASS'
    for item in manifest['figures']:
        source_path=ROOT/item['source_json'];assert sha256(source_path)==item['source_sha256']
        source=read_json(source_path);audit=draw(source,ROOT/'05_figures'/item['name'])
        audit.update({'source_json':item['source_json'],'source_sha256':item['source_sha256'],
            'checkpoint_SHA256':source['checkpoint_SHA256'],'same_actor_and_GT':True,
            'redrawn_from_existing_true_JSON_without_inference_or_training':True})
        atomic_json(ROOT/'05_figures'/(item['name']+'_audit.json'),audit)
        print('STAGE3B_CASE_LAYOUT=PASS',item['name'],flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
    if '--redraw-only' in sys.argv:redraw_only()
    else:main()
