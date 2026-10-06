"""Four matched Stage3B/Stage5A cases; unchanged trajectories in identical axes."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage5a_common import (SceneDataset, config, model_input, errors_with_top1, atomic_json,
    read_json, sha256, BEST, ACTORS, BASE_ACTORS, BASE_BEST, canonical_stage3b_model, verify_previous)
from stage5a_evaluate import read_actors, actor_key, load_final_model
from stage3b_common import GT_fingerprint
import numpy as np
import torch
from torch_geometric.data import Batch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from PIL import Image
FIELDS=('minADE6','minFDE6','Top1ADE6','Top1FDE6')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})


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
    # Bound tall-case canvas height while preserving equal physical x/y scale.
    width=max(width,height/1.2)
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
    vertical_inset=bool(inset_rule['add_inset'] and zhi[1]-zlo[1]>1.3*(zhi[0]-zlo[0]))
    inset_width,inset_height=(.13,.95) if vertical_inset else (.16,.44)
    if inset_rule['add_inset']:
        # Fill a legible inset rectangle without distorting physical x/y scale.
        zoom_aspect=9.5*inset_width/inset_height
        zoom_center=(zlo+zhi)/2
        zoom_width=max(zhi[0]-zlo[0],(zhi[1]-zlo[1])*zoom_aspect)
        zoom_height=max(zhi[1]-zlo[1],zoom_width/zoom_aspect)
        zlo=zoom_center-np.array([zoom_width,zoom_height])/2
        zhi=zoom_center+np.array([zoom_width,zoom_height])/2
    header_height=(2.15 if vertical_inset else 1.55) if inset_rule['add_inset'] else 1.25
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
        label_y=(body_top+(1.45 if vertical_inset else (.78 if inset_rule['add_inset'] else .48)))/figure_height
        fig.text(left,label_y,panel['method'],ha='left',va='top',fontsize=10)
        text=fig.text(left+.4,label_y,f"Best FDE = {panel['recomputed_metrics']['minFDE6']:.2f} m\nTop1 FDE = {panel['recomputed_metrics']['Top1FDE6']:.2f} m",
            ha='right',va='top',fontsize=8,
            bbox={'facecolor':'white','edgecolor':'#DEE3E6','linewidth':.5,'pad':3},zorder=25)
        annotations.append((ax,text,panel))
        if inset_rule['add_inset']:
            zoom=fig.add_axes([left+.005,(body_top+.10)/figure_height,inset_width,inset_height/figure_height],zorder=20)
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
        'vertical_inset_layout':vertical_inset,'inset_equal_physical_aspect':True,
        'lane_alpha':.22,'lane_linewidth':.5,'lanes_do_not_set_axes_limits':True,'endpoint_annotation_layout':'PASS',
        'trajectory_edits':False,'smoothing':False,'interpolation':False,'exports_SHA256':exports}


@torch.no_grad()
def replay(ds,index,node,model):
    bs=config()['batch_size'];start=index//bs*bs
    graphs=[ds[i] for i in range(start,min(start+bs,len(ds)))]
    data=Batch.from_data_list(graphs).cuda();output_node=int(data.ptr[index-start])+node
    out=model(model_input(data));prediction,errors=errors_with_top1(model,out,data)
    modes=prediction[output_node].cpu().numpy();gt=graphs[index-start].positions[node,5:].numpy()
    best=int(errors['best_mode'][output_node]);top=int(errors['top1_mode'][output_node])
    best_error=np.linalg.norm(modes[best].astype(np.float64)-gt,axis=-1)
    top_error=np.linalg.norm(modes[top].astype(np.float64)-gt,axis=-1)
    metrics=dict(zip(FIELDS,map(float,(best_error.mean(),best_error[-1],top_error.mean(),top_error[-1]))))
    assert modes.shape==(6,12,2) and np.isfinite(modes).all()
    return {'all_mode_trajectories_m':modes.tolist(),'mode_probabilities':out['mode_prob'][output_node].cpu().tolist(),
        'best_FDE_mode_zero_based':best,'top1_mode_zero_based':top,'recomputed_metrics':metrics,
        'reproduced_VAL_batch_start_index':start,'reproduced_VAL_batch_size':len(graphs)}


def candidates(b,e):
    full=[(float(e[k]['minFDE6'])-float(b[k]['minFDE6']),k,e[k],b[k]) for k in b if k[-1]=='full_horizon']
    groups=[('stage5a_case_moving_vehicle','moving vehicle improvement',
        [x for x in full if x[2]['agent_type']=='vehicle' and x[2]['motion_state']=='vehicle.moving']),
        ('stage5a_case_parked_stopped_vehicle','parked/stopped vehicle',
        [x for x in full if x[2]['agent_type']=='vehicle' and x[2]['motion_state'] in ('vehicle.parked','vehicle.stopped')]),
        ('stage5a_case_moving_pedestrian','moving pedestrian',
        [x for x in full if x[2]['agent_type']=='pedestrian' and float(x[2]['GT_endpoint_displacement_m'])>5])]
    selected=[];used=set();notes=[]
    for stem,kind,items in groups:
        assert items,'No real case for '+kind
        items=[x for x in items if x[1] not in used];choice=min(items,key=lambda x:(x[0],x[1]))
        if 'improvement' in kind and choice[0]>=0:
            notes.append('No moving-vehicle improvement exists; display least degraded real example.')
            kind='moving vehicle (no improvement available)'
        selected.append((stem,kind,choice));used.add(choice[1])
    harmed=[x for x in full if x[1] not in used and x[0]>0]
    if harmed:choice=max(harmed,key=lambda x:(x[0],x[1]));kind='degradation case'
    else:
        remaining=[x for x in full if x[1] not in used];choice=max(remaining,key=lambda x:(x[0],x[1]))
        kind='least improved case (no degradation available)';notes.append('No positive delta FDE exists; do not fabricate degradation.')
    selected.append(('stage5a_case_degradation',kind,choice))
    return selected,notes


@torch.no_grad()
def main():
    verify_previous();assert read_json(ROOT/'04_evaluation/stage5a_pairing_audit.json')['status']=='PASS'
    b=read_actors(BASE_ACTORS);e=read_actors(ACTORS);selected,notes=candidates(b,e)
    ds=SceneDataset('val');lookup={(r['scene_token'],r['sample_token']):i for i,r in enumerate(ds.rows)}
    assert len(lookup)==len(ds)==3603
    baseline=canonical_stage3b_model()
    baseline.load_state_dict(torch.load(BASE_BEST,map_location='cpu',weights_only=False)['state_dict'],strict=True);baseline.eval()
    experiment,_,_=load_final_model()
    checkpoint_sha={'Stage3B':sha256(BASE_BEST),'Stage5A':sha256(BEST)}
    actor_sha={'Stage3B':sha256(BASE_ACTORS),'Stage5A':sha256(ACTORS)};figures=[]
    for stem,kind,(delta,key,row,old) in selected:
        index=lookup[row['scene_token'],row['sample_token']];graph=ds[index];node=int(row['node_in_graph'])
        assert graph.instance_tokens[node]==row['instance_token'] and graph.future_mask[node].all()
        fingerprint=GT_fingerprint(graph.positions[node,5:],graph.future_mask[node],graph.agent_type[node])
        assert fingerprint['GT_trajectory_sha256']==row['GT_trajectory_sha256']==old['GT_trajectory_sha256']
        panels=[]
        for name,model,source in (('Stage3B',baseline,old),('Stage5A',experiment,row)):
            panel=replay(ds,index,node,model)
            difference={k:abs(panel['recomputed_metrics'][k]-float(source[k])) for k in FIELDS}
            assert max(difference.values())<1e-4,(name,difference)
            assert panel['best_FDE_mode_zero_based']==int(source['best_mode'])
            assert panel['top1_mode_zero_based']==int(source['top1_mode'])
            panel.update(method=name,source_metrics={k:float(source[k]) for k in FIELDS},
                         metric_absolute_differences_m=difference,metric_tolerance_m=1e-4)
            panels.append(panel)
        source={'name':stem,'case_kind':kind,**{k:row[k] for k in ('scene_name','scene_token','sample_token','instance_token','agent_type','motion_state')},
            'node_in_graph':node,'GT_endpoint_displacement_m':float(row['GT_endpoint_displacement_m']),
            'delta_FDE_Stage5A_minus_Stage3B_m':delta,'history_trajectory_m':graph.positions[node,:5].tolist(),
            'history_mask':(~graph.padding_mask[node,:5]).tolist(),'GT_trajectory_m':graph.positions[node,5:].tolist(),
            'future_mask':graph.future_mask[node].tolist(),'history_times_seconds':graph.history_times.tolist(),
            'future_times_seconds':graph.future_times.tolist(),'lane_positions_m':graph.lane_positions.tolist(),
            'lane_vectors_m':graph.lane_vectors.tolist(),'origin_global_m':graph.origin.tolist(),
            'ego_yaw_global_rad':float(graph.ego_yaw),'coordinate_frame':'t0 ego +x forward +y left; meters',
            'panels':panels,'GT_fingerprint':fingerprint,'VP_context':None,'checkpoint_SHA256':checkpoint_sha,'actor_CSV_SHA256':actor_sha,
            'selection':'deterministic largest FDE gain in required groups and largest degradation; tied by actor key',
            'representativeness':'purposive examples; not an unbiased population estimate',
            'trajectory_edits':False,'smoothing':False,'interpolation':False}
        source_path=ROOT/'04_evaluation/cases'/(stem+'.json');atomic_json(source_path,source)
        audit=draw(read_json(source_path),ROOT/'05_figures'/stem)
        audit.update(source_json=str(source_path.relative_to(ROOT)),source_sha256=sha256(source_path),
                     checkpoint_SHA256=checkpoint_sha,same_actor_and_GT=True)
        atomic_json(ROOT/'05_figures'/(stem+'_audit.json'),audit)
        figures.append({'name':stem,'source_json':str(source_path.relative_to(ROOT)),'source_sha256':sha256(source_path),
                        'case_kind':kind,'delta_FDE_m':delta,'agent_type':row['agent_type'],'actor_key':list(key),'metric_audit':'PASS'})
        print('STAGE5A_MATCHED_CASE_PASS',stem,delta,flush=True)
    ds.clear();assert len(figures)==4
    atomic_json(ROOT/'04_evaluation/stage5a_qualitative_case_manifest.json',{'status':'PASS','figures':figures,
        'case_count':4,'honest_absence_notes':notes,'checkpoint_SHA256':checkpoint_sha,'actor_CSV_SHA256':actor_sha,
        'selected_after_locked_quantitative_evaluation':True,'no_training':True})
    verify_previous();print('STAGE5A_QUALITATIVE_PASS',flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
