"""Publication exports with paired scene intervals and matched numerical cases."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage7a_formal_common import *
from stage7a_evaluate import load_pair,group_mask,IDS
from stage3b_common import model_new as baseline_new
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
  'font.size':7,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':0.7,
  'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
BLUE='#477FA8';BASE='#89979F';GT='#377C64';CONTROL='#C5A34D';TURN='#9B769B'
FIGURES=ROOT/'05_figures'

def export(fig,name,source):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox
    clipped=[];undrawn=set()
    # Matplotlib marks out-of-view Tick labels visible although Axis.draw excludes them.
    for ax in fig.axes:
        for axis,limits in ((ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())):
            lower,upper=sorted(limits)
            for tick in [*axis.get_major_ticks(),*axis.get_minor_ticks()]:
                if tick.get_loc()<lower-1e-12 or tick.get_loc()>upper+1e-12:
                    undrawn.update((tick.label1,tick.label2))
    for obj in fig.findobj(matplotlib.text.Text):
        if obj not in undrawn and obj.get_visible() and obj.get_text():
            box=obj.get_window_extent(renderer)
            if box.x0<bounds.x0-1 or box.y0<bounds.y0-1 or box.x1>bounds.x1+1 or box.y1>bounds.y1+1:clipped.append(obj.get_text())
    assert not clipped,clipped
    for suffix in ('svg','pdf','png'):
        path=FIGURES/(name+'.'+suffix);fig.savefig(path,dpi=300,facecolor='white')
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    atomic_json(FIGURES/(name+'_audit.json'),{'status':'PASS','source':source,'backend':'Python matplotlib',
      'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,'PNG_dpi':300,
      'SVG_editable_text':True,'PDF_fonttype':42,'out_of_canvas_text':clipped,
      'undrawn_outside_axis_tick_labels_ignored':[o.get_text() for o in undrawn if o.get_text()]})
    plt.close(fig)

def comparison(name,groups,labels):
    ci=pd.read_csv(ROOT/'06_tables/stage7a_bootstrap_ci.csv');ci=ci[ci.metric=='minFDE6'].set_index('group').loc[groups]
    source=[{'group':g,'label':l,**ci.loc[g].to_dict()} for g,l in zip(groups,labels)]
    write_csv(FIGURES/(name+'_source.csv'),source)
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,85/25.4),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
    y=np.arange(len(groups));a=axes[0];a.barh(y-0.16,ci.Stage3B,height=.29,color=BASE,label='Stage3B')
    a.barh(y+0.16,ci.Stage7A,height=.29,color=BLUE,label='Stage7A');a.set_yticks(y,labels);a.invert_yaxis()
    a.set_xlabel('minFDE6 (m), lower is better');a.legend(loc='upper right',fontsize=6)
    a.set_title('a  Independent trained models',loc='left',fontweight='bold',fontsize=8)
    a=axes[1];delta=ci.delta.to_numpy();low=ci.CI_lower.to_numpy();high=ci.CI_upper.to_numpy()
    # Draw endpoints directly: percentile intervals need not contain the sample estimate.
    for j,d,l,h in zip(y,delta,low,high):a.plot([l,h],[j,j],color=BLUE,lw=1.2);a.scatter(d,j,color=BLUE,s=14,zorder=3)
    a.axvline(0,color='#777777',lw=.7,ls='--');a.set_yticks(y,labels);a.invert_yaxis();a.set_xlabel('Δ minFDE6: Stage7A − Stage3B (m)')
    a.set_title('b  Paired scene bootstrap, 95% CI',loc='left',fontweight='bold',fontsize=8)
    for a in axes:a.grid(axis='x',color='#E9E9E9',lw=.5);a.set_axisbelow(True)
    export(fig,name,'06_tables/stage7a_bootstrap_ci.csv;1000 paired150-scene replicates; one training seed each; secondary CIs unadjusted')

def ablation():
    df=pd.read_csv(ROOT/'06_tables/stage7a_semantic_zero_ablation.csv');groups=['overall','vehicle','vehicle.moving','TurningVehicle_GT']
    labels=['Overall','Vehicle','Moving vehicle','Turning vehicle (GT)'];fig,ax=plt.subplots(figsize=(183/25.4,85/25.4),layout='constrained')
    x=np.arange(4);source=[]
    for offset,name,color in [(-.17,'Semantic-ON',BLUE),(.17,'Semantic-ZERO',BASE)]:
        values=df[df.model==name].set_index('group').loc[groups];ax.bar(x+offset,values.minFDE6,width=.31,color=color,label=name)
        for i,(v,n) in enumerate(zip(values.minFDE6,values['count'])):source.append({'setting':name,'group':groups[i],'count':int(n),'minFDE6':float(v)})
    ax.set_xticks(x,labels);ax.set_ylabel('minFDE6 (m), lower is better');ax.legend(loc='upper left')
    ax.set_title('Same Stage7A checkpoint: semantic input inference ablation',loc='left',fontsize=9)
    ax.grid(axis='y',color='#E9E9E9',lw=.5);ax.set_axisbelow(True)
    write_csv(FIGURES/'stage7a_semantic_on_zero_source.csv',source)
    export(fig,'stage7a_semantic_on_zero','06_tables/stage7a_semantic_zero_ablation.csv;point estimates;ZERO retains learned semantic MLP biases;not retrained baseline')

@torch.no_grad()
def cases():
    b,o,z=load_pair();b=b[b.horizon=='full_horizon'].reset_index(drop=True);o=o[o.horizon=='full_horizon'].reset_index(drop=True)
    paired=b.copy();paired['delta']=o.minFDE6-b.minFDE6;paired['Stage7A_minFDE']=o.minFDE6
    selections=[];used=set()
    for label,predicate,ascending in [
      ('turning_vehicle',paired.TurningVehicle_GT==1,True),
      ('traffic_control', (paired.NearTrafficControl20==1)&(paired.motion_state=='vehicle.moving')&(paired.GT_endpoint_displacement_m>5),True),
      ('degradation',(paired.agent_type=='vehicle')&(paired.GT_endpoint_displacement_m>5),False)]:
        candidates=paired[predicate].sort_values(['delta',*IDS],ascending=[ascending,True,True,True,True])
        selected=next((r for _,r in candidates.iterrows() if tuple(r[k] for k in IDS) not in used),None)
        assert selected is not None;used.add(tuple(selected[k] for k in IDS));selections.append((label,selected))
    baseline=baseline_new().eval();saved=torch.load(BASE_BEST,map_location='cpu',weights_only=False);baseline.load_state_dict(saved['state_dict']);del saved
    model=model_new().eval();saved=torch.load(BEST,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict']);del saved
    ds=Stage7ASemanticDataset('val');index={(r['scene_token'],r['sample_token']):i for i,r in enumerate(ds.rows)};manifest=[]
    for label,row in selections:
        g=ds[index[row.scene_token,row.sample_token]];node=int(row.node_in_graph);assert g.instance_tokens[node]==row.instance_token
        data=Batch.from_data_list([g]).cuda();outputs=[baseline(data),model(data)]
        predictions=[m.ego_predictions(out,data)[node].cpu().numpy() for m,out in zip((baseline,model),outputs)]
        history=g.positions[node,:5].numpy()[g.history_mask[node].numpy()];gt=g.positions[node,5:].numpy()
        top=[int(out['mode_prob'][node].argmax()) for out in outputs];best=[int(np.linalg.norm(p[:,-1]-gt[-1],axis=-1).argmin()) for p in predictions]
        fde=[float(np.linalg.norm(p[mode,-1]-gt[-1])) for p,mode in zip(predictions,best)]
        assert abs(fde[0]-row.minFDE6)<1e-3 and abs(fde[1]-row.Stage7A_minFDE)<1e-3
        allpoints=np.concatenate([history,gt,*[p.reshape(-1,2) for p in predictions]])
        low=allpoints.min(0)-6;high=allpoints.max(0)+6;span=max(high-low);center=(low+high)/2
        low=center-span/2;high=center+span/2
        figure,axes=plt.subplots(1,2,figsize=(183/25.4,105/25.4),layout='constrained')
        lanes=np.stack([g.lane_positions.numpy(),g.lane_positions.numpy()+g.lane_vectors.numpy()],axis=1);s=g.lane_semantic.numpy()
        titles=['Stage3B','Stage7A']
        for ax,p,mode,first,title,fd in zip(axes,predictions,best,top,titles,fde):
            ax.add_collection(LineCollection(lanes,color='#D5D9DB',linewidth=.5,zorder=0))
            ax.add_collection(LineCollection(lanes[(s[:,1]+s[:,3])>0],color=TURN,linewidth=.8,alpha=.65,zorder=1))
            ax.add_collection(LineCollection(lanes[s[:,5:8].any(-1)],color=CONTROL,linewidth=1.,alpha=.75,zorder=1))
            for path in p:ax.plot(*np.vstack([history[-1],path]).T,color=BLUE,alpha=.23,lw=.7)
            ax.plot(*np.vstack([history[-1],p[mode]]).T,color=BLUE,lw=1.6,zorder=5)
            ax.scatter(*p[mode,-1],s=17,facecolors='white',edgecolors=BLUE,linewidths=.9,zorder=10)
            ax.plot(*np.vstack([history[-1],p[first]]).T,color=BLUE,lw=1.1,ls='--',zorder=6)
            ax.plot(*history.T,color='#333333',lw=1.5,zorder=7);ax.plot(*np.vstack([history[-1],gt]).T,color=GT,lw=1.7,zorder=8)
            ax.scatter(*history[-1],s=15,color='#333333',zorder=9);ax.scatter(*gt[-1],s=16,color=GT,zorder=9)
            ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.set_xlabel('Ego x (m)');ax.set_ylabel('Ego y (m)')
            ax.set_title(title+f'  minFDE6={fd:.2f}m',fontsize=8,loc='left')
        outcome='improvement' if row.delta<0 else ('degradation' if row.delta>0 else 'unchanged')
        case_title={'turning_vehicle':'Turning vehicle (GT subgroup)',
                    'traffic_control':'Traffic-control context','degradation':'Failure case'}[label]
        figure.suptitle(case_title+f': {outcome}, Δ={row.delta:+.2f}m\n{g.scene_name}; same actor, frozen map and axes',fontsize=8)
        handles=[Line2D([0],[0],color='#333333',lw=1.5,label='History'),Line2D([0],[0],color=GT,lw=1.5,label='GT'),
          Line2D([0],[0],color=BLUE,lw=1.5,label='Best FDE'),Line2D([0],[0],color=BLUE,ls='--',label='Top1'),
          Line2D([0],[0],color=TURN,label='Turn connector'),Line2D([0],[0],color=CONTROL,label='Control-associated lane')]
        axes[0].legend(handles=handles,loc='upper left',fontsize=5)
        source={'scene_token':g.scene_token,'sample_token':g.sample_token,'instance_token':row.instance_token,
          'scene_name':g.scene_name,'node_in_graph':node,'selection':'prespecified extreme delta illustration, not frequency evidence',
          'baseline_checkpoint_sha256':sha256(BASE_BEST),'Stage7A_checkpoint_sha256':sha256(BEST),
          'history':history.tolist(),'GT':gt.tolist(),'Stage3B_candidates':predictions[0].tolist(),'Stage7A_candidates':predictions[1].tolist(),
          'best_FDE_modes':best,'Top1_modes':top,'minFDE6':fde,'delta':float(row.delta),
          'xlim':[float(low[0]),float(high[0])],'ylim':[float(low[1]),float(high[1])],'same_map_geometry':True,
          'TurningVehicle_GT':int(row.TurningVehicle_GT),'NearTrafficControl20':int(row.NearTrafficControl20),'outcome':outcome}
        name='stage7a_case_'+label;atomic_json(FIGURES/(name+'_source.json'),source);export(figure,name,name+'_source.json')
        manifest.append({'name':name,'outcome':outcome,**{k:source[k] for k in ['scene_token','sample_token','instance_token','delta','minFDE6']}})
    ds.clear();atomic_json(ROOT/'04_evaluation/stage7a_case_selection.json',{'selection_rule':read_json(ROOT/'00_manifest/stage7a_formal_figure_contract.json')['case_selection'],'cases':manifest})

def main():
    comparison('stage7a_main_fde_comparison',['overall','vehicle','pedestrian','vehicle.moving'],['Overall','Vehicle','Pedestrian','Moving vehicle'])
    comparison('stage7a_vehicle_difficulty_groups',['Vehicle >5m','IntersectionVehicle20','NearTurnConnector20','TurningVehicle_GT'],['Vehicle >5m','Intersection20','Near turn20','Turning vehicle (GT)'])
    ablation();cases();print('FORMAL_FIGURES_COMPLETE',flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
