"""Four predeclared, measured, same-actor map/attention/error illustrations."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'01_attention_capture')]
from stage7c_common import *
from stage7c_hooks import LaneCapture
from torch_geometric.data import Batch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':7,'axes.grid':False,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
BLUE='#477FA8';BASE='#8D6E63';GT='#377C64'
SEM_COLORS={0:'#B4BBC0',1:'#A46F76',2:'#758895',3:'#6C82AE',4:'#A881A3'}

def export(fig,name,source):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();box=fig.bbox;outside=[];skip=set()
    for ax in fig.axes:
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            lo,hi=sorted(limits)
            for t in [*axis.get_major_ticks(),*axis.get_minor_ticks()]:
                if t.get_loc()<lo-1e-10 or t.get_loc()>hi+1e-10:skip.update([t.label1,t.label2])
    for o in fig.findobj(matplotlib.text.Text):
        if o in skip or not o.get_visible() or not o.get_text():continue
        b=o.get_window_extent(renderer)
        if b.x0<box.x0-1 or b.y0<box.y0-1 or b.x1>box.x1+1 or b.y1>box.y1+1:outside.append(o.get_text())
    assert not outside,outside
    artifacts=[]
    for ext in ['png','pdf','svg']:
        p=ROOT/'05_figures'/f'{name}.{ext}';fig.savefig(p,dpi=300,facecolor='white')
        if ext=='svg':p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        artifacts.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p)})
    plt.close(fig)
    atomic_json(ROOT/'05_figures'/f'{name}_audit.json',{'status':'PASS','backend':'Python matplotlib',
        'source':source,'same_axes':True,'all_text_within_canvas':True,'PNG_dpi':300,'SVG_text_editable':True,'PDF_fonttype':42,'artifacts':artifacts})

def select(f):
    v=(f.agent_type=='vehicle')&(f.GT_endpoint_displacement_m>5)
    turn=f.TurningVehicle_GT==1
    rules=[('case1_relevance_shift',v&(f.DeltaMinFDE>1)&(f.DeltaRelevantMass<-.05),False,
        'Degradation with reduced GT-relevant attention'),
        ('case2_opposite_turn',turn&(f.DeltaMinFDE>1)&((f.Stage7A_OppositeTurnMass-f.Stage3B_OppositeTurnMass)>0),False,
        'Turning vehicle: increased opposite-turn attention'),
        ('case3_correct_turn',turn&(f.DeltaMinFDE<-1)&(f.DeltaCorrectTurnMass>0),True,
        'Turning vehicle: improvement with increased correct-turn attention'),
        ('case4_reasonable_attention',v&(f.DeltaMinFDE>1)&(f.Stage7A_GTRelevantMass2m>=.5)&(f.DeltaRelevantMass>=0)&(f.Stage7A_Top1LaneGTDistance<=2),False,
        'GT-relevant attention with prediction degradation')]
    chosen=[];used=set()
    for name,mask,ascending,title in rules:
        candidates=f[mask].sort_values(['DeltaMinFDE',*IDS],ascending=[ascending,True,True,True])
        row=next((r for _,r in candidates.iterrows() if tuple(r[k] for k in IDS) not in used),None)
        if row is None and name=='case1_relevance_shift':
            candidates=f[v&(f.DeltaMinFDE>1)&(f.DeltaRelevantMass<0)].sort_values(['DeltaMinFDE',*IDS],ascending=[False,True,True,True])
            row=next((r for _,r in candidates.iterrows() if tuple(r[k] for k in IDS) not in used),None)
        assert row is not None,'No qualifying measured case: '+name
        used.add(tuple(row[k] for k in IDS));chosen.append((name,title,row,len(candidates)))
    return chosen

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    f=pd.read_csv(ROOT/'02_relevance_analysis/stage7c_actor_attention.csv')
    chosen=select(f);models={n:load_model(n) for n in ['Stage3B','Stage7A']}
    ds=Stage7ASemanticDataset('val');indices={(r['scene_token'],r['sample_token']):i for i,r in enumerate(ds.rows)}
    manifest=[]
    for name,title,row,n in chosen:
        idx=indices[row.scene_token,row.sample_token];g=ds[idx];node=int(row.node_in_graph)
        assert g.instance_tokens[node]==row.instance_token
        data=Batch.from_data_list([g]).cuda();pred=[];attention=[];best=[];fresh=[];alpha_diffs=[]
        archive=np.load(ROOT/f'01_attention_capture/stage7c_edges_batch_{idx//16:04d}.npz')
        graphidx=idx%16
        archive_edges=np.flatnonzero((archive['edge_graph_index']==graphidx)&(archive['edge_actor_node_index']==node))
        archive_lanes=archive['edge_lane_local_index'][archive_edges]
        for model_name,m in models.items():
            hook=LaneCapture(m,representations=False);out=m(input_for(data,model_name));hook.close()
            p=m.ego_predictions(out,data)[node].cpu().numpy();pred.append(p)
            mode=int(np.linalg.norm(p[:,-1]-g.positions[node,-1].numpy(),axis=1).argmin());best.append(mode)
            fd=float(np.linalg.norm(p[mode,-1]-g.positions[node,-1].numpy()));fresh.append(fd)
            assert abs(fd-row[model_name+'_minFDE6'])<1e-3
            e=hook.data['edge_index'];ix=np.flatnonzero(e[1]==node)
            assert np.array_equal(e[0,ix],archive_lanes)
            alpha=archive[model_name+'_attention_per_head'][archive_edges]
            alpha_diffs.append(float(np.abs(hook.data['alpha'][ix]-alpha).max()) if len(ix) else 0.)
            a=np.zeros(len(g.lane_tokens));a[archive_lanes]=alpha.mean(1);attention.append(a)
        archive.close()
        history=g.positions[node,:5].numpy()[g.history_mask[node].numpy()];gt=g.positions[node,5:].numpy()
        pbest=[p[b] for p,b in zip(pred,best)]
        points=np.concatenate([history,gt,*pbest]);center=(points.min(0)+points.max(0))/2;span=float((points.max(0)-points.min(0)).max()+12)
        low=center-span/2;high=center+span/2
        lines=np.stack([g.lane_positions.numpy(),g.lane_positions.numpy()+g.lane_vectors.numpy()],1)
        sem=g.lane_semantic.numpy();kind=np.where(sem[:,0]==1,sem[:,1:5].argmax(1)+1,0)
        colors=[SEM_COLORS[int(k)] for k in kind]
        fig,axes=plt.subplots(2,2,figsize=(183/25.4,178/25.4),layout='constrained')
        vmax=max(a.max() for a in attention);norm=Normalize(0,max(vmax,1e-12));lc=None
        for col,(model_name,a) in enumerate(zip(models,attention)):
            axes[0,col].add_collection(LineCollection(lines,colors=colors,linewidths=.8,alpha=.8))
            # One token marker retains the turn color along the whole centerline;
            # overlapping control/crosswalk flags are never collapsed to one color.
            for token in dict.fromkeys(g.lane_tokens):
                ix=np.array([k for k,t in enumerate(g.lane_tokens) if t==token]);k=int(ix[len(ix)//2]);xy=lines[k].mean(0)
                if sem[k,5:8].any():axes[0,col].scatter(*xy,marker='s',s=8,color='#C1A052',zorder=3)
                if sem[k,8]:axes[0,col].scatter(*xy,marker='o',s=22,facecolors='none',edgecolors='#56A7A0',linewidths=.8,zorder=4)
            axes[0,col].set_title(('a  ' if col==0 else 'b  ')+model_name+f': minFDE6={row[model_name+"_minFDE6"]:.2f}m',loc='left',fontsize=8)
            axes[1,col].add_collection(LineCollection(lines,colors='#DFE2E3',linewidths=.45))
            lc=LineCollection(lines[archive_lanes],array=a[archive_lanes],cmap='viridis',norm=norm,linewidths=1.9)
            axes[1,col].add_collection(lc)
            axes[1,col].set_title(('c  ' if col==0 else 'd  ')+f'Attention: relevant mass2m={row[model_name+"_GTRelevantMass2m"]:.3f}',loc='left',fontsize=8)
            for ax in axes[:,col]:
                ax.plot(*history.T,color='#222222',lw=1.2,zorder=5)
                ax.plot(*np.vstack([history[-1],gt]).T,color=GT,lw=1.7,zorder=7)
                for p,color in zip(pbest,[BASE,BLUE]):
                    ax.plot(*np.vstack([history[-1],p]).T,color=color,lw=1.2,zorder=6)
                    ax.scatter(*p[-1],s=14,facecolors='white',edgecolors=color,linewidths=.8,zorder=8)
                ax.scatter(*gt[-1],color=GT,s=12,zorder=9)
                ax.scatter(*history[-1],color='#222222',s=10,zorder=8)
                ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_aspect('equal');ax.set_xlabel('Ego x (m)');ax.set_ylabel('Ego y (m)')
        cb=fig.colorbar(lc,ax=axes[1,:],shrink=.75,pad=.02);cb.set_label('Mean-head attention (common scale)',fontsize=7)
        handles=[Line2D([0],[0],color=GT,label='GT'),Line2D([0],[0],color=BASE,label='Stage3B best'),Line2D([0],[0],color=BLUE,label='Stage7A best'),
            Line2D([0],[0],color=SEM_COLORS[1],label='Left'),Line2D([0],[0],color=SEM_COLORS[2],label='Straight'),Line2D([0],[0],color=SEM_COLORS[3],label='Right'),
            Line2D([0],[0],color='#C1A052',marker='s',ls='none',markersize=3,label='Control'),
            Line2D([0],[0],color='#56A7A0',marker='o',markerfacecolor='none',ls='none',markersize=4,label='Crosswalk')]
        axes[0,0].legend(handles=handles,loc='upper left',fontsize=5,ncol=2)
        fig.suptitle(title+f'\n{g.scene_name}; ΔminFDE={row.DeltaMinFDE:+.2f}m; illustrative case',fontsize=8)
        source={'case':name,'title':title,'candidate_count':n,'selection':'predeclared illustrative criteria; extreme FDE delta; not prevalence evidence',
            **{k:str(row[k]) for k in IDS},'scene_name':g.scene_name,'node_in_graph':node,'dataset_index':idx,
            'history':history.tolist(),'GT':gt.tolist(),'Stage3B_best':pbest[0].tolist(),'Stage7A_best':pbest[1].tolist(),
            'best_modes':best,'fresh_minFDE':fresh,'frozen_minFDE':[float(row.Stage3B_minFDE6),float(row.Stage7A_minFDE6)],
            'lane_positions':g.lane_positions.numpy().tolist(),'lane_vectors':g.lane_vectors.numpy().tolist(),
            'lane_semantic':sem.tolist(),'incoming_lane_indices':archive_lanes.tolist(),
            'Stage3B_mean_attention':attention[0].tolist(),'Stage7A_mean_attention':attention[1].tolist(),
            'shared_attention_color_max':float(vmax),'attention_fresh_vs_archived_maxdiff':alpha_diffs,
            'same_GT':True,'same_map':True,'same_axes':True,'xlim':[float(low[0]),float(high[0])],'ylim':[float(low[1]),float(high[1])],
            'GT_turn_heading_deg':float(row.GT_heading_change_deg),'TurningVehicle_GT':int(row.TurningVehicle_GT),
            'DeltaMinFDE':float(row.DeltaMinFDE),'DeltaRelevantMass':float(row.DeltaRelevantMass),
            'DeltaCorrectTurnMass':float(row.DeltaCorrectTurnMass) if np.isfinite(row.DeltaCorrectTurnMass) else None,
            'DeltaOppositeTurnMass':float(row.Stage7A_OppositeTurnMass-row.Stage3B_OppositeTurnMass) if row.TurningVehicle_GT else None,
            'checkpoint_sha256':{'Stage3B':BASE_SHA,'Stage7A':BEST_SHA}}
        atomic_json(ROOT/'07_cases'/f'stage7c_{name}_source.json',source)
        export(fig,'stage7c_'+name,'07_cases/stage7c_'+name+'_source.json')
        manifest.append({k:source[k] for k in ['case','title',*IDS,'candidate_count','DeltaMinFDE','DeltaRelevantMass','DeltaCorrectTurnMass','DeltaOppositeTurnMass']})
    ds.clear();atomic_json(ROOT/'07_cases/stage7c_case_selection.json',{'cases':manifest,'count':4,'distinct_identities':True,'training':False})
    print('FOUR CASES COMPLETE',manifest,flush=True)

if __name__=='__main__':main()
