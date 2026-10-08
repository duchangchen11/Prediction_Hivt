"""Post-aggregate case selection; examples do not replace bootstrap evidence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage8a1_common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from preprocessing.coordinates import global_to_ego
MODELS=('R0','R2','G1','G2','G3')

def main():
    assert (ROOT/'09_reports/stage8a1_scientific_decision.json').exists(),'Cases only after aggregate decisions'
    f=pd.read_csv(ROOT/'03_evaluation/stage8a1_actor_results.csv')
    hit=lambda v:f[v+'_top1_mode']==f.best_mode
    specifications=[('case1','R2 misses FDE-best mode; G1 selects it',~hit('R2')&hit('G1'),f.R2_Top1FDE-f.G1_Top1FDE),
        ('case2','R2 misses; semantic branch selects FDE-best mode',~hit('R2')&(hit('G2')|hit('G3'))&(f.MapNonEmpty==1)&
            ((f.NearTurnConnector20==1)|(f.NearTrafficControl20==1)|(f.agent_type=='Pedestrian')),f.R2_Top1FDE-np.minimum(f.G2_Top1FDE,f.G3_Top1FDE)),
        ('case3','G1 misses FDE-best mode; G3 selects it',~hit('G1')&hit('G3'),f.G1_Top1FDE-f.G3_Top1FDE),
        ('case4','G3 misses FDE-best mode and worsens versus R2',~hit('G3')&(f.G3_Top1FDE>f.R2_Top1FDE),f.G3_Top1FDE-f.R2_Top1FDE)]
    index=SparseSemanticIndex();records=read_json(C/'02_graph_cache/stage8a0c_selector_manifest.json')['batches'];manifest=[];seen=set()
    colors=['#4477aa','#ee6677','#228833','#ccbb44','#66ccee','#aa3377']
    for name,title,mask,score in specifications:
        eligible=f[mask].copy();eligible['score']=score[mask];eligible=eligible.sort_values(['score','scene_token','sample_token','instance_token'],ascending=[False,True,True,True])
        eligible=eligible[~eligible.instance_token.isin(seen)];assert len(eligible)>0,name+' requested case category unavailable'
        row=eligible.iloc[0];seen.add(row.instance_token);di=int(row.dataset_index);t=int(row.node_in_graph);bi=di//16
        rec=[r for r in records if r['split']=='val'][bi]
        source=torch.load(S6/rec['source_path'],map_location='cpu',weights_only=False);w=next(w for w in source['windows'] if w['dataset_index']==di)
        payload=torch.load(ROOT/'03_evaluation/cache'/(Path(rec['path']).stem+'.pt'),map_location='cpu',weights_only=False)
        ew=next(x for x in payload['windows'] if x['dataset_index']==di);a=ew['targets'].tolist().index(t)
        with np.load(C/rec['path']) as sel,np.load(ROOT/'02_graph_cache'/rec['frame_path']) as frame:
            j=di%16;obs,selected,node,idx,keep,feature=graph_window(w,rec,j,index,sel,frame)
            rr=selected.tolist().index(t);entityids=feature['entity_ids'][rr]
        start=w['current_position'][t].numpy();pred=w['ego_prediction'][t].numpy();gt=np.vstack((start,w['GT'][t].numpy()))
        xy=np.vstack((gt,pred.reshape(-1,2)));lo=xy.min(0)-5;hi=xy.max(0)+5
        extent=np.full(2,max(float((hi-lo).max()),15.));middle=(hi+lo)/2;lo=middle-extent/2;hi=middle+extent/2
        fig=plt.figure(figsize=(13,5));grid=fig.add_gridspec(1,3,width_ratios=(1.1,1.1,1.2));ax=fig.add_subplot(grid[0]);ctx=fig.add_subplot(grid[1]);bars=fig.add_subplot(grid[2])
        reg=index.regions[obs.map_location]
        for eid in np.unique(entityids[entityids>=0]):
            e=reg['entities'][reg['id_to_local'][int(eid)]];ty=e['type']
            for part in e['parts']:
                if ty<2:coordinates=np.asarray(part.coords)
                else:coordinates=np.asarray(part.exterior.coords)
                local=global_to_ego(coordinates,obs.origin,obs.yaw)
                for x in (ax,ctx):x.plot(local[:,0],local[:,1],color=colors[ty],lw=1,alpha=.6)
        for k,line in enumerate(pred):
            full=np.vstack((start,line));ax.plot(full[:,0],full[:,1],color='#999999',lw=1,alpha=.65);ax.text(*line[-1],str(k),fontsize=8)
        styles=[':','--','-.','-',(0,(5,2))];modelcolors=['#999999','#4477aa','#ee6677','#228833','#aa3377']
        for v,style,col in zip(MODELS,styles,modelcolors):
            k=int(row[v+'_top1_mode']);line=np.vstack((start,pred[k]));ax.plot(line[:,0],line[:,1],ls=style,color=col,lw=1.8,label=f'{v} mode{k}')
        for nb in idx[t][keep[t]].tolist():
            for line in w['ego_prediction'][nb].numpy():ctx.plot(line[:,0],line[:,1],color='#888888',lw=.6,alpha=.3)
        top=int(row.G3_top1_mode);alpha=ew['map_alpha']['G3'][a,top].numpy();marked=[]
        for slot,eid in enumerate(entityids[top]):
            if eid<0:continue
            entry=index.dictionary[int(eid)];marked.append({'entity_token':entry['token'],'entity_type':entry['entity_type'],'attention':float(alpha[slot])})
        for x in (ax,ctx):
            x.plot(gt[:,0],gt[:,1],color='black',lw=2,label='GT');x.scatter(*start,color='black',s=18);x.set_xlim(lo[0],hi[0]);x.set_ylim(lo[1],hi[1]);x.set_aspect('equal');x.set_xlabel('t0 ego x (m)');x.set_ylabel('t0 ego y (m)');x.grid(alpha=.15)
        ax.set_title('Shared six candidates / top1 selections',fontsize=9);ax.legend(fontsize=7,loc='best')
        ctx.set_title('Neighbor modes / selected map entities',fontsize=9)
        for i,v in enumerate(MODELS):
            p=ew['probabilities'][v][a].numpy();bars.bar(np.arange(6)+(i-2)*.14,p,width=.14,label=v,color=modelcolors[i])
        bars.set_xticks(range(6));bars.set_xlabel('Candidate mode');bars.set_ylabel('Probability');bars.legend(fontsize=8);bars.set_ylim(0,1)
        bars.set_title('Frozen candidate ranking probabilities',fontsize=9)
        fig.suptitle(name+': '+title+'\n'+row.agent_type+' | descriptive context; semantic attribution is unproven',fontsize=10)
        handles=[Line2D([0],[0],color=c,lw=1.5,label=ty) for c,ty in zip(colors,TYPES)]
        fig.legend(handles=handles,ncol=3,fontsize=7,loc='lower left',bbox_to_anchor=(.05,.005),frameon=False)
        info='G3 top-mode map mass: '+', '.join(f"{m['entity_type']}={m['attention']:.2f}" for m in sorted(marked,key=lambda d:-d['attention'])[:3])
        fig.text(.52,.018,info if marked else 'G3 top mode: no semantic map edges',fontsize=7,ha='left')
        fig.tight_layout(rect=(0,.075,1,1));fig.savefig(ROOT/'07_cases'/f'stage8a1_{name}.png',dpi=160);fig.savefig(ROOT/'07_cases'/f'stage8a1_{name}.svg');plt.close(fig)
        svg=ROOT/'07_cases'/f'stage8a1_{name}.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        record={'case':name,'description':title,'scene_token':row.scene_token,'sample_token':row.sample_token,'instance_token':row.instance_token,
            'dataset_index':di,'node_in_graph':t,'GT_best_mode':int(row.best_mode),'selected_top1':{v:int(row[v+'_top1_mode']) for v in MODELS},
            'FDE':{v:float(row[v+'_Top1FDE']) for v in MODELS},'G3_selected_mode_map_entities':sorted(marked,key=lambda d:-d['attention']),
            'aggregate_decision_precedes_selection':True,'semantic_causal_attribution':False}
        atomic_json(ROOT/'07_cases'/f'stage8a1_{name}.json',record);manifest.append(record)
    atomic_json(ROOT/'07_cases/stage8a1_case_manifest.json',{'status':'PASS','cases':manifest,'selection':'largest requested-category FDE effect, lexical identities break ties, distinct actor instances',
        'axis_shared_between_geometry_panels':True,'case2_interpretation':'semantic map context co-occurs with corrected ranking; attention cannot establish why'})
    print('CASES_PASS',flush=True)

if __name__=='__main__':main()
