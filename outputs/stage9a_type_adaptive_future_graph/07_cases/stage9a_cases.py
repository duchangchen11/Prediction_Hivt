"""After aggregate only: four requested cases, no causal interpretation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage9a_common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODELS=('R0','R2','G1','G3','T1','T2','T3')

def main():
    assert (ROOT/'09_reports/stage9a_scientific_decision.json').exists()
    f=pd.read_csv(ROOT/'03_evaluation/stage9a_actor_results.csv');v=f.agent_type=='Vehicle';p=f.agent_type=='Pedestrian'
    hit=lambda m:f[m+'_top1_mode']==f.best_mode
    cutoff=float(f.loc[v,'T3_gate'].quantile(.75))
    cases=[('case1','MovingVehicle: R2 misses oracle mode; T3 selects it',v&(f.motion_state=='vehicle.moving')&~hit('R2')&hit('T3'),f.R2_Top1FDE-f.T3_Top1FDE),
        ('case2','Pedestrian: G1 worsens R2; T3 recovers oracle mode',p&hit('R2')&~hit('G1')&hit('T3')&(f.G1_Top1FDE>f.R2_Top1FDE),f.G1_Top1FDE-f.T3_Top1FDE),
        ('case3','High interaction and high gate Vehicle: T3 improves R2',v&(f.neighbor_count>=6)&(f.T3_gate>=cutoff)&(f.T3_Top1FDE<f.R2_Top1FDE),f.R2_Top1FDE-f.T3_Top1FDE),
        ('case4','Failure: T3 worsens R2',f.T3_Top1FDE>f.R2_Top1FDE,f.T3_Top1FDE-f.R2_Top1FDE)]
    palette=['#999999','#4477aa','#ee6677','#ccbb44','#66ccee','#228833','#aa3377'];records=[];seen=set();unavailable=[]
    for name,title,mask,score in cases:
        eligible=f[mask].copy();eligible['score']=score[mask];eligible=eligible[~eligible.instance_token.isin(seen)].sort_values(['score','scene_token','sample_token','instance_token'],ascending=[False,True,True,True])
        if not len(eligible):unavailable.append({'case':name,'reason':'No target satisfies frozen requested category; no substitute claim'});continue
        row=eligible.iloc[0];seen.add(row.instance_token);di=int(row.dataset_index);t=int(row.node_in_graph)
        source=torch.load(S6/'01_cache/val'/f'stage6a_val_batch_{di//16*16:05d}.pt',map_location='cpu',weights_only=False);w=next(x for x in source['windows'] if x['dataset_index']==di)
        result=torch.load(ROOT/'03_evaluation/cache'/f'stage9a_val_batch_{di//16*16:05d}.pt',map_location='cpu',weights_only=False);ew=next(x for x in result['windows'] if x['dataset_index']==di);a=ew['targets'].tolist().index(t)
        _,_,_,idx,keep=graph_from_window(w,torch.tensor([t]));start=w['current_position'][t].numpy();pred=w['ego_prediction'][t].numpy();gt=np.vstack((start,w['GT'][t].numpy()))
        xy=np.vstack((gt,pred.reshape(-1,2)));middle=(xy.min(0)+xy.max(0))/2;extent=max(float(np.ptp(xy,axis=0).max())+10,15);lo=middle-extent/2;hi=middle+extent/2
        fig,axs=plt.subplots(1,3,figsize=(14,5),gridspec_kw={'width_ratios':[1,1,1.2]});ax,ctx,bars=axs
        for k,line in enumerate(pred):
            line=np.vstack((start,line));ax.plot(line[:,0],line[:,1],color='#aaaaaa',lw=.8,alpha=.6);ax.text(*line[-1],str(k),fontsize=8)
        styles=[':', '--', '-.',(0,(1,1)),(0,(5,2)),(0,(3,1,1,1)),'-']
        for m,col,style in zip(MODELS,palette,styles):
            mode=int(row[m+'_top1_mode']);line=np.vstack((start,pred[mode]));ax.plot(line[:,0],line[:,1],color=col,ls=style,lw=1.7,label=f'{m}: mode{mode}')
        for nb in idx[t][keep[t]].tolist():
            for line in w['ego_prediction'][nb].numpy():ctx.plot(line[:,0],line[:,1],color='#888888',lw=.6,alpha=.4)
        selected=np.vstack((start,pred[int(row.T3_top1_mode)]));ctx.plot(selected[:,0],selected[:,1],color=palette[-1],lw=1.7,label='T3 selected')
        for x in (ax,ctx):
            x.plot(gt[:,0],gt[:,1],color='black',lw=2,label='GT');x.scatter(*start,color='black',s=18);x.set_xlim(lo[0],hi[0]);x.set_ylim(lo[1],hi[1]);x.set_aspect('equal');x.grid(alpha=.15);x.set_xlabel('t0 ego x (m)');x.set_ylabel('t0 ego y (m)')
        ax.set_title('Shared candidates / ranking selections',fontsize=9);ax.legend(fontsize=6,loc='best');ctx.set_title('Neighbor predicted modes',fontsize=9);ctx.legend(fontsize=7)
        for i,(m,col) in enumerate(zip(MODELS,palette)):bars.bar(np.arange(6)+(i-3)*.11,ew['probabilities'][m][a].numpy(),width=.11,color=col,label=m)
        bars.set_xticks(range(6));bars.set_xlabel('Candidate mode');bars.set_ylabel('Probability');bars.set_ylim(0,1);bars.legend(fontsize=7);bars.set_title('Fixed candidates, different rankings',fontsize=9)
        fig.suptitle(title+'\n'+f'T3 gate={row.T3_gate:.3f}; current neighbors={int(row.neighbor_count)}; descriptive example',fontsize=10);fig.tight_layout()
        fig.savefig(ROOT/'07_cases'/f'stage9a_{name}.png',dpi=160);fig.savefig(ROOT/'07_cases'/f'stage9a_{name}.svg');plt.close(fig)
        svg=ROOT/'07_cases'/f'stage9a_{name}.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
        record={'case':name,'description':title,'scene_token':row.scene_token,'sample_token':row.sample_token,'instance_token':row.instance_token,'dataset_index':di,'node_in_graph':t,'best_mode':int(row.best_mode),'T3_gate':float(row.T3_gate),'neighbor_count':int(row.neighbor_count),'top1_modes':{m:int(row[m+'_top1_mode']) for m in MODELS},'Top1FDE':{m:float(row[m+'_Top1FDE']) for m in MODELS},'aggregate_precedes_selection':True,'causal_attribution':False}
        atomic_json(ROOT/'07_cases'/f'stage9a_{name}.json',record);records.append(record)
    atomic_json(ROOT/'07_cases/stage9a_case_manifest.json',{'status':'PASS' if len(records)==4 else 'REQUESTED_CATEGORY_UNAVAILABLE','cases':records,'unavailable':unavailable,'high_gate_vehicle_p75':cutoff,'same_axis_between_geometry_panels':True,'selection':'fixed requested category; largest FDE effect; lexical identity tie-break; distinct instances','oracle_mode_not_behavior_prediction_accuracy':True})
    print('STAGE9_CASES',len(records),'UNAVAILABLE',unavailable,flush=True)

if __name__=='__main__':main()
