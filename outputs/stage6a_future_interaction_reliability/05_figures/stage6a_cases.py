"""Four purposive cases from the completed official pass; no new predictor inference."""
from pathlib import Path
import sys,csv
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage6a_common import *
from stage6a_evaluate import ACTORS,membership
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.text import Text
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'path.simplify':False})
COLORS={'R0':'#C18336','R1':'#9C83A9','R2':'#3488AD'}

def selection(rows,members):
    full=[r for r in rows if r['horizon']=='full_horizon'];used=set();chosen=[]
    gain=lambda r:float(r['R2_Top1FDE'])-float(r['R0_Top1FDE'])
    specs=[('stage6a_case_moving_vehicle','Moving vehicle: better Top1',
        lambda r:r['motion_state']=='vehicle.moving' and float(r['R0_Top1HitRate'])==0 and gain(r)<0,gain),
        ('stage6a_case_pedestrian','Pedestrian: best mode recovered',
        lambda r:r['agent_type']=='pedestrian' and float(r['R0_Top1HitRate'])==0 and float(r['R2_Top1HitRate'])==1,gain),
        ('stage6a_case_heterogeneous','Heterogeneous context: R2 better than R1',
        lambda r:members[actor_key(r)]['Heterogeneous-20m']=='1' and float(r['R2_Top1FDE'])<float(r['R1_Top1FDE']),
        lambda r:float(r['R2_Top1FDE'])-float(r['R1_Top1FDE'])),
        ('stage6a_case_failure','Failure: wrong Top1 after reranking',
        lambda r:float(r['R2_Top1HitRate'])==0 and gain(r)>0,lambda r:-gain(r))]
    for stem,title,predicate,score in specs:
        candidates=[r for r in full if predicate(r) and r['instance_token'] not in used]
        assert candidates,'Requested real case absent: '+title
        r=min(candidates,key=lambda x:(score(x),actor_key(x)));used.add(r['instance_token'])
        chosen.append((stem,title,r,len(candidates)))
    return chosen

def draw(source):
    modes=np.array(source['all_mode_trajectories_m']);gt=np.array(source['GT_trajectory_m'])
    hist=np.array(source['history_trajectory_m']);hm=np.array(source['history_mask'],dtype=bool);hist[~hm]=np.nan
    visible=np.concatenate([modes.reshape(-1,2),gt,hist[hm]])
    low,high=visible.min(0),visible.max(0);pad=max(1.,float(np.ptp(visible,axis=0).max())*.07)
    low-=pad;high+=pad
    # Shared limits include every candidate; physical x/y scale remains equal.
    center=(low+high)/2;span=high-low;span[0]=max(span[0],span[1]/1.2);span[1]=max(span[1],span[0]*.35)
    low=center-span/2;high=center+span/2
    segments=np.array(source['lane_segments_m']).reshape(-1,2,2)
    body_height=1.80*float(span[1]/span[0]);body_bottom=2.82
    height=body_bottom+body_height+1.05;fig=plt.figure(figsize=(7.2,height))
    probability_limit=max(.35,max(max(p['mode_probabilities']) for p in source['rankings'].values())+.13)
    checks=[];prob_artists=[]
    for j,v in enumerate(('R0','R1','R2')):
        left=.085+j*.302;ax=fig.add_axes([left,body_bottom/height,.25,body_height/height]);p=source['rankings'][v];top=p['top1_mode_zero_based']
        ax.add_collection(LineCollection(segments,colors='#ABB4BC',linewidths=.4,alpha=.20,zorder=0))
        lines=[]
        for k,mode in enumerate(modes):
            line,=ax.plot(mode[:,0],mode[:,1],color='#A8B1B8',lw=.9,alpha=.65,zorder=1)
            assert np.array_equal(line.get_xydata(),mode);lines.append(line)
        ax.plot(hist[:,0],hist[:,1],'o-',color='#777777',ms=2.5,lw=1,label='History',zorder=3)
        line,=ax.plot(modes[top,:,0],modes[top,:,1],color=COLORS[v],marker=('^','D','o')[j],
            ms=3,lw=1.7,ls='--',label=v+' Top1',zorder=4)
        assert np.array_equal(line.get_xydata(),modes[top])
        line,=ax.plot(gt[:,0],gt[:,1],'s-',color='#151515',ms=2.8,mfc='white',lw=1.4,label='GT',zorder=5)
        assert np.array_equal(line.get_xydata(),gt)
        ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),xlabel='Ego x (m)')
        if j==0:ax.set_ylabel('Ego y (m)')
        ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.12,lw=.5)
        ax.set_title(f'{v}: mode {top+1}\nTop1 FDE = {p["Top1FDE_m"]:.2f} m',fontsize=8,pad=7)
        ax.xaxis.set_major_locator(MaxNLocator(3));ax.yaxis.set_major_locator(MaxNLocator(3));ax.tick_params(labelsize=7)
        bx=fig.add_axes([left,.90/height,.25,1.12/height]);probs=np.array(p['mode_probabilities']);bars=bx.bar(np.arange(1,7),probs,color=COLORS[v],width=.65)
        for k,(bar,value) in enumerate(zip(bars,probs)):
            assert bar.get_height()==value
            bx.text(k+1,value+.018,f'{value:.3f}',ha='center',fontsize=6)
            if k==top:bar.set_edgecolor('#222222');bar.set_linewidth(1.5)
        bx.set(xlabel='Candidate mode (1–6)',ylim=(0,probability_limit),xticks=np.arange(1,7))
        if j==0:bx.set_ylabel('Mode probability')
        bx.set_title(v+' probabilities',fontsize=8);bx.grid(axis='y',alpha=.12);bx.tick_params(labelsize=7)
        checks.append({'variant':v,'all_six_candidates_exact':True,'GT_exact':True,'selected_candidate_exact':True,
            'future_points_per_trajectory':12,'Top1FDE_m':p['Top1FDE_m'],'metric_difference_m':p['metric_difference_m']})
        prob_artists.append({'variant':v,'probability_artist_max_abs_diff':0.})
    fig.suptitle(source['case_title'],fontsize=10,y=1-.12/height)
    fig.text(.5,1-.39/height,f"{source['agent_type']} | scene {source['scene_name']} | actor {source['instance_token'][:8]}",ha='center',fontsize=8)
    fig.text(.5,1-.61/height,f"Shared oracle minFDE = {source['minFDE6_m']:.2f} m; best-FDE candidate = mode {source['best_mode_zero_based']+1}",ha='center',fontsize=8)
    handles=[Line2D([],[],color='#777777',marker='o',lw=1,label='History'),
        Line2D([],[],color='#151515',marker='s',mfc='white',lw=1,label='GT'),
        Line2D([],[],color='#A8B1B8',lw=1,label='Six candidates')]
    handles.extend(Line2D([],[],color=COLORS[v],marker=('^','D','o')[j],ls='--',lw=1,label=v+' Top1') for j,v in enumerate(('R0','R1','R2')))
    legend=fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,(body_bottom-.56)/height),ncol=6,frameon=False,fontsize=6)
    caption='Identical six candidates in every panel. GT selects the oracle mode only for evaluation.\n12 original future points; no interpolation or smoothing. Dark bar edge marks predicted Top1.\nPurposive example selected after locked evaluation; it does not estimate population benefit.'
    footer=fig.text(.5,.055/height,caption,ha='center',va='bottom',fontsize=7,color='#53616B')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=list(fig.texts)+[t for legend in fig.legends for t in legend.get_texts()]
    for ax in fig.axes:
        texts.extend([ax.xaxis.label,ax.yaxis.label,ax.title]+list(ax.texts))
        for ticks,labels,limits in ((ax.get_xticks(),ax.get_xticklabels(),ax.get_xlim()),(ax.get_yticks(),ax.get_yticklabels(),ax.get_ylim())):
            texts.extend(t for x,t in zip(ticks,labels) if min(limits)<=x<=max(limits))
    for text in texts:
        if not text.get_visible() or not text.get_text():continue
        b=text.get_window_extent(renderer)
        assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1,(text.get_text(),b.bounds)
    for ax in fig.axes:
        for text in (ax.xaxis.label,ax.title):
            assert not legend.get_window_extent(renderer).overlaps(text.get_window_extent(renderer))
            assert not footer.get_window_extent(renderer).overlaps(text.get_window_extent(renderer))
    exports={}
    for ext in ('png','pdf','svg'):
        p=ROOT/'05_figures'/(source['name']+'.'+ext);fig.savefig(p,dpi=300,facecolor='white')
        if ext=='svg':p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
        exports[str(p.relative_to(ROOT))]=sha256(p)
    plt.close(fig)
    return {'status':'PASS','panels':checks,'probability_checks':prob_artists,'same_xy_limits':True,
        'equal_aspect':True,'trajectory_edits':False,'smoothing':False,'interpolation':False,
        'exports_SHA256':exports,'xlim_m':low[[0]].tolist()+high[[0]].tolist(),
        'ylim_m':low[[1]].tolist()+high[[1]].tolist(),'no_new_predictor_forward':True}

def main():
    assert read_json(ROOT/'04_evaluation/stage6a_evaluation_complete.json')['status']=='PASS'
    with ACTORS.open() as f:rows=list(csv.DictReader(f))
    members=membership();chosen=selection(rows,members);ds=SceneDataset('val')
    lookup={(r['scene_token'],r['sample_token']):i for i,r in enumerate(ds.rows)};records=[]
    for stem,title,r,population in chosen:
        index=lookup[r['scene_token'],r['sample_token']];start=index//16*16;node=int(r['node_in_graph'])
        cache_path=cache_file('val',start);rank_path=ROOT/'01_cache/val_ranking'/f'stage6a_val_ranking_batch_{start:05d}.pt'
        w=torch.load(cache_path,map_location='cpu',weights_only=False)['windows'][index-start]
        out=torch.load(rank_path,map_location='cpu',weights_only=False)['windows'][index-start]
        graph=ds[index];assert w['instance_tokens'][node]==r['instance_token']==graph.instance_tokens[node]
        assert w['raw_prediction_sha256']==out['raw_prediction_sha256'] and w['ego_prediction_sha256']==out['ego_prediction_sha256']
        assert tensor_sha(w['ego_prediction'])==out['ego_prediction_sha256']
        fp=GT_fingerprint(w['GT'][node],w['future_mask'][node],w['agent_type'][node]);assert fp['GT_trajectory_sha256']==r['GT_trajectory_sha256']
        modes=w['ego_prediction'][node].numpy();gt=w['GT'][node].numpy();fde=np.linalg.norm(modes.astype(float)-gt[None].astype(float),axis=-1)[:,-1]
        assert int(fde.argmin())==int(r['best_mode']);rankings={}
        for v in ('R0','R1','R2'):
            top=int(out['rankings'][v]['top1_mode'][node]);probs=out['rankings'][v]['mode_prob'][node].tolist()
            diff=abs(float(fde[top])-float(r[v+'_Top1FDE']));assert diff<1e-4
            assert top==int(r[v+'_top1_mode'])==int(np.argmax(probs))
            rankings[v]={'top1_mode_zero_based':top,'mode_probabilities':probs,'Top1FDE_m':float(r[v+'_Top1FDE']),
                'metric_difference_m':diff,'best_mode_hit':float(r[v+'_Top1HitRate'])==1}
        source={'name':stem,'case_title':title,'selection_candidate_count':population,
            **{k:r[k] for k in ('scene_name','scene_token','sample_token','instance_token','agent_type','motion_state')},
            'node_in_graph':node,'dataset_index':index,'cache_batch_start':start,'coordinate_frame':'t0 ego x-forward y-left meters',
            'history_trajectory_m':w['history'][node].tolist(),'history_mask':(~w['history_padding'][node]).tolist(),
            'GT_trajectory_m':gt.tolist(),'all_mode_trajectories_m':modes.tolist(),'GT_fingerprint':fp,
            'lane_segments_m':torch.stack((graph.lane_positions,graph.lane_positions+graph.lane_vectors),1).tolist(),
            'best_mode_zero_based':int(r['best_mode']),'minFDE6_m':float(r['minFDE6']),'rankings':rankings,
            'frozen_20m_membership':members[actor_key(r)],'raw_prediction_sha256':out['raw_prediction_sha256'],
            'ego_prediction_sha256':out['ego_prediction_sha256'],'prediction_cache_SHA256':sha256(cache_path),
            'official_ranking_cache_SHA256':sha256(rank_path),'actor_csv_SHA256':sha256(ACTORS),
            'checkpoint_SHA256':{'predictor':PREDICTOR_SHA,'R1':sha256(head_path('R1')),'R2':sha256(head_path('R2'))},
            'selection':'largest specified gain or failure degradation; ties by actor key; distinct instances',
            'future_GT_role':'offline case selection and evaluation only','no_new_predictor_forward':True}
        p=ROOT/'04_evaluation/cases'/(stem+'.json');atomic_json(p,source);audit=draw(source)
        audit.update(source_json=str(p.relative_to(ROOT)),source_sha256=sha256(p),same_actor_and_GT=True)
        atomic_json(ROOT/'05_figures'/(stem+'_audit.json'),audit)
        records.append({'name':stem,'case_kind':title,'agent_type':r['agent_type'],'actor_key':list(actor_key(r)),
            'R0_Top1FDE':float(r['R0_Top1FDE']),'R1_Top1FDE':float(r['R1_Top1FDE']),'R2_Top1FDE':float(r['R2_Top1FDE']),
            'source_json':str(p.relative_to(ROOT)),'source_sha256':sha256(p),'metric_audit':'PASS'})
        print('STAGE6A_CASE_PASS',stem,flush=True)
    ds.clear();atomic_json(ROOT/'04_evaluation/stage6a_qualitative_case_manifest.json',{'status':'PASS','case_count':4,
        'figures':records,'purposive':True,'new_predictor_forwards':0,'selected_after_completed_official_VAL':True})

if __name__=='__main__':torch.set_num_threads(4);main()
