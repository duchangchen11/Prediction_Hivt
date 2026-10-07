"""Fresh reload, exact paired identities, scene bootstrap and frozen decisions."""
from pathlib import Path
import sys, math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7a_formal_common import *
from stage3b_common import model_new as baseline_new
import numpy as np
import pandas as pd
import torch

IDS=['scene_token','sample_token','instance_token','horizon']
AUDIT=['node_in_graph','GT_trajectory_sha256','future_mask_bits','agent_type_id','agent_type','motion_state','valid_future_steps']
GROUPS_FINAL=['overall',*CLASSES,'vehicle.moving','vehicle.stopped','vehicle.parked','unknown',
             'Vehicle >5m','Pedestrian <5m','Pedestrian >5m','IntersectionVehicle20',
             'NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT','Left-context','Right-context']

def group_mask(frame,group):
    if group=='overall':return np.ones(len(frame),dtype=bool)
    if group in CLASSES:return (frame.agent_type==group).to_numpy()
    if group.startswith('vehicle.') or group=='unknown':return ((frame.agent_type=='vehicle')&(frame.motion_state==group)).to_numpy()
    if group=='Vehicle >5m':return ((frame.agent_type=='vehicle')&(frame.GT_endpoint_displacement_m>5)).to_numpy()
    if group.startswith('Pedestrian '):
        return ((frame.agent_type=='pedestrian')&((frame.GT_endpoint_displacement_m<5) if '<' in group else (frame.GT_endpoint_displacement_m>5))).to_numpy()
    if group in ('Left-context','Right-context'):
        return ((frame.turn_context==group.split('-')[0].lower())&((frame.TurningVehicle_GT==1)|(frame.NearTurnConnector20==1))).to_numpy()
    return (frame[group]==1).to_numpy()

def load_pair():
    base=pd.read_csv(ROOT/'04_evaluation/stage7a_baseline_actor_errors.csv',dtype={'future_mask_bits':str})
    on=pd.read_csv(ROOT/'04_evaluation/stage7a_on_actor_errors.csv',dtype={'future_mask_bits':str})
    zero=pd.read_csv(ROOT/'04_evaluation/stage7a_zero_actor_errors.csv',dtype={'future_mask_bits':str})
    frames=[]
    for frame in (base,on,zero):
        assert not frame.duplicated(IDS).any() and len(frame)==85027
        frames.append(frame.set_index(IDS).sort_index())
    base,on,zero=frames
    assert base.index.equals(on.index) and base.index.equals(zero.index)
    for name in AUDIT:
        assert (base[name].to_numpy()==on[name].to_numpy()).all(),name
        assert (base[name].to_numpy()==zero[name].to_numpy()).all(),name
    assert np.allclose(base.GT_endpoint_displacement_m,on.GT_endpoint_displacement_m,atol=0,rtol=0)
    assert np.isfinite(base[list(METRICS)].to_numpy()).all() and np.isfinite(on[list(METRICS)].to_numpy()).all() and np.isfinite(zero[list(METRICS)].to_numpy()).all()
    for f in (base,on,zero):
        assert (f.index.get_level_values('horizon')=='full_horizon').sum()==54990
        assert (f.index.get_level_values('horizon')=='partial_future').sum()==30037
    sidecar=pd.read_csv(ROOT/'01_data_audit/stage7a_formal_actor_val_groups.csv',dtype={'future_mask_bits':str}).set_index(IDS)
    for frame in (base,on,zero):
        full=frame.loc[frame.index.get_level_values('horizon')=='full_horizon']
        s=sidecar.loc[full.index]
        for k in ('GT_trajectory_sha256','future_mask_bits','agent_type_id'):assert (full[k].to_numpy()==s[k].to_numpy()).all()
    atomic_json(ROOT/'04_evaluation/stage7a_pairing_audit.json',{'status':'PASS','paired_actor_windows':85027,
      'full_horizon':54990,'partial':30037,'NaN':0,'Inf':0,'identity_fields':IDS+AUDIT,
      'Stage3B_checkpoint_sha256':sha256(BASE_BEST),'Stage7A_checkpoint_sha256':sha256(BEST),
      'baseline_actor_sha256':sha256(ROOT/'04_evaluation/stage7a_baseline_actor_errors.csv'),
      'ON_actor_sha256':sha256(ROOT/'04_evaluation/stage7a_on_actor_errors.csv'),
      'ZERO_actor_sha256':sha256(ROOT/'04_evaluation/stage7a_zero_actor_errors.csv')})
    def attach(f):
        f=f.reset_index();return f.merge(sidecar.reset_index()[IDS+['IntersectionVehicle20','NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT','turn_context']],on=IDS,how='left',validate='one_to_one')
    return tuple(attach(f) for f in (base,on,zero))

def tables_bootstrap_decision():
    b,o,z=load_pair();full=[f[f.horizon=='full_horizon'].reset_index(drop=True) for f in (b,o,z)];bf,of,zf=full
    rows=[];ablation=[]
    for horizon in HORIZONS:
        bb=b[b.horizon==horizon];oo=o[o.horizon==horizon]
        for group in GROUPS_FINAL:
            mask=group_mask(bb,group);count=int(mask.sum())
            if not count:continue
            for label,f in [('Stage3B',bb),('Stage7A',oo)]:
                rows.append({'model':label,'horizon':horizon,'group':group,'count':count,
                             **{k:float(f.loc[mask,k].mean()) for k in METRICS}})
    table=pd.DataFrame(rows)
    table[table.group.isin(['overall',*CLASSES,'Vehicle >5m','Pedestrian <5m','Pedestrian >5m'])].to_csv(ROOT/'06_tables/stage7a_main_results.csv',index=False)
    table[table.group.isin(['vehicle.moving','vehicle.stopped','vehicle.parked','unknown'])].to_csv(ROOT/'06_tables/stage7a_vehicle_motion_results.csv',index=False)
    table[table.group.isin(GROUPS_FINAL[11:])].to_csv(ROOT/'06_tables/stage7a_semantic_subgroup_results.csv',index=False)
    # Explicit lists prevent omission when the main group ordering changes.
    table[table.group.isin(['IntersectionVehicle20','NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT','Left-context','Right-context'])].to_csv(ROOT/'06_tables/stage7a_semantic_subgroup_results.csv',index=False)
    for group in GROUPS_FINAL:
        mask=group_mask(bf,group);count=int(mask.sum())
        if not count:continue
        for name,f in [('Semantic-ON',of),('Semantic-ZERO',zf)]:
            ablation.append({'model':name,'horizon':'full_horizon','group':group,'count':count,**{k:float(f.loc[mask,k].mean()) for k in METRICS}})
    write_csv(ROOT/'06_tables/stage7a_semantic_zero_ablation.csv',ablation)
    scene_tokens=sorted(SceneDataset('val').scene_indices);assert len(scene_tokens)==150
    scene_index={t:i for i,t in enumerate(scene_tokens)};indices=np.array([scene_index[t] for t in bf.scene_token])
    rng=np.random.default_rng(2022);draws=rng.integers(0,150,(1000,150));bootstrap=[]
    comparisons=[(g,'minFDE6') for g in ['overall','vehicle','pedestrian','vehicle.moving','Vehicle >5m','IntersectionVehicle20','NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT']]+[('overall','Top1FDE6')]
    scene_rows=[]
    for group,metric in comparisons:
        mask=group_mask(bf,group);counts=np.bincount(indices[mask],minlength=150)
        sb=np.bincount(indices[mask],weights=bf.loc[mask,metric],minlength=150)
        so=np.bincount(indices[mask],weights=of.loc[mask,metric],minlength=150)
        den=counts[draws].sum(1);assert (den>0).all()
        deltas=(so[draws].sum(1)-sb[draws].sum(1))/den
        low,high=np.quantile(deltas,[0.025,0.975]);bm=float(sb.sum()/counts.sum());om=float(so.sum()/counts.sum())
        bootstrap.append({'group':group,'metric':metric,'count':int(counts.sum()),'Stage3B':bm,'Stage7A':om,'delta':om-bm,
          'CI_lower':float(low),'CI_upper':float(high),'replicates':1000,'seed':2022,'clusters':150,
          'nonempty_clusters':int((counts>0).sum()),'unit':'whole_scene','multiplicity':'unadjusted_secondary'})
        for t,n,a,c in zip(scene_tokens,counts,sb,so):scene_rows.append({'scene_token':t,'group':group,'metric':metric,'count':int(n),'Stage3B_sum':float(a),'Stage7A_sum':float(c)})
    write_csv(ROOT/'06_tables/stage7a_bootstrap_ci.csv',bootstrap);write_csv(ROOT/'06_tables/stage7a_paired_scene_sums.csv',scene_rows)
    ci={r['group']:r for r in bootstrap if r['metric']=='minFDE6'};overall=ci['overall']
    prereg=read_json(PREREG);difficult=prereg['difficult_groups']
    harm=[g for g in ('vehicle','pedestrian') if ci[g]['delta']>0.05*ci[g]['Stage3B'] and ci[g]['CI_lower']>0]
    reliable=[g for g in difficult if ci[g]['CI_upper']<0];direction=[g for g in difficult if ci[g]['delta']<0]
    basically_flat=overall['delta']<=0.01*overall['Stage3B']
    if overall['delta']<0 and overall['CI_upper']<0 and not harm and direction:decision='SUPPORTED'
    elif basically_flat and overall['CI_lower']<=0<=overall['CI_upper'] and len(reliable)>=2 and not harm:decision='TARGETED_SUPPORTED'
    else:decision='NOT_SUPPORTED'
    paper=decision in ('SUPPORTED','TARGETED_SUPPORTED')
    ready=paper or (basically_flat and overall['CI_lower']<=0 and all(ci[g]['CI_upper']<0 for g in ('vehicle.moving','TurningVehicle_GT')) and not harm)
    atomic_json(ROOT/'04_evaluation/stage7a_scientific_decision.json',{'SemanticMap':decision,'PaperUsableSemantic':'YES' if paper else 'NO',
      'ReadyStage7B':'YES' if ready else 'NO','marked_reliable_harm':harm,'reliable_difficult_groups':reliable,
      'point_improved_difficult_groups':direction,'registered_before_training':True,'test_used':False,'Stage7B_executed':False})
    print('SCIENTIFIC_DECISION',decision,'reliable groups',reliable,'harm',harm,flush=True)

def main():
    verify_previous();prereg=read_json(PREREG)
    assert sha256(ROOT/'01_data_audit/stage7a_formal_actor_val_groups.csv')==prereg['group_sidecar_sha256']
    assert read_json(SUMMARY)['status']=='COMPLETE'
    saved=torch.load(BEST,map_location='cpu',weights_only=False)
    for name in ('baseline','on','zero'):
        result_path=ROOT/f'04_evaluation/stage7a_{name}_final_metrics.json'
        actor_path=ROOT/f'04_evaluation/stage7a_{name}_actor_errors.csv'
        if result_path.exists() and actor_path.exists():continue
        if name=='baseline':
            model=baseline_new();cp=torch.load(BASE_BEST,map_location='cpu',weights_only=False)
            model.load_state_dict(cp['state_dict'],strict=True);del cp;ds=SceneDataset('val')
        else:
            model=model_new();model.load_state_dict(saved['state_dict'],strict=True)
            ds=Stage7ASemanticDataset('val',semantic_zero=name=='zero')
        assert len(ds.scene_indices)==150 and len(ds)==3603
        measured=evaluate(ds,model,actor_path=actor_path,progress=True)
        assert measured['metrics']['full_horizon']['overall']['count']==54990
        assert measured['metrics']['partial_future']['overall']['count']==30037
        measured.update(fresh_checkpoint_reload=True,checkpoint_sha256=sha256(BASE_BEST if name=='baseline' else BEST),
                        semantic_setting=name,NaN=0,Inf=0)
        if name=='on':
            selected=saved['metadata']['full_horizon_metrics']
            for group in GROUPS:
                for metric in METRICS:assert abs(measured['metrics']['full_horizon'][group][metric]-selected[group][metric])<1e-5
            measured['fresh_reload_max_metric_difference']=max(abs(measured['metrics']['full_horizon'][g][k]-selected[g][k]) for g in GROUPS for k in METRICS)
        atomic_json(result_path,measured);del model,ds;torch.cuda.empty_cache()
        print('FINAL_EVALUATION',name,measured['metrics']['full_horizon']['overall'],flush=True)
    tables_bootstrap_decision();verify_previous()

if __name__=='__main__':torch.set_num_threads(4);main()
