"""Fresh best-checkpoint VAL, exact three-model pairing and registered inference."""
import csv
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage4f_common import (model_new, model_input, SceneDataset, evaluate, config,
    atomic_json, read_json, write_csv, sha256, state_digest, verify_previous,
    BEST, SUMMARY, CONFIG, ACTORS, GROUPS, CLASSES, OLD, STAGE3_ROOT, PREREG)
from stage4a_evaluate import (read_actors, read_membership, actor_key, summarize,
    select as old_select, INTERACTION, MOTION, FIELDS)
from stage4f_global_interactor import FEATURE_NAMES
import numpy as np
import torch
from scipy.stats import spearmanr

B_ACTORS=STAGE3_ROOT/'04_evaluation/stage3b_type_embedding_actor_errors.csv'
C_ACTORS=OLD/'04_evaluation/stage4a_actor_errors.csv'
GATES=ROOT/'04_evaluation/stage4f_actor_gates.csv'
BINS=((0,1),(1,2),(2,5),(5,10),(10,20))


def load_final_model():
    summary=read_json(SUMMARY);assert summary['status']=='COMPLETE'
    assert sha256(BEST)==summary['checkpoint_sha256'] and sha256(CONFIG)==summary['config_sha256']
    saved=torch.load(BEST,map_location='cpu',weights_only=False)
    assert saved['metadata']['phase']=='original_nll'
    model=model_new();model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    assert state_digest(model.state_dict())==summary['model_state_content_sha256']
    return model,saved,summary


@torch.no_grad()
def fresh_val():
    verify_previous();model,saved,summary=load_final_model();gi=model.global_interactor
    fields=['scene_token','sample_token','instance_token','node_in_graph','horizon','agent_type','motion_state','necessity_gate',*FEATURE_NAMES]
    temp=GATES.with_suffix('.csv.tmp');handle=temp.open('w',newline='');writer=csv.DictWriter(handle,fields,lineterminator='\n');writer.writeheader()
    def capture(module,args,output):
        data=args[0];obs=module.necessity_gate_observation
        ids=obs['node_ids'].cpu().numpy();values=obs['gate'][obs['node_ids']].cpu().numpy();features=obs['features'].cpu().numpy()
        graph=data.batch.cpu().numpy();ptr=data.ptr.cpu().numpy();types=data.agent_type.cpu().numpy()
        future=data.future_mask.cpu();target=data.target_mask.cpu()
        for j,node in enumerate(ids):
            k=int(graph[node]);local=int(node-ptr[k]);steps=int(future[node].sum())
            horizon='full_horizon' if bool(target[node]) and steps==12 else ('partial_future' if bool(target[node]) and steps>0 else 'context_only')
            row=dict(zip(fields,[data.scene_token[k],data.sample_token[k],data.instance_tokens[k][local],local,horizon,CLASSES[int(types[node])],data.t0_motion_state[k][local],float(values[j]),*features[j].tolist()]))
            writer.writerow(row)
    gi.record_necessity_gate=True;hook=gi.register_forward_hook(capture)
    try:measured=evaluate(SceneDataset('val'),model,actor_path=ACTORS,progress=True)
    finally:hook.remove();handle.close();gi.record_necessity_gate=False;gi.necessity_gate_observation=None
    os.replace(temp,GATES)
    assert measured['windows']==3603 and len(measured['scenes'])==150
    checks=[];actual=measured['metrics']['full_horizon'];selected=summary['selected_full_horizon_metrics']
    tolerances=read_json(PREREG)['final_reload_metric_tolerance']
    for group in GROUPS:
        assert actual[group]['count']==selected[group]['count']
        if actual[group]['count']:
            for metric in FIELDS:
                difference=abs(actual[group][metric]-selected[group][metric]);tol=tolerances[metric]
                checks.append({'group':group,'metric':metric,'absolute_difference':difference,'tolerance':tol,'passed':difference<tol})
    atomic_json(ROOT/'04_evaluation/stage4f_fresh_val_reconciliation.json',{'status':'PASS' if all(r['passed'] for r in checks) else 'FAIL','checks':checks,'fresh_metrics_authoritative':True,'checkpoint_sha256':sha256(BEST)})
    assert all(r['passed'] for r in checks),'Fresh/training reconciliation failed'
    measured.update(status='PASS',checkpoint_sha256=sha256(BEST),checkpoint_metadata=saved['metadata'],
        actor_errors_sha256=sha256(ACTORS),actor_gates_sha256=sha256(GATES),interaction_necessity_gate=True,
        fresh_complete_official_VAL=True,NaN=0,Inf=0)
    atomic_json(ROOT/'04_evaluation/stage4f_metrics.json',measured)
    del model;torch.cuda.empty_cache();print('FRESH_VAL_PASS',actual['overall'],flush=True)


def paired():
    sources={'Stage3B':B_ACTORS,'Stage4A':C_ACTORS,'Stage4F':ACTORS}
    maps={name:{actor_key(r):r for r in read_actors(path)} for name,path in sources.items()};membership=read_membership()
    b=maps['Stage3B'];assert all(m.keys()==b.keys()==membership.keys() for m in maps.values())
    identity=('scene_token','sample_token','instance_token','horizon','node_in_graph','agent_type',
        'agent_type_id','motion_state','valid_future_steps','future_mask_bits','GT_trajectory_sha256')
    max_disp=0.
    for key,base in b.items():
        for name,m in maps.items():
            row=m[key];assert all(row[f]==base[f] for f in identity),(name,key)
            max_disp=max(max_disp,abs(float(row['GT_endpoint_displacement_m'])-float(base['GT_endpoint_displacement_m'])))
            assert float(row['MR6'])==float(float(row['minFDE6'])>2)
        assert all(base[f]==membership[key][f] for f in ('node_in_graph','agent_type','agent_type_id'))
    assert max_disp<=1e-6
    atomic_json(ROOT/'04_evaluation/stage4f_pairing_audit.json',{'status':'PASS','full':54990,'partial':30037,'total':85027,
        'identity_fields':identity,'all_three_models_identical':True,'GT_displacement_max_diff':max_disp,
        'actor_csv_sha256':{k:sha256(v) for k,v in sources.items()},'frozen_interaction_ledger_reused':True})
    return maps,membership


def select_keys(b,group,membership):
    if group=='Pedestrian <5m':return [k for k,r in b.items() if k[-1]=='full_horizon' and r['agent_type']=='pedestrian' and float(r['GT_endpoint_displacement_m'])<5]
    if group in tuple(f'Pedestrian {lo}-{hi}m' for lo,hi in BINS):
        lo,hi=map(float,group.split()[1].rstrip('m').split('-'))
        return [k for k,r in b.items() if k[-1]=='full_horizon' and r['agent_type']=='pedestrian' and lo<=float(r['GT_endpoint_displacement_m'])<hi]
    if group=='non-heterogeneous':return [k for k in b if k[-1]=='full_horizon' and membership[k]['Heterogeneous-20m']=='0']
    return [actor_key(r) for r in old_select(list(b.values()),group,membership)]


def tables(maps,membership):
    groups=GROUPS+tuple(cls.capitalize()+f' >{t}m' for cls,t in MOTION)+INTERACTION+('Pedestrian <5m',)+tuple(f'Pedestrian {lo}-{hi}m' for lo,hi in BINS)
    ablation=[];long=[]
    for group in groups:
        keys=select_keys(maps['Stage3B'],group,membership)
        row={'Group':group,'Count':len(keys),'Instances':len({k[2] for k in keys}),'Scenes':len({k[0] for k in keys})}
        measured={}
        for name,m in maps.items():
            values=summarize([m[k] for k in keys]) if keys else {f:None for f in FIELDS}
            measured[name]=values
            for f in FIELDS:row[name+'_'+f]=values[f]
            long.append({'Group':group,'Model':name,'Count':len(keys),'Instances':row['Instances'],'Scenes':row['Scenes'],**{f:values[f] for f in FIELDS}})
        for f in FIELDS:
            row['D-B_'+f]=measured['Stage4F'][f]-measured['Stage3B'][f] if keys else None
            row['D-C_'+f]=measured['Stage4F'][f]-measured['Stage4A'][f] if keys else None
        ablation.append(row)
    write_csv(ROOT/'06_tables/stage4f_three_model_ablation.csv',ablation)
    write_csv(ROOT/'06_tables/stage4f_main_results.csv',[r for r in long if r['Group'] in GROUPS])
    write_csv(ROOT/'06_tables/stage4f_nontrivial_motion_metrics.csv',[r for r in ablation if ' >' in r['Group']])
    write_csv(ROOT/'06_tables/stage4f_pedestrian_motion_bin_ablation.csv',[r for r in ablation if r['Group'] in [f'Pedestrian {lo}-{hi}m' for lo,hi in BINS]])
    write_csv(ROOT/'06_tables/stage4f_low_motion_recovery.csv',[r for r in ablation if r['Group'] in ('Pedestrian <5m','Pedestrian 5-10m')])
    write_csv(ROOT/'06_tables/stage4f_interaction_subgroup_metrics.csv',[r for r in ablation if r['Group'] in INTERACTION])
    return ablation


def bootstrap(maps,membership):
    b=maps['Stage3B'];scenes=sorted({r['scene_token'] for r in b.values()});assert len(scenes)==150
    si={s:i for i,s in enumerate(scenes)};draws=np.random.default_rng(2022).integers(0,150,(1000,150));weights=np.stack([np.bincount(d,minlength=150) for d in draws])
    planned={'overall':('minADE6','minFDE6','Top1FDE6'),'vehicle':('minADE6','minFDE6'),
        'pedestrian':('minADE6','minFDE6','Top1FDE6'),'bicycle':('minFDE6',),
        **{g:('minFDE6',) for g in ('vehicle.moving','Vehicle >5m','Pedestrian <5m','Pedestrian 5-10m',*INTERACTION)},
        **{f'Pedestrian {lo}-{hi}m':('minFDE6',) for lo,hi in BINS}}
    rows=[];result={'status':'PASS','replicates':1000,'seed':2022,'scenes':150,'method':'paired scene-cluster percentile bootstrap; equally weighted actor-windows pooled within resampled clusters','comparisons':{}}
    for label,reference in (('D-B','Stage3B'),('D-C','Stage4A')):
        result['comparisons'][label]={}
        for group,metrics in planned.items():
            keys=select_keys(b,group,membership);counts=np.bincount([si[k[0]] for k in keys],minlength=150);den=weights@counts
            assert (den>0).all();result['comparisons'][label][group]={}
            for metric in metrics:
                sums=np.zeros(150)
                for k in keys:sums[si[k[0]]]+=float(maps['Stage4F'][k][metric])-float(maps[reference][k][metric])
                samples=(weights@sums)/den;lo,hi=np.quantile(samples,[.025,.975]);delta=float(sums.sum()/counts.sum())
                item={'delta':delta,'CI95':[float(lo),float(hi)],'Count':len(keys),'Scenes':int((counts>0).sum())}
                result['comparisons'][label][group][metric]=item
                rows.append({'Comparison':label,'Group':group,'Metric':metric,'Delta':delta,'CI95_lower':float(lo),'CI95_upper':float(hi),'Count':len(keys),'Scenes':item['Scenes'],'Replicates':1000})
    atomic_json(ROOT/'04_evaluation/stage4f_bootstrap_ci.json',result);write_csv(ROOT/'06_tables/stage4f_bootstrap_ci.csv',rows)
    return result


def gate_statistics(maps,membership):
    with GATES.open() as f:all_rows=list(csv.DictReader(f))
    assert all(0<=float(r['necessity_gate'])<=1 for r in all_rows)
    gates={actor_key(r):r for r in all_rows if r['horizon']!='context_only'}
    assert gates.keys()==maps['Stage3B'].keys();full=[r for r in gates.values() if r['horizon']=='full_horizon']
    groups=GROUPS[:-1]+tuple(f'Pedestrian {lo}-{hi}m' for lo,hi in BINS)+('Heterogeneous-20m','non-heterogeneous')
    rows=[]
    def stats(label,rr,population):
        g=np.array([float(r['necessity_gate']) for r in rr]);assert len(g)
        return {'Group':label,'Population':population,'Count':len(g),'mean':float(g.mean()),'std':float(g.std()),'median':float(np.median(g)),
            **{f'p{n}':float(np.percentile(g,n)) for n in (10,25,75,90)},'fraction_lt0p1':float((g<.1).mean()),'fraction_gt0p5':float((g>.5).mean()),'fraction_gt0p9':float((g>.9).mean()),
            'fraction_lt0p05':float((g<.05).mean()),'fraction_gt0p95':float((g>.95).mean())}
    for group in groups:
        keys=select_keys(maps['Stage3B'],group,membership)
        if keys:rows.append(stats(group,[gates[k] for k in keys],'full_horizon'))
    rows.append(stats('overall',[r for r in gates.values() if r['horizon']=='partial_future'],'partial_future'))
    rows.append(stats('overall',all_rows,'all_current_valid_context'))
    correlations=[]
    for population,rr in (('full_horizon',full),('all_current_valid_context',all_rows)):
        g=np.array([float(r['necessity_gate']) for r in rr])
        for feature in FEATURE_NAMES[3:]:
            v=np.array([float(r[feature]) for r in rr]);rho,p=spearmanr(v,g)
            defined=bool(np.isfinite(rho) and np.isfinite(p))
            correlations.append({'Population':population,'Feature':feature,'Spearman_rho':float(rho) if defined else None,'Descriptive_p_value':float(p) if defined else None,'Count':len(g),'Interpretation':'descriptive association, not causal; overlapping windows' if defined else 'undefined because an input is constant; no fabricated correlation'})
    context=rows[-1];collapse=context['fraction_lt0p05']>.95 or context['fraction_gt0p95']>.95
    result={'status':'PASS','groups':rows,'gate_collapse':'YES' if collapse else 'NO','collapse_population':'all current valid actor-window contexts',
        'motion_bins_GT_offline_only':True,'actor_gates_sha256':sha256(GATES),'all_current_valid_actor_observations':len(all_rows),'full':len(full),'partial':len(gates)-len(full),'correlation_is_not_causal':True}
    atomic_json(ROOT/'04_evaluation/stage4f_gate_statistics.json',result)
    write_csv(ROOT/'06_tables/stage4f_gate_statistics.csv',rows);write_csv(ROOT/'06_tables/stage4f_gate_feature_correlations.csv',correlations)
    return result


def decision(boot):
    db=boot['comparisons']['D-B'];dc=boot['comparisons']['D-C'];get=lambda x,g:x[g]['minFDE6']
    overall=get(db,'overall');ped=get(db,'pedestrian');low=get(db,'Pedestrian <5m');recovery=get(dc,'Pedestrian <5m')
    moving=any(get(db,g)['CI95'][1]<0 for g in ('vehicle.moving','Vehicle >5m'))
    harm=any(get(db,g)['CI95'][0]>0 for g in ('vehicle','pedestrian','bicycle'))
    recovered=recovery['delta']<0;low_safe=low['CI95'][0]<=0
    eligible=not harm and low_safe and recovered and moving
    supported=eligible and overall['delta']<0 and overall['CI95'][1]<0
    partial=eligible and overall['CI95'][0]<=0<=overall['CI95'][1]
    science='SUPPORTED' if supported else ('PARTIAL' if partial else 'NOT SUPPORTED')
    readiness=supported or (partial and recovery['CI95'][1]<0)
    result={'scientific_decision':science,'Stage4F':'PASS','Ready_Reliability_Head':'YES' if readiness else 'NO',
        'overall_gain_reliable':overall['CI95'][1]<0,'pedestrian_reliable_harm':ped['CI95'][0]>0,
        'major_class_harm_guard':harm,'moving_vehicle_gain_reliable':moving,
        'low_motion_recovery_from_C':recovered,'low_motion_recovery_from_C_reliable':recovery['CI95'][1]<0,
        'low_motion_D_minus_B_no_reliable_harm':low_safe,'Reliability_executed':False,'additional_seeds_executed':False,'STOP':True}
    atomic_json(ROOT/'04_evaluation/stage4f_scientific_decision.json',result);return result


def main():
    fresh_val();maps,membership=paired();tables(maps,membership);boot=bootstrap(maps,membership);gate_statistics(maps,membership);decision(boot);verify_previous(shards=True)
    print('STAGE4F_EVALUATION_PASS',flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
