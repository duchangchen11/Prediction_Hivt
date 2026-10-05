"""Fresh Type VAL, exact frozen-GT pairing, descriptive motion and paired scene CIs."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import (CONFIG,FREEZE,PREREG,BEST,SUMMARY,ACTORS,BASE_ACTORS,GT_LEDGER,
    CLASSES,GROUPS,HORIZONS,METRICS,SceneDataset,config,model_new,evaluate,atomic_json,
    read_json,write_csv,sha256,verify_previous,state_digest)
from stage3b_pairing_ledger import actor_key
import numpy as np
import torch

FIELDS=('minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL')
LABELS={'overall':'Overall','vehicle':'Vehicle','pedestrian':'Pedestrian','bicycle':'Bicycle'}
MOTION=(('vehicle',5),('pedestrian',1),('pedestrian',5),('bicycle',1),('bicycle',5))

def read_actors(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==85027 and len({actor_key(row) for row in rows})==len(rows)
    values=np.array([[float(r[k]) for k in FIELDS+('independent_minADE6','GT_endpoint_displacement_m')] for r in rows])
    assert np.isfinite(values).all()
    assert sum(r['horizon']=='full_horizon' for r in rows)==54990
    return rows

def select(rows,group):
    if ' >' in group:
        cls,threshold=group.split(' >');threshold=float(threshold.rstrip('m'));cls=cls.lower()
        return [r for r in rows if r['horizon']=='full_horizon' and r['agent_type']==cls and float(r['GT_endpoint_displacement_m'])>threshold]
    return [r for r in rows if r['horizon']=='full_horizon' and
            (group=='overall' or r['agent_type']==group or (r['agent_type']=='vehicle' and r['motion_state']==group))]

def summary(rows):
    assert rows;values=np.array([[float(r[k]) for k in FIELDS] for r in rows])
    assert np.isfinite(values).all()
    return {'Count':len(rows),**dict(zip(FIELDS,map(float,values.mean(0))))}

def strict_pairing(no_type,typed):
    with GT_LEDGER.open() as f:ledger=list(csv.DictReader(f))
    ground={actor_key(r):r for r in ledger};a={actor_key(r):r for r in no_type};b={actor_key(r):r for r in typed}
    assert len(ground)==len(ledger)==len(a)==len(b)==85027
    assert a.keys()==b.keys()==ground.keys(),'Actor mismatch; stop comparison'
    ledger_audit=read_json(ROOT/'04_evaluation/stage3b_frozen_GT_ledger_audit.json')
    assert ledger_audit['status']=='PASS' and sha256(GT_LEDGER)==ledger_audit['GT_ledger_sha256']
    for name,digest in ledger_audit['VAL_shards_SHA256_verified'].items():assert sha256(ROOT/name)==digest
    maximum_displacement_difference=0.0
    for key,old in a.items():
        new=b[key];gt=ground[key]
        for name in ('scene_token','sample_token','instance_token','horizon','node_in_graph','agent_type','motion_state','valid_future_steps'):
            assert old[name]==new[name]==gt[name],(key,name)
        for name in ('GT_trajectory_sha256','future_mask_bits','agent_type_id'):assert new[name]==gt[name],(key,name)
        assert int(new['agent_type_id'])==CLASSES.index(new['agent_type'])
        valid=int(new['valid_future_steps']);assert new['future_mask_bits'].count('1')==valid
        assert valid==12 if new['horizon']=='full_horizon' else 1<=valid<12
        maximum_displacement_difference=max(maximum_displacement_difference,abs(float(old['GT_endpoint_displacement_m'])-float(new['GT_endpoint_displacement_m'])))
        assert float(new['MR6'])==float(float(new['minFDE6'])>2)
    assert maximum_displacement_difference<=1e-6
    audit={'status':'PASS','paired_full_horizon_actors':54990,'paired_partial_future_actors':30037,
        'paired_total_actor_windows':85027,'scene_sample_instance_horizon_node_identity_equal':True,
        'agent_type_equal':True,'future_mask_bitwise_equal':True,'GT_trajectory_float32_tensor_SHA256_equal':True,
        'GT_audit_source':'Original unchanged Stage3A frozen graph ledger; old No-Type CSV has no per-timestep GT columns',
        'maximum_GT_endpoint_displacement_difference_m':maximum_displacement_difference,
        'NoType_actor_errors_sha256':sha256(BASE_ACTORS),'Type_actor_errors_sha256':sha256(ACTORS),
        'frozen_GT_ledger_sha256':sha256(GT_LEDGER),'VAL_scenes':150,'test_used':False}
    atomic_json(ROOT/'04_evaluation/stage3b_pairing_audit.json',audit);return a,b

def main_tables(no_type,typed,measured):
    full=measured['metrics']['full_horizon']
    for group in GROUPS:
        aggregated=summary(select(typed,group));assert aggregated['Count']==full[group]['count']
        assert all(abs(aggregated[k]-full[group][k])<1e-10 for k in FIELDS)
    write_csv(ROOT/'06_tables/stage3b_type_embedding_main_results.csv',[
        {'Group':LABELS.get(group,group),'Count':values['count'],**{k:values[k] for k in FIELDS}} for group,values in full.items()])
    motion=[]
    for cls,threshold in MOTION:
        name=cls.capitalize()+f' >{threshold}m';chosen=select(typed,name);values=summary(chosen)
        motion.append({'Group':name,'AgentType':cls,'GT_endpoint_displacement_gt_m':threshold,'Count':values['Count'],
            'UniqueInstances':len({r['instance_token'] for r in chosen}),'UniqueScenes':len({r['scene_token'] for r in chosen}),
            **{k:values[k] for k in FIELDS[:-1]}})
    assert [r['Count'] for r in motion]==[9744,8810,7581,155,132]
    write_csv(ROOT/'06_tables/stage3b_nontrivial_motion_metrics.csv',motion)
    atomic_json(ROOT/'04_evaluation/stage3b_nontrivial_motion_metrics.json',{'status':'PASS','metrics':motion,
        'count_unit':'actor-window; overlapping windows not independent','limited_unique_bicycle_instances_scenes':True,
        'Bicycle_gt5_significance_test':False,'source_actor_csv_sha256':sha256(ACTORS)})
    rows=[]
    for group in ('overall','vehicle','pedestrian','bicycle','vehicle.moving','Vehicle >5m','Pedestrian >5m','Bicycle >5m'):
        old=summary(select(no_type,group));new=summary(select(typed,group));assert old['Count']==new['Count']
        row={'Group':LABELS.get(group,group),'Count':old['Count']}
        for metric,short in (('minADE6','ADE'),('minFDE6','FDE')):
            row.update({f'NoType_min{short}6':old[metric],f'Type_min{short}6':new[metric],
                f'Delta_{short}':new[metric]-old[metric],f'Relative_{short}_change':(new[metric]-old[metric])/old[metric]})
        row.update(NoType_MR6=old['MR6'],Type_MR6=new['MR6'],Delta_MR=new['MR6']-old['MR6'],
                   NoType_Top1FDE6=old['Top1FDE6'],Type_Top1FDE6=new['Top1FDE6'],Delta_Top1FDE=new['Top1FDE6']-old['Top1FDE6'])
        rows.append(row)
    write_csv(ROOT/'06_tables/stage3b_type_embedding_ablation.csv',rows)
    atomic_json(ROOT/'04_evaluation/stage3b_type_embedding_ablation.json',{'status':'PASS','delta':'Type-NoType; negative favors Type',
        'relative_change_unit':'fraction, not percent','groups':rows,'NoType_actor_errors_sha256':sha256(BASE_ACTORS),
        'Type_actor_errors_sha256':sha256(ACTORS),'pairing_audit_sha256':sha256(ROOT/'04_evaluation/stage3b_pairing_audit.json')})
    return rows

def bootstrap(no_type,typed,scenes):
    assert len(scenes)==150;scenes=sorted(scenes);scene_index={token:i for i,token in enumerate(scenes)}
    rng=np.random.default_rng(2022);draws=rng.integers(0,150,size=(1000,150))
    multiplicities=np.stack([np.bincount(row,minlength=150) for row in draws])
    baseline={actor_key(r):r for r in no_type}
    planned={'overall':('minADE6','minFDE6','Top1FDE6'),'vehicle':('minADE6','minFDE6'),
        'pedestrian':('minADE6','minFDE6'),'vehicle.moving':('minFDE6',),
        'Vehicle >5m':('minFDE6',),'Pedestrian >5m':('minFDE6',),'bicycle':('minADE6','minFDE6')}
    result={'status':'PASS','method':'paired scene-cluster percentile bootstrap','replicates':1000,'seed':2022,
        'resampling_unit':'official VAL scene','scene_count':150,'confidence':0.95,'delta':'Type-NoType; negative favors Type',
        'weighting':'pool equally weighted actor-window deltas across complete resampled scene clusters',
        'Bicycle_gt5_significance_test':False,'Bicycle_gt5_reason':'limited unique bicycle instances/scenes; descriptive only','groups':{}}
    csv_rows=[]
    for group,metrics in planned.items():
        selected=select(typed,group);count=np.zeros(150);sums={k:np.zeros(150) for k in metrics}
        for row in selected:
            i=scene_index[row['scene_token']];count[i]+=1;old=baseline[actor_key(row)]
            for k in metrics:sums[k][i]+=float(row[k])-float(old[k])
        denominator=multiplicities@count;assert np.all(denominator>0)
        fields={}
        for metric,values in sums.items():
            samples=(multiplicities@values)/denominator;lo,hi=np.quantile(samples,[.025,.975]);point=float(values.sum()/count.sum())
            fields[metric]={'estimate_m':point,'CI95_m':[float(lo),float(hi)],'bootstrap_std_m':float(samples.std(ddof=1)),
                'valid_replicates':1000,'actor_windows':int(count.sum()),'unique_scenes_with_targets':int((count>0).sum())}
            csv_rows.append({'Group':LABELS.get(group,group),'Metric':metric,'Delta_Type_minus_NoType_m':point,
                'CI95_lower_m':float(lo),'CI95_upper_m':float(hi),'ActorWindows':int(count.sum()),
                'UniqueScenes':int((count>0).sum()),'ResamplingUnit':'scene','Replicates':1000})
        result['groups'][group]=fields
    atomic_json(ROOT/'04_evaluation/stage3b_bootstrap_ci.json',result)
    write_csv(ROOT/'06_tables/stage3b_bootstrap_ci.csv',csv_rows)
    return result

@torch.no_grad()
def main():
    torch.set_num_threads(4);verify_previous();training=read_json(SUMMARY);assert training['status']=='COMPLETE'
    assert sha256(BEST)==training['checkpoint_sha256'] and sha256(CONFIG)==training['config_sha256']
    saved=torch.load(BEST,map_location='cpu',weights_only=False);assert saved['metadata']['phase']=='original_nll'
    model=model_new();model.load_state_dict(saved['state_dict']);assert state_digest(model.state_dict())==training['model_state_content_sha256']
    assert all(torch.isfinite(p).all() for p in model.parameters())
    measured=evaluate(SceneDataset('val'),model,actor_path=ACTORS,progress=True)
    assert measured['windows']==3603 and len(measured['scenes'])==150
    selected=training['selected_full_horizon_metrics'];actual=measured['metrics']['full_horizon']
    assert all(actual[g]['count']==selected[g]['count'] for g in GROUPS)
    assert all(abs(actual[g][m]-selected[g][m])<1e-6 for g in GROUPS for m in FIELDS)
    no_type=read_actors(BASE_ACTORS);typed=read_actors(ACTORS);strict_pairing(no_type,typed)
    measured.update(status='PASS',checkpoint_sha256=sha256(BEST),checkpoint_relative_path=str(BEST.relative_to(ROOT)),
        checkpoint_metadata=saved['metadata'],actor_errors_sha256=sha256(ACTORS),NaN=0,Inf=0,
        fresh_complete_official_VAL=True,full_horizon_paired_actor_windows=54990)
    atomic_json(ROOT/'04_evaluation/stage3b_type_embedding_metrics.json',measured)
    tables=main_tables(no_type,typed,measured);ci=bootstrap(no_type,typed,measured['scenes'])
    embedding=model.type_embedding.weight.detach().cpu().double().numpy()
    norms=np.linalg.norm(embedding,axis=1);normalized=embedding/norms[:,None];cosine=normalized@normalized.T
    atomic_json(ROOT/'04_evaluation/stage3b_type_embedding_vectors.json',{'status':'PASS','class_order':list(CLASSES),
        'embedding_vectors':embedding.tolist(),'L2_norm':dict(zip(CLASSES,map(float,norms))),
        'cosine_similarity_matrix':cosine.tolist(),'embedding_distances_are_not_physical_semantics':True,
        'checkpoint_sha256':sha256(BEST)})
    verify_previous();print('STAGE3B_FRESH_EVALUATION=PASS',actual['overall'],flush=True)
    print('STAGE3B_PAIRED_BOOTSTRAP',ci['groups']['overall'],flush=True)

if __name__=='__main__':main()
