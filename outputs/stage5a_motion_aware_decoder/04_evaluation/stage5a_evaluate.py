"""Fresh official VAL, strict B/E pairing, scene bootstrap and decoder behavior."""
from pathlib import Path
import sys
import csv
import os
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage5a_common import (BEST, SUMMARY, ACTORS, BASE_ACTORS, GT_LEDGER, CONFIG, PREREG,
    CLASSES, GROUPS, HORIZONS, SceneDataset, model_new, model_input, evaluate, config,
    atomic_json, read_json, write_csv, sha256, state_digest, verify_previous)
from stage5a_decoder import FEATURE_NAMES
import torch
import numpy as np
from scipy.stats import spearmanr

FIELDS=('minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL')
BINS=((0,1),(1,2),(2,5),(5,10),(10,20))
BIN_GROUPS=tuple(f'Pedestrian {lo}-{hi}m' for lo,hi in BINS)
ALL_GROUPS=GROUPS+('Vehicle >5m','Pedestrian <5m','Pedestrian >5m')+BIN_GROUPS
ROUTERS=ROOT/'04_evaluation/stage5a_router_actor_records.csv'


def actor_key(row):
    return tuple(row[k] for k in ('scene_token','sample_token','instance_token','horizon'))


def load_final_model():
    training=read_json(SUMMARY)
    assert training['status']=='COMPLETE' and sha256(BEST)==training['checkpoint_sha256']
    assert sha256(CONFIG)==training['config_sha256']
    saved=torch.load(BEST,map_location='cpu',weights_only=False)
    assert saved['metadata']['phase']=='original_nll'
    model=model_new();model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    assert state_digest(model.state_dict())==training['model_state_content_sha256']
    return model,saved,training


@torch.no_grad()
def fresh_val():
    verify_previous();model,saved,training=load_final_model()
    fields=['scene_token','sample_token','instance_token','node_in_graph','horizon','agent_type','motion_state',
            'r1','r2','entropy','expert1_norm','expert2_norm','expert_cosine','expert_relative_difference',
            'residual_norm',*FEATURE_NAMES]
    temp=ROUTERS.with_suffix('.csv.tmp');handle=temp.open('w',newline='')
    writer=csv.DictWriter(handle,fields,lineterminator='\n');writer.writeheader()
    def capture(module,args,output):
        data=args[0];obs=model.decoder.observation
        condition=obs['condition'].cpu().double().numpy();routing=obs['routing'].cpu().double().numpy()
        expert=obs['expert_outputs'].double()
        a,b=expert[:,:,0],expert[:,:,1]
        norm1=a.norm(dim=-1).mean(0);norm2=b.norm(dim=-1).mean(0)
        cosine=torch.nn.functional.cosine_similarity(a,b,dim=-1,eps=1e-12).mean(0)
        rms1=a.square().mean((0,2)).sqrt();rms2=b.square().mean((0,2)).sqrt()
        relative=(a-b).square().mean((0,2)).sqrt()/torch.maximum(rms1,rms2).clamp(min=1e-12)
        residual=obs['residual'].double().norm(dim=-1).mean(0)
        statistics=torch.stack((norm1,norm2,cosine,relative,residual),-1).cpu().numpy()
        entropy=-(routing*np.log(np.clip(routing,1e-300,1))).sum(-1)
        assert np.isfinite(condition).all() and np.isfinite(routing).all() and np.isfinite(statistics).all()
        graph=data.batch.cpu().numpy();ptr=data.ptr.cpu().numpy();types=data.agent_type.cpu().numpy()
        future=data.future_mask.cpu();target=data.target_mask.cpu();valid_current=~data.padding_mask[:,4].cpu()
        for node in torch.where(valid_current)[0].tolist():
            k=int(graph[node]);local=int(node-ptr[k]);steps=int(future[node].sum())
            horizon='full_horizon' if bool(target[node]) and steps==12 else ('partial_future' if bool(target[node]) and steps>0 else 'context_only')
            writer.writerow(dict(zip(fields,[data.scene_token[k],data.sample_token[k],data.instance_tokens[k][local],local,
                horizon,CLASSES[int(types[node])],data.t0_motion_state[k][local],*routing[node].tolist(),float(entropy[node]),
                *statistics[node].tolist(),*condition[node].tolist()])))
    model.decoder.record_observation=True;hook=model.register_forward_hook(capture)
    try:measured=evaluate(SceneDataset('val'),model,actor_path=ACTORS,progress=True)
    finally:hook.remove();handle.close();model.decoder.record_observation=False;model.decoder.observation=None
    os.replace(temp,ROUTERS)
    assert measured['windows']==3603 and len(measured['scenes'])==150
    actual=measured['metrics']['full_horizon'];selected=training['selected_full_horizon_metrics']
    tolerance=read_json(PREREG)['final_reload_metric_tolerance'];checks=[]
    for group in GROUPS:
        assert actual[group]['count']==selected[group]['count']
        for metric in FIELDS:
            if actual[group]['count']:
                difference=abs(actual[group][metric]-selected[group][metric])
                checks.append({'group':group,'metric':metric,'difference':difference,'tolerance':tolerance[metric],
                               'passed':difference<tolerance[metric]})
    assert all(r['passed'] for r in checks),checks
    atomic_json(ROOT/'04_evaluation/stage5a_fresh_val_reconciliation.json',{'status':'PASS','checks':checks,'fresh_metrics_authoritative':True})
    measured.update(status='PASS',checkpoint_sha256=sha256(BEST),checkpoint_metadata=saved['metadata'],
        actor_errors_sha256=sha256(ACTORS),router_actor_records_sha256=sha256(ROUTERS),fresh_complete_official_VAL=True,NaN=0,Inf=0)
    atomic_json(ROOT/'04_evaluation/stage5a_metrics.json',measured)
    del model;torch.cuda.empty_cache();print('FRESH_VAL_PASS',actual['overall'],flush=True)
    return measured


def read_actors(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==85027 and len({actor_key(r) for r in rows})==85027
    assert sum(r['horizon']=='full_horizon' for r in rows)==54990
    assert sum(r['horizon']=='partial_future' for r in rows)==30037
    assert np.isfinite([[float(r[k]) for k in FIELDS] for r in rows]).all()
    return {actor_key(r):r for r in rows}


def paired():
    baseline=read_actors(BASE_ACTORS);experiment=read_actors(ACTORS)
    with GT_LEDGER.open() as f:ledger={actor_key(r):r for r in csv.DictReader(f)}
    assert baseline.keys()==experiment.keys()==ledger.keys()
    identity=('scene_token','sample_token','instance_token','node_in_graph','horizon','agent_type','motion_state',
              'valid_future_steps','GT_trajectory_sha256','future_mask_bits','agent_type_id')
    maximum=0.
    for key,old in baseline.items():
        new=experiment[key];gt=ledger[key]
        assert all(old[k]==new[k]==gt[k] for k in identity),(key,'GT identity mismatch')
        assert int(new['agent_type_id'])==CLASSES.index(new['agent_type'])
        assert new['future_mask_bits'].count('1')==int(new['valid_future_steps'])
        maximum=max(maximum,abs(float(old['GT_endpoint_displacement_m'])-float(new['GT_endpoint_displacement_m'])))
        assert float(new['MR6'])==float(float(new['minFDE6'])>2)
    assert maximum<=1e-6
    atomic_json(ROOT/'04_evaluation/stage5a_pairing_audit.json',{'status':'PASS','full':54990,'partial':30037,'total':85027,
        'identity_fields':identity,'GT_displacement_max_diff':maximum,'VAL_scenes':150,
        'Stage3B_actor_csv_sha256':sha256(BASE_ACTORS),'Stage5A_actor_csv_sha256':sha256(ACTORS),
        'frozen_GT_ledger_sha256':sha256(GT_LEDGER),'all_fields_equal':True})
    return baseline,experiment


def select_keys(rows,group):
    keys=[]
    for key,row in rows.items():
        if row['horizon']!='full_horizon':continue
        cls=row['agent_type'];d=float(row['GT_endpoint_displacement_m'])
        chosen=(group=='overall' or group==cls or (cls=='vehicle' and group==row['motion_state']))
        if group=='Vehicle >5m':chosen=cls=='vehicle' and d>5
        if group=='Pedestrian <5m':chosen=cls=='pedestrian' and d<5
        if group=='Pedestrian >5m':chosen=cls=='pedestrian' and d>5
        if group in BIN_GROUPS:
            lo,hi=BINS[BIN_GROUPS.index(group)];chosen=cls=='pedestrian' and lo<=d<hi
        if chosen:keys.append(key)
    return keys


def tables(b,e):
    rows=[];long=[]
    for group in ALL_GROUPS:
        keys=select_keys(b,group)
        assert keys
        row={'Group':group,'Count':len(keys),'Instances':len({k[2] for k in keys}),'Scenes':len({k[0] for k in keys})}
        for label,source in (('Stage3B',b),('Stage5A',e)):
            values=np.array([[float(source[k][f]) for f in FIELDS] for k in keys]).mean(0)
            row.update({label+'_'+f:float(v) for f,v in zip(FIELDS,values)})
            long.append({'Group':group,'Model':label,'Count':len(keys),'Instances':row['Instances'],'Scenes':row['Scenes'],
                         **{f:float(v) for f,v in zip(FIELDS,values)}})
        for f in FIELDS:row['Delta_'+f]=row['Stage5A_'+f]-row['Stage3B_'+f]
        rows.append(row)
    write_csv(ROOT/'06_tables/stage5a_main_results.csv',[r for r in long if r['Group'] in ('overall',)+CLASSES])
    write_csv(ROOT/'06_tables/stage5a_comparison_all_groups.csv',rows)
    write_csv(ROOT/'06_tables/stage5a_vehicle_motion.csv',[r for r in rows if r['Group'] in ('vehicle.moving','vehicle.stopped','vehicle.parked','unknown','Vehicle >5m')])
    write_csv(ROOT/'06_tables/stage5a_pedestrian_motion_bins.csv',[r for r in rows if r['Group'] in BIN_GROUPS+('Pedestrian <5m','Pedestrian >5m')])
    return rows


def bootstrap(b,e):
    scenes=sorted({k[0] for k in b});assert len(scenes)==150
    si={s:i for i,s in enumerate(scenes)}
    draws=np.random.default_rng(2022).integers(0,150,(1000,150))
    weights=np.stack([np.bincount(d,minlength=150) for d in draws])
    planned={g:('minFDE6',) for g in ALL_GROUPS}
    planned['overall']=('minADE6','minFDE6','Top1FDE6')
    planned['vehicle']=('minADE6','minFDE6')
    planned['pedestrian']=('minADE6','minFDE6','Top1FDE6')
    result={'status':'PASS','method':'paired scene-cluster percentile bootstrap','seed':2022,'replicates':1000,
        'scenes':150,'confidence':.95,'delta':'Stage5A - Stage3B; negative favors Stage5A',
        'weighting':'pool all actor-window deltas within resampled complete scene clusters','groups':{}}
    rows=[]
    for group,metrics in planned.items():
        keys=select_keys(b,group);count=np.bincount([si[k[0]] for k in keys],minlength=150)
        denominator=weights@count;assert (denominator>0).all()
        result['groups'][group]={}
        for metric in metrics:
            sums=np.zeros(150)
            for k in keys:sums[si[k[0]]]+=float(e[k][metric])-float(b[k][metric])
            replicates=(weights@sums)/denominator;lo,hi=np.quantile(replicates,[.025,.975])
            item={'delta':float(sums.sum()/count.sum()),'CI95':[float(lo),float(hi)],'Count':len(keys),'Scenes':int((count>0).sum())}
            result['groups'][group][metric]=item
            rows.append({'Group':group,'Metric':metric,'Delta':item['delta'],'CI95_lower':float(lo),'CI95_upper':float(hi),
                         'Count':len(keys),'Scenes':item['Scenes'],'Replicates':1000})
    atomic_json(ROOT/'04_evaluation/stage5a_bootstrap_ci.json',result)
    write_csv(ROOT/'06_tables/stage5a_bootstrap_ci.csv',rows)
    return result


def decoder_statistics(b):
    with ROUTERS.open() as f:all_rows=list(csv.DictReader(f))
    routes={actor_key(r):r for r in all_rows if r['horizon']!='context_only'}
    assert routes.keys()==b.keys()
    assert len({actor_key(r) for r in all_rows})==len(all_rows)
    router_rows=[];expert_rows=[]
    def summarize(group,rr,population):
        r=np.array([[float(x['r1']),float(x['r2'])] for x in rr]);assert np.isfinite(r).all()
        assert np.max(np.abs(r.sum(-1)-1))<1e-6 and ((r>=0)&(r<=1)).all()
        entropy=np.array([float(x['entropy']) for x in rr]);dominant=r.argmax(-1)
        router_rows.append({'Group':group,'Population':population,'Count':len(rr),
            'r1_mean':float(r[:,0].mean()),'r2_mean':float(r[:,1].mean()),
            'r1_median':float(np.median(r[:,0])),'r2_median':float(np.median(r[:,1])),
            'entropy_mean':float(entropy.mean()),'entropy_median':float(np.median(entropy)),
            'expert1_dominant_rate':float((dominant==0).mean()),'expert2_dominant_rate':float((dominant==1).mean()),
            'expert1_probability_gt0p9_rate':float((r[:,0]>.9).mean()),'expert2_probability_gt0p9_rate':float((r[:,1]>.9).mean()),
            'tie_rate':float((r[:,0]==r[:,1]).mean())})
        a=np.array([[float(x[k]) for k in ('expert1_norm','expert2_norm','expert_cosine','expert_relative_difference','residual_norm')] for x in rr])
        assert np.isfinite(a).all()
        zero=(a[:,0]<1e-12)&(a[:,1]<1e-12)
        near=(a[:,3]<1e-3)&(a[:,2]>.999)
        expert_rows.append({'Group':group,'Population':population,'Count':len(rr),
            **{k+'_mean':float(a[:,i].mean()) for i,k in enumerate(('expert1_norm','expert2_norm','expert_cosine','expert_relative_difference','residual_norm'))},
            'relative_difference_median':float(np.median(a[:,3])),'near_identical_rate':float(near.mean()),
            'both_numerically_zero_rate':float(zero.mean()),'cosine_zero_vector_convention':0.})
    for group in ALL_GROUPS:
        keys=select_keys(b,group);summarize(group,[routes[k] for k in keys],'full_horizon')
    summarize('overall',all_rows,'all_current_valid_context')
    summarize('overall',[r for r in all_rows if r['horizon']=='partial_future'],'partial_future')
    router_context=router_rows[-2];expert_context=expert_rows[-2]
    router_collapse=max(router_context['expert1_probability_gt0p9_rate'],router_context['expert2_probability_gt0p9_rate'])>.95
    expert_collapse=expert_context['both_numerically_zero_rate']>.95 or expert_context['near_identical_rate']>.95
    correlations=[]
    for group in ('overall','vehicle','pedestrian'):
        rr=[routes[k] for k in select_keys(b,group)]
        for feature in FEATURE_NAMES[3:]:
            rho=spearmanr([float(r['r1']) for r in rr],[float(r[feature]) for r in rr]).statistic
            correlations.append({'Group':group,'Feature':feature,'Spearman_rho':float(rho) if np.isfinite(rho) else None,
                                 'Count':len(rr),'Interpretation':'descriptive, overlapping actor-windows; not causal'})
    write_csv(ROOT/'06_tables/stage5a_router_statistics.csv',router_rows)
    write_csv(ROOT/'06_tables/stage5a_expert_statistics.csv',expert_rows)
    write_csv(ROOT/'06_tables/stage5a_router_motion_correlations.csv',correlations)
    result={'status':'PASS','router_collapse':'YES' if router_collapse else 'NO',
        'expert_functional_collapse':'YES' if expert_collapse else 'NO',
        'all_current_valid_actor_windows':len(all_rows),'supervised_actor_windows':len(routes),
        'full':54990,'partial':30037,'router_groups':router_rows,'expert_groups':expert_rows,
        'correlations':correlations,'collapse_population':'all current-valid actor-windows',
        'collapse_rules_sha256':sha256(PREREG),'expert_semantics_preassigned':False,'cosine_zero_vector_convention':0.}
    atomic_json(ROOT/'04_evaluation/stage5a_decoder_statistics.json',result)
    return result


def decision(boot,stats):
    get=lambda g:boot['groups'][g]['minFDE6']
    overall=get('overall');harm=any(get(g)['CI95'][0]>0 for g in ('vehicle','pedestrian'))
    important=read_json(PREREG)['scientific_rule']['important_motion_groups']
    gain=[g for g in important if get(g)['delta']<0 and get(g)['CI95'][1]<0]
    collapse=stats['router_collapse']=='YES' or stats['expert_functional_collapse']=='YES'
    supported=overall['delta']<0 and overall['CI95'][1]<0 and not harm and len(gain)>=1 and not collapse
    partial=overall['CI95'][0]<=0<=overall['CI95'][1] and not harm and len(gain)>=2 and not collapse
    science='SUPPORTED' if supported else ('PARTIAL' if partial else 'NOT SUPPORTED')
    result={'Motion_Aware_Decoder':science,'Stage5A':'PASS','Ready_Reliability':'YES' if supported or partial else 'NO',
        'reliable_main_class_harm':harm,'important_motion_groups_with_reliable_gain':gain,
        'router_collapse':stats['router_collapse'],'expert_functional_collapse':stats['expert_functional_collapse'],
        'overall':overall,'rules_sha256':sha256(PREREG),'Stage3B_conclusion':'SUPPORTED',
        'Stage4A_conclusion':'NOT SUPPORTED','Stage4F_conclusion':'NOT SUPPORTED','Stage4FD_conclusion':'MIXED',
        'additional_seeds_executed':False,'Reliability_executed':False,'STOP':True}
    atomic_json(ROOT/'04_evaluation/stage5a_scientific_decision.json',result)
    return result


def main():
    measured=fresh_val();b,e=paired();rows=tables(b,e);boot=bootstrap(b,e);stats=decoder_statistics(b)
    full=measured['metrics']['full_horizon']
    for group in GROUPS:
        row=next(r for r in rows if r['Group']==group)
        assert row['Count']==full[group]['count']
        assert all(abs(row['Stage5A_'+f]-full[group][f])<1e-10 for f in FIELDS)
    result=decision(boot,stats);verify_previous(shards=True)
    print('STAGE5A_EVALUATION_PASS',result,flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
