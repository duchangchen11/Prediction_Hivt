"""One locked official VAL pass: shared immutable geometry, three rankings."""
from pathlib import Path
import sys,csv,os
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'01_cache')]
from stage6a_common import *
from stage6a_cache import forward_batch
from stage6a_features import observable_features,normalize,FEATURE_NAMES
from stage6a_head import ReliabilityHead
import numpy as np

RANK_METRICS=('Top1ADE','Top1FDE','OracleGap_FDE','Top1HitRate','BestModeRank','MRR_best_mode')
GEOMETRY=('minADE6','minFDE6','MR6')
INTERACTION=('Heterogeneous-20m','Vehicle hetero-20m','Pedestrian hetero-20m','VP-context-20m')
GROUPS=('overall','vehicle','pedestrian','bicycle','vehicle.moving','vehicle.stopped','vehicle.parked','unknown',
    'Vehicle >5m','Pedestrian <5m','Pedestrian >5m','Pedestrian 5-10m')+INTERACTION
ACTORS=ROOT/'04_evaluation/stage6a_actor_ranking.csv'

def load_head(variant,device='cuda'):
    summary=read_json(ROOT/'03_training'/f'stage6a_{variant.lower()}_summary.json')
    assert summary['status']=='COMPLETE' and sha256(head_path(variant))==summary['checkpoint_sha256']
    saved=torch.load(head_path(variant),map_location='cpu',weights_only=False)
    for name,path in (('config_sha256',CONFIG),('normalization_sha256',NORM),('split_sha256',SPLIT)):
        assert saved['metadata'][name]==sha256(path)
    model=ReliabilityHead().to(device);model.load_state_dict(saved['state_dict'],strict=True)
    model.eval();model.requires_grad_(False);return model,summary

def membership():
    audit=read_json(MEMBERSHIP_AUDIT);assert audit['membership_sha256']==sha256(MEMBERSHIP)
    with MEMBERSHIP.open() as f:rows=list(csv.DictReader(f))
    result={actor_key(r):r for r in rows};assert len(result)==len(rows)==85027
    return result

def selected(rows,group,members):
    out=[]
    for row in rows:
        if row['horizon']!='full_horizon':continue
        cls=row['agent_type'];d=float(row['GT_endpoint_displacement_m'])
        include=group=='overall' or group==cls or (cls=='vehicle' and group==row['motion_state'])
        if group=='Vehicle >5m':include=cls=='vehicle' and d>5
        if group=='Pedestrian <5m':include=cls=='pedestrian' and d<5
        if group=='Pedestrian >5m':include=cls=='pedestrian' and d>5
        if group=='Pedestrian 5-10m':include=cls=='pedestrian' and 5<=d<10
        if group in INTERACTION:include=members[actor_key(row)][group]=='1'
        if include:out.append(row)
    assert out,group;return out

@torch.no_grad()
def ranking_arrays(w,probabilities):
    prediction=w['ego_prediction'].cuda();gt=w['GT'].cuda();mask=w['future_mask'].cuda()
    valid=mask.sum(-1);last=torch.where(mask,torch.arange(12,device='cuda'),-1).max(-1).values
    distances=(prediction-gt[:,None]).norm(dim=-1)*mask[:,None]
    fde=distances.gather(2,last.clamp(min=0)[:,None,None].expand(-1,6,1)).squeeze(-1)
    ade=distances.sum(-1)/valid.clamp(min=1)[:,None]
    best=fde.argmin(-1);idx=torch.arange(len(best),device='cuda')
    min_fde=fde[idx,best];min_ade=ade[idx,best]
    result={'geometry':torch.stack((min_ade,min_fde,(min_fde>2).float()),-1).cpu(),
        'FDE_by_mode':fde.cpu(),'ADE_by_mode':ade.cpu(),'best_mode':best.cpu(),'valid_steps':valid.cpu(),
        'last_valid_index':last.cpu(),'rankings':{}}
    for variant,prob in probabilities.items():
        p=prob.cuda();top=p.argmax(-1);order=p.argsort(dim=-1,descending=True,stable=True)
        rank=(order==best[:,None]).long().argmax(-1)+1
        values=torch.stack((ade[idx,top],fde[idx,top],fde[idx,top]-min_fde,
            (top==best).float(),rank.float(),1./rank.float()),-1)
        assert torch.isfinite(values).all()
        result['rankings'][variant]={'mode_prob':prob,'top1_mode':top.cpu(),'values':values.cpu()}
    return result

def metric_rows(w,result):
    rows=[]
    for node in torch.where(w['target_mask']&w['future_mask'].any(-1))[0].tolist():
        horizon='full_horizon' if w['future_mask'][node].all() else 'partial_future'
        last=int(result['last_valid_index'][node]);gt_delta=w['GT'][node,last]-w['current_position'][node]
        row={'scene_name':w['scene_name'],'scene_token':w['scene_token'],'sample_token':w['sample_token'],
            'instance_token':w['instance_tokens'][node],'node_in_graph':node,'horizon':horizon,
            'agent_type':CLASSES[int(w['agent_type'][node])],'motion_state':w['motion_state'][node],
            'valid_future_steps':int(result['valid_steps'][node]),'GT_endpoint_displacement_m':float(gt_delta.norm()),
            'best_mode':int(result['best_mode'][node]),**GT_fingerprint(w['GT'][node],w['future_mask'][node],w['agent_type'][node]),
            **dict(zip(GEOMETRY,map(float,result['geometry'][node])))}
        for variant in ('R0','R1','R2'):
            row.update({variant+'_'+name:float(value) for name,value in zip(RANK_METRICS,result['rankings'][variant]['values'][node])})
            row[variant+'_top1_mode']=int(result['rankings'][variant]['top1_mode'][node])
        rows.append(row)
    return rows

@torch.no_grad()
def fresh_val():
    assert read_json(ROOT/'03_training/stage6a_training_summary.json')['status']=='COMPLETE'
    registration=read_json(ROOT/'00_manifest/stage6a_training_registration.json')
    for name,digest in registration['source_sha256'].items():assert sha256(ROOT/name)==digest,name
    verify_frozen();heads={v:load_head(v)[0] for v in ('R1','R2')};model=frozen_predictor()
    initial=state_digest(model.state_dict());norm=read_json(NORM);ds=SceneDataset('val')
    official=ROOT/'00_manifest/stage6a_official_val_registration.json'
    specification={'status':'LOCKED_BEFORE_FINAL_OFFICIAL_VAL','predictor_sha256':PREDICTOR_SHA,
        'R1_checkpoint_sha256':sha256(head_path('R1')),'R2_checkpoint_sha256':sha256(head_path('R2')),
        'normalization_sha256':sha256(NORM),'config_sha256':sha256(CONFIG),'split_sha256':sha256(SPLIT),
        'evaluation_code_sha256':sha256(Path(__file__)),'official_VAL_passes':1,'official_VAL_checkpoint_selection':False,
        'cache_determinism':'torch deterministic algorithms, identical original16-window batches',
        'ranking_ties':'smallest mode index first; stable descending probability order'}
    if official.exists():assert read_json(official)==specification,'Official evaluation specification changed'
    else:atomic_json(official,specification)
    assert not (ROOT/'04_evaluation/stage6a_evaluation_complete.json').exists(),'Final official VAL already complete; do not re-evaluate'
    cache=read_json(ROOT/'01_cache/stage6a_cache_manifest.json');cache_by_start={r['start']:r for r in cache['batches'] if r['split']=='val'}
    rows=[];feature_blocks=[];raw_blocks=[];flag_blocks=[];mode_fde=[];mode_ade=[];logit_blocks=[];full_keys=[];geometry_sha=[]
    for start in range(0,len(ds),16):
        path=ROOT/'01_cache/val_ranking'/f'stage6a_val_ranking_batch_{start:05d}.pt'
        if path.exists():
            payload=torch.load(path,map_location='cpu',weights_only=False)
            assert payload['specification_sha256']==sha256(official)
        else:
            live=forward_batch(model,ds,start);cached=torch.load(ROOT/cache_by_start[start]['path'],map_location='cpu',weights_only=False)
            results=[]
            for w,old in zip(live['windows'],cached['windows']):
                assert w['raw_prediction_sha256']==old['raw_prediction_sha256'] and w['ego_prediction_sha256']==old['ego_prediction_sha256'],'Geometry changed from frozen deterministic cache'
                assert w['instance_tokens']==old['instance_tokens'] and torch.equal(w['future_mask'],old['future_mask'])
                assert torch.equal(w['mode_logits'],old['mode_logits']) and torch.equal(w['mode_prob'],old['mode_prob'])
                x,flags,_,_=observable_features(*[w[n].cuda() for n in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
                r2x=normalize(x,flags,norm,'R2');r1x=normalize(x,flags,norm,'R1')
                assert (r1x[:,:,12:]==0).all() and torch.equal(r1x[:,:,:12],r2x[:,:,:12])
                probs={'R0':w['mode_prob']}
                for variant,xx in (('R1',r1x),('R2',r2x)):
                    out=heads[variant](xx,w['mode_logits'].cuda());probs[variant]=out['mode_prob'].cpu()
                result=ranking_arrays(w,probs);selected=torch.where(w['target_mask']&w['future_mask'].all(-1))[0]
                result.update(scene_token=w['scene_token'],sample_token=w['sample_token'],instance_tokens=w['instance_tokens'],
                    dataset_index=w['dataset_index'],selected_nodes=selected,raw_prediction_sha256=w['raw_prediction_sha256'],
                    ego_prediction_sha256=w['ego_prediction_sha256'],features_R2=r2x[selected].cpu(),raw_features=x[selected].cpu(),
                    flags=flags[selected].cpu(),base_logits=w['mode_logits'][selected],actor_rows=metric_rows(w,result))
                results.append(result)
            payload={'status':'PASS','specification_sha256':sha256(official),'batch_start':start,'windows':results}
            atomic_torch(path,payload)
        for result in payload['windows']:
            rows.extend(result['actor_rows']);sel=result['selected_nodes']
            feature_blocks.append(result['features_R2']);raw_blocks.append(result['raw_features']);flag_blocks.append(result['flags'])
            mode_fde.append(result['FDE_by_mode'][sel]);mode_ade.append(result['ADE_by_mode'][sel]);logit_blocks.append(result['base_logits'])
            full_keys.extend([actor_key(r) for r in result['actor_rows'] if r['horizon']=='full_horizon'])
            geometry_sha.append({'scene_token':result['scene_token'],'sample_token':result['sample_token'],
                'raw_prediction_sha256_R0_R1_R2':result['raw_prediction_sha256'],
                'ego_prediction_sha256_R0_R1_R2':result['ego_prediction_sha256']})
        if start%400==0 or start+16>=len(ds):print('FINAL_VAL',min(start+16,len(ds)),'/',len(ds),flush=True)
    assert len(rows)==85027 and len({actor_key(r) for r in rows})==85027
    assert sum(r['horizon']=='full_horizon' for r in rows)==54990
    assert sum(r['horizon']=='partial_future' for r in rows)==30037
    write_csv(ACTORS,rows)
    atomic_torch(ROOT/'02_features/stage6a_val_features.pt',{'status':'PASS','features_R2':torch.cat(feature_blocks),
        'raw_features':torch.cat(raw_blocks),'flags':torch.cat(flag_blocks),'FDE_by_mode':torch.cat(mode_fde),
        'ADE_by_mode':torch.cat(mode_ade),'base_logits':torch.cat(logit_blocks),'actor_keys':full_keys,
        'normalization_sha256':sha256(NORM),'official_VAL_used_for_training':False})
    assert initial==state_digest(model.state_dict()) and not model.training
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    atomic_json(ROOT/'04_evaluation/stage6a_geometry_identity_audit.json',{'status':'PASS','windows':3603,'VAL_scenes':150,
        'full':54990,'partial':30037,'total':85027,'raw_and_ego_prediction_bitwise_identical':True,
        'shared_geometry_computed_once':True,'R1_R2_return_geometry':False,
        'raw_prediction_SHA_R0_R1_R2_equal_all_windows':True,'deterministic_cache_SHA_equal_fresh_VAL':True,
        'minADE6_max_R0_R1_R2_diff':0.,'minFDE6_max_R0_R1_R2_diff':0.,'MR6_max_R0_R1_R2_diff':0.,
        'tolerance':1e-8,'predictor_state_SHA_unchanged':initial,'predictor_gradient_tensors':0,
        'per_window_geometry_SHA':geometry_sha})
    ds.clear();del model,heads;torch.cuda.empty_cache();return rows

def pair(rows,members):
    with GT_LEDGER.open() as f:ledger={actor_key(r):r for r in csv.DictReader(f)}
    with BASE_ACTORS.open() as f:b={actor_key(r):r for r in csv.DictReader(f)}
    with STAGE5_ACTORS.open() as f:e={actor_key(r):r for r in csv.DictReader(f)}
    current={actor_key(r):r for r in rows};assert ledger.keys()==b.keys()==e.keys()==current.keys()==members.keys()
    identity=('scene_token','sample_token','instance_token','node_in_graph','horizon','agent_type','motion_state',
        'valid_future_steps','GT_trajectory_sha256','future_mask_bits','agent_type_id')
    old_metric_difference={name:0. for name in GEOMETRY+('Top1ADE6','Top1FDE6')}
    for key,row in current.items():
        assert all(str(row[name])==ledger[key][name]==b[key][name]==e[key][name] for name in identity),(key,'identity mismatch')
        assert str(row['node_in_graph'])==members[key]['node_in_graph']
        assert row['agent_type']==members[key]['agent_type']
    # Numerical reconciliation to Stage5A's earlier nondeterministic CUDA run is
    # kept separate from exact geometry identity among current R0/R1/R2.
    group_reconciliation=[]
    for group in ('overall','vehicle','pedestrian','bicycle','vehicle.moving','vehicle.stopped','vehicle.parked','unknown'):
        chosen=selected(rows,group,members)
        for name in GEOMETRY+('Top1ADE6','Top1FDE6'):
            newname=name if name in GEOMETRY else ('R0_Top1ADE' if name=='Top1ADE6' else 'R0_Top1FDE')
            difference=abs(np.mean([float(r[newname]) for r in chosen])-np.mean([float(e[actor_key(r)][name]) for r in chosen]))
            old_metric_difference[name]=max(old_metric_difference[name],float(difference))
            assert difference<1e-6,(group,name,difference)
            group_reconciliation.append({'Group':group,'Metric':name,'historical_Stage5A_mean_difference':float(difference)})
    atomic_json(ROOT/'04_evaluation/stage6a_pairing_audit.json',{'status':'PASS','full':54990,'partial':30037,'total':85027,
        'identity_fields':identity,'frozen_interaction_membership_sha256':sha256(MEMBERSHIP),
        'frozen_GT_ledger_sha256':sha256(GT_LEDGER),'historical_Stage5A_numerical_reconciliation_tolerance':1e-6,
        'historical_reconciliation':group_reconciliation,'historical_max_mean_differences':old_metric_difference,
        'current_R0_R1_R2_geometry_identity_tolerance':1e-8,'current_geometry_max_difference':0.})
    return b,e

def tables(rows,members,b):
    main=[];long=[]
    for group in GROUPS:
        rr=selected(rows,group,members);keys=[actor_key(r) for r in rr]
        row={'Group':group,'Count':len(rr),'Scenes':len({r['scene_token'] for r in rr}),
            'Instances':len({r['instance_token'] for r in rr}),'Stage3B_Top1FDE':float(np.mean([float(b[k]['Top1FDE6']) for k in keys]))}
        geometry={name:float(np.mean([r[name] for r in rr])) for name in GEOMETRY}
        for variant in ('R0','R1','R2'):
            metrics={name:float(np.mean([r[variant+'_'+name] for r in rr])) for name in RANK_METRICS}
            row.update({variant+'_'+name:v for name,v in {**metrics,**geometry}.items()})
            long.append({'Group':group,'Variant':variant,'Count':len(rr),'Scenes':row['Scenes'],'Instances':row['Instances'],**geometry,**metrics})
        row['Stage5A_Top1FDE']=row['R0_Top1FDE'];row['Stage5A_minFDE']=row['R0_minFDE6']
        row['R1_minFDE']=row['R1_minFDE6'];row['R2_minFDE']=row['R2_minFDE6'];main.append(row)
    write_csv(ROOT/'06_tables/stage6a_main_ranking_results.csv',main)
    write_csv(ROOT/'06_tables/stage6a_reliability_ablation.csv',long)
    return main,long

def bootstrap(rows,members):
    scenes=sorted({r['scene_token'] for r in rows});assert len(scenes)==150
    lookup={s:i for i,s in enumerate(scenes)};rng=np.random.default_rng(2022)
    weights=np.stack([np.bincount(draw,minlength=150) for draw in rng.integers(0,150,(1000,150))])
    metrics=('Top1FDE','Top1ADE','Top1HitRate','OracleGap_FDE');comparisons=(('R1','R0'),('R2','R0'),('R2','R1'))
    result={'status':'PASS','seed':2022,'replicates':1000,'scene_clusters':150,'weighting':'actor-window pooled after resampling whole paired scenes',
        'confidence':.95,'method':'paired scene-cluster percentile bootstrap','comparisons':{}}
    table=[]
    for new,old in comparisons:
        label=new+'-'+old;result['comparisons'][label]={}
        for group in GROUPS:
            rr=selected(rows,group,members);indices=np.array([lookup[r['scene_token']] for r in rr])
            counts=np.bincount(indices,minlength=150);denominator=weights@counts;assert (denominator>0).all()
            result['comparisons'][label][group]={}
            for metric in metrics:
                delta=np.array([r[new+'_'+metric]-r[old+'_'+metric] for r in rr])
                sums=np.bincount(indices,weights=delta,minlength=150);replicates=(weights@sums)/denominator
                lo,hi=map(float,np.quantile(replicates,[.025,.975]))
                item={'delta':float(delta.mean()),'CI95':[lo,hi],'Count':len(rr),'Scenes':int((counts>0).sum())}
                result['comparisons'][label][group][metric]=item
                table.append({'Comparison':label,'Group':group,'Metric':metric,'Delta':item['delta'],
                    'CI95_lower':lo,'CI95_upper':hi,'Count':len(rr),'Scenes':item['Scenes'],'Replicates':1000})
    atomic_json(ROOT/'04_evaluation/stage6a_bootstrap_ci.json',result)
    write_csv(ROOT/'06_tables/stage6a_bootstrap_ci.csv',table);return result

def changes(rows,members):
    table=[]
    for new,old in (('R1','R0'),('R2','R0'),('R2','R1')):
        for group in GROUPS:
            rr=selected(rows,group,members);delta=np.array([r[new+'_Top1FDE']-r[old+'_Top1FDE'] for r in rr])
            changed=np.array([r[new+'_top1_mode']!=r[old+'_top1_mode'] for r in rr]);improved=changed&(delta<0);worsened=changed&(delta>0)
            equal=changed&(delta==0);assert int(changed.sum())==int(improved.sum()+worsened.sum()+equal.sum())
            table.append({'Comparison':new+'-'+old,'Group':group,'Count':len(rr),'Number_changed':int(changed.sum()),
                'Number_improved':int(improved.sum()),'Number_worsened':int(worsened.sum()),'Number_changed_equal':int(equal.sum()),
                'Top1_changed_rate':float(changed.mean()),'Changed_and_improved_rate_all_actors':float(improved.mean()),
                'Changed_and_worsened_rate_all_actors':float(worsened.mean()),
                'Improved_rate_among_changed':float(improved.sum()/changed.sum()) if changed.any() else 0.,
                'Worsened_rate_among_changed':float(worsened.sum()/changed.sum()) if changed.any() else 0.,
                'Mean_improvement_when_improved_m':float(-delta[improved].mean()) if improved.any() else 0.,
                'Mean_degradation_when_worsened_m':float(delta[worsened].mean()) if worsened.any() else 0.,
                'Net_Top1FDE_delta':float(delta.mean())})
    write_csv(ROOT/'06_tables/stage6a_net_reranking_benefit.csv',table)
    atomic_json(ROOT/'04_evaluation/stage6a_probability_change_audit.json',{'status':'PASS','rows':table,
        'definition':'changed=argmax probability mode differs; improved/worsened compare new versus old Top1 FDE strictly',
        'equal_quality_changed_modes_reported_separately':True})

def feature_distribution(members):
    payload=load_features('val');keys=[tuple(k) for k in payload['actor_keys']]
    with ACTORS.open() as f:rr={actor_key(r):r for r in csv.DictReader(f)}
    values=payload['raw_features'].double().numpy();flags=payload['flags'].numpy();table=[]
    for group in ('overall','vehicle','pedestrian','vehicle.moving','Heterogeneous-20m','VP-context-20m'):
        chosen={actor_key(r) for r in selected(list(rr.values()),group,members)}
        mask=np.array([k in chosen for k in keys])
        for column in range(12,19):
            valid=mask&flags[:,0] if column<15 else mask
            for mode in range(6):
                x=values[valid,mode,column];assert len(x)
                table.append({'Group':group,'Feature':FEATURE_NAMES[column],'Mode':mode,'Count':len(x),
                    'Mean':float(x.mean()),'Std':float(x.std()),'Q05':float(np.quantile(x,.05)),
                    'Median':float(np.median(x)),'Q95':float(np.quantile(x,.95)),
                    'No_neighbor_distance_sentinel_excluded':column<15})
    write_csv(ROOT/'06_tables/stage6a_interaction_feature_distributions.csv',table)

def interpret(main,boot):
    lookup={r['Group']:r for r in main};overall=lookup['overall'];safe={};usable={};evidence={}
    for variant in ('R1','R2'):
        comp=boot['comparisons'][variant+'-R0']
        severe=[]
        for group in ('vehicle','pedestrian'):
            relative=(lookup[group][variant+'_Top1FDE']/lookup[group]['R0_Top1FDE']-1)
            if relative>=.10 and comp[group]['Top1FDE']['CI95'][0]>0:severe.append(group)
        safe[variant]=not severe
        benefit=overall[variant+'_Top1FDE']<overall['R0_Top1FDE'] and overall[variant+'_OracleGap_FDE']<overall['R0_OracleGap_FDE'] and overall[variant+'_Top1HitRate']>overall['R0_Top1HitRate'] and safe[variant]
        usable[variant]=benefit
        interval=comp['overall']['Top1FDE']['CI95']
        evidence[variant]='SUPPORTED' if benefit and interval[1]<0 else ('PROMISING' if benefit and interval[0]<=0<=interval[1] else 'NOT SUPPORTED')
    choices=['R0']+[v for v in ('R1','R2') if safe[v]]
    recommended=min(choices,key=lambda v:(overall[v+'_Top1FDE'],int(v[-1])))
    candidate=min(('R1','R2'),key=lambda v:(overall[v+'_Top1FDE'],int(v[-1])))
    reliability=evidence[candidate] if recommended!='R0' else 'NOT SUPPORTED'
    comparison=boot['comparisons']['R2-R1'];primary=comparison['overall']['Top1FDE']
    context_better=[g for g in ('Heterogeneous-20m','VP-context-20m') if comparison[g]['Top1FDE']['delta']<0]
    if not safe['R2'] or primary['CI95'][0]>0:contribution='NOT SUPPORTED'
    elif primary['CI95'][1]<0 and context_better:contribution='SUPPORTED'
    else:contribution='WEAK'
    result={'status':'PASS','ReliabilityHead':reliability,'FutureInteractionContribution':contribution,
        'PaperUsableReliability':'YES' if recommended!='R0' and usable[recommended] else 'NO',
        'RecommendedFinalVariant':recommended,'head_evidence':evidence,'head_paper_usable':usable,
        'no_severe_major_class_harm':safe,'severe_harm_guard':'relative Top1FDE increase >=10% and paired CI lower>0',
        'R2_R1_overall':primary,'R2_better_frozen_interaction_groups':context_better,
        'variant_choice':'descriptive comparison among two prespecified fixed HeadDev-selected checkpoints on final VAL',
        'Stage5A_prior_science':'NOT SUPPORTED; unchanged','predictor_geometry':'PASS','official_VAL_checkpoint_selection':False,
        'additional_heads':False,'joint_finetuning':False,'STOP':True}
    atomic_json(ROOT/'04_evaluation/stage6a_scientific_interpretation.json',result);return result

def main():
    rows=fresh_val();members=membership();b,e=pair(rows,members);main_table,_=tables(rows,members,b)
    boot=bootstrap(rows,members);changes(rows,members);feature_distribution(members);result=interpret(main_table,boot)
    assert len(selected(rows,'Heterogeneous-20m',members))==25113
    assert len(selected(rows,'VP-context-20m',members))==23209
    verify_frozen(shards=True)
    atomic_json(ROOT/'04_evaluation/stage6a_evaluation_complete.json',{'status':'PASS','official_VAL_final_ranking_passes':1,
        'supervised_windows':3603,'scenes':150,'full':54990,'partial':30037,'total':85027,
        'actor_csv_sha256':sha256(ACTORS),'NaN':0,'Inf':0,'interpretation':result})
    print('STAGE6A_FINAL_EVALUATION_PASS',result,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
