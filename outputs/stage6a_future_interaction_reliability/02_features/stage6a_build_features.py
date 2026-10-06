"""TRAIN-only feature normalization and full-horizon head supervision."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage6a_common import *
from stage6a_features import observable_features,normalize,FEATURE_NAMES

@torch.no_grad()
def main():
    verify_frozen();cache=read_json(ROOT/'01_cache/stage6a_cache_manifest.json')
    assert cache['status']=='PASS' and read_json(ROOT/'01_cache/stage6a_cache_audit.json')['status']=='PASS'
    split=read_json(SPLIT);dev=set(split['HeadDev']);train=set(split['HeadTrain']);assert not train&dev
    blocks=[];flag_blocks=[];fde_blocks=[];ade_blocks=[];logit_blocks=[];rows=[]
    files=[r for r in cache['batches'] if r['split']=='train']
    for fi,record in enumerate(files):
        path=ROOT/record['path'];assert sha256(path)==record['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=False)
        for w in payload['windows']:
            feature,flags,_,_=observable_features(*[w[n].cuda() for n in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
            selected=torch.where(w['target_mask']&w['future_mask'].all(-1))[0]
            assert (~w['history_padding'][selected,4]).all()
            if not len(selected):continue
            blocks.append(feature[selected].cpu());flag_blocks.append(flags[selected].cpu());logit_blocks.append(w['mode_logits'][selected])
            distances=(w['ego_prediction'][selected]-w['GT'][selected,None]).norm(dim=-1)
            fde_blocks.append(distances[:,:,-1]);ade_blocks.append(distances.mean(-1))
            for node in selected.tolist():
                rows.append({'scene_token':w['scene_token'],'scene_name':w['scene_name'],'sample_token':w['sample_token'],
                    'instance_token':w['instance_tokens'][node],'node_in_graph':node,'horizon':'full_horizon',
                    'agent_type':CLASSES[int(w['agent_type'][node])],'motion_state':w['motion_state'][node],
                    'cache_path':record['path'],'dataset_index':w['dataset_index'],
                    **GT_fingerprint(w['GT'][node],w['future_mask'][node],w['agent_type'][node])})
        if (fi+1)%50==0 or fi+1==len(files):print('FEATURES_TRAIN',fi+1,'/',len(files),'actors',len(rows),flush=True)
    features=torch.cat(blocks);flags=torch.cat(flag_blocks);fde=torch.cat(fde_blocks);ade=torch.cat(ade_blocks);logits=torch.cat(logit_blocks)
    train_mask=torch.tensor([r['scene_token'] in train for r in rows]);dev_mask=~train_mask
    assert all(r['scene_token'] in train|dev for r in rows) and len(rows)==cache['counts']['train']['full']
    mean=torch.zeros(19,dtype=torch.float64);std=mean.clone();statistics_count=[]
    for column in range(19):
        mask=train_mask&(flags[:,0] if 12<=column<15 else torch.ones_like(train_mask))
        values=features[mask,:,column].double().flatten();assert len(values)
        mean[column]=values.mean();std[column]=values.std(unbiased=False);statistics_count.append(len(values))
    statistics={'status':'PASS','mean':mean.tolist(),'std':std.tolist(),'feature_names':list(FEATURE_NAMES),
        'per_column_observations':statistics_count,'epsilon':1e-6,'type_one_hot_standardized':False,
        'continuous_standardized_columns':list(range(3,19)),'base_probability_standardized':True,
        'split_sha256':sha256(SPLIT),'config_sha256':sha256(CONFIG),'source_scenes':'HeadTrain630 only',
        'TRAIN630_actor_count':int(train_mask.sum()),'HeadDev_actor_count':int(dev_mask.sum()),'VAL_used':False,
        'missing_neighbor':{'distance_columns':[12,13,14],'normalized_fill':1.,'conflict_fill':0.,
            'absent_vehicle_column17':0.,'absent_pedestrian_column18':0.}}
    atomic_json(NORM,statistics)
    normalized=normalize(features,flags,statistics)
    assert not normalized.requires_grad and not logits.requires_grad
    partitions={}
    for name,mask in (('headtrain',train_mask),('headdev',dev_mask)):
        indices=torch.where(mask)[0].tolist();rr=[rows[i] for i in indices]
        payload={'status':'PASS','partition':name,'features_R2':normalized[mask].clone(),'flags':flags[mask].clone(),
            'base_logits':logits[mask].clone(),'FDE_by_mode':fde[mask].clone(),'ADE_by_mode':ade[mask].clone(),
            'rows':rr,'normalization_sha256':sha256(NORM),'split_sha256':sha256(SPLIT),'full_horizon_only':True}
        path=ROOT/'02_features'/f'stage6a_{name}_features.pt';atomic_torch(path,payload)
        partitions[name]={'actors':len(rr),'scenes':len({r['scene_token'] for r in rr}),
            'class_counts':{c:sum(r['agent_type']==c for r in rr) for c in CLASSES},'path':str(path.relative_to(ROOT)),
            'sha256':sha256(path),'bytes':path.stat().st_size}
    assert partitions['headtrain']['scenes']==630 and partitions['headdev']['scenes']==70
    atomic_json(ROOT/'02_features/stage6a_feature_manifest.json',{'status':'PASS','feature_count':19,'partitions':partitions,
        'normalization_sha256':sha256(NORM),'split_sha256':sha256(SPLIT),'config_sha256':sha256(CONFIG),
        'cache_manifest_sha256':sha256(ROOT/'01_cache/stage6a_cache_manifest.json'),'VAL_used_for_normalization':False,
        'R1_only_difference':'last7 normalized input columns set exactly0','predictor_requires_grad':False,
        'feature_inputs':'history/types/frozen mode logits/probabilities/ego-frame predictions only; GT used solely for FDE labels',
        'supervised_target_temperature_m':1.,'target_formula':'softmax(-full-horizon FDE_by_mode /1m)'})
    verify_frozen();print('STAGE6A_FEATURES_PASS',partitions,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
