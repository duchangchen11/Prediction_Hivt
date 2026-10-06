"""Reproducible frozen-predictor batches, resume without overwriting valid cache."""
from pathlib import Path
import sys,random
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage6a_common import *
from torch_geometric.data import Batch
import numpy as np

@torch.no_grad()
def forward_batch(model,dataset,start):
    graphs=[dataset[i] for i in range(start,min(start+16,len(dataset)))]
    data=Batch.from_data_list(graphs).cuda();out=model(model_input(data));pred=model.ego_predictions(out,data)
    assert all(torch.isfinite(out[name]).all() for name in out)
    raw=out['raw_prediction'].cpu();logits=out['mode_logits'].cpu();prob=out['mode_prob'].cpu()
    rotation=out['rotation'].cpu();ego=pred.cpu();ptr=data.ptr.cpu().tolist();windows=[]
    for j,graph in enumerate(graphs):
        lo,hi=ptr[j:j+2]
        w={'dataset_index':start+j,'cache_batch_start':start,'cache_batch_size':len(graphs),
            'scene_name':graph.scene_name,'scene_token':graph.scene_token,'sample_token':graph.sample_token,
            'instance_tokens':list(graph.instance_tokens),'agent_type':graph.agent_type.clone(),
            'history':graph.positions[:,:5].clone(),'history_padding':graph.padding_mask[:,:5].clone(),
            'current_position':graph.positions[:,4].clone(),'GT':graph.positions[:,5:].clone(),
            'future_mask':graph.future_mask.clone(),'target_mask':graph.target_mask.clone(),
            'motion_state':list(graph.t0_motion_state),'rotate_angles':graph.rotate_angles.clone(),
            'raw_prediction':raw[:,lo:hi].clone(),'mode_logits':logits[lo:hi].clone(),
            'mode_prob':prob[lo:hi].clone(),'rotation':rotation[lo:hi].clone(),'ego_prediction':ego[lo:hi].clone()}
        assert w['raw_prediction'].shape==(6,graph.num_nodes,12,4)
        # Independent coordinate identity, evaluated using the exact frozen logic.
        explicit=torch.matmul(w['raw_prediction'][...,:2].permute(1,0,2,3),
            w['rotation'].transpose(-1,-2)[:,None])+w['current_position'][:,None,None]
        assert torch.allclose(explicit,w['ego_prediction'],atol=1e-5,rtol=1e-6)
        w['raw_prediction_sha256']=tensor_sha(w['raw_prediction']);w['ego_prediction_sha256']=tensor_sha(w['ego_prediction'])
        windows.append(w)
    return {'status':'PASS','split':dataset.split,'batch_start':start,'predictor_sha256':PREDICTOR_SHA,
            'windows':windows,'geometry_frame':'t0 ego x-forward/y-left, meters'}

def main():
    verify_frozen();model=frozen_predictor();initial=state_digest(model.state_dict())
    manifest_path=ROOT/'01_cache/stage6a_cache_manifest.json'
    prior=read_json(manifest_path) if manifest_path.exists() else {'batches':[]}
    done={r['path']:r for r in prior['batches']};records=[];counts={}
    for split in ('train','val'):
        ds=SceneDataset(split);full=partial=current=0;scene=set()
        for start in range(0,len(ds),16):
            path=cache_file(split,start);rel=str(path.relative_to(ROOT))
            if path.exists():
                assert rel in done and sha256(path)==done[rel]['sha256'],'Unregistered or modified cache: '+rel
                record=done[rel]
            else:
                payload=forward_batch(model,ds,start);atomic_torch(path,payload)
                record={'path':rel,'sha256':sha256(path),'bytes':path.stat().st_size,'split':split,'start':start,
                    'windows':len(payload['windows']),'full':0,'partial':0,'current_valid':0,'scene_tokens':[]}
                for w in payload['windows']:
                    full_mask=w['target_mask']&w['future_mask'].all(-1)
                    partial_mask=w['target_mask']&w['future_mask'].any(-1)&~w['future_mask'].all(-1)
                    record['full']+=int(full_mask.sum());record['partial']+=int(partial_mask.sum())
                    record['current_valid']+=int((~w['history_padding'][:,4]).sum())
                    record['scene_tokens'].append(w['scene_token'])
                record['scene_tokens']=sorted(set(record['scene_tokens']))
            records.append(record);full+=record['full'];partial+=record['partial'];current+=record['current_valid'];scene.update(record['scene_tokens'])
            atomic_json(manifest_path,{'status':'RUNNING','predictor_sha256':PREDICTOR_SHA,'batch_size':16,
                'batches':records,'counts':counts,'copied_original_data':False})
            if start%400==0 or start+16>=len(ds):
                print('CACHE',split,min(start+16,len(ds)),'/',len(ds),'full',full,flush=True)
        counts[split]={'supervised_windows':len(ds),'candidate_windows':len(ds.all_rows),'empty_windows':len(ds.empty_rows),
                       'full':full,'partial':partial,'current_valid':current,'scenes':len(scene)}
        if split=='val':assert (full,partial,current,len(scene))==(54990,30037,91092,150)
        else:assert len(scene)==700
        ds.clear()
    assert initial==state_digest(model.state_dict())
    assert not model.training and not any(p.requires_grad or p.grad is not None for p in model.parameters())
    atomic_json(manifest_path,{'status':'PASS','predictor_sha256':PREDICTOR_SHA,'predictor_state_sha256':initial,
        'predictor_eval':True,'predictor_requires_grad':False,'batch_size':16,'batches':records,'counts':counts,
        'total_cache_bytes':sum(r['bytes'] for r in records),'copied_original_data':False,
        'raw_shape':'[K,N,T,4]','ego_shape':'[N,K,T,2]','test_used':False,
        'official_VAL_cache_role':'frozen prediction generation only; not used for head tuning or selection'})
    verify_frozen();print('STAGE6A_CACHE_PASS',counts,flush=True)

@torch.no_grad()
def audit():
    manifest=read_json(ROOT/'01_cache/stage6a_cache_manifest.json');assert manifest['status']=='PASS'
    model=frozen_predictor();rng=random.Random(2022);samples=[];seen_instances=set();maximum={k:0. for k in ('raw_prediction','mode_logits','mode_prob','ego_prediction','rotation')}
    for split in ('train','val'):
        ds=SceneDataset(split);rows=[r for r in manifest['batches'] if r['split']==split]
        rng.shuffle(rows)
        target_count=60
        for record in rows:
            cached=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)
            replay=forward_batch(model,ds,record['start'])
            candidates=[(j,int(i)) for j,w in enumerate(cached['windows']) for i in torch.where(~w['history_padding'][:,4])[0]
                        if w['instance_tokens'][int(i)] not in seen_instances]
            rng.shuffle(candidates)
            selected=[]
            for j,i in candidates:
                old=cached['windows'][j];new=replay['windows'][j];instance=old['instance_tokens'][i]
                if instance in seen_instances:continue
                assert (old['scene_token'],old['sample_token'],old['instance_tokens'])==(new['scene_token'],new['sample_token'],new['instance_tokens'])
                assert old['raw_prediction_sha256']==tensor_sha(old['raw_prediction'])
                for name in maximum:
                    a=old[name][:,i] if name=='raw_prediction' else old[name][i]
                    b=new[name][:,i] if name=='raw_prediction' else new[name][i]
                    difference=float((a-b).abs().max());maximum[name]=max(maximum[name],difference)
                samples.append({'split':split,'scene_token':old['scene_token'],'sample_token':old['sample_token'],
                    'instance_token':instance,'node':i,'cache_path':record['path']})
                selected.append(i);seen_instances.add(instance)
                if len(selected)>=12 or sum(r['split']==split for r in samples)>=target_count:break
            if sum(r['split']==split for r in samples)>=target_count:break
        ds.clear()
    assert len(samples)>=120 and len(seen_instances)>=100
    assert all(v<1e-6 for v in maximum.values()),maximum
    verify_frozen(shards=True)
    atomic_json(ROOT/'01_cache/stage6a_cache_audit.json',{'status':'PASS','seed':2022,'actor_windows':len(samples),
        'unique_instances':len(seen_instances),'max_abs_difference':maximum,'tolerance':1e-6,'identity_equal':True,
        'predictor_eval':not model.training,'predictor_all_parameters_frozen':True,'samples':samples,
        'geometry_conversion':'existing predictor.ego_predictions: local row-vector @ rotation transpose + current t0 position'})
    print('STAGE6A_CACHE_AUDIT_PASS',maximum,flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    if '--audit' in sys.argv:audit()
    else:main()
