"""Past-only extraction with original coordinates and separate future labels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage13a_common import *

def extract_observation(db,samples,t0,future_times):
    assert t0>=4 and np.asarray(future_times).shape==(12,) and (np.diff(future_times)>0).all() and np.min(future_times)>0
    past=samples[t0-4:t0+1];limited=PastOnly(db,past);mapping=category_mapping(limited);anns=[annotation_index(limited,s) for s in past]
    tokens=sorted(t for t,a in anns[-1].items() if a['category_name'] in mapping);n=len(tokens)
    pos=np.zeros((n,5,2),np.float64);head=np.zeros((n,5),np.float64);mask=np.zeros((n,5),bool)
    types=np.array([mapping[anns[-1][t]['category_name']] for t in tokens],np.int64);sizes=np.zeros((n,2),np.float64);source=np.empty(n,dtype='U32');extras,_=original_extras()
    prior=read_json(PROTOCOL)['Collision']['MissingActorLengthWidth']
    for i,token in enumerate(tokens):
        for j,rows in enumerate(anns):
            if token in rows:pos[i,j]=rows[token]['translation'][:2];head[i,j]=quaternion_yaw(rows[token]['rotation']);mask[i,j]=True
        current=anns[-1][token];original=extras.get(current['token'])
        if original is not None and 'size' in original and np.all(np.array(original['size'])>0):
            assert np.array_equal(original['translation'],current['translation']) and original['sample_token']==samples[t0]['token'];sizes[i]=[original['size'][1],original['size'][0]];source[i]='current_annotation'
        else:sizes[i]=prior[TYPES[types[i]]];source[i]='class_prior_missing_source'
    poses=[];sd_times=[];pose_tokens=[];lidar_tokens=[]
    for sample in past:
        sd=limited.get('sample_data',sample['data']['LIDAR_TOP']);assert sd['is_key_frame'] and sd['sample_token']==sample['token']
        pose=limited.get('ego_pose',sd['ego_pose_token']);poses.append(pose);sd_times.append(sd['timestamp']);pose_tokens.append(pose['token']);lidar_tokens.append(sd['token'])
    ego_global=np.array([p['translation'][:2] for p in poses],np.float64);yaw=np.array([quaternion_yaw(p['rotation']) for p in poses]);origin=ego_global[-1].copy();angle=float(yaw[-1])
    local=np.zeros_like(pos);local[mask]=global_to_ego(pos[mask],origin,angle);local_head=np.zeros_like(head);local_head[mask]=global_heading_to_ego(head[mask],angle)
    stamps=np.array([s['timestamp'] for s in past],np.int64);ht=(stamps-stamps[-1])/1e6;assert (np.diff(ht)>0).all()
    ego=global_to_ego(ego_global,origin,angle);ego_head=global_heading_to_ego(yaw,angle)
    roundtrip=max(float(np.abs(ego_to_global(ego,origin,angle)-ego_global).max()),float(np.abs(ego_to_global(local[mask],origin,angle)-pos[mask]).max()) if mask.any() else 0.)
    assert roundtrip<1e-5 and np.allclose(ego[-1],0,atol=1e-12) and abs(float(ego_head[-1]))<1e-12
    result=dict(scene_token=db.scene[0]['token'],sample_token=samples[t0]['token'],t0_index=t0,t0_timestamp=int(stamps[-1]),history_times=ht,
        ego_history=ego,ego_history_heading=ego_head,history=local,history_heading=local_head,history_mask=mask,agent_type=types,instance_tokens=tuple(tokens),
        actor_size_length_width=sizes,actor_size_source=source,origin=origin,yaw=angle,location=db.get('log',db.scene[0]['log_token'])['location'],
        future_times=np.asarray(future_times,np.float64),prediction_valid_mask=(mask.sum(-1)>=2)&mask[:,-1],
        current_position=local[:,-1].copy(),current_heading=local_head[:,-1].copy(),history_sample_timestamps_us=stamps,
        history_lidar_timestamps_us=np.array(sd_times,np.int64),history_pose_tokens=tuple(pose_tokens),history_lidar_tokens=tuple(lidar_tokens),
        coordinate_roundtrip_max_error_m=roundtrip,past_metadata_reads=tuple(limited.reads),coordinate_frame='t0 ego planar: +x forward, +y left')
    assert not any(k in result for k in ['future_mask','target_mask','full_horizon_mask','GT','ego_future_gt']);return result

def extract_evaluation(db,samples,t0,obs):
    n=len(obs['instance_tokens']);ego=np.full((12,2),np.nan);eh=np.full(12,np.nan);gt=np.full((n,12,2),np.nan);heading=np.full((n,12),np.nan);mask=np.zeros((n,12),bool);em=np.zeros(12,bool)
    samplets=np.full(12,-1,np.int64);lidarts=np.full(12,-1,np.int64);pose_ts=np.full(12,-1,np.int64);_,originalposes=original_extras()
    for j,sample in enumerate(samples[t0+1:t0+13]):
        sd=db.get('sample_data',sample['data']['LIDAR_TOP']);pose=db.get('ego_pose',sd['ego_pose_token']);assert sd['sample_token']==sample['token']
        ego[j]=global_to_ego(np.array(pose['translation'][:2],np.float64),obs['origin'],obs['yaw']);eh[j]=global_heading_to_ego(quaternion_yaw(pose['rotation']),obs['yaw']);em[j]=True
        samplets[j]=sample['timestamp'];lidarts[j]=sd['timestamp'];sourcepose=originalposes.get(pose['token'])
        if sourcepose is not None:pose_ts[j]=sourcepose['timestamp'];assert np.array_equal(sourcepose['translation'],pose['translation'])
        anns=annotation_index(db,sample)
        for i,token in enumerate(obs['instance_tokens']):
            if token not in anns:continue
            ann=anns[token];gt[i,j]=global_to_ego(ann['translation'][:2],obs['origin'],obs['yaw']);heading[i,j]=global_heading_to_ego(quaternion_yaw(ann['rotation']),obs['yaw']);mask[i,j]=True
    return dict(ego_future_gt=ego,ego_future_heading_gt=eh,ego_future_mask=em,agent_future_gt=gt,agent_future_heading_gt=heading,agent_future_mask=mask,
        future_sample_timestamps_us=samplets,future_lidar_timestamps_us=lidarts,future_ego_pose_timestamps_us=pose_ts,EvaluationOnly=True)

def select_scenes():
    con=connection();heads=set(scene_folds());logs={r['token']:r['location'] for r in read_json(RAW/'v1.0-trainval/log.json')}
    scenes={t:json.loads(p) for t,p in con.execute('SELECT token,payload FROM scenes') if t in heads}
    q="""SELECT s.scene,s.token,
      sum(json_extract(a.payload,'$.category_name') LIKE 'vehicle.%' AND json_extract(a.payload,'$.category_name')!='vehicle.bicycle'),
      sum(json_extract(a.payload,'$.category_name') LIKE 'human.pedestrian.%'),sum(json_extract(a.payload,'$.category_name')='vehicle.bicycle')
      FROM samples s LEFT JOIN annotations a ON a.sample=s.token
      WHERE s.scene IN ("""+','.join('?' for _ in heads)+") GROUP BY s.token"
    maxima={s:np.zeros(3,np.int64) for s in heads}
    for scene,sample,v,p,b in con.execute(q,sorted(heads)):
        if scene in heads:maxima[scene]=np.maximum(maxima[scene],[v or 0,p or 0,b or 0])
    picked=[];rows=[]
    for region in sorted(set(logs[scenes[s]['log_token']] for s in heads)):
        candidates=sorted(s for s in heads if logs[scenes[s]['log_token']]==region)
        for role,col in [('Vehicle-rich',0),('Pedestrian-rich',1),('Bicycle-coverage',2)]:
            unused=[s for s in candidates if s not in picked]
            if role=='Bicycle-coverage':positive=[s for s in unused if maxima[s][2]>0];chosen=(positive or unused)[0]
            else:chosen=min(unused,key=lambda s:(-int(maxima[s][col]),s))
            picked.append(chosen);rows.append(dict(SceneToken=chosen,MapRegion=region,SelectionRole=role,Fold=scene_folds()[chosen],
                MaxObservedVehicle=int(maxima[chosen][0]),MaxObservedPedestrian=int(maxima[chosen][1]),MaxObservedBicycle=int(maxima[chosen][2])))
    assert len(picked)==12 and len(set(picked))==12;dump('01_dataset_audit/stage13a_scene_selection.csv',rows)
    anchors=[]
    for token in picked:
        db=load_scene(con,token);chain=scene_samples(db,db.scene[0]);last=len(chain)-13;assert last>=4
        for i,role in [(4,'first-evaluable'),((4+last)//2,'middle-evaluable'),(last,'last-evaluable'),(len(chain)-1,'inference-only-missing-future')]:
            scheduled=i+12<len(chain);times=(np.array([s['timestamp'] for s in chain[i+1:i+13]])-chain[i]['timestamp'])/1e6 if scheduled else np.arange(1,13)*.5
            anchors.append(dict(SceneToken=token,SampleToken=chain[i]['token'],T0Index=i,Fold=scene_folds()[token],Role=role,Has12FutureKeyframes=scheduled,
                FutureTimeSource='recorded keyframe query timestamps' if scheduled else 'externally requested nominal0.5s query',FutureTimes=times.tolist(),Region=db.get('log',db.scene[0]['log_token'])['location']))
    con.close();atomic_json(ROOT/'01_dataset_audit/stage13a_fixed_samples.json',dict(Status='FIXED_BEFORE_INFERENCE',Scenes=picked,Samples=anchors,ProtocolSHA256=sha256(PROTOCOL)))
    print('STAGE13A_FIXED12',len(anchors),'anchors',flush=True)
if __name__=='__main__':select_scenes()
