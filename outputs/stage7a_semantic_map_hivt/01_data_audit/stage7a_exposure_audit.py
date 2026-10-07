"""All official scene shards, every map segment and frozen VAL actor identity."""
from pathlib import Path
import sys,csv,collections,json,hashlib,random,sqlite3
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'02_semantic_cache')]
from stage7a_common import *
sys.path.insert(0,str(STAGE3/'02_preprocessed'))
from stage7a_semantic_cache import CONTROL
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from shapely.geometry import LineString,Point
from shapely.strtree import STRtree
from preprocessing.coordinates import ego_to_global,rotation_matrix,wrap_angle
from nuscenes.utils.splits import create_splits_scenes

FEATURES=['NearConnector','NearTurnConnector','NearTrafficControl','NearCrosswalk']
SEG_FIELDS=['Total','Connector','Left','Straight','Right','Unknown','UnknownConnector','TrafficLight','StopSign','Yield','OtherControl','MixedControl','Crosswalk','Crosswalk2m','Crosswalk5m','DirectCrosswalk','DirectCrosswalk2m','DirectCrosswalk5m','SourceTopologyGap']
def fingerprint(t):
    t=t.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes());return h.hexdigest()
def group(df,name):
    if name=='overall':return df
    if name=='vehicle.moving':return df[(df.agent_type=='vehicle')&(df.motion_state=='vehicle.moving')]
    if name=='Vehicle >5m':return df[(df.agent_type=='vehicle')&(df.GT_endpoint_displacement_m>5)]
    return df[df.agent_type==name]
def main():
    torch.set_num_threads(1);verify_v1();reg=read_json(ROOT/'00_manifest/stage7a_exposure_registration.json')
    cache=read_json(ROOT/'02_semantic_cache/stage7a_semantic_metadata.json');z=np.load(ROOT/'02_semantic_cache/stage7a_centerlines.npz');lines={k:z[k] for k in z.files}
    # Arc junctions may repeat starts. A segment is identified by start AND vector.
    trees={k:cKDTree(np.concatenate((v[:-1],np.diff(v,axis=0)),axis=1)) for k,v in lines.items()}
    connector_trees={loc:STRtree([LineString(lines[loc+'__'+t]) for t,v in ms.items() if v['is_connector']]) for loc,ms in cache.items()}
    frozen=read_json(ROOT/'00_manifest/stage7a_frozen_previous.json');official=read_json(PROJECT/'outputs/stage2c_trainval_vehicle_baseline/01_data_audit/stage2c_nuscenes_split_audit.json');split_api=create_splits_scenes()
    with ACTORS.open() as f:prior_rows=list(csv.DictReader(f))
    previous={key(r):r for r in prior_rows};assert len(previous)==len(prior_rows)==85027
    actors=[];seen_val=set();scene_sets={};samples_seen=set();observed=collections.defaultdict(set);segments=collections.defaultdict(collections.Counter);windows=collections.Counter();sceneloc=[]
    max_start=max_vector=max_roundtrip=max_edge=0.;nonzero=collections.Counter();sha_checked=0;empty=collections.Counter();numeric_nonfinite=0;relation_mismatch=0;placeholder_segments=0;null_graphs=collections.Counter()
    dbpath=PROJECT/'outputs/stage2c_trainval_vehicle_baseline/02_preprocessed/stage2c_metadata_cache/stage2c_trajectory_metadata.sqlite';db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
    for split in ('train','val'):
        rows=indexes(split);by_file=collections.defaultdict(list)
        for r in rows:by_file[r['file_path']].append(r)
        scene_sets[split]={r['scene_token'] for r in rows};assert len(by_file)==len(scene_sets[split])==reg['splits'][split]
        source={r['token']:r for r in official['scenes'][split]};assert scene_sets[split]==set(source)
        assert {r['scene_name'] for r in rows}==set(split_api[split])
        for sn,(name,rs) in enumerate(sorted(by_file.items()),1):
            path=STAGE3/name;assert sha256(path)==frozen['scene_shards'][name],name;sha_checked+=1
            payload=torch.load(path,map_location='cpu',weights_only=False);graphs=payload['graphs'];scene=rs[0]['scene_token'];assert payload['scene_token']==scene
            stored=db.execute('SELECT payload FROM scenes WHERE token=?',(scene,)).fetchone();assert stored and json.loads(stored[0])['name']==source[scene]['name']
            assert len(rs)==len(graphs) and {int(r['window_index']) for r in rs}==set(range(len(graphs)))
            loc=None
            for r in rs:
                g=graphs[int(r['window_index'])]
                if g is None:
                    assert r['window_status']=='no supported current actors' and int(r['actor_count'])==int(r['full_horizon_target_count'])==int(r['partial_target_count'])==0
                    null_graphs[split]+=1;windows[split]+=1;empty[split]+=1
                    assert (split,r['sample_token']) not in samples_seen;samples_seen.add((split,r['sample_token']))
                    continue
                assert g.scene_token==scene and g.sample_token==r['sample_token'] and g.scene_name==r['scene_name']
                assert (split,g.sample_token) not in samples_seen;samples_seen.add((split,g.sample_token));assert loc is None or loc==g.map_location;loc=g.map_location
                for k,v in g.to_dict().items():
                    if isinstance(v,torch.Tensor) and v.is_floating_point():numeric_nonfinite+=int((~torch.isfinite(v)).sum())
                windows[split]+=1;full=g.full_horizon_mask.numpy();assert int(full.sum())==int(r['full_horizon_target_count']);empty[split]+=int(not g.target_mask.any())
                origin=g.origin.numpy();yaw=float(g.ego_yaw);assert abs(np.linalg.det(rotation_matrix(yaw))-1)<1e-12
                lp=g.lane_positions.numpy();lv=g.lane_vectors.numpy();tokens=np.asarray(g.lane_tokens);n=len(lp);assert n==len(tokens)==len(lv)
                for field in ('is_intersections','turn_directions','traffic_controls'):nonzero[field]+=int(torch.count_nonzero(g[field]));assert len(g[field])==n
                placeholder_segments+=n;global_starts=ego_to_global(lp,origin,yaw);global_vectors=lv.astype(np.float64)@rotation_matrix(yaw).T
                local_roundtrip=(global_starts-origin)@rotation_matrix(yaw);max_roundtrip=max(max_roundtrip,float(np.max(np.abs(local_roundtrip-lp),initial=0)))
                flags=np.zeros((n,len(SEG_FIELDS)),dtype=np.int64);actor_sem=np.zeros((int(g.num_nodes),4),dtype=bool)
                breaks=np.r_[np.flatnonzero(np.r_[True,tokens[1:]!=tokens[:-1]]),n] if n else []
                for start,end in zip(breaks[:-1],breaks[1:]):
                    token=str(tokens[start]);sm=cache[loc][token];assert sum(sm['turn_onehot'])==sum(sm['traffic_control_onehot'])==1
                    observed[(loc,token)].add((split,scene,g.scene_name));line=lines[loc+'__'+token]
                    _,indices=trees[loc+'__'+token].query(np.concatenate((global_starts[start:end],global_vectors[start:end]),axis=1))
                    distances=np.linalg.norm(line[indices]-global_starts[start:end],axis=-1);assert np.max(distances)<1e-4,(loc,token,'segment start drift',np.max(distances));max_start=max(max_start,float(np.max(distances)))
                    delta=np.diff(line,axis=0);error=np.linalg.norm(delta[indices]-global_vectors[start:end],axis=-1);assert np.max(error)<1e-4,(loc,token,'vector drift',np.max(error));max_vector=max(max_vector,float(np.max(error)))
                    assert (np.linalg.norm(delta[indices],axis=-1)>1e-6).all()
                    valid=np.flatnonzero(np.linalg.norm(delta,axis=-1)>1e-6);ordinal=np.searchsorted(valid,indices)
                    attrs=[1,sm['is_connector'],sm['turn_type']=='left',sm['turn_type']=='straight',sm['turn_type']=='right',sm['turn_type']=='unknown',
                        sm['is_connector'] and sm['turn_type']=='unknown','traffic_light' in sm['control_types_present'],'stop_sign' in sm['control_types_present'],'yield' in sm['control_types_present'],
                        'other_control' in sm['control_types_present'],sm['traffic_control_type']=='mixed_control',sm['near_ped_crossing'],sm['near_ped_crossing_2m'],sm['near_ped_crossing_5m']]
                    flags[start:end,:15]=attrs
                    for j,label in enumerate(('intersects','within2m','within5m')):flags[start:end,15+j]=np.isin(ordinal,sm['direct_crosswalk_segment_indices'][label])
                    flags[start:end,18]=sm['has_source_topology_gap']
                assert np.all(flags[:,2:6].sum(-1)==1)
                segments[(split,loc)].update(dict(zip(SEG_FIELDS,flags.sum(0).tolist())))
                current=g.positions[:,4].numpy();edge=g.lane_actor_index.numpy();assert edge.shape[0]==2
                offsets=lp[:,None,:]-current[None,:,:];expected=np.stack(np.where(np.linalg.norm(offsets,axis=-1)<50)).astype(np.int64)
                relation_mismatch+=int(not np.array_equal(edge,expected));assert np.array_equal(edge,expected),'Old 50m map relation changed'
                if edge.shape[1]:
                    diff=np.abs(g.lane_actor_vectors.numpy()-offsets[edge[0],edge[1]]);max_edge=max(max_edge,float(diff.max(initial=0)))
                    near_flags=np.stack((flags[:,1]>0,(flags[:,2]+flags[:,4])>0,flags[:,7:11].sum(-1)>0,flags[:,12]>0),-1)
                    for j in range(4):np.logical_or.at(actor_sem[:,j],edge[1],near_flags[edge[0],j])
                nodes=np.flatnonzero(full);global_current=ego_to_global(current[nodes],origin,yaw);point_geoms=np.asarray([Point(p) for p in global_current],dtype=object)
                pairs=connector_trees[loc].query(point_geoms,predicate='dwithin',distance=30);min_dist=np.full(len(nodes),np.inf)
                if pairs.size:
                    ds=np.array([point_geoms[i].distance(connector_trees[loc].geometries[j]) for i,j in pairs.T]);np.minimum.at(min_dist,pairs[0],ds)
                gt=g.positions[:,5:];disp=torch.linalg.vector_norm(gt[:,-1]-g.positions[:,4],dim=-1).numpy()
                if split=='val':
                    for node in torch.where(g.target_mask)[0].tolist():
                        horizon='full_horizon' if full[node] else 'partial_future';k=(scene,g.sample_token,g.instance_tokens[node],horizon);assert k in previous and k not in seen_val;seen_val.add(k);old=previous[k]
                        assert int(old['node_in_graph'])==node and old['agent_type']==CLASSES[int(g.agent_type[node])] and old['motion_state']==g.t0_motion_state[node]
                        assert old['GT_trajectory_sha256']==fingerprint(gt[node]);assert old['future_mask_bits']==''.join('1' if x else '0' for x in g.future_mask[node].tolist())
                        assert int(old['agent_type_id'])==int(g.agent_type[node])
                for j,node in enumerate(nodes):
                    trajectory=g.positions[node,4:].numpy();vin=trajectory[2]-trajectory[0];vout=trajectory[-1]-trajectory[-3]
                    heading_eligible=int(g.agent_type[node])==0 and disp[node]>5 and np.linalg.norm(vin)>0.5 and np.linalg.norm(vout)>0.5
                    delta=float(np.degrees(wrap_angle(np.arctan2(vout[1],vout[0])-np.arctan2(vin[1],vin[0])))) if heading_eligible else None
                    a={'split':split,'scene_name':g.scene_name,'scene_token':scene,'sample_token':g.sample_token,'instance_token':g.instance_tokens[node],
                        'horizon':'full_horizon','node_in_graph':int(node),'map_location':loc,'agent_type':CLASSES[int(g.agent_type[node])],'motion_state':g.t0_motion_state[node],
                        'GT_endpoint_displacement_m':float(disp[node]),**{k:int(v) for k,v in zip(FEATURES,actor_sem[node])},
                        'IntersectionVehicle_A':int(g.agent_type[node]==0 and min_dist[j]<20),'IntersectionVehicle_B':int(g.agent_type[node]==0 and min_dist[j]<30),
                        'GT_heading_eligible':int(heading_eligible),'GT_heading_delta_deg':delta,'TurningVehicle_GT':int(heading_eligible and abs(delta)>20)}
                    if split=='val':
                        old=previous[key(a)];assert abs(float(old['GT_endpoint_displacement_m'])-a['GT_endpoint_displacement_m'])<1e-5
                        a.update(Stage5A_minFDE=float(old['minFDE6']),Stage5A_Top1FDE=float(old['R0_Top1FDE']),R2_Top1FDE=float(old['R2_Top1FDE']))
                    else:a.update(Stage5A_minFDE=None,Stage5A_Top1FDE=None,R2_Top1FDE=None)
                    actors.append(a)
            sceneloc.append({'split':split,'scene_name':rs[0]['scene_name'],'scene_token':scene,'map_location':loc or 'NOT_AVAILABLE_ALL_GRAPHS_EMPTY',
                'window_count':len(rs),'shard_sha256':frozen['scene_shards'][name]})
            if sn%25==0:print('SHARDS',split,sn,'/',len(by_file),'windows',windows[split],flush=True)
    db.close();assert not scene_sets['train']&scene_sets['val'];assert len(seen_val)==85027 and seen_val==set(previous)
    assert numeric_nonfinite==relation_mismatch==max_edge==0 and not any(nonzero.values())
    df=pd.DataFrame(actors);assert len(df[df.split=='val'])==54990;assert not df.duplicated(['split','scene_token','sample_token','instance_token','horizon']).any()
    for split in ('train','val'):write_csv(ROOT/'01_data_audit'/f'stage7a_actor_{split}_semantic_context.csv',[a for a in actors if a['split']==split])
    exposure=[];groups=['overall',*CLASSES,'vehicle.moving','Vehicle >5m'];moving_n=[];screen=reg['sufficient_moving_exposure_candidate']
    for split in ('train','val','combined'):
        subset=df if split=='combined' else df[df.split==split]
        for name in groups:
            g=group(subset,name);assert len(g)>0;exposure.append({'Split':split,'Group':name,'Count':len(g),
                **{f+'Rate':float(g[f].mean()) for f in FEATURES},'Scenes':int(g.scene_token.nunique()),'Instances':int(g.instance_token.nunique())})
        for f in FEATURES:
            g=group(subset,'vehicle.moving');near=g[g[f]==1]
            moving_n.append({'Split':split,'Semantic':f,'Count':len(near),'Scenes':int(near.scene_token.nunique()),'Instances':int(near.instance_token.nunique()),
                'sufficient':len(near)>=screen['minimum_actor_windows_each_semantic'] and near.scene_token.nunique()>=screen['minimum_scenes_each_semantic'] and near.instance_token.nunique()>=screen['minimum_instances_each_semantic']})
    valmoving=group(df[df.split=='val'],'vehicle.moving');errors=[]
    for f in FEATURES:
        for value in (1,0):
            g=valmoving[valmoving[f]==value];name=f if value else f.replace('Near','No',1)
            errors.append({'Context':name,'Count':len(g),**{metric:float(g[metric].mean()) if len(g) else None for metric in ['Stage5A_minFDE','Stage5A_Top1FDE','R2_Top1FDE']},
                'Scenes':int(g.scene_token.nunique()),'Instances':int(g.instance_token.nunique()),'Split':'val','Horizon':'full_horizon','Group':'vehicle.moving'})
    write_csv(ROOT/'06_tables/stage7a_actor_semantic_exposure.csv',exposure);write_csv(ROOT/'06_tables/stage7a_semantic_context_error_audit.csv',errors)
    write_csv(ROOT/'06_tables/stage7a_moving_semantic_sample_counts.csv',moving_n);write_csv(ROOT/'06_tables/stage7a_scene_map_provenance.csv',sceneloc)
    out=[]
    for (split,loc),c in segments.items():out.append({'Split':split,'location':loc,**c,**{k+'Rate':c[k]/c['Total'] for k in SEG_FIELDS[1:]}})
    for split in ('train','val','combined'):
        c=collections.Counter()
        for (s,l),v in segments.items():
            if split=='combined' or s==split:c.update(v)
        out.append({'Split':split,'location':'all',**c,**{k+'Rate':c[k]/c['Total'] for k in SEG_FIELDS[1:]}})
    write_csv(ROOT/'06_tables/stage7a_stage3_segment_semantic_coverage.csv',out)
    diagnostics=[]
    for split in ('train','val'):
        vehicle=group(df[df.split==split],'vehicle')
        for candidate in ('IntersectionVehicle_A','IntersectionVehicle_B','GT_heading_eligible','TurningVehicle_GT'):
            selected=vehicle[vehicle[candidate]==1];diagnostics.append({'Split':split,'Candidate':candidate,'Count':len(selected),'VehicleDenominator':len(vehicle),
                'VehicleRate':len(selected)/len(vehicle),'Scenes':selected.scene_token.nunique(),'Instances':selected.instance_token.nunique(),
                'Stage5A_minFDE':selected.Stage5A_minFDE.mean() if split=='val' and len(selected) else None,
                'Stage5A_Top1FDE':selected.Stage5A_Top1FDE.mean() if split=='val' and len(selected) else None,'R2_Top1FDE':selected.R2_Top1FDE.mean() if split=='val' and len(selected) else None})
    write_csv(ROOT/'06_tables/stage7a_intersection_turning_group_candidates.csv',diagnostics)
    rng=random.Random(2022);observed_keys=sorted(observed);predicates={'lane':lambda v:not v['is_connector'],'connector':lambda v:v['is_connector'],
        'traffic_light':lambda v:'traffic_light' in v['control_types_present'],'stop_yield':lambda v:'stop_sign' in v['control_types_present'] or 'yield' in v['control_types_present'],'crosswalk':lambda v:v['near_ped_crossing']}
    samples={}
    for name,pred in predicates.items():
        candidates=[k for k in observed_keys if pred(cache[k[0]][k[1]])];chosen=rng.sample(candidates,min(10,len(candidates)));assert len(chosen)==10
        samples[name]=[{'location':loc,'token':token,'scenes_observed':[{'split':s,'scene_token':t,'scene_name':n} for s,t,n in sorted(observed[(loc,token)])],
            'semantic_metadata':cache[loc][token],'geometry_summary':{'bounds_xy':[*lines[loc+'__'+token].min(0).tolist(),*lines[loc+'__'+token].max(0).tolist()],
                'point_count':len(lines[loc+'__'+token]),'length_m':cache[loc][token]['centerline_length_m'],'coordinate_frame':'global'}} for loc,token in chosen]
    atomic_json(ROOT/'01_data_audit/stage7a_semantic_manual_samples.json',{'status':'AWAITING_VISUAL_REVIEW','random_seed':2022,'observed_trainval_tokens_only':True,'groups':samples})
    qa={'status':'PASS','TRAIN_scenes':len(scene_sets['train']),'VAL_scenes':len(scene_sets['val']),'scene_overlap':0,'shard_SHA_checked':sha_checked,
        'windows':dict(windows),'empty_supervision_windows':dict(empty),'no_current_actor_null_graphs':dict(null_graphs),
        'all_empty_graph_scenes':[r['scene_name'] for r in sceneloc if r['map_location']=='NOT_AVAILABLE_ALL_GRAPHS_EMPTY'],
        'all_candidate_windows_audited':True,'placeholder_lane_segments_checked':placeholder_segments,
        'nonzero_old_placeholders':dict(nonzero),'NaN_or_Inf_count':numeric_nonfinite,'unresolved_segment_tokens':0,
        'maximum_global_segment_start_difference_m':max_start,'maximum_global_segment_vector_difference_m':max_vector,
        'maximum_roundtrip_difference_m':max_roundtrip,'maximum_edge_vector_difference_m':max_edge,'map_radius_relation_mismatch_windows':relation_mismatch,
        'frozen_VAL_GT_identity_matches':len(seen_val),'full_VAL_actor_count':54990,'TRAIN_full_actor_count':len(df[df.split=='train']),
        'actor_results_SHA256':sha256(ACTORS),'semantic_metadata_SHA256':sha256(ROOT/'02_semantic_cache/stage7a_semantic_metadata.json'),
        'all_control_references_valid':all(int(r[k])==0 for r in csv.DictReader(open(ROOT/'06_tables/stage7a_stop_line_raw_statistics.csv')) for k in ['invalid_traffic_light_references','invalid_ped_crossing_references','invalid_road_block_references']),
        'segment_identity_match':'joint global start XY and vector XY; resolves repeated arc-junction starts without relaxing tolerances',
        'raw_topology_unresolved_reference_count':read_json(ROOT/'01_data_audit/stage7a_source_topology_gaps.json')['unresolved_reference_count'],'raw_topology_repaired':False,'raw_topology_used_in_candidate_features':False,
        'old_model_inference_rerun':False,'new_model_trained':False,'test_used':False,
        'moving_exposure_sufficient':all(r['sufficient'] for r in moving_n if r['Split']!='combined'),
        'source_mount_note':'Raw trainval mount absent; audit reuses byte-frozen official trainval shards and metadata SQLite, verifies each map segment against actual regional Map Expansion JSON.'}
    atomic_json(ROOT/'01_data_audit/stage7a_all_scene_data_quality.json',qa);print('EXPOSURE_DONE',json.dumps(qa),flush=True)
if __name__=='__main__':main()
