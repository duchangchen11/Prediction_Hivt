"""Complete TRAIN/VAL audit; predictions and source maps remain immutable."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0b_common import *
from stage8a0b_entities import EntityIndex
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def initialize_worker():
    global worker_index
    torch.set_num_threads(1)
    worker_index=EntityIndex()

def retrieve_worker(job):
    points,location,types=job
    return worker_index.retrieve(points,location,types),worker_index.lanes.retrieve(points,location)

class NoFuture(dict):
    def __getitem__(self,key):
        assert key not in {'GT','future_mask','target_mask'},'forbidden retrieval access: '+key
        return super().__getitem__(key)

def coordinate_check(index,w,selected,result,rng):
    local=w.predicted[selected].numpy().reshape(-1,12,2).astype(np.float64)
    global_points=ego_to_global(local,w.origin,w.yaw)
    roundtrip=float(np.abs(global_to_ego(global_points,w.origin,w.yaw)-local).max())
    region=index.regions[w.map_location]
    available=np.flatnonzero(result['counts'].reshape(-1,7)[:,2:].sum(-1)>0)
    chosen=rng.choice(available,size=min(3,len(available)),replace=False) if len(available) else []
    pairs=[]
    # Check actual matched polygons, including all components of multipart drivable records.
    for row in chosen:
        ids=result['all_ids'].reshape(-1,8)[row]
        for eid in ids[ids>=0]:
            info=index.dictionary[int(eid)]
            if info['type_id']>=2:
                entity=next(e for e in region['entities'] if e['id']==eid)
                pairs.append((int(row),entity['geometry']));break
    # Always include random source polygon geometry, even if there is no match.
    for i in rng.choice(len(region['polygons']),size=min(3,len(region['polygons'])),replace=False):
        pairs.append((int(rng.integers(len(local))),region['polygons'][i]))
    maximum=0.;boundary_max=0.;polygon_rt=0.
    for row,polygon in pairs:
        p_local=shapely.transform(polygon,lambda xy:global_to_ego(xy,w.origin,w.yaw))
        p_back=shapely.transform(p_local,lambda xy:ego_to_global(xy,w.origin,w.yaw))
        xy=shapely.get_coordinates(polygon);back=shapely.get_coordinates(p_back)
        polygon_rt=max(polygon_rt,float(np.abs(xy-back).max()))
        gl=shapely.LineString(global_points[row]);ll=shapely.LineString(local[row])
        maximum=max(maximum,abs(gl.distance(polygon)-ll.distance(p_local)))
        # Polygon collections are kept as original components; compare component boundaries.
        for pg,pl in zip(shapely.get_parts(polygon),shapely.get_parts(p_local)):
            boundary_max=max(boundary_max,abs(gl.distance(pg.boundary)-ll.distance(pl.boundary)))
    assert max(roundtrip,polygon_rt,maximum,boundary_max)<1e-5
    return {'CandidateRoundtripMaxM':roundtrip,'PolygonRoundtripMaxM':polygon_rt,'GeometryDistanceMaxM':maximum,
        'BoundaryDistanceMaxM':boundary_max,'PolygonsChecked':len(pairs)}

def main():
    torch.set_num_threads(2)
    registration=read_json(ROOT/'00_manifest/stage8a0b_registration.json');assert not registration['training']
    frozen=read_json(ROOT/'00_manifest/stage8a0b_frozen_inputs.json')
    assert sha256(ROOT/'00_manifest/stage8a0b_execution_plan.md')==frozen['preregistration_sha256']
    index=EntityIndex();old=read_json(STAGE8/'02_graph_cache/stage8a_graph_cache_manifest.json')
    identity=list(csv.DictReader((STAGE8/'06_tables/stage8a_candidate_identity_windows.csv').open()))
    sample={(r['Split'],int(r['DatasetIndex'])) for r in identity};assert len(sample)==200
    rng=np.random.default_rng(2022);audits=[];manifest=[];stats={'train':[],'val':[]};began=time.monotonic()
    # Each process owns its GEOS context and prepared polygons. No shared-geometry threads.
    pool=ProcessPoolExecutor(max_workers=6,initializer=initialize_worker,mp_context=multiprocessing.get_context('spawn'))
    for batch_number,b in enumerate(old['batches']):
        source=STAGE6/b['source_cache_path'];oldpath=STAGE8/'02_graph_cache'/b['path']
        assert sha256(source)==b['source_cache_sha256'] and sha256(oldpath)==b['sha256']
        payload=torch.load(source,map_location='cpu',weights_only=False)
        with np.load(oldpath) as z: old_arrays={k:z[k] for k in z.files}
        arrays={k:[] for k in ['counts','all_ids','all_distance','relevant_ids','relevant_distance','relevant_counts','type_aware_coverage']}
        selected_rows=[];offsets=[0];split=b['split']
        assert len(payload['windows'])==b['windows']
        observed=[];jobs=[]
        for j,c in enumerate(payload['windows']):
            location=list(index.lanes.regions)[int(old_arrays['window_location'][j])]
            origin=old_arrays['window_origin'][j];yaw=float(old_arrays['window_yaw'][j])
            w=observable_window(NoFuture(c),location,origin,yaw)
            selected=torch.where(~w.history_padding[:,4])[0].numpy()
            points=ego_to_global(w.predicted[selected].numpy().reshape(-1,12,2),origin,yaw)
            jobs.append((points,location,np.repeat(w.actor_type[selected].numpy(),6)))
            observed.append((w,selected))
        retrieved=list(pool.map(retrieve_worker,jobs))
        for j,c in enumerate(payload['windows']):
            assert c['dataset_index']==b['dataset_indices'][j]
            location=list(index.lanes.regions)[int(old_arrays['window_location'][j])]
            origin=old_arrays['window_origin'][j];yaw=float(old_arrays['window_yaw'][j])
            w,selected=observed[j];flat_result,exact=retrieved[j]
            result={key:value.reshape((len(selected),6)+value.shape[1:]) for key,value in flat_result.items()}
            assert w.predicted.data_ptr()==c['ego_prediction'].data_ptr()
            sl=slice(int(old_arrays['node_offset'][j]),int(old_arrays['node_offset'][j+1]))
            assert np.array_equal(selected,old_arrays['actor_node'][sl])
            lane_counts=result['counts'][...,:2].sum(-1)
            assert np.array_equal(lane_counts,old_arrays['within_radius_count'][sl])
            assert np.array_equal(lane_counts==0,old_arrays['map_fallback'][sl])
            # The same frozen rule yields the same top8 tokens/distances BEFORE fallback.
            actual_lane=index.lanes.regions[location]['ids'][np.maximum(exact['region_token'],0)]
            actual_lane[~exact['mask']]=-1
            assert np.array_equal(actual_lane.reshape(-1,6,8),old_arrays['map_token_index'][sl])
            assert np.array_equal(exact['minimum_distance'].reshape(-1,6,8),old_arrays['map_minimum_distance'][sl])
            typ=c['agent_type'][selected].numpy().astype(np.int8)
            motion=np.array([MOTIONS.index(c['motion_state'][int(i)]) if c['motion_state'][int(i)] in MOTIONS else 3 for i in selected],dtype=np.int8)
            # Accounting only: future/target masks do not reach retrieval.
            full=(c['target_mask']&c['future_mask'].all(-1))[selected].numpy()
            displacement=np.linalg.norm(c['ego_prediction'][selected,:,-1].numpy()-c['ego_prediction'][selected,:,0].numpy(),axis=-1)
            bins=np.where(displacement<=1.,0,np.where(displacement<=5.,1,2)).astype(np.int8)
            stat={'actor_type':typ,'motion':motion,'full':full,'counts':result['counts'].astype(np.int16),
                'net_bin':bins,'lane_top8_count':np.where(lane_counts>0,np.minimum(lane_counts,8),0).astype(np.int16),
                'neighbors':old_arrays['neighbor_mask'][sl].sum(-1).astype(np.int8)}
            stats[split].append(stat)
            if (split,c['dataset_index']) in sample:
                poison=dict(c);poison['GT']=torch.full_like(c['GT'],float('nan'))
                poison['future_mask']=~c['future_mask'];poison['target_mask']=~c['target_mask']
                sw=observable_window(NoFuture(poison),location,origin,yaw)
                ss,sr=index.build(sw);assert np.array_equal(ss,selected)
                for key in result:assert np.array_equal(result[key],sr[key]),key
                cr=coordinate_check(index,w,selected,result,rng)
                audits.append({'Split':split,'DatasetIndex':c['dataset_index'],'SceneToken':c['scene_token'],
                    'SampleToken':c['sample_token'],'Actors':len(selected),'GTLeakage':'PASS','CandidateIdentity':'PASS',**cr})
            for key in arrays:arrays[key].append(result[key])
            selected_rows.append(selected);offsets.append(offsets[-1]+len(selected))
        out={k:np.concatenate(v,axis=0) for k,v in arrays.items()}
        out.update(actor_node=np.concatenate(selected_rows),node_offset=np.array(offsets,dtype=np.int32),
            dataset_indices=np.array(b['dataset_indices'],dtype=np.int32))
        path=ROOT/'03_entity_retrieval'/('stage8a0b_'+split+'_'+str(b['dataset_indices'][0]).zfill(5)+'.npz')
        atomic_npz(path,out)
        manifest.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'bytes':path.stat().st_size,
            'source_path':b['source_cache_path'],'source_sha256':b['source_cache_sha256'],'split':split,
            'windows':b['windows'],'actors':offsets[-1]})
        if batch_number%20==0 or batch_number+1==len(old['batches']):
            print('BATCH',batch_number+1,'/',len(old['batches']),'elapsed_s',round(time.monotonic()-began,1),'audit_windows',len(audits),flush=True)
    pool.shutdown()
    assert len(audits)==200 and sum(r['Split']=='train' for r in audits)==100
    for split,parts in stats.items():
        combined={k:np.concatenate([p[k] for p in parts],axis=0) for k in parts[0]}
        assert len(combined['actor_type'])=={'train':459612,'val':91092}[split]
        assert combined['full'].sum()=={'train':290085,'val':54990}[split]
        atomic_npz(ROOT/'04_coverage'/f'stage8a0b_{split}_accounting.npz',combined)
    write_csv(ROOT/'06_tables/stage8a0b_integrity_windows.csv',audits)
    atomic_json(ROOT/'03_entity_retrieval/stage8a0b_retrieval_manifest.json',{'status':'PASS','batches':manifest,
        'source_batches':len(manifest),'candidate_modes':3304224,'full_horizon_modes':2070450,
        'source_sha_verified':True,'old_lane_tokens_and_distances_bitwise_identical':True,'cache_has_GT':False,
        'cache_has_future_masks':False,'all_future_mask_use':'offline population accounting only',
        'cache_bytes':sum(r['bytes'] for r in manifest),'elapsed_seconds':time.monotonic()-began})
    atomic_json(ROOT/'04_coverage/stage8a0b_integrity_audit.json',{'candidate_identity':'PASS',
        'candidate_identity_evidence':'Historical exact predictor replay on 100 TRAIN +100 VAL windows; all 1283 source hashes reverified; candidates aliased without modification; lane retrieval reproduced bitwise.',
        'historical_replay_sha256':sha256(STAGE8/'01_cache_audit/stage8a_candidate_identity.json'),
        'coordinate':'PASS','GT_leakage':'PASS','semantic_attachment':'PASS','windows':200,
        'candidate_roundtrip_max_m':max(r['CandidateRoundtripMaxM'] for r in audits),
        'polygon_roundtrip_max_m':max(r['PolygonRoundtripMaxM'] for r in audits),
        'geometry_distance_max_m':max(r['GeometryDistanceMaxM'] for r in audits),
        'boundary_distance_max_m':max(r['BoundaryDistanceMaxM'] for r in audits),
        'polygons_checked':sum(r['PolygonsChecked'] for r in audits),'poison_GT':'NaN',
        'poison_future_and_target_masks':'inverted','forbidden_access_guard':True,'all_arrays_bitwise_unchanged':True,
        'optimizer_steps':0,'backward_calls':0,'training_executed':False,'test_used':False})
    print('COMPLETE',round(time.monotonic()-began,1),'s',flush=True)

if __name__=='__main__':main()
