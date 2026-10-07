"""Original-batch frozen predictor replay and independent map-frame checks."""
from pathlib import Path
import sys,random,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage8a_common import *
from stage8a_graph import *
from stage8a_map import MapIndex
from preprocessing.coordinates import ego_to_global,global_to_ego,rotation_matrix
import shapely
sys.path.insert(0,str(STAGE6/'01_cache'))
from stage6a_cache import forward_batch

def coordinate_check(g,w,index):
    origin=g.origin.numpy();yaw=float(g.ego_yaw);region=index.regions[g.map_location]
    local=w['ego_prediction'].numpy().reshape(-1,12,2);glob=ego_to_global(local,origin,yaw)
    rt=float(np.abs(global_to_ego(glob,origin,yaw)-local).max())
    basis=global_to_ego(origin+np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]]),origin,yaw)
    assert np.allclose(basis,np.eye(2),atol=1e-10) and rt<1e-9
    starts=g.lane_positions.numpy();vectors=g.lane_vectors.numpy();start_error=vector_error=0.
    lookup={t:i for i,t in enumerate(region['tokens'])}
    tokens=np.asarray(g.lane_tokens);count=0
    for token in np.unique(tokens):
        rows=np.flatnonzero(tokens==token);line=region['coordinates'][lookup[token]]
        source=global_to_ego(line,origin,yaw)
        distance=np.linalg.norm(starts[rows,None]-source[None,:-1],axis=-1)
        endpoint_distance=np.linalg.norm((starts[rows]+vectors[rows])[:,None]-source[None,1:],axis=-1)
        # Preserve duplicated source vertices; both endpoints disambiguate segments.
        nearest=(distance+endpoint_distance).argmin(-1)
        start_error=max(start_error,float(distance[np.arange(len(rows)),nearest].max()))
        vector_error=max(vector_error,float(np.abs(vectors[rows]-np.diff(source,axis=0)[nearest]).max()));count+=len(rows)
    assert start_error<1e-4 and vector_error<1e-4,(start_error,vector_error)
    # Brute full-region distance verifies spatial-index completeness and frame invariance.
    chosen=glob[:2];retr=index.retrieve(chosen,g.map_location)
    distance=shapely.distance(shapely.linestrings(chosen)[:,None],region['geometries'][None])
    local_lines=np.asarray([shapely.LineString(global_to_ego(x,origin,yaw)) for x in region['coordinates']],dtype=object)
    local_distance=shapely.distance(shapely.linestrings(local[:2])[:,None],local_lines[None])
    invariant=float(np.abs(distance-local_distance).max());assert invariant<1e-8
    for row in range(len(chosen)):
        eligible=np.flatnonzero(distance[row]<=10.)
        wanted=eligible[np.lexsort((eligible,distance[row,eligible]))][:8] if len(eligible) else np.array([np.argmin(distance[row])])
        got=retr['region_token'][row,retr['mask'][row]]
        assert np.array_equal(wanted,got),(wanted,got)
    return {'roundtrip_max_m':rt,'map_start_max_m':start_error,'map_vector_max_m':vector_error,
            'distance_frame_max_m':invariant,'source_segments_checked':count,'brute_candidates':len(chosen)}

@torch.no_grad()
def main():
    verify_frozen();manifest=read_json(CACHE_MANIFEST);assert manifest['status']=='PASS'
    rng=random.Random(2022);model=frozen_predictor();before=state_digest(model.state_dict());maps=MapIndex()
    maxima={k:0. for k in ('raw_prediction','mode_logits','mode_prob','ego_prediction','rotation')}
    coordinate={k:0. for k in ('roundtrip_max_m','map_start_max_m','map_vector_max_m','distance_frame_max_m')}
    rows=[];segments=0;actors=0;chosen_records=[]
    for split in ('train','val'):
        ds=SceneDataset(split);selected=sorted(rng.sample(range(len(ds)),100));selected_set=set(selected)
        records=[r for r in manifest['batches'] if r['split']==split and any(i in selected_set for i in range(r['start'],r['start']+r['windows']))]
        for ri,record in enumerate(records):
            path=STAGE6/record['path'];assert sha256(path)==record['sha256']
            cached=torch.load(path,map_location='cpu',weights_only=False)
            replay=forward_batch(model,ds,record['start']);by_index={w['dataset_index']:w for w in replay['windows']}
            for old in cached['windows']:
                i=old['dataset_index']
                if i not in selected_set:continue
                new=by_index[i];g=ds[i]
                for field in ('scene_token','sample_token','instance_tokens','motion_state'):
                    assert old[field]==new[field],field
                for field in ('agent_type','history','history_padding','current_position','GT','future_mask','target_mask','rotate_angles'):
                    assert torch.equal(old[field],new[field]),field
                for field in maxima:
                    difference=float((old[field]-new[field]).abs().max());maxima[field]=max(maxima[field],difference)
                    assert torch.equal(old[field],new[field]),(split,i,field,difference)
                assert tensor_sha(old['raw_prediction'])==old['raw_prediction_sha256']
                assert tensor_sha(old['ego_prediction'])==old['ego_prediction_sha256']
                c=coordinate_check(g,old,maps)
                for k in coordinate:coordinate[k]=max(coordinate[k],c[k])
                segments+=c['source_segments_checked'];actors+=len(old['history'])
                rows.append({'Split':split,'DatasetIndex':i,'SceneToken':old['scene_token'],'SampleToken':old['sample_token'],
                    'Actors':len(old['history']),'CandidateMaxDiff':0.,'LogitMaxDiff':0.,'ProbMaxDiff':0.,**c})
            chosen_records.append({'split':split,'path':record['path'],'sha256':record['sha256'],'selected_windows':sum(w['dataset_index'] in selected_set for w in cached['windows'])})
            if (ri+1)%10==0 or ri+1==len(records):print('IDENTITY',split,ri+1,'/',len(records),'checked_windows',sum(x['Split']==split for x in rows),flush=True)
        ds.clear()
    assert len(rows)==200 and all(v==0 for v in maxima.values())
    assert state_digest(model.state_dict())==before and not model.training
    assert not any(p.requires_grad or p.grad is not None for p in model.parameters())
    write_csv(ROOT/'06_tables/stage8a_candidate_identity_windows.csv',rows)
    atomic_json(ROOT/'01_cache_audit/stage8a_candidate_identity.json',{'status':'PASS','TRAIN_windows':100,'VAL_windows':100,
        'seed':2022,'actors_checked':actors,'candidate_maxdiff':maxima['ego_prediction'],'logit_maxdiff':maxima['mode_logits'],
        'prob_maxdiff':maxima['mode_prob'],'all_output_maxdiff':maxima,'identity_history_type_masks_bitwise_equal':True,
        'predictor_eval':True,'predictor_requires_grad':False,'predictor_gradient_count':0,'predictor_state_unchanged':True,
        'original_batch_size_and_membership_preserved':True,'selected_source_batches':chosen_records,'test_used':False})
    atomic_json(ROOT/'01_cache_audit/stage8a_coordinate_audit.json',{'status':'PASS','windows':200,'coordinate_frame':'t0 ego x-forward/y-left meters',
        'origin_yaw_source':'same original scene shard, current LIDAR_TOP ego pose','maxima':coordinate,
        'source_map_segments_checked':segments,'global_local_full_region_distance_agrees':True,'brute_force_retrieval_candidates':400,
        'centerline_resolution_source':'unaltered Stage7A 2m discretized complete centerline archive',
        'duplicate_vertices':'source unchanged; segment match uses both endpoints','future_GT_used_for_frame':False})
    print('STAGE8A_IDENTITY_COORDINATE_PASS',maxima,coordinate,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
