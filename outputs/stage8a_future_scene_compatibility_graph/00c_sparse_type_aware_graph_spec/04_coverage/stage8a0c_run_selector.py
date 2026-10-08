"""Full pure-geometry quota audit. GT/masks are offline population accounting only."""
from pathlib import Path
import sys,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'00_manifest'),str(Path(__file__).resolve().parents[1]/'01_selector')]
from stage8a0c_common import *
from stage8a0c_selector import SparseSemanticIndex

def initialize_worker():
    global maps
    torch.set_num_threads(1);maps=SparseSemanticIndex()
def select_worker(job):
    points,location,types=job
    return maps.select_batch(types,points,location)

def main():
    torch.set_num_threads(2);index=SparseSemanticIndex()
    atomic_json(ROOT/'02_graph_cache/stage8a0c_entity_dictionary.json',index.dictionary)
    old_dictionary=read_json(STAGE0B/'03_entity_retrieval/stage8a0b_entity_dictionary.json')
    new_ids={(e['location'],e['type_id'],e['token']):e['index'] for e in index.dictionary}
    translation=np.array([new_ids.get((e['location'],e['type_id'],e['token']),-1) for e in old_dictionary],dtype=np.int32)
    for e in old_dictionary:
        if e['type_id']<6:
            current=index.dictionary[int(translation[e['index']])]
            assert current['semantic']==e['semantic']
    manifests=read_json(STAGE0B/'03_entity_retrieval/stage8a0b_retrieval_manifest.json')['batches']
    old8=read_json(STAGE8/'02_graph_cache/stage8a_graph_cache_manifest.json')['batches']
    assert len(manifests)==len(old8)==1283
    parts={'train':[],'val':[]};output=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=6,initializer=initialize_worker,mp_context=multiprocessing.get_context('spawn')) as pool:
        for bn,(b,frame) in enumerate(zip(manifests,old8)):
            source=STAGE6/b['source_path'];oldpath=STAGE0B/b['path'];framepath=STAGE8/'02_graph_cache'/frame['path']
            assert sha256(source)==b['source_sha256'] and sha256(oldpath)==b['sha256'] and sha256(framepath)==frame['sha256']
            payload=torch.load(source,map_location='cpu',weights_only=False)
            with np.load(oldpath) as z:old={k:z[k] for k in z.files}
            with np.load(framepath) as z:frames={k:z[k] for k in ['window_origin','window_yaw','window_location','neighbor_mask']}
            jobs=[];selected_rows=[]
            for j,c in enumerate(payload['windows']):
                selected=torch.where(~c['history_padding'][:,4])[0].numpy();selected_rows.append(selected)
                location=list(index.regions)[int(frames['window_location'][j])]
                w=observable_window(c,location,frames['window_origin'][j],float(frames['window_yaw'][j]))
                assert w.predicted.data_ptr()==c['ego_prediction'].data_ptr()
                points=ego_to_global(w.predicted[selected].numpy().reshape(-1,12,2),w.origin,w.yaw)
                jobs.append((points,location,np.repeat(w.actor_type[selected].numpy(),6)))
            retrieved=list(pool.map(select_worker,jobs));saved={};offsets=[0]
            for j,(c,selected,result) in enumerate(zip(payload['windows'],selected_rows,retrieved)):
                result={key:value.reshape((len(selected),6)+value.shape[1:]) for key,value in result.items()}
                sl=slice(int(old['node_offset'][j]),int(old['node_offset'][j+1]))
                assert np.array_equal(selected,old['actor_node'][sl])
                assert np.array_equal(result['uncapped_counts'],old['counts'][sl,...,:6])
                assert np.array_equal(result['map_mask'].any(-1),old['type_aware_coverage'][sl])
                for current,previous in [('global_top8_ids','relevant_ids'),('all_six_top8_ids','all_ids')]:
                    previous_ids=old[previous][sl];expected=translation[np.maximum(previous_ids,0)];expected[previous_ids<0]=-1
                    assert np.array_equal(result[current],expected)
                types=c['agent_type'][selected].numpy().astype(np.int8)
                states=('vehicle.moving','vehicle.stopped','vehicle.parked','unknown')
                motion=np.array([states.index(c['motion_state'][int(i)]) if c['motion_state'][int(i)] in states else 3 for i in selected],dtype=np.int8)
                full=(c['target_mask']&c['future_mask'].all(-1))[selected].numpy()
                stat={key:result[key] for key in ['uncapped_counts','selected_counts','entity_types','global_top8_types','all_six_top8_types']}
                stat.update(actor_type=types,motion=motion,full=full,neighbor_count=frames['neighbor_mask'][sl].sum(-1).astype(np.int8))
                parts[b['split']].append(stat)
                for key,value in result.items():saved.setdefault(key,[]).append(value)
                offsets.append(offsets[-1]+len(selected))
            arrays={k:np.concatenate(v) for k,v in saved.items()}
            arrays.update(actor_node=np.concatenate(selected_rows),node_offset=np.array(offsets,dtype=np.int32),dataset_indices=np.array(frame['dataset_indices'],dtype=np.int32))
            path=ROOT/'02_graph_cache'/('stage8a0c_'+b['split']+'_'+str(frame['dataset_indices'][0]).zfill(5)+'.npz');atomic_npz(path,arrays)
            output.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'bytes':path.stat().st_size,
                'source_path':b['source_path'],'source_sha256':b['source_sha256'],'split':b['split'],'windows':b['windows'],
                'actors':offsets[-1],'frame_path':frame['path'],'frame_sha256':frame['sha256']})
            if bn%40==0 or bn+1==len(manifests):print('QUOTA',bn+1,'/',len(manifests),'seconds',round(time.monotonic()-start,1),flush=True)
    for split,p in parts.items():
        combined={k:np.concatenate([r[k] for r in p]) for k in p[0]}
        assert len(combined['actor_type'])=={'train':459612,'val':91092}[split]
        assert combined['full'].sum()=={'train':290085,'val':54990}[split]
        atomic_npz(ROOT/'04_coverage'/f'stage8a0c_{split}_accounting.npz',combined)
    atomic_json(ROOT/'02_graph_cache/stage8a0c_selector_manifest.json',{'status':'PASS','batches':output,'source_batches':1283,
        'all_source_SHA_verified':True,'uncapped_counts_equal_0B':True,'nonempty_rate_equal_0B':True,'GlobalTop8_equal_0B':True,
        'cache_has_GT_or_future_masks':False,'candidate_tensors_copied':False,'nearest_fallback':False,'learned_NULL':False,
        'six_type_frozen_semantics_equal_0B':True,'invalid_component_policy_equal_0B':True,
        'bytes':sum(r['bytes'] for r in output),'runtime_seconds':time.monotonic()-start})
    print('QUOTA COMPLETE',round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
