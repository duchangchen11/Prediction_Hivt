"""Join frozen actor identities to original Stage8 selector/frame caches."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'00_manifest'),str(Path(__file__).resolve().parents[1]/'02_semantic_cache')]
from stage12a_common import *
def main():
    os.environ['STAGE12A_PHASE']='extract'
    torch.set_num_threads(1);verify();f=observed_identities();n=len(f)
    dest=ROOT/'01_map_integrity/cache';dest.mkdir(exist_ok=True)
    if (dest/'stage12a_frame_manifest.json').exists():
        audit=read_json(dest/'stage12a_frame_manifest.json')
        for name,h in audit['files'].items():assert sha256(dest/name)==h
        print('FROZEN_FRAMES_RESUME_VERIFIED');return
    origins=np.zeros((n,2),np.float64);yaw=np.zeros(n,np.float64);location=np.full(n,-1,np.int8)
    current=np.zeros((n,2),np.float32);heading=np.zeros(n,np.float64);hvalid=np.zeros(n,bool)
    entities=np.full((n,6,8),-1,np.int32);etypes=np.full((n,6,8),-1,np.int8);dists=np.zeros((n,6,8),np.float64)
    filled=np.zeros(n,bool);rows=[];regions=sorted(read_json(SEMANTICS));groups=f.groupby('cache_path',sort=False).indices
    selector={r['source_path']:r for r in read_json(S8C/'02_graph_cache/stage8a0c_selector_manifest.json')['batches'] if r['split']=='train'}
    frozen_candidates=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r')
    frozen_mask=np.load(TRAIN/'arg5.npy',mmap_mode='r')
    for count,(path,ids) in enumerate(groups.items(),1):
        rec=selector[path];frpath=S8/'02_graph_cache'/rec['frame_path'];selpath=S8C/rec['path'];src=S6/path
        assert sha256(frpath)==rec['frame_sha256'] and sha256(selpath)==rec['sha256'] and sha256(src)==rec['source_sha256']
        with np.load(frpath) as fr,np.load(selpath) as sel:
            payload=torch.load(src,map_location='cpu',weights_only=False)
            lookup={int(w['dataset_index']):(j,w) for j,w in enumerate(payload['windows'])}
            for di,within in f.iloc[ids].groupby('dataset_index',sort=False).indices.items():
                ix=ids[within];j,w=lookup[int(di)];assert int(sel['dataset_indices'][j])==int(di)
                assert (f.iloc[ix].scene_token==w['scene_token']).all() and (f.iloc[ix].sample_token==w['sample_token']).all()
                nodes=f.iloc[ix].node_in_graph.to_numpy()
                assert [w['instance_tokens'][int(k)] for k in nodes]==f.iloc[ix].instance_token.tolist()
                assert np.array_equal(w['agent_type'][nodes].numpy(),f.iloc[ix].agent_type_id.to_numpy())
                source_index=f.iloc[ix].source_index.to_numpy()
                assert np.array_equal(w['ego_prediction'][nodes].numpy(),frozen_candidates[source_index])
                assert (~w['history_padding'][nodes,4]).all()
                lo,hi=map(int,sel['node_offset'][j:j+2]);selected=sel['actor_node'][lo:hi]
                rank={int(a):r for r,a in enumerate(selected)};si=lo+np.array([rank[int(a)] for a in nodes])
                entities[ix]=sel['entity_ids'][si];etypes[ix]=sel['entity_types'][si];dists[ix]=sel['geometry_distances'][si]
                assert np.array_equal(entities[ix]>=0,frozen_mask[source_index])
                origins[ix]=fr['window_origin'][j];yaw[ix]=fr['window_yaw'][j];location[ix]=fr['window_location'][j]
                current[ix]=w['history'][nodes,4].numpy()
                for actor,node in zip(ix,nodes):
                    valid=torch.where(~w['history_padding'][int(node)])[0]
                    if len(valid)>=2:
                        delta=(w['history'][int(node),valid[-1]]-w['history'][int(node),valid[-2]]).numpy().astype(np.float64)
                        if np.linalg.norm(delta)>1e-6:heading[actor]=np.arctan2(delta[1],delta[0]);hvalid[actor]=True
                filled[ix]=True
        rows.append({'Source':path,'Frame':str(frpath.relative_to(PROJECT)),'FrameSHA256':rec['frame_sha256'],
            'Selector':str(selpath.relative_to(PROJECT)),'SelectorSHA256':rec['sha256'],'SourceSHA256':rec['source_sha256'],'OOFActors':len(ids)})
        if count%100==0:print('OBSERVATION_FRAME_JOIN',count,'/',len(groups),'actors',int(filled.sum()),flush=True)
    assert filled.all() and np.isin(location,np.arange(4)).all()
    arrays={'origin':origins,'yaw':yaw,'location':location,'current':current,'history_heading':heading,'history_heading_valid':hvalid,
        'entity_ids':entities,'entity_types':etypes,'geometry_distances':dists}
    for name,a in arrays.items():np.save(dest/f'stage12a_{name}.npy',a)
    dump('01_map_integrity/stage12a_source_frames.csv',rows)
    atomic_json(dest/'stage12a_frame_manifest.json',{'Status':'PASS','Actors':n,'Regions':regions,'ActorsExactlyJoined':True,
      'InputFields':'Stage8 origin/yaw/location/selector; Stage6 historical observations and actor identities only',
      'LabelArraysOpened':False,'HeadDevRowsUsed':False,'files':{p.name:sha256(p) for p in dest.glob('*.npy')}})
    print('ALL_OBSERVABLE_FRAMES_AND_QUOTA_ENTITY_IDS_JOINED',flush=True)
if __name__=='__main__':main()
