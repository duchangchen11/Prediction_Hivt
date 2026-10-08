"""Fresh 200-window feature poisoning, geometry, semantics and neutral-input audit."""
from pathlib import Path
import sys,ast
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/d) for d in ('00_manifest','01_selector','03_model_audit')]
from stage8a0c_common import *
from stage8a0c_selector import SparseSemanticIndex
from stage8a0c_build import build_window
from stage8a0c_model import pack_targets
import shapely

class Protected(dict):
    def __getitem__(self,key):
        assert key not in {'GT','future_mask','target_mask'},'Forbidden feature access '+key
        return super().__getitem__(key)
    def get(self,key,default=None):
        assert key not in {'GT','future_mask','target_mask'},'Forbidden feature access '+key
        return super().get(key,default)

def coordinate_check(w,index,selected,features,rng):
    local=w.predicted[selected].numpy().reshape(-1,12,2).astype(np.float64)
    points=ego_to_global(local,w.origin,w.yaw);reg=index.regions[w.map_location]
    maximum=float(np.abs(global_to_ego(points,w.origin,w.yaw)-local).max());polygon_rt=0.;distance_max=0.
    ids=features['entity_ids'].reshape(-1,8);mask=ids>=0;r,s=np.where(mask)
    unique=np.unique(ids[mask]);local_geometries={}
    for eid in unique:
        geometry=reg['entities'][reg['id_to_local'][int(eid)]]['geometry']
        transformed=shapely.transform(geometry,lambda xy:global_to_ego(xy,w.origin,w.yaw))
        back=shapely.transform(transformed,lambda xy:ego_to_global(xy,w.origin,w.yaw))
        rt=float(np.abs(shapely.get_coordinates(geometry)-shapely.get_coordinates(back)).max())
        polygon_rt=max(polygon_rt,rt);local_geometries[int(eid)]=transformed
    if len(r):
        geoms=np.array([local_geometries[int(i)] for i in ids[r,s]],dtype=object)
        d=shapely.distance(shapely.linestrings(local[r]),geoms)
        old=features['geometry_distances'].reshape(-1,8)[r,s]
        distance_max=float(np.abs(d-old).max())
    for i in rng.choice(len(reg['polygon_parts']),size=3,replace=False):
        geometry=reg['polygon_parts'][i];transformed=shapely.transform(geometry,lambda xy:global_to_ego(xy,w.origin,w.yaw))
        back=shapely.transform(transformed,lambda xy:ego_to_global(xy,w.origin,w.yaw))
        polygon_rt=max(polygon_rt,float(np.abs(shapely.get_coordinates(geometry)-shapely.get_coordinates(back)).max()))
        row=int(rng.integers(len(local)))
        distance_max=max(distance_max,abs(shapely.LineString(points[row]).distance(geometry)-shapely.LineString(local[row]).distance(transformed)))
    assert max(maximum,polygon_rt,distance_max)<1e-5
    return maximum,polygon_rt,distance_max,len(r)

@torch.no_grad()
def main():
    torch.set_num_threads(2);index=SparseSemanticIndex()
    samples=list(csv.DictReader((STAGE0B/'06_tables/stage8a0b_integrity_windows.csv').open()))
    desired={(r['Split'],int(r['DatasetIndex'])) for r in samples};assert len(desired)==200
    records=read_json(STAGE8/'02_graph_cache/stage8a_graph_cache_manifest.json')['batches']
    audited=[];rng=np.random.default_rng(2022);neutral={s:[] for s in ('train','val')};neutral_prob={s:[] for s in neutral};identities=[];seen=set();counts={'train':0,'val':0}
    # Frozen dictionary gives an independent label attachment reference for every lane token.
    old=read_json(STAGE0B/'03_entity_retrieval/stage8a0b_entity_dictionary.json')
    labels={(e['location'],e['type_id'],e['token']):np.array(e['semantic'],dtype=np.float32) for e in old if e['type_id']<6}
    semantic_edges=0
    for b in records:
        wanted=[j for j,i in enumerate(b['dataset_indices']) if (b['split'],i) in desired]
        if not wanted:continue
        source=STAGE6/b['source_cache_path'];frame=STAGE8/'02_graph_cache'/b['path']
        assert sha256(source)==b['source_cache_sha256'] and sha256(frame)==b['sha256']
        payload=torch.load(source,map_location='cpu',weights_only=False)
        with np.load(frame) as z:frame_arrays={k:z[k] for k in z.files}
        for j in wanted:
            c=payload['windows'][j];split=b['split'];location=list(index.regions)[int(frame_arrays['window_location'][j])]
            snapshots={key:c[key].clone() for key in ['raw_prediction','ego_prediction','mode_logits','mode_prob','rotation']}
            w=observable_window(Protected(c),location,frame_arrays['window_origin'][j],float(frame_arrays['window_yaw'][j]))
            selected,f,node,idx,keep,edge=build_window(w,index)
            poison=dict(c);poison['GT']=torch.full_like(c['GT'],float('nan'));poison['future_mask']=~c['future_mask'];poison['target_mask']=~c['target_mask']
            p=observable_window(Protected(poison),location,w.origin,w.yaw)
            ss,ff,nn,ii,kk,ee=build_window(p,index)
            assert np.array_equal(selected,ss)
            for key in f:assert np.array_equal(f[key],ff[key]),key
            for a,bb in [(node,nn),(idx,ii),(keep,kk),(edge,ee)]:assert torch.equal(a,bb)
            sl=slice(int(frame_arrays['node_offset'][j]),int(frame_arrays['node_offset'][j+1]))
            assert np.array_equal(keep[selected].numpy(),frame_arrays['neighbor_mask'][sl])
            assert np.array_equal(idx[selected].numpy(),frame_arrays['neighbor_actor_node'][sl])
            for key,t in snapshots.items():assert torch.equal(t,c[key]),key
            valid=f['map_mask'];entity_ids=f['entity_ids'][valid];types=f['entity_types'][valid];features=f['map_node'][valid]
            assert np.array_equal(features[:,:6],np.eye(6,dtype=np.float32)[types])
            ref=np.stack([labels[(index.dictionary[int(i)]['location'],index.dictionary[int(i)]['type_id'],index.dictionary[int(i)]['token'])] for i in entity_ids]) if len(entity_ids) else np.empty((0,9))
            assert np.array_equal(features[:,6:15],ref)
            assert not features[types>=2,6:].any() and not f['map_edge'][valid][types>=2,6:9].any()
            assert (features[types<2,17]==1).all() and (f['map_edge'][valid][types<2,8]==1).all()
            assert not f['map_edge'][valid][types<2,9:].any()
            semantic_edges+=len(entity_ids)
            rt,pr,dm,geometry_edges=coordinate_check(w,index,selected,f,rng)
            audited.append({'Split':split,'DatasetIndex':c['dataset_index'],'Actors':len(selected),'CandidateMaxDiff':0,
                'LogitMaxDiff':0,'ProbabilityMaxDiff':0,'PoisonedFeatures':'PASS','FrozenSemantic':'PASS',
                'CandidateRoundtripMaxM':rt,'PolygonRoundtripMaxM':pr,'GeometryDistanceMaxM':dm,'GeometryEdgesChecked':geometry_edges})
            if counts[split]<64:
                full=(c['target_mask']&c['future_mask'].all(-1)).numpy();targets=[]
                for actor in selected:
                    key=(c['scene_token'],c['instance_tokens'][int(actor)])
                    if full[int(actor)] and key not in seen and len(targets)<min(8,64-counts[split]):
                        seen.add(key);targets.append(int(actor));identities.append({'Split':split,'SceneToken':c['scene_token'],
                            'SampleToken':c['sample_token'],'InstanceToken':c['instance_tokens'][int(actor)],'ActorNode':int(actor)})
                if targets:
                    targets=torch.tensor(targets);target_edge=interaction_edges(w,idx,keep,targets)
                    neutral[split].append(pack_targets(node,idx,keep,target_edge,selected,f,targets))
                    neutral_prob[split].append(c['mode_prob'][targets]);counts[split]+=len(targets)
            if len(audited)%20==0:print('INTEGRITY',len(audited),'/200','neutral',counts,flush=True)
    assert len(audited)==200 and sum(r['Split']=='train' for r in audited)==100 and counts=={'train':64,'val':64}
    arguments=[torch.cat([part[k] for split in ('train','val') for part in neutral[split]],dim=0).numpy() for k in range(7)]
    original_prob=torch.cat([p for split in ('train','val') for p in neutral_prob[split]]).numpy()
    atomic_npz(ROOT/'02_graph_cache/stage8a0c_neutral_arguments.npz',{**{f'arg{k}':a for k,a in enumerate(arguments)},'original_probability':original_prob})
    write_csv(ROOT/'06_tables/stage8a0c_integrity_windows.csv',audited);write_csv(ROOT/'06_tables/stage8a0c_neutral_actor_identities.csv',identities)
    # Selector and its project dependencies may not import any evaluation/error/label module.
    audited_sources=[ROOT/'01_selector/stage8a0c_selector.py',ROOT/'01_selector/stage8a0c_build.py',ROOT/'00_manifest/stage8a0c_common.py',STAGE8/'00_manifest/stage8a_graph.py',PROJECT/'preprocessing/coordinates.py']
    imports=[]
    for path in audited_sources:
        tree=ast.parse(path.read_text())
        for n in ast.walk(tree):
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
                for name in names:
                    assert not any(x in name.lower() for x in ('evaluation','fde','ade','ranking_label','best_mode','actor_errors'))
                    imports.append({'source':str(path.relative_to(PROJECT)),'module':name})
    atomic_json(ROOT/'03_model_audit/stage8a0c_integrity_audit.json',{'CandidateIdentity':'PASS','Coordinate':'PASS','NoFutureLeakage':'PASS',
        'SemanticAttachment':'PASS','SelectorUsesPredictionError':'NO','windows':200,'train_windows':100,'val_windows':100,
        'candidate_maxdiff':0,'original_logit_maxdiff':0,'original_probability_maxdiff':0,
        'geometry_distance_max_m':max(r['GeometryDistanceMaxM'] for r in audited),
        'candidate_roundtrip_max_m':max(r['CandidateRoundtripMaxM'] for r in audited),
        'polygon_roundtrip_max_m':max(r['PolygonRoundtripMaxM'] for r in audited),'semantic_edges_checked':semantic_edges,
        'neutral_real_unique_targets':128,'poisoned_GT':'NaN','poisoned_masks':'inverted',
        'forbidden_access_guard':True,'all_node_map_interaction_features_bitwise_unchanged':True,
        'unchanged_Stage8A0_neighbor_rule':True,'source_import_audit':imports,'training_executed':False,'test_used':False})
    print('INTEGRITY PASS',flush=True)

if __name__=='__main__':main()
