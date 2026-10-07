"""All TRAIN/VAL modes: frozen-candidate references, neighbor selectors and map retrieval."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage8a_common import *
from stage8a_graph import observable_window,neighbors
from stage8a_map import MapIndex
from preprocessing.coordinates import ego_to_global

GROUPS=('Overall','Vehicle','Pedestrian','Bicycle','MovingVehicle')
POPULATIONS=('current_valid','full_horizon_ranking_targets')

def masks(w,selected):
    t=w['agent_type'][selected].numpy();moving=np.array([w['motion_state'][int(i)]=='vehicle.moving' for i in selected])
    return {'Overall':np.ones(len(selected),bool),'Vehicle':t==0,'Pedestrian':t==1,'Bicycle':t==2,'MovingVehicle':moving}

def main():
    verify_frozen();assert read_json(ROOT/'01_cache_audit/stage8a_coordinate_audit.json')['status']=='PASS'
    source=read_json(CACHE_MANIFEST);maps=MapIndex();progress=ROOT/'02_graph_cache/stage8a_graph_cache_manifest.json'
    prior=read_json(progress) if progress.exists() else {'batches':[]}
    done={r['path']:r for r in prior['batches']};records=[];stats={};distribution={};counts={};max_n=0
    started=time.perf_counter()
    for split in ('train','val'):
        ds=SceneDataset(split);population_count=0;full_count=0;window_count=0
        files=[r for r in source['batches'] if r['split']==split]
        for fi,record in enumerate(files):
            path=STAGE6/record['path'];assert sha256(path)==record['sha256']
            cache=torch.load(path,map_location='cpu',weights_only=False);name=f'stage8a_{split}_retrieval_{record["start"]:05d}.npz'
            out=ROOT/'02_graph_cache'/name;batch_stats={};batch_hist={};frames=[];node_offset=[0];all_arrays=[]
            # Resume reuses validated retrieval results; metrics are replayed from saved arrays.
            loaded=np.load(out) if name in done and out.exists() else None
            if loaded is not None:assert sha256(out)==done[name]['sha256']
            elapsed_start=time.perf_counter()
            for wi,w in enumerate(cache['windows']):
                g=ds[w['dataset_index']]
                assert (g.scene_token,g.sample_token,list(g.instance_tokens))==(w['scene_token'],w['sample_token'],w['instance_tokens'])
                obs=observable_window(w,g.map_location,g.origin.numpy(),float(g.ego_yaw));selected=np.flatnonzero(~w['history_padding'][:,4].numpy())
                n=len(selected);max_n=max(max_n,n);idx,keep=neighbors(obs)
                if loaded is None:
                    points=ego_to_global(w['ego_prediction'][selected].numpy().reshape(-1,12,2),obs.origin,obs.yaw)
                    ret=maps.retrieve(points,obs.map_location);region=maps.regions[obs.map_location]
                    ids=ret['region_token'].copy();valid=ids>=0;ids[valid]=region['ids'][ids[valid]]
                    arrays={'actor_node':selected.astype(np.int32),'map_token_index':ids.reshape(n,6,8),
                        'map_minimum_distance':ret['minimum_distance'].reshape(n,6,8),
                        'map_fallback':ret['fallback'].reshape(n,6),'within_radius_count':ret['within_radius_token_count'].reshape(n,6).astype(np.int32),
                        'neighbor_actor_node':idx[selected].numpy().astype(np.int32),'neighbor_mask':keep[selected].numpy()}
                    all_arrays.append(arrays)
                else:
                    lo,hi=loaded['node_offset'][wi:wi+2]
                    arrays={k:loaded[k][lo:hi] for k in ('actor_node','map_token_index','map_minimum_distance','map_fallback','within_radius_count','neighbor_actor_node','neighbor_mask')}
                    assert np.array_equal(arrays['actor_node'],selected)
                    assert np.array_equal(arrays['neighbor_actor_node'],idx[selected].numpy()) and np.array_equal(arrays['neighbor_mask'],keep[selected].numpy())
                assert np.isfinite(arrays['map_minimum_distance']).all()
                token_count=(arrays['map_token_index']>=0).sum(-1);near=arrays['within_radius_count'];fallback=arrays['map_fallback']
                assert np.array_equal(fallback,near==0) and (token_count>=1).all() and (token_count<=8).all()
                assert ((~fallback)&(arrays['map_minimum_distance'][:,:,0]<=10.+1e-9)|fallback).all()
                for token_row in arrays['map_token_index'].reshape(-1,8):
                    v=token_row[token_row>=0];assert len(v)==len(set(v.tolist()))
                full=(w['target_mask']&w['future_mask'].all(-1))[selected].numpy()
                assert int(full.sum())==int((w['target_mask']&w['future_mask'].all(-1)).sum())
                gm=masks(w,selected);neighbors_count=arrays['neighbor_mask'].sum(-1)
                for population in POPULATIONS:
                    pop=np.ones(n,bool) if population=='current_valid' else full
                    for group in GROUPS:
                        mask=pop&gm[group];key=(population,group);stat=batch_stats.setdefault(key,np.zeros(8,dtype=np.float64))
                        stat+=np.array([mask.sum(),mask.sum()*6,(near[mask]>=1).sum(),(near[mask]>=4).sum(),fallback[mask].sum(),
                            token_count[mask].sum(),neighbors_count[mask].sum(),((neighbors_count[mask]==0)).sum()])
                        hist=batch_hist.setdefault(key,np.zeros(9,dtype=np.int64));hist+=np.bincount(neighbors_count[mask],minlength=9)
                node_offset.append(node_offset[-1]+n);frames.append({'dataset_index':w['dataset_index'],'scene_token':w['scene_token'],
                    'sample_token':w['sample_token'],'location':obs.map_location,'origin':obs.origin.tolist(),'yaw':obs.yaw})
                population_count+=n;full_count+=int(full.sum());window_count+=1
            if loaded is None:
                arrays={k:np.concatenate([a[k] for a in all_arrays]) for k in all_arrays[0]}
                arrays['node_offset']=np.asarray(node_offset,dtype=np.int32)
                arrays['window_origin']=np.array([f['origin'] for f in frames],dtype=np.float64)
                arrays['window_yaw']=np.array([f['yaw'] for f in frames],dtype=np.float64)
                arrays['window_location']=np.array([list(maps.regions).index(f['location']) for f in frames],dtype=np.uint8)
                atomic_npz(out,arrays)
                item={'path':name,'sha256':sha256(out),'bytes':out.stat().st_size,'source_cache_path':record['path'],
                    'source_cache_sha256':record['sha256'],'split':split,'windows':len(frames),
                    'dataset_indices':[x['dataset_index'] for x in frames],'construction_seconds':time.perf_counter()-elapsed_start}
            else:item=done[name];loaded.close()
            for key,v in batch_stats.items():stats.setdefault((split,*key),np.zeros(8))[:]+=v
            for key,v in batch_hist.items():distribution.setdefault((split,*key),np.zeros(9,dtype=np.int64))[:]+=v
            records.append(item)
            atomic_json(progress,{'status':'RUNNING','batches':records,'training_executed':False,'test_used':False})
            if (fi+1)%20==0 or fi+1==len(files):print('GRAPH_COVERAGE',split,fi+1,'/',len(files),'current_actors',population_count,'full',full_count,'seconds',round(time.perf_counter()-started,1),flush=True)
        assert population_count==source['counts'][split]['current_valid'] and full_count==source['counts'][split]['full']
        counts[split]={'current_valid_actors':population_count,'current_valid_candidates':population_count*6,
            'full_target_actors':full_count,'full_target_candidates':full_count*6,'windows':window_count,'scenes':source['counts'][split]['scenes']}
        ds.clear()
    coverage=[];graph=[];hist=[]
    for (split,pop,group),s in stats.items():
        a,c,one,four,fallback,tokens,ng,zero=s
        coverage.append({'Split':split,'Population':pop,'Group':group,'Actors':int(a),'Candidates':int(c),
            'AtLeast1Within10m':int(one),'AtLeast1Within10mRate':float(one/c) if c else 0.,'AtLeast4Within10mRate':float(four/c) if c else 0.,
            'FallbackCandidates':int(fallback),'FallbackRate':float(fallback/c) if c else 0.,'RetrievedAtLeast1IncludingFallbackRate':1. if c else 0.,
            'MeanMapTokensPerMode':float(tokens/c) if c else 0.})
        graph.append({'Split':split,'Population':pop,'Group':group,'Targets':int(a),'MeanNeighborActors':float(ng/a) if a else 0.,
            'MeanLocalModeNodes':float(6+6*ng/a) if a else 0.,'MeanModeModeEdgesPerTarget':float(36*ng/a) if a else 0.,
            'MeanMapNodesPerMode':float(tokens/c) if c else 0.,'MeanModeMapEdgesPerTarget':float(tokens/a) if a else 0.,
            'ZeroNeighborTargetRate':float(zero/a) if a else 0.,'FallbackRate':float(fallback/c) if c else 0.})
        for j,n in enumerate(distribution[split,pop,group]):hist.append({'Split':split,'Population':pop,'Group':group,'Neighbors':j,'Targets':int(n)})
    write_csv(ROOT/'06_tables/stage8a_map_coverage.csv',coverage);write_csv(ROOT/'06_tables/stage8a_graph_statistics.csv',graph)
    write_csv(ROOT/'06_tables/stage8a_neighbor_distribution.csv',hist)
    atomic_json(ROOT/'02_graph_cache/stage8a_map_token_dictionary.json',{'source_semantics_sha256':sha256(SEMANTICS),
        'source_centerlines_sha256':sha256(CENTERLINES),'token_count':len(maps.token_dictionary),'tokens':maps.token_dictionary})
    atomic_json(progress,{'status':'PASS','batches':records,'counts':counts,'total_graph_cache_bytes':sum(x['bytes'] for x in records),
        'candidate_tensors_copied':False,'cache_contents':'GT-free distinct map token retrieval and Stage6A-identical neighbor selectors; mode/edge features computed on demand from same source candidates',
        'cache_has_GT':False,'source_cache_manifest_sha256':sha256(CACHE_MANIFEST),'maximum_current_valid_actors_per_window':max_n,
        'all_source_batch_SHA_verified':True,'all_source_scene_sample_instance_ids_checked':True,'test_used':False,'training_executed':False})
    print('STAGE8A_COVERAGE_PASS',counts,'bytes',sum(x['bytes'] for x in records),flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
