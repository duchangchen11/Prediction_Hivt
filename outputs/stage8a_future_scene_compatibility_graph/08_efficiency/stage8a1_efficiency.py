"""Separate RTX3080 head forward, live CPU graph work, live predictor forward."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage8a1_common import *
sys.path.insert(0,str(ROOT/'03_evaluation'))
from stage8a1_evaluate import loaded
sys.path[:0]=[str(S6/'00_manifest'),str(S6/'01_cache')]
from stage6a_common import frozen_predictor,SceneDataset,model_input
from stage6a_features import observable_features,normalize as r2normalize
from torch_geometric.data import Batch

def timed_cuda(fn,warm=100,repeats=500):
    for _ in range(warm):fn()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();baseline=torch.cuda.memory_allocated();ms=[]
    for _ in range(repeats):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();fn();end.record();end.synchronize();ms.append(start.elapsed_time(end))
    return {'Mean_ms':float(np.mean(ms)),'Median_ms':float(np.median(ms)),'P95_ms':float(np.percentile(ms,95)),
        'PeakCUDA_MiB':torch.cuda.max_memory_allocated()/2**20,'IncrementalCUDA_MiB':(torch.cuda.max_memory_allocated()-baseline)/2**20,
        'Warmup':warm,'MeasuredForwards':repeats}

def main():
    seed();assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3080';models=loaded()
    store=Store('train');ids=np.flatnonzero(store.partition==1)[:128];args,_,_=store.batch(ids)
    r2norm=read_json(S6/'02_features/stage6a_normalization.json');records=read_json(C/'02_graph_cache/stage8a0c_selector_manifest.json')['batches']
    # Find the same first128 HeadTrain identities, with all neighbors/context intact.
    identities=[]
    with (ROOT/'01_training/cache/identities.csv').open() as f:
        allids=list(csv.DictReader(f));identities=[allids[int(i)] for i in ids]
    desired={}
    for row in identities:desired.setdefault(int(row['dataset_index']),[]).append(int(row['node_in_graph']))
    chosen=[];rx=[];index=SparseSemanticIndex()
    for rec in records:
        if rec['split']!='train':continue
        cache=torch.load(S6/rec['source_path'],map_location='cpu',weights_only=False)
        with np.load(ROOT/'02_graph_cache'/rec['frame_path']) as frame:
            for j,w in enumerate(cache['windows']):
                di=w['dataset_index']
                if di not in desired:continue
                targets=torch.tensor(desired[di]);loc=sorted(index.regions)[int(frame['window_location'][j])]
                obs=observable_window(w,loc,frame['window_origin'][j],float(frame['window_yaw'][j]));chosen.append((obs,targets))
                x,flags,_,_=observable_features(*[w[k].cuda() for k in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')])
                rx.append(r2normalize(x,flags,r2norm,'R2')[targets])
        if sum(len(t) for _,t in chosen)==128:break
    rx=torch.cat(rx);assert rx.shape==(128,6,19)
    rows=[]
    for v in ('R2','G1','G2','G3'):
        fn=(lambda:models[v](rx,args[-1])) if v=='R2' else (lambda:models[v](*args))
        rows.append({'Component':'A reranker forward','Model':v,'TargetBatch':128,'SceneWindows':'mixed frozen inputs',
            'Params':673 if v=='R2' else PARAMS[v],**timed_cuda(fn),'Scope':'normalized tensors already on CUDA; excludes feature construction, retrieval and predictor'})
    def graph_work():
        result=[]
        for obs,targets in chosen:
            node=node_features(obs);idx,keep=neighbors(obs);edge=interaction_edges(obs,idx,keep,targets)
            local=obs.predicted[targets].numpy().reshape(-1,12,2);points=ego_to_global(local,obs.origin,obs.yaw)
            selected=index.select_batch(np.repeat(obs.actor_type[targets].numpy(),6),points,obs.map_location)
            delta=local[:,-1]-local[:,-3];feature=index.features(points,obs.map_location,obs.yaw,np.arctan2(delta[:,1],delta[:,0]),selected)
            feature={k:x.reshape((len(targets),6)+x.shape[1:]) for k,x in feature.items()}
            feature['map_mask']=selected['map_mask'].reshape(len(targets),6,8)
            result.append(normalize_args(pack_targets(node,idx,keep,edge,targets.numpy(),feature,targets),store.stats))
        return result
    for _ in range(2):graph_work()
    ms=[]
    for _ in range(10):start=time.perf_counter();graph_work();ms.append((time.perf_counter()-start)*1000)
    rows.append({'Component':'B live CPU graph+retrieval','Model':'G3 features','TargetBatch':128,'SceneWindows':len(chosen),'Params':0,
        'Mean_ms':float(np.mean(ms)),'Median_ms':float(np.median(ms)),'P95_ms':float(np.percentile(ms,95)),
        'PeakCUDA_MiB':0.,'IncrementalCUDA_MiB':0.,'Warmup':2,'MeasuredForwards':10,
        'Scope':'live selector, all neighbor-mode edges, GEOS semantic features, normalization; static HD map index already loaded; frozen predicted candidates supplied; CPU→CUDA transfer excluded'})
    del models,args,rx;torch.cuda.empty_cache()
    predictor=frozen_predictor();ds=SceneDataset('train');graphs=[ds[i] for i in range(16)];data=Batch.from_data_list(graphs).cuda();inp=model_input(data)
    fn=lambda:predictor(inp)
    full=sum(int(g.full_horizon_mask.sum()) for g in graphs)
    rows.append({'Component':'C frozen predictor forward','Model':'Stage5A','TargetBatch':full,'SceneWindows':16,
        'Params':sum(p.numel() for p in predictor.parameters()),**timed_cuda(fn),
        'Scope':'original16 full scene-window batch, all actors and lanes; CUDA input prepared; excludes dataset loading and batch construction; different unit from128-target reranker'})
    write_csv(ROOT/'06_tables/stage8a1_efficiency.csv',rows)
    atomic_json(ROOT/'08_efficiency/stage8a1_efficiency_audit.json',{'status':'PASS','device':torch.cuda.get_device_name(),
        'heads_batch128_warmup100_measured500':True,'cache_not_online_latency':True,'graph_measurements':10,
        'end_to_end_claim':False,'B_cpu_threads':torch.get_num_threads(),'C_actual_scene_windows':16,'C_actual_full_targets':full})
    ds.clear();print('EFFICIENCY_PASS',rows,flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
