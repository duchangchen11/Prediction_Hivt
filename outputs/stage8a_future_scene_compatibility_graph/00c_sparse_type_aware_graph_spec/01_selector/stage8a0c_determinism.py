"""Three independent replay orders/worker counts on the registered 200 windows."""
from pathlib import Path
import sys,multiprocessing,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/d) for d in ('00_manifest','01_selector')]
from stage8a0c_common import *
from stage8a0c_selector import SparseSemanticIndex

def initialize():
    global maps
    torch.set_num_threads(1);maps=SparseSemanticIndex()
def replay(chunk):
    output=[]
    for key,points,location,types,order in chunk:
        result=maps.select_batch(types[order],points[order],location);restore=np.argsort(order)
        output.append((key,{k:v[restore] for k,v in result.items()}))
    return output

def main():
    index=SparseSemanticIndex();torch.set_num_threads(1)
    samples=list(csv.DictReader((STAGE0B/'06_tables/stage8a0b_integrity_windows.csv').open()))
    wanted={(r['Split'],int(r['DatasetIndex'])) for r in samples}
    jobs=[]
    for b in read_json(STAGE8/'02_graph_cache/stage8a_graph_cache_manifest.json')['batches']:
        js=[j for j,i in enumerate(b['dataset_indices']) if (b['split'],i) in wanted]
        if not js:continue
        source=STAGE6/b['source_cache_path'];assert sha256(source)==b['source_cache_sha256']
        cached=torch.load(source,map_location='cpu',weights_only=False)
        with np.load(STAGE8/'02_graph_cache'/b['path']) as z:
            for j in js:
                c=cached['windows'][j];selected=torch.where(~c['history_padding'][:,4])[0].numpy()
                location=list(index.regions)[int(z['window_location'][j])]
                points=ego_to_global(c['ego_prediction'][selected].numpy().reshape(-1,12,2),z['window_origin'][j],float(z['window_yaw'][j]))
                jobs.append(((b['split'],c['dataset_index']),points,location,np.repeat(c['agent_type'][selected].numpy(),6)))
    assert len(jobs)==200
    golden={};rows=[];rng=np.random.default_rng(2022)
    for pass_number,workers,chunk_size in [(1,1,1),(2,2,4),(3,3,7)]:
        ordering=np.arange(len(jobs)) if pass_number==1 else (np.arange(len(jobs))[::-1] if pass_number==2 else rng.permutation(len(jobs)))
        arranged=[]
        for j in ordering:
            key,points,location,types=jobs[int(j)]
            order=np.arange(len(points)) if pass_number==1 else (np.arange(len(points))[::-1] if pass_number==2 else rng.permutation(len(points)))
            arranged.append((key,points,location,types,order))
        chunks=[arranged[i:i+chunk_size] for i in range(0,len(arranged),chunk_size)]
        received={}
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(replay,chunk) for chunk in chunks]
            for f in as_completed(futures):
                for key,result in f.result():received[key]=result
        assert len(received)==200
        if pass_number==1:golden=received
        else:
            for key,result in received.items():
                for field,value in result.items():assert np.array_equal(value,golden[key][field]),(key,field)
        digest=hashlib.sha256();candidate_count=0
        for key,result in sorted(received.items()):
            digest.update(str(key).encode())
            for field in ('entity_ids','entity_types','geometry_distances'):digest.update(result[field].tobytes())
            candidate_count+=len(result['entity_ids'])
        rows.append({'Run':pass_number,'Workers':workers,'ChunkWindows':chunk_size,'WindowOrdering':('ascending','reverse','seed2022_shuffle')[pass_number-1],
            'CandidateOrdering':('ascending','reverse','seed2022_shuffle')[pass_number-1],'Windows':200,'Candidates':candidate_count,
            'SelectorSHA256':digest.hexdigest(),'BitwiseIdentical':'PASS'})
        print('DETERMINISM RUN',pass_number,'PASS',digest.hexdigest(),flush=True)
    assert len({r['SelectorSHA256'] for r in rows})==1
    write_csv(ROOT/'06_tables/stage8a0c_determinism_runs.csv',rows)
    atomic_json(ROOT/'01_selector/stage8a0c_determinism_audit.json',{'SelectorDeterminism':'PASS','runs':rows,
        'checks':'every selected ID/type/distance, count, mask and comparison selector bitwise equal after restoring row order',
        'selected_tokens_order_equal':True,'worker_counts':[1,2,3],'sample_seed':2022})

if __name__=='__main__':main()
