"""RTX3080 FP32 forward timing; normalized CUDA inputs, no end-to-end claim."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage9a_common import *

@torch.no_grad()
def measure(fn,params,name,boundary,includes_r2=False):
    for _ in range(100):fn()
    torch.cuda.synchronize();allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
    for _ in range(500):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();fn();b.record();b.synchronize();times.append(a.elapsed_time(b))
    x=np.array(times)
    return {'Boundary':boundary,'Model':name,'BatchTargets':128,'Params':params,'IncludesFrozenR2Forward':int(includes_r2),'FrozenR2Params':673 if name in VARIANTS else 0,'MeanMs':float(x.mean()),'MedianMs':float(np.median(x)),'P95Ms':float(np.percentile(x,95)),'PeakMiB':torch.cuda.max_memory_allocated()/1048576,'IncrementalMiB':(torch.cuda.max_memory_allocated()-allocated)/1048576,'Warmup':100,'Measured':500,'Scope':('includes frozen R2 head plus graph residual; ' if includes_r2 else 'R2 logits prepared for graph-residual-only rows; ' if boundary=='graph residual only' else '')+'FP32 normalized CUDA feature tensors prepared; excludes predictor, feature construction and transfer; gate-only cost is included in graph and combined rows'}

@torch.no_grad()
def main():
    seed();verify();assert read_json(ROOT/'03_evaluation/stage9a_complete.json')['status']=='PASS'
    store=Store();ids=np.flatnonzero(store.partition==1)[:128];args,_,_=store.batch(ids);rows=[]
    data=torch.load(S6/'02_features/stage6a_headtrain_features.pt',map_location='cpu',weights_only=False)
    assert np.array_equal(data['FDE_by_mode'][:128].numpy(),store.fde[ids]) and np.array_equal(data['base_logits'][:128].numpy(),store.args[0][ids,0,:,3])
    rx=data['features_R2'][:128].cuda();base=data['base_logits'][:128].cuda();del data
    r2=frozen_r2();rows.append(measure(lambda:r2(rx,base),673,'R2','reranker forward'))
    g1=SparseGraphReranker('G1').cuda();p=S8/'01_training/G1/formal/best_dev_loss.pt';g1.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state_dict']);g1.eval().requires_grad_(False)
    orig=torch.from_numpy(np.array(store.args[0][ids,0,:,3],copy=True)).cuda();ra=(*args[:3],torch.zeros(128,6,8,18,device='cuda'),torch.zeros(128,6,8,11,device='cuda'),torch.zeros(128,6,8,dtype=torch.bool,device='cuda'),orig)
    rows.append(measure(lambda:g1(*ra),24066,'G1','reranker forward'));del g1,ra
    for v in VARIANTS:
        m=fresh(v);m.load_state_dict(torch.load(ROOT/'02_checkpoints'/f'{v}_best.pt',map_location='cpu',weights_only=False)['state_dict']);m.eval().requires_grad_(False)
        rows.append(measure(lambda:m(*args[:3],r2(rx,base)['mode_logits'],args[4]),PARAMS[v],v,'combined R2+graph reranker',True))
        rows.append(measure(lambda:m(*args),PARAMS[v],v,'graph residual only'))
        if v!='T1':rows.append(measure(lambda:m.gate_network(args[4]),161,v,'gate only'))
        del m;torch.cuda.empty_cache()
    write_csv(ROOT/'06_tables/stage9a_efficiency.csv',rows)
    atomic_json(ROOT/'08_efficiency/stage9a_efficiency_audit.json',{'status':'PASS','device':torch.cuda.get_device_name(0),'precision':'FP32','targets':128,'warmup':100,'measured':500,'semantic_retrieval':False,'Stage9_combined_includes_frozen_R2_forward':True,'residual_only_and_gate_costs_separately_labeled':True,'end_to_end_latency_claim':False})
    print('STAGE9_EFFICIENCY_PASS',flush=True)

if __name__=='__main__':main()
