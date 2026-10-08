"""Small frozen-head timing audit on predeclared TRAIN tensors; no model fitting."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage10a_common import *
from stage10a_model import DualExpert, Expert

@torch.no_grad()
def main():
    seed(); verify(); assert read_json(FROZEN)['status']=='FROZEN_ALL_COMPLETE'
    models={e:fresh(e) for e in EXPERTS}
    for e,m in models.items():
        cp=torch.load(ROOT/'02_checkpoints'/CPNAMES[e],map_location='cpu',weights_only=False)
        m.load_state_dict(cp['state_dict']); m.eval().requires_grad_(False)
    r2=frozen_r2(); g1=Expert(2022).cuda()
    g1.load_state_dict(torch.load(G1PATH,map_location='cpu',weights_only=False)['state_dict']); g1.eval().requires_grad_(False)
    dual=DualExpert(models['vehicle'],models['pedestrian'],r2).eval()
    store=Store(); rng=np.random.default_rng(2022)
    ids=np.concatenate([rng.choice(np.flatnonzero((store.partition==1)&(store.types==t)),n,replace=False)
        for t,n in enumerate((64,48,16))])
    args,_,_=store.batch(ids); feat=store.r2_batch(ids)
    probe=args[3].new_zeros((128,6,12,2))
    states=[state_sha(m) for m in (models['vehicle'],models['pedestrian'],r2,g1)]
    functions={'R2':lambda:r2(feat,args[3]),'G1':lambda:g1(*args),
        'DualExpert':lambda:dual(*args,feat,probe)}
    rows=[]
    for name,fn in functions.items():
        for _ in range(50): fn()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        baseline=torch.cuda.memory_allocated(); times=[]
        for _ in range(200):
            a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); b.synchronize(); times.append(a.elapsed_time(b))
        rows.append({'Model':name,'TargetBatch':128,'VehicleTargets':64,'PedestrianTargets':48,'BicycleTargets':16,
            'Parameters':48805 if name=='DualExpert' else 673 if name=='R2' else 24066,
            'Stage10TrainableExpertParameters':48132 if name=='DualExpert' else 0,
            'FrozenParameters':673 if name in ('R2','DualExpert') else 24066,
            'Mean_ms':float(np.mean(times)),'Median_ms':float(np.median(times)),'P95_ms':float(np.percentile(times,95)),
            'PeakCUDA_MiB':torch.cuda.max_memory_allocated()/1048576.,
            'IncrementalCUDA_MiB':(torch.cuda.max_memory_allocated()-baseline)/1048576.,
            'Warmup':50,'MeasuredForwards':200,
            'Scope':'reranker forward on normalized CUDA tensors; DualExpert includes frozen R2 plus type routing; excludes predictor, graph/feature construction and transfer'})
    assert states==[state_sha(m) for m in (models['vehicle'],models['pedestrian'],r2,g1)]
    write_csv(ROOT/'06_tables/stage10a_efficiency.csv',rows)
    atomic_json(ROOT/'08_efficiency/stage10a_efficiency_audit.json',{'status':'PASS','device':torch.cuda.get_device_name(0),
        'precision':'FP32','partition':'HeadTrain only','sampling_seed':2022,'counts':[64,48,16],'measured':200,
        'parameters_each':24066,'independent_expert_total_parameters':48132,'frozen_R2_parameters':673,
        'combined_reranker_parameters':48805,'models_unchanged':True,'end_to_end_latency_claim':False,
        'official_VAL_opened':False,'test_opened':False})
    print('STAGE10_EFFICIENCY_PASS',rows,flush=True)

if __name__=='__main__': main()
