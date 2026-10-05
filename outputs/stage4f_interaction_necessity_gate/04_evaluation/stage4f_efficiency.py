"""500 interleaved paired measurements; isolated three-model peak memory."""
from pathlib import Path
import itertools
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'));sys.path.insert(0,str(ROOT/'04_evaluation'))
from stage4f_common import (canonical_stage3b_model, canonical_stage4a_model,
    model_input, SceneDataset, atomic_json, write_csv, sha256, OLD, STAGE3_ROOT, BEST)
from stage4f_evaluate import load_final_model
from torch_geometric.data import Batch
import torch
import numpy as np


def networks():
    b=canonical_stage3b_model();c=canonical_stage4a_model();d,_,_=load_final_model()
    b.load_state_dict(torch.load(STAGE3_ROOT/'07_checkpoints/stage3b_best_overall_minfde.pt',map_location='cpu',weights_only=False)['state_dict'])
    c.load_state_dict(torch.load(OLD/'07_checkpoints/stage4a_best_overall_minfde.pt',map_location='cpu',weights_only=False)['state_dict'])
    models={'Stage3B':b.eval(),'Stage4A':c.eval(),'Stage4F':d.eval()}
    d.global_interactor.record_necessity_gate=False;d.global_interactor.record_relation_bias=False
    return models


@torch.no_grad()
def main():
    models=networks();ds=SceneDataset('val');indices=[list(range(s,min(s+16,len(ds)))) for s in range(0,len(ds),16)]
    assert len(ds)==3603 and len(indices)==226
    def batch(i):return Batch.from_data_list([ds[j] for j in indices[i]]).cuda()
    for rep in range(20):
        data=batch(rep)
        for model in models.values():model(model_input(data))
        del data
    torch.cuda.synchronize();records=[]
    orders=list(itertools.permutations(models))
    for rep in range(500):
        index=rep%len(indices);data=batch(index)
        row={'paired_batch_index':rep,'VAL_batch_index':index,'graphs':data.num_graphs,'nodes':data.num_nodes,'order':'/'.join(orders[rep%6])}
        for name in orders[rep%6]:
            model=models[name];working=model_input(data);torch.cuda.synchronize()
            before=int(torch.cuda.memory_allocated());torch.cuda.reset_peak_memory_stats()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();out=model(working);end.record();end.synchronize()
            elapsed=float(start.elapsed_time(end));assert elapsed>0 and torch.isfinite(out['raw_prediction']).all()
            row[name+'_ms']=elapsed;row[name+'_forward_incremental_bytes']=int(torch.cuda.max_memory_allocated())-before
            del out,working
        records.append(row);del data
        if (rep+1)%100==0:print('EFFICIENCY_TRIPLETS',rep+1,'/500',flush=True)
    for model in models.values():model.cpu()
    torch.cuda.synchronize();torch.cuda.empty_cache();memory={}
    for name,model in models.items():
        model.cuda()
        for rep in range(20):
            data=batch(rep);out=model(model_input(data));del data,out
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        for index in range(len(indices)):
            data=batch(index);out=model(model_input(data));assert torch.isfinite(out['raw_prediction']).all();del data,out
        torch.cuda.synchronize()
        memory[name]={'allocated_bytes':int(torch.cuda.max_memory_allocated()),'reserved_bytes':int(torch.cuda.max_memory_reserved()),'unique_VAL_batches':226,'GPU_model_count':1}
        model.cpu();torch.cuda.synchronize();torch.cuda.empty_cache()
    ds.clear();rows=[]
    for name,model in models.items():
        values=np.array([r[name+'_ms'] for r in records])
        rows.append({'Model':name,'parameters':sum(p.numel() for p in model.parameters()),'mean_inference_ms':float(values.mean()),
            'median_inference_ms':float(np.median(values)),'std_inference_ms':float(values.std(ddof=1)),
            'paired_measurements':500,'warmup_batches':20,'peak_CUDA_memory_MiB':memory[name]['allocated_bytes']/2**20,
            'peak_reserved_MiB':memory[name]['reserved_bytes']/2**20})
    assert [r['parameters'] for r in rows]==[646001,647609,647786]
    write_csv(ROOT/'06_tables/stage4f_efficiency.csv',rows);write_csv(ROOT/'06_tables/stage4f_efficiency_batch_timings.csv',records)
    audit={'status':'PASS','GPU':torch.cuda.get_device_name(),'torch':torch.__version__,'CUDA':torch.version.cuda,
        'paired_measurements':500,'execution_order':'cycle through all6 permutations of B/C/D, same preloaded GPU batch per triplet',
        'timing':'CUDA events around eval forward; I/O, H2D and external input clone excluded; actual model-internal clone included equally',
        'memory':'separate untimed single-GPU-model pass; warm20, empty_cache, reset_peak, all226 VAL batches',
        'models':rows,'isolated_memory':memory,'gate_parameters':177,
        'mean_overhead_D_vs_B_percent':100*(rows[2]['mean_inference_ms']/rows[0]['mean_inference_ms']-1),
        'mean_overhead_D_vs_C_percent':100*(rows[2]['mean_inference_ms']/rows[1]['mean_inference_ms']-1),
        'source_csv_sha256':sha256(ROOT/'06_tables/stage4f_efficiency_batch_timings.csv')}
    atomic_json(ROOT/'04_evaluation/stage4f_efficiency_audit.json',audit);print('EFFICIENCY_PASS',audit,flush=True)


if __name__=='__main__':torch.set_num_threads(4);main()
