"""Actual learned semantic residual and 500 paired forward measurements."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7a_formal_common import *
from stage3b_common import model_new as baseline_new
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch

@torch.no_grad()
def main():
    model=model_new().eval();saved=torch.load(BEST,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict']);del saved
    encoder=model.local_encoder.al_encoder.semantic_mlp
    stats_path=ROOT/'06_tables/stage7a_semantic_residual_statistics.csv'
    if not stats_path.exists():
        # Frequency over actual VAL graph lane segments (one occurrence per stored segment/window).
        ds=Stage7ASemanticDataset('val');patterns={}
        for i in range(len(ds)):
            s=ds[i].lane_semantic.numpy();unique,counts=np.unique(s,axis=0,return_counts=True)
            for u,c in zip(unique,counts):key=tuple(u.tolist());patterns[key]=patterns.get(key,0)+int(c)
            if (i+1)%1000==0:print('RESIDUAL_INPUTS',i+1,flush=True)
        ds.clear();p=np.array(list(patterns),dtype=np.float32);counts=np.array(list(patterns.values()))
        values=encoder(torch.from_numpy(p).cuda()).cpu().numpy();norms=np.linalg.norm(values,axis=1)
        predicates={'ordinary_lane':p[:,0]==0,'connector':p[:,0]==1,'left':p[:,1]==1,'straight':p[:,2]==1,
          'right':p[:,3]==1,'unknown_connector':p[:,4]==1,'traffic_light':p[:,5]==1,'stop_sign':p[:,6]==1,
          'other_control':p[:,7]==1,'crosswalk':p[:,8]==1,'non_crosswalk':p[:,8]==0,'all':np.ones(len(p),bool)}
        rows=[]
        for group,mask in predicates.items():
            n=counts[mask].sum();v=norms[mask];w=counts[mask];mean=np.average(v,weights=w)
            order=np.argsort(v);cumulative=np.cumsum(w[order]);middle=(n-1)/2
            median=float(v[order[np.searchsorted(cumulative,middle+1)]])
            if n%2==0:median=(median+float(v[order[np.searchsorted(cumulative,n/2+1)]]))/2
            rows.append({'group':group,'VAL_segment_occurrences':int(n),'unique_semantic_patterns':int(mask.sum()),
              'mean_norm':float(mean),'median_norm':median,'std_norm':float(np.sqrt(np.average((v-mean)**2,weights=w))),
              'min_norm':float(v.min()),'max_norm':float(v.max()),'weighting':'stored VAL lane segment/window occurrences'})
        write_csv(stats_path,rows)
        atomic_json(ROOT/'04_evaluation/stage7a_semantic_residual_audit.json',{'checkpoint_sha256':sha256(BEST),
          'all_features_zero_residual_norm':float(encoder(torch.zeros((1,9),device='cuda')).norm()),
          'semantic_last_weight_norm':float(encoder[2].weight.norm()),'semantic_last_bias_norm':float(encoder[2].bias.norm()),
          'near_zero_threshold':1e-6,'effectively_unused':float(np.average(norms,weights=counts))<1e-6,
          'meaning':'feature-conditioned learned representation norm, not physical effect or causal influence'})
    efficiency_path=ROOT/'06_tables/stage7a_efficiency.csv'
    if efficiency_path.exists():return
    baseline=baseline_new().eval();saved=torch.load(BASE_BEST,map_location='cpu',weights_only=False);baseline.load_state_dict(saved['state_dict']);del saved
    ds=Stage7ASemanticDataset('val');batches=[]
    for start in range(0,16*8,16):batches.append(Batch.from_data_list([ds[i] for i in range(start,start+16)]))
    ds.clear();timings={'Stage3B':[],'Stage7A':[]};peaks={'Stage3B':[],'Stage7A':[]};incremental={'Stage3B':[],'Stage7A':[]}
    for b in batches:
        data=b.cuda()
        for _ in range(3):baseline(data);model(data)
        del data
    torch.cuda.synchronize();torch.cuda.empty_cache()
    for pair in range(500):
        data=batches[pair%len(batches)].cuda();torch.cuda.synchronize()
        order=[('Stage3B',baseline),('Stage7A',model)]
        if pair%2:order.reverse()
        for name,m in order:
            torch.cuda.reset_peak_memory_stats();initial=torch.cuda.memory_allocated()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();out=m(data);end.record();end.synchronize()
            timings[name].append(float(start.elapsed_time(end)))
            peaks[name].append(torch.cuda.max_memory_allocated()/2**20)
            incremental[name].append((torch.cuda.max_memory_allocated()-initial)/2**20);del out
        del data
        if (pair+1)%100==0:print('PAIRED_LATENCY',pair+1,flush=True)
    count_base=sum(p.numel() for p in baseline.parameters());rows=[]
    for name,m in [('Stage3B',baseline),('Stage7A',model)]:
        count=sum(p.numel() for p in m.parameters());rows.append({'model':name,'parameters':count,'additional_parameters':count-count_base,
          'parameter_increase_percent':100*(count-count_base)/count_base,'paired_measurements':500,
          'mean_forward_ms':float(np.mean(timings[name])),'median_forward_ms':float(np.median(timings[name])),
          'peak_CUDA_allocated_MiB':float(max(peaks[name])),'peak_forward_increment_MiB':float(max(incremental[name])),
          'device':torch.cuda.get_device_name(),'batch_size':16,'unique_VAL_batches':8,
          'benchmark':'CUDA events; synchronized; alternating method order; both models resident; data loading excluded; warmed8batches3pairs'})
    write_csv(efficiency_path,rows)
    write_csv(ROOT/'06_tables/stage7a_efficiency_paired_measurements.csv',[
      {'pair':i+1,'batch':i%8,'Stage3B_ms':timings['Stage3B'][i],'Stage7A_ms':timings['Stage7A'][i]} for i in range(500)])
    print('EFFICIENCY_COMPLETE',rows,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
