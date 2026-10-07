"""500 paired CUDA-event timings, alternating order, original baseline inputs."""
from stage7d_analysis_common import *
from stage3b_common import model_new as baseline_new
import copy
from torch_geometric.data import Batch

@torch.no_grad()
def main():
    model=model_new().eval();saved=torch.load(BEST,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['state_dict'],strict=True);del saved
    efficiency_path=ROOT/'06_tables/stage7d_efficiency.csv'
    if efficiency_path.exists():return
    baseline=baseline_new().eval();saved=torch.load(BASE_BEST,map_location='cpu',weights_only=False);baseline.load_state_dict(saved['state_dict']);del saved
    ds=Stage7DSemanticDataset('val');batches=[]
    for start in range(0,16*8,16):batches.append(Batch.from_data_list([ds[i] for i in range(start,start+16)]))
    ds.clear();timings={'Stage3B':[],'Stage7D':[]};peaks={'Stage3B':[],'Stage7D':[]};incremental={'Stage3B':[],'Stage7D':[]}
    for b in batches:
        data=b.cuda()
        base_data=copy.copy(data);del base_data.lane_semantic
        assert 'lane_semantic' in data and 'lane_semantic' not in base_data
        assert data.x.data_ptr()==base_data.x.data_ptr()
        for _ in range(3):baseline(base_data);model(data)
        del base_data,data
    torch.cuda.synchronize();torch.cuda.empty_cache()
    for pair in range(500):
        data=batches[pair%len(batches)].cuda();torch.cuda.synchronize()
        base_data=copy.copy(data);del base_data.lane_semantic
        assert 'lane_semantic' in data and 'lane_semantic' not in base_data
        assert data.x.data_ptr()==base_data.x.data_ptr()
        order=[('Stage3B',baseline,base_data),('Stage7D',model,data)]
        if pair%2:order.reverse()
        for name,m,working in order:
            torch.cuda.reset_peak_memory_stats();initial=torch.cuda.memory_allocated()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();out=m(working);end.record();end.synchronize()
            timings[name].append(float(start.elapsed_time(end)))
            peaks[name].append(torch.cuda.max_memory_allocated()/2**20)
            incremental[name].append((torch.cuda.max_memory_allocated()-initial)/2**20);del out
        del order,working,base_data,data
        if (pair+1)%100==0:print('PAIRED_LATENCY',pair+1,flush=True)
    count_base=sum(p.numel() for p in baseline.parameters());rows=[]
    for name,m in [('Stage3B',baseline),('Stage7D',model)]:
        count=sum(p.numel() for p in m.parameters());rows.append({'model':name,'parameters':count,'additional_parameters':count-count_base,
          'parameter_increase_percent':100*(count-count_base)/count_base,'paired_measurements':500,
          'mean_forward_ms':float(np.mean(timings[name])),'median_forward_ms':float(np.median(timings[name])),
          'peak_CUDA_allocated_MiB':float(max(peaks[name])),'peak_forward_increment_MiB':float(max(incremental[name])),
          'device':torch.cuda.get_device_name(),'batch_size':16,'unique_VAL_batches':8,
          'benchmark':'CUDA events; synchronized; alternating order; both models resident; baseline original fields and Stage7D augmented fields share original tensors; data loading excluded; warmed8batches3pairs'})
    write_csv(efficiency_path,rows)
    write_csv(ROOT/'06_tables/stage7d_efficiency_paired_measurements.csv',[
      {'pair':i+1,'batch':i%8,'Stage3B_ms':timings['Stage3B'][i],'Stage7D_ms':timings['Stage7D'][i]} for i in range(500)])
    print('EFFICIENCY_COMPLETE',rows,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
