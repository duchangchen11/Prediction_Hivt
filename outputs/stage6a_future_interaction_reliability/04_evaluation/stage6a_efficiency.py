"""Feature extraction + head scoring overhead on identical preloaded windows."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage6a_common import *
from stage6a_evaluate import load_head
from stage6a_features import observable_features,normalize
import numpy as np

@torch.no_grad()
def main():
    verify_frozen();heads={v:load_head(v)[0] for v in ('R1','R2')};norm=read_json(NORM)
    manifest=read_json(ROOT/'01_cache/stage6a_cache_manifest.json');records=[r for r in manifest['batches'] if r['split']=='val']
    chosen=[records[i] for i in np.linspace(0,len(records)-1,20,dtype=int)]
    batches=[]
    for record in chosen:
        payload=torch.load(ROOT/record['path'],map_location='cpu',weights_only=False)
        windows=[]
        for w in payload['windows']:
            inputs=[w[name].cuda() for name in ('history','history_padding','agent_type','ego_prediction','mode_logits','mode_prob')]
            windows.append(inputs)
        batches.append(windows)
    def execute(variant,windows,events=False):
        start,feature_end,end=[torch.cuda.Event(enable_timing=True) for _ in range(3)]
        prepared=[];start.record()
        for inputs in windows:
            x,flags,_,_=observable_features(*inputs,interaction=variant=='R2')
            prepared.append((normalize(x,flags,norm,variant),inputs[4]))
        feature_end.record()
        for x,logits in prepared:out=heads[variant](x,logits)
        end.record();end.synchronize()
        assert torch.isfinite(out['mode_prob']).all()
        return float(start.elapsed_time(feature_end)),float(feature_end.elapsed_time(end)),float(start.elapsed_time(end))
    for rep in range(20):
        for variant in ('R1','R2'):execute(variant,batches[rep])
    timings=[]
    for rep in range(200):
        windows=batches[rep%len(batches)];order=('R1','R2') if rep%2==0 else ('R2','R1')
        row={'Measurement':rep,'preloaded_VAL_batch':rep%len(batches),'windows':len(windows),'order':'/'.join(order)}
        for variant in order:
            features,head,total=execute(variant,windows)
            row.update({variant+'_feature_ms':features,variant+'_head_ms':head,variant+'_total_ms':total})
        timings.append(row)
        if (rep+1)%50==0:print('RANKING_EFFICIENCY',rep+1,'/200',flush=True)
    rows=[]
    for variant in ('R1','R2'):
        total=np.array([r[variant+'_total_ms'] for r in timings])
        rows.append({'Variant':variant,'Predictor_params':650403,'Head_params':673,'Total_params':651076,
            'Parameter_increase_percent':100*673/650403,'Mean_feature_extraction_ms':float(np.mean([r[variant+'_feature_ms'] for r in timings])),
            'Mean_head_scoring_ms':float(np.mean([r[variant+'_head_ms'] for r in timings])),
            'Mean_ranking_overhead_ms':float(total.mean()),'Median_ranking_overhead_ms':float(np.median(total)),
            'Std_ranking_overhead_ms':float(total.std(ddof=1)),'Measurements':200})
    write_csv(ROOT/'06_tables/stage6a_efficiency.csv',rows);write_csv(ROOT/'06_tables/stage6a_efficiency_timings.csv',timings)
    atomic_json(ROOT/'04_evaluation/stage6a_efficiency_audit.json',{'status':'PASS','GPU':torch.cuda.get_device_name(),'torch':torch.__version__,
        'paired_measurements':200,'preloaded_original_VAL_batches':20,'warmup_pairs':20,'models':rows,
        'scope':'additional feature extraction + scoring only; excludes predictor forward, I/O and H2D',
        'R1_interaction_distances_computed':False,'R2_neighbor_features_computed':True,
        'official_VAL_metric_evaluation_or_selection':False,'data_copies_excluded_from_timing':True,
        'predictor_geometry_modified':False,'total_predictor_peak_memory_remeasured':False})
    verify_frozen();print('STAGE6A_EFFICIENCY_PASS',rows,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
