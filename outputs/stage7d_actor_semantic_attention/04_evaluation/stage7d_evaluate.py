"""Fresh checkpoint reload and untouched canonical full/partial VAL metrics."""
from stage7d_analysis_common import *
from stage3b_common import model_new as baseline_new,evaluate as baseline_evaluate

def main():
    verify_previous();assert read_json(SUMMARY)['status']=='COMPLETE'
    saved=torch.load(BEST,map_location='cpu',weights_only=False)
    for name in ('baseline','on'):
        result_path=ROOT/f'04_evaluation/stage7d_{name}_final_metrics.json'
        actor_path=ROOT/f'04_evaluation/stage7d_{name}_actor_errors.csv'
        if result_path.exists() and actor_path.exists():continue
        if name=='baseline':
            model=baseline_new();cp=torch.load(BASE_BEST,map_location='cpu',weights_only=False)
            model.load_state_dict(cp['state_dict'],strict=True);del cp;ds=SceneDataset('val');fn=baseline_evaluate
        else:
            model=model_new();model.load_state_dict(saved['state_dict'],strict=True);ds=Stage7DSemanticDataset('val');fn=evaluate
        assert len(ds)==3603 and len(ds.scene_indices)==150
        measured=fn(ds,model,actor_path=actor_path,progress=True)
        assert measured['metrics']['full_horizon']['overall']['count']==54990
        assert measured['metrics']['partial_future']['overall']['count']==30037
        measured.update(fresh_checkpoint_reload=True,checkpoint_sha256=sha256(BASE_BEST if name=='baseline' else BEST),NaN=0,Inf=0)
        if name=='on':
            selected=saved['metadata']['full_horizon_metrics']
            diff=max(abs(measured['metrics']['full_horizon'][g][k]-selected[g][k]) for g in GROUPS for k in METRICS)
            assert diff<1e-5;measured['fresh_reload_max_metric_difference']=diff
        atomic_json(result_path,measured);print('FINAL_VAL',name,measured['metrics']['full_horizon']['overall'],flush=True)
        del model,ds;torch.cuda.empty_cache()
    paired_frames();verify_previous()

if __name__=='__main__':torch.set_num_threads(4);main()
