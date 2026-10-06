"""Freeze the single prescribed decoder experiment before any formal updates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage5a_common import atomic_json, sha256, CONFIG, PREREG, verify_previous

SOURCES=('00_manifest/stage5a_decoder.py','00_manifest/stage5a_model.py',
         '00_manifest/stage5a_common.py','03_training/stage5a_audits_tiny.py',
         '03_training/stage5a_train.py')


def main():
    verify_previous(shards=True)
    value={'status':'REGISTERED_BEFORE_FORMAL_TRAINING','seed':2022,
        'base_commit':'9c0a91a02012565f36f2214d24b8f05baa3066b7',
        'Stage5A_config_sha256':sha256(CONFIG),
        'training_source_sha256':{name:sha256(ROOT/name) for name in SOURCES},
        'from_scratch':True,'trained_Stage3B_weights_loaded_for_training':False,
        'warmup_steps':5000,'NLL_maximum_steps':16000,'global_hard_max':21000,
        'validation_interval':500,'NLL_patience':5,'batch':16,
        'warmup_LR':.001,'NLL_LR':.0001,'weight_decay':.0001,
        'selection':'full-horizon official VAL overall minFDE6, strict improvement; final from original NLL',
        'phase_transition':'own warmup best model + AdamW + RNG; LR only changed; NLL sampler reset as frozen Stage3B',
        'primary_baseline':'Stage3B TypeEmbedding','formal_models':1,
        'formal_train_windows':16898,'formal_val_windows':3603,
        'TRAIN_scenes':700,'VAL_scenes':150,'test_used':False,
        'final_actor_counts':{'full':54990,'partial':30037,'total':85027},
        'history_features':['type one-hot V/P/B','log1p recent displacement','log1p net displacement','log1p path length'],
        'experts':2,'bottleneck':16,'router_hidden':16,'residual_scale':1,
        'tiny_updates':1000,'tiny_selection':'TRAIN only, three classes plus moving/low-motion history V/P; full context',
        'offline_motion_bins_m':[[0,1],[1,2],[2,5],[5,10],[10,20]],
        'bin_convention':'[lo,hi); offline future endpoint displacement; <5 and >5 strictly as frozen definitions',
        'bootstrap':{'unit':'paired official VAL scene cluster','scenes':150,'replicates':1000,'seed':2022,
            'CI':'percentile 95%','weighting':'actor-window deltas pooled within resampled complete clusters',
            'primary':'Overall minFDE6 E-B',
            'secondary':['Vehicle FDE','Pedestrian FDE','vehicle.moving FDE','Vehicle >5m FDE',
                'Pedestrian <5m FDE','Pedestrian 5-10m FDE','Pedestrian >5m FDE','Overall Top1FDE','Pedestrian Top1FDE']},
        'scientific_rule':{
            'SUPPORTED':'Overall delta<0 and CI upper<0; no reliable Vehicle/Pedestrian harm; >=1 important motion group reliably improves; no collapse',
            'PARTIAL':'Overall CI spans0; >=2 important motion groups reliably improve; no reliable Vehicle/Pedestrian harm; no collapse',
            'NOT_SUPPORTED':'reliable Overall or Vehicle/Pedestrian harm, no reliable important motion gain, or remaining support criteria unmet',
            'important_motion_groups':['vehicle.moving','Vehicle >5m','Pedestrian <5m','Pedestrian 5-10m'],
            'major_reliable_harm_guard':'conservative: any Vehicle/Pedestrian FDE CI lower>0; no effect-size threshold invented'},
        'router_collapse':'same expert gets >0.9 probability in >95% of all current-valid actor-windows',
        'expert_functional_collapse':{
            'population':'all current-valid actor-window, all6 modes evaluated on same hidden',
            'rule':'both experts numerically zero OR >95% actor-windows have relative RMS difference<1e-3 and mean cosine>0.999',
            'relative_difference':'RMS(A1-A2) / max(RMS(A1), RMS(A2), 1e-12)',
            'purpose':'descriptive numerical near-identity criterion declared before training; not a tuning objective'},
        'efficiency':{'paired_measurements':500,'batch':16,'same_graphs':True,'order':'alternating B/E and E/B',
            'timing':'CUDA events, synchronized model forward only; preloaded GPU batch','warmup':20},
        'final_reload_metric_tolerance':{'minADE6':1e-6,'minFDE6':1e-6,'MR6':1e-6,'Top1ADE6':1e-6,'Top1FDE6':1e-6,'NLL':1e-5},
        'Reliability_executed':False,'additional_seeds':False,'hyperparameter_search':False,
        'attention_modified':False,'pi_structure_modified':False}
    atomic_json(PREREG,value)
    print('STAGE5A_REGISTERED',flush=True)


if __name__=='__main__': main()
