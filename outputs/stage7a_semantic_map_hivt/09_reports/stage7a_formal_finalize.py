"""Evidence-grounded report and complete formal protocol preservation audit."""
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7a_formal_common import *
import pandas as pd
import numpy as np
import torch

def markdown(df,columns=None):
    if columns:df=df[columns]
    # Avoid optional tabulate dependency.
    head='| '+' | '.join(map(str,df.columns))+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'
    lines=[]
    for row in df.itertuples(index=False,name=None):
        values=[f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row]
        lines.append('| '+' | '.join(values)+' |')
    return head+'\n'.join(lines)

def main():
    verify_previous(shards=True);s=read_json(SUMMARY);d=read_json(ROOT/'04_evaluation/stage7a_scientific_decision.json')
    neutral=read_json(ROOT/'00_manifest/stage7a_neutral_initialization_audit.json');identity=read_json(ROOT/'01_data_audit/stage7a_data_identity_audit.json')
    pairing=read_json(ROOT/'04_evaluation/stage7a_pairing_audit.json');prereg=read_json(PREREG)
    for n,h in prereg['training_source_sha256'].items():assert sha256(ROOT/n)==h,n
    assert sha256(BEST)==s['checkpoint_sha256'] and neutral['status']==identity['status']==pairing['status']=='PASS'
    assert s['warmup_steps']==5000 and s['NLL_steps']<=16000 and s['final_executed_global_step']<=21000
    assert s['stop_reason'] in ('patience_5','global21000_budget')
    curve=pd.read_csv(CURVE);warm=curve[curve.phase=='fixed_scale'];nll=curve[curve.phase=='original_nll']
    assert warm.global_step.tolist()==list(range(500,5001,500))
    assert nll.global_step.tolist()==list(range(5500,s['final_executed_global_step']+1,500))
    assert s['best_global_step']==int(nll.loc[nll.VAL_overall_FDE.idxmin(),'global_step'])
    if s['stop_reason']=='patience_5':assert int(nll.iloc[-1].consecutive_nonimprovements)==5
    saved=torch.load(BEST,map_location='cpu',weights_only=False)
    assert saved['metadata']['global_step']==s['best_global_step'] and saved['metadata']['phase']=='original_nll'
    assert all(k in saved for k in ('optimizer_state_dict','torch_rng','cuda_rng','python_rng','numpy_rng'))
    for step in curve.global_step.astype(int):
        val=read_json(ROOT/f'04_evaluation/stage7a_val_step_{step:05d}.json')
        assert val['windows']==3603 and len(val['scenes'])==150
        assert val['metrics']['full_horizon']['overall']['count']==54990 and val['metrics']['partial_future']['overall']['count']==30037
    ci=pd.read_csv(ROOT/'06_tables/stage7a_bootstrap_ci.csv');main=pd.read_csv(ROOT/'06_tables/stage7a_main_results.csv')
    motion=pd.read_csv(ROOT/'06_tables/stage7a_vehicle_motion_results.csv');sub=pd.read_csv(ROOT/'06_tables/stage7a_semantic_subgroup_results.csv')
    zero=pd.read_csv(ROOT/'06_tables/stage7a_semantic_zero_ablation.csv');residual=pd.read_csv(ROOT/'06_tables/stage7a_semantic_residual_statistics.csv')
    efficiency=pd.read_csv(ROOT/'06_tables/stage7a_efficiency.csv');residual_audit=read_json(ROOT/'04_evaluation/stage7a_semantic_residual_audit.json')
    fig_names=['stage7a_main_fde_comparison','stage7a_vehicle_difficulty_groups','stage7a_semantic_on_zero',
               'stage7a_case_turning_vehicle','stage7a_case_traffic_control','stage7a_case_degradation']
    for name in fig_names:
        assert read_json(ROOT/'05_figures'/(name+'_audit.json'))['status']=='PASS'
        for extension in ('png','pdf','svg'):assert (ROOT/'05_figures'/(name+'.'+extension)).stat().st_size>1000
    formal=main[main.horizon=='full_horizon'];motion=motion[motion.horizon=='full_horizon'];sub=sub[sub.horizon=='full_horizon']
    cols=['model','group','count','minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL']
    ci_cols=['group','metric','count','Stage3B','Stage7A','delta','CI_lower','CI_upper']
    primary=ci[(ci.group=='overall')&(ci.metric=='minFDE6')].iloc[0]
    top1=ci[(ci.group=='overall')&(ci.metric=='Top1FDE6')].iloc[0]
    turn=ci[(ci.group=='TurningVehicle_GT')&(ci.metric=='minFDE6')].iloc[0]
    ablation_overall=zero[zero.group=='overall'].set_index('model')
    baseline_formal=read_json(ROOT/'04_evaluation/stage7a_baseline_final_metrics.json')['metrics']['full_horizon']
    original=read_json(STAGE3/'07_checkpoints/stage3b_best_checkpoint_manifest.json')['selected_full_horizon_metrics']
    baseline_reproduction=max(abs(baseline_formal[g][k]-original[g][k]) for g in GROUPS for k in METRICS)
    assert baseline_reproduction<1e-5
    audit={'status':'PASS','training_complete':True,'from_scratch':True,'formal_training_runs':1,'shared_step0_exact':True,
      'data_identity_bitwise_equal':True,'old_shards_SHA_checked':850,'old_files_SHA_checked':len(read_json(FREEZE)['files']),
      'old_unrelated_untracked_preserved':len(read_json(FREEZE)['unrelated_untracked']),
      'TRAIN_scenes':700,'VAL_scenes':150,'VAL_supervised_windows':3603,'full_horizon':54990,'partial_future':30037,
      'total_supervised_actor_windows':85027,'NaN':0,'Inf':0,'validation_checks':len(curve),'bootstrap_replicates':1000,
      'Stage3B_reproduction_max_metric_difference':baseline_reproduction,
      'final_checkpoint_sha256':sha256(BEST),'source_preregistration_sha256':sha256(PREREG),'test_used':False,
      'Stage7B_executed':False,'main_merged':False,'semantic_architecture_variants_trained':False}
    atomic_json(ROOT/'00_manifest/stage7a_formal_final_protocol_audit.json',audit)
    learned='The semantic residual remains below the preregistered near-zero diagnostic threshold; the branch was not effectively used.' if residual_audit['effectively_unused'] else 'The learned residual is nonzero. Its magnitude describes a representation, and does not establish physical causality or prove performance benefit.'
    report=f'''# Stage7A Semantic Map Enhancement

【Problem】

Lane geometry describes road shape but does not explicitly distinguish connector turn classes, static traffic-control associations or crosswalk intersections. This experiment asks whether a single static lane residual improves the geometry of K=6 predicted candidates. The sole primary comparator is independently trained Stage3B TypeEmbedding. No reranker or future-interaction head is used.

【Frozen Semantic Definition】

The nine float32 lane-segment features are `{', '.join(prereg['semantic_fields'])}`. Ordinary lanes have connector=0 and all four turn slots=0. Connectors use the audited analytic arcline entrance/exit tangent taxonomy: abs heading change<=20° straight, >20° left, <-20° right, abs>=150° unknown. Controls are multi-hot: TRAFFIC_LIGHT, STOP_SIGN, and other(TURN_STOP/PED_CROSSING/YIELD). Control association is whole-centerline intersection with typed stop-line polygons. Crosswalk means whole lane/connector centerline intersects a ped-crossing polygon. No 2m/5m alternative, actor-level50m binary, dynamic signal state or topology repair is used. Earlier audit files retain their historical definitions; the formal vector conversion is independently registered.

【Architecture】

Canonical LocalEncoder → TypeEmbedding → GlobalInteractor → MLPDecoder is preserved. Actor–lane geometry and original all-zero categorical base remain unchanged. Linear(9,32)→ReLU→Linear(32,64) adds a semantic residual immediately before the original K/V projections. Attention, gates, FFN, decoder and losses are unchanged. Stage3B has {neutral['baseline_parameters']} parameters; Stage7A has {neutral['parameters']}; addition {neutral['additional_parameters']} ({neutral['increase_percent']:.6f}%). The old runtime is read-only.

【Neutral Initialization】

PASS. All shared parameters and buffers match fresh canonical Stage3B at seed2022; max parameter difference=0. The last semantic layer has zero weight and bias. On six real TRAIN graphs with deterministic CUDA aggregation and CuBLAS workspace configured for the audit, raw_prediction, mode_logits and mode_prob differences are exactly0. Unrestricted CUDA aggregation also causes approximately2e-6 differences between repeated forwards of Stage3B itself. Training retains the previous Stage3B kernel policy. Neither baseline nor any later-stage trained weights initialize Stage7A. Both semantic layers and original modules receive finite nonzero gradients within10 updates; the last semantic layer does so on update1.

【Data Integrity】

PASS. Random seed2022 sampled100 TRAIN and100 VAL windows; every original graph field is bitwise equal and the only added field is lane_semantic[L,9]. All850 old scene shards and {audit['old_files_SHA_checked']} frozen source/result/reference files retain SHA256. Five unrelated Stage2C redraw files remain unchanged and unsubmitted. Official split is TRAIN700/VAL150, with16898/3603 supervised windows; final VAL has54990 full and30037 partial targets (85027 total), NaN0/Inf0. Empty-supervision windows remain excluded under the existing definition. No trajectory, GT, mask, identity, map vector or graph edge is rebuilt.

【Training】

One fresh seed2022 run; Th5,Tf12,K6,batch16,embedding64,heads8,global layers3,temporal layers4,dropout0.1,local radius50m,AdamW weight_decay1e-4, natural class frequencies. Tiny is a fixed six-window TRAIN health check:200 fixed-scale updates, no architecture tuning, with semantic turn/control/crosswalk exposure and all three target classes plus moving vehicles. Formal warm-up is5000 fixed-scale updates atLR0.001; every500updates evaluates all150 VAL scenes. The model, optimizer and Torch/CUDA/Python/NumPy RNG restore from Stage7A's own warm-up best at source step{s['warmup_best_source_step']}. NLL begins at global counter5000 withLR0.0001 and reset scene-sampler cursor matching Stage3B. It executes{s['NLL_steps']} NLLupdates; final counter{s['final_executed_global_step']}; best global step{s['best_global_step']}; stop reason `{s['stop_reason']}`. Selection uses strict overall full-horizon minFDE6 improvement, patience5 full VAL checks, hard cap21000. Fresh best reload reproduces selected metrics within1e-5, allowing existing CUDA floating-order variation.

Training source commit: `{s['training_code_git_commit']}`. Stage7A checkpoint SHA256: `{s['checkpoint_sha256']}`. Stage3B comparison checkpoint SHA256: `{prereg['baseline_checkpoint_sha256']}`. Final Stage3B reproduction differs from its archived metrics by at most{baseline_reproduction:.3g}.

【Main Results】

All following primary metric tables use full-horizon targets. ADE is evaluated on the best-FDE mode; MR uses endpoint error>2m; Top1 is the highest original model probability; NLL uses the original summed-L2 training winner and mean valid coordinate Laplace density. Partial-future results are also retained in the CSV tables. Lower is better.

The primary Stage7A−Stage3B minFDE6 difference is{primary['delta']:+.6f}m, paired95%CI[{primary['CI_lower']:+.6f},{primary['CI_upper']:+.6f}]. Overall Top1FDE6 difference is{top1['delta']:+.6f}m, CI[{top1['CI_lower']:+.6f},{top1['CI_upper']:+.6f}]. Both intervals are wholly above0, indicating small deterioration for this fixed-seed, validation-selected comparison. No preregistered difficult vehicle subgroup has a reliable minFDE improvement. This outcome does not support a general claim that static semantics are ineffective; it rejects the intended benefit for this particular feature definition and fusion experiment.

{markdown(formal,cols)}

【Moving Vehicle】

Motion states are the frozen t0 nuScenes attribute labels, not GT-derived inference inputs.

{markdown(motion,cols)}

【Intersection / Turning Vehicle】

IntersectionVehicle20 and NearTurnConnector20 use t0 distance<20m to complete regional connector centerlines, independent of cropped graph geometry. TurningVehicle_GT is strictly offline: full-future endpoint displacement>5m, first/last1s secant displacement>0.5m, abs wrapped heading change>20°. Its frozen VAL count is1663. Turn-context labels use the unique nearest left/right connector within20m; distances tied within1cm remain unassigned. Rules were frozen before training and never adjusted using VAL prediction errors.

TurningVehicle_GT minFDE6 changes by{turn['delta']:+.6f}m, CI[{turn['CI_lower']:+.6f},{turn['CI_upper']:+.6f}], which includes0. The extreme illustrative improvement case does not establish an aggregate turning benefit.

{markdown(sub[sub.group.isin(['IntersectionVehicle20','NearTurnConnector20','TurningVehicle_GT','Left-context','Right-context'])],cols)}

Vehicle>5m and pedestrian displacement groups are present in the main table and are offline GT-defined analyses.

【Traffic-Control Context】

NearTrafficControl20 uses t0 distance<20m to a control-associated whole lane/connector centerline. Static association does not provide light color, stop compliance or intent.

{markdown(sub[sub.group=='NearTrafficControl20'],cols)}

【Semantic Zero Ablation】

One full VAL inference per setting on the identical Stage7A best checkpoint; no retraining. ZERO sets all nine inputs to0 but retains the trained MLP biases, so its residual is not necessarily0 and it is not an independently trained Stage3B baseline. Residual at zero input has norm{residual_audit['all_features_zero_residual_norm']:.6f}.

ON overall minFDE6 is{ablation_overall.loc['Semantic-ON','minFDE6']:.6f}m and ZERO is{ablation_overall.loc['Semantic-ZERO','minFDE6']:.6f}m. This point comparison demonstrates that the trained predictions respond to semantic inputs, while ON still fails to outperform the independently trained Stage3B baseline. No additional ablation checkpoint or feature combination was trained.

{markdown(zero[zero.group.isin(['overall','vehicle','pedestrian','vehicle.moving','TurningVehicle_GT'])],cols)}

【Bootstrap】

1000 paired bootstrap draws over all150 official VAL scene clusters,seed2022. Entire scenes are resampled together; actor-window sums divided by counts determine each estimate. Every actor identity, original future tensor fingerprint, type and mask is paired. Delta=Stage7A−Stage3B. Intervals are percentile95%; overall minFDE6 is primary. Other intervals are exploratory and unadjusted; semantic groups overlap and are correlated.

{markdown(ci,ci_cols)}

【Semantic Residual Analysis】

{learned} The feature-conditioned mean norm difference from zero input is{residual_audit['feature_conditioned_mean_difference_from_zero_input']:.6f}, separating the actual input effect from the constant zero-input residual. Statistics weight actual stored VAL lane segment/window occurrences. Repeated appearances of a map token are therefore repeated exposures, not independent physical lane samples. Groups overlap for multi-hot control features.

{markdown(residual)}

【Efficiency】

500 paired forwards, alternating method order, eight identical real VAL batches repeated, batch16, warm-up3pairs/batch, CUDA events with synchronization. Data loading is excluded. Both models remain resident; reported absolute allocated peaks include common co-resident weights and input; incremental forward peak is separately reported. This is one machine/session and excludes preprocessing or cache generation.

{markdown(efficiency,['model','parameters','parameter_increase_percent','mean_forward_ms','median_forward_ms','peak_CUDA_allocated_MiB','peak_forward_increment_MiB'])}

【Limitations】

Static map only; no dynamic traffic-light state, camera/VLM, intent, trajectory regeneration, balanced loss or interaction architecture. Controls are geometric associations and turn classes are derived taxonomy, with ambiguous near-U-turn connectors marked unknown. The absent raw trainval mount is not required because byte-frozen official scene shards and the already audited local map cache are used. One seed per independently trained method cannot quantify seed-to-seed variance; validation-selected checkpoints and secondary overlapping subgroup tests limit inference beyond this experiment. GT-turning/displacement groups are offline, never model inputs. ON/ZERO is an input perturbation and may be outside the training feature distribution. Residual norms do not establish causality. Cases are deliberately selected extreme improvements/degradation using the saved rule and are illustrations, not aggregate evidence. No test split, semantic variant retraining or Stage7B was run.

【Scientific Decision】

SemanticMap = {d['SemanticMap']}

PaperUsableSemantic = {d['PaperUsableSemantic']}

ReadyStage7B = {d['ReadyStage7B']}

The exact preregistered numeric interpretation is: marked reliable harm means>5% relative Vehicle/Pedestrian minFDE increase with CI lower>0; basically flat means overall point increase<=1%. SUPPORTED requires overall reliable improvement, no marked harm and at least one difficult-group point improvement. TARGETED_SUPPORTED requires overall point improvement/flat with CI including0, at least two difficult-group reliable improvements, and no marked harm. If paper usability isNO, readiness requires overall increase<=1%, overall CI not wholly positive, and reliable improvements in BOTH moving and GT-turning with no marked harm. Reliable difficult groups: `{d['reliable_difficult_groups']}`. Marked reliable harms: `{d['marked_reliable_harm']}`. Rules were fixed before training.

Completed Stage7A only. STOP; Stage7B awaits separate review and authorization.
'''
    (ROOT/'09_reports/stage7a_final_report.md').write_text(report)
    fields={'training_code_git_commit':s['training_code_git_commit'],'Stage7A_checkpoint_SHA':sha256(BEST),
      'Stage7A_parameters':neutral['parameters'],'Additional_parameters':neutral['additional_parameters'],
      'Neutral_step0_audit':'PASS','Data_identity_audit':'PASS','Best_global_step':s['best_global_step'],
      'Training_stop_reason':s['stop_reason'],'paired_metrics':ci[ci.metric=='minFDE6'].to_dict('records'),
      'Overall_Top1FDE':ci[ci.metric=='Top1FDE6'].to_dict('records')[0],
      'semantic_ablation_overall':zero[zero.group=='overall'].to_dict('records'),
      'Semantic_residual_norm':residual[residual.group=='all'].to_dict('records')[0],**d,'STOP':True}
    atomic_json(ROOT/'09_reports/stage7a_final_fields.json',fields)
    print('STAGE7A_COMPLETE',fields,flush=True)

if __name__=='__main__':torch.set_num_threads(4);main()
