"""Audit the full evidence package and write the requested Stage5A final report."""
from pathlib import Path
import sys
import csv
import ast
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage5a_common import (PROJECT, SUMMARY, BEST, CONFIG, PREREG, atomic_json, read_json, sha256,
    verify_previous, git)


def read_table(name):
    with (ROOT/'06_tables'/name).open() as f:return list(csv.DictReader(f))


def markdown_table(rows,columns,decimals=6):
    def value(v):
        if v is None:return '—'
        if isinstance(v,float):return f'{v:.{decimals}f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |',
                      *['| '+' | '.join(value(row.get(c)) for c in columns)+' |' for row in rows]])


def numeric_rows(rows):
    return [{k:(int(v) if k in ('Count','Instances','Scenes') else float(v))
             if k not in ('Group','Model','Population') and v not in ('','—') else v for k,v in row.items()} for row in rows]


def main():
    frozen=verify_previous(shards=True);training=read_json(SUMMARY);prereg=read_json(PREREG)
    assert training['status']=='COMPLETE' and sha256(BEST)==training['checkpoint_sha256']
    assert sha256(CONFIG)==training['config_sha256'] and sha256(CONFIG)==prereg['Stage5A_config_sha256']
    assert training['warmup_steps']==5000 and training['NLL_steps']<=16000 and training['final_executed_global_step']<=21000
    for path in ('00_manifest/stage5a_initialization_audit.json','00_manifest/stage5a_neutral_initialization_audit.json',
        '00_manifest/stage5a_no_future_leakage_audit.json','00_manifest/stage5a_unit_tests.json',
        '00_manifest/stage5a_decoder_branch_audit.json',
        '04_evaluation/stage5a_gradient_audit.json','04_evaluation/stage5a_tiny_overfit.json',
        '04_evaluation/stage5a_metrics.json','04_evaluation/stage5a_pairing_audit.json',
        '04_evaluation/stage5a_fresh_val_reconciliation.json','04_evaluation/stage5a_bootstrap_ci.json',
        '04_evaluation/stage5a_decoder_statistics.json','04_evaluation/stage5a_efficiency_audit.json',
        '04_evaluation/stage5a_qualitative_case_manifest.json','00_manifest/stage5a_export_QA.json',
        '00_manifest/stage5a_manual_visual_QA.json'):
        assert read_json(ROOT/path)['status']=='PASS',path
    for name,digest in prereg['training_source_sha256'].items():assert sha256(ROOT/name)==digest,name
    init=read_json(ROOT/'00_manifest/stage5a_initialization_audit.json')
    neutral=read_json(ROOT/'00_manifest/stage5a_neutral_initialization_audit.json')
    tiny=read_json(ROOT/'04_evaluation/stage5a_tiny_overfit.json')
    metrics=read_json(ROOT/'04_evaluation/stage5a_metrics.json')
    pair=read_json(ROOT/'04_evaluation/stage5a_pairing_audit.json')
    boot=read_json(ROOT/'04_evaluation/stage5a_bootstrap_ci.json')
    stats=read_json(ROOT/'04_evaluation/stage5a_decoder_statistics.json')
    efficiency=read_json(ROOT/'04_evaluation/stage5a_efficiency_audit.json')
    cases=read_json(ROOT/'04_evaluation/stage5a_qualitative_case_manifest.json')
    decision=read_json(ROOT/'04_evaluation/stage5a_scientific_decision.json')
    export_QA=read_json(ROOT/'00_manifest/stage5a_export_QA.json')
    manual_QA=read_json(ROOT/'00_manifest/stage5a_manual_visual_QA.json')
    inspected={r['name']:r['PNG_SHA256'] for r in manual_QA['figure_records']}
    assert len(inspected)==8 and set(inspected)=={r['name'] for r in export_QA['figure_records']}
    for name,digest in inspected.items():assert sha256(ROOT/'05_figures'/(name+'.png'))==digest
    assert (pair['full'],pair['partial'],pair['total'])==(54990,30037,85027)
    assert metrics['windows']==3603 and len(metrics['scenes'])==150
    for name in ('stage5a_decoder.py','stage5a_model.py','stage5a_common.py'):
        tree=ast.parse((ROOT/'00_manifest'/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module or '']
                assert not any('stage4' in m for m in modules),'Stage4 code imported'
    all_rows=numeric_rows(read_table('stage5a_comparison_all_groups.csv'))
    main_rows=numeric_rows(read_table('stage5a_main_results.csv'))
    vehicle=[r for r in all_rows if r['Group'] in ('vehicle.moving','vehicle.stopped','vehicle.parked','unknown','Vehicle >5m')]
    pedestrian=[r for r in all_rows if r['Group'].startswith('Pedestrian ')]
    comparison_cols=('Group','Count','Stage3B_minFDE6','Stage5A_minFDE6','Delta_minFDE6')
    bootstrap_rows=[]
    for group,fields in boot['groups'].items():
        for metric,v in fields.items():
            bootstrap_rows.append({'Group':group,'Metric':metric,'Delta':v['delta'],'CI lower':v['CI95'][0],'CI upper':v['CI95'][1],
                                   'Count':v['Count'],'Scenes':v['Scenes']})
    router=[r for r in stats['router_groups'] if r['Population']=='full_horizon']
    experts=[r for r in stats['expert_groups'] if r['Population']=='full_horizon']
    text=['# Stage5A: Motion-Aware Heterogeneous Residual Decoder',
        '', '【Motivation】', '',
        'Stage3B TypeEmbedding remains SUPPORTED and is the sole formal backbone baseline. Stage4A always-on relation bias and Stage4F necessity-gated relation bias remain technically PASS but scientifically NOT SUPPORTED. Stage4F-D GateCompensation remains MIXED: pooled attenuation coexists with a localized Layer2/Head3 exception. These results motivate stopping attention-logit relation-bias changes and testing actor-type/history-dependent trajectory decoding.',
        '', '【Current Decoder】', '',
        'The actual frozen decoder is an MLPDecoder. pi = pi(concat(local, global)); h = aggr_embed(concat(global, local)); loc = loc(h); scale = ELU(scale(h)) + 1 + min_scale. There is no mode query, decoder cross-attention or Transformer decoder. The pi branch retains its original structure and embedding inputs.',
        '', '【Method】', '',
        'The original LocalEncoder, TypeEmbedding fusion, GlobalInteractor and multihead_proj are retained. c is six-dimensional: type one-hot [V,P,B] plus log1p(recent displacement), log1p(history net displacement) and log1p(history path length). Padding gaps are skipped and successive valid observations are connected in time; fewer than two valid observations produce zero motion features.',
        '', 'r = softmax(Linear(16,2)(ReLU(Linear(6,16)(c)))). Each actor shares its two routing weights across all six modes. Two unnamed experts use Linear(64,16) → ReLU → Linear(16,64). Δh[k,i] = Σₑ r[i,e] Aₑ(h[k,i]); h′ = h + Δh; the original shared loc and scale heads receive h′. Residual scale is one. The experts do not receive future labels or neighbor statistics, and they do not directly modify pi.',
        '', markdown_table([{'Model':'Stage3B','Parameters':init['Stage3B_params']},
            {'Model':'Stage5A','Parameters':init['Stage5A_params']},{'Model':'Additional','Parameters':init['Additional_params']}],('Model','Parameters')),
        '', '【No Future Leakage】', '',
        'The condition function accepts only [N,5,2] historical positions, [N,5] historical padding and actor types. Perturbing future trajectory, future masks/padding, GT endpoint, future labels, future times, ego future and target mask changes neither condition nor routing (maximum difference zero). The real-batch output comparison also has zero difference. Future-displacement bins are used only after inference for offline analysis.',
        '', '【Initialization】', '',
        'Seed 2022 reproduces canonical Stage3B step0 from scratch. Every shared parameter and buffer is bitwise equal. Both expert final Linear layers have zero weight/bias; the router final layer is also zero, so r=[0.5,0.5] and Δh=0. No trained Stage3B checkpoint is loaded for optimization.',
        '', f"Neutral real-TRAIN-batch differences: raw_prediction={neutral['raw_prediction_max_abs_diff']}; mode_logits={neutral['mode_logits_max_abs_diff']}; mode_prob={neutral['mode_prob_max_abs_diff']}. All required shape, finite, broadcast and padding tests pass. All six new gradient modules are positive within ten updates. Tiny fixed-scale training uses the frozen six TRAIN windows, covers 11 vehicle / 12 pedestrian / 6 bicycle targets and moving/low-history-motion V/P; all three regression diagnostics decrease. Mean absolute residual={tiny['mean_absolute_residual']:.6f}; maximum router deviation from half={tiny['router_max_deviation_from_half']:.6f}. Tiny is discarded before formal optimization.",
        '', '【Training】', '',
        f"One formal from-scratch model, seed2022; official TRAIN700/VAL150, test unused; Th=5, Tf=12, K=6; batch16, embed64, heads8, global layers3, dropout0.1, local radius50m, AdamW weight_decay1e-4. Fixed-scale warm-up executes exactly5000 updates at LR0.001. NLL restores this experiment's own warm-up best model/optimizer/RNG, then uses the unchanged original learnable Laplace NLL at LR0.0001. Python, NumPy, torch and CUDA RNG states are saved. Complete VAL150 occurs every500 updates, strict overall full-horizon minFDE6 selection and NLL patience5; global hard limit21000. There is no extra loss, oversampling, class weighting, expert balancing or hyperparameter search.",
        '', f"Warm-up best source step={training['warmup_best_source_step']}; NLL updates={training['NLL_steps']}; best global step={training['best_global_step']}; final executed global step={training['final_executed_global_step']}; stop reason={training['stop_reason']}. Training code commit={training['training_code_git_commit']}. Best checkpoint SHA256={training['checkpoint_sha256']}. Fresh reload reconciles all metrics to the selected checkpoint within preregistered tolerances. NaN=0, Inf=0.",
        '', '【Main Results】', '',
        'All main values use full-horizon actor-window means. minADE6 is ADE of the best-FDE mode, minFDE6 the smallest endpoint error, MR6 endpoint error>2m, Top1 argmax predicted probability, NLL original best-summed-L2 mode and valid-time coordinate density mean.',
        '', markdown_table(main_rows,('Group','Model','Count','minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL')),
        '', '【Vehicle Motion】', '',markdown_table(vehicle,comparison_cols),
        '', '【Pedestrian Motion Bins】', '',
        'Frozen endpoint bins are [0,1), [1,2), [2,5), [5,10), [10,20) meters. <5m and >5m use strict inequalities and remain offline only.',
        '',markdown_table(pedestrian,comparison_cols),
        '', '【Bootstrap】', '',
        'Stage3B and Stage5A match scene/sample/instance/node/type/motion/future-mask/GT-SHA exactly: 54,990 full +30,037 partial =85,027 actor-windows across150 VAL scenes and3603 supervised windows. The primary endpoint is overall full-horizon minFDE6. Paired scene-cluster percentile bootstrap uses1000 replicates and seed2022, pooling all actor-window deltas in each set of resampled whole scenes. Negative E−B favors Stage5A. These intervals condition on the two selected checkpoints and do not measure variation across training seeds or correct for checkpoint selection on the same VAL split. Secondary subgroup intervals are descriptive; no multiplicity correction is applied. Overlapping windows and nested motion groups are not independent observations. Bicycle has limited unique instances/scenes and receives cautious interpretation.',
        '',markdown_table(bootstrap_rows,('Group','Metric','Delta','CI lower','CI upper','Count','Scenes')),
        '', '【Router Behavior】', '',
        'Routing is actor-level and shared across six modes. Entropy uses natural logarithms. Dominance is argmax; exact ties are separately reported. Group tables below use full-horizon targets; the collapse check additionally covers all current-valid context actors.',
        '',markdown_table(router,('Group','Count','r1_mean','r2_mean','r1_median','r2_median','entropy_mean','expert1_dominant_rate','expert2_dominant_rate')),
        '',f"Router collapse={stats['router_collapse']}. Preregistered criterion: the same expert receives probability>0.9 in>95% of all current-valid actor-windows. Context population={stats['all_current_valid_actor_windows']}. The learned experts retain neutral numerical names; a routing association alone does not prove a static/dynamic specialization.",
        '',markdown_table(stats['correlations'],('Group','Feature','Spearman_rho','Count')),
        '', '【Expert Diversity】', '',
        'Both expert outputs are evaluated on the same hidden h. Norms and cosine are computed per mode then averaged over all six modes. Relative difference is RMS(A1−A2)/max(RMS(A1),RMS(A2),1e-12). Numerical cosine is zero when a vector is zero; this convention and the zero-output rate are reported. Before training, near-identity was defined as relative difference<1e-3 and cosine>0.999; functional collapse means this holds for>95% of current-valid actor-windows, or both outputs are numerically zero for>95%. This is a descriptive implementation criterion, not a training objective.',
        '',markdown_table(experts,('Group','Count','expert1_norm_mean','expert2_norm_mean','expert_cosine_mean','expert_relative_difference_mean','residual_norm_mean','near_identical_rate')),
        '',f"Expert functional collapse={stats['expert_functional_collapse']}.",
        '', '【Efficiency】', '',
        '500 paired measurements alternate B/E and E/B on identical preloaded GPU batches;20 warm-up batches. CUDA events cover synchronized model forward, excluding I/O, H2D and the external clone but including both models\' internal clone. An independent one-GPU-model pass over all226 VAL batches measures peak CUDA allocation. These are batch-forward latency measurements on the available GPU, not a deployment throughput claim.',
        '',markdown_table(efficiency['models'],('Model','parameters','mean_inference_ms','median_inference_ms','std_inference_ms','peak_CUDA_memory_MiB')),
        '',f"Mean inference overhead={efficiency['mean_inference_overhead_percent']:.6f}%; median overhead={efficiency['median_inference_overhead_percent']:.6f}%. GPU={efficiency['GPU']}; torch={efficiency['torch']}; CUDA={efficiency['CUDA']}.",
        '', 'Interpret small latency differences within the observed timing dispersion; this single-device measurement includes runtime noise.',
        '', '【Qualitative Cases】', '',
        'Four deterministic, purposive matched cases show moving vehicle, parked/stopped vehicle, moving pedestrian and degradation. The same actor/GT/map/frame/axis limits are used across Stage3B and Stage5A. Original twelve future observations are displayed without interpolation or smoothing. History=gray circles, GT=black squares, Best-FDE=blue circles, Top1=orange dashed triangles. Best-FDE uses GT and is distinct from predicted mode ranking. Replayed predictions use the original16-window VAL batch, match best/top1 mode IDs and reproduce CSV metrics within1e-4m. Examples do not estimate population effects.',
        '',markdown_table(cases['figures'],('name','case_kind','agent_type','delta_FDE_m')),
        '', '【Scientific Decision】', '',
        f"Motion-Aware Heterogeneous Decoder = **{decision['Motion_Aware_Decoder']}**. Stage5A = **{decision['Stage5A']}**. Ready Reliability = **{decision['Ready_Reliability']}**.",
        '',f"Important motion groups with reliable FDE gains: {', '.join(decision['important_motion_groups_with_reliable_gain']) or 'none'}. Reliable major-class harm guard={decision['reliable_main_class_harm']}. The guard conservatively treats any Vehicle/Pedestrian FDE95% CI lower>0 as harm; no post-hoc effect-size threshold is introduced. SUPPORTED requires overall CI upper<0 plus at least one reliable important motion gain and no class harm/collapse. PARTIAL requires overall CI spanning0 plus at least two such motion gains and no class harm/collapse. Remaining outcomes are NOT SUPPORTED.",
        '', f"Primary overall minFDE6 delta={boot['groups']['overall']['minFDE6']['delta']:.6f} m, 95% CI={boot['groups']['overall']['minFDE6']['CI95']}; reliable important motion gains={len(decision['important_motion_groups_with_reliable_gain'])}. Pedestrian <5m delta={boot['groups']['Pedestrian <5m']['minFDE6']['delta']:.6f} m, 95% CI={boot['groups']['Pedestrian <5m']['minFDE6']['CI95']}. Localized improvement and degradation must be read together with the primary interval, rather than inferring broad motion-adaptation benefit from routing separation alone.",
        '', 'All earlier scientific conclusions remain frozen. Only one Stage5A formal model was trained. Type-only/motion-only routing ablations, additional seeds, attention changes and Reliability are not executed. STOP: any next experiment requires the next explicit user instruction.',
        '', '【Artifacts】', '',
        'All new files reside in outputs/stage5a_motion_aware_decoder. Existing scene shards and checkpoints are referenced read-only; no dataset is copied or reprocessed. Code, configuration, small audit/report/table files, the complete training curve and figures are versioned. Raw actor/router records, per-validation scene-detailed JSON, logs and checkpoints remain local; counts, schemas and SHA256 are recorded as applicable. The artifact inventory excludes its own hash and transient Git-upload payloads. Old Stage2C redraw files are preserved and not submitted.',
        '']
    (ROOT/'09_reports/stage5a_final_report.md').write_text('\n'.join(text))
    branch=git('branch','--show-current');assert branch=='stage5a/motion-aware-residual-decoder'
    changed=git('diff','--name-only','9c0a91a02012565f36f2214d24b8f05baa3066b7').splitlines()
    assert all(p.startswith(str(ROOT.relative_to(PROJECT))+'/') for p in changed),changed
    audit={'status':'PASS','base_commit':frozen['base_commit'],'branch':branch,
        'previous_files_SHA256_unchanged':len(frozen['files']),'scene_shards_SHA256_unchanged':len(frozen['scene_shards']),
        'training_source_SHA256_unchanged':True,'formal_models':1,'training_seed':2022,
        'warmup_steps':5000,'NLL_steps':training['NLL_steps'],'best_global_step':training['best_global_step'],
        'final_executed_global_step':training['final_executed_global_step'],'stop_reason':training['stop_reason'],
        'full':54990,'partial':30037,'total':85027,'VAL_windows':3603,'VAL_scenes':150,
        'NaN':0,'Inf':0,'pairing':'PASS','future_leakage':'PASS','neutral_initialization':'PASS',
        'quantitative_figures':4,'matched_cases':4,'manual_visual_QA':'PASS',
        'technical_result':decision['Stage5A'],'scientific_result':decision['Motion_Aware_Decoder'],
        'Ready_Reliability':decision['Ready_Reliability'],'Reliability_executed':False,'STOP':True}
    atomic_json(ROOT/'00_manifest/stage5a_final_audit.json',audit)
    manifest=ROOT/'00_manifest/stage5a_artifact_manifest.json';artifacts=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p==manifest or p.name.startswith('stage5a_git_upload_') or p.suffix in ('.tmp','.pyc') or '__pycache__' in p.parts:continue
        ignored=p.suffix in ('.pt','.log') or p.name.startswith(('stage5a_actor_','stage5a_val_step_')) or p.name=='stage5a_router_actor_records.csv'
        artifacts.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p),
                          'Git_eligible':not ignored,'stage':'Stage5A'})
    with (ROOT/'04_evaluation/stage5a_actor_errors.csv').open() as f:actor_schema=next(csv.reader(f))
    with (ROOT/'04_evaluation/stage5a_router_actor_records.csv').open() as f:router_schema=next(csv.reader(f))
    atomic_json(manifest,{'status':'PASS','stage':'Stage5A','root':str(ROOT),'self_hash_excluded':True,'artifacts':artifacts,
        'raw_actor_records':{'full':54990,'partial':30037,'total':85027,'schema_fields':actor_schema},
        'raw_router_records':{'all_current_valid':stats['all_current_valid_actor_windows'],'supervised':85027,
                              'schema_fields':router_schema}})
    print('STAGE5A_FINAL_AUDIT_PASS',audit,flush=True)


if __name__=='__main__':main()
