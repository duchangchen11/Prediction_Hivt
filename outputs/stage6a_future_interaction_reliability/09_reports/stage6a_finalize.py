"""Evidence-led final report and frozen-file audit; no model fitting or inference."""
from pathlib import Path
import sys,csv,ast
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage6a_common import *

def table(name):
    with (ROOT/'06_tables'/name).open() as f:return list(csv.DictReader(f))

def markdown(rows,fields):
    def value(x):
        if x is None:return '—'
        if isinstance(x,(float,int)):return f'{x:.6f}' if isinstance(x,float) else str(x)
        try:return f'{float(x):.6f}' if any(c in x for c in '.e') else x
        except (ValueError,TypeError):return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join('---' for _ in fields)+' |']+
        ['| '+' | '.join(value(r.get(k)) for k in fields)+' |' for r in rows])

def main():
    frozen=read_json(FREEZE)
    for i,(name,digest) in enumerate(frozen['files'].items()):
        assert sha256(PROJECT/name)==digest,name
        if i%300==0:print('VERIFY_PREVIOUS_FILES',i,'/',len(frozen['files']),flush=True)
    for i,(name,digest) in enumerate(frozen['scene_shards'].items()):
        assert sha256(STAGE3/name)==digest,name
        if i%100==0:print('VERIFY_ORIGINAL_SHARDS',i,'/',len(frozen['scene_shards']),flush=True)
    checks=('00_manifest/stage6a_unit_audit.json','00_manifest/stage6a_real_batch_audit.json',
        '00_manifest/stage6a_no_future_leakage_audit.json','01_cache/stage6a_cache_manifest.json',
        '01_cache/stage6a_cache_audit.json','02_features/stage6a_feature_manifest.json',
        '07_checkpoints/stage6a_checkpoint_manifest.json','04_evaluation/stage6a_evaluation_complete.json',
        '04_evaluation/stage6a_geometry_identity_audit.json','04_evaluation/stage6a_pairing_audit.json',
        '04_evaluation/stage6a_bootstrap_ci.json','04_evaluation/stage6a_probability_change_audit.json',
        '04_evaluation/stage6a_headdev_importance_audit.json','04_evaluation/stage6a_efficiency_audit.json',
        '04_evaluation/stage6a_qualitative_case_manifest.json','00_manifest/stage6a_export_QA.json',
        '00_manifest/stage6a_manual_visual_QA.json')
    for name in checks:assert read_json(ROOT/name)['status']=='PASS',name
    registration=read_json(ROOT/'00_manifest/stage6a_training_registration.json')
    for name,digest in registration['source_sha256'].items():assert sha256(ROOT/name)==digest,name
    for name,path in (('config_sha256',CONFIG),('split_sha256',SPLIT),('normalization_sha256',NORM)):
        assert registration[name]==sha256(path)
    official=read_json(ROOT/'00_manifest/stage6a_official_val_registration.json')
    assert official['evaluation_code_sha256']==sha256(ROOT/'04_evaluation/stage6a_evaluate.py')
    assert official['official_VAL_passes']==1 and not official['official_VAL_checkpoint_selection']
    training=read_json(ROOT/'03_training/stage6a_training_summary.json');assert training['status']=='COMPLETE'
    assert training['formal_heads']==2 and training['same_initial_weights'] and training['same_batch_orders']
    for h in training['heads']:
        assert sha256(head_path(h['variant']))==h['checkpoint_sha256']
        assert h['head_params']==673 and h['seed']==2022 and not h['official_VAL_used'] and h['full_horizon_labels_only']
        saved=torch.load(head_path(h['variant']),map_location='cpu',weights_only=False)
        assert sum(t.numel() for t in saved['state_dict'].values())==673
    geometry=read_json(ROOT/'04_evaluation/stage6a_geometry_identity_audit.json')
    assert (geometry['full'],geometry['partial'],geometry['total'],geometry['windows'],geometry['VAL_scenes'])==(54990,30037,85027,3603,150)
    for field in ('minADE6_max_R0_R1_R2_diff','minFDE6_max_R0_R1_R2_diff','MR6_max_R0_R1_R2_diff'):assert geometry[field]==0
    assert geometry['raw_and_ego_prediction_bitwise_identical'] and geometry['predictor_gradient_tensors']==0
    cacheaudit=read_json(ROOT/'01_cache/stage6a_cache_audit.json');assert cacheaudit['unique_instances']>=100
    assert max(cacheaudit['max_abs_difference'].values())<1e-6
    split=read_json(SPLIT);assert len(split['HeadTrain'])==630 and len(split['HeadDev'])==70
    assert not set(split['HeadTrain'])&set(split['HeadDev'])
    features=read_json(ROOT/'02_features/stage6a_feature_manifest.json');assert not features['VAL_used_for_normalization']
    for row in features['partitions'].values():assert sha256(ROOT/row['path'])==row['sha256']
    exports=read_json(ROOT/'00_manifest/stage6a_export_QA.json')
    manual=read_json(ROOT/'00_manifest/stage6a_manual_visual_QA.json')
    inspected={r['name']:r['PNG_SHA256'] for r in manual['figure_records']}
    assert len(inspected)==8 and set(inspected)=={r['name'] for r in exports['figure_records']}
    for name,digest in inspected.items():assert sha256(ROOT/'05_figures'/(name+'.png'))==digest
    for source in ROOT.rglob('*.py'):
        if source.name.startswith('stage6a_git_upload_'):continue
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or '']
                assert not any('stage4' in m for m in modules),'Stage4 model code import: '+str(source)
    rows=table('stage6a_main_ranking_results.csv');bygroup={r['Group']:r for r in rows};overall=bygroup['overall']
    for r in rows:
        for field in ('minADE6','minFDE6','MR6'):assert r['R0_'+field]==r['R1_'+field]==r['R2_'+field]
    boot=table('stage6a_bootstrap_ci.csv');net=table('stage6a_net_reranking_benefit.csv')
    for row in net:
        assert int(row['Number_changed'])==int(row['Number_improved'])+int(row['Number_worsened'])+int(row['Number_changed_equal'])
        reconstructed=(int(row['Number_worsened'])*float(row['Mean_degradation_when_worsened_m'])-
            int(row['Number_improved'])*float(row['Mean_improvement_when_improved_m']))/int(row['Count'])
        assert abs(reconstructed-float(row['Net_Top1FDE_delta']))<1e-12
    efficiency=read_json(ROOT/'04_evaluation/stage6a_efficiency_audit.json')
    importance=read_json(ROOT/'04_evaluation/stage6a_headdev_importance_audit.json')
    cases=read_json(ROOT/'04_evaluation/stage6a_qualitative_case_manifest.json')
    science=read_json(ROOT/'04_evaluation/stage6a_scientific_interpretation.json')
    pair=read_json(ROOT/'04_evaluation/stage6a_pairing_audit.json')
    rankfields=('Top1ADE','Top1FDE','OracleGap_FDE','Top1HitRate','BestModeRank','MRR_best_mode')
    rankrows=[{'Variant':v,**{k:float(overall[v+'_'+k]) for k in rankfields}} for v in ('R0','R1','R2')]
    headrows=[{'Variant':h['variant'],'Parameters':h['head_params'],'Best epoch':h['best_epoch'],
        'Epochs executed':h['executed_epochs'],'HeadDev Top1FDE':h['selected_HeadDev_metrics']['Top1FDE'],
        'HeadDev step0 Top1FDE':h['headdev_step0']['Top1FDE']} for h in training['heads']]
    mainfields=('Group','Count','Scenes','Instances','Stage3B_Top1FDE','R0_Top1FDE','R1_Top1FDE','R2_Top1FDE')
    interaction=[r for r in rows if '20m' in r['Group']]
    e0={'Variant':'R0','Predictor_params':650403,'Head_params':0,'Total_params':650403,'Parameter_increase_percent':0,
        'Mean_feature_extraction_ms':None,'Mean_head_scoring_ms':None,'Mean_ranking_overhead_ms':None}
    text=['# Stage6A: Future Interaction Reliability Re-ranking','',
        '【Problem】','',
        'The six candidate trajectories already contain better endpoint choices than the probability-selected Top1. This experiment changes only mode logits and probabilities, testing whether inference-computable future interaction features close part of that ranking gap. Stage3B TypeEmbedding remains the stable baseline. Stage5A remains technically PASS and scientifically NOT SUPPORTED under its earlier strict protocol; its geometry is authorized as a candidate backbone here. Those earlier conclusions are unchanged.',
        '',f"Moving-vehicle R0 minFDE is {float(bygroup['vehicle.moving']['R0_minFDE6']):.6f} m versus Top1FDE {float(bygroup['vehicle.moving']['R0_Top1FDE']):.6f} m. minFDE is a GT-selected oracle metric. Top1FDE selects a mode from predicted probabilities and is the ranking metric used for inference.",
        '', '【Frozen Predictor】','',
        f'Base commit: `{BASE_COMMIT}`. Branch: `stage6a/future-interaction-reliability`. All new files are isolated under `outputs/stage6a_future_interaction_reliability`.',
        '',f'Frozen checkpoint: `outputs/stage5a_motion_aware_decoder/07_checkpoints/stage5a_best_overall_minfde.pt`. SHA256: `{PREDICTOR_SHA}`. Parameters: 650,403. The predictor is in eval mode, every parameter has requires_grad=False, and no backward or optimization reaches it. The encoders, TypeEmbedding, residual decoder and all six candidate trajectories remain frozen.',
        '', 'TRAIN700 and VAL150 predictions are cached in the original 16-window batch order. Positions use the existing ego_predictions conversion: actor-local predictions times inverse actor rotation, plus the actor current position. Distances between actors use one common t0 ego frame (x forward, y left), in meters. The cache audit replays 120 unique actor instances, 60 TRAIN and 60 VAL; raw predictions, logits, probabilities, rotations and ego predictions have maximum difference zero, below 1e-6.',
        '', 'Runtime determinism was fixed and registered before caching and head optimization: torch deterministic algorithms, deterministic cuDNN and CUBLAS_WORKSPACE_CONFIG=:4096:8. A pre-cache repeat check initially exceeded tolerance (raw 7.629e-6, ego 1.526e-5); no formal cache or head training proceeded under that runtime. The corrected real-batch repeat and neutral probabilities are exactly equal. This changes runtime reproducibility, not checkpoint weights or architecture.',
        '', '【Reliability Features】','',
        'Each actor-mode input has 19 columns: type one-hot (3); log1p historical recent/net/path displacements (3); original logit and probability (2); predicted endpoint displacement, path length, mean step and maximum step from current position through the 12 future points (4); predicted future interaction features (7). History features reuse the existing padding-aware condition function and do not read future observations.',
        '', 'Neighbors must be valid at t0, exclude self, lie within 50 m and comprise at most the nearest eight. Ties retain node order. All neighbor futures and six-mode probabilities come from frozen Stage5A. For each target mode and neighbor mode, minimum and mean distance are computed over all 12 shared timestamps. Soft conflict is mean_t exp(−d²/(2·2²)). Expectation over the six frozen neighbor probabilities is taken before neighbor aggregation. There is no recursive reranking.',
        '', 'The seven aggregates are minimum expected minimum distance; mean expected minimum distance; minimum expected mean distance; maximum expected conflict; mean expected conflict; maximum vehicle-neighbor conflict; maximum pedestrian-neighbor conflict. Radius=50 m, M=8, sigma=2 m and label temperature=1 m are fixed without search.',
        '', 'Normalization uses only HeadTrain630 full-horizon actor-modes: population mean/std and (x−mean)/(std+1e-6) for continuous columns 3–18, including base probability. One-hot columns stay unchanged. No-neighbor raw distance sentinels are excluded from the three distance statistics. After normalization, absent-neighbor distances become one and conflict columns become zero; missing vehicle or pedestrian neighbor conflict becomes zero. R1 sets all seven normalized interaction columns exactly zero; R2 uses all seven. Their first twelve normalized inputs are identical.',
        '', 'Both heads are Linear(19,32) → ReLU → Linear(32,1), applied to each mode. z′=z+Δz and p′=softmax(z′). Both receive identical initial weights; last Linear weight and bias are zero. Initial shared-parameter difference is zero; step0 real-batch logits and probabilities exactly reproduce R0. Each head has 673 parameters.',
        '', '【No Future Leakage】','',
        'The feature API takes only historical positions/padding, actor type, frozen predicted ego trajectories and frozen original logits/probabilities. It has no GT, future mask, target mask, map, future label or refined-neighbor probability argument. Nonneutral-head perturbation tests independently alter target GT, neighbor GT, future masks, target masks and future labels; features and probabilities remain bitwise equal. Positive controls changing neighbor predicted trajectories or base probabilities alter interaction features. Future GT is used only to construct training labels, compute offline metrics/bins and select illustrative cases.',
        '', '【Head Training Split】','',
        'Sorted official TRAIN700 scene tokens are permuted by NumPy default_rng(2022). First70 are HeadDev; the other630 are HeadTrain. No scene overlap exists. HeadTrain has260,151 full-horizon targets (191,026 vehicle;66,145 pedestrian;2,980 bicycle); HeadDev has29,934 (22,719;7,057;158). Partial targets are excluded from normalization and ranking supervision, while current-valid context actors can contribute frozen future predictions.',
        '', 'HeadDev is held out from head optimization only: the previously trained frozen predictor already saw all TRAIN700 scenes. The frozen predictor checkpoint was also selected in the earlier Stage5A experiment on official VAL. Thus this stage uses official VAL only once for final ranking evaluation, but the overall experiment chain is not an untouched test evaluation.',
        '', 'Soft targets are q=softmax(−FDE_k/1 m), using full-horizon endpoint errors. The only loss is −sum q log p′. Both heads use seed2022, the same actor batches/permutations, batch1024, AdamW LR1e-3, weight_decay1e-4, maximum50 epochs. Each epoch evaluates HeadDev Top1FDE; strict improvement and patience5 select the best post-epoch checkpoint. Both stop after7 epochs with best epoch2. R1 best trained HeadDev FDE is slightly worse than the neutral R0; step0 was audited separately and was not an eligible trained checkpoint. No extra head, seed, feature selection or retraining follows.',
        '',markdown(headrows,('Variant','Parameters','Best epoch','Epochs executed','HeadDev Top1FDE','HeadDev step0 Top1FDE')),
        '', '\n'.join(f"{h['variant']} checkpoint SHA256: `{h['checkpoint_sha256']}`." for h in training['heads']),
        '',f"Split SHA256: `{sha256(SPLIT)}`. Normalization SHA256: `{sha256(NORM)}`. Configuration SHA256: `{sha256(CONFIG)}`.",
        '', '【Geometry Identity Audit】','',
        'PASS. Fresh frozen-predictor VAL outputs are bitwise identical to the registered deterministic cache in all3,603 windows. R0/R1/R2 share exactly the same raw and ego candidate tensors; the two head APIs return only ranking. Actor identity, future mask and GT fingerprints pair exactly with the frozen Stage3B/Stage5A ledger. minADE6/minFDE6/MR6 maximum difference across R0/R1/R2 is zero (<1e-8); predictor state hash is unchanged and no predictor gradient tensors exist.',
        '',f"Overall shared minADE6={float(overall['R0_minADE6']):.9f} m, minFDE6={float(overall['R0_minFDE6']):.9f} m, MR6={float(overall['R0_MR6']):.9f}. minADE6 retains the prior ADE-of-best-FDE-mode convention; MR6 means best endpoint error>2 m.",
        '',f"Historical Stage5A CSV reconciliation is a separate check, because its earlier GPU run did not use the deterministic runtime. Maximum group-mean differences are {pair['historical_max_mean_differences']}, all below1e-6. Historical old CSV tensors are not claimed bitwise identical. The exact invariant concerns frozen Stage5A predictions versus their current R1/R2 rerankings; all ranking comparisons use freshly reproduced R0.",
        '', '【Main Ranking Results】','',
        'Official VAL:150 scenes,3,603 supervised windows,54,990 full-horizon +30,037 partial =85,027 supervised actor-windows. Primary tables use only the54,990 full targets, representing4,323 distinct actor instances. Complete identity pairing and finite checks pass (NaN=0, Inf=0). Values are actor-window means, not scene means. Probabilities tie by smallest mode index; best-mode probability ranks use stable descending order. The GT best mode is the first minimum-FDE index.',
        '',markdown(rows,mainfields),
        '',markdown(rankrows,('Variant',)+rankfields),
        '', 'Source tables: [main ranking results](../06_tables/stage6a_main_ranking_results.csv), [reliability ablation](../06_tables/stage6a_reliability_ablation.csv). The comparison chain is B=Stage3B, E=Stage5A/R0, E+R1 and E+R2. Differences between B and R2 in this table are descriptive; the paired bootstrap comparisons registered for this stage are R1−R0, R2−R0 and R2−R1.',
        '', '【Oracle Gap】','',
        f"Overall mean oracle gap is R0={float(overall['R0_OracleGap_FDE']):.6f}, R1={float(overall['R1_OracleGap_FDE']):.6f}, R2={float(overall['R2_OracleGap_FDE']):.6f} m. R2 closes only part of the gap; it generates no new geometry. The gap delta equals the Top1FDE delta up to float32 subtraction rounding, and is not an independent geometric improvement.",
        '', '【Hit Rate】','',
        f"Overall best-FDE-mode hit rate is R0={100*float(overall['R0_Top1HitRate']):.6f}%, R1={100*float(overall['R1_Top1HitRate']):.6f}%, R2={100*float(overall['R2_Top1HitRate']):.6f}%. R2 improves the point estimate, but the paired HitRate CI spans zero; this is not a statistically resolved hit-rate gain. Mean best-mode rank and MRR are reported above. Moving vehicle hit rates are {100*float(bygroup['vehicle.moving']['R0_Top1HitRate']):.6f}%, {100*float(bygroup['vehicle.moving']['R1_Top1HitRate']):.6f}% and {100*float(bygroup['vehicle.moving']['R2_Top1HitRate']):.6f}%.",
        '', '【R1 vs R2】','',
        'R1 and R2 have identical head architecture, parameter budget, initialization, optimization and checkpoint-selection protocol. Their only input difference is the seven future interaction columns. R1 slightly worsens Overall Top1FDE, with a paired CI above zero. R2 improves Overall Top1FDE relative to both R0 and R1. Under this fixed protocol, the improvement is supported by the interaction-enabled head rather than a generic extra MLP alone. This does not isolate individual interaction features or establish causality for observed physical collisions.',
        '',markdown([r for r in net if r['Group']=='overall'],('Comparison','Count','Number_changed','Number_improved','Number_worsened',
            'Top1_changed_rate','Improved_rate_among_changed','Worsened_rate_among_changed',
            'Mean_improvement_when_improved_m','Mean_degradation_when_worsened_m','Net_Top1FDE_delta')),
        '', 'Changed rates use all full actor-windows as denominator. Improved/worsened conditional rates use only changed Top1s. Improved means strictly smaller endpoint error, worsened strictly larger. The full table also includes rates over all actors and equal changes. Net delta=(worsened count×mean harm−improved count×mean gain)/total count; an independent reconstruction matches within1e-12 m. Improvements and degradations coexist.',
        '', '【Interaction-sensitive Groups】','',
        'The original frozen20 m membership CSV and its SHA are reused without recomputation. These subgroup definitions are distinct from the fixed50 m/nearest8 inference features. Heterogeneous20 m includes25,113 full targets; vehicle heterogeneous14,813; pedestrian heterogeneous9,732; VP context23,209. R2−R1 is the controlled interaction contrast; R2−R0 subgroup intervals may cross zero even where the controlled R2−R1 interval is below zero.',
        '',markdown(interaction,mainfields),
        '', 'Interaction feature distributions are recorded by candidate mode, actor type and frozen context group in [the distribution table](../06_tables/stage6a_interaction_feature_distributions.csv). No-neighbor raw distance sentinels are excluded from distance distributions; conflict zeros remain included. First-layer column norms and one fixed actor-wise permutation per interaction feature use only HeadDev70 after training. The same actor permutation moves all six mode values together. These are descriptive diagnostics of correlated features; no features are removed and no new model is fit.',
        '',markdown([r for r in importance['rows'] if r['Column']>=12],('Feature','First_layer_weight_L2_norm',
            'HeadDev_permutation_Top1FDE_delta','HeadDev_permutation_rank_loss_delta','HeadDev_top1_changed_rate')),
        '', '【Bootstrap】','',
        'Paired whole-scene percentile bootstrap resamples the same150 official VAL scene clusters with replacement,1,000 replicates, seed2022, pooling actor-window sums/counts in each sampled set. It uses the same resampled weights for all contrasts, groups and metrics. Negative FDE/ADE/gap deltas favor the new model; positive HitRate deltas favor it. HitRate values in the following table are proportions, not percent. Full results cover16 groups×3 contrasts×4 metrics.',
        '',markdown([r for r in boot if r['Metric']=='Top1FDE' or r['Group']=='overall'],
            ('Comparison','Group','Metric','Delta','CI95_lower','CI95_upper','Count','Scenes')),
        '', 'Intervals condition on these fixed HeadDev-selected heads and the previously selected frozen predictor. One training seed is used; intervals do not capture training-seed variability or correct the descriptive final choice among fixed variants. No multiple-comparison correction is applied; secondary subgroup intervals are descriptive. Overlapping windows and nested subgroups are not independent actor observations. Bicycle support is limited to656 full actor-windows,71 instances and47 scenes. No SOTA or broad deployment claim follows from this experiment.',
        '', '【Case Studies】','',
        'Four deterministic purposive examples cover a moving vehicle with improved Top1, a pedestrian recovering its best-FDE mode, a frozen heterogeneous-context actor with R2 better than R1, and a failure where R2 still selects the wrong mode and increases endpoint error. Selection maximizes the specified gain or failure harm with stable identity tie-breaks and distinct instances. GT is used only for this offline selection. No new predictor forward is performed: original prediction and final ranking caches provide all curves/probabilities.',
        '',markdown(cases['figures'],('name','case_kind','agent_type','R0_Top1FDE','R1_Top1FDE','R2_Top1FDE')),
        '', 'Each case displays the same actor, GT, map, all six original candidates and matched axis limits in three panels, with the R0/R1/R2 selected mode and all six probabilities. Original12 future points are unmodified; no smoothing or interpolation. Float64 endpoint recomputation agrees with the source actor CSV within1e-4 m. Gray candidates can overlap because the frozen geometry is genuinely similar; they are not shifted for display. Cases illustrate possibilities and cannot estimate population benefit.',
        '', '【Efficiency】','',
        'Predictor=650,403 parameters; each complete variant adds one673-parameter head, giving651,076 and0.103474% increase. The two experiment heads are separate ablations, not combined at inference. R0 has no added feature/head stage; its overhead entries below are unmeasured reference dashes.',
        '',markdown([e0]+efficiency['models'],('Variant','Predictor_params','Head_params','Total_params','Parameter_increase_percent',
            'Mean_feature_extraction_ms','Mean_head_scoring_ms','Mean_ranking_overhead_ms')),
        '', 'Timing uses20 preloaded original16-window VAL batches,20 paired warm-up rounds and200 paired measurements, alternating R1/R2 execution order with CUDA events on the existing RTX3080 / torch2.5.1+cu124. Only extra feature extraction, normalization and head scoring are timed; predictor forward, disk I/O and H2D are excluded. R1 skips neighbor distance computation. Feature extraction includes Python loops and validation checks in this implementation; these batch latencies are not deployment-optimized throughput. Whole predictor memory/latency are not remeasured.',
        '', '【Scientific Interpretation】','',
        f"ReliabilityHead = **{science['ReliabilityHead']}**. FutureInteractionContribution = **{science['FutureInteractionContribution']}**. PaperUsableReliability = **{science['PaperUsableReliability']}**. RecommendedFinalVariant = **{science['RecommendedFinalVariant']}**.",
        '', 'The numerical interpretation rules were operationalized and saved before fitting: ReliabilityHead SUPPORTED requires at least one fixed head with Overall FDE CI upper<0, smaller gap, higher point-estimate HitRate and no severe main-class harm. PROMISING requires favorable point estimates with CI crossing zero. Future interaction SUPPORTED requires R2−R1 Overall FDE CI upper<0 plus improvement in at least one frozen context group and no severe class harm. Severe Vehicle/Pedestrian harm means a≥10% relative FDE increase together with a paired CI lower>0. This effect-size guard is an explicitly disclosed implementation of the qualitative requirement; it was not chosen after seeing results.',
        '', 'R2 satisfies these rules and is the lowest-Overall-FDE safe variant. Its improvement is modest and concentrated in ranking; R1 is NOT SUPPORTED. Neither main class exhibits severe collapse. HitRate improvement is unresolved by the paired CI; interaction subgroup and case findings support limited interpretation. Final R2 recommendation is descriptive selection among prespecified fixed heads on final VAL, not a new checkpoint/feature/threshold search. Stage5A prior NOT SUPPORTED and all prior conclusions remain unchanged.',
        '', 'All new code, configuration, small tables, audits, source-case JSON, reports and eight PNG300dpi/PDF/SVG bundles are versioned in this stage root. Prediction caches, tensor features, raw actor records, logs and head weights remain local; their sizes/counts/schemas/hashes are recorded. Old data/shards and earlier stage files are read-only; the five Stage2C redraw files remain untouched and unsubmitted. The artifact inventory excludes itself, transient Git payloads and the active finalizer log, whose contents are still being appended.',
        '', 'STOP. No joint fine-tuning, decoder changes, third reranker, extra seed, altered radius/M/sigma/temperature or next experiment is executed.','']
    report=ROOT/'09_reports/stage6a_final_report.md';report.write_text('\n'.join(text))
    assert git('branch','--show-current')=='stage6a/future-interaction-reliability'
    for path in git('diff','--name-only',BASE_COMMIT).splitlines():assert path.startswith(str(ROOT.relative_to(PROJECT))+'/')
    atomic_json(ROOT/'00_manifest/stage6a_pipeline_state.json',{'status':'COMPLETE','phase':'FINAL_AUDIT_AND_AUTHORIZED_BRANCH_PUSH','STOP_after_push':True})
    audit={'status':'PASS','stage':'Stage6A','base_commit':BASE_COMMIT,'branch':'stage6a/future-interaction-reliability',
        'previous_files_SHA256_unchanged':len(frozen['files']),'original_scene_shards_SHA256_unchanged':len(frozen['scene_shards']),
        'training_source_SHA256_unchanged':True,'formal_heads':2,'head_params':{'R1':673,'R2':673},
        'best_epochs':{h['variant']:h['best_epoch'] for h in training['heads']},'training_seed':2022,
        'official_VAL_ranking_passes':1,'VAL_windows':3603,'VAL_scenes':150,'full':54990,'partial':30037,'total':85027,
        'geometry':'PASS','maximum_geometry_metric_difference':0.,'cache_replay_unique_instances':120,
        'future_leakage':'PASS','neutral_initialization':'PASS','predictor_finetuning':False,
        'quantitative_figures':4,'cases':4,'manual_visual_QA':'PASS','report_sha256':sha256(report),
        'ReliabilityHead':science['ReliabilityHead'],'FutureInteractionContribution':science['FutureInteractionContribution'],
        'PaperUsableReliability':science['PaperUsableReliability'],'RecommendedFinalVariant':science['RecommendedFinalVariant'],
        'NaN':0,'Inf':0,'extra_heads_or_seeds':False,'STOP':True}
    atomic_json(ROOT/'00_manifest/stage6a_final_audit.json',audit)
    target=ROOT/'00_manifest/stage6a_artifact_manifest.json';artifacts=[]
    for i,p in enumerate(sorted(ROOT.rglob('*'))):
        if not p.is_file() or p==target or p.name.startswith('stage6a_git_upload_') or p.suffix in ('.tmp','.pyc') or '__pycache__' in p.parts or p.name=='stage6a_finalize.log':continue
        ignored=p.suffix in ('.pt','.log') or p.name.startswith('stage6a_actor_')
        artifacts.append({'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha256(p),'Git_eligible':not ignored})
        if len(artifacts)%200==0:print('ARTIFACT_HASH',len(artifacts),flush=True)
    expected={r['path']:r['sha256'] for r in artifacts}
    for r in read_json(ROOT/'01_cache/stage6a_cache_manifest.json')['batches']:assert expected[r['path']]==r['sha256']
    with (ROOT/'04_evaluation/stage6a_actor_ranking.csv').open() as f:schema=next(csv.reader(f))
    atomic_json(target,{'status':'PASS','stage':'Stage6A','root':str(ROOT),'artifacts':artifacts,'self_hash_excluded':True,
        'active_finalizer_log_excluded':True,'raw_actor_records':{'full':54990,'partial':30037,'total':85027,'schema_fields':schema},
        'cached_predictions_raw':'[6,N,12,4]','ego':'[N,6,12,2]','normalized_head_features':'[full_targets,6,19]'})
    print('STAGE6A_FINAL_AUDIT_PASS',audit,flush=True)

if __name__=='__main__':main()
