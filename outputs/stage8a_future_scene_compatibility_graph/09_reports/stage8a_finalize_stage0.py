"""Finalize feasibility gates only. No scientific support claim before head training."""
from pathlib import Path
import sys,ast,json,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage8a_common import *
from stage8a_graph import NODE_FIELDS,INTERACTION_FIELDS,ObservableWindow
from stage8a_map import MAP_NODE_FIELDS,MAP_EDGE_FIELDS,SEMANTIC_FIELDS
import pandas as pd

def table(f):
    def cell(v):return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v).replace('|','\\|')
    return '\n'.join(['| '+' | '.join(f.columns)+' |','| '+' | '.join(['---']*len(f.columns))+' |',
        *['| '+' | '.join(cell(v) for v in row)+' |' for row in f.itertuples(index=False,name=None)]])

def main():
    frozen=verify_frozen(shards=True)
    identity=read_json(ROOT/'01_cache_audit/stage8a_candidate_identity.json')
    coordinate=read_json(ROOT/'01_cache_audit/stage8a_coordinate_audit.json')
    leak=read_json(ROOT/'03_model_audit/stage8a_gt_leakage_audit.json')
    graph_audit=read_json(ROOT/'03_model_audit/stage8a_graph_definition_audit.json')
    init=read_json(ROOT/'03_model_audit/stage8a_initialization_manifest.json')
    memory=read_json(ROOT/'03_model_audit/stage8a_memory_audit.json')
    manifest=read_json(ROOT/'02_graph_cache/stage8a_graph_cache_manifest.json')
    assert all(a['status']=='PASS' for a in (identity,coordinate,leak,graph_audit,init,memory,manifest))
    assert identity['candidate_maxdiff']==identity['logit_maxdiff']==identity['prob_maxdiff']==0
    assert not manifest['training_executed'] and memory['optimizer_steps']==0 and memory['new_checkpoint_count']==0
    assert not any((ROOT/'07_checkpoints').glob('*.pt')) and not any((ROOT/'04_training').glob('*'))
    expected={r['source_cache_path']:r['source_cache_sha256'] for r in manifest['batches']}
    source=read_json(CACHE_MANIFEST);assert expected=={r['path']:r['sha256'] for r in source['batches']}
    for row in manifest['batches']:assert sha256(ROOT/'02_graph_cache'/row['path'])==row['sha256'],row['path']
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
    for p in ROOT.rglob('*.py'):
        code=ast.parse(p.read_text())
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='backward' for n in ast.walk(code))
    cov=pd.read_csv(ROOT/'06_tables/stage8a_map_coverage.csv');stats=pd.read_csv(ROOT/'06_tables/stage8a_graph_statistics.csv')
    primary=cov[cov.Population=='current_valid'];ranking=cov[cov.Population=='full_horizon_ranking_targets']
    all_graph=stats[(stats.Population=='current_valid')&(stats.Group=='Overall')]
    combined=[]
    for group in ('Overall','Vehicle','Pedestrian','Bicycle','MovingVehicle'):
        c=primary[primary.Group==group];count=int(c.Candidates.sum())
        combined.append({'Group':group,'Candidates':count,'CoverageWithin10m':float(c.AtLeast1Within10m.sum()/count),
            'AtLeast4TokensWithin10m':float((c.AtLeast4Within10mRate*c.Candidates).sum()/count),
            'FallbackRate':float(c.FallbackCandidates.sum()/count),'MeanMapTokensPerMode':float((c.MeanMapTokensPerMode*c.Candidates).sum()/count)})
    write_csv(ROOT/'06_tables/stage8a_combined_map_coverage.csv',combined)
    failure=[]
    for row in primary[primary.Group=='Vehicle'].itertuples():
        if row.FallbackRate>.05:failure.append(f'{row.Split} Vehicle fallback={row.FallbackRate:.6%} exceeds fixed5%')
    if max(x['Parameters'] for x in init['variants'])>150000:failure.append('Parameter estimate>150k')
    decision={'Stage8A_0':'NOT_READY' if failure else 'READY','failures':failure,'candidate_identity':'PASS','coordinate_audit':'PASS',
        'semantic_attachment':graph_audit['semantic_attachment'],'GT_leakage_audit':'PASS','memory_audit':'PASS',
        'G1_G2_G3_training_executed':False,'tiny_training_executed':False,'Stage8A_1_started':False,'test_used':False,
        'map_radius_m':10.,'map_topM':8,'neighbors_max':8,'neighbor_radius_m':50.,'thresholds_changed':False,
        'counts':manifest['counts'],'combined_map_coverage':combined,
        'mean_neighbor_actors_per_target':float((all_graph.MeanNeighborActors*all_graph.Targets).sum()/all_graph.Targets.sum()),
        'mean_mode_mode_edges_per_target':float((all_graph.MeanModeModeEdgesPerTarget*all_graph.Targets).sum()/all_graph.Targets.sum()),
        'mean_mode_map_edges_per_target':float((all_graph.MeanModeMapEdgesPerTarget*all_graph.Targets).sum()/all_graph.Targets.sum()),
        'estimated_G3_parameters':next(x['Parameters'] for x in init['variants'] if x['Variant']=='G3'),
        'graph_cache_bytes':manifest['total_graph_cache_bytes'],'estimated_padded_raw_bytes_per_target':memory['dense_padded_float32_input_bytes_per_target'],
        'G3_prototype_batch128_peak_CUDA_MiB':memory['rows'][-1]['PeakCUDAAllocatedMiB'],'STOP':True}
    atomic_json(ROOT/'00_manifest/stage8a_stage0_decision.json',decision)
    fields={'node':list(NODE_FIELDS),'interaction_edge':list(INTERACTION_FIELDS),'map_node':list(MAP_NODE_FIELDS),'map_edge':list(MAP_EDGE_FIELDS),
        'feature_dimensions':{'node':15,'interaction_edge':17,'map_node':13,'map_edge':17},
        'history_motion':'raw meters, valid chronological observed history; bridge missing-history gaps like frozen Stage5A',
        'candidate_path':'t0 current position then12 predicted future positions; mean speed=path/6s',
        'candidate_heading':'last1s secant prediction[-1]-prediction[-3]; atan2(0,0)=0 for stationary secants',
        'minimum_relative_displacement':'minimum norm of target candidate step displacement minus neighbor candidate step displacement over12 steps, including t0->first step',
        'closest_interaction_time':'(first argmin index+1)/12','closing_tendency':'(distance at first future point-distance at endpoint)/6s; descriptive signed tendency',
        'map_geometry':'12 future points only form a continuous predicted polyline; complete frozen centerline, distinct tokens',
        'map_direction':'directed centerline tangent at nearest polyline-centerline point; transformed into t0 ego coordinates',
        'map_closest_time':'fractional nearest polyline segment index+1, divided by12; stationary polylines use first future point1/12',
        'semantics':'frozen9D vector retained in map node and mode-map edge; duplicate connector flag is intentional',
        'GT_is_not_a_feature':True,'node_features_not_standardized_or_fitted_in_Stage0':True}
    atomic_json(ROOT/'00_manifest/stage8a_feature_dictionary.json',fields)
    audit={'status':'PASS','historical_files_verified':len(frozen['files']),'scene_shards_verified':len(frozen['scene_shards']),
        'unrelated_untracked_preserved':len(frozen['unrelated_untracked']),'source_candidate_cache_batches':len(manifest['batches']),
        'all_candidate_cache_SHA_verified':True,'all_graph_cache_SHA_verified':True,'new_checkpoint_count':0,'optimizer_steps':0,
        'graph_inputs_contain_GT':False,'official_VAL_training_used':False,'head_split_reused':True,
        'graph_cache_manifest_sha256':sha256(ROOT/'02_graph_cache/stage8a_graph_cache_manifest.json'),
        'source_SHA256':{str(p.relative_to(ROOT)):sha256(p) for p in ROOT.rglob('*.py')},'merged_main':False}
    atomic_json(ROOT/'00_manifest/stage8a_final_integrity_audit.json',audit)
    parts=[]
    def section(name,text):parts.append('【'+name+'】\n\n'+text+'\n')
    section('Scope','Stage8A-0 only. All new files and this execution plan live under outputs/stage8a_future_scene_compatibility_graph/. No head fitting, tiny overfit, optimizer step, new checkpoint or official ranking comparison was run. G1/G2/G3 here are feasibility prototypes only. Frozen Stage7A and Stage7D NOT_SUPPORTED decisions remain unchanged. Stage8A-1 requires separate review and authorization even if readiness passes.')
    section('Frozen Candidate Predictor',f'Stage5A checkpoint SHA256={PREDICTOR_SHA}. Stage6A R2 checkpoint SHA256={R2_SHA}; its historical Overall Top1FDE=2.645563 is retained as the future strong baseline, not a new Stage8A measurement. Official TRAIN700/VAL150, K6/Tf12. Candidate cache is referenced, never regenerated/overwritten. HeadTrain630/HeadDev70 split SHA256={sha256(SPLIT)}; future head checkpoint selection must use HeadDev ranking loss as the new requirements specify, not the old R2 Top1FDE selection and never official VAL.')
    section('Candidate Identity',f"PASS. Seed2022 random100 TRAIN +100 VAL windows replayed in original batch16 membership. All actors in these windows checked: {identity['actors_checked']}. Maximum raw/candidate/logit/probability/rotation differences are exactly0. Identities, actor types, history and masks are bitwise identical. Predictor eval()/requires_grad=False, gradient count0, state unchanged. Entire1283 source cache batches were SHA-verified during coverage construction. R0/R2/G1/G2/G3 use references to the same cached geometry; graph head outputs only logits/probabilities. Trained G1/G2/G3 oracle/ranking metrics are not yet evaluated.")
    section('Map Coordinates','```json\n'+json.dumps(coordinate,indent=2)+'\n```\n\nBoth endpoints identify original segments even where the unchanged source centerline contains repeated vertices. Whole-region brute-force retrieval and global/local distance agreement verify the coordinate/index mapping. No centerline repair, semantic change or radius tuning occurred.')
    section('Graph Definition','```json\n'+json.dumps(fields,indent=2)+'\n```\n\nEach target has6 mode nodes and at most8 current-valid nearest neighbor actors with6 modes each: at most54 local mode nodes and288 directed mode-mode relations. Stable current-distance neighbor selection matches Stage6A. Each6x6 neighbor-mode pair remains an independent17D relation before a learned64D message and softmax attention; no pre-aggregation into min/mean/max across neighbors. Map messages use separate retrieved tokens with geometry and frozen semantics. One LayerNorm update of node+interaction+map messages and a zero-final-layer64-32-1 head changes logits only.')
    section('Graph Coverage',table(primary)+'\n\nCoverage is measured before fallback: at least1 token within10m, at least4 distinct tokens within10m. Nearest-one fallback yields100% nonempty retrieved sets, but is never counted as within-radius coverage. Primary counts include all current-valid context/partial/full actor modes. This matches graph-neighbor inference availability and avoids selecting graph nodes using future labels. Full-horizon ranking-target-only coverage is separately shown below; its future mask is an audit population label, never a graph input.\n\n'+table(ranking)+'\n\nCombined TRAIN/VAL:\n\n'+table(pd.DataFrame(combined)))
    section('Graph Statistics',table(stats)+'\n\nNeighbor-count distributions0..8 are in stage8a_neighbor_distribution.csv. Source cache counts and dataset membership agree exactly. Retrieval cache stores token indices, distances, neighbor selectors and current frame metadata; candidate tensors/history/GT are not copied into it. Node/edge features are constructed on demand from the frozen source predictions. Maximum current-valid actors per scene-window='+str(manifest['maximum_current_valid_actors_per_window'])+'.')
    section('Semantic Attachment',f"PASS. All8251 frozen lane/connector tokens have exact Stage7A semantic_vector labels. On16 real graph windows, {graph_audit['semantic_attachment_edges_checked']} mode-map relations checked bitwise. Turns20°/150°, static control associations and centerline-crosswalk intersection definitions are unchanged. No dynamic signal state, priority label or invented right-of-way is used.")
    section('No-Future-Leakage Audit','```json\n'+json.dumps(leak,indent=2)+'\n```\n\nThe ObservableWindow dataclass is an explicit observed/predicted allowlist. Access-protected GT/future-mask/target-mask fields raise if read; changing their contents leaves every node, selector, interaction relation and map relation identical. GT labels are not constructed in this stage. Later ranking supervision/evaluation belongs outside graph construction.')
    section('Neutral Initialization',f"PASS. Shared node encoder, LayerNorm and scoring head initialize bitwise identically withseed2022. Interaction/map branches use fixed independent2123/2124 initialization. Final head linear weight/bias exact0. Neutral logit maxdiff0, same-device probability maxdiff0. CPU versus original GPU cached softmax maxdiff={graph_audit['neutral_probability_maxdiff_vs_original_GPU_cache']:.9g}, below1e-6; exact predictor replay on the original device has probability maxdiff0. No claim that these untrained graph heads already improve ranking.")
    section('Memory and Parameter Audit',table(pd.read_csv(ROOT/'06_tables/stage8a_efficiency.csv'))+f"\n\nG3 parameters={decision['estimated_G3_parameters']}, below preferred50k and mandatory150k pause threshold. Dense padded raw graph inputs={memory['dense_padded_float32_input_bytes_per_target']} bytes/target. Batch128 prototype forward CUDA peak={decision['G3_prototype_batch128_peak_CUDA_MiB']:.6f}MiB. Device={memory['device']}. Actual compressed retrieval cache={manifest['total_graph_cache_bytes']/1024**2:.3f}MiB; candidates remain in the old1.19GB cache. Inference only; autograd training activations and optimizer memory have not been measured. Cache generation includes file loading/spatial lookup and is not presented as real-time predictor or reranker latency.")
    section('Limitations','Map entities here are road lane/connector centerlines with frozen static semantics. Within10m coverage may be poor for some candidate locations; nearest-one fallback guarantees a nonempty set but does not make a distant relation reliable. Coverage and map-only geometry do not establish semantic usefulness or ranking gains. One fixed graph architecture, no radius/M/hidden/neighbor search. No G1/G2/G3 training, formal ablation, bootstrap, ranked case studies or scientific support decision has been performed. Current readiness is separate from AgentGraph/SemanticGraph/FSCG effectiveness.')
    section('Stage8A-0 Decision','```json\n'+json.dumps(decision,indent=2)+'\n```\n\n'+('The fixed Vehicle fallback gate fails. Stage8A-0=NOT_READY; do not train G1/G2/G3 or start Stage8A-1. No retrieval parameter was adjusted to obtain a pass.' if failure else 'All feasibility gates pass. Stage8A-0=READY; still do not train G1/G2/G3 or start Stage8A-1 before separate review.')+'\n\nSTOP. Push only this stage branch, do not merge main. Await review.')
    (ROOT/'09_reports/stage8a_stage0_report.md').write_text('\n'.join(parts))
    print('STAGE8A_FINAL_INTEGRITY_PASS',json.dumps(decision,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
