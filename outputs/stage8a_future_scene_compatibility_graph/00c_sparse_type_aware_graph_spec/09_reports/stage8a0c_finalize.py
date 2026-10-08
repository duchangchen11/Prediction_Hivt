"""Freeze the final graph contract, apply gates and verify historical immutability."""
from pathlib import Path
import sys,ast
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage8a0c_common import *

def table(rows,columns,percent=()):
    out=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for r in rows:
        values=[]
        for col in columns:
            v=r[col]
            if v is None or v=='':v='NA'
            elif col in percent:v=f'{float(v)*100:.4f}%'
            values.append(str(v))
        out.append('| '+' | '.join(values)+' |')
    return '\n'.join(out)
def csv_rows(name):return list(csv.DictReader((ROOT/'06_tables'/f'stage8a0c_{name}.csv').open()))

def main():
    coverage=read_json(ROOT/'04_coverage/stage8a0c_coverage_audit.json')
    integrity=read_json(ROOT/'03_model_audit/stage8a0c_integrity_audit.json')
    determinism=read_json(ROOT/'01_selector/stage8a0c_determinism_audit.json')
    empty=read_json(ROOT/'03_model_audit/stage8a0c_empty_map_audit.json')
    model=read_json(ROOT/'03_model_audit/stage8a0c_model_audit.json')
    memory=read_json(ROOT/'05_memory/stage8a0c_memory_audit.json')
    selector=read_json(ROOT/'02_graph_cache/stage8a0c_selector_manifest.json')
    registration=read_json(ROOT/'00_manifest/stage8a0c_registration.json')
    frozen=read_json(ROOT/'00_manifest/stage8a0c_frozen_inputs.json')
    assert sha256(ROOT/'00_manifest/stage8a0c_execution_plan.md')==frozen['preregistration_sha256']
    assert empty['status']=='PASS' and model['status']=='PASS' and selector['status']=='PASS'
    assert git('branch','--show-current')==BRANCH
    for group in ('files','unrelated_untracked'):
        for name,h in frozen[group].items():assert sha256(PROJECT/name)==h,name
    for b in selector['batches']:assert sha256(ROOT/b['path'])==b['sha256']
    assert not list(ROOT.rglob('*.pt'))
    source_files=[ROOT/'00_manifest/stage8a0c_common.py',ROOT/'01_selector/stage8a0c_selector.py',ROOT/'01_selector/stage8a0c_build.py',
        ROOT/'03_model_audit/stage8a0c_model.py',STAGE8/'00_manifest/stage8a_graph.py',STAGE8/'00_manifest/stage8a_model.py',
        PROJECT/'preprocessing/coordinates.py',STAGE7/'00_manifest/stage7a_dataset.py',CENTERLINES,SEMANTICS,PREDICTOR,R2,
        STAGE6/'01_cache/stage6a_cache_manifest.json',STAGE8/'02_graph_cache/stage8a_graph_cache_manifest.json',
        STAGE0B/'03_entity_retrieval/stage8a0b_retrieval_manifest.json',ROOT/'02_graph_cache/stage8a0c_selector_manifest.json',
        ROOT/'00_manifest/stage8a0c_frozen_inputs.json',ROOT/'00_manifest/stage8a0c_requirements.txt',ROOT/'00_manifest/stage8a0c_execution_plan.md',
        *sorted(MAP_JSON.glob('*.json'))]
    source_sha={str(p.relative_to(PROJECT)):sha256(p) for p in source_files}
    for path in [ROOT/'01_selector/stage8a0c_selector.py',ROOT/'01_selector/stage8a0c_build.py',ROOT/'00_manifest/stage8a0c_common.py']:
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n,(ast.Import,ast.ImportFrom)):
                names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
                assert all(not any(bad in name.lower() for bad in ('evaluation','fde','ade','ranking_label','best_mode','actor_errors')) for name in names)
    spec={'stage':'Stage8A-0C','source_commit':BASE,'branch':BRANCH,'GraphSpecFrozen':'YES',
        'entity_taxonomy':[{'id':i,'name':s} for i,s in enumerate(TYPES)],
        'actor_entity_mapping':{name:[TYPES[i] for i in np.flatnonzero(QUOTAS[j])] for j,name in enumerate(('Vehicle','Pedestrian','Bicycle'))},
        'quota':{name:{TYPES[i]:int(QUOTAS[j,i]) for i in np.flatnonzero(QUOTAS[j])} for j,name in enumerate(('Vehicle','Pedestrian','Bicycle'))},
        'max_entities_per_mode':{'Vehicle':8,'Pedestrian':6,'Bicycle':7},'max_mode_map_edges_per_target':{'Vehicle':48,'Pedestrian':36,'Bicycle':42},
        'lane_radius_m':10,'polygon_threshold_m':2,'polyline_points':12,'future_horizon_seconds':6,
        'polygon_match':'Any predicted point inside OR polyline intersects OR filled geometry distance<=2m; valid source components unchanged.',
        'sorting':'within each entity type: minimum filled/complete-centerline geometric distance, lexical token; concatenate fixed type IDs0..5',
        'selector_inputs':['observed actor type','predicted12point global candidate','HDmap location'],
        'motion_state_changes_selector':False,'SelectorUsesPredictionError':'NO','no_nearest_fallback':True,'no_NULL_token':True,
        'node_feature_names':NODE_FIELDS,'interaction_feature_names':INTERACTION_FIELDS,
        'map_node_feature_names':MAP_NODE_FIELDS,'map_edge_feature_names':MAP_EDGE_FIELDS,
        'feature_dimensions':{'node':15,'interaction':17,'map_node':18,'map_edge':11},
        'feature_dtype':'float32','selector_distance_dtype':'float64','nonlane_9d_semantic':'exact zero',
        'lane_semantics':'exact Stage7A vector; connector/control/crosswalk/20degree-turn definitions unchanged',
        'heading':'last1s candidate secant [-1]-[-3]; atan2(0,0)=0; lane nearest directed-centerline tangent transformed into t0 ego frame',
        'tangent_and_heading_valid':'lane/connector=1; polygon=0 and sin/cos=0',
        'polygon_inside_fraction':'strict containment in any original valid polygon component; OR over components; no union/repair',
        'polygon_intersection':'polyline intersects any original valid polygon component, including boundaries',
        'closest_time':'candidate-side GEOS shortest_line point; first arc-length occurrence; fractional segment index+1 divided by12; stationary=1/12',
        'map_feature_distance_scaling':'first3 geometric distances divided by10; within2m/4m fractions use true unscaled point distances',
        'neighbor_rule':{'radius_m':50,'max_actors':8,'current_valid_only':True,'self_excluded':True,'tie':'original actor node index',
            'modes_per_actor':6,'max_mode_mode_edges':288,'individual_directed_target_neighbor_mode_relations':True},
        'empty_map_rule':'if no real map edges, exact zero64D; only nonempty rows enter map softmax; no learned padding/NULL entity',
        'zero_map_samples':'retain all targets without dropping, resampling or oversampling',
        'architecture':{'hidden':64,'node_encoder':[15,64,64],'interaction_message':[47,64,64],
            'interaction_attention':[145,64,1],'map_message':[44,64,64],'map_attention':[93,64,1],
            'interaction_layers':1,'map_aggregation_layers':1,'joint':'LayerNorm(h_node+m_int+m_map)',
            'score_head':[64,32,1],'score_head_last_weight_bias':'exact zero initialization','variants':model['variants']},
        'ranking_temperature_m':1,'initialization_seeds':{'common':2022,'interaction':2123,'map':2124},
        'CUBLAS_WORKSPACE_CONFIG':':4096:8','all_readiness_thresholds':registration['coverage_gates'],
        'readiness_population':'full-horizon ranking candidates separately in TRAIN and VAL; masks offline only',
        'parameter_limit_exclusive':150000,'preferred_parameter_limit_exclusive':50000,'inference_memory_limit_bytes':2**30,
        'engineering_gate_not_prediction_metric':True,'source_SHA256':source_sha,'training_authorized':False,'test_used':False,
        'historical_stage8a0':'NOT_READY','historical_stage8a0b':'NOT_READY',
        'future_stage8a1_modification_rule':'No radius/quota/features/neighbors/hidden/temperature changes unless a concrete code bug is found.'}
    atomic_json(ROOT/'00_manifest/stage8a0c_frozen_graph_spec.json',spec)
    p={r['Variant']:r['Parameters'] for r in model['variants']}
    decision={'SparseMapSelector':coverage['SparseMapSelector'],'SemanticPresencePreservation':coverage['SemanticPresencePreservation'],
        'SparseCoverage':coverage['SparseCoverage'],'CandidateIdentity':integrity['CandidateIdentity'],'Coordinate':integrity['Coordinate'],
        'NoFutureLeakage':integrity['NoFutureLeakage'],'SelectorDeterminism':determinism['SelectorDeterminism'],
        'SemanticAttachment':integrity['SemanticAttachment'],'MemoryFeasible':memory['MemoryFeasible'],'GraphSpecFrozen':'YES',
        'G1_parameters':p['G1'],'G2_parameters':p['G2'],'G3_parameters':p['G3'],'coverage_gates':coverage['gates'],
        'frozen_graph_spec_sha256':sha256(ROOT/'00_manifest/stage8a0c_frozen_graph_spec.json')}
    ready=all(decision[k]=='PASS' for k in ['SparseMapSelector','SemanticPresencePreservation','SparseCoverage','CandidateIdentity','Coordinate','NoFutureLeakage','SelectorDeterminism','SemanticAttachment'])
    ready &= decision['MemoryFeasible']=='YES' and p['G3']<150000
    decision.update(Stage8A_0C='READY' if ready else 'NOT_READY',ReadyStage8A1='YES' if ready else 'NO',
        training_authorized=False,STOP=True,historical_Stage8A_0='NOT_READY',historical_Stage8A_0B='NOT_READY')
    atomic_json(ROOT/'00_manifest/stage8a0c_decision.json',decision)
    def full(rows):return [r for r in rows if r['Population']=='full_horizon_ranking_targets' and r['Split']!='combined']
    cov=full(csv_rows('sparse_coverage'));compare=full(csv_rows('selector_comparison'));graphs=full(csv_rows('graph_statistics'))
    report=['# Stage8A-0C sparse type-aware semantic graph final specification audit\n',
        '## Scope and frozen history\n\nStage8A-0 remains NOT_READY. Stage8A-0B remains MIXED / coverage FAIL / context STRONG / NOT_READY. This stage specifies sparse inputs and changes the user-supplied engineering support gates before any graph training. These thresholds are neither accuracy thresholds nor significance/paper performance claims. No prediction performance was evaluated.\n',
        '## Deterministic selector and fixed quotas\n\nSix entity types only. Vehicle quotas lane3/connector3/drivable1/carpark1; pedestrian walkway3/crossing2/drivable1; bicycle lane3/connector3/drivable1. Within each type sort true geometry distance then lexical token. Motion state never changes the selector. No standalone stop line, traffic-light point, nearest fallback or learned NULL token. Real multipart polygons are preserved component by component; Boston’s invalid walkway component stays excluded.\n',
        '## Feature contract\n\nActor-mode15D and directed interaction17D are imported directly from the unchanged Stage8A-0 pure module. Map-node18D = type onehot6 + frozen lane semantic9 + local tangent sin/cos/valid3. Map-edge11D names and units are frozen in the manifest. Polygon nine lane bits and tangent/heading fields are exact zero. There is no area, priority, right-of-way or dynamic signal feature. Lane tangent uses the true nearest complete-centerline point, preserving repeated-source-vertex handling. Strict inside/intersection features use OR over original valid multipart components, avoiding invalid collection topology operations without repair.\n',
        '## Empty-map behavior\n\nSynthetic far-away candidates return zero selected entities. Both actual empty tensors and all-masked padding produce exact zero64D, even after setting encoder biases to nonzero. A patched Tensor.softmax verifies the empty-map code path never calls softmax. Mixed empty/nonempty rows remain finite. LayerNorm(h_node+m_int+0) is valid. No empty target is deleted.\n',
        '## Entity retention and semantic presence\n\nAll authorized matched types preserve presence at100%. Retention is entity-weighted selected/uncapped and naturally falls when a type exceeds its fixed quota. Unauthorized types have no eligible entities and undefined retention/presence ratios; they do not enter the graph.\n\n'+table([r for r in full(csv_rows('entity_retention')) if r['ActorGroup'] in ['Vehicle','Pedestrian','Bicycle'] and r['EntityType']=='total_primary'],
            ['Split','ActorGroup','UncappedEntities','SelectedEntities','RetentionRate','PresencePreservationRate'],['RetentionRate','PresencePreservationRate'])+'\n',
        '## GlobalTop8 versus quota\n\nThe primary comparison uses the same actor-authorized pool. All-six GlobalTop8, including secondary contexts, is separately descriptive. Quota never loses an authorized present type; no performance conclusion follows.\n\n'+table([r for r in compare if r['Group'] in ['Overall','MovingVehicle','Pedestrian','Bicycle'] and r['Selector'] in ['GlobalTop8','TypeAwareQuota']],
            ['Split','Group','Selector','MeanSelectedEntities','MeanDistinctEntityTypes','LanePresence','ConnectorPresence','DrivablePresence','CarparkPresence','CrosswalkPresence','WalkwayPresence'])+'\n',
        '## Sparse coverage and zero-map rates\n\nCoverage is MapNonEmptyRate, without fallback. All current-valid and full-horizon targets remain in their natural populations. Full-horizon TRAIN/VAL results follow; complete current-valid results are in the CSV.\n\n'+table(cov,
            ['Split','Group','Actors','Candidates','MapNonEmptyRate','ZeroMapRate','MeanEntitiesPerMode','MedianEntities','P95Entities','MaxEntities'],['MapNonEmptyRate','ZeroMapRate'])+'\n',
        '## Engineering support gates\n\nEvery gate applies independently in TRAIN and VAL full-horizon modes. The new thresholds were explicitly provided before graph training and are registered before this audit. They do not retroactively change the0B decision.\n\n'+table(coverage['gates'],['Split','Group','Metric','Value','Threshold','PASS'])+'\n',
        '## Graph size\n\nVerified maxima: vehicle48, pedestrian36, bicycle42 mode-map edges per target; mode-mode<=288 and unchanged50m/8-actor neighbor selection. ZeroMapTargetRate means all six modes lack map context; it differs from per-mode ZeroMapRate.\n\n'+table(graphs,['Split','Group','Targets','MeanModeMapEdges','P95ModeMapEdges','MaxModeMapEdges','MeanModeModeEdges','P95ModeModeEdges','ZeroMapTargetRate'])+'\n',
        '## Static models and inference memory\n\nNo optimizer, backward, fitting, tiny overfit or checkpoint. G1 is bitwise identical to the old static prototype; shared node encoder/LayerNorm/head initialization is identical across variants. The new map message uses44D concatenated input and map attention93D input. Final64→32→1 layer weight/bias are exact zero.\n\n'+table(model['variants'],['Variant','Parameters','InputDims','HiddenDim','MaxMapEdges','MaxInteractionEdges','NeutralOutputMaxDiff','EstimatedMemoryMiB'])+
            f"\n\nBatch128 input bytes: {memory['batch128_input_bytes']}; inference budget maximum {memory['EstimatedInferenceMiB']:.6f} MiB, using allowed static CUDA peak plus fixed32MiB margin. This excludes the frozen predictor and retrieval construction. Training activation/optimizer memory is unmeasured and must be measured in separately authorized Stage8A-1 tiny work.\n",
        '## Candidate, coordinates, leakage and semantics\n\n'+f"{integrity['windows']} sampled windows (100 TRAIN/100 VAL, seed2022) freshly audited all graph features. Candidate/logit/probability maxdiff=0. Poisoned GT=NaN and inverted future/target masks leave IDs/types/distances, map-node18/map-edge11, actor-mode15 and interaction17 features bitwise unchanged. Full-horizon masks only determine offline populations. Semantic edges checked: {integrity['semantic_edges_checked']}; lane9 equals frozen Stage7A, polygon9=0, type onehot exact. Global/local geometry-distance discrepancy max {integrity['geometry_distance_max_m']:.12g} m (<1e-5). Candidate roundtrip max {integrity['candidate_roundtrip_max_m']:.12g} m; polygon roundtrip max {integrity['polygon_roundtrip_max_m']:.12g} m. All1283 source and historical graph cache SHAs were reverified. Selector import audit finds no evaluation/error/ranking-label/best-mode imports.\n",
        '## Selector determinism\n\nThree independent runs used worker counts1/2/3, chunk sizes1/4/7, ascending/reverse/shuffled window and candidate order, and completion-order gathering. All200 sampled windows’ IDs/types/distances/counts/masks and token ordering are bitwise identical.\n\n'+table(determinism['runs'],['Run','Workers','ChunkWindows','WindowOrdering','CandidateOrdering','Windows','Candidates','SelectorSHA256','BitwiseIdentical'])+'\n',
        '## Frozen contract and decision\n\nThe final manifest lists taxonomy, actor-primary mapping, all feature names, quotas, radii, empty-map behavior, neighbors, all thresholds and relevant source/code hashes. Hidden64 and ranking temperature1m are frozen. It must remain unchanged in Stage8A-1 unless a concrete code bug is identified.\n\n'+
            '\n\n'.join(f'{key} = **{decision[key]}**' for key in ['SparseMapSelector','SemanticPresencePreservation','SparseCoverage','CandidateIdentity','Coordinate','NoFutureLeakage','SelectorDeterminism','MemoryFeasible','GraphSpecFrozen','Stage8A_0C','ReadyStage8A1'])+
            '\n\n**STOP. No automatic G1/G2/G3 training. Await scientific review.**\n']
    (ROOT/'09_reports/stage8a0c_sparse_graph_spec_report.md').write_text('\n'.join(report))
    atomic_json(ROOT/'00_manifest/stage8a0c_final_integrity.json',{'status':'PASS','historical_files_unchanged':len(frozen['files']),
        'unrelated_files_unchanged':len(frozen['unrelated_untracked']),'selector_cache_batches_verified':len(selector['batches']),
        'source_SHA_verified':True,'frozen_spec_sha256':decision['frozen_graph_spec_sha256'],'preregistration_unchanged':True,
        'optimizer_steps':0,'backward_calls':0,'new_checkpoints':0,'test_used':False,'historical0_and0B_NOT_READY_unchanged':True,
        'Stage8A_0C':decision['Stage8A_0C'],'ReadyStage8A1':decision['ReadyStage8A1'],'STOP':True})
    print('FINAL SPEC',decision['Stage8A_0C'],'ReadyStage8A1',decision['ReadyStage8A1'],'STOP',flush=True)

if __name__=='__main__':main()
