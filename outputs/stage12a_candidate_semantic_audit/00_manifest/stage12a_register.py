"""Freeze inputs and prespecify all signal/case/negative-control choices."""
from stage12a_common import *
import subprocess
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE and not REG.exists()
    proto={'Stage':'Stage12A','BaseCommit':BASE,'Branch':'stage12a/candidate-semantic-consistency-audit',
      'NeuralTraining':False,'NewCheckpoints':False,'ModifyCandidatesOrScores':False,
      'Population':{'Scenes':630,'Actors':260151,'Vehicle':191026,'Pedestrian':66145,'Bicycle':2980,'Source':'unchanged Stage11B 3-fold OOF'},
      'Evidence':'development exploration, prior OOF already inspected; no independent/cause/performance-gain claim',
      'Selector':'exact Stage8A-0C SparseSemanticIndex, candidate-specific future predicted polyline, no GT and no nearest fallback',
      'MapTypes':MAP_TYPES,'Quotas':QUOTAS.tolist(),'Thresholds':{'lane_connector_m':10,'polygon_m':2},
      'Geometry':'original Stage7A whole-region 2m centerlines; original valid polygon components, no union/buffer/repair; component OR',
      'SemanticFields':SEM_FIELDS,'GeometryInteractionFields':GEO_FIELDS,
      'CenterlineFeatures':'nearest selected lane/connector by whole-polyline minimum distance, selector order for ties; its mean12point/endpoint distance and acos clipped cos heading mismatch',
      'PolygonFeatures':'minimum selected-entity distance; strict containment OR over all selected valid original components; boundary crossing = any component boundary crossed by predicted polyline',
      'Missing':'zero finite values with explicit per-feature valid masks; preserve every actor and six modes',
      'Coordinate':'frozen t0 ego x-forward/y-left; row-vector rotation R(yaw).T plus origin; max roundtrip <1e-5m',
      'Heading':'candidate frozen last1s secant endpoint[-1]-[-3]; history last two valid observed positions, zero+missing when stationary',
      'Preflight':'lexicographically first HeadTrain scene in each of four map regions, up to 16 actor/windows per type per scene; use no error labels for sample choice',
      'PrimarySignals':{'Vehicle':[['centerline_mean_distance',1],['centerline_heading_error',1],['drivable_inside_fraction',-1]],
        'Pedestrian':[['crosswalk_distance',1],['walkway_inside_fraction',-1],['crosswalk_intersects',-1]]},
      'Association':'within-actor Spearman between signed semantic feature and six frozen FDE; at least3 valid modes with variation; scene-weighted means of actor rho',
      'SignalRule':{'PROMISING':'at least one registered feature: pooled oriented mean rho>=.05, >=500 actors and50 scenes; rho>0 in >=2 folds with >=100 actors and20 scenes each; paired real-minus-each-control delta>=.03 with descriptive95CI lower>0; C-minus-oracle signed feature difference>0 in >=2 folds',
        'WEAK':'pooled rho>=.05 and positive direction in >=2 eligible folds, but fails control or selected-mode checks',
        'NONE':'no registered feature meets weak reproducibility criterion; insufficient estimability explicitly reported'},
      'NegativeControls':{'seed':2022,'NC1':'independent row permutations of all six full semantic records per actor, including masks; coordinates,FDE,scores unchanged',
        'NC2':'permutation of flattened candidate semantic records within exact observed actor type and map region; masks/features distributions preserved',
        'Comparison':'common eligible actors across real and both controls for paired association; no change to frozen predictor outputs',
        'PASS':'permutations preserve registered distributions and do not change identity/scores; report absence of real-vs-control signal honestly'},
      'MatchedMotion':'within same actor and predicted-displacement bin [0,1,5,10,20,inf); >=3 valid varying candidate modes in a bin; average eligible-bin rho within actor; type,mapregion,history context fixed by actor',
      'IncrementalDecision':'SUGGESTED only if primary PROMISING and matched-motion association also rho>=.05 in >=2 folds and real-minus-both-controls paired95CI lower>0; otherwise UNRESOLVED for a promising/weak signal, NOT_OBSERVED if neither type has a weak signal; genuine independent prediction gains always unresolved',
      'FeatureCorrelations':'scene-level mean correlations semantic vs displacement,endpoint,heading,length,curvature and original interaction summary; descriptive, no fitted predictive model',
      'Switch':'FoldR2->C Raw; changed mode split by exact FDE delta sign; high-cost harm fixed >5m; mean/median/p90/p95/p99,gross gain/harm; no posthoc thresholds',
      'Bootstrap':{'replicates':2000,'seed':2022,'unit':'paired whole scene sampled within each frozen outer210 fold','CI':[2.5,97.5],'scope':'descriptive development evidence, unadjusted exploratory feature family'},
      'Cases':{'VehicleError':10,'PedestrianError':10,'VehicleSuccess':5,'PedestrianSuccess':5,
        'ErrorSort':'C!=oracle; descending(C FDE - minFDE), actor_id tie-break; prefer one per scene until quota filled; no map-coverage filtering',
        'SuccessSort':'R2!=C and C FDE<R2 FDE; ascending deltaFDE, actor_id tie-break; prefer one per scene; no semantic-feature selection'},
      'GO':'all engineering gates PASS and >=1 PROMISING type; report matched-motion evidence; CONDITIONAL_GO if engineering PASS and only WEAK; otherwise STOP',
      'ReadyForStage12B':'YES only Stage12A GO; actual execution must STOP after audit, any training requires new authorization',
      'Forbidden':['HeadDev70 evaluation/selection','official VAL150','test','training','reranking candidates','Stage12B implementation','loss-parameter search'],
      'StaticOnly':'no realtime traffic light colors, future occupancy or unobserved yielding',
      'Seed':2022,'Versions':{'numpy':np.__version__,'shapely':shapely.__version__,'torch':torch.__version__}}
    atomic_json(PROTOCOL,proto)
    old=read_json(S11C/'01_frozen_audit/stage11c_frozen_manifest.json');cps=old['checkpoints']
    for p,h in cps.items():assert sha256(PROJECT/p)==h,p
    stageb=read_json(S11B/'04_checkpoints/stage11b_all_frozen.json')
    assert len(stageb['Checkpoints'])==12
    for r in stageb['Checkpoints']:assert cps[r['Path']]==r['SHA256']
    mapfiles=[*MAP_JSON.glob('*.json'),CENTERLINES,SEMANTICS,S8C/'01_selector/stage8a0c_selector.py',S8C/'00_manifest/stage8a0c_common.py',
      S8C/'02_graph_cache/stage8a0c_entity_dictionary.json',S8C/'02_graph_cache/stage8a0c_selector_manifest.json',S8/'02_graph_cache/stage8a_graph_cache_manifest.json']
    maps={str(p.relative_to(PROJECT)):sha256(p) for p in mapfiles}
    oldschema=read_json(S8/'00b_type_aware_semantic_audit/01_schema_audit/stage8a0b_schema_audit.json')
    for p in MAP_JSON.glob('*.json'):assert sha256(p)==oldschema['map_sources'][p.name]
    data=dict(old['data_files'])
    for n,h in read_json(TRAIN/'manifest.json')['files'].items():
        if n.startswith('arg') and n not in ['arg3.npy','arg4.npy','arg5.npy']:continue
        p=TRAIN/n;assert sha256(p)==h;data[str(p.relative_to(PROJECT))]=h
    for n,h in read_json(S11A/'01_identity_audit/cache/prepared.json')['files'].items():
        p=S11A/'01_identity_audit/cache'/n;assert sha256(p)==h;data[str(p.relative_to(PROJECT))]=h
    for n,h in read_json(S11C/'03_oof_evaluation/cache/stage11c_evaluation_complete.json')['cache_files'].items():
        p=S11C/'03_oof_evaluation/cache'/n;assert sha256(p)==h;data[str(p.relative_to(PROJECT))]=h
    tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines()
    historical={p:sha256(PROJECT/p) for p in tracked}
    preserved=old['preserved_untracked_files']
    for p,h in preserved.items():assert sha256(PROJECT/p)==h
    atomic_json(FROZEN,{'Status':'PASS','BaseCommit':BASE,'checkpoints':cps,'map_sources':maps,'data_files':data,
      'historical_files':historical,'preserved_untracked_files':preserved,
      'HistoricalConclusions':'Stage8/9/10/11A/11B/11C unchanged','Stage11BCheckpoints':stageb['Checkpoints']})
    atomic_json(REG,{'Status':'REGISTERED_BEFORE_SEMANTIC_EXTRACTION_AND_LABEL_ANALYSIS','protocol_sha256':sha256(PROTOCOL),
      'requirements_sha256':sha256(ROOT/'00_manifest/stage12a_requirements.txt'),'frozen_manifest_sha256':sha256(FROZEN)})
    verify();print('STAGE12A_REGISTERED_FROZEN_PASS',len(historical),len(cps),flush=True)
if __name__=='__main__':main()
