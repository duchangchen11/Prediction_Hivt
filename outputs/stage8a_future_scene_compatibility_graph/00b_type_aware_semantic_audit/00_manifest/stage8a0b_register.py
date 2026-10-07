from stage8a0b_common import *
from stage8a0b_entities import EntityIndex

assert git('rev-parse','HEAD')==BASE and git('branch','--show-current')==BRANCH
prefix=str(ROOT.relative_to(PROJECT))+'/'
tracked=git('ls-files').splitlines()
files={p:sha256(PROJECT/p) for p in tracked if not p.startswith(prefix)}
unrelated={p:sha256(PROJECT/p) for p in git('ls-files','--others','--exclude-standard').splitlines() if not p.startswith(prefix)}
extras=[SEMANTICS,CENTERLINES,PREDICTOR,R2,STAGE6/'01_cache/stage6a_cache_manifest.json',
        STAGE8/'02_graph_cache/stage8a_graph_cache_manifest.json',*sorted(MAP_JSON.glob('*.json'))]
files.update({str(p.relative_to(PROJECT)):sha256(p) for p in extras})
assert sha256(PREDICTOR)==PREDICTOR_SHA and sha256(R2)==R2_SHA
atomic_json(ROOT/'00_manifest/stage8a0b_frozen_inputs.json',{'base_commit':BASE,'files':files,'unrelated_untracked':unrelated,
    'requirements_sha256':sha256(ROOT/'00_manifest/stage8a0b_requirements.txt'),
    'preregistration_sha256':sha256(ROOT/'00_manifest/stage8a0b_execution_plan.md')})
atomic_json(ROOT/'00_manifest/stage8a0b_registration.json',{'status':'REGISTERED','branch':BRANCH,'base':BASE,
    'populations':['current_valid','full_horizon_ranking_targets'],'splits':['train','val'],
    'lane_radius_m':10,'polygon_boundary_threshold_m':2,'topM':8,'seed':2022,'coordinate_windows':200,
    'gate_scope':'each split full-horizon ranking candidates separately','generic_denominator':'all candidates',
    'recovery_denominator':'lane-fallback candidates','training':False,'test':False,'new_checkpoint':False,
    'historical_stage8a0':'NOT_READY','future_node_schema_proposal':{'type_onehot':6,'lane_semantic':9,
        'nonlane_semantic':'all zero','geometry':'relative geometry and polygon/centerline attributes; dimensions not implemented'},
    'forbidden':'no optimizer/backward/forward new G2 or G3, no predictor/R2 changes'})
index=EntityIndex(write_audit=True)
print('SCHEMA PASS',len(index.dictionary),'entities; invalid components',sum(x['InvalidComponents'] for x in index.schema),flush=True)
