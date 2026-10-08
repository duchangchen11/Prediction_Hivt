from stage8a0c_common import *
assert git('rev-parse','HEAD')==BASE and git('branch','--show-current')==BRANCH
prefix=str(ROOT.relative_to(PROJECT))+'/'
files={p:sha256(PROJECT/p) for p in git('ls-files').splitlines() if not p.startswith(prefix)}
extra=[CENTERLINES,SEMANTICS,PREDICTOR,R2,STAGE6/'01_cache/stage6a_cache_manifest.json',
    STAGE8/'02_graph_cache/stage8a_graph_cache_manifest.json',STAGE0B/'03_entity_retrieval/stage8a0b_entity_dictionary.json',*sorted(MAP_JSON.glob('*.json'))]
files.update({str(p.relative_to(PROJECT)):sha256(p) for p in extra})
unrelated={p:sha256(PROJECT/p) for p in git('ls-files','--others','--exclude-standard').splitlines() if not p.startswith(prefix)}
atomic_json(ROOT/'00_manifest/stage8a0c_frozen_inputs.json',{'source_commit':BASE,'files':files,'unrelated_untracked':unrelated,
    'preregistration_sha256':sha256(ROOT/'00_manifest/stage8a0c_execution_plan.md'),'requirements_sha256':sha256(ROOT/'00_manifest/stage8a0c_requirements.txt')})
atomic_json(ROOT/'00_manifest/stage8a0c_registration.json',{'source_commit':BASE,'branch':BRANCH,'entity_types':TYPES,
    'quotas':QUOTAS.tolist(),'lane_radius_m':10,'polygon_threshold_m':2,'gate_scope':'each TRAIN/VAL full-horizon population separately',
    'coverage_gates':{'MovingRoute':.95,'VehicleNonempty':.80,'ParkedNonempty':.70,'PedestrianNonempty':.85,'PedestrianSpecific':.70,'BicycleNonempty':.80},
    'hidden_dim':64,'ranking_temperature_m':1,'memory_limit_bytes':2**30,'parameter_limit_exclusive':150000,
    'no_fallback':True,'no_NULL_token':True,'zero_map_samples_retained':True,'training':False,'test_used':False,
    'historical_stage8a0':'NOT_READY','historical_stage8a0b':'NOT_READY'})
print('STAGE8A0C REGISTERED',flush=True)
