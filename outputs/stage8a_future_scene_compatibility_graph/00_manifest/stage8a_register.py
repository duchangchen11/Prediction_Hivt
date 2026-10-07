"""Record Stage8A-0 authorization and frozen inputs; no optimization allowed."""
from stage8a_common import *
import datetime

def main():
    assert git('rev-parse','HEAD')==BASE and git('branch','--show-current')==BRANCH
    old=read_json(ROOT.parent/'stage7d_actor_semantic_attention/00_manifest/stage7d_frozen_previous.json')
    files=dict(old['files'])
    files.update({n:sha256(PROJECT/n) for n in git('ls-files').splitlines() if not n.startswith(str(ROOT.relative_to(PROJECT))+'/')})
    for p in (PREDICTOR,R2,CACHE_MANIFEST,CENTERLINES,SEMANTICS,SPLIT,STAGE6/'02_features/stage6a_normalization.json'):
        files[str(p.relative_to(PROJECT))]=sha256(p)
    for n,h in old['files'].items():assert sha256(PROJECT/n)==h,n
    frozen={'base_commit':BASE,'files':files,'scene_shards':old['scene_shards'],'unrelated_untracked':old['unrelated_untracked'],
            'cache_batches':read_json(CACHE_MANIFEST)['batches'],'cache_source_is_read_only':True}
    atomic_json(ROOT/'00_manifest/stage8a_frozen_inputs.json',frozen)
    split=read_json(SPLIT)
    assert len(split['HeadTrain'])==630 and len(split['HeadDev'])==70 and len(split['official_VAL'])==150
    assert not (set(split['HeadTrain'])&set(split['HeadDev'])) and not ((set(split['HeadTrain'])|set(split['HeadDev']))&set(split['official_VAL']))
    atomic_json(ROOT/'00_manifest/stage8a_stage0_registration.json',{
        'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'Stage8A-0',
        'base_commit':BASE,'branch':BRANCH,'training_authorized':False,'Stage8A_1_authorized':False,
        'G1_G2_G3_training_executed':False,'tiny_training_authorized':False,'test_used':False,
        'seed':2022,'K':6,'Tf':12,'neighbors':8,'neighbor_radius_m':50.,'map_topM':8,'map_radius_m':10.,
        'map_no_token_policy':'nearest one token across the same full map region; no radius change',
        'coverage_primary_population':'all current-valid actor modes, including context and partial targets',
        'coverage_secondary_population':'full-horizon ranking targets; future mask used only outside graph to label audit populations',
        'map_gate':'NOT_READY if Vehicle fallback>5% in TRAIN or VAL primary coverage, or unreliable coordinate mapping',
        'feature_dimensions':{'node':15,'interaction_edge':17,'map_node':13,'map_edge':17},
        'message_hidden':64,'message_layers':1,'head_architecture':[64,32,1],
        'future_training_selection':'HeadDev ranking soft cross entropy, never official VAL; deferred for separate authorization',
        'head_split_sha256':sha256(SPLIT),'predictor_sha256':PREDICTOR_SHA,'R2_sha256':R2_SHA,
        'Stage7A_decision':'NOT_SUPPORTED','Stage7D_decision':'NOT_SUPPORTED','historical_decisions_changed':False})
    verify_frozen(shards=True)
    print('STAGE8A_REGISTER_PASS',len(files),len(old['scene_shards']),flush=True)

if __name__=='__main__':main()
