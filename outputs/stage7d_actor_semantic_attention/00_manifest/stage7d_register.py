"""Freeze protocol, historical inputs and decision rules before optimization."""
from pathlib import Path
import sys,subprocess,datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7d_common import *
from stage7d_dataset import SEMANTIC_FIELDS

def main():
    assert not BEST.exists() and not CURVE.exists()
    old=read_json(ROOT.parent/'stage7c_actor_semantic_relevance_audit/00_manifest/stage7c_frozen_inputs.json')
    files=dict(old['files'])
    for dirname in ('stage7a_semantic_map_hivt','stage7c_actor_semantic_relevance_audit'):
        for path in (ROOT.parent/dirname).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and 'git_upload_' not in path.name:
                files[str(path.relative_to(PROJECT))]=sha256(path)
    unrelated={str(path.relative_to(PROJECT)):sha256(path) for path in (ROOT.parent/'stage2c_trainval_vehicle_baseline/04_evaluation').glob('stage2c_qualitative_main_case_new*')}
    atomic_json(FREEZE,{'files':files,'scene_shards':old['scene_shards'],'unrelated_untracked':unrelated,
                       'base_commit':'7df27d2091425c6836a66d6987714b1d9569efb5'})
    assert sha256(BASE_BEST)=='461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547'
    sidecar=ROOT.parent/'stage7a_semantic_map_hivt/01_data_audit/stage7a_formal_actor_val_groups.csv'
    attention=ROOT.parent/'stage7c_actor_semantic_relevance_audit/02_relevance_analysis/stage7c_actor_attention.csv'
    sources=[ROOT/'00_manifest'/name for name in ('stage7d_dataset.py','stage7d_model.py','stage7d_local_encoder.py',
             'stage7d_common.py','stage7d_evaluation_core.py','stage7d_bias_statistics.py')]+[ROOT/'03_training/stage7d_train.py']
    atomic_json(PREREG,{'registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'training_authorized':True,'Stage7E_authorized':False,'Stage7B_authorized':False,'R2_authorized':False,
      'from_scratch':True,'formal_runs':1,'seed':2022,'config_sha256':sha256(CONFIG),
      'training_source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in sources},
      'semantic_fields':SEMANTIC_FIELDS,'metadata_source':'Stage7A audited cache, no regeneration',
      'baseline_checkpoint_sha256':sha256(BASE_BEST),'baseline_checkpoint_used_for_initialization':False,
      'group_sidecar_sha256':sha256(sidecar),'baseline_attention_sidecar_sha256':sha256(attention),
      'offline_only':['GT trajectory','turn direction','semantic subgroup masks','errors'],
      'warmup_updates':5000,'NLL_updates_max':16000,'global_hard_max':21000,'validation_interval':500,'patience':5,
      'primary_selection':'original-NLL overall full-horizon VAL minFDE6 strict improvement',
      'bootstrap':{'unit':'whole official VAL scene','scenes':150,'replicates':1000,'seed':2022,
          'pooling':'paired actor-window sums/counts','interval':'percentile2.5/97.5','delta':'Stage7D minus Stage3B'},
      'scientific_rules':{
          'marked_reliable_harm_SUPPORTED':'Vehicle or Pedestrian relative minFDE increase>5% AND paired CI lower>0; same Stage7A guard',
          'SUPPORTED':'Overall delta<0 and upper<0; no marked reliable Vehicle/Pedestrian harm',
          'TARGETED_SUPPORTED':'Overall increase<=0.5%, CI includes0; >=2 of Moving,Vehicle>5m,NearTurn,Turning upper<0; neither Vehicle nor Pedestrian lower>0',
          'AttentionMechanism_SUPPORTED':'Vehicle OR Moving relevant-mass2m lower>0 AND Turning correct-mass OR connector-match lower>0',
          'AttentionMechanism_PARTIAL':'exactly one of those two reliable-improvement legs; otherwise NOT_SUPPORTED',
          'PaperUsableSemantic':'YES iff SUPPORTED/TARGETED_SUPPORTED; SemanticRoute and ReadySemanticReranking follow this only',
          'secondary_multiplicity':'unadjusted exploratory overlapping subgroup CIs, no independent replication claim'},
      'primary_attention_population':'Overall Vehicle; Moving is prespecified secondary',
      'nearest_rank_definition':'Stage7C all-current-graph lanes, absent edges weight0, average ties; lowest-index GT argmin',
      'bias_audit':'every500 updates all retained VAL edges and 8 heads; exact absolute mean/median/p95/max by actor type',
      'tiny':'fixed6 TRAIN graphs/200 fixed-scale updates; fresh constructor for formal train',
      'test_used':False})
    verify_previous(shards=True)
    print('REGISTER_PASS',len(files),'historical files',len(old['scene_shards']),'scene shards',flush=True)

if __name__=='__main__':main()
