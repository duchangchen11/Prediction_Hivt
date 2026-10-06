"""Register split, normalization/missing-data conventions and frozen references."""
from pathlib import Path
import sys,csv
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage6a_common import *
import numpy as np

def main():
    assert git('branch','--show-current')=='stage6a/future-interaction-reliability'
    assert git('rev-parse','HEAD')==BASE_COMMIT
    assert not CONFIG.exists() and not SPLIT.exists(),'Registration already exists'
    ds=SceneDataset('train');val=SceneDataset('val')
    scenes=sorted({r['scene_token'] for r in ds.all_rows});vscenes=sorted({r['scene_token'] for r in val.all_rows})
    assert len(scenes)==700 and len(vscenes)==150 and not set(scenes)&set(vscenes)
    order=np.random.default_rng(2022).permutation(scenes).tolist()
    dev=sorted(order[:70]);train=sorted(order[70:]);assert len(train)==630 and not set(train)&set(dev)
    atomic_json(SPLIT,{'status':'PASS','seed':2022,'method':'sorted official TRAIN700 scene tokens; NumPy default_rng(2022) permutation; first70 HeadDev',
        'HeadTrain':train,'HeadDev':dev,'official_VAL':vscenes,'HeadTrain_scenes':630,'HeadDev_scenes':70,
        'head_split_scene_overlap':0,'official_VAL_used_for_head_selection':False,
        'head_dev_is_held_out_for_head_optimization_only':'frozen Stage5A predictor was already trained on all official TRAIN700 scenes'})
    c={'stage':'Stage6A','seed':2022,'predictor_batch':16,'head_batch':1024,'evaluation_batch':4096,
       'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'maximum_epochs':50,'patience':5,
       'selection':'HeadDev full-horizon Top1FDE strict improvement after every epoch','head_architecture':[19,32,1],
       'neighbors_maximum':8,'neighbor_radius_m':50.,'conflict_sigma_m':2.,'target_temperature_m':1.,
       'normalization_epsilon':1e-6,'standardization':'TRAIN630 full-horizon actor-modes only; population std; all continuous columns3:19 including base probability',
       'distance_statistics':'columns12:15 exclude actors with zero neighbors; no-neighbor distances set1 after normalization',
       'missing_conflicts':'no-neighbor conflict columns15:19 set0 after normalization; absent V/P neighbor column17/18 set0 after normalization',
       'R1_interaction':'columns12:19 exactly zero after common normalization','type_one_hot':'columns0:3 unchanged',
       'predictor_requires_grad':False,'predictor_mode':'eval','ranking_labels':'only full-horizon targets; q=softmax(-FDE/1m)',
       'predictor_deterministic_algorithms':True,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
       'formal_heads':2,'additional_seed':False,'official_VAL_evaluations_for_ranking':1,'test_used':False,
       'scientific_interpretation':{'severe_major_class_harm':'Vehicle or Pedestrian Top1FDE increases by >=10% relative to R0 AND paired CI lower>0; descriptive operational guard set before fitting',
          'ReliabilityHead_SUPPORTED':'at least one fixed head improves overall Top1FDE with CI upper<0, reduces OracleGap, increases HitRate, no severe class harm, geometry PASS',
          'ReliabilityHead_PROMISING':'point improvement, smaller gap and higher HitRate with overall CI crossing0; no severe class harm and geometry PASS',
          'FutureInteraction_SUPPORTED':'R2-R1 overall Top1FDE CI upper<0 and at least one frozen interaction group point improves; no severe class harm',
          'FutureInteraction_NOT_SUPPORTED':'R2-R1 overall Top1FDE CI lower>0, or R2 causes severe main-class harm; otherwise WEAK',
          'final_variant':'lowest final overall Top1FDE among R0/R1/R2 subject to geometry and severe-harm guard; descriptive selection among two prespecified heads, no VAL checkpoint selection',
          'PaperUsable':'recommended R1/R2 lowers overall Top1FDE and OracleGap and raises HitRate without severe class harm; report supported versus promising evidence separately'}}
    atomic_json(CONFIG,c)
    previous=read_json(STAGE5/'00_manifest/stage5a_frozen_references.json')
    files=dict(previous['files'])
    for name in git('ls-files').splitlines():
        if not name.startswith(str(ROOT.relative_to(PROJECT))+'/'):files[name]=sha256(PROJECT/name)
    for p in (PREDICTOR,STAGE5_ACTORS,BASE_ACTORS,GT_LEDGER,MEMBERSHIP,MEMBERSHIP_AUDIT):files[str(p.relative_to(PROJECT))]=sha256(p)
    assert sha256(PREDICTOR)==PREDICTOR_SHA
    atomic_json(FREEZE,{'status':'PASS','base_commit':BASE_COMMIT,'files':files,'scene_shards':previous['scene_shards'],
        'Stage5A_checkpoint_sha256':PREDICTOR_SHA,'Stage3B_science':'SUPPORTED','Stage5A_science':'NOT SUPPORTED',
        'candidate_geometry_backbone':'Stage5A; prior scientific conclusion unchanged','data_copied':False})
    verify_frozen(shards=True)
    audit=read_json(MEMBERSHIP_AUDIT);assert audit['membership_sha256']==sha256(MEMBERSHIP)
    assert audit['Stage3B_actor_errors_sha256']==sha256(BASE_ACTORS)
    atomic_json(ROOT/'00_manifest/stage6a_registration.json',{'status':'REGISTERED_BEFORE_CACHE_AND_HEAD_TRAINING',
        'base_commit':BASE_COMMIT,'config_sha256':sha256(CONFIG),'split_sha256':sha256(SPLIT),
        'frozen_predictor_sha256':PREDICTOR_SHA,'frozen_membership_sha256':sha256(MEMBERSHIP),
        'TRAIN_scenes':700,'VAL_scenes':150,'TRAIN_windows':len(ds),'VAL_windows':len(val),
        'prior_files_frozen':len(files),'scene_shards_frozen':len(previous['scene_shards']),
        'official_VAL_final_ranking_only':True,'models':['R0 frozen original ranking','R1 without interaction','R2 with interaction']})
    print('STAGE6A_REGISTERED',len(ds),len(val),flush=True)

if __name__=='__main__':main()
