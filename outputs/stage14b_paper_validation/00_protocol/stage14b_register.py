"""Freeze one capacity model and its reused head protocol before fitting."""
from stage14b_common import *

def main():
    assert git('rev-parse','HEAD')==BASE
    assert git('branch','--show-current')=='stage14b/paper-validation-design'
    assert not PROTOCOL.exists() and not REG.exists()
    old=read_json(S14A/'00_protocol/stage14a_frozen_history.json')
    checkpoints=dict(old['checkpoints'])
    for row in read_json(S14A/'04_checkpoints/stage14a_all_frozen.json')['Checkpoints']:
        checkpoints[row['Path']]=row['SHA256']
    hist={p:sha256(PROJECT/p) for p in git('ls-tree','-r','--name-only',BASE).splitlines()}
    preserved={p:sha256(PROJECT/p) for p in git('ls-files','--others','--exclude-standard').splitlines() if not p.startswith(str(ROOT.relative_to(PROJECT))+'/')}
    inputs=dict(old['data_files'])
    inputs.update({str(p.relative_to(PROJECT)):sha256(p) for p in (S14A/'05_evaluation/cache').glob('*') if p.is_file()})
    for p,h in inputs.items():assert sha256(PROJECT/p)==h,p
    for p,h in checkpoints.items():assert sha256(PROJECT/p)==h,p
    freeze=dict(BaseCommit=BASE,historical_files=hist,preserved_untracked=preserved,checkpoints=checkpoints,
                split_normalization=old['split_normalization'],data_files=inputs)
    atomic_json(FROZEN,freeze)
    protocol=dict(read_json(S14A/'00_protocol/stage14a_protocol.json'))
    for k in ('decisions','FigureBackend','FigureProtocolLabel','RequirementsSHA256','same_initial_state_per_fold','control','NoGraphImplementation'):
        protocol.pop(k,None)
    protocol.update(Stage='Stage14B',status='REGISTERED_BEFORE_FITTING',base_commit=BASE,
        branch='stage14b/paper-validation-design',formal_variants=['Matched-NG-C'],FormalNewTrainingRuns=3,
        architecture='G1 own-node encoder64 + residual Linear64x128/ReLU/Linear128x64 + original LayerNorm64/head64x32x1; original-logit residual',
        ActiveParameters=24001,GraphParameters=24066,Difference=65,RelativeDifference=65/24066,
        ExtraOwnAdapterParameters=16576,AdapterInitialization='foldseed+101 once; no architecture search',
        SharedInitialization='G1 node_encoder/norm/head bitwise same per fold; score last layer zero',
        FutureGT='labels only objective/checkpoint evaluation; absent from seven forward inputs',
        CandidateGenerator='same historical Stage5A checkpoint and byte-frozen candidates; no new HiVT training',
        reused_variants=['Stage14A NG-A/NG-C/G-A/G-C, historical fixed foldR2'],
        OuterTest='new matched outputs only after all3 checkpoints frozen; cached historical outputs joined by scene/sample/instance identities',
        primary='three registered Overall Top1FDE contrasts; two historical known results and one new capacity comparison',
        secondary=['minFDE6','Top1ADE','HitRate','Vehicle','Pedestrian','MovingVehicle','head latency and GPU memory'],
        tiny=dict(fold=1,targets=128,sampling_seed=2022,updates=300,variants=['Matched-NG-C'],reuse_original_target_ids=True,
            C_pass='1-final/initial >=0.8 on original normalized C objective',any_failure='STOP; no architecture/hyperparameter changes or formal runs'),
        bootstrap=dict(replicates=2000,seed=2022,unit='paired whole scenes independently resampled210 within each fold',
            descriptive_percentiles=[2.5,97.5],family_size=3,FWER=.05,adjusted_coverage=1-.05/3,
            adjusted_percentiles=[100*.05/6,100*(1-.05/6)],primary_groups=['Overall'],
            co_primary=['G-C-NG-C','G-C-Matched-NG-C','NG-C-NG-A']),
        decisions=dict(CapacityAlternativeNotSufficient='SUPPORTED iff G-C minus Matched-NG-C Overall point<0, Bonferroni family3 CI upper<0 and >=2 of3 folds negative; else NOT_ESTABLISHED; no causal proof'),
        limitations='known historical contrasts reused; internal ranking OOF only; frozen predictor trained all TRAIN700; architecture/depth still differ; approximate rather than exact active capacity match',
        after_stage='commit/push Stage14B only then STOP brainAI review; no Stage15 or HiVT retraining',
        FreezeSHA256=sha256(FROZEN),ModelSourceSHA256=sha256(ROOT/'02_models/stage14b_matched_nograph.py'))
    atomic_json(PROTOCOL,protocol)
    atomic_json(REG,dict(Status='REGISTERED_BEFORE_FITTING',RegisteredUTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        ProtocolSHA256=sha256(PROTOCOL),Base=BASE,ModelSourceSHA256=protocol['ModelSourceSHA256'],
        HistoricalTrainingCodeSHA256=sha256(S11B/'03_training/stage11b_train.py'),
        LossAndStoreSourceSHA256=sha256(S11B/'00_manifest/stage11b_common.py'),
        OriginalTinyTargetsSHA256=sha256(S11B/'01_preflight/stage11b_tiny_targets.json'),
        NewOuterOutputsSeen=False,FormalHiVTTrainingAuthorized=False))
    verify(history=True,data=True)
    print('STAGE14B_REGISTERED',len(hist),'historical files;',len(checkpoints),'old checkpoints; matched head runs3; HiVT runs0',flush=True)
if __name__=='__main__':main()
