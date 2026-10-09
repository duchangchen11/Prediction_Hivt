"""Freeze the complete 2x2 comparison before any new fitting or OOF inspection."""
from stage14a_common import *

def main():
    assert git('rev-parse','HEAD')==BASE and git('branch','--show-current')=='stage14a/paper-graph-ablation'
    assert not PROTOCOL.exists()
    previous=read_json(S13/'00_manifest/stage13a_frozen_history.json')
    checkpoints=dict(previous['checkpoints'])
    for row in read_json(S11B/'04_checkpoints/stage11b_all_frozen.json')['Checkpoints']:
        assert sha256(PROJECT/row['Path'])==row['SHA256'];checkpoints[row['Path']]=row['SHA256']
    hist={p:sha256(PROJECT/p) for p in git('ls-tree','-r','--name-only',BASE).splitlines()}
    preserved={p:sha256(PROJECT/p) for p in git('ls-files','--others','--exclude-standard').splitlines() if not p.startswith(str(ROOT.relative_to(PROJECT))+'/')}
    normalization={}
    for k in (1,2,3):
        for name in ('split','normalization'):
            p=S11B/f'02_splits/stage11b_fold{k}_{name}.json';normalization[str(p.relative_to(PROJECT))]=sha256(p)
    inputs=[S11A/'01_identity_audit/cache/candidates.npy',S11B/'01_preflight/cache/identities.csv',
            S11B/'01_preflight/cache/r2_raw.npy',S11B/'01_preflight/cache/r2_flags.npy',
            *[RAW/name for name in ['arg0.npy','arg1.npy','arg2.npy','arg6.npy','fde.npy','ade.npy']],
            OOF/'stage11b_oof_predictions_actor_records.csv',OOF/'stage11b_oof_predictions_logits.npy',
            OOF/'stage11b_oof_predictions_probabilities.npy',OOF/'stage11b_oof_metrics.npy']
    f=frame();assert len(f)==260151 and f.scene_token.nunique()==630
    foldsets=[];splitrows=[]
    for k in (1,2,3):
        s=split(k);assert [len(s[p]) for p in ('InnerTrain','InnerDev','OuterTest')]==[378,42,210]
        sets=[set(s[p]) for p in ('InnerTrain','InnerDev','OuterTest')]
        assert all(not sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2)))
        assert set.union(*sets)==set(f.scene_token);foldsets.append(sets[2])
        for part in ('InnerTrain','InnerDev','OuterTest'):splitrows.append(dict(Fold=k,Part=part,Scenes=len(s[part]),Actors=len(indices(k,part)),IndicesSHA256=array_sha(indices(k,part))))
    assert len(set.union(*foldsets))==630 and all(not foldsets[a]&foldsets[b] for a,b in ((0,1),(0,2),(1,2)))
    freeze=dict(BaseCommit=BASE,historical_files=hist,preserved_untracked=preserved,checkpoints=checkpoints,
                split_normalization=normalization,data_files={str(p.relative_to(PROJECT)):sha256(p) for p in inputs})
    atomic_json(FROZEN,freeze)
    old=read_json(S11B/'00_manifest/stage11b_protocol.json')
    # The original optimizer, stream, loss, normalization and stopping fields
    # remain verbatim. Stage14A changes only the registered architecture contrast.
    protocol=dict(old)
    protocol.update(Stage='Stage14A',status='REGISTERED_BEFORE_FITTING',client_date='2026-10-10',timezone='Asia/Shanghai',
        base_commit=BASE,branch='stage14a/paper-graph-ablation',formal_variants=['NG-A','NG-C'],
        reused_variants=['G-A = Stage11B A','G-C = Stage11B C'],FormalNewTrainingRuns=6,
        architecture='original G1 node_encoder + LayerNorm + scoring head + original-logit residual; interaction_message exactly0; no interaction network parameters',
        control='identical shared module initialization, own candidates/node features, folds, normalization, sample ordering, optimizer, learning rate, loss and checkpoint selection; full unchanged graph-input cache remains available',
        CandidateGenerator='unchanged Stage5A frozen; no new predictor training',
        NoGraphImplementation='real structural zero-message ablation, not shuffled/deleted/corrupted graph inputs',
        OuterTest='unified evaluation only after all6 new checkpoints frozen; reuse historical G-A/G-C outputs through identity keys',
        tiny=dict(old['tiny'],variants=['NG-A','NG-C'],reuse_original_target_ids=True,A_pass='original entropy-floor adjusted excess CE reduction>=0.9',C_pass='original normalized objective reduction>=0.8',any_failure='STOP; no formal training or tuning'),
        bootstrap=dict(replicates=2000,seed=2022,unit='paired whole scenes, independent210-scene resampling within each of3folds',
            descriptive_percentiles=[2.5,97.5],family_size=4,FWER=.05,adjusted_coverage=.9875,
            adjusted_percentiles=[.625,99.375],primary_groups=['Overall'],
            co_primary=['NG-C-NG-A','G-A-NG-A','G-C-NG-C','G-C-G-A'],
            interaction='(G-C-G-A)-(NG-C-NG-A), descriptive95%CI only'),
        primary='four preregistered controlled OverallTop1FDE comparisons; type/motion and interaction contrasts exploratory',
        decisions=dict(GraphIncrementSupported='SUPPORTED iff G-C minus NG-C Overall point<0, family4 adjusted CI upper<0 and>=2/3fold deltas<0; otherwise NOT_SUPPORTED',
            LossIncrementSupported='SUPPORTED iff both NG-C minus NG-A and G-C minus G-A each meet point<0, adjusted CI upper<0,>=2/3negativefolds; otherwise NOT_SUPPORTED',
            Limits='type and MovingVehicle gains/harms disclosed separately; no causal/independent end-to-end validation claim; no outer-test tuning'),
        high_cost='original Stage11B definition: sum and count of largest10% positive per-actor switch FDE harms; ceil(0.1*n_harms)',
        after_stage='push ownbranch without merge then STOP for brain-AI review; no Stage14B, new models/semantic modules, HiVT retraining or officialVAL/test evaluation',
        FigureBackend='python',FigureProtocolLabel='nuScenes HeadTrain630 internal ranking OOF,3fold; not official independent test',
        FreezeSHA256=sha256(FROZEN),RequirementsSHA256=sha256(ROOT/'00_protocol/stage14a_requirements.txt'))
    atomic_json(PROTOCOL,protocol)
    atomic_json(REG,dict(Status='REGISTERED_BEFORE_FITTING',ProtocolSHA256=sha256(PROTOCOL),Base=BASE,
        HistoricalTrainingCodeSHA256=sha256(S11B/'03_training/stage11b_train.py'),
        LossAndStoreSourceSHA256=sha256(S11B/'00_manifest/stage11b_common.py'),
        OriginalTinyTargetsSHA256=sha256(S11B/'01_preflight/stage11b_tiny_targets.json')))
    dump('00_protocol/stage14a_fold_split_audit.csv',splitrows)
    verify(history=True,data=True)
    print('STAGE14A_REGISTERED',len(hist),'historical files;',len(checkpoints),'checkpoints; 6 new runs only',flush=True)

if __name__=='__main__':main()
