"""Register every choice before normalized inputs, tiny and formal training."""
from stage12b_common import *
import subprocess
def main():
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE and not REG.exists()
    protocol={'Stage':'Stage12B','BaseCommit':BASE,'Branch':'stage12b/motion-conditioned-semantic-reranking',
        'Population':{'Scenes':630,'Actors':260151,'Vehicle':191026,'Pedestrian':66145,'Bicycle':2980,'K':6,'Th':5,'Tf':12},
        'Split':'exact frozen Stage11B 378/42/210 threefold scenes; no resplit',
        'FrozenCForward':'corresponding fold C and R2 on each separate partition, FP32 CUDA batch128, source normalizer unchanged; bitwise OuterTest logits/probability replay gate',
        'Temperatures':TEMPERATURES,'RefitTemperature':False,'Variants':VARIANTS,'Features':FEATURES,
        'Validity':'candidate geometry valid; recent_speed/history_displacement valid when observed history_count>=2; interaction minimum valid when original neighbor mask nonempty',
        'SemanticSlots':['walkway_inside_fraction','walkway_valid'],'GSlots':'both exact zero','Missing':'continuous standardized sentinel0 plus mask; unmatched walkway0+valid0; no actor dropped',
        'Normalization':'population mean/std per feature over valid Pedestrian InnerTrain378 candidate rows only; denominator max(std,1e-6); FP64 fit then FP32 network inputs, shared G/S/P',
        'Architecture':{'Inputs':18,'Hidden':32,'LinearLayers':2,'Activation':'ReLU','SharedAcrossSixModes':True,'Params':641,'Delta':'2*tanh(raw)'},
        'Initialization':'output weight/bias zero; same complete state_dict in G/S/P per fold; neutral Top1 exactly C0',
        'Seeds':{1:2022,2:2122,3:2222},
        'Permutation':{'Seed':2022,'Generator':'numpy SeedSequence(2022).spawn(9), key3*(fold-1)+partitionCode; codes InnerTrain0,InnerDev1,OuterTest2',
            'WithinActorSixModes':True,'PermuteTogether':['walkway_inside_fraction','walkway_valid'],'TrainingAndDevShuffled':True,'ErrorsOrScoresUsed':False},
        'Loss':{'Risk':'sum p*(FDE-minFDE)/max(1,mean(FDE-minFDE))','Anchor':'KL(softmax(Craw/T) || student)','AnchorWeight':.1,'DeltaPenaltyWeight':.001,'DeltaPenalty':'mean6 squared delta'},
        'Training':{'Optimizer':'AdamW','LR':.001,'WeightDecay':.0001,'FP32':True,'AMP':False,'Microbatch':128,'EffectiveBatch':512,'Accumulation':4,'MaxEpoch':50,'Patience':5,
            'Order':'same np.default_rng(foldseed+epoch) permutation of all pedestrian InnerTrain IDs, prepend previous pending occurrences; optimize complete512 groups only, carry <512 to nextepoch, final pending saved; all distinct targets trained',
            'Selection':'strict minimum Pedestrian InnerDev42 Top1FDE, exact ties earliest epoch; step0 audited but best checkpoint selected among trained epochs1..50',
            'Sequence':['Fold1 G','Fold1 S','Fold1 P','Fold2 G','Fold2 S','Fold2 P','Fold3 G','Fold3 S','Fold3 P']},
        'Tiny':{'Fold':1,'Targets':128,'Selection':'first128 InnerTrain pedestrian actor_ids in lexicographic order, no error/coverage filtering','OptimizerUpdates':300,'RequiredRelativeLossDecrease':.05,
            'EffectiveBatch':'four repetitions of same fixed128 targets, gradient accumulation4','DiscardTinyWeights':True},
        'OOF':'no new residual OuterTest evaluation until all9 selected checkpoints frozen; frozen C baseline replay before training permitted solely identity gate',
        'Routing':{'Vehicle':'direct copy corresponding frozen C Raw z/p; no softmax recomputation','Pedestrian':'Craw/T + variant delta','Bicycle':'direct copy corresponding frozen FoldR2 z/p'},
        'CoverageGroups':'fixed actor-level walkway_valid1 if ANY of original6 candidate walkway masks valid;0 if allmissing, paired population identical across variants',
        'AdditionalCoverage':'C0 Top1 walkway mask and model selected-mode coverage reported separately, no filtering',
        'Bootstrap':{'Replicates':2000,'Seed':2022,'Unit':'whole scene resampled within each frozen210 outerfold, paired actors/modes; actor-weighted pooled FDE estimand',
            'CI95Percentiles':[2.5,97.5],'PrimaryFamily':['S-G Pedestrian Top1FDE','S-P Pedestrian Top1FDE'],'BonferroniAdjustedPercentiles':[1.25,98.75],
            'Evidence':'Development-stage descriptive intervals, not independent confirmation'},
        'DecisionPrecedence':'integrity failure or S not strictly better than G/P or C0 => NOT_SUPPORTED; if S improvesC0 but significant5-8m harm => PARTIAL; else STRONG if adjusted S-G and S-P upper<0 and Overall improved; SUPPORTED if at least2 improving S-G folds and Overall notworse; otherwise PARTIAL',
        'Clause40vs41':'conservative precedence of explicit failure clause41 when point S fails to beat a control; PARTIAL retains motion-subgroup or crossfold inconsistency cases; no posthoc threshold changes',
        'Pedestrian5_8mDegraded':'YES iff pooled S-C0 delta>0; significant subgroup harm iff descriptive95CI lower>0, overrides support toPARTIAL when S overall Pedestrian improves',
        'HighCostSwitch':'diagnostic harm>5m fixed; no selection or hyperparameter adjustment',
        'Probability':{'NLL':'hard oracle minFDE mode, lowest-index ties','Brier':'sum6 squared error vs onehot oracle','ECEBins':10,'ECE':'confidence vs hard oracle hit, equal-width bins','ExpectedRegret':'sum p*(FDE-minFDE)',
            'NoNewCalibration':True,'C0':'original C Raw probability; teacher-scaled C distribution additionally identified'},
        'Cases':'10 Pedestrian examples; 5 best gains and5 largest harms C0->S, actor_id ties, distinct scene preference, no map coverage filtering, all6 candidates and GT+walkway',
        'EvidenceBoundary':'Stage12A already inspected these630 scenes; same development OOF; frozen Stage5 saw related historical training scenes; not fully independent end-to-end test',
        'Forbidden':['HeadDev70','official VAL150','test','HiVT/C/R2/temperature retraining','new fields','new architectures','hyperparameter or loss/seed searches','Stage12C'],
        'AfterCompletion':'STOP and wait for 大脑AI review'}
    atomic_json(PROTOCOL,protocol)
    old=read_json(S12A/'00_manifest/stage12a_frozen_manifest.json')
    cps=old['checkpoints'];extra={}
    for fold in (1,2,3):
        for path in [S11B/f'02_splits/stage11b_fold{fold}_split.json',S11B/f'02_splits/stage11b_fold{fold}_normalization.json',S11C/f'02_calibration/fold{fold}/stage11c_temperature.json']:
            extra[str(path.relative_to(PROJECT))]=sha256(path)
        temp=read_json(S11C/f'02_calibration/fold{fold}/stage11c_temperature.json');assert temp['Temperature']==TEMPERATURES[fold]
        assert temp['CheckpointSHA256']==cps[str((S11B/f'04_checkpoints/fold{fold}/C_best.pt').relative_to(PROJECT))]
    historical={p:sha256(PROJECT/p) for p in subprocess.check_output(['git','ls-tree','-r','--name-only',BASE],text=True).splitlines()}
    data=dict(old['data_files'])
    for name,h in read_json(RAW/'manifest.json')['files'].items():
        if name in ['arg0.npy','arg1.npy','arg2.npy','arg6.npy']:assert sha256(RAW/name)==h;data[str((RAW/name).relative_to(PROJECT))]=h
    for name,h in read_json(S11B/'01_preflight/cache/prepared.json')['cache_files'].items():
        path=S11B/'01_preflight/cache'/name;assert sha256(path)==h;data[str(path.relative_to(PROJECT))]=h
    folder=S12A/'03_feature_statistics/cache'
    for name,h in read_json(folder/'stage12a_combined_manifest.json')['files'].items():assert sha256(folder/name)==h;data[str((folder/name).relative_to(PROJECT))]=h
    manifest=read_json(S12A/'02_semantic_cache/stage12a_semantic_cache_manifest.json');assert manifest['Status']=='FROZEN_ALL_630'
    for rec in manifest['Records']:
        path=S12A/rec['Path'];assert sha256(path)==rec['SHA256'];data[str(path.relative_to(PROJECT))]=rec['SHA256']
    atomic_json(FROZEN,{'Status':'PASS','BaseCommit':BASE,'checkpoints':cps,'split_normalization_temperature':extra,
        'historical_files':historical,'preserved_untracked_files':old['preserved_untracked_files'],'data_files':data,
        'SemanticManifestSHA256':sha256(S12A/'02_semantic_cache/stage12a_semantic_cache_manifest.json')})
    atomic_json(REG,{'Status':'REGISTERED_BEFORE_PREFLIGHT_TINY_FORMAL_AND_RESIDUAL_OOF','ProtocolSHA256':sha256(PROTOCOL),
        'RequirementsSHA256':sha256(ROOT/'00_manifest/stage12b_requirements.txt'),'FrozenManifestSHA256':sha256(FROZEN)})
    verify(history=True);print('STAGE12B_REGISTERED',len(historical),'historical files',len(cps),'checkpoints',flush=True)
if __name__=='__main__':main()
