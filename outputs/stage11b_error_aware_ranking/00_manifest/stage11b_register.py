"""Register all split, initialization and decision rules before fitting."""
from stage11b_common import *
import subprocess,datetime
def main():
    assert not REG.exists();assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
    oldfreeze=read_json(S11/'01_identity_audit/stage11a_frozen_history.json')
    tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
    frozen={'base_commit':BASE,'historical_files':{p:sha256(PROJECT/p) for p in tracked},
        'checkpoints':oldfreeze['checkpoints'],'preserved_untracked_files':oldfreeze['preserved_untracked_files']}
    atomic_json(ROOT/'00_manifest/stage11b_frozen_history.json',frozen)
    protocol={'status':'REGISTERED_BEFORE_FITTING','client_date':'2026-10-09','timezone':'Asia/Shanghai','base_commit':BASE,
        'branch':'stage11b/controlled-ranking-objectives','formal_variants':['A','B','C'],
        'A':'mean(-sum softmax(-FDE/1m)*log_softmax(logits))','B':'mean(-log p[argmin FDE]); lowest index tie',
        'C':'mean(sum p*(FDE-minFDE)/max(1m,mean(FDE-minFDE))); normalized formulation, differs from raw Stage11A expected risk',
        'architecture':'unchanged Stage8 G1, 24066 parameters; original Stage5A logits + graph delta',
        'node_dim':15,'edge_dim':17,'hidden':64,'message_layers':1,'neighbor_radius_m':50,'max_neighbors':8,'K':6,'Tf':12,'Th':5,
        'official_VAL_or_test_permitted':False,'historical_HeadDev_for_training_selection_or_main_evaluation':False,
        'source_scenes':'HeadTrain630 only','split_seed':2022,
        'outer_split':'sort scene token by SHA256 of UTF8(2022|outer|scene_token), lexical token tie; consecutive210 blocks',
        'inner_split':'remaining420 sort by SHA256 of UTF8(2022|inner|foldN|scene_token); first42 InnerDev, rest378 InnerTrain',
        'ABC_seeds':[2022,2122,2222],'same_initial_state_per_fold':True,'last_score_layer':'zero weight and bias',
        'batch_order':'NumPy default_rng(fold_seed+epoch) permutation; identical cached order hashes A/B/C',
        'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'precision':'FP32','AMP':False,
        'microbatch':128,'accumulation':8,'effective_batch':1024,'max_epochs':50,'patience':5,
        'carry':'continuous shuffled epoch stream; pending final remainder prepended next epoch; no smaller optimizer step; pending saved at stop, every distinct actor must have participated',
        'early_stop':'variant-specific patience5 on identical selection rule; same maximum budget, no equalization based on test performance',
        'ABC_selection':'minimum Sdev=.5*VehicleInnerDevFDE/FoldR2VehicleFDE+.5*PedestrianInnerDevFDE/FoldR2PedestrianFDE; strict improvement; epoch1 first eligible',
        'R2_protocol':{'architecture':[19,32,1],'parameters':673,'loss':'original Stage6 ranking_loss softmax(-FDE/1m)',
            'seed':2022,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'head_batch':1024,'evaluation_batch':4096,
            'max_epochs':50,'patience':5,'order':'torch.randperm with CPU Generator seed2022+epoch',
            'remainder':'original last partial batch optimizer step, no carry','selection':'InnerDev Overall Top1FDE strict improvement',
            'changes_only':'data partition and normalization; historical R2 checkpoint never loaded'},
        'normalization':'each fold fitted only on InnerTrain378, population std +1e-6; original G1 valid mask and R2 missing-neighbor conventions',
        'Bicycle':'neighbors and training targets retained; final ABC scores routed to corresponding frozen FoldR2',
        'tiny':{'fold':1,'targets':128,'sampling_seed':2022,'updates':300,'optimizer':'same AdamW FP32',
            'A_pass':'1-(finalCE-Hq)/(initialCE-Hq)>=.9','B_C_pass':'1-final/initial>=.8','any_failure':'STOP without formal CV or tuning'},
        'OuterTest':'only after all four fold checkpoints frozen; exactly one fold prediction per630scene; never use InnerDev in OOF',
        'bootstrap':{'replicates':2000,'seed':2022,'unit':'whole paired scene within each210-scene fold, then actor-window pooled across folds',
            'descriptive_percentiles':[2.5,97.5],'two_co_primary_Bonferroni_percentiles':[1.25,98.75]},
        'primary':'C vs A Vehicle and Pedestrian Top1FDE; Overall direction safeguard',
        'secondary':['B vs A','C vs B','C vs R2','A vs R2'],'motion_groups':'exploratory',
        'decisions':{'ErrorAwareRanking':'exact user STRONG_SUPPORTED/SUPPORTED/PARTIAL/NOT_SUPPORTED; joint point directions, adjusted primary intervals, R2 type guard and Bicycle bitwise identity',
            'HardCE':'SUPPORTED if B improves A and FoldR2 in Vehicle/Pedestrian/Overall and primary-type descriptive B-A CI is not wholly harmful; secondary evidence only',
            'VehicleImproved_PedestrianImproved_OverallImproved':'C improves both A and FoldR2 point estimates for stated group',
            'CostlySwitchReduced':'C-vs-R2 Vehicle gross positive harm and sum of largest10% positive harms both below A-vs-R2',
            'PedestrianWrongSwitchReduced':'C-vs-R2 Pedestrian worsened count below A-vs-R2',
            'Stage11B_GO_Ready':'GO/YES only if C SUPPORTED or STRONG_SUPPORTED and all engineering audits pass; otherwise STOP/NO; good secondary B still waits for review'},
        'limitations':'hypothesis informed by Stage11A; historical HeadDev/VAL used earlier; frozen Stage5A may have trained on CV scenes; internal ranking OOF, not independent end-to-end test',
        'after_stage':'STOP; no official VAL/test, Stage11C, loss search or architecture changes'}
    atomic_json(PROTOCOL,protocol)
    config6=read_json(S6/'00_manifest/stage6a_config.json');assert config6['head_architecture']==[19,32,1] and config6['head_batch']==1024 and config6['evaluation_batch']==4096
    for key,value in [('lr',.001),('weight_decay',.0001),('maximum_epochs',50),('patience',5),('seed',2022)]:assert config6[key]==value
    atomic_json(REG,{'status':'REGISTERED_BEFORE_FITTING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'protocol_sha256':sha256(PROTOCOL),'requirements_sha256':sha256(ROOT/'00_manifest/stage11b_requirements.txt'),
        'R2_original_source_sha256':sha256(S6/'03_training/stage6a_train.py'),
        'R2_original_config_sha256':sha256(S6/'00_manifest/stage6a_config.json'),
        'G1_architecture_source_sha256':sha256(S8/'00c_sparse_type_aware_graph_spec/03_model_audit/stage8a0c_model.py')})
    verify(history=True);print('STAGE11B_REGISTERED',flush=True)
if __name__=='__main__':main()
