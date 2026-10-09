"""Assemble execution evidence, registered decisions and reviewable results.

This produces an experiment report, not manuscript text. Brain-AI retains paper
writing and scientific interpretation. No fitting or checkpoint selection.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_protocol'))
from stage14a_common import *

def markdown_table(rows,fields):
    def value(x):
        if isinstance(x,(float,np.floating)):return f'{x:.6f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(fields)+' |','| '+' | '.join(['---']*len(fields))+' |']+
        ['| '+' | '.join(value(r[k]) for k in fields)+' |' for r in rows])

def main():
    verify(history=True,data=True)
    expected={'01_preflight/stage14a_model_integrity.json':'PASS',
        '01_preflight/stage14a_input_loss_integrity.json':'PASS','01_preflight/stage14a_tiny_audit.json':'PASS',
        '04_checkpoints/stage14a_all_frozen.json':'FROZEN_ALL_COMPLETE',
        '05_evaluation/stage14a_identity_audit.json':'PASS','06_bootstrap/stage14a_bootstrap_audit.json':'PASS',
        '07_diagnostics/stage14a_mode_switch_audit.json':'PASS','07_diagnostics/stage14a_efficiency_audit.json':'PASS',
        '09_reports/stage14a_predictor_data_provenance.json':'PASS'}
    audits={p:read_json(ROOT/p) for p in expected}
    assert all(audits[p]['Status']==status for p,status in expected.items())
    identity=audits['05_evaluation/stage14a_identity_audit.json']
    frozen=audits['04_checkpoints/stage14a_all_frozen.json']
    expected_checkpoint_paths={str(cp_path(fold,variant).relative_to(PROJECT))
        for fold in (1,2,3) for variant in VARIANTS}
    assert len(frozen['Checkpoints'])==6
    assert {row['Path'] for row in frozen['Checkpoints']}==expected_checkpoint_paths
    tiny=audits['01_preflight/stage14a_tiny_audit.json']
    provenance=audits['09_reports/stage14a_predictor_data_provenance.json']
    evidence=read_json(ROOT/'06_bootstrap/stage14a_decision_evidence.json')
    metrics=pd.read_csv(ROOT/'05_evaluation/stage14a_ablation_metrics.csv')
    contrasts=pd.read_csv(ROOT/'06_bootstrap/stage14a_bootstrap_ci.csv')
    interaction=pd.read_csv(ROOT/'06_bootstrap/stage14a_interaction_ci.csv')
    switches=pd.read_csv(ROOT/'07_diagnostics/stage14a_mode_switch.csv')
    motion=pd.read_csv(ROOT/'07_diagnostics/stage14a_motion_contribution.csv')
    training=pd.read_csv(ROOT/'03_training/stage14a_training_summary.csv')
    efficiency=pd.read_csv(ROOT/'07_diagnostics/stage14a_efficiency.csv')
    assert len(training)==6 and list(zip(training.Fold,training.PaperModel))==[(f,'NG-'+v) for f in (1,2,3) for v in VARIANTS]
    assert not training.OuterTestUsed.any() and (training.Params==7425).all()
    init=pd.read_csv(ROOT/'01_preflight/stage14a_initialization.csv')
    cp_rows=[]
    for row in frozen['Checkpoints']:
        fold=row['Fold'];variant=row['Model'][-1];path=PROJECT/row['Path']
        assert sha256(path)==row['SHA256']
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['Fold']==fold and saved['Model']==variant
        model=fresh(fold,'cpu');assert state_sha(model)==init.loc[init.Fold==fold,'StateSHA256'].iloc[0]
        assert set(saved['state_dict'])==set(model.state_dict())
        model.load_state_dict(saved['state_dict'])
        assert sum(p.numel() for p in model.parameters())==7425
        config_path=ROOT/f'03_training/fold{fold}/{variant}/stage14a_training_config.json'
        c=read_json(config_path);assert sha256(config_path)==saved['config_sha256']
        assert c['protocol_sha256']==sha256(PROTOCOL)
        for p,h in c['new_sources_sha256'].items():assert sha256(ROOT/p)==h,p
        actual=pd.read_csv(ROOT/f'03_training/fold{fold}/{variant}/stage14a_batch_order.csv')
        for oldvariant in VARIANTS:
            old=pd.read_csv(S11B/f'03_training/fold{fold}/{oldvariant}/stage11b_batch_order.csv')
            prefix=min(len(old),len(actual))
            assert actual.OrderSHA256.iloc[:prefix].tolist()==old.OrderSHA256.iloc[:prefix].tolist()
        assert saved['initial_state_sha256']==init.loc[init.Fold==fold,'StateSHA256'].iloc[0]
        cp_rows.append({**row,'SharedInitialization':'PASS','HistoricalBatchOrdering':'PASS','OnlyNoGraphParameters':'PASS'})
    dump('09_reports/stage14a_checkpoint_control_audit.csv',cp_rows)
    figures=sorted((ROOT/'08_figures').glob('*.svg'))
    assert len(figures)>=6,'required six figure families must be rendered before finalization'
    assert all(p.with_suffix('.pdf').exists() for p in figures)
    figure_audits=sorted((ROOT/'08_figures').glob('*figure_audit.json'))
    assert figure_audits,'figure QA audit must exist before finalization'
    for path in figure_audits:
        assert read_json(path)['Status']=='PASS',str(path)
    decision=dict(Stage='Stage14A',Status='COMPLETE_AWAITING_BRAIN_AI_REVIEW',
        GraphIncrementSupported='SUPPORTED' if evidence['GraphIncrementRegisteredConditionsMet'] else 'NOT_SUPPORTED',
        LossIncrementSupported='SUPPORTED' if evidence['LossAcrossStructuresRegisteredConditionsMet'] else 'NOT_SUPPORTED',
        Rule='verbatim preregistered point<0, Bonferroni 98.75% CI upper<0 and at least 2/3 negative folds; graph=C, cross-structure loss=A AND D',
        RegisteredEvidence=evidence,ProtocolSHA256=sha256(PROTOCOL),NewFormalTrainingRuns=6,
        FrozenCandidateIdentity='PASS',NoGraphImplementationAudit='PASS',TrainingStatus='COMPLETE_6_OF_6',
        Bicycle='all four models bitwise-identical frozen FoldR2 route; no graph-Bicycle improvement claim',
        PredictorTrainingSceneOverlap='630/630; each OuterTest 210/210; Stage5A trained all TRAIN700',
        IndependentEvaluationStatus='NOT_INDEPENDENT_FROZEN_PREDICTOR_OOF',
        EvaluationLabel='nuScenes HeadTrain630 internal three-fold ranking OOF',
        OfficialVALTestExecuted=False,EndToEndRetrainingExecuted=False,
        ScientificInterpretationOwner='brain-AI; this JSON only applies registered evidence gates',
        NextAction='STOP after publication; await brain-AI review',FutureStageTrainingAuthorized=False)
    atomic_json(ROOT/'09_reports/stage14a_scientific_decision.json',decision)
    main_groups=['Overall','Vehicle','Pedestrian','Bicycle','MovingVehicle','StoppedVehicle','ParkedVehicle']
    main_metrics=metrics[metrics.Group.isin(main_groups)].copy()
    primary=contrasts[contrasts.Group=='Overall']
    required_switches=switches[(switches.Group.isin(main_groups)) & switches.Comparison.isin(['NG-C-NG-A','G-C-NG-C','G-C-G-A'])]
    freeze_time=datetime.fromtimestamp((ROOT/'04_checkpoints/stage14a_all_frozen.json').stat().st_mtime,timezone.utc).isoformat()
    total_seconds=float(training.Seconds.sum())
    lines=['# Stage14A controlled graph ablation execution report','',
        '**Execution complete. Awaiting brain-AI review.**',
        f"Registered qualification: GraphIncrementSupported={decision['GraphIncrementSupported']}; LossIncrementSupported={decision['LossIncrementSupported']}.",
        'These labels apply the frozen statistical rules. Paper writing and broader scientific judgment remain with brain-AI.','',
        '## Evaluation scope','',
        'nuScenes HeadTrain630 internal three-fold ranking OOF; 260151 full-horizon actor/window targets. Each scene belongs to exactly one 210-scene OuterTest fold. Corresponding ranking-head training uses 378 InnerTrain and 42 InnerDev scenes. The frozen Stage5A predictor trained on all 700 official TRAIN scenes, including 630/630 of these ranking scenes and 210/210 of every OuterTest fold. This is not independent end-to-end validation. Historical VAL150 was used for predictor selection and method development. No official VAL/test evaluation was executed in Stage14A.','',
        '## Frozen controls and integrity','',
        f'Base commit: `{BASE}`. Branch: `stage14a/paper-graph-ablation`. Protocol SHA256: `{sha256(PROTOCOL)}`.',
        'NoGraph registers only original G1 node_encoder, LayerNorm and scoring head, with the same 64-dimensional embedding and original-logit residual. Interaction message is exactly zero. Full cached node/edge/neighbor inputs are retained unchanged; they are not corrupted or shuffled. NoGraph has 7425 total/trainable parameters; G1 has 24066. The extra G1 interaction parameters and compute are part of the structural contrast, so this is not a parameter-count-matched or causal test.',
        'The original Stage11B split, InnerTrain-fitted normalization, shared initialization, optimizer, learning rate, weight decay, FP32, 128×8 microbatch accumulation, continuous 1024-target carry ordering, maximum 50 epochs and patience 5 are reused. Historical batch-order prefixes and shared initial weights match exactly. Checkpoint selection uses only the original fixed Vehicle/Pedestrian relative InnerDev score. The protocol retains original Stage11B R2 configuration metadata; Stage14A actually reuses frozen fold R2 and does not train R2.',
        'Loss A is original SoftCE; loss C is original normalized expected regret with unchanged 1 m floor. Exact original AST definitions are reused. Future labels remain detached and outside the seven forward inputs. GT poison, zero-message equivalence, shared initialization, frozen predictor gradient/state and candidate coordinate checks passed.',
        'All four models use identical six coordinate arrays. G-A/G-C scores and all 15 actor metrics reproduce original binary arrays bitwise. Published CSV summaries reproduce their original 12-significant-digit serialization precision. Bicycle uses the same predeclared corresponding fold R2 logits/probabilities/choices in all four models, bitwise. Its result is a routing policy, not evidence that Graph improves Bicycle.',
        f'Global six-checkpoint freeze file mtime (UTC): `{freeze_time}`. This is the filesystem modification time of the all-frozen manifest, reported with the gate/hash evidence; it is not independent timestamp notarization. Unified OuterTest evaluation is guarded until all 6 checkpoints exist and their SHA256 is verified. No OuterTest-based tuning or reselection occurred.','',
        '## Tiny preflight and six formal runs','',
        markdown_table(tiny['Results'],['Model','Status','Updates','InitialLoss','FinalLoss','Reduction','Threshold']), '',
        'A reduction is relative to the original entropy floor; C reduction is relative to initial normalized objective. Tiny weights are discarded before formal runs.',
        markdown_table(training.to_dict('records'),['Fold','PaperModel','SelectedEpoch','ExecutedEpochs','CheckpointScore','Seconds','Params']),
        f'Total six-run measured training wall time: {total_seconds:.3f} s ({total_seconds/3600:.4f} h), including InnerDev/I/O/checkpoint work; not pure CUDA kernel time.',
        'A pre-optimizer isolation block occurred when a source-hash scan touched evaluation code. It produced zero optimizer updates and zero checkpoints; the trace is preserved in 03_training/stage14a_pre_optimizer_guard_block.txt. The source-hash scope was restricted to training sources without changing model/loss/hyperparameters or weakening the guard. A preliminary label-gradient audit also required enabling gradients within its detached-label test; forward checks remained in eval/no_grad. Neither engineering correction was an experimental search.','',
        '## Four-model metrics','',
        'FDE/ADE/OracleGap/minFDE6 are meters. Count is full 12-step future actor/window targets, exactly the historical Stage11B denominator/mask. HitRate is the fraction whose top1 mode matches the lowest-index endpoint-FDE oracle. MR6 is the fraction with oracle endpoint FDE>2 m. These are sorting results over the same frozen candidates.',
        markdown_table(main_metrics.to_dict('records'),['Group','Model','Count','Top1FDE','Top1ADE','OracleGap','HitRate','minFDE6','MR6']), '',
        '## Preregistered comparisons and uncertainty','',
        'Delta=first model−second model; negative favors the first. 2000 paired whole-scene bootstrap draws, seed 2022, independently resample 210 scenes within each of three folds. Every actor/window of a drawn scene stays clustered and paired across models. Four co-primary Overall Top1FDE comparisons use Bonferroni family 4: 98.75% individual intervals, giving the preregistered familywise .05 criterion. 95% intervals are also reported descriptively. Type/motion groups and interaction are exploratory; they do not replace the registered comparisons.',
        markdown_table(primary.to_dict('records'),['Comparison','Count','DeltaTop1FDE','CI95Lower','CI95Upper','BonferroniCILower','BonferroniCIUpper','Fold1DeltaTop1FDE','Fold2DeltaTop1FDE','Fold3DeltaTop1FDE']), '',
        'Graph gate requires comparison C (G-C−NG-C) to satisfy all conditions. Cross-structure Loss gate requires both A (NG-C−NG-A) and D (G-C−G-A) to satisfy them. No post-hoc criterion changes or parameter searches were performed.',
        markdown_table(interaction[interaction.Group=='Overall'].to_dict('records'),['Comparison','Count','DeltaTop1FDE','CI95Lower','CI95Upper']),
        'The interaction is descriptive association of controlled contrasts, not causal identification.','',
        '## Mode changes, high-cost harm and motion states','',
        'Gross gain/harm are sums of negative/positive paired target FDE changes in meters. High-cost harm is the original comparison/group-specific largest ceil(0.1×number of positive harms) tail. Different comparisons have different tail memberships; this is not a fixed identical actor subset. Unchanged selected modes have exactly zero FDE difference.',
        markdown_table(required_switches.to_dict('records'),['Comparison','Group','Count','ChangedCount','ImprovedCount','WorsenedCount','GrossGain','GrossHarm','HighCostHarmCount','HighCostHarmSum','NetFDEDelta']), '',
        'Disjoint MovingVehicle/StoppedVehicle/ParkedVehicle/OtherVehicleState contributions are in 07_diagnostics/stage14a_motion_contribution.csv. These contributions sum to the Vehicle mean delta, preventing a static-target count effect from being silently described as a moving-vehicle improvement. Diagnostics alone do not establish a causal switching mechanism.','',
        '## Compute and parameters','',
        markdown_table(efficiency.to_dict('records'),['Fold','Model','TotalParameters','OptimizationTrainableParameters','BatchSize','MeanMSPerActor','PeakIncrementalAllocatedGPUMemoryBytes']),
        'Benchmark uses identical on-device cached seven-argument inputs, FP32, CUDA synchronization, five warmups and 20 rotating-order repeats. Only head forward is timed; frozen HiVT, packing, transfer and R2 routing are excluded. At inference every head has zero trainable parameters. CPU linear FLOP proxies and timing are separately documented in 01_preflight/stage14a_model_integrity.json. Full graph-input construction is retained for control and its removal is not claimed as measured end-to-end acceleration.','',
        '## Predictor data provenance and independent evaluation','',
        'Detailed scene-token evidence, original sampler replay, selected checkpoint ancestry and 700 training-shard hashes are in 09_reports/stage14a_predictor_data_provenance.json and its four CSV appendices. The selected Stage5A checkpoint saw all 700 training scenes and 16898 supervised windows. Historical VAL150 was checked every 500 steps, selected the predictor checkpoint, and informed Stage8/Stage9 development. No clean independent evaluation scene is verified within the registered local 850-scene corpus. A static official test name list does not establish available labels or a compatible custom three-type evaluator.',
        'Historical RTX3080 timing gives three predictor retraining runs a 6.40–11.69 h planning anchor, depending on 11500 vs 21000 update schedules. Adding historical nine A/C/R2 heads and candidate forward yields 7.48–12.77 h before new NG heads, preprocessing, full I/O and diagnostics. Stage14A measures the six NG runs separately above. These are wall/GPU-reservation estimates, not guarantees or pure CUDA training time. 700 training shards occupy 3.784 GiB; per 378 scene InnerTrain fold approximately 1.96–2.07 GiB. End-to-end retraining was not executed or authorized. Proper outer isolation would address predictor training overlap but would not make previously developed-on scenes pristine research holdouts.','',
        '## Figures and review package','',
        'Python figures provide editable SVG and PDF, plus PNG previews, source CSV, figure contract, captions and QA. Dataset, internal OOF protocol, sample counts, units and statistical scope are explicit. Cases are descriptive selections of actual improvements/failures, not an unbiased performance estimate.',
        *[f'- [ {p.stem} ](../08_figures/{p.name}) / [PDF](../08_figures/{p.with_suffix(".pdf").name})' for p in figures], '',
        'All new files remain under outputs/stage14a_paper_graph_ablation/. Local .pt/.npy/.log/cache artifacts are excluded from Git; checkpoint manifests and hashes are included. History, Stage12 failed experiments and five untracked Stage2C redraw files remain unchanged.','',
        '## End condition','',
        'Publish this branch without merge, then STOP. Await brain-AI review. No Stage14B, new predictor/semantic module, end-to-end retraining, official VAL/test evaluation or Diffusion Planning is started.']
    # Keep compact code literals readable in the delivered prose.
    text='\n\n'.join(lines)+'\n'
    (ROOT/'09_reports/stage14a_final_report.md').write_text(text)
    final_audit=dict(Status='PASS',HistoricalFilesAndCheckpoints='UNCHANGED',FrozenDataArrays='UNCHANGED',
        ProtocolSHA256=sha256(PROTOCOL),TrainingCheckpoints=cp_rows,FormalNewRuns=6,NoOtherModelsTrained=True,
        InitializationAndBatchOrdering='PASS',HistoricalGraphReproduction='BITWISE_PASS',
        CandidateIdentity='BITWISE_PASS',BicycleRoute='BITWISE_PASS',WholeSceneBootstrap='PASS',
        FigureFamilies=len(figures),PredictorOverlap='630/630',IndependentEvaluation=False,
        FigureQAAuditSHA256={str(path.relative_to(ROOT)):sha256(path) for path in figure_audits},
        FreezeFileMTimeUTC=freeze_time,FreezeTimeSource='filesystem manifest mtime with gate/hash evidence; not independent timestamp notarization',
        AuditFiles={p:sha256(ROOT/p) for p in expected},
        FinalReportSHA256=sha256(ROOT/'09_reports/stage14a_final_report.md'),
        ScientificDecisionSHA256=sha256(ROOT/'09_reports/stage14a_scientific_decision.json'),
        CompletedUTC=datetime.now(timezone.utc).isoformat(),NextAction='publish branch; STOP')
    atomic_json(ROOT/'09_reports/stage14a_final_integrity.json',final_audit)
    print('STAGE14A_FINAL_PACKAGE_PASS',decision['GraphIncrementSupported'],decision['LossIncrementSupported'],flush=True)

if __name__=='__main__':main()
