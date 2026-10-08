"""Evidence-linked feasibility report and complete freeze/history audits; no training API."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage10a_common import *
import pandas as pd

def table(rows):
    if not rows: return '(no rows)'
    cols=list(rows[0]); text='|'+'|'.join(cols)+'|\n|'+'|'.join('---' for _ in cols)+'|\n'
    for r in rows:
        values=[]
        for c in cols:
            v=r[c]
            if isinstance(v,(float,np.floating)): v=f'{v:.6f}'
            values.append(str(v))
        text+='|'+'|'.join(values)+'|\n'
    return text
def records(name): return pd.read_csv(ROOT/'06_tables'/name).to_dict('records')
def main():
    history=verify(history=True); frozen=read_json(FROZEN)
    assert frozen['status']=='FROZEN_ALL_COMPLETE' and frozen['training_prohibited']
    for rec in frozen['checkpoints']: assert sha256(PROJECT/rec['Path'])==rec['SHA256']
    assert sha256(REG)==frozen['registration_sha256']
    checks=('01_training/stage10a_engineering_audit.json','01_training/stage10a_tiny_gate.json',
        '03_evaluation/stage10a_complete.json','03_evaluation/stage10a_candidate_identity.json',
        '03_evaluation/stage10a_predictor_replay.json','03_evaluation/stage10a_reference_reproduction.json',
        '04_bootstrap/stage10a_bootstrap_audit.json','08_efficiency/stage10a_efficiency_audit.json',
        '05_diagnostics/stage10a_all_actor_inference.json')
    assert all(read_json(ROOT/p)['status']=='PASS' for p in checks)
    complete=read_json(ROOT/'03_evaluation/stage10a_complete.json')
    assert sha256(ROOT/'03_evaluation/stage10a_actor_results.csv')==complete['actor_csv_sha256']
    evaluation=read_json(ROOT/'03_evaluation/stage10a_headdev_registration.json')
    assert evaluation['evaluation_source_sha256']==sha256(ROOT/'03_evaluation/stage10a_evaluate.py')
    spec=read_json(PROTOCOL); decision=read_json(ROOT/'09_reports/stage10a_scientific_decision.json')
    counts=records('stage10a_target_counts.csv'); checkpoint=records('stage10a_checkpoint_manifest.csv')
    mainrows=records('stage10a_main_results.csv'); types=records('stage10a_type_results.csv'); motion=records('stage10a_motion_results.csv')
    ci=records('stage10a_bootstrap_ci.csv'); cimain=[r for r in ci if r['Metric']=='Top1FDE' and r['Group'] in ('Overall','Vehicle','Pedestrian','Bicycle')]
    tiny=[read_json(ROOT/'01_training'/e/'stage10a_tiny_audit.json') for e in EXPERTS]
    summaries=[read_json(ROOT/'01_training'/e/'formal/stage10a_training_summary.json') for e in EXPERTS]
    engineering=read_json(ROOT/'01_training/stage10a_engineering_audit.json')
    candidate=read_json(ROOT/'03_evaluation/stage10a_candidate_identity.json')
    reference=read_json(ROOT/'03_evaluation/stage10a_reference_reproduction.json')
    replay=read_json(ROOT/'03_evaluation/stage10a_predictor_replay.json')
    neighbor=records('stage10a_neighbor_type_counts.csv'); changes=records('stage10a_mode_change.csv')
    def concise(rows):
        return [{k:r[k] for k in ('Group','Model','Count','Top1FDE','Top1ADE','OracleGap','HitRate','MRR','minADE6','minADEOracle6','minFDE6','MR6','SoftCE')} for r in rows]
    def group(g): return concise([r for r in types if r['Group']==g])
    def section(name,body): return '【'+name+'】\n\n'+body.strip()+'\n\n'
    deltas={r['Group']:r for r in ci if r['Metric']=='Top1FDE'}
    def delta_text(g):
        r=deltas[g]
        return f"{r['Delta']:+.6f}m, descriptive HeadDev 95% CI [{r['CI_lower']:+.6f}, {r['CI_upper']:+.6f}]"
    report=section('Research Question',
        'Result: Stage10A='+decision['Stage10A']+'. Both expert types fail the predeclared feasibility targets; Bicycle stays exactly R2. '
        'Vehicle DeltaTop1FDE='+delta_text('Vehicle')+'; Pedestrian '+delta_text('Pedestrian')+'; Overall '+delta_text('Overall')+'. '
        'The research question is whether fully independent Vehicle/Pedestrian G1 reranking experts improve Vehicle ranking while preserving or improving Pedestrian ranking, with Bicycle exactly frozen R2. '
        'This is one predeclared low-cost feasibility run on HeadTrain630/HeadDev70. Stage10 uses scene-level multi-actor history and predicted future mode interactions, Th=5, Tf=12, K=6. '
        'The new branch starts from `'+BASE+'`. All new files are confined to `outputs/stage10a_type_specific_future_graph/`. '
        'Historical conclusions remain Stage8 AgentGraph=SUPPORTED, SemanticGraph=NOT_SUPPORTED, FSCG=NOT_SUPPORTED; '
        'Stage9 TAFIG=NOT_SUPPORTED, PedestrianNegativeTransferResolved=NO, VehicleBenefitRetained=NO.')
    report+=section('Frozen Predictor',
        'Stage5A remains eval(), requires_grad=False, without optimizer membership. Checkpoint SHA256: `'+PREDICTOR_SHA+'`. '
        'No candidate trajectory training occurred. The fresh HeadDev replay checked '+str(replay['source_windows_bitwise_replayed'])+
        ' original TRAIN scene-windows against frozen raw/ego prediction hashes; '+str(replay['HeadDev_windows'])+' windows belong to HeadDev. '
        'All 29,934 full-horizon HeadDev target identities, candidate coordinates, GT and future masks retain the frozen protocol.')
    report+=section('Type-Specific Architecture',
        'Two separate complete Stage8 G1 networks, each 24,066 trainable parameters; total 48,132. Independent NodeEncoder, InteractionMessage, InteractionAttention, LayerNorm and RankingHead. '
        'No trainable tensor storage is shared. Node15D/edge17D, hidden64, one message layer, nearest eight current-valid non-self actors within <=50m, all six neighbor modes. '
        'Vehicle, Pedestrian and Bicycle remain eligible neighbors for both experts. No semantic input, map module, adapter, learned gate or macro loss. '
        'The wrapper invokes the unchanged frozen G1 forward with unused map slots set to None; bitwise forward equivalence passed. '
        'Vehicle/Pedestrian logits are original Stage5A logits plus their own expert residual. Bicycle logits/probabilities route directly to frozen R2.\n\n'+table(neighbor))
    report+=section('Vehicle Expert',
        'Fresh independent seed2022, with G1 interaction submodule seed2123, final score weight/bias zero. Vehicle target loss only; all neighbor types retained. '
        'No trained Stage8/Stage9 graph weights loaded.\n\n'+table([checkpoint[0]]))
    report+=section('Pedestrian Expert',
        'Fresh independent seed2123, with G1 interaction submodule seed2224, final score weight/bias zero. Pedestrian target loss only; all neighbor types retained. '
        'No shared backbone or optimizer and no trained Stage8/Stage9 weights loaded.\n\n'+table([checkpoint[1]]))
    report+=section('Bicycle Frozen R2',
        'R2 SHA256: `'+R2SHA+'`. Bicycle keeps full histories, six predictor candidates, graph context, six final probabilities, and contributes to Overall. '
        'Bicycle is excluded only from expert training losses. No Bicycle sample was removed. '
        'Before training, all 3,138 TRAIN700 Bicycle targets retained bitwise R2 logits/probabilities; 128 actual Bicycle candidate tensors and all 3,138 candidate routing probes retained identity. '
        'After training, all 158 HeadDev Bicycle logits, probabilities and actual candidate tensors remained bitwise R2. '
        'A real TRAIN all-actor inference audit also routes every current-valid actor, including those without full future supervision, without a future-target selection mask; all three types receive6 candidates/probabilities. '
        'Changing GT/future/target masks leaves trained outputs bitwise identical.')
    report+=section('Training Split',
        'Exact frozen Stage6 seed2022 HeadTrain630/HeadDev70 scene split, SHA256 `'+sha256(SPLIT)+'`. '
        'Full target scene/sample/instance/node/window identities, FDE/ADE arrays and original logits join exactly. '
        'No altered time windows, horizon masks, candidate geometry or type selection.\n\n'+table(counts)+
        '\nNormalization directly reuses frozen Stage8 HeadTrain-only values, SHA256 `'+sha256(NORM)+'`; no refitting with HeadDev or VAL. '
        'The R2 branch separately reuses its existing frozen 19D normalized inputs. '
        'A pre-registration metadata preflight found Stage6 lowercase display labels versus Stage10 uppercase display labels; the new metadata join checks exact lowercase labels and actor_type_id, then changes presentation only. '
        'No frozen data/model/protocol was changed and no optimizer had run; this is logged in `00_manifest/stage10a_engineering_preflight.json`.')
    report+=section('No Future Leakage',
        'The observable-window allowlist exposes only histories, current type, frozen predicted trajectories and original logits/probabilities. '
        'Future GT is used solely for q=softmax(-FDE/1m) supervision and offline evaluation/displacement groups. '
        'NaN GT, inverted future/target masks and NaN semantic metadata leave graph features bitwise identical on fixed target rows. '
        'Live graph and frozen R2 input identity passed for all HeadDev targets. Runtime file-access guards reject official VAL/test cache/shard paths. '
        'Only official TRAIN source batches intersecting HeadDev70 are replayed. Official VAL was not opened, evaluated, normalized, used for selection or training.')
    report+=section('Tiny Overfit',
        'One fixed 128-target same-type HeadTrain set per expert, respective expert seed, 300 AdamW optimizer updates. '
        'ExcessLossReduction=1-(finalCE-H(q))/(initialCE-H(q)); mean target entropy is the feasible lower bound. Both exceed the fixed 90% gate. '
        'Each graph module has finite nonzero gradients; predictor/R2 gradient counts zero; candidates detached; no NaN/Inf. '
        'Formal models were reset to registered initial states, without tiny weight reuse.\n\n'+table([{k:r[k] for k in (
            'Expert','status','InitialLoss','FinalLoss','EntropyFloor','ExcessLossReduction','Updates','graph_gradient_norm')} for r in tiny]))
    report+=section('Independent Training',
        'Two fully independent optimizers, AdamW lr1e-3/weight_decay1e-4, FP32, effective batch1024 using 128 microbatches x8. '
        'Loss is mean SoftCE over only the expert target type; no type weighting, macro objective, uncertainty, semantic or gate losses. '
        'Every epoch residual is prepended to the next shuffled epoch; final residual occurrences are retained in each last checkpoint. '
        'Every own-type HeadTrain identity was optimized at least once; zero other-type or HeadDev backprop targets. '
        'No short effective batches or target deletion. Max50 epochs, strict improvement, patience5.\n\n'+table(records('stage10a_training_summary.csv')))
    report+=section('Checkpoint Selection',
        'Registered before any tiny or formal update: Vehicle selects minimum complete Vehicle HeadDev SoftCE; Pedestrian selects minimum complete Pedestrian HeadDev SoftCE. '
        'Top1ADE/Top1FDE are recorded each epoch but never select checkpoints. Earlier epoch wins exact ties. '
        'Both checkpoints were frozen together before unified HeadDev evaluation, and further training/overwriting is prohibited by the manifest. '
        'Unified own-type selected checkpoint CE/ADE/FDE matches training records within1e-6; frozen G1 HeadDev SoftCE reproduction difference '+str(reference['G1_absdiff'])+'.\n\n'+table(checkpoint))
    report+=section('HeadDev Results',
        'Only 70 HeadDev scenes, 29,934 full-horizon targets. R0 original logits, frozen R2, frozen Stage8 G1 reference, and the combined DualExpert are evaluated on identical candidates. '
        'All required groups appear in `06_tables/stage10a_type_results.csv` and `stage10a_motion_results.csv`. '
        'Top1FDE is primary; negative DualExpert-R2 difference improves ranking. HitRate/MRR use the lowest-index FDE-best candidate and stable descending probability order; HitRate is a ranking oracle agreement metric, not behavior-prediction accuracy. '
        'OracleGap=Top1FDE-minFDE6; MR6 threshold2m. Historic minADE6 is ADE at the FDE-best mode; independent ADE oracle is additionally reported as minADEOracle6.\n\n'+table(concise(mainrows))+'\n'+table(concise(motion)))
    report+=section('Vehicle Results',
        'Vehicle DeltaTop1FDE='+delta_text('Vehicle')+'; this misses the -0.03m GO target and the conditional requirement of any Vehicle improvement. '
        'MovingVehicle delta='+delta_text('MovingVehicle')+' and Vehicle>5m delta='+delta_text('Vehicle>5m')+'. '
        'Those subgroup point improvements coexist with Vehicle aggregate deterioration; both corresponding intervals cross0. '
        'ParkedVehicle delta='+delta_text('ParkedVehicle')+' and StoppedVehicle delta='+delta_text('StoppedVehicle')+'. '
        'All groups are retained in the aggregate and reported without selecting only improvements.\n\n'+
        table(group('Vehicle'))+'\n'+table([r for r in cimain if r['Group']=='Vehicle']))
    report+=section('Pedestrian Results',
        'Pedestrian DeltaTop1FDE='+delta_text('Pedestrian')+'; its descriptive selected-split interval lies above0 and the point delta also exceeds the +0.01m conditional tolerance. '
        'Pedestrian<5m delta='+delta_text('Pedestrian<5m')+' while Pedestrian>5m delta='+delta_text('Pedestrian>5m')+'. '
        'The independent Pedestrian expert has lower own-type SoftCE than frozen R2 and G1, but its aggregate Top1FDE remains worse. '
        'The fixed SoftCE checkpoint rule was followed; no FDE-driven reselection or model change follows this result.\n\n'+
        table(group('Pedestrian'))+'\n'+table([r for r in cimain if r['Group']=='Pedestrian']))
    report+=section('Bicycle Results',table(group('Bicycle'))+'\nAll Bicycle model outputs match frozen R2 exactly; DualExpert-R2 Top1FDE difference and all bootstrap draws equal0. '+
        'The 158 target count limits independent claims about Bicycle generalization.\n\n'+table([r for r in cimain if r['Group']=='Bicycle']))
    report+=section('Overall Results',table(concise(mainrows))+'\n'+table([r for r in cimain if r['Group']=='Overall']))
    report+=section('Bootstrap',
        'Paired whole-scene cluster bootstrap over all70 HeadDev scenes, 1000 draws, seed2022, shared resampling indices for every group/metric. '
        'Keep each selected scene intact, including repeated windows/actors, and compute actor-weighted means with duplicated scene multiplicities. '
        'Percentile95% intervals on DualExpert-R2, with Top1FDE/Top1ADE/HitRate for all10 requested groups; 30 rows. '
        'These intervals screen feasibility on a checkpoint-selection split and do not provide independent confirmation or causal evidence. '
        'No interval-based seed/model/hyperparameter choice occurred.\n\n'+table(cimain))
    report+=section('Candidate Identity',
        'PASS: fresh predictor raw/ego tensors match frozen source bitwise; all model routes share the same detached six candidates. '
        'maxdiff candidates/minADE6/minADEOracle6/minFDE6/MR6 =0. '
        'Bicycle logits/probability/candidates match frozen R2 bitwise for every158 HeadDev Bicycle target. '
        'Predictor/R2 states unchanged, gradients0. Initialization logits/probability maxdiff0 on1024 Vehicle and1024 Pedestrian targets.\n\n'+table(records('stage10a_candidate_identity.csv')))
    report+=section('Negative Transfer Analysis',
        'The design removes shared trainable parameters across Vehicle/Pedestrian target losses, while preserving heterogeneous interaction neighbors. '
        'HeadDev group deltas and mode-change counts characterize whether this single independent-expert run meets the frozen screening goals. '
        'Historical Stage8/Stage9 official VAL failures stay unchanged; results on different splits cannot be combined into a causal proof that parameter sharing caused those failures.\n\n'+
        table(changes)+'\nThe larger Overall HitRate/MRR values do not establish a Top1FDE benefit: the primary Overall delta is '+delta_text('Overall')+'. '
        'Complete parameter separation did not meet either aggregate Vehicle improvement or Pedestrian preservation in this run. '
        'These observations reject the specified feasibility screen; they do not identify a causal failure mechanism or prove all independent experts ineffective. '
        'NegativeTransferMitigated='+decision['NegativeTransferMitigated']+' under the predeclared HeadDev rule. '
        'YES requires Vehicle point improvement plus nonpositive Pedestrian delta and exact Bicycle preservation; PARTIAL allows Pedestrian<=+0.01m; otherwise NO.')
    report+=section('Limitations',
        'The Stage10 hypothesis was motivated by prior official VAL outcomes. HeadDev also selects both expert checkpoints, and Stage5A/R2 have historical training/checkpoint-selection exposure. '
        'This report therefore supplies HeadDev feasibility evidence, not an independent end-to-end evaluation. '
        'One fixed seed per expert, overlapping motion groups and small Bicycle sample; bootstrap is unadjusted descriptive screening. '
        'Candidate oracle geometry is frozen and does not improve through reranking. '
        'Efficiency below measures only normalized-tensor reranker forward, excluding predictor/graph construction/transfer, not end-to-end latency. '
        'No case-based model selection or additional model/seed/hyperparameter search. Official VAL/test and Stage10B were not executed.\n\n'+table(records('stage10a_efficiency.csv')))
    report+=section('Scientific Decision',
        'Frozen GO rule: Vehicle delta<=-0.03m, Pedestrian delta<=0m, Bicycle bitwise R2, Overall delta<0m and all engineering gates PASS. '
        'Otherwise CONDITIONAL_GO requires Vehicle delta<0m, Pedestrian delta<=+0.01m, Bicycle exact and Overall delta<0m with all gates PASS; otherwise STOP. '
        'Auxiliary label definitions are frozen in `00_manifest/stage10a_protocol.json`; bootstrap CI does not substitute for the point-estimate screening rule.\n\n'+
        '\n'.join(k+' = '+v for k,v in decision.items())+'\n\nSTOP after this phase, regardless of feasibility label. '
        'ReadyStage10B is only a recommendation for review. Do not run official VAL, train more models or execute Stage10B automatically. Wait for 大脑AI review.')
    headings=('Research Question','Frozen Predictor','Type-Specific Architecture','Vehicle Expert','Pedestrian Expert',
        'Bicycle Frozen R2','Training Split','No Future Leakage','Tiny Overfit','Independent Training','Checkpoint Selection',
        'HeadDev Results','Vehicle Results','Pedestrian Results','Bicycle Results','Overall Results','Bootstrap','Candidate Identity',
        'Negative Transfer Analysis','Limitations','Scientific Decision')
    assert all(report.count('【'+h+'】')==1 for h in headings)
    out=ROOT/'09_reports/stage10a_feasibility_report.md'; out.write_text(report)
    atomic_json(ROOT/'09_reports/stage10a_final_audit.json',{'status':'COMPLETE','engineering_status':'PASS','STOP':True,
        'official_VAL_executed':False,'test_executed':False,'Stage10B_executed':False,
        'historical_tracked_files_unchanged':len(history['historical_files']),'historical_dependencies_unchanged':True,
        'registration_protocol_training_sources_unchanged':True,'frozen_checkpoints_unchanged':True,
        'checkpoint_manifest_sha256':sha256(FROZEN),'actor_csv_sha256':complete['actor_csv_sha256'],
        'report_sha256':sha256(out),'required_sections':len(headings),'scientific_decision':decision,
        'HeadDev_targets':29934,'HeadDev_scenes':70,'Bicycle_targets':158,'bootstrap_rows':len(ci),
        'HeadDev_feasibility_only':True,'independent_final_statistical_confirmation':False})
    print('STAGE10_FINAL_PASS_AND_STOP',decision,flush=True)

if __name__=='__main__': main()
