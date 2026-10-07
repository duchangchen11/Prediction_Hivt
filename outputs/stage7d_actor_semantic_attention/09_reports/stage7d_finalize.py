"""Evidence-first experiment report and final scope/integrity checks."""
from pathlib import Path
import sys,json,ast,subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'04_evaluation'))
from stage7d_analysis_common import *

def block(frame):
    # Native Markdown serialization avoids adding any environment dependency.
    def cell(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6f}' if np.isfinite(v) else 'NA'
        return str(v).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,frame.columns))+' |',
                      '| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(cell(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])

def main():
    frozen=verify_previous(shards=True);prereg=read_json(PREREG)
    for name,digest in prereg['training_source_sha256'].items():assert sha256(ROOT/name)==digest
    training=read_json(SUMMARY);decision=read_json(ROOT/'04_evaluation/stage7d_scientific_decision.json')
    neutral=read_json(ROOT/'00_manifest/stage7d_neutral_initialization_audit.json')
    identity=read_json(ROOT/'01_data_audit/stage7d_data_identity_audit.json')
    capture=read_json(ROOT/'02_model_audit/stage7d_attention_capture_integrity.json')
    final=read_json(ROOT/'04_evaluation/stage7d_on_final_metrics.json')
    assert all(a['status']=='PASS' for a in (neutral,identity,capture))
    assert read_json(ROOT/'02_model_audit/stage7d_scope_audit.json')['status']=='PASS'
    assert training['from_scratch'] and not training['baseline_trained_weights_loaded'] and not training['test_used']
    assert training['warmup_steps']==5000 and training['NLL_steps']<=16000 and training['final_executed_global_step']<=21000
    assert training['checkpoint_sha256']==sha256(BEST)
    cp=torch.load(BEST,map_location='cpu',weights_only=False)
    assert cp['metadata']['additional_parameter_count']==744
    assert cp['metadata']['selection_metric']=='overall minFDE6'
    curve=pd.read_csv(CURVE);assert not curve[['VAL_overall_FDE','train_loss']].isna().any().any()
    bias_curve=pd.read_csv(ROOT/'03_training/stage7d_bias_curve.csv')
    assert len(bias_curve)==len(curve)*4 and set(bias_curve.global_step)==set(curve.global_step)
    assert np.isfinite(bias_curve[['mean_abs_bias','median_abs_bias','p95_abs_bias','max_abs_bias']].to_numpy()).all()
    ci=pd.read_csv(ROOT/'06_tables/stage7d_bootstrap_ci.csv')
    main=pd.read_csv(ROOT/'06_tables/stage7d_main_results.csv');motion=pd.read_csv(ROOT/'06_tables/stage7d_vehicle_motion_results.csv')
    semantic=pd.read_csv(ROOT/'06_tables/stage7d_semantic_subgroup_results.csv')
    att=pd.read_csv(ROOT/'06_tables/stage7d_attention_relevance.csv');turn=pd.read_csv(ROOT/'06_tables/stage7d_turn_attention.csv')
    bias=pd.read_csv(ROOT/'06_tables/stage7d_bias_statistics.csv');eff=pd.read_csv(ROOT/'06_tables/stage7d_efficiency.csv')
    corr=pd.read_csv(ROOT/'06_tables/stage7d_attention_error_correlation.csv')
    conditioning=read_json(ROOT/'02_model_audit/stage7d_actor_conditioning_audit.json')
    # Validate actual vector exports and embedded selectable text, not rcParams alone.
    import xml.etree.ElementTree as ET
    exports=[]
    for p in sorted((ROOT/'05_figures').glob('*.pdf')):
        text=subprocess.check_output(['pdftotext',str(p),'-'],text=True)
        fonts=subprocess.check_output(['pdffonts',str(p)],text=True)
        assert text.strip() and 'TrueType' in fonts and 'Type 3' not in fonts
        svg=p.with_suffix('.svg');tree=ET.parse(svg);texts=tree.findall('.//{http://www.w3.org/2000/svg}text')
        assert texts and p.with_suffix('.png').exists()
        exports.append({'name':p.stem,'PDF_text_characters':len(text),'PDF_font_listing':fonts,
                       'SVG_text_nodes':len(texts),'bundle_sha256':{e:sha256(p.with_suffix('.'+e)) for e in ('pdf','svg','png')}})
    assert len(exports)>=7
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
    changed=subprocess.check_output(['git','diff','--name-only','7df27d2091425c6836a66d6987714b1d9569efb5'],text=True).splitlines()
    assert all(n.startswith('outputs/stage7d_actor_semantic_attention/') for n in changed)
    audit={'status':'PASS','historical_files_verified':len(frozen['files']),'scene_shards_verified':len(frozen['scene_shards']),
        'unrelated_untracked_preserved':len(frozen['unrelated_untracked']),'training_sources_match_preregistration':True,
        'parameters':646745,'additional_parameters':744,'best_checkpoint_sha256':sha256(BEST),
        'bias_audits_at_every500_steps':True,'formal_runs':1,'seed':2022,'test_used':False,
        'Stage7E_executed':False,'Stage7B_executed':False,'R2_executed':False,'merged_main':False,
        'exports':exports,'visual_QA':'final PNG bundles inspected separately before publication'}
    atomic_json(ROOT/'00_manifest/stage7d_final_integrity_audit.json',audit)
    full=main[main.Horizon=='full_horizon'];sem=semantic[semantic.Horizon=='full_horizon']
    bias_final=final['bias_audit'][0]
    sections=[]
    def section(title,text):sections.append('【'+title+'】\n\n'+text+'\n')
    section('Motivation','Stage7A remains NOT_SUPPORTED / PaperUsableSemantic=NO / ReadyStage7B=NO. Frozen Stage7C classified RELEVANCE_MISALLOCATION and recommended score-only actor-conditioned semantic attention. This is a candidate explanation with localized descriptive evidence, not established causality. Stage7D tests forecasting performance and attention allocation independently; Stage3B is the sole primary baseline.')
    section('Architecture',(ROOT/'02_model_audit/stage7d_architecture.md').read_text()+'\n\nFrozen nine semantic fields and original categorical zeros are reused. Th=5, Tf=12, K=6, width64, eight heads, temporal4/global3, dropout0.1, radius50m. No semantic residual, decoder modification, reliability or dynamic signal input. Geometry-only K/V describes direct forward dependence: learned shared weights can differ after separately training Stage7D.')
    section('Neutral Initialization',f"PASS. Shared parameters and buffers: exact equality. raw_prediction/mode_logits/mode_prob maximum differences: {neutral['output_max_differences']}. Original attention score difference: {neutral['original_attention_score_max_diff']}. Bias starts at exactly0. Six real TRAIN graphs; deterministic CUDA only for controlled equality. Nonzero-bias semantic perturbation K/V differences: {neutral['nonzero_bias_semantic_perturbation_KV_max_differences']}.")
    section('Data Integrity','PASS: 100 random TRAIN and100 random VAL windows, seed2022; every original field bitwise equal, only lane_semantic added. Audited Stage7A map metadata reused, no map/trajectory/edge regeneration. Official TRAIN700/VAL150; test unused. Fresh evaluation:150 scenes,3603 supervised windows,54990 full-horizon +30037 partial =85027 targets; prediction NaN=0/Inf=0. Full-horizon semantic groups and signed GT turns are identity/GT-hash checked frozen offline sidecars, never branch inputs.')
    tiny=read_json(ROOT/'03_training/stage7d_tiny_overfit.json')
    section('Training',f"Fresh seed2022 canonical initialization; no trained baseline weights. Fixed six TRAIN graphs cover V/P/B,moving,left/right connectors,controls,crosswalks; tiny200 updates PASS, first20 mean loss={tiny['initial20_loss_mean']:.6f}, last20={tiny['final20_loss_mean']:.6f}, bias and attention become finite nonzero. Both MLP layers have finite nonzero gradients within10 updates, shared modules connected.\n\nFormal Protocol1: AdamW, decay1e-4,natural taxonomy distribution,batch16. Fixed warmup5000 at LR.001, own best source step={training['warmup_best_source_step']}. Model/optimizer/Python/NumPy/Torch/CUDA RNG restored at transition; LR alone becomes.0001, canonical NLL sampler resets. NLL updates={training['NLL_steps']}, final global={training['final_executed_global_step']}, best global={training['best_global_step']}, stop={training['stop_reason']}. Every500 fullVAL, strict overall full-horizon minFDE selection,patience5,hard21000. No subgroup/Top1 checkpoint selection or tuning. Training source commit={training['training_code_git_commit']}.\n\nEvery validation logs exact absolute-bias statistics over all retained VAL edges/heads, including context actors; maximum observed |bias|={bias_curve.max_abs_bias.max():.6f}; all finite. Final mean/median/p95/max |bias|={bias_final['mean_abs_bias']:.6f}/{bias_final['median_abs_bias']:.6f}/{bias_final['p95_abs_bias']:.6f}/{bias_final['max_abs_bias']:.6f}. Large absolute score biases alone cannot establish mechanism because actor/head constants cancel in softmax.")
    section('Main Forecasting Results',block(full)+'\n\nFull and partial metrics are in the source tables. minADE uses the best-FDE candidate, MR is endpoint>2m, Top1 is argmax mode probability; NLL retains the original best summed-L2 mode/valid-coordinate definition. Stage3B frozen reference Overall minFDE=1.343260 /Top1FDE=2.687984; Stage7A=1.350546/2.749482. Historical scientific decisions remain unchanged.')
    section('Moving Vehicle',block(motion[motion.Horizon=='full_horizon'])+'\n\nVehicle>5m uses the frozen GT endpoint displacement threshold and is offline evaluation only. Moving is the original t0 taxonomy annotation, not a new speed threshold.')
    section('Turning Vehicle',block(sem)+'\n\nTurningVehicle_GT remains1663, GT-left831/GT-right832. Endpoint displacement>5m, first/last1s secants>0.5m, absolute wrapped heading change>20°. Intersection/NearTurn/NearControl strict20m whole-centerline context definitions are reused without changes. These masks are never used for training or selection.')
    section('Attention Relevance',block(att[['Group','Model','Count',*ATTENTION_METRICS]])+'\n\nStage7C post-softmax observer reused;100 random VAL batches PASS, all three output differences<1e-6, dropout disabled, model state unchanged. Baseline/Stage7A attention ledger is frozen. GT relevance is future polyline-to-current-line-segment distance, not segment-start distance. Primary mass population is Vehicle; Moving is prespecified secondary. Nearest rank considers all current graph segments, zero for nonedges, average ties; tie/coverage limitations match Stage7C. Entropy/Top-K distance have missing values for actors with zero incoming edges; paired finite denominators are explicitly recorded in source tables. Attention sums are checked per actor/head. Captured-forward metrics reproduce fresh evaluation within declared numerical tolerance.')
    section('Turn Consistency',block(turn)+'\n\nHighest-attention connector direction match is an offline explanatory match rate, not prediction accuracy. NO_CONNECTOR counts as a nonmatch in the primary denominator. Correct/opposite labels use frozen signed GT heading, and straight/unknown mass are reported separately.')
    mechanism_rows={(row.Group,row.Metric):row for row in ci.itertuples()}
    relevance=mechanism_rows['Vehicle','GTRelevantMass2m']
    correct=mechanism_rows['TurningVehicle_GT','CorrectTurnMass']
    match=mechanism_rows['TurningVehicle_GT','TopConnectorMatchRate']
    sections[-2]+=f"\nVehicle relevant mass decreases by {relevance.Delta:.6f},95%CI=[{relevance.CI_lower:.6f},{relevance.CI_upper:.6f}]; Moving relevant mass also decreases reliably. The lower entropy accompanies lower GT relevance and does not establish better selection.\n"
    sections[-1]+=f"\nTurning correct-turn mass decreases by {correct.Delta:.6f},95%CI=[{correct.CI_lower:.6f},{correct.CI_upper:.6f}]; connector match decreases by {match.Delta:.6f},CI=[{match.CI_lower:.6f},{match.CI_upper:.6f}]. Opposite-turn mass also decreases, while straight mass increases. Neither prespecified mechanism improvement leg is supported; Stage7D does not restore the Stage3B attention pattern. These are descriptive comparisons, not causal explanations of errors.\n"
    section('Bias Interpretation',block(bias[['ActorGroup','SemanticGroup','Edges','mean_bias','mean_abs_bias']])+f"\n\nEight signed head means and absolute means are in stage7d_bias_statistics.csv. Groups overlap. Empty groups have undefined means and do not imply prediction NaNs. Observed type-conditioned means mix different positions; the additional controlled table holds semantic pattern and relative XY fixed, varying only actor type across observed patterns and five fixed XY positions. Maximum raw type response difference={conditioning['same_semantics_same_position_type_response_max_difference']:.6f}; maximum type difference in semantic-versus-zero contrast={conditioning['type_difference_in_semantic_vs_zero_contrast_max']:.6f}. These describe network response, not a physical causal effect or independent evidence of forecasting gain.")
    section('Bootstrap',block(ci)+'\n\nPaired whole-scene bootstrap:150 official VAL scenes,1000 draws,seed2022, actor-window sums/counts pooled for each draw,percentile95CI. Delta=Stage7D−Stage3B. Forecasting improvement has negative delta; attention mass/match improvement has positive delta. Primary Overall minFDE is distinct from exploratory unadjusted overlapping subgroup/mechanism comparisons. No multiple-comparison correction or seed variance estimate. Per-scene sufficient statistics are archived.')
    section('Efficiency',block(eff)+'\n\n500 paired forward measurements, alternating model order,eight fixed VAL batches,batch16; warmed3 paired passes per batch,CUDA events and synchronization. Data loading excluded; both models resident. Baseline receives original fields and shares original tensors with semantic input. Absolute and incremental peak allocated memory are reported; hardware/timing environment limits generalization.')
    section('Attention-Error Association',block(corr)+'\n\nSpearman and p-values are descriptive only, with clustered/overlapping observations. They do not replace scene bootstrap or support causality. Checkpoint selection did not use these associations.')
    cases=read_json(ROOT/'05_figures/stage7d_case_selection.json')
    section('Case Studies','```json\n'+json.dumps(cases,ensure_ascii=False,indent=2)+'\n```\n\nThree prespecified extreme-delta illustrations: turning improvement,moving improvement,degradation,distinct actor identities. Every panel shares actor,GT,map,axis limits, and attention color scale. Both model best-FDE trajectories appear with semantic turn colors and separate control/crosswalk markers. Cases illustrate possibilities and cannot establish their prevalence or the overall decision.\n\nThe turning improvement has zero relevant/correct-turn mass in both models for its retained edges. The moving improvement accompanies lower relevant mass; the degradation accompanies higher relevant mass. These examples show that relevance shifts and error shifts need not follow a monotonic relationship for individual actors.')
    section('Limitations','Static map semantics only; no dynamic light states. Turn labels derive from audited centerlines and fixed20°/150° rules. GT relevance/turning/endpoint groups are offline explanatory definitions unavailable to model inputs. One training seed; checkpoint selection and analysis use official VAL, with no independent test claim. The same early-stop protocol can select different actual update counts; this is not an equal-compute comparison. Attention rank ties, segment density, local-edge truncation and empty incoming sets constrain interpretation. Secondary groups overlap and their CIs are unadjusted. Semantic bias changes selection directly, while shared geometry-only representations can change indirectly through training. The bias also consumes relative geometry and actor type, so performance differences do not isolate semantic information from those additional score inputs. No reliability fusion was executed.')
    conclusion='```text\n'+'\n'.join(f"{k} = {decision[k]}" for k in ('Stage7D','AttentionMechanism','PaperUsableSemantic','SemanticRoute','ReadySemanticReranking'))+'\n```'
    conclusion+='\n\nRegistered performance and mechanism rules are evaluated separately. SUPPORTED uses the inherited marked-harm guard (>5% relative AND reliable worsening); TARGETED_SUPPORTED rejects any reliable Vehicle/Pedestrian worsening and permits at most0.5% overall point increase with CI including0. Mechanism PARTIAL means exactly one of the two prespecified improvement legs. No result alters these rules.'
    forecast={(row.Group,row.Metric):row for row in ci.itertuples()}
    overall=forecast['Overall','minFDE6'];vehicle=forecast['Vehicle','minFDE6'];ped=forecast['Pedestrian','minFDE6']
    conclusion+=f"\n\nOverall delta={overall.Delta:+.6f},95%CI=[{overall.CI_lower:+.6f},{overall.CI_upper:+.6f}]. Vehicle delta={vehicle.Delta:+.6f},CI=[{vehicle.CI_lower:+.6f},{vehicle.CI_upper:+.6f}]. Pedestrian delta={ped.Delta:+.6f},CI=[{ped.CI_lower:+.6f},{ped.CI_upper:+.6f}]. Reliably improved prespecified targeted groups={decision['reliably_improved_targeted_groups']}; reliable Vehicle/Pedestrian harms={decision['any_reliable_Vehicle_Pedestrian_harm']}. These, rather than case examples or attention changes, determine forecasting support."
    if decision['Stage7D']=='NOT_SUPPORTED':conclusion+='\n\nSemantic modeling experiments Stage7A + Stage7D did not produce reliable forecasting gains under the frozen protocol. This decision concerns the prespecified overall/priority-vehicle support rules; any Pedestrian improvement is reported above and does not override those rules. Stop the semantic architecture route; the paper primary model returns to Stage6A R2. R2 is not executed in this stage.'
    conclusion+='\n\nSTOP. Stage7E, Stage7B and R2 remain unexecuted. Await review; no merge to main.'
    section('Scientific Decision',conclusion)
    (ROOT/'09_reports/stage7d_final_report.md').write_text('\n'.join(sections))
    print('FINAL_REPORT_AND_INTEGRITY_PASS',decision,flush=True)

if __name__=='__main__':main()
