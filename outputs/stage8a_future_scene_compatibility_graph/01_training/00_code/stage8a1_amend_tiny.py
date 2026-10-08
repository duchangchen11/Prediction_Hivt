"""Rejudge saved tiny outputs only; no forward, optimizer, sampling or updates."""
from stage8a1_common import *
import datetime,subprocess,math
SOURCE='2a79b0d687192002537073147d93658bf4c6423e'
AMEND=ROOT/'00_manifest/stage8a1_tiny_gate_amendment.json'

def main():
    verify();assert not FROZEN.exists();assert not (ROOT/'03_evaluation/stage8a1_val_registration.json').exists()
    historical=read_json(ROOT/'01_training/stage8a1_stop_gate.json')
    assert historical['status']=='STOP_BEFORE_FORMAL_TRAINING' and historical['full_training_updates']==0
    feasibility=read_json(ROOT/'01_training/stage8a1_tiny_feasibility.json');h=feasibility['soft_target_entropy_lower_bound'];rows=[];evidence={}
    for v in PARAMS:
        p=ROOT/'01_training'/v/'stage8a1_tiny_audit.json';g=ROOT/'01_training'/v/'gradient_audit.json'
        tiny=read_json(p);grad=read_json(g);assert tiny['status']=='FAIL' and tiny['updates']==300 and tiny['precision']=='FP32'
        initial=tiny['initial_loss'];final=tiny['final_loss'];initial_excess=initial-h;final_excess=final-h
        assert initial_excess>0 and final_excess>=0
        reduction=1-final_excess/initial_excess
        numeric=all(math.isfinite(x) for x in (initial,final,tiny['delta_abs_max'],grad['graph_gradient_norm']))
        conditions={'ExcessLossReduction_ge_0_90':reduction>=.90,'delta_finite_and_nonzero':numeric and tiny['delta_abs_max']>0,
            'graph_gradients_finite_nonzero':grad['graph_finite'] and grad['graph_nonzero'] and grad['graph_gradient_norm']>0,
            'predictor_gradient_count_zero':tiny['predictor_gradient_count']==grad['predictor_gradient_count']==0,
            'candidate_requires_grad_false':tiny['candidate_requires_grad']==grad['candidate_requires_grad']==False,'no_NaN_Inf':numeric}
        rows.append({'Variant':v,'HistoricalStatus':'FAIL','HistoricalFailureSource':'INFEASIBLE_TINY_CRITERION',
            'H':h,'InitialLoss':initial,'FinalLoss':final,'InitialExcessLoss':initial_excess,'FinalExcessLoss':final_excess,
            'ExcessLossReduction':reduction,'Threshold':.90,'AmendedStatus':'PASS' if all(conditions.values()) else 'FAIL','conditions':conditions})
        evidence[str(p.relative_to(PROJECT))]=sha256(p);evidence[str(g.relative_to(PROJECT))]=sha256(g)
    for p in (ROOT/'01_training/stage8a1_tiny_targets.json',ROOT/'01_training/stage8a1_tiny_feasibility.json',ROOT/'01_training/stage8a1_stop_gate.json'):
        evidence[str(p.relative_to(PROJECT))]=sha256(p)
    passed=all(r['AmendedStatus']=='PASS' for r in rows)
    record={'protocol_revision':'Stage8A-1 tiny gate amendment only','registered_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'historical_commit':SOURCE,'historical_first_attempt':'STOP_TINY_GATE_FAILURE','failure_source':'INFEASIBLE_TINY_CRITERION','classification':'PROTOCOL_BUG',
        'old_gate':'FinalLoss < 0.8 * InitialLoss','proof':'L(q,p)=H(q)+KL(q||p)>=H(q); old required loss below target entropy',
        'H':h,'old_required_final_loss_below':feasibility['required_final_loss_strictly_below'],
        'new_formula':'ExcessLoss=L-H(q); ExcessLossReduction=1-(FinalLoss-H)/(InitialLoss-H)','threshold':.90,
        'results':rows,'TinyGateAmended':'YES','TinyGate':'PASS' if passed else 'FAIL','ReadyFormalTraining':'YES' if passed else 'NO',
        'amendment_before_formal_training':True,'amendment_before_first_official_VAL':True,'tiny_repeated':False,'additional_tiny_updates':0,
        'sample_seed_updates_optimizer_LR_unchanged':True,'graph_and_formal_training_protocol_unchanged':True,'historical_evidence_sha256':evidence,
        'historical_registration_sha256':sha256(ROOT/'00_manifest/stage8a1_registration.json'),'new_audit_source_sha256':sha256(Path(__file__))}
    assert not AMEND.exists(),'Amendment already recorded; never rewrite history'
    atomic_json(AMEND,record)
    write_csv(ROOT/'06_tables/stage8a1_amended_tiny_gate_results.csv',[{k:x for k,x in r.items() if k!='conditions'} for r in rows])
    (ROOT/'00_manifest/stage8a1_tiny_gate_amendment.md').write_text(
        '# Stage8A-1 tiny gate protocol amendment\n\n'
        f'Historical commit `{SOURCE}` retains `STOP_TINY_GATE_FAILURE` and the original three FAIL outcomes. '
        'The failure source is `INFEASIBLE_TINY_CRITERION`, classified as `PROTOCOL_BUG`.\n\n'
        f'Original criterion: `FinalLoss < 0.8 * InitialLoss`. On the unchanged128 targets, '
        f'`H(q)={h:.15f}` and the old upper bound is `{feasibility["required_final_loss_strictly_below"]:.15f}`. '
        'Because `SoftCE(q,p)=H(q)+KL(q||p)>=H(q)`, that bound is unattainable.\n\n'
        'Amended engineering criterion: `ExcessLoss=L-H(q)`, '
        '`ExcessLossReduction=1-(FinalLoss-H)/(InitialLoss-H) >=0.90`. '
        'Finite/nonzero residuals and graph gradients, frozen predictor gradients0, candidate.requires_grad=False and no NaN/Inf remain required.\n\n'
        'Only saved seed2022 fixed128/300-update FP32 results are rejudged. No resampling, new tiny update or hyperparameter change. '
        'This amendment is recorded before any formal training or officialVAL.\n\n'
        '|Variant|Initial excess|Final excess|Reduction|Amended gate|\n|---|---:|---:|---:|---|\n'+
        ''.join(f"|{r['Variant']}|{r['InitialExcessLoss']:.12f}|{r['FinalExcessLoss']:.12f}|{100*r['ExcessLossReduction']:.6f}%|{r['AmendedStatus']}|\n" for r in rows)+
        '\nFormal outputs use `01_training/G{1,2,3}/formal/` to preserve first-attempt config/gradient/tiny records. '
        'The final resumed report has a new filename; the first STOP report is immutable.\n')
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=PROJECT).decode().split('\0')
    atomic_json(ROOT/'00_manifest/stage8a1_resume_frozen_history.json',{'source_commit':SOURCE,
        'files':{p:sha256(PROJECT/p) for p in tracked if p},'historical_first_attempt':'STOP_TINY_GATE_FAILURE',
        'all_original_tiny_outputs_preserved':True})
    print('AMENDED_TINY',[(r['Variant'],r['ExcessLossReduction'],r['AmendedStatus']) for r in rows],flush=True)
    assert passed,'STOP: amended tiny gate FAIL'

if __name__=='__main__':main()
