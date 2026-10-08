"""Frozen point-estimate feasibility criteria; descriptive paired HeadDev scene bootstrap."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage10a_common import *
import pandas as pd
MODELS=('R0','R2','G1','DualExpert')
METRICS=('minADE6','minADEOracle6','minFDE6','MR6','Top1ADE','Top1FDE','OracleGap','HitRate','BestModeRank','MRR','SoftCE')

def masks(f):
    v=f.agent_type=='Vehicle'; p=f.agent_type=='Pedestrian'; d=f.GT_endpoint_displacement_m
    return {'Overall':np.ones(len(f),bool),'Vehicle':v,'Pedestrian':p,'Bicycle':f.agent_type=='Bicycle',
        **{g:v&(f.motion_state==m) for g,m in zip(('MovingVehicle','StoppedVehicle','ParkedVehicle'),('vehicle.moving','vehicle.stopped','vehicle.parked'))},
        'Vehicle>5m':v&(d>5),'Pedestrian<5m':p&(d<5),'Pedestrian>5m':p&(d>5)}

def main():
    verify(); complete=read_json(ROOT/'03_evaluation/stage10a_complete.json'); assert complete['status']=='PASS'
    assert sha256(ROOT/'03_evaluation/stage10a_actor_results.csv')==complete['actor_csv_sha256']
    f=pd.read_csv(ROOT/'03_evaluation/stage10a_actor_results.csv'); assert len(f)==29934
    groups=masks(f); spec=read_json(PROTOCOL); assert list(groups)==spec['evaluation']['groups']
    rows=[]; params={'R0':0,'R2':673,'G1':24066,'DualExpert':48805}
    for g,mask in groups.items():
        assert int(np.sum(mask))>0,g
        for m in MODELS:
            rows.append({'Group':g,'Model':m,'Count':int(np.sum(mask)),**{x:float(f.loc[mask,m+'_'+x].mean()) for x in METRICS},
                'RerankerParameters':params[m],'Stage10TrainableExpertParameters':48132 if m=='DualExpert' else 0,
                'FrozenR2Parameters':673 if m in ('R2','DualExpert') else 0,
                'EvidenceRole':'HeadDev feasibility screening'})
    for name,gs in (('main',('Overall',)),('type',('Overall','Vehicle','Pedestrian','Bicycle')),('motion',list(groups)[4:])):
        write_csv(ROOT/'06_tables'/f'stage10a_{name}_results.csv',[r for r in rows if r['Group'] in gs])
    scenes=sorted(read_json(SPLIT)['HeadDev']); assert set(f.scene_token)==set(scenes) and len(scenes)==70
    si=pd.Categorical(f.scene_token,categories=scenes).codes
    indices=np.random.default_rng(2022).integers(0,70,size=(1000,70))
    weights=np.array([np.bincount(x,minlength=70) for x in indices],dtype=np.float64)
    np.savez_compressed(ROOT/'04_bootstrap/stage10a_shared_resampling.npz',scene_tokens=np.array(scenes),indices=indices)
    ci=[]
    for g,mask in groups.items():
        mask=np.asarray(mask,dtype=bool); count=np.bincount(si[mask],minlength=70); den=weights@count
        assert (den>0).all(), 'Empty bootstrap population; requires explicit reporting'
        for metric in spec['bootstrap']['metrics']:
            delta=(f['DualExpert_'+metric]-f['R2_'+metric]).to_numpy()
            sums=np.bincount(si[mask],weights=delta[mask],minlength=70)
            draws=(weights@sums)/den; lo,hi=np.percentile(draws,[2.5,97.5])
            ci.append({'Group':g,'New':'DualExpert','Baseline':'R2','Metric':metric,'Count':int(mask.sum()),
                'Delta':float(delta[mask].mean()),'CI_lower':float(lo),'CI_upper':float(hi),'Replicates':1000,'Seed':2022,
                'EvidenceRole':'HeadDev checkpoint-selected feasibility screening; not independent confirmation'})
    write_csv(ROOT/'06_tables/stage10a_bootstrap_ci.csv',ci)
    lookup={r['Group']:r for r in ci if r['Metric']=='Top1FDE'}
    candidate=read_json(ROOT/'03_evaluation/stage10a_candidate_identity.json')
    engineering=read_json(ROOT/'01_training/stage10a_engineering_audit.json')['status']=='PASS'
    engineering=engineering and read_json(ROOT/'01_training/stage10a_tiny_gate.json')['status']=='PASS' and candidate['status']=='PASS'
    bike=candidate['Bicycle_logits_bitwise_R2'] and candidate['Bicycle_probability_bitwise_R2'] and candidate['Bicycle_candidates_bitwise_R2']
    dv,dp,db,do=(lookup[g]['Delta'] for g in ('Vehicle','Pedestrian','Bicycle','Overall'))
    assert db==0. and bike
    go=engineering and bike and dv<=-.03 and dp<=0 and do<0
    conditional=engineering and bike and dv<0 and dp<=.01 and do<0
    result='GO' if go else 'CONDITIONAL_GO' if conditional else 'STOP'
    mitigation='YES' if engineering and bike and dv<0 and dp<=0 else 'PARTIAL' if engineering and bike and dv<0 and dp<=.01 else 'NO'
    decision={'VehicleExpert':'PROMISING' if engineering and dv<=-.03 else 'NOT_PROMISING',
        'PedestrianExpert':'PROMISING' if engineering and dp<=0 else 'NOT_PROMISING','BicyclePreserved':'YES' if bike else 'NO',
        'OverallImproved':'YES' if do<0 else 'NO','NegativeTransferMitigated':mitigation,'Stage10A':result,
        'ReadyStage10B':'YES' if result!='STOP' else 'NO'}
    atomic_json(ROOT/'09_reports/stage10a_scientific_decision.json',decision)
    changes=[]
    for g in ('Overall','Vehicle','Pedestrian','Bicycle'):
        sub=f[groups[g]]; delta=sub.DualExpert_Top1FDE-sub.R2_Top1FDE
        changed=sub.DualExpert_top1_mode!=sub.R2_top1_mode
        changes.append({'Group':g,'Count':len(sub),'ChangedCount':int(changed.sum()),'ChangedRate':float(changed.mean()),
            'ImprovedCount':int((changed&(delta<0)).sum()),'WorsenedCount':int((changed&(delta>0)).sum()),
            'UnchangedCount':int((~changed).sum()),'DeltaTop1FDE':float(delta.mean())})
    write_csv(ROOT/'06_tables/stage10a_mode_change.csv',changes)
    atomic_json(ROOT/'04_bootstrap/stage10a_bootstrap_audit.json',{'status':'PASS','scenes':70,'replicates':1000,'seed':2022,
        'paired_whole_scene':True,'actor_weighted':True,'groups':list(groups),'metrics':spec['bootstrap']['metrics'],
        'shared_resampling_sha256':sha256(ROOT/'04_bootstrap/stage10a_shared_resampling.npz'),
        'bootstrap_rows':len(ci),'checkpoint_selected_HeadDev':True,'independent_confirmation':False,'official_VAL_opened':False})
    # Validate checkpoint-selected own-type metrics against the frozen training records.
    comparisons=[]
    for e,g in zip(EXPERTS,('Vehicle','Pedestrian')):
        s=read_json(ROOT/'01_training'/e/'formal/stage10a_training_summary.json')['selected']
        r=next(r for r in rows if r['Group']==g and r['Model']=='DualExpert')
        for old,new in (('DevSoftCE','SoftCE'),('DevTop1ADE','Top1ADE'),('DevTop1FDE','Top1FDE')):
            diff=abs(s[old]-r[new]); assert diff<1e-6,(e,new,diff)
            comparisons.append({'Expert':e,'Metric':new,'FrozenSelectedValue':s[old],'UnifiedValue':r[new],'AbsDiff':diff})
    g1score=next(r['SoftCE'] for r in rows if r['Group']=='Overall' and r['Model']=='G1')
    historical=read_json(S8/'02_checkpoints/stage8a1_checkpoint_manifest.json')['checkpoints'][0]['HeadDevLoss']
    assert abs(g1score-historical)<1e-6
    atomic_json(ROOT/'03_evaluation/stage10a_reference_reproduction.json',{'status':'PASS',
        'expert_selected_metric_checks':comparisons,'G1_HeadDevSoftCE':g1score,'historical_G1_HeadDevSoftCE':historical,
        'G1_absdiff':abs(g1score-historical),'R2_frozen_sha256':R2SHA,'predictor_frozen_sha256':PREDICTOR_SHA})
    print('STAGE10_HEADDEV_DECISION',decision,'DELTAS',{'Vehicle':dv,'Pedestrian':dp,'Bicycle':db,'Overall':do},flush=True)

if __name__=='__main__': main()
