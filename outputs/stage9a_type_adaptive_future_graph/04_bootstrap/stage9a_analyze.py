"""Frozen secondary evidence rules; paired scenes, no variant search."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage9a_common import *
import pandas as pd
MODELS=('R0','R2','G1','G3','T1','T2','T3')
METRICS=('minADE6','minFDE6','MR6','Top1ADE','Top1FDE','OracleGap','HitRate','BestModeRank','MRR')
GENERAL=('Overall','Vehicle','Pedestrian','Bicycle','MovingVehicle','StoppedVehicle','ParkedVehicle','Vehicle>5m','Pedestrian<5m','Pedestrian>5m')
CONTEXT=('Neighbor0','Neighbor1-2','Neighbor3-5','Neighbor6-8','0-5m','5-10m','10-20m','20-50m')

def masks(f):
    v=f.agent_type=='Vehicle';p=f.agent_type=='Pedestrian';d=f.GT_endpoint_displacement_m
    return {'Overall':np.ones(len(f),bool),'Vehicle':v,'Pedestrian':p,'Bicycle':f.agent_type=='Bicycle',
        **{g:v&(f.motion_state==m) for g,m in zip(('MovingVehicle','StoppedVehicle','ParkedVehicle'),('vehicle.moving','vehicle.stopped','vehicle.parked'))},
        'Vehicle>5m':v&(d>5),'Pedestrian<5m':p&(d<5),'Pedestrian>5m':p&(d>5),
        **{g:f.NeighborCountGroup==g for g in CONTEXT[:4]},**{g:f.NearestDistanceGroup==g for g in CONTEXT[4:]}}

def main():
    assert read_json(ROOT/'03_evaluation/stage9a_complete.json')['status']=='PASS'
    protocol=read_json(ROOT/'00_manifest/stage9a_protocol.json');assert sha256(ROOT/'00_manifest/stage9a_protocol.json')==read_json(REG)['protocol_sha256']
    f=pd.read_csv(ROOT/'03_evaluation/stage9a_actor_results.csv');assert len(f)==54990
    groups=masks(f);params={'R0':0,'R2':673,'G1':24066,'G3':37187,**PARAMS};rows=[]
    for g,mask in groups.items():
        assert int(np.sum(mask))>0,g
        for v in MODELS:rows.append({'Group':g,'Model':v,'Count':int(np.sum(mask)),**{m:float(f.loc[mask,v+'_'+m].mean()) for m in METRICS},'Params':params[v]})
    for name,gs in (('main',('Overall',)),('type',GENERAL[:4]),('motion',GENERAL[4:]),('interaction_context',CONTEXT)):
        write_csv(ROOT/'06_tables'/f'stage9a_{name}_results.csv',[r for r in rows if r['Group'] in gs])
    scenes=sorted(f.scene_token.unique());assert len(scenes)==150;si=pd.Categorical(f.scene_token,categories=scenes).codes
    indices=np.random.default_rng(2022).integers(0,150,size=(1000,150));counts=np.array([np.bincount(x,minlength=150) for x in indices],dtype=np.float64)
    np.savez_compressed(ROOT/'04_bootstrap/stage9a_shared_resampling.npz',indices=indices)
    ci=[]
    for g,mask in groups.items():
        mask=np.asarray(mask,dtype=bool);scene_counts=np.bincount(si[mask],minlength=150);den=counts@scene_counts;assert (den>0).all()
        for new,base in protocol['bootstrap']['pairs']:
            for metric in ('Top1FDE','Top1ADE','HitRate'):
                delta=(f[new+'_'+metric]-f[base+'_'+metric]).to_numpy();sums=np.bincount(si[mask],weights=delta[mask],minlength=150)
                draws=(counts@sums)/den;lo,hi=np.percentile(draws,[2.5,97.5])
                ci.append({'Group':g,'New':new,'Baseline':base,'Metric':metric,'Count':int(mask.sum()),'Delta':float(delta[mask].mean()),'CI_lower':float(lo),'CI_upper':float(hi),'Replicates':1000,'Seed':2022})
    write_csv(ROOT/'06_tables/stage9a_bootstrap_ci.csv',ci);lookup={(r['New'],r['Baseline'],r['Group'],r['Metric']):r for r in ci}
    def result(n,b,g='Overall'):return lookup[n,b,g,'Top1FDE']
    def improves(n,b,g='Overall'):r=result(n,b,g);return r['Delta']<0 and r['CI_upper']<0
    def harm(n,b,g):return result(n,b,g)['CI_lower']>0
    def safe(n,b,gs):return not any(harm(n,b,g) for g in gs)
    p21=result('T2','T1','Pedestrian');p32=result('T3','T2','Pedestrian');ped=result('T3','R2','Pedestrian')
    adaptation='SUPPORTED' if improves('T2','T1','Pedestrian') and safe('T2','T1',('Overall','Vehicle')) else 'WEAK_SUPPORTED' if p21['Delta']<0 and p21['CI_lower']<=0<=p21['CI_upper'] and safe('T2','T1',('Overall','Vehicle')) else 'NOT_SUPPORTED'
    balanced=p32['Delta']<=0 and safe('T3','T2',('Overall','Vehicle'))
    taf=improves('T3','R2') and safe('T3','R2',('Vehicle','Pedestrian','Bicycle')) and ped['Delta']<=.01
    vr2=float(f.loc[groups['Vehicle'],'R2_Top1FDE'].mean());vg1=float(f.loc[groups['Vehicle'],'G1_Top1FDE'].mean());vt3=float(f.loc[groups['Vehicle'],'T3_Top1FDE'].mean())
    retention=(vr2-vt3)/(vr2-vg1) if vr2-vg1>0 else None
    pr2=float(f.loc[groups['Pedestrian'],'R2_Top1FDE'].mean());pg1=float(f.loc[groups['Pedestrian'],'G1_Top1FDE'].mean());pt3=float(f.loc[groups['Pedestrian'],'T3_Top1FDE'].mean())
    recovery=1-(pt3-pr2)/(pg1-pr2) if pg1-pr2>0 else None
    retained=retention is not None and retention>=.7;resolved=ped['Delta']<=.01 and not harm('T3','R2','Pedestrian')
    paper=taf and retained and resolved;strong=taf and improves('T3','R2','Vehicle') and ped['Delta']<=0 and not harm('T3','R2','Pedestrian')
    decision={'ResidualGraph':'SUPPORTED' if improves('T1','R2') else 'NOT_SUPPORTED','TypeAdaptation':adaptation,'BalancedTraining':'SUPPORTED' if balanced else 'NOT_SUPPORTED','TAFIG':'SUPPORTED' if taf else 'NOT_SUPPORTED','TAFIG_STRONG':'YES' if strong else 'NO','VehicleBenefitRetained':'YES' if retained else 'NO','PedestrianNegativeTransferResolved':'YES' if resolved else 'NO','PaperUsableTAFIG':'YES' if paper else 'NO','RecommendedFinalModel':'T3' if paper else 'R2','ReadyStage9BConfirmatoryCV':'YES' if taf else 'NO'}
    atomic_json(ROOT/'09_reports/stage9a_scientific_decision.json',decision)
    atomic_json(ROOT/'05_diagnostics/stage9a_retention_recovery.json',{'VehicleRetentionRatio':retention,'VehicleRetentionTarget':.7,'Vehicle_G1_benefit_m':vr2-vg1,'Vehicle_T3_benefit_m':vr2-vt3,'PedestrianRecoveryRatio':recovery,'PedestrianRecoveryTarget_descriptive':.8,'G1PedDegradation':pg1-pr2,'T3PedDegradation':pt3-pr2,'STRONG_PLUS':bool(strong and improves('T3','R2','Pedestrian')),'Official_VAL_role':'development-reuse evidence'})
    gates=[]
    for g in GENERAL:
        for v in VARIANTS:
            x=f.loc[groups[g],v+'_gate'].to_numpy();assert np.isfinite(x).all() and ((x>=0)&(x<=1)).all()
            gates.append({'Group':g,'Model':v,'Count':len(x),'MeanGate':float(x.mean()),'MedianGate':float(np.median(x)),'P10Gate':float(np.percentile(x,10)),'P90Gate':float(np.percentile(x,90)),'FixedGate':int(v=='T1')})
    write_csv(ROOT/'06_tables/stage9a_gate_statistics.csv',gates)
    changes=[]
    for g in GENERAL[:4]:
        mask=groups[g];sub=f[mask];d=sub.T3_Top1FDE-sub.R2_Top1FDE;changed=sub.T3_top1_mode!=sub.R2_top1_mode;good=changed&(d<0);bad=changed&(d>0)
        changes.append({'Group':g,'Count':len(sub),'ChangedCount':int(changed.sum()),'ChangedRate':float(changed.mean()),'ImprovedCount':int(good.sum()),'WorsenedCount':int(bad.sum()),'ChangedErrorUnchanged':int((changed&(d==0)).sum()),'UnchangedCount':int((~changed).sum()),'MeanGain':float(-d[good].mean()) if good.any() else 0.,'MeanHarm':float(d[bad].mean()) if bad.any() else 0.,'NetTop1FDEDelta':float(d.mean())})
    write_csv(ROOT/'06_tables/stage9a_mode_change.csv',changes)
    atomic_json(ROOT/'04_bootstrap/stage9a_bootstrap_audit.json',{'status':'PASS','scenes':150,'replicates':1000,'seed':2022,'shared_indices_sha256':sha256(ROOT/'04_bootstrap/stage9a_shared_resampling.npz'),'pairs':protocol['bootstrap']['pairs'],'groups':list(groups),'delta':'new-baseline','Official_VAL_role':'development-reuse evidence','subgroup_intervals':'unadjusted overlapping secondary analyses; not independent confirmatory evidence'})
    print('STAGE9_SCIENTIFIC_DECISION',decision,'RETENTION',retention,'RECOVERY',recovery,flush=True)

if __name__=='__main__':main()
