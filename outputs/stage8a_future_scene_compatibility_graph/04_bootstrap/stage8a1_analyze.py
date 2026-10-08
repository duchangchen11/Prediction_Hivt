"""Preregistered paired whole-scene bootstrap and exact scientific gates."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage8a1_common import *
import pandas as pd
MODELS=('R0','R2','G1','G2','G3')
METRICS=('minADE6','minFDE6','MR6','Top1ADE','Top1FDE','OracleGap','HitRate','BestModeRank','MRR')
GENERAL=('Overall','Vehicle','Pedestrian','Bicycle','MovingVehicle','StoppedVehicle','ParkedVehicle','Vehicle>5m','Pedestrian<5m','Pedestrian>5m')
SEMANTIC=('IntersectionVehicle20','NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT','GT-left','GT-right')
MAP=('MapNonEmpty','ZeroMap','RouteCenterlineAvailable','PedSemanticSpecificAvailable')
PAIRS=(('G3','R2'),('G1','R0'),('G1','R2'),('G2','R0'),('G3','G1'),('G3','G2'))

def masks(f):
    v=f.agent_type=='Vehicle';p=f.agent_type=='Pedestrian';d=f.GT_endpoint_displacement_m
    return {'Overall':np.ones(len(f),dtype=bool),'Vehicle':v,'Pedestrian':p,'Bicycle':f.agent_type=='Bicycle',
        **{n:v&(f.motion_state==m) for n,m in zip(('MovingVehicle','StoppedVehicle','ParkedVehicle'),('vehicle.moving','vehicle.stopped','vehicle.parked'))},
        'Vehicle>5m':v&(d>5),'Pedestrian<5m':p&(d<5),'Pedestrian>5m':p&(d>5),
        **{g:f[g]==1 for g in (*SEMANTIC[:4],*MAP)},
        'GT-left':(f.TurningVehicle_GT==1)&(f.GT_heading_change_deg>0),'GT-right':(f.TurningVehicle_GT==1)&(f.GT_heading_change_deg<0)}

def main():
    assert read_json(ROOT/'03_evaluation/stage8a1_complete.json')['status']=='PASS'
    f=pd.read_csv(ROOT/'03_evaluation/stage8a1_actor_results.csv');assert len(f)==54990
    groups=masks(f);rows=[]
    for g,mask in groups.items():
        assert np.sum(mask)>0,g
        for v in MODELS:rows.append({'Group':g,'Model':v,'Count':int(np.sum(mask)),
            **{m:float(f.loc[mask,v+'_'+m].mean()) for m in METRICS},'Params':0 if v=='R0' else 673 if v=='R2' else PARAMS[v]})
    write_csv(ROOT/'06_tables/stage8a1_main_results.csv',[r for r in rows if r['Group']=='Overall'])
    for name,gs in (('group',GENERAL),('semantic_group',SEMANTIC),('map_availability',MAP)):
        write_csv(ROOT/'06_tables'/f'stage8a1_{name}_results.csv',[r for r in rows if r['Group'] in gs])
    scenes=sorted(f.scene_token.unique());assert len(scenes)==150
    si=pd.Categorical(f.scene_token,categories=scenes).codes
    resampling=np.random.default_rng(2022).integers(0,150,size=(1000,150))
    counts=np.array([np.bincount(row,minlength=150) for row in resampling],dtype=np.float64)
    atomic_npz(ROOT/'04_bootstrap/stage8a1_shared_resampling.npz',{'indices':resampling})
    cis=[]
    for g,mask in groups.items():
        mask=np.asarray(mask,dtype=bool);scene_counts=np.bincount(si[mask],minlength=150).astype(float);den=counts@scene_counts
        assert (den>0).all()
        for new,base in PAIRS:
            for metric in ('Top1FDE','Top1ADE','HitRate'):
                delta=(f[new+'_'+metric]-f[base+'_'+metric]).to_numpy();sums=np.bincount(si[mask],weights=delta[mask],minlength=150)
                draws=(counts@sums)/den;lo,hi=np.percentile(draws,[2.5,97.5])
                cis.append({'Group':g,'New':new,'Baseline':base,'Metric':metric,'Count':int(mask.sum()),
                    'Delta':float(delta[mask].mean()),'CI_lower':float(lo),'CI_upper':float(hi),'Replicates':1000,'Seed':2022})
    write_csv(ROOT/'06_tables/stage8a1_bootstrap_ci.csv',cis)
    lookup={(r['New'],r['Baseline'],r['Group'],r['Metric']):r for r in cis}
    def ci(new,base,g='Overall'):return lookup[new,base,g,'Top1FDE']
    def improves(n,b,g='Overall'):return ci(n,b,g)['Delta']<0 and ci(n,b,g)['CI_upper']<0
    def harm(n,b,g):return ci(n,b,g)['CI_lower']>0
    def safe(n,b):return not any(harm(n,b,g) for g in ('Vehicle','Pedestrian'))
    def semantic(n,b):
        a=ci(n,b);relative=a['Delta']/float(f[b+'_Top1FDE'].mean())
        return improves(n,b) or (relative<=.005 and a['CI_lower']<=0<=a['CI_upper'] and safe(n,b) and
            sum(improves(n,b,g) for g in ('NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT'))>=2)
    primary=ci('G3','R2');target=sum(improves('G3','R2',g) for g in ('MovingVehicle','Vehicle>5m','NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT'))
    fscg='SUPPORTED' if improves('G3','R2') and safe('G3','R2') else 'TARGETED_SUPPORTED' if primary['Delta']<0 and primary['CI_lower']<=0<=primary['CI_upper'] and target>=2 and safe('G3','R2') else 'NOT_SUPPORTED'
    sc=semantic('G3','G1')
    joint=sc and ((improves('G3','G1') and not harm('G3','G2','Overall')) or (improves('G3','G2') and not harm('G3','G1','Overall')))
    decision={'AgentGraph':'SUPPORTED' if improves('G1','R0') else 'NOT_SUPPORTED',
        'LearnedGraphAdvantageOverR2':'YES' if improves('G1','R2') else 'NO',
        'SemanticGraph':'SUPPORTED' if semantic('G2','R0') else 'NOT_SUPPORTED',
        'SemanticContributionToJoint':'SUPPORTED' if sc else 'NOT_SUPPORTED','FSCG':fscg,
        'JointComplementarity':'YES' if joint else 'NO','PaperUsableFSCG':'YES' if fscg!='NOT_SUPPORTED' else 'NO',
        'RecommendedFinalModel':'G3' if fscg!='NOT_SUPPORTED' else 'G1' if improves('G1','R2') and safe('G1','R2') else 'R2'}
    atomic_json(ROOT/'09_reports/stage8a1_scientific_decision.json',decision)
    changed=f.R2_top1_mode!=f.G3_top1_mode;delta=f.G3_Top1FDE-f.R2_Top1FDE
    improved=changed&(delta<0);worsened=changed&(delta>0);unchanged=changed&(delta==0)
    mode=[{'Analysis':'R2→G3','Targets':len(f),'ChangedCount':int(changed.sum()),'ChangedRate':float(changed.mean()),
        'ChangedImproved':int(improved.sum()),'ChangedWorsened':int(worsened.sum()),'ChangedErrorUnchanged':int(unchanged.sum()),
        'MeanGainImproved':float(-delta[improved].mean()),'MeanHarmWorsened':float(delta[worsened].mean()),
        'NetFDEDeltaOverall':float(delta.mean()),'AgreementRate':float((~changed).mean())}]
    for a,b in (('G1','G2'),('G1','G3'),('G2','G3')):
        mode.append({**dict.fromkeys(mode[0],''),'Analysis':a+'↔'+b,'Targets':len(f),'AgreementRate':float((f[a+'_top1_mode']==f[b+'_top1_mode']).mean())})
    write_csv(ROOT/'06_tables/stage8a1_mode_change_analysis.csv',mode)
    atomic_json(ROOT/'04_bootstrap/stage8a1_bootstrap_audit.json',{'status':'PASS','scenes':150,'replicates':1000,'seed':2022,
        'shared_indices_sha256':sha256(ROOT/'04_bootstrap/stage8a1_shared_resampling.npz'),'pairs':PAIRS,'delta':'new-baseline',
        'actor_weighted_within_resampled_whole_scenes':True,'subgroup_intervals':'unadjusted, overlapping descriptive secondary groups'})
    print('SCIENTIFIC_DECISION',decision,flush=True)

if __name__=='__main__':main()
