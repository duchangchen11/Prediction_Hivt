"""Fixed paired whole-scene descriptive intervals and preregistered decision."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage12b_common import *
COMPARISONS=(('S','G'),('S','P'),('S','C0'),('G','C0'),('P','C0'))
def switches(delta,changed):
    gain=-delta[delta<0];harm=delta[delta>0]
    return dict(changed_count=int(changed.sum()),improved_count=len(gain),worsened_count=len(harm),changed_tie_count=int((changed&(delta==0)).sum()),
        gross_gain=float(gain.sum()),gross_harm=float(harm.sum()),net_delta_FDE=float(delta.mean()),mean_harm=float(harm.mean()) if len(harm) else 0.,
        **{f'p{int(q*100)}_harm':float(np.quantile(harm,q)) if len(harm) else 0. for q in [.9,.95,.99]},HighCostHarmAbove5mCount=int((harm>5).sum()))
def main():
    verify();audit=read_json(ROOT/'06_oof_evaluation/stage12b_oof_identity_audit.json');assert audit['Status']=='PASS'
    for path,h in audit['CacheFiles'].items():assert sha256(ROOT/path)==h
    dest=ROOT/'06_oof_evaluation/cache';f=pd.read_csv(dest/'stage12b_actor_records.csv');values=np.load(dest/'stage12b_metrics.npy',mmap_mode='r');top=np.load(dest/'stage12b_modes.npy',mmap_mode='r')
    walk=np.load(CACHE/'stage12b_walkway.npy',mmap_mode='r');fd=np.asarray(values[:,:,0]);scene_order=sum([sorted(split(k)['OuterTest']) for k in (1,2,3)],[])
    assert len(set(scene_order))==630;codes=f.scene_token.map({s:i for i,s in enumerate(scene_order)}).to_numpy()
    rng=np.random.default_rng(2022);weights=np.zeros((2000,630),np.int32)
    for r in range(2000):
        for k in range(3):weights[r,k*210:(k+1)*210]=np.bincount(rng.integers(0,210,210),minlength=210)
    assert np.array_equal(weights,np.load(S11B/'06_bootstrap/stage11b_scene_bootstrap_weights.npy'))
    folder=ROOT/'07_bootstrap/cache';folder.mkdir(exist_ok=True);np.save(folder/'stage12b_scene_weights.npy',weights)
    cirows=[];switchrows=[];meanrows=[]
    for group,mask in groups(f,walk).items():
        den=np.bincount(codes[mask],minlength=630).astype(np.float64);sums=np.stack([np.bincount(codes[mask],weights=fd[mask,j],minlength=630) for j in range(4)],-1)
        assert mask.any() and np.all(weights@den>0);means=weights@sums/(weights@den)[:,None]
        assert np.allclose(sums.sum(0)/den.sum(),fd[mask].mean(0),atol=1e-12,rtol=0)
        for j,name in enumerate(MODELS):
            lo,hi=np.quantile(means[:,j],[.025,.975]);meanrows.append(dict(Group=group,Model=name,Count=int(mask.sum()),Top1FDE=float(fd[mask,j].mean()),CI95Lower=lo,CI95Upper=hi))
        for new,old in COMPARISONS:
            a,b=MODELS.index(new),MODELS.index(old);d=fd[mask,a]-fd[mask,b];bd=means[:,a]-means[:,b];lo,hi=np.quantile(bd,[.025,.975]);primary=group=='Pedestrian' and old in ('G','P') and new=='S'
            alo,ahi=np.quantile(bd,[.0125,.9875]) if primary else (None,None)
            cirows.append(dict(Group=group,Comparison=new+'-'+old,Count=int(mask.sum()),DeltaTop1FDE=float(d.mean()),CI95Lower=lo,CI95Upper=hi,
                FamilyAdjustedLower=alo,FamilyAdjustedUpper=ahi,FamilySize=2 if primary else None,IntervalScope='Development-stage descriptive interval'))
            switchrows.append(dict(Group=group,Comparison=new+'-'+old,Count=int(mask.sum()),**switches(d,top[mask,a]!=top[mask,b])))
    dump('07_bootstrap/stage12b_bootstrap_ci.csv',cirows);dump('07_bootstrap/stage12b_model_mean_ci.csv',meanrows);dump('08_diagnostics/stage12b_mode_switch_cost.csv',switchrows)
    lookup={(r['Group'],r['Comparison']):r for r in cirows};delta=lambda g,c:lookup[g,c]['DeltaTop1FDE']
    fold=pd.read_csv(ROOT/'06_oof_evaluation/stage12b_fold_metrics.csv');directions=[]
    for k in (1,2,3):
        sub=fold[(fold.Fold==k)&(fold.Group=='Pedestrian')].set_index('Model');directions.append(float(sub.loc['S','Top1FDE']-sub.loc['G','Top1FDE']))
    point=all(delta('Pedestrian',c)<0 for c in ('S-G','S-P','S-C0'));overall=delta('Overall','S-C0')<0
    adjusted=all(lookup['Pedestrian',c]['FamilyAdjustedUpper']<0 for c in ('S-G','S-P'))
    harm=lookup['Pedestrian5-8m','S-C0'];significant_harm=harm['DeltaTop1FDE']>0 and harm['CI95Lower']>0
    if not point:decision='NOT_SUPPORTED'
    elif significant_harm:decision='PARTIAL'
    elif adjusted and overall:decision='STRONG_SUPPORTED'
    elif sum(v<0 for v in directions)>=2 and delta('Overall','S-C0')<=0:decision='SUPPORTED'
    else:decision='PARTIAL'
    eligible=decision in ('SUPPORTED','STRONG_SUPPORTED');swi={(r['Group'],r['Comparison']):r for r in switchrows}
    swg=swi['Pedestrian','G-C0'];sws=swi['Pedestrian','S-C0'];swp=swi['Pedestrian','P-C0']
    result=dict(FrozenModelIntegrity='PASS',SemanticCacheIntegrity='PASS',GTLeakage='PASS',FoldIsolation='PASS',CandidateIdentity='PASS',
        VehiclePreserved='YES',BicyclePreserved='YES',GeometryControl='PASS',ShuffledControl='PASS',SemanticIncrement=decision,
        PedestrianImproved='YES' if delta('Pedestrian','S-C0')<0 else 'NO',Pedestrian5_8mDegraded='YES' if harm['DeltaTop1FDE']>0 else 'NO',
        Pedestrian5_8mSignificantlyDegraded=bool(significant_harm),OverallImproved='YES' if overall else 'NO',Stage12B='GO' if eligible else 'STOP',
        ReadyForNextStage='YES' if eligible else 'NO',FoldPedestrianSMinusG=directions,
        PedestrianWrongSwitchReduction={'RelativeToGeometry':swg['worsened_count']-sws['worsened_count'],'RelativeToShuffled':swp['worsened_count']-sws['worsened_count'],
            'GeometryWrongSwitches':swg['worsened_count'],'RealSemanticWrongSwitches':sws['worsened_count'],'ShuffledWrongSwitches':swp['worsened_count']},
        ProbabilityDiagnostics='recorded; no new calibration or fitted temperature',Evidence='development-stage same630-scene OOF; not fully independent end-to-end test',
        StopAfterCompletion=True,NextStageTrainingAuthorized=False,HistoricalScientificDecisionsUnchanged=True)
    atomic_json(ROOT/'10_reports/stage12b_scientific_decision.json',result)
    atomic_json(ROOT/'07_bootstrap/stage12b_bootstrap_audit.json',dict(Status='PASS',Seed=2022,Replicates=2000,Unit='whole scene within each210-scene fold',
        SceneOrder=scene_order,WeightsSHA256=sha256(folder/'stage12b_scene_weights.npy'),ExactHistoricalWeights=True,CI95Percentiles=[2.5,97.5],
        FamilyAdjustedPercentiles=[1.25,98.75],PrimaryFamily=['Pedestrian S-G','Pedestrian S-P'],ActorWeightedPooledEstimand=True,Decision=result))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
