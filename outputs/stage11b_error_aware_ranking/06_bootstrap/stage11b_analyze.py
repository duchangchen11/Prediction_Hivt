"""Registered comparisons: within-fold paired scene bootstrap and switch costs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11b_common import *

COMPARISONS=(('A','R2'),('B','R2'),('C','R2'),('B','A'),('C','A'),('C','B'))

def switch_statistics(delta,changed):
    gain=-delta[delta<0];harm=delta[delta>0];tail_count=int(np.ceil(.1*len(harm)))
    tail=float(np.sort(harm)[-tail_count:].sum()) if tail_count else 0.
    return {'ChangedCount':int(changed.sum()),'ImprovedCount':len(gain),'WorsenedCount':len(harm),
        'NeutralCount':int((delta==0).sum()),'ChangedNeutralCount':int((changed&(delta==0)).sum()),
        'MeanGain':float(gain.mean()) if len(gain) else 0.,'MeanHarm':float(harm.mean()) if len(harm) else 0.,
        'GrossGain':float(gain.sum()),'GrossHarm':float(harm.sum()),'NetFDEDelta':float(delta.mean()),
        **{f'P{int(q*100)}Harm':float(np.quantile(harm,q)) if len(harm) else 0. for q in (.9,.95,.99)},
        'Top10HarmCount':tail_count,'Top10HarmSum':tail,'Top10HarmShare':tail/float(harm.sum()) if len(harm) else 0.}

def main():
    verify(history=True);audit=read_json(ROOT/'05_oof_evaluation/stage11b_oof_identity_audit.json');assert audit['Status']=='PASS'
    dest=ROOT/'05_oof_evaluation/cache'
    for name,h in audit['cache_files'].items():assert sha256(dest/name)==h
    f=pd.read_csv(dest/'stage11b_oof_predictions_actor_records.csv',dtype={'future_mask_bits':str});n=len(f)
    values=np.load(dest/'stage11b_oof_metrics.npy',mmap_mode='r');prob=np.load(dest/'stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    assert audit['models']==list(MODELS) and audit['fields'][0]=='Top1FDE' and n==260151
    fd=np.asarray(values[:,:,0]);selected=prob.argmax(-1);masks=groups(f)
    scene_order=[]
    for fold in (1,2,3):
        scenes=sorted(split(fold)['OuterTest']);assert len(scenes)==210
        assert set(f.loc[f.Fold==fold,'scene_token'])==set(scenes);scene_order.extend(scenes)
    assert len(set(scene_order))==630
    scene_index={s:i for i,s in enumerate(scene_order)};codes=f.scene_token.map(scene_index).to_numpy()
    # Each replicate independently draws 210 paired whole scenes in each fold.
    rng=np.random.default_rng(2022);weights=np.zeros((2000,630),np.int32)
    for b in range(2000):
        for fold in range(3):weights[b,fold*210:(fold+1)*210]=np.bincount(rng.integers(0,210,210),minlength=210)
    assert np.all(weights.reshape(2000,3,210).sum(-1)==210)
    np.save(ROOT/'06_bootstrap/stage11b_scene_bootstrap_weights.npy',weights)
    bootrows=[];switchrows=[];probrows=[];meanrows=[]
    for group,keep in masks.items():
        count=int(keep.sum());assert count>0
        den=np.bincount(codes[keep],minlength=630).astype(np.float64);bootden=weights@den;assert (bootden>0).all()
        sums=np.stack([np.bincount(codes[keep],weights=fd[keep,j],minlength=630) for j in range(5)],-1)
        means=weights@sums/bootden[:,None]
        assert np.allclose(sums.sum(0)/den.sum(),fd[keep].mean(0),atol=1e-12,rtol=0.)
        for j,name in enumerate(MODELS):
            ci=np.quantile(means[:,j],[.025,.975]);meanrows.append({'Group':group,'Model':name,'Count':count,
                'Top1FDE':float(fd[keep,j].mean()),'CI95Lower':ci[0],'CI95Upper':ci[1]})
            maximum=np.asarray(values[keep,j,9]);entropy=np.asarray(values[keep,j,8])
            probrows.append({'Group':group,'Model':name,'Count':count,'PredictionEntropy':float(entropy.mean()),
                'Top1Probability':float(maximum.mean()),'Top1Top2Margin':float(values[keep,j,10].mean()),
                'OracleModeProbability':float(values[keep,j,11].mean()),'SoftCE':float(values[keep,j,5].mean()),
                'EntropyMedian':float(np.median(entropy)),'Top1ProbabilityMedian':float(np.median(maximum)),
                'Top1ProbabilityP90':float(np.quantile(maximum,.9)),'ProbabilityAbove0p9Fraction':float((maximum>.9).mean()),
                'ProbabilityAbove0p99Fraction':float((maximum>.99).mean()),
                'ExpectedRegret':float(values[keep,j,6].mean()),'NormalizedExpectedRegret':float(values[keep,j,7].mean())})
        for new,old in COMPARISONS:
            a,b=MODELS.index(new),MODELS.index(old);delta=fd[keep,a]-fd[keep,b];bootstrap_delta=means[:,a]-means[:,b]
            ci=np.quantile(bootstrap_delta,[.025,.975]);primary=(new,old)==('C','A') and group in TYPES[:2]
            adjusted=np.quantile(bootstrap_delta,[.0125,.9875]) if primary else (None,None)
            row={'Comparison':f'{new}-{old}','Group':group,'Count':count,'DeltaTop1FDE':float(delta.mean()),
                'CI95Lower':float(ci[0]),'CI95Upper':float(ci[1]),'CoPrimary':primary,
                'BonferroniFamilySize':2 if primary else None,'AdjustedIndividualCoverage':.975 if primary else None,
                'BonferroniCILower':adjusted[0],'BonferroniCIUpper':adjusted[1],
                'Interpretation':'co-primary' if primary else ('exploratory' if group not in ('Overall',*TYPES) else 'secondary/direction guard')}
            bootrows.append(row);switchrows.append({'Comparison':f'{new}-{old}','Group':group,'Count':count,
                **switch_statistics(delta,selected[keep,a]!=selected[keep,b])})
    dump('06_bootstrap/stage11b_bootstrap_ci.csv',bootrows)
    dump('06_bootstrap/stage11b_model_mean_ci.csv',meanrows)
    dump('07_diagnostics/stage11b_mode_switch_cost.csv',switchrows)
    dump('07_diagnostics/stage11b_probability_statistics.csv',probrows)
    boot={(r['Comparison'],r['Group']):r for r in bootrows};switch={(r['Comparison'],r['Group']):r for r in switchrows}
    def delta(comp,group):return boot[comp,group]['DeltaTop1FDE']
    bike=audit['BicyclePreserved']=='YES';types=('Vehicle','Pedestrian');both=all(delta('C-A',g)<0 for g in types)
    r2guard=all(delta('C-R2',g)<0 for g in types);overall=delta('C-A','Overall')<0
    adjusted=all(boot['C-A',g]['BonferroniCIUpper']<0 for g in types)
    no_wholly_harmful=all(boot['C-A',g]['CI95Lower']<=0 for g in types)
    if both and r2guard and overall and adjusted and bike:decision='STRONG_SUPPORTED'
    elif both and r2guard and overall and no_wholly_harmful and bike:decision='SUPPORTED'
    elif (delta('C-A','Vehicle')<0 and delta('C-A','Pedestrian')>0) or (delta('C-A','Pedestrian')<0 and delta('C-A','Vehicle')>0):decision='PARTIAL'
    else:decision='NOT_SUPPORTED'
    hard=all(delta(c,g)<0 for c in ('B-A','B-R2') for g in ('Overall',*types)) and all(boot['B-A',g]['CI95Lower']<=0 for g in types) and bike
    va,vc=switch['A-R2','Vehicle'],switch['C-R2','Vehicle'];pa,pc=switch['A-R2','Pedestrian'],switch['C-R2','Pedestrian']
    reductions={'VehicleGrossHarmReduction':va['GrossHarm']-vc['GrossHarm'],
        'VehicleGrossHarmReductionFraction':1-vc['GrossHarm']/va['GrossHarm'] if va['GrossHarm'] else None,
        'VehicleTop10HarmSumReduction':va['Top10HarmSum']-vc['Top10HarmSum'],
        'VehicleTop10HarmSumReductionFraction':1-vc['Top10HarmSum']/va['Top10HarmSum'] if va['Top10HarmSum'] else None,
        'PedestrianWrongSwitchCountReduction':pa['WorsenedCount']-pc['WorsenedCount'],
        'PedestrianWrongSwitchCountReductionFraction':1-pc['WorsenedCount']/pa['WorsenedCount'] if pa['WorsenedCount'] else None}
    conclusions={'ErrorAwareRanking':decision,'HardCE':'SUPPORTED' if hard else 'NOT_SUPPORTED',
        **{f'{g}Improved':'YES' if delta('C-A',g)<0 and delta('C-R2',g)<0 else 'NO' for g in ('Vehicle','Pedestrian','Overall')},
        'CostlySwitchReduced':'YES' if vc['GrossHarm']<va['GrossHarm'] and vc['Top10HarmSum']<va['Top10HarmSum'] else 'NO',
        'PedestrianWrongSwitchReduced':'YES' if pc['WorsenedCount']<pa['WorsenedCount'] else 'NO',
        'BicyclePreserved':'YES' if bike else 'NO','Stage11B':'GO' if decision in ('SUPPORTED','STRONG_SUPPORTED') else 'STOP',
        'ReadyForFurtherConfirmation':'YES' if decision in ('SUPPORTED','STRONG_SUPPORTED') else 'NO',
        'EngineeringGate':'pending independent final audit; any failure overrides GO to STOP',
        'PrimaryRegisteredBeforeTraining':True,'SecondaryBNotPromotedToPrimary':True,**reductions}
    atomic_json(ROOT/'09_reports/stage11b_scientific_decision.json',conclusions)
    atomic_json(ROOT/'06_bootstrap/stage11b_bootstrap_audit.json',{'Status':'PASS','Replicates':2000,'Seed':2022,
        'Unit':'whole scene with every actor/window paired between all models','WithinFoldDraws':210,'Folds':3,
        'SceneOrder':scene_order,'ReplicateWeightsSHA256':sha256(ROOT/'06_bootstrap/stage11b_scene_bootstrap_weights.npy'),
        '95DescriptivePercentiles':[2.5,97.5],'BonferroniFamilySize':2,'IndividualAdjustedPercentiles':[1.25,98.75],
        'CoPrimary':['C-A Vehicle Top1FDE','C-A Pedestrian Top1FDE'],'PointEstimatesReproduced':True,
        'FoldCheckpointsReselected':False,'MotionGroups':'exploratory','Science':conclusions})
    verify(history=True);print(json.dumps(conclusions,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
