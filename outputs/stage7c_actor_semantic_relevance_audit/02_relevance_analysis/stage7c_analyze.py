"""Paired descriptive attention/residual tables and whole-scene uncertainty."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7c_common import *
from scipy.stats import spearmanr

ATTENTION_METRICS=['GTRelevantMass2m','GTRelevantMass4m','GTNearestAttentionWeight',
    'GTNearestAttentionRank','Top1LaneGTDistance','Top3MinGTDistance','Top5MinGTDistance','AttentionEntropy',
    'IncomingGTNearestAttentionWeight','IncomingGTNearestAttentionRank','BestCoNearestAttentionRank',
    'BestCoNearestAttentionWeight','GTNearestIncoming','GTNearestTieCount']

def bootstrap(f,group,mask,metric,draws,scene_index):
    b=f['Stage3B_'+metric].to_numpy(float);o=f['Stage7A_'+metric].to_numpy(float)
    valid=mask&np.isfinite(b)&np.isfinite(o)
    if not valid.any():return None
    idx=scene_index[valid];counts=np.bincount(idx,minlength=150)
    sb=np.bincount(idx,weights=b[valid],minlength=150);so=np.bincount(idx,weights=o[valid],minlength=150)
    den=counts[draws].sum(1);okay=den>0
    ds=(so[draws[okay]].sum(1)-sb[draws[okay]].sum(1))/den[okay]
    lower,upper=np.quantile(ds,[.025,.975])
    return {'Group':group,'Metric':metric,'Count':int(valid.sum()),'Stage3B':float(b[valid].mean()),
        'Stage7A':float(o[valid].mean()),'Delta':float((o[valid]-b[valid]).mean()),
        'CI_lower':float(lower),'CI_upper':float(upper),'replicates':1000,'valid_replicates':int(okay.sum()),
        'scenes':150,'nonempty_scenes':int((counts>0).sum()),'seed':2022,'unit':'paired whole-scene',
        'secondary_intervals':'descriptive unadjusted overlapping groups'}

def main():
    f=pd.read_csv(ROOT/'02_relevance_analysis/stage7c_actor_attention.csv',dtype={'future_mask_bits':str})
    assert len(f)==54990 and f.TurningVehicle_GT.sum()==1663 and not f.duplicated(IDS).any()
    for name in ['Stage3B','Stage7A']:
        assert np.isfinite(f[name+'_GTNearestDistance']).all()
        assert np.isfinite(f[name+'_GTRelevantMass2m']).all()
    gg=groups(f);rows=[];coverage=[]
    for group,mask in gg.items():
        if not mask.any():continue
        for model in ['Stage3B','Stage7A']:
            rows.append({'Group':group,'Model':model,'Count':int(mask.sum()),
                'AttentionCount':int((mask&(f.IncomingLaneCount>0)).sum()),
                **{m:float(f.loc[mask,model+'_'+m].mean()) for m in ATTENTION_METRICS}})
        coverage.append({'Group':group,'Count':int(mask.sum()),'ZeroIncomingEdges':int((mask&(f.IncomingLaneCount==0)).sum()),
            'GTNearestInsideIncomingRate':float(f.loc[mask,'Stage7A_GTNearestIncoming'].mean()),
            'GTNearestTiedRate':float((f.loc[mask,'Stage7A_GTNearestTieCount']>1).mean()),
            'MeanIncomingLaneCount':float(f.loc[mask,'IncomingLaneCount'].mean()),
            'MeanGraphLaneCount':float(f.loc[mask,'GraphLaneCount'].mean())})
    pd.DataFrame(rows).to_csv(ROOT/'06_tables/stage7c_attention_relevance.csv',index=False)
    pd.DataFrame(coverage).to_csv(ROOT/'06_tables/stage7c_lane_coverage.csv',index=False)
    turn_rows=[]
    for group,mask in gg.items():
        mask=mask&(f.TurningVehicle_GT==1)
        if not mask.any():continue
        r={'Group':group if group in ('TurningVehicle_GT','GT-left','GT-right') else group+' (turning subset)',
            'Count':int(mask.sum())}
        for model in ['Stage3B','Stage7A']:
            for metric in ['CorrectTurnMass','OppositeTurnMass','StraightMass','UnknownTurnMass','TopConnectorMatchRate','ConnectorPresent']:
                r[model+'_'+metric]=float(f.loc[mask,model+'_'+metric].mean())
            present=mask&(f[model+'_ConnectorPresent']==1)
            r[model+'_ConnectorPresentMatchRate']=float(f.loc[present,model+'_TopConnectorMatchRate'].mean()) if present.any() else np.nan
        turn_rows.append(r)
    pd.DataFrame(turn_rows).to_csv(ROOT/'06_tables/stage7c_turn_attention.csv',index=False)
    ambiguity=[]
    for n in range(4):
        mask=(f.agent_type=='vehicle')&(f.TurnOptionCount20==n)
        ambiguity.append({'TurnOptionCount20':'>=3' if n==3 else str(n),'Count':int(mask.sum()),
            'Stage3B_minFDE':float(f.loc[mask,'Stage3B_minFDE6'].mean()),'Stage7A_minFDE':float(f.loc[mask,'Stage7A_minFDE6'].mean()),
            'DeltaMinFDE':float(f.loc[mask,'DeltaMinFDE'].mean()),
            'Stage3B_Top1FDE':float(f.loc[mask,'Stage3B_Top1FDE6'].mean()),'Stage7A_Top1FDE':float(f.loc[mask,'Stage7A_Top1FDE6'].mean()),
            'DeltaTop1FDE':float(f.loc[mask,'DeltaTop1FDE'].mean()),'RelevantAttentionDelta':float(f.loc[mask,'DeltaRelevantMass'].mean())})
    pd.DataFrame(ambiguity).to_csv(ROOT/'06_tables/stage7c_turn_ambiguity.csv',index=False)
    tokens=sorted(f.scene_token.unique());assert len(tokens)==150
    mapping={t:i for i,t in enumerate(tokens)};idx=f.scene_token.map(mapping).to_numpy()
    draws=np.random.default_rng(2022).integers(0,150,(1000,150));cis=[]
    for group,mask in gg.items():
        for metric in ['GTRelevantMass2m','GTNearestAttentionRank','AttentionEntropy',
            'IncomingGTNearestAttentionRank','BestCoNearestAttentionRank']:
            r=bootstrap(f,group,mask,metric,draws,idx)
            if r:cis.append(r)
        turnmask=mask&(f.TurningVehicle_GT.to_numpy()==1)
        for metric in ['CorrectTurnMass','OppositeTurnMass','TopConnectorMatchRate']:
            r=bootstrap(f,group+' (turning subset)' if group not in ('TurningVehicle_GT','GT-left','GT-right') else group,
                turnmask,metric,draws,idx)
            if r:cis.append(r)
    pd.DataFrame(cis).to_csv(ROOT/'06_tables/stage7c_attention_bootstrap_ci.csv',index=False)
    associations=[]
    for group,mask in gg.items():
        for x in ['DeltaRelevantMass','DeltaGTNearestAttentionRank','DeltaCorrectTurnMass']:
            for y in ['DeltaMinFDE','DeltaTop1FDE']:
                valid=mask&np.isfinite(f[x])&np.isfinite(f[y]);n=int(valid.sum())
                if n<3:continue
                if f.loc[valid,x].nunique()<2 or f.loc[valid,y].nunique()<2:r,p=np.nan,np.nan
                else:r,p=spearmanr(f.loc[valid,x],f.loc[valid,y])
                associations.append({'Group':group,'X':x,'Y':y,'SpearmanR':float(r),'p_value':float(p),
                    'Count':n,'interpretation':'descriptive; correlated windows; not causal; no multiplicity adjustment'})
    pd.DataFrame(associations).to_csv(ROOT/'06_tables/stage7c_attention_error_correlation.csv',index=False)
    direction=[]
    for group,mask in gg.items():
        for label,sub in [('improved',f.DeltaMinFDE<0),('degraded',f.DeltaMinFDE>0)]:
            selected=mask&sub
            if not selected.any():continue
            direction.append({'Group':group,'Outcome':label,'Count':int(selected.sum()),
                'MeanDeltaMinFDE':float(f.loc[selected,'DeltaMinFDE'].mean()),
                'MeanDeltaRelevantMass':float(f.loc[selected,'DeltaRelevantMass'].mean()),
                'MeanDeltaCorrectTurnMass':float(f.loc[selected,'DeltaCorrectTurnMass'].mean())})
    pd.DataFrame(direction).to_csv(ROOT/'06_tables/stage7c_attention_by_error_direction.csv',index=False)
    z=np.load(ROOT/'03_representation_analysis/stage7c_residual_patterns.npz')
    s=z['patterns'];counts=z['counts'];used=counts>0;r0=float(np.linalg.norm(z['r0']))
    predicates={'all':used,'ordinary_lane':used&(s[:,0]==0),'connector':used&(s[:,0]==1),
        'left':used&(s[:,1]==1),'straight':used&(s[:,2]==1),'right':used&(s[:,3]==1),
        'unknown_connector':used&(s[:,4]==1),'traffic_light':used&(s[:,5]==1),'stop_sign':used&(s[:,6]==1),
        'other_control':used&(s[:,7]==1),'crosswalk':used&(s[:,8]==1),'non_crosswalk':used&(s[:,8]==0)}
    residual_rows=[]
    for group,mask in predicates.items():
        n=int(counts[mask].sum());assert n>0
        residual_rows.append({'SemanticGroup':group,'Count':n,'UniquePatterns':int(mask.sum()),
            'mean_norm_r':float(np.average(z['norm_r'][mask],weights=counts[mask])),
            'mean_norm_r0':r0,'mean_norm_delta_r':float(np.average(z['norm_delta_r'][mask],weights=counts[mask])),
            'mean_delta_ratio':float(np.average(z['ratio'][mask],weights=counts[mask])),
            'weighting':'actual stored VAL segment/window occurrences'})
    pd.DataFrame(residual_rows).to_csv(ROOT/'06_tables/stage7c_residual_decomposition.csv',index=False)
    pattern_rows=[]
    for i in np.flatnonzero(used):
        pattern_rows.append({'PatternCode':int(i),'Count':int(counts[i]),**dict(zip(SEMANTIC_FIELDS,s[i].astype(int))),
            'norm_r':float(z['norm_r'][i]),'norm_r0':r0,'norm_delta_r':float(z['norm_delta_r'][i]),'delta_ratio':float(z['ratio'][i])})
    pd.DataFrame(pattern_rows).to_csv(ROOT/'06_tables/stage7c_residual_by_pattern.csv',index=False)
    original=pd.read_csv(STAGE7A/'06_tables/stage7a_semantic_residual_statistics.csv').set_index('group').loc['all']
    assert int(counts.sum())==int(original.VAL_segment_occurrences)
    assert abs(residual_rows[0]['mean_norm_r']-original.mean_norm)<1e-5
    assert abs(residual_rows[0]['mean_norm_delta_r']-original.mean_norm_difference_from_zero_input)<1e-5
    representation_metrics=['Stage3B_MeanGeometryNorm','Stage7A_MeanGeometryNorm','Stage3B_MeanKeyNorm','Stage7A_MeanKeyNorm',
        'Stage3B_MeanValueNorm','Stage7A_MeanValueNorm','MeanValueDeltaNorm','MeanValueCosine',
        'SemanticEffect','GTRelevantSemanticEffect','AttentionWeightedResidualValueNorm','AttentionWeightedSpecificValueNorm',
        'AttentionWeightedResidualKeyNorm','ConstantValueComponentNorm']
    pd.DataFrame([{'Group':group,'Count':int(mask.sum()),**{k:float(f.loc[mask,k].mean()) for k in representation_metrics}}
        for group,mask in gg.items() if mask.any()]).to_csv(ROOT/'06_tables/stage7c_representation_effect.csv',index=False)
    atomic_json(ROOT/'02_relevance_analysis/stage7c_analysis_audit.json',{'status':'PASS','full_targets':len(f),
        'vehicle_count':int((f.agent_type=='vehicle').sum()),'turn_count':1663,'GT_left':int(((f.TurningVehicle_GT==1)&(f.GT_heading_change_deg>0)).sum()),
        'GT_right':int(((f.TurningVehicle_GT==1)&(f.GT_heading_change_deg<0)).sum()),
        'HighAmbiguityCount':int(((f.agent_type=='vehicle')&(f.TurnOptionCount20>=3)).sum()),
        'bootstrap_rows':len(cis),'correlation_rows':len(associations),'residual_segment_occurrences':int(counts.sum()),
        'future_GT_never_input':True,'historical_residual_statistics_reproduced':True,'training_updates':0})
    print('ANALYSIS COMPLETE',residual_rows[0],flush=True)

if __name__=='__main__':main()
