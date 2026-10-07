"""Registered forecasting/mechanism decisions from paired scene bootstraps."""
from stage7d_analysis_common import *
from scipy.stats import spearmanr

def main():
    f=paired_frames();full=f[f.horizon=='full_horizon'].reset_index(drop=True)
    att=pd.read_csv(ROOT/'04_evaluation/stage7d_actor_attention.csv')
    assert len(att)==54990 and att.TurningVehicle_GT.sum()==1663
    perf=[]
    for horizon in HORIZONS:
        frame=f[f.horizon==horizon].reset_index(drop=True)
        for group in FINAL_GROUPS:
            mask=group_mask(frame,group);n=int(mask.sum())
            if not n:continue
            for model in ('Stage3B','Stage7D'):
                perf.append({'Group':group,'Model':model,'Horizon':horizon,'Count':n,
                    **{k:float(frame.loc[mask,model+'_'+k].mean()) for k in METRICS}})
    table=pd.DataFrame(perf)
    for name,groups in [('main',MAIN_GROUPS),('vehicle_motion',MOTION_GROUPS),('semantic_subgroup',SEMANTIC_GROUPS)]:
        table[table.Group.isin(groups)].to_csv(ROOT/f'06_tables/stage7d_{name}_results.csv',index=False)
    rows=[];turn=[]
    attention_groups=['Overall','Vehicle','Pedestrian','vehicle.moving','Vehicle >5m','TurningVehicle_GT','GT-left','GT-right']
    for group in attention_groups:
        mask=group_mask(att,group)
        for model in ('Stage3B','Stage7A','Stage7D'):
            rows.append({'Group':group,'Model':model,'Count':int(mask.sum()),
                **{k:float(att.loc[mask,model+'_'+k].mean()) for k in ATTENTION_METRICS},
                **{'FiniteCount_'+k:int(att.loc[mask,model+'_'+k].notna().sum()) for k in ATTENTION_METRICS}})
    for group in ('TurningVehicle_GT','GT-left','GT-right'):
        mask=group_mask(att,group)
        for model in ('Stage3B','Stage7A','Stage7D'):
            turn.append({'Group':group,'Model':model,'Count':int(mask.sum()),
                         **{k:float(att.loc[mask,model+'_'+k].mean()) for k in TURN_METRICS}})
    write_csv(ROOT/'06_tables/stage7d_attention_relevance.csv',rows)
    write_csv(ROOT/'06_tables/stage7d_turn_attention.csv',turn)
    scenes=sorted(SceneDataset('val').scene_indices);assert len(scenes)==150
    scene_index={t:i for i,t in enumerate(scenes)};draws=np.random.default_rng(2022).integers(0,150,(1000,150))
    cis=[];scene_sums=[]
    comparisons=[(g,'minFDE6') for g in ['Overall','Vehicle','Pedestrian','vehicle.moving','Vehicle >5m',
        'IntersectionVehicle20','NearTurnConnector20','NearTrafficControl20','TurningVehicle_GT']]+[('Overall','Top1FDE6')]
    for group,metric in comparisons:
        row,sums=bootstrap_row(full,group_mask(full,group),metric,draws,scene_index,group);cis.append(row);scene_sums+=sums
    for group,metric in [('Vehicle','GTRelevantMass2m'),('vehicle.moving','GTRelevantMass2m'),
        ('Vehicle','GTNearestAttentionRank'),('Vehicle','AttentionEntropy'),
        ('TurningVehicle_GT','CorrectTurnMass'),('TurningVehicle_GT','TopConnectorMatchRate'),('TurningVehicle_GT','OppositeTurnMass')]:
        row,sums=bootstrap_row(att,group_mask(att,group),metric,draws,scene_index,group);cis.append(row);scene_sums+=sums
    write_csv(ROOT/'06_tables/stage7d_bootstrap_ci.csv',cis);write_csv(ROOT/'06_tables/stage7d_paired_scene_sums.csv',scene_sums)
    ci={(r['Group'],r['Metric']):r for r in cis};overall=ci['Overall','minFDE6']
    harm=[g for g in ('Vehicle','Pedestrian') if ci[g,'minFDE6']['Delta']>.05*ci[g,'minFDE6']['Stage3B'] and ci[g,'minFDE6']['CI_lower']>0]
    any_reliable_harm=[g for g in ('Vehicle','Pedestrian') if ci[g,'minFDE6']['CI_lower']>0]
    targeted=['vehicle.moving','Vehicle >5m','NearTurnConnector20','TurningVehicle_GT']
    reliable=[g for g in targeted if ci[g,'minFDE6']['CI_upper']<0]
    if overall['Delta']<0 and overall['CI_upper']<0 and not harm:decision='SUPPORTED'
    elif overall['Delta']<=.005*overall['Stage3B'] and overall['CI_lower']<=0<=overall['CI_upper'] and len(reliable)>=2 and not any_reliable_harm:decision='TARGETED_SUPPORTED'
    else:decision='NOT_SUPPORTED'
    relevant=any(ci[g,'GTRelevantMass2m']['CI_lower']>0 for g in ('Vehicle','vehicle.moving'))
    turns=any(ci['TurningVehicle_GT',m]['CI_lower']>0 for m in ('CorrectTurnMass','TopConnectorMatchRate'))
    mechanism='SUPPORTED' if relevant and turns else ('PARTIAL' if relevant or turns else 'NOT_SUPPORTED')
    paper=decision in ('SUPPORTED','TARGETED_SUPPORTED')
    atomic_json(ROOT/'04_evaluation/stage7d_scientific_decision.json',{
        'Stage7D':decision,'AttentionMechanism':mechanism,'PaperUsableSemantic':'YES' if paper else 'NO',
        'SemanticRoute':'CONTINUE' if paper else 'STOP','ReadySemanticReranking':'YES' if paper else 'NO',
        'marked_reliable_harm':harm,'any_reliable_Vehicle_Pedestrian_harm':any_reliable_harm,
        'reliably_improved_targeted_groups':reliable,'relevance_mechanism_leg':relevant,'turn_mechanism_leg':turns,
        'registered_before_formal_training':True,'Stage7E_executed':False,'R2_executed':False,'test_used':False})
    correlations=[]
    for group in attention_groups:
        mask=group_mask(att,group)
        pairs=[('DeltaRelevantMass','DeltaMinFDE'),('DeltaRelevantMass','DeltaTop1FDE'),
               ('DeltaGTNearestAttentionRank','DeltaMinFDE'),('DeltaGTNearestAttentionRank','DeltaTop1FDE')]
        if group in ('TurningVehicle_GT','GT-left','GT-right'):pairs+=[('DeltaCorrectTurnMass','DeltaMinFDE')]
        for x,y in pairs:
            use=mask&np.isfinite(att[x])&np.isfinite(att[y]);r,p=spearmanr(att.loc[use,x],att.loc[use,y])
            correlations.append({'Group':group,'X':x,'Y':y,'SpearmanR':float(r),'p_value':float(p),'Count':int(use.sum()),
                                 'Interpretation':'descriptive association; clustered observations; p-value does not replace scene bootstrap'})
    write_csv(ROOT/'06_tables/stage7d_attention_error_correlation.csv',correlations)
    print('STAGE7D_SCIENTIFIC_DECISION',decision,mechanism,'targeted',reliable,flush=True)

if __name__=='__main__':main()
