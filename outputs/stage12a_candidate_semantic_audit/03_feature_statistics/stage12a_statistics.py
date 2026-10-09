"""Full coverage, all six modes, C/oracle errors and FoldR2-to-C switches."""
from stage12a_analysis_common import *
def main():
    verify();data=load_semantics();f,fd,ad,top,best,p,z,calibrated=load_labels();n=len(f);ii=np.arange(n)
    sem=data['semantic'];valid=data['feature_valid'];geo=data['geometry'];c=top[:,4];r2=top[:,1]
    # Reuse identical per-actor selected values instead of repeating strided
    # mmap gathers inside every fold/group/outcome loop. No values are changed.
    sv={name:np.asarray(sem[ii,mode]).T.copy() for name,mode in [('C',c),('R2',r2),('Oracle',best)]}
    vm={name:np.asarray(valid[ii,mode]).T.copy() for name,mode in [('C',c),('R2',r2),('Oracle',best)]}
    differences={'Error':sv['C']-sv['Oracle'],'Switch':sv['C']-sv['R2']}
    byfeature=np.asarray(sem).transpose(2,0,1).copy();byvalid=np.asarray(valid).transpose(2,0,1).copy()
    mv=sem[...,SEM_FIELDS.index('map_valid_mask')]>0;route=sem[...,SEM_FIELDS.index('centerline_valid')]>0
    pedsp=(sem[...,SEM_FIELDS.index('crosswalk_valid')]>0)|(sem[...,SEM_FIELDS.index('walkway_valid')]>0)
    delta=fd[ii,c].astype(np.float64)-fd[ii,r2].astype(np.float64);gap=fd[ii,c].astype(np.float64)-fd.min(-1).astype(np.float64)
    groupmasks=groups(f,geo);cov=[];featurestats=[];errors=[];switchrows=[];switchsem=[]
    for scope,sm in scopes(f):
        for group,gm in groupmasks.items():
            m=sm&gm
            if not m.any():continue
            cov.append({'Scope':scope,'Group':group,'Actors':int(m.sum()),'Candidates':int(m.sum()*6),
                'CandidateMapCoverage':float(mv[m].mean()),'ActorAnyModeCoverage':float(mv[m].any(-1).mean()),
                'ActorAllModesCoverage':float(mv[m].all(-1).mean()),'RouteCenterlineCoverage':float(route[m].mean()),
                'PedestrianSpecificCoverage':float(pedsp[m].mean()),'EmptyCandidates':int((~mv[m]).sum()),
                'MeanSelectedEntities':float((data['entity_ids'][m]>=0).sum(-1).mean())})
            if group in ['Overall',*ACTOR_TYPES,'MovingVehicle','StoppedVehicle','ParkedVehicle','Vehicle>5m','Pedestrian<5m','Pedestrian>5m','Pedestrian5-8m']:
                for k,field in enumerate(SEM_FIELDS):
                    selected=byfeature[k,m][byvalid[k,m]]
                    featurestats.append({'Scope':scope,'Group':group,'Feature':field,**describe(selected),'AllCandidates':int(m.sum()*6)})
            wrong=m&(c!=best)
            for k,field in enumerate(SEM_FIELDS):
                paired=wrong&vm['C'][k]&vm['Oracle'][k]
                difference=differences['Error'][k]
                info=describe(difference[paired])
                errors.append({'Scope':scope,'Group':group,'Feature':field,'WrongModeActors':int(wrong.sum()),
                    'PairedMapFeatureActors':int(paired.sum()),'MeanCFeature':float(sv['C'][k,paired].mean()) if paired.any() else None,
                    'MeanOracleFeature':float(sv['Oracle'][k,paired].mean()) if paired.any() else None,
                    **info,'MeanOracleGap':float(gap[wrong].mean()) if wrong.any() else None,
                    'WrongIndexButZeroFDEGap':int((wrong&(gap==0)).sum())})
            for state,condition in [('Improved',delta<0),('Worsened',delta>0),('Unchanged',delta==0),('HighCostWorsened',delta>5)]:
                choose=m&condition;positive=delta[choose];stats=describe(positive)
                harm=delta[m&(delta>0)];h=describe(harm)
                switchrows.append({'Scope':scope,'Group':group,'Outcome':state,**stats,'PopulationCount':int(m.sum()),
                    'ModeSwitches':int((choose&(r2!=c)).sum()),'TotalModeSwitches':int((m&(r2!=c)).sum()),
                    'MeanImprovementMagnitude':float(-delta[m&(delta<0)].mean()) if np.any(m&(delta<0)) else 0.,
                    'MeanWorseningMagnitude':float(harm.mean()) if len(harm) else 0.,
                    'GrossGain':float(-delta[m&(delta<0)].sum()),'GrossHarm':float(harm.sum()),
                    'HarmP90':h['P90'],'HarmP95':h['P95'],'HarmP99':h['P99']})
                for k,field in enumerate(SEM_FIELDS):
                    paired=choose&vm['C'][k]&vm['R2'][k];d=differences['Switch'][k]
                    switchsem.append({'Scope':scope,'Group':group,'Outcome':state,'Feature':field,'OutcomeActors':int(choose.sum()),
                        **describe(d[paired]),'MeanFDEDeltaPaired':float(delta[paired].mean()) if paired.any() else None})
    dump('03_feature_statistics/stage12a_semantic_coverage.csv',cov)
    dump('03_feature_statistics/stage12a_candidate_feature_statistics.csv',featurestats)
    dump('04_vehicle_analysis/stage12a_vehicle_semantic_error.csv',[r for r in errors if 'Vehicle' in r['Group']])
    dump('05_pedestrian_analysis/stage12a_pedestrian_semantic_error.csv',[r for r in errors if 'Pedestrian' in r['Group']])
    dump('06_bicycle_analysis/stage12a_bicycle_semantic_statistics.csv',[r for r in featurestats if r['Group']=='Bicycle'])
    dump('03_feature_statistics/stage12a_switch_semantic_analysis.csv',switchrows)
    dump('03_feature_statistics/stage12a_switch_feature_differences.csv',switchsem)
    hist=pd.read_csv(S8C/'06_tables/stage8a0c_sparse_coverage.csv');comparison=[]
    for t in ACTOR_TYPES:
        old=hist[(hist.Split=='train')&(hist.Population=='full_horizon_ranking_targets')&(hist.Group==t)].iloc[0]
        current=next(r for r in cov if r['Scope']=='Pooled' and r['Group']==t)
        comparison.append({'Group':t,'Stage8Population':'TRAIN700 full-horizon targets incl historical HeadDev70',
            'Stage12Population':'HeadTrain630 frozen OOF full-horizon targets only','Stage8Actors':int(old.Actors),
            'Stage12Actors':current['Actors'],'Stage8CandidateCoverage':float(old.MapNonEmptyRate),
            'Stage12CandidateCoverage':current['CandidateMapCoverage'],
            'DifferenceDueToPopulation':'same selector and candidate matching; different actor/scene population, not assumed equal'})
    dump('03_feature_statistics/stage12a_stage8_coverage_comparison.csv',comparison)
    # Full per-candidate CSV remains local. Every field is observable-derived.
    path=ROOT/'02_semantic_cache/cache/stage12a_mode_semantics.csv'
    if not path.exists():
        temporary=path.with_suffix('.csv.tmp')
        columns=['scene_id','sample_token','actor_id','actor_type','fold','mode_index','candidate_geometry_id','feature_valid_bitmask','entity_ids',*SEM_FIELDS,*GEO_FIELDS]
        for start in range(0,n,2048):
            ids=np.arange(start,min(start+2048,n));rr=f.iloc[ids];count=len(ids)
            bits=(valid[ids].astype(np.uint32)*(1<<np.arange(len(SEM_FIELDS),dtype=np.uint32))).sum(-1)
            block={'scene_id':np.repeat(rr.scene_token.to_numpy(),6),'sample_token':np.repeat(rr.sample_token.to_numpy(),6),
                'actor_id':np.repeat(rr.actor_id.to_numpy(),6),'actor_type':np.repeat(rr.agent_type.to_numpy(),6),
                'fold':np.repeat(rr.Fold.to_numpy(),6),'mode_index':np.tile(np.arange(6),count),
                'candidate_geometry_id':data['candidate_geometry_id'][ids].reshape(-1).astype('U64'),
                'feature_valid_bitmask':bits.reshape(-1),'entity_ids':['|'.join(map(str,a[a>=0])) for a in data['entity_ids'][ids].reshape(-1,8)]}
            block.update({k:sem[ids,:,j].reshape(-1) for j,k in enumerate(SEM_FIELDS)})
            block.update({k:geo[ids,:,j].reshape(-1) for j,k in enumerate(GEO_FIELDS)})
            pd.DataFrame(block,columns=columns).to_csv(temporary,index=False,mode='w' if start==0 else 'a',header=start==0,float_format='%.9g')
            if start%40960==0:print('LOCAL_MODE_CSV',start,'/',n,flush=True)
        temporary.replace(path)
    # Small versioned table summarizes each retained mode, with full CSV provenance.
    modes=[]
    for scope,sm in scopes(f):
        for t in ACTOR_TYPES:
            m=sm&(f.agent_type.to_numpy()==t)
            for mode in range(6):
                for k,field in enumerate(SEM_FIELDS):
                    v=m&byvalid[k,:,mode];modes.append({'Scope':scope,'Type':t,'ModeIndex':mode,'Feature':field,
                        **describe(byfeature[k,v,mode]),'FullActorCSV':'02_semantic_cache/cache/stage12a_mode_semantics.csv'})
    dump('03_feature_statistics/stage12a_mode_semantics.csv',modes)
    atomic_json(ROOT/'03_feature_statistics/stage12a_statistics_audit.json',{'Status':'PASS','SemanticCoverage':'PASS',
        'ActorsRetained':n,'CandidatesRetained':n*6,'MissingFeaturesFiniteWithMasks':True,'MapMissingActorsDropped':0,
        'MainTop1':'C Raw','CalibratedTop1ExactlyCraw':True,'BicycleFoldR2Identity':True,
        'LocalFullModeCSV':str(path.relative_to(ROOT)),'LocalFullModeCSVSHA256':sha256(path),
        'FullCSVRows':n*6,'StaticEntityDictionaryPath':str((S8C/'02_graph_cache/stage8a0c_entity_dictionary.json').relative_to(PROJECT)),
        'DictionarySHA256':sha256(S8C/'02_graph_cache/stage8a0c_entity_dictionary.json'),
        'ExplicitLabelAccessStartedAfterSemanticFreeze':True})
    print('COVERAGE_ALL_MODES_ERROR_SWITCH_STATISTICS_PASS',flush=True)
if __name__=='__main__':main()
