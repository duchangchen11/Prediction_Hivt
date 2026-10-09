"""Prespecified semantic association and two deterministic negative controls."""
from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'00_manifest'),str(Path(__file__).resolve().parents[1]/'03_feature_statistics')]
from stage12a_analysis_common import *
def matched_rho(x,y,mask,bins):
    sums=np.zeros(len(x));counts=np.zeros(len(x),np.int8)
    for b in range(5):
        rho,eligible=rho_rows(x,y,mask&(bins==b));sums+=rho;counts+=eligible.astype(np.int8)
    return np.divide(sums,counts,out=np.zeros(len(x)),where=counts>0),counts>0
def comparisons(scene,arrays,masks,population):
    common=population&masks[0]&masks[1]&masks[2]
    real,nc1,nc2=arrays
    result={'CommonActors':int(common.sum()),'RealOnlyEligibleActors':int((population&masks[0]).sum())}
    for label,a in [('Real',real),('NC1',nc1),('NC2',nc2)]:
        stat=scene.mean(a,common,bootstrap=label=='Real')
        result[label+'_SceneMeanRho']=stat['SceneMean'];result[label+'_CI95Low']=stat['CI95Low'];result[label+'_CI95High']=stat['CI95High']
        for q in ['SceneMedian','SceneP90','SceneP95']:result[label+'_'+q]=stat[q]
        result['Scenes']=stat['Scenes']
    for label,a in [('NC1',nc1),('NC2',nc2)]:
        stat=scene.mean(real-a,common,bootstrap=True)
        result['DeltaRealMinus'+label]=stat['SceneMean'];result['Delta'+label+'_CI95Low']=stat['CI95Low'];result['Delta'+label+'_CI95High']=stat['CI95High']
        for q in ['SceneMedian','SceneP90','SceneP95']:result['Delta'+label+'_'+q]=stat[q]
    return result
def main():
    verify();data=load_semantics();f,fd,ad,top,best,prob,logits,calibrated=load_labels();n=len(f)
    scene=SceneStatistics(f);sem=data['semantic'];valid=data['feature_valid'];geom=data['geometry'];ii=np.arange(n)
    root=ROOT/'08_negative_controls/cache';root.mkdir(exist_ok=True)
    rng=np.random.default_rng(2022);nc1=np.stack([rng.permutation(6) for _ in range(n)]).astype(np.int8)
    rng=np.random.default_rng(2022);nc2=np.arange(n*6,dtype=np.int32).reshape(n,6)
    for region in range(4):
        for typ in range(3):
            actors=np.flatnonzero((data['map_region']==region)&(f.agent_type_id.to_numpy()==typ))
            flat=(actors[:,None]*6+np.arange(6)).reshape(-1)
            if len(flat):nc2[actors]=rng.permutation(flat).reshape(-1,6)
    assert np.all(np.sort(nc1,axis=1)==np.arange(6)) and np.array_equal(np.sort(nc2.ravel()),np.arange(n*6))
    assert np.array_equal(f.agent_type_id.to_numpy()[nc2//6],np.repeat(f.agent_type_id.to_numpy()[:,None],6,axis=1))
    assert np.array_equal(data['map_region'][nc2//6],np.repeat(data['map_region'][:,None],6,axis=1))
    np.save(root/'stage12a_candidate_internal_permutation.npy',nc1);np.save(root/'stage12a_region_type_permutation.npy',nc2)
    bins=np.searchsorted([1,5,10,20],geom[...,GEO_FIELDS.index('predicted_displacement')],side='right')
    rows=[];matched=[];grouprows=[];regionrows=[];perfeature={};protocol=read_json(PROTOCOL)
    for typ,spec in protocol['PrimarySignals'].items():
        population=f.agent_type.to_numpy()==typ
        for field,sign in spec:
            k=SEM_FIELDS.index(field);x=np.asarray(sem[:,:,k])*sign;v=np.asarray(valid[:,:,k])
            xs=[x,x[ii[:,None],nc1],x.reshape(-1)[nc2]];vs=[v,v[ii[:,None],nc1],v.reshape(-1)[nc2]]
            associations=[rho_rows(a,fd,m) for a,m in zip(xs,vs)]
            conditioned=[matched_rho(a,fd,m,bins) for a,m in zip(xs,vs)]
            a=[p[0] for p in associations];m=[p[1] for p in associations]
            b=[p[0] for p in conditioned];bm=[p[1] for p in conditioned]
            for scope,scopeMask in scopes(f):
                info=comparisons(scene,a,m,population&scopeMask)
                selection=population&scopeMask&(top[:,4]!=best)&v[ii,top[:,4]]&v[ii,best]
                d=x[ii,top[:,4]]-x[ii,best];selectedstat=scene.mean(d,selection,bootstrap=False)
                rows.append({'Scope':scope,'ActorType':typ,'Feature':field,'OrientedSign':sign,**info,
                    'CMinusOracleSignedSceneMean':selectedstat['SceneMean'],'COraclePairedActors':selectedstat['Count']})
                matched.append({'Scope':scope,'ActorType':typ,'Feature':field,'OrientedSign':sign,
                    'Matching':'same actor/type/mapregion/history context, predicted displacement bin',**comparisons(scene,b,bm,population&scopeMask)})
            for region in range(4):
                regionmask=population&(data['map_region']==region)
                for binid in range(5):
                    conditionedbin=[rho_rows(xx,fd,vv&(bins==binid)) for xx,vv in zip(xs,vs)]
                    common=regionmask&conditionedbin[0][1]&conditionedbin[1][1]&conditionedbin[2][1]
                    for scope,sm in scopes(f):
                        rr={'Scope':scope,'ActorType':typ,'Feature':field,'MapRegionIndex':region,'PredictedDisplacementBin':binid}
                        for label,pair in zip(['Real','NC1','NC2'],conditionedbin):
                            st=scene.mean(pair[0],common&sm,bootstrap=False)
                            rr.update({label+'_'+key:val for key,val in st.items() if not key.startswith('CI')})
                        regionrows.append(rr)
            # Motion and history subgroups are exploratory, never select features.
            for group,groupmask in groups(f,geom).items():
                if group=='Overall' or group in ACTOR_TYPES:continue
                if (typ=='Vehicle')!=('Vehicle' in group):continue
                for scope,scopeMask in scopes(f):
                    realEligible=population&groupmask&scopeMask&m[0]
                    stat=scene.mean(a[0],realEligible,bootstrap=False)
                    grouprows.append({'Scope':scope,'Group':group,'ActorType':typ,'Feature':field,**stat})
            perfeature[field]={'rho':a[0],'eligible':m[0],'matched_rho':b[0],'matched_eligible':bm[0]}
            print('SEMANTIC_CONTROLS_AND_MATCHED_MOTION',typ,field,flush=True)
    dump('08_negative_controls/stage12a_negative_controls.csv',rows)
    dump('07_incremental_information/stage12a_incremental_information.csv',matched)
    dump('03_feature_statistics/stage12a_motion_semantic_associations.csv',grouprows)
    dump('07_incremental_information/stage12a_region_displacement_associations.csv',regionrows)
    # Correlate map features with existing motion/interaction summaries, without
    # fitting a predictor, selecting a new feature or changing C mode scores.
    correlation=[]
    controls=['predicted_displacement','endpoint_x','endpoint_y','predicted_heading','trajectory_length','trajectory_curvature',
        'interaction_minimum_distance','interaction_mean_distance','interaction_closing_mean']
    for typ,spec in protocol['PrimarySignals'].items():
        pop=f.agent_type.to_numpy()==typ
        for field,sign in spec:
            k=SEM_FIELDS.index(field);x=sem[:,:,k]*sign;v=np.asarray(valid[:,:,k])
            for control in controls:
                y=geom[:,:,GEO_FIELDS.index(control)];mask=v.copy()
                if control.startswith('interaction_'):mask&=geom[:,:,GEO_FIELDS.index('interaction_valid')]>0
                if control=='predicted_heading':mask&=geom[:,:,GEO_FIELDS.index('predicted_heading_valid')]>0
                rho,eligible=rho_rows(x,y,mask)
                for scope,scopemask in scopes(f):correlation.append({'Scope':scope,'ActorType':typ,'SemanticFeature':field,
                    'ObservableControl':control,**scene.mean(rho,pop&scopemask&eligible,bootstrap=False),
                    'CircularAngleCaution':'heading Spearman uses wrapped angle order; descriptive only' if control=='predicted_heading' else ''})
    dump('07_incremental_information/stage12a_semantic_geometry_interaction_correlations.csv',correlation)
    signals={};criteria=[]
    for typ,spec in protocol['PrimarySignals'].items():
        strongest='NONE'
        for feature,sign in spec:
            pool=next(r for r in rows if r['Scope']=='Pooled' and r['ActorType']==typ and r['Feature']==feature)
            folds=[r for r in rows if r['Scope']!='Pooled' and r['ActorType']==typ and r['Feature']==feature]
            replicates=sum(r['CommonActors']>=100 and r['Scenes']>=20 and r['Real_SceneMeanRho'] is not None and r['Real_SceneMeanRho']>0 for r in folds)
            selected=sum(r['CMinusOracleSignedSceneMean'] is not None and r['CMinusOracleSignedSceneMean']>0 for r in folds)
            weak=pool['Real_SceneMeanRho'] is not None and pool['Real_SceneMeanRho']>=.05 and replicates>=2
            controlpass=all(pool['DeltaRealMinus'+c] is not None and pool['DeltaRealMinus'+c]>=.03 and pool['Delta'+c+'_CI95Low'] is not None and pool['Delta'+c+'_CI95Low']>0 for c in ['NC1','NC2'])
            promising=weak and pool['CommonActors']>=500 and pool['Scenes']>=50 and controlpass and selected>=2
            grade='PROMISING' if promising else ('WEAK' if weak else 'NONE')
            if ['NONE','WEAK','PROMISING'].index(grade)>['NONE','WEAK','PROMISING'].index(strongest):strongest=grade
            mp=next(r for r in matched if r['Scope']=='Pooled' and r['ActorType']==typ and r['Feature']==feature)
            mf=[r for r in matched if r['Scope']!='Pooled' and r['ActorType']==typ and r['Feature']==feature]
            incremental=promising and mp['Real_SceneMeanRho'] is not None and mp['Real_SceneMeanRho']>=.05 and sum(r['Real_SceneMeanRho'] is not None and r['Real_SceneMeanRho']>=.05 for r in mf)>=2 and all(mp['Delta'+c+'_CI95Low'] is not None and mp['Delta'+c+'_CI95Low']>0 for c in ['NC1','NC2'])
            criteria.append({'ActorType':typ,'Feature':feature,'Grade':grade,'PositiveEligibleFolds':replicates,
                'PositiveSelectedDifferenceFolds':selected,'ControlContrastPass':controlpass,'MatchedMotionIncrementalSuggested':bool(incremental)})
        signals[typ]=strongest
    value='SUGGESTED' if any(r['MatchedMotionIncrementalSuggested'] for r in criteria) else ('UNRESOLVED' if any(s!='NONE' for s in signals.values()) else 'NOT_OBSERVED')
    decision={'Status':'PASS','NegativeControl':'PASS','VehicleSemanticSignal':signals['Vehicle'],
        'PedestrianSemanticSignal':signals['Pedestrian'],'SemanticIncrementalValue':value,'FeatureCriteria':criteria,
        'AllFeatureAndMaskDistributionsPreservedByPermutation':True,'CandidateFDEModelScoresUnchanged':True,
        'NC1PermutationSHA256':sha256(root/'stage12a_candidate_internal_permutation.npy'),
        'NC2PermutationSHA256':sha256(root/'stage12a_region_type_permutation.npy'),'Seed':2022,
        'BootstrapUnit':'whole scene within each frozen outer fold, paired real and both controls',
        'GenuineIndependentPredictionGain':'UNRESOLVED','Evidence':'descriptive development evidence, no independent confirmation or causal inference'}
    atomic_json(ROOT/'08_negative_controls/stage12a_negative_control_audit.json',decision)
    print('SEMANTIC_SCREEN_DECISION',signals,value,flush=True)
if __name__=='__main__':main()
