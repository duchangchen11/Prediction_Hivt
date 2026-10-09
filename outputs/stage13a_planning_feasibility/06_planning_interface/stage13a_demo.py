"""Fixed12-scene feasibility demonstration, identity joins and full GT poison."""
from pathlib import Path
import sys,copy
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/s) for s in ['00_manifest','02_ego_trajectory','04_observation_only_inference','05_map_alignment','06_planning_interface','07_evaluation_preflight']]
from stage13a_common import *
from stage13a_trajectory import extract_observation,extract_evaluation
from stage13a_forecast import FrozenForecaster
from stage13a_map import MapInterface,save_map
from stage13a_interface import PlanningObservation,PlanningEvaluationLabels,PlanningSample,cv_reference
from stage13a_metrics import evaluate,numerical_audit
def obs_digest(obs):
    h=hashlib.sha256()
    for k in ['history','history_heading','history_mask','agent_type','ego_history','ego_history_heading','future_times','prediction_valid_mask','current_position','current_heading','actor_size_length_width']:
        h.update(k.encode());h.update(np.ascontiguousarray(obs[k]).tobytes())
    h.update('|'.join(obs['instance_tokens']).encode());return h.hexdigest()
def main():
    seed();verify();fixed=read_json(ROOT/'01_dataset_audit/stage13a_fixed_samples.json');assert fixed['Status']=='FIXED_BEFORE_INFERENCE'
    metricgate=numerical_audit();atomic_json(ROOT/'07_evaluation_preflight/stage13a_metric_unit_audit.json',metricgate)
    forecaster=FrozenForecaster();maps=MapInterface();con=connection();f=pd.read_csv(S11B/'05_oof_evaluation/cache/stage11b_oof_predictions_actor_records.csv')
    keys=list(zip(f.scene_token,f.sample_token,f.instance_token));assert len(set(keys))==260151;lookup={key:i for i,key in enumerate(keys)}
    old=np.load(S11A/'01_identity_audit/cache/candidates.npy',mmap_mode='r');origin=np.load(S12A/'01_map_integrity/cache/stage12a_origin.npy',mmap_mode='r');yaw=np.load(S12A/'01_map_integrity/cache/stage12a_yaw.npy',mmap_mode='r')
    source_z=np.load(S11B/'05_oof_evaluation/cache/stage11b_oof_predictions_logits.npy',mmap_mode='r');source_p=np.load(S11B/'05_oof_evaluation/cache/stage11b_oof_predictions_probabilities.npy',mmap_mode='r')
    selected=[];joined=[];evaluations=[];timestamps=[];poisonrows=[];coverage=[];cached={};shards={};roundtrip=0.;pose_known=0;labels_count=0;matching_history=0
    index=pd.read_csv(S3/'02_preprocessed/stage3_train_index.csv');graphlookup={(r.scene_token,r.sample_token):r for r in index.itertuples()}
    for ordinal,rec in enumerate(fixed['Samples'],1):
        token=rec['SceneToken']
        if token not in cached:db=load_scene(con,token);chain=scene_samples(db,db.scene[0]);cached[token]=(db,chain)
        db,chain=cached[token];i=rec['T0Index'];obs=extract_observation(db,chain,i,np.array(rec['FutureTimes']));assert obs['sample_token']==rec['SampleToken']
        before=obs_digest(obs);output=forecaster(obs);after=forecaster(obs,poison_unused_future=True)
        assert all(np.array_equal(output[k],after[k]) for k in output),'STOP: future placeholder poison changed prediction'
        geometry=maps(obs);cv_info=dict(scene_token=token,sample_token=obs['sample_token'],fold=rec['Fold'],t0_timestamp=obs['t0_timestamp'],ego_history=obs['ego_history'],
            ego_history_heading=obs['ego_history_heading'],history_times=obs['history_times'],future_times=obs['future_times'],instance_tokens=obs['instance_tokens'],
            other_agent_predictions=output['prediction'],other_agent_probabilities=output['probabilities'],other_agent_raw_probabilities=output['raw_probabilities'],other_agent_type=obs['agent_type'],
            other_agent_current_position=obs['current_position'],other_agent_current_heading=obs['current_heading'],other_agent_sizes_length_width=obs['actor_size_length_width'],other_agent_size_source=obs['actor_size_source'],
            prediction_valid_mask=output['valid'],other_agent_history=obs['history'],other_agent_history_mask=obs['history_mask'],map_geometry=geometry,
            model_identity=dict(Stage5ASHA256=sha256(ROOT.parent/'stage5a_motion_aware_decoder/07_checkpoints/stage5a_best_overall_minfde.pt'),
                CSHA256=sha256(S11B/f'04_checkpoints/fold{rec["Fold"]}/C_best.pt'),R2SHA256=sha256(S11B/f'04_checkpoints/fold{rec["Fold"]}/R2_best.pt'),Temperature=TEMPERATURES[rec['Fold']],
                NormalizationSHA256=sha256(S11B/f'02_splits/stage11b_fold{rec["Fold"]}_normalization.json')))
        observation=PlanningObservation(**cv_info).validate();plan=cv_reference(observation);assert plan.shape==(12,2) and np.isfinite(plan).all()
        # Future labels are first constructed AFTER frozen predictions/planning.
        labeldict=extract_evaluation(db,chain,i,obs);labels=PlanningEvaluationLabels(**labeldict);sample=PlanningSample(observation,labels)
        poisoned={k:(np.full_like(v,np.nan) if v.dtype.kind=='f' else np.zeros_like(v)) if isinstance(v,np.ndarray) else v for k,v in labeldict.items()}
        poisoned['ego_future_mask']=np.ones(12,bool);poisoned['agent_future_mask']=np.ones_like(labels.agent_future_mask,bool)
        poisoned_sample=PlanningSample(observation,PlanningEvaluationLabels(**poisoned));assert np.array_equal(cv_reference(sample.Observation),cv_reference(poisoned_sample.Observation))
        assert obs_digest(obs)==before
        try:cv_reference(poisoned_sample);raise AssertionError('fullsample accepted by inference')
        except TypeError:pass
        rawpoison=False
        if rec['Role']=='first-evaluable':
            altered=copy.deepcopy(db);future_samples={s['token'] for s in chain[i+1:]};future_poses={altered.get('sample_data',s['data']['LIDAR_TOP'])['ego_pose_token'] for s in chain[i+1:]}
            for pose_token in future_poses:altered.tables['ego_pose'][pose_token]['translation']=[float('nan')]*3
            for ann in altered.tables['sample_annotation'].values():
                if ann['sample_token'] in future_samples:ann['translation']=[float('nan')]*3;ann['rotation']=[float('nan')]*4
            again=extract_observation(altered,chain,i,obs['future_times']);again_out=forecaster(again);assert obs_digest(again)==before and all(np.array_equal(output[k],again_out[k]) for k in output);rawpoison=True
        row,pc,gc=evaluate(observation,labels,plan,'CV_Ego_EngineeringReference');row.update(SampleOrdinal=ordinal,SceneToken=token,SampleToken=obs['sample_token'],Role=rec['Role']);evaluations.append(row)
        if labels.ego_future_mask.all():
            recorded=labels.ego_future_gt.copy();r2,_,_=evaluate(observation,labels,recorded,'RecordedEgoFuture_EvaluationOnly');r2.update(SampleOrdinal=ordinal,SceneToken=token,SampleToken=obs['sample_token'],Role=rec['Role']);evaluations.append(r2);labels_count+=1
            # Independently reuse the historical builder for label/frame identity.
            legacy=build_window(db,db.scene[0],i,samples=chain)
            assert np.array_equal(legacy['agent_pos'].numpy(),obs['history'].astype(np.float32)) and np.array_equal(legacy['ego_future'].numpy(),labels.ego_future_gt.astype(np.float32))
            assert np.array_equal(legacy['future_times'].numpy(),obs['future_times'])
        historical_found=0;legacy_graph=graphlookup.get((token,obs['sample_token']))
        if legacy_graph is not None:
            path=S3/legacy_graph.file_path
            if str(path) not in shards:shards[str(path)]=torch.load(path,map_location='cpu',weights_only=False)['graphs']
            g=shards[str(path)][int(legacy_graph.window_index)];assert g.scene_token==token and g.sample_token==obs['sample_token']
            assert list(g.instance_tokens)==list(obs['instance_tokens']) and np.array_equal(g.positions[:,:5].numpy(),obs['history'].astype(np.float32)) and np.array_equal(g.history_mask.numpy(),obs['history_mask'])
            assert np.array_equal(g.future_times.numpy(),obs['future_times']);matching_history+=1
        for node,instance in enumerate(obs['instance_tokens']):
            key=(token,obs['sample_token'],instance);oldrow=lookup.get(key);expected=bool(obs['prediction_valid_mask'][node] and labels.agent_future_mask[node].all())
            assert (oldrow is not None)==expected,(key,'historical fulltarget identity inconsistent with current history/future labels')
            row=dict(SceneToken=token,SampleToken=obs['sample_token'],InstanceToken=instance,ObservationNode=node,ObservationEligible=bool(output['valid'][node]),HistoricalOOFJoined=oldrow is not None,
                AgentType=TYPES[obs['agent_type'][node]],HistoricalOOFMissingReason='' if oldrow is not None else ('insufficient_history' if not output['valid'][node] else 'future_incomplete_or_outside_recording'),
                FutureValidSteps=int(labels.agent_future_mask[node].sum()),Fold=rec['Fold'])
            if oldrow is not None:
                r=f.iloc[oldrow];assert int(r.Fold)==rec['Fold'] and abs(float(yaw[oldrow])-obs['yaw'])<1e-12 and np.max(np.abs(origin[oldrow]-obs['origin']))<1e-9
                pred=old[int(r.source_index)];identifier=[hashlib.sha256(a.tobytes()).hexdigest() for a in pred];row.update(HistoricalActorRow=int(oldrow),CandidateGeometryIDs='|'.join(identifier),
                    FrameOriginMaxDiffM=float(np.abs(origin[oldrow]-obs['origin']).max()),ReplaySingleGraphCandidateMaxDiffM=float(np.abs(pred-output['prediction'][node]).max()),
                    OldCMode=int(source_p[oldrow,4].argmax()),CoordinateFrame=observation.coordinate_frame);historical_found+=1
            joined.append(row)
        # Explicitly retain every supported actor, including GT-incomplete and
        # history-insufficient actors absent from the old OOF target table.
        for typ,name in enumerate(TYPES):
            chosen=obs['agent_type']==typ;coverage.append(dict(SampleOrdinal=ordinal,SceneToken=token,SampleToken=obs['sample_token'],AgentType=name,
                CurrentActors=int(chosen.sum()),HistoryEligible=int((chosen&output['valid']).sum()),MissingHistory=int((chosen&~output['valid']).sum()),
                HistoricalOOFActors=sum(lookup.get((token,obs['sample_token'],t)) is not None for t in np.array(obs['instance_tokens'])[chosen]),
                FutureIncompleteEligible=int((chosen&output['valid']&~labels.agent_future_mask.all(-1)).sum()),FiniteForwardRows=int(chosen.sum())))
        roundtrip=max(roundtrip,obs['coordinate_roundtrip_max_error_m'])
        for j,source_sample in enumerate(chain[max(0,i-4):min(len(chain),i+13)]):
            sd=db.get('sample_data',source_sample['data']['LIDAR_TOP']);pose=db.get('ego_pose',sd['ego_pose_token']);original=original_extras()[1].get(pose['token']);known=original is not None;pose_known+=known
            timestamps.append(dict(SampleOrdinal=ordinal,SceneToken=token,T0SampleToken=obs['sample_token'],FrameSampleToken=source_sample['token'],FrameRelativeIndex=j-4,
                SampleTimestampUS=int(source_sample['timestamp']),LidarTimestampUS=int(sd['timestamp']),SampleMinusLidarUS=int(source_sample['timestamp']-sd['timestamp']),
                EgoPoseToken=pose['token'],EgoPoseTimestampUS=int(original['timestamp']) if known else None,PoseTimestampSource='originalJSON' if known else 'not_retained_in_SQLite',
                PoseMinusLidarUS=int(original['timestamp']-sd['timestamp']) if known else None))
        folder=ROOT/'06_planning_interface/cache';stem=f'stage13a_sample_{ordinal:03d}'
        atomic_npz(folder/(stem+'_observation.npz'),**{k:v for k,v in obs.items() if isinstance(v,np.ndarray)},other_agent_predictions=output['prediction'],
            other_agent_probabilities=output['probabilities'],other_agent_raw_probabilities=output['raw_probabilities'],other_agent_logits=output['logits'],cv_ego_reference=plan)
        atomic_npz(folder/(stem+'_evaluation_only.npz'),**{k:v for k,v in labeldict.items() if isinstance(v,np.ndarray)})
        save_map(ROOT/'05_map_alignment/cache'/f'{stem}_map.json',geometry)
        atomic_json(folder/(stem+'_identity.json'),dict(SceneToken=token,SampleToken=obs['sample_token'],InstanceTokens=list(obs['instance_tokens']),Fold=rec['Fold'],T0Timestamp=obs['t0_timestamp'],
            CoordinateFrame=obs['coordinate_frame'],Region=obs['location'],FutureTimeSource=rec['FutureTimeSource'],ModelIdentity=observation.model_identity,PastMetadataReads=list(obs['past_metadata_reads'])))
        selected.append(dict(SampleOrdinal=ordinal,SceneToken=token,SampleToken=obs['sample_token'],Fold=rec['Fold'],T0Index=i,Role=rec['Role'],MapRegion=obs['location'],
            CurrentActors=len(obs['instance_tokens']),VehicleActors=int((obs['agent_type']==0).sum()),PedestrianActors=int((obs['agent_type']==1).sum()),BicycleActors=int((obs['agent_type']==2).sum()),
            PredictionEligible=int(output['valid'].sum()),HistoricalOOFJoined=historical_found,CompleteEgoEvaluation=bool(labels.ego_future_mask.all()),
            PotentialPredictionConflict=pc['AnyFootprintOverlap'],GTProxyCoverage=float(labels.agent_future_mask.mean()),ObservationDigest=before,
            ObservationCache=str((folder/(stem+'_observation.npz')).relative_to(ROOT)),EvaluationOnlyCache=str((folder/(stem+'_evaluation_only.npz')).relative_to(ROOT)),
            IdentityCache=str((folder/(stem+'_identity.json')).relative_to(ROOT)),MapCache=str((ROOT/'05_map_alignment/cache'/f'{stem}_map.json').relative_to(ROOT))))
        poisonrows.append(dict(SampleOrdinal=ordinal,EvaluationLabelsNaNAndMaskPoison='PASS',FuturePlaceholderModelPoison='PASS',SourceFutureMetadataPoison=rawpoison,PredictionMaxDiff=0.,PlanningMaxDiff=0.))
        print('STAGE13A_SAMPLE_PASS',ordinal,token,'actors',len(obs['instance_tokens']),'eligible',int(output['valid'].sum()),'oldOOF',historical_found,'futureGT',bool(labels.ego_future_mask.all()),flush=True)
    forecaster.verify_frozen();con.close();assert len(selected)==48 and labels_count==36 and matching_history==36 and roundtrip<1e-5
    dump('06_planning_interface/stage13a_planning_samples_manifest.csv',selected);dump('03_prediction_alignment/stage13a_prediction_join.csv',joined)
    dump('04_observation_only_inference/stage13a_small_prediction_coverage.csv',coverage);dump('02_ego_trajectory/stage13a_timestamp_alignment.csv',timestamps)
    dump('07_evaluation_preflight/stage13a_evaluation_feasibility.csv',evaluations);dump('06_planning_interface/stage13a_gt_poison_details.csv',poisonrows)
    atomic_json(ROOT/'04_observation_only_inference/stage13a_observation_only_audit.json',dict(Status='PASS',ObservationOnlyInference='PASS',Scenes=12,PlanningSamples=48,InferenceOnlyMissingFutureSamples=12,
        HistoricalGraphHistoryBitwiseMatchedSamples=matching_history,PastOnlyFacadeBlocksFutureReads=True,ActorSelection='present supported,historycount>=2; no future labels',FrozenAllModelsEval=True,FrozenParameterGradients=0,FrozenStatesUnchanged=True,
        OriginalGraphNumericalDefinitionReused=True,CAndR2FoldIdentity=True,TemperatureUnchanged=True,BicycleR2ProbabilityDirectCopy=True,AllCurrentActorsRetained=True,NoTraining=True))
    atomic_json(ROOT/'06_planning_interface/stage13a_gt_poison_audit.json',dict(Status='PASS',GTLeakage='PASS',PlanningGTIsolation='PASS',Samples=48,RawFutureMetadataPoisonSamples=12,
        EvaluationLabelsNaNAndMasksPoisoned=True,FutureGraphPositionsAndMasksPoisoned=True,ObservationIdentitiesAndFeaturesUnchanged=True,PredictionMaxDiff=0.,PlanningMaxDiff=0.,
        CVInputSignature='PlanningObservation only; full PlanningSample and EvaluationLabels rejected'))
    atomic_json(ROOT/'02_ego_trajectory/stage13a_coordinate_audit.json',dict(Status='PASS',CoordinateAlignment='PASS',EgoAndActorFP64RoundtripMaxM=roundtrip,
        MapFP64RoundtripMaxM=maps.roundtrip,ThresholdM=1e-5,Coordinates='reuse preprocessing.coordinates',ForwardAxis=[1,0],LeftAxis=[0,1],YawUnits='radians counterclockwise',DistanceUnits='meters'))
    atomic_json(ROOT/'05_map_alignment/stage13a_map_alignment.json',dict(Status='PASS',MapCompatibility='PASS',Samples=48,Regions=4,MapDrivableMaskShape=[301,301],RasterSpacingM=1,
        PatchRadiusM=150,BoundaryAndLaneGeometryPresent=True,SourceHashes=read_json(S12A/'01_map_integrity/stage12a_map_integrity.json')['MapJSONSHA256'],RoundtripMaxM=maps.roundtrip,
        RouteInferredFromFutureGT=False,ExistingInvalidSourceComponentsExcluded=maps.index.invalid_components))
    atomic_json(ROOT/'06_planning_interface/stage13a_small_preflight.json',dict(Status='PASS',PlanningInterface='PASS',EgoTrajectoryExtraction='PASS',PredictionIdentityJoin='PASS',
        PlanningSamples=48,CompleteOpenLoopEgoSamples=36,InferenceOnlySamples=12,MatchedOOFActors=sum(r['HistoricalOOFJoined'] for r in selected),
        OriginalPoseTimestampVerifiedFrameOccurrences=pose_known,NumericUnitAudit='PASS',ProceedFull630MetadataAudit=True))
    verify();print('STAGE13A_SMALL_ALL_PASS',flush=True)
if __name__=='__main__':main()
