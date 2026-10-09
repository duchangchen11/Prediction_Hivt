"""Full630 metadata eligibility audit, without full630 frozen-model inference."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage13a_common import *
def main():
    seed();verify();assert read_json(ROOT/'06_planning_interface/stage13a_small_preflight.json')['ProceedFull630MetadataAudit']
    con=connection();scenes=scene_folds();extras,originalposes=original_extras();stats=[];actorrows=[];dts=[];offsets=[];poseoffsets=[];pose_missing=0;validpose=0;coordinate_max=0.;historical_matched=0;full_expected=0
    f=pd.read_csv(S11B/'05_oof_evaluation/cache/stage11b_oof_predictions_actor_records.csv');keys=set(zip(f.scene_token,f.sample_token,f.instance_token));assert len(keys)==260151
    calibrations={r['token']:r for r in read_json(RAW/'v1.0-trainval/calibrated_sensor.json')};sensors={r['token']:r for r in read_json(RAW/'v1.0-trainval/sensor.json')}
    source_timestamps=[];current_counts=np.zeros(3,np.int64);eligible_counts=np.zeros(3,np.int64);original_counts=np.zeros(3,np.int64);missing_counts=np.zeros(3,np.int64);size_known_counts=np.zeros(3,np.int64)
    for ordinal,token in enumerate(sorted(scenes),1):
        db=load_scene(con,token);chain=scene_samples(db,db.scene[0]);mapping=category_mapping(db);region=db.get('log',db.scene[0]['log_token'])['location'];map_available=(MAP_ROOT/'maps/expansion'/f'{region}.json').exists()
        anns=[annotation_index(db,s) for s in chain];pose_positions=[];yaws=[];lidar=[];pt=[];poseknown=[]
        for i,s in enumerate(chain):
            sd=db.get('sample_data',s['data']['LIDAR_TOP']);p=db.get('ego_pose',sd['ego_pose_token']);assert sd['is_key_frame'] and sd['sample_token']==s['token'] and p['token']==sd['ego_pose_token']
            cal=calibrations.get(sd['calibrated_sensor_token']);assert cal is not None and sensors[cal['sensor_token']]['channel']=='LIDAR_TOP'
            pose_positions.append(p['translation'][:2]);yaws.append(quaternion_yaw(p['rotation']));lidar.append(int(sd['timestamp']));offsets.append(int(s['timestamp']-sd['timestamp']))
            original=originalposes.get(p['token']);known=original is not None;poseknown.append(known)
            if known:
                assert np.array_equal(original['translation'],p['translation']) and np.array_equal(original['rotation'],p['rotation']);poseoffsets.append(int(original['timestamp']-sd['timestamp']));validpose+=1;pt.append(int(original['timestamp']))
                source_timestamps.append(dict(SceneToken=token,SampleToken=s['token'],EgoPoseToken=p['token'],SampleTimestampUS=int(s['timestamp']),LidarTimestampUS=int(sd['timestamp']),
                    EgoPoseTimestampUS=int(original['timestamp']),SampleMinusLidarUS=int(s['timestamp']-sd['timestamp']),PoseMinusLidarUS=int(original['timestamp']-sd['timestamp'])))
            else:pose_missing+=1;pt.append(None)
        stamps=np.array([s['timestamp'] for s in chain],np.int64);dt=np.diff(stamps)/1e6;dts.extend(dt.tolist());assert (dt>0).all()
        positions=np.asarray(pose_positions,np.float64);yaws=np.asarray(yaws,np.float64);assert np.isfinite(positions).all() and np.isfinite(yaws).all()
        sceneeligible=0
        for i,s in enumerate(chain):
            hist=i>=4;future=len(chain)-i-1;full=future>=12
            local=global_to_ego(positions[max(0,i-4):min(len(chain),i+13)],positions[i],yaws[i]);coordinate_max=max(coordinate_max,float(np.abs(ego_to_global(local,positions[i],yaws[i])-positions[max(0,i-4):min(len(chain),i+13)]).max()))
            counts=np.zeros(3,np.int64);eligible=np.zeros(3,np.int64);old=np.zeros(3,np.int64);complete=np.zeros(3,np.int64);sizes=np.zeros(3,np.int64)
            for instance,ann in anns[i].items():
                if ann['category_name'] not in mapping:continue
                typ=mapping[ann['category_name']];counts[typ]+=1;history_count=sum(instance in a for a in anns[max(0,i-4):i+1]);supported=bool(hist and history_count>=2)
                eligible[typ]+=supported;joined=(token,s['token'],instance) in keys;old[typ]+=joined;sizes[typ]+=ann['token'] in extras
                future_complete=full and all(instance in a for a in anns[i+1:i+13]);complete[typ]+=supported and future_complete
                expected=bool(supported and future_complete);assert joined==expected,(token,s['token'],instance,'historical OOF metadata membership')
                historical_matched+=joined;full_expected+=expected
            if hist:current_counts+=counts;eligible_counts+=eligible;original_counts+=old;missing_counts+=counts-eligible;size_known_counts+=sizes;sceneeligible+=int(eligible.sum())
            stats.append(dict(SceneToken=token,SceneName=db.scene[0]['name'],SampleToken=s['token'],T0Index=i,Fold=scenes[token],Region=region,
                T0SampleTimestampUS=int(stamps[i]),T0LidarTimestampUS=int(lidar[i]),SampleMinusLidarUS=int(stamps[i]-lidar[i]),EgoPoseTimestampVerified=poseknown[i],
                EgoHistoryComplete=hist,EgoFutureGTAny=future>0,EgoFutureGTComplete=full,AvailableFutureFrames=min(future,12),MapAvailable=map_available,
                CurrentSupportedActors=int(counts.sum()),MetadataPredictionEligible=int(eligible.sum()),MissingHistoryActors=int(counts.sum()-eligible.sum()),
                VehicleCurrent=int(counts[0]),PedestrianCurrent=int(counts[1]),BicycleCurrent=int(counts[2]),VehicleEligible=int(eligible[0]),PedestrianEligible=int(eligible[1]),BicycleEligible=int(eligible[2]),
                HistoricalOOFActors=int(old.sum()),FutureIncompleteEligible=int(eligible.sum()-complete.sum()),
                EgoObservationInputValid=hist and map_available,OpenLoopEgoEvaluationValid=hist and full and map_available,
                CompleteSourcePoseTimestampVerificationForHorizon=bool(all(poseknown[max(0,i-4):min(len(chain),i+13)]))))
        actorrows.append(dict(SceneToken=token,Fold=scenes[token],Region=region,MetadataForecastEligibleOccurrences=sceneeligible,NoForecastEligibleActors=sceneeligible==0))
        if ordinal%50==0:print('METADATA_SCENE_AUDIT',ordinal,'/630','windows',len(stats),'oldmatched',historical_matched,flush=True)
    con.close();assert historical_matched==full_expected==260151 and coordinate_max<1e-5
    folder=ROOT/'01_dataset_audit/cache';folder.mkdir(exist_ok=True);pd.DataFrame(stats).to_csv(folder/'stage13a_all_windows.csv',index=False,float_format='%.17g')
    dump('01_dataset_audit/stage13a_scene_coverage.csv',actorrows);coverage=[]
    for typ,name in enumerate(TYPES):coverage.append(dict(AgentType=name,Denominator='all supported actor occurrences at t0 with5 ego history frames, including final12 scene anchors',
        CurrentActors=int(current_counts[typ]),HistoryEligibleActors=int(eligible_counts[typ]),MetadataPredictionCoverage=float(eligible_counts[typ]/current_counts[typ]),
        MissingHistoryActors=int(missing_counts[typ]),HistoricalOOFActors=int(original_counts[typ]),EligibleButAbsentOOF=int(eligible_counts[typ]-original_counts[typ]),
        CurrentOriginalDimensionAvailable=int(size_known_counts[typ]),ActualFrozenModelReplayScope='only fixed12scenes/48windows; full630 counts are metadata eligibility, not all630 forward validation'))
    dump('03_prediction_alignment/stage13a_prediction_coverage.csv',coverage);windows=pd.DataFrame(stats);hist=windows.EgoHistoryComplete
    audit=dict(Status='PASS',Scenes=630,TotalT0Windows=len(windows),CompleteEgoHistoryWindows=int(hist.sum()),AnyEgoFutureGTWindows=int(windows.EgoFutureGTAny.sum()),
        CompleteEgoFutureGTWindows=int(windows.EgoFutureGTComplete.sum()),ObservationInputWindows=int(windows.EgoObservationInputValid.sum()),
        OpenLoopEgoEvaluationWindows=int(windows.OpenLoopEgoEvaluationValid.sum()),HistoryAndAnyPredictionWindows=int((hist&(windows.MetadataPredictionEligible>0)).sum()),
        VehiclePredictionWindows=int((hist&(windows.VehicleEligible>0)).sum()),PedestrianPredictionWindows=int((hist&(windows.PedestrianEligible>0)).sum()),BicyclePredictionWindows=int((hist&(windows.BicycleEligible>0)).sum()),
        MapAvailableWindows=int(windows.MapAvailable.sum()),NoPredictableActorsSceneFraction=float(np.mean([r['NoForecastEligibleActors'] for r in actorrows])),
        NoPredictableActorsHistoryWindowFraction=float((windows.loc[hist,'MetadataPredictionEligible']==0).mean()),HistoricalOOFActorCount=historical_matched,
        EligibilityIndependentOfFuture=True,ObservationAndEvaluationDenominatorsSeparated=True,All630FrozenForwardReplayed=False,
        ConfiguredRawVolume='/media/lrj/54926A1D926A0438/nuscenes-trainval unavailable',MetadataSource='unchanged full Stage2C read-only SQLite, official850scenes/34149samples; onlyHeadTrain630 processed',
        OriginalJSONScenesAvailable=len(read_json(RAW/'v1.0-trainval/scene.json')),CurrentOriginalPoseTimestampKnownKeyframes=validpose,CurrentOriginalPoseTimestampUnknownKeyframes=pose_missing,
        CoordinateFP64MaxErrorM=coordinate_max,SceneSplitIsolation='PASS',OfficialVALTestModelSelection=False,NoIndependentPerformanceClaim=True,
        RouteInformation='UNAVAILABLE',RouteAudit='sample/ego_pose/log/HD Map and current metadata contain no supplied route/goal/high-level navigation command; map lane connectivity is not an intended route',
        CacheSHA256=sha256(folder/'stage13a_all_windows.csv'),DatasetIndexManifestSHA256=sha256(S2C/'02_preprocessed/stage2c_metadata_cache/stage2c_metadata_index_manifest.json'))
    atomic_json(ROOT/'01_dataset_audit/stage13a_dataset_audit.json',audit)
    dump('02_ego_trajectory/stage13a_source_pose_timestamp_checks.csv',source_timestamps)
    atomic_json(ROOT/'02_ego_trajectory/stage13a_ego_pose_audit.json',dict(Status='PASS_WITH_SOURCE_LIMIT',EgoTrajectoryExtraction='PASS',EgoPoseTokenAndLidarKeyframeIdentity='PASS',
        OriginalPoseTimestampVerification='PARTIAL',KnownKeyframes=validpose,UnknownKeyframes=pose_missing,PoseMinusLidarUS=distribution(poseoffsets),
        SourceFieldsDroppedByHistoricalIndex=['ego_pose.timestamp','sample_annotation.size'],NoPoseTimestampImputation=True,NoNewRawDataCopy=True))
    atomic_json(ROOT/'02_ego_trajectory/stage13a_time_audit.json',dict(Status='PASS',TimestampAlignment='PASS',SampleAndForecastKeyframeIdentity='PASS',SampleMinusLidarUS=distribution(offsets),
        SamplingIntervalSeconds=distribution(dts),NoForcedHalfSecondResampling=True,FutureQueryTimes='exact recorded sample schedule supplied externally for replay; nominal schedule only for no-future demonstration',
        PoseTimestampSourceVerification='PARTIAL',AcquisitionAlignment='sample vs LIDAR offset explicitly reported; no unrecorded correction',FrameCoordinateDefinition='pose at matching LIDAR_TOP keyframe'))
    print('STAGE13A_METADATA_ALL630_PASS',json.dumps(audit),flush=True)
if __name__=='__main__':main()
