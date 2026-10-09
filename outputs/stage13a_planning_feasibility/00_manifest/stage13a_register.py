"""Freeze history and engineering definitions before any planning demonstration."""
from stage13a_common import *
def main():
    assert git('rev-parse','HEAD')==BASE and not PROTOCOL.exists();old=read_json(S12B/'00_manifest/stage12b_frozen_history.json')
    cps=dict(old['checkpoints']);cps.update({str((S12B/p).relative_to(PROJECT)):h for p,h in read_json(S12B/'05_checkpoints/stage12b_all9_frozen.json')['Checkpoints'].items()})
    historical={p:sha256(PROJECT/p) for p in git('ls-tree','-r','--name-only',BASE).splitlines()}
    preserved={p:sha256(PROJECT/p) for p in git('ls-files','--others','--exclude-standard').splitlines() if not p.startswith(str(ROOT.relative_to(PROJECT))+'/')}
    sources=[DB,S2C/'02_preprocessed/stage2c_metadata_cache/stage2c_metadata_index_manifest.json',*sorted((RAW/'v1.0-trainval').glob('*.json')),
        S11B/'05_oof_evaluation/cache/stage11b_oof_predictions_actor_records.csv',S11B/'05_oof_evaluation/cache/stage11b_oof_predictions_logits.npy',S11B/'05_oof_evaluation/cache/stage11b_oof_predictions_probabilities.npy',
        S11A/'01_identity_audit/cache/candidates.npy',S12A/'01_map_integrity/cache/stage12a_origin.npy',S12A/'01_map_integrity/cache/stage12a_yaw.npy',
        *sorted((MAP_ROOT/'maps/expansion').glob('*.json'))]
    data={str(p.relative_to(PROJECT)):sha256(p) for p in sources}
    freeze=dict(BaseCommit=BASE,historical_files=historical,preserved_untracked=preserved,checkpoints=cps,splits_norm_temperatures=old['split_normalization_temperature'],data_files=data)
    atomic_json(FREEZE,freeze)
    protocol=dict(Stage='Stage13A',Branch='stage13a/prediction-to-ego-planning-audit',BaseCommit=BASE,NoTraining=True,NoOptimizers=True,
        HistorySteps=5,FutureSteps=12,Modes=6,Coordinates='existing FP64 global_to_ego/ego_to_global: t0 origin, +x forward,+y left,yaw counterclockwise radians',
        PredictionEligibility='supported and present at t0, at least2 historical observations among5; independent of any future GT/mask/displacement/error',
        OtherActors='retain every supported t0 actor, including insufficient-history rows with PredictionValidMask0 and recorded unknown-risk status; all retained as original model context',
        SceneSelection='three distinct lexicographic scenes per region: largest observable Vehicle count, largest observable Pedestrian count, first Bicycle-present remaining scene if available; density ties scene_token. Regions sorted; only HeadTrain630. No prediction/error/GT trajectory selection.',
        Windows='per selected scene first/middle/last anchor with5 history and12 scheduled future keyframes, plus finalscene keyframe as inference-only missing-future example; fixed before predictor replay',
        FutureTime='explicit query schedule from recorded sample timestamps for replay; planner/forecaster never read future poses or annotations. Inference-only final anchor uses externally requested nominal0.5s grid, clearly distinguished. No asynchronous pose resampling.',
        InferenceModes=dict(A='historical OOF identity/frame/keyframe alignment only, future-complete evaluated subset',B='past-only metadata facade, neutral unused future placeholders, frozen Stage5A + corresponding-fold C/R2 on allcurrent contexts'),
        Probability='C raw logits and original raw probability retained; Vehicle/Pedestrian additionally scale by existing foldT, Bicycle directly routesR2 original probabilities; neither distribution is actual collision probability',
        Map=dict(RadiusMeters=150,RasterResolutionMeters=1,Source='unchanged original local HD Map; no semantic training or route extraction from future GT',OutOfPatch='evaluation mask unknown rather than offroad'),
        Collision=dict(Footprint='oriented rectangles; current annotation width/length when available; otherwise explicit class prior',EgoLengthWidth=[4.8,2.0],
            MissingActorLengthWidth={'Vehicle':[4.5,1.8],'Pedestrian':[.6,.6],'Bicycle':[1.8,.6]},InterpolationMaxStepSeconds=.05,
            Heading='prediction tangent, retain observed heading for zero displacement; interpolate wrapped heading',
            GTMask='both segment endpoints observed, including t0; never treat missing future as empty scene',Score='uncalibrated mode-weighted overlap indicator/expected impacted actors, not actual collision probability'),
        References=['Recorded Ego Future:EvaluationOnly','CV from last2 observed ego poses and their true elapsedseconds; externally supplied forecast schedule'],
        CaseSelection='within fixed12 scenes: up to3 distinct vehicle-rich>=10,3 pedestrian-rich>=5,3 forecast-CV footprint-proxy conflict scenes, then ordinary; include final-anchor missing-future cases. No selection by GT success/ADE/FDE.',
        RouteInformation='audit metadata for actual route/command; no GT endpoint as navigation input',
        Decision='GO requires all required engineering gates and observation-only inference; CONDITIONAL_GO for usable geometry/interface but missing source verification, route or evaluation coverage limitations; STOP on unextractable ego GT or broken identity/time/frame or inputGTdependency',
        DataLimit='configured original trainval volume currently unmounted; complete frozen SQLite, subset rawJSON and unchanged HD Map reused; source ego_pose timestamps/dimensions audited as available/unknown, never invented',
        FrozenHistorySHA256=sha256(FREEZE),RequirementsSHA256=sha256(ROOT/'00_manifest/stage13a_requirements.txt'),
        AfterCompletion='push ownbranch without merge then STOP; no Stage13B/complexplanner/training/officialVAL/test',Evidence='engineering feasibility, nonreactive openloop, not independent generalization or closedloop safety')
    atomic_json(PROTOCOL,protocol);verify(history=True,data=True);print('STAGE13A_REGISTERED',len(historical),'historical files',len(cps),'checkpoints',flush=True)
if __name__=='__main__':main()
