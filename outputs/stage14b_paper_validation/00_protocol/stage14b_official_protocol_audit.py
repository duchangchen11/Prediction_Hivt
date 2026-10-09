"""Read-only local devkit/target-manifest compatibility audit; no evaluation."""
from pathlib import Path
import hashlib,json,sys,datetime
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parents[1]
from nuscenes.utils.splits import create_splits_scenes
DEV=Path(sys.prefix)/'lib/python3.10/site-packages/nuscenes/eval/prediction'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
sources={str(p):sha(p) for p in [DEV/'splits.py',DEV/'metrics.py',DEV/'data_classes.py',DEV/'configs/predict_2020_icra.json']}
paths=[PROJECT/'outputs/stage2/trainval/cache/maps/prediction/prediction_scenes.json',Path('/media/lrj/54926A1D926A0438/nuscenes-trainval/maps/prediction/prediction_scenes.json'),Path('/home/lrj/datasets/nuscenes-mini/maps/prediction/prediction_scenes.json')]
splits=create_splits_scenes()
out=dict(Status='PARTIAL_NEEDS_ADAPTER_AND_PROTOCOL_RESET',CheckedUTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    LocalDevkitSourceSHA256=sources,ChallengeTrainScenes=len(splits['train'][200:]),ChallengeTrainValScenes=len(splits['train'][:200]),ChallengeValScenes=len(splits['val']),
    ManifestChecks=[dict(Path=str(p),Exists=p.exists()) for p in paths],FullTrainValTargetManifestVerified=any(p.exists() for p in paths[:2]),
    CurrentProtocol='custom HeadTrain630 ranking OOF, all full-horizon V/P/B targets',
    HorizonCompatible=True,HistorySeconds=2,PredictionSeconds=6,Timesteps=12,NumModes=6,ModeLimit=25,
    DefaultOfficialK=[1,5,10],ActualSixModeSupportsK=[1,5,6],Genuine10ModesAvailable=False,
    OfficialMiss='min over probability top-k of max_t L2 >=2m',ProjectMR6='min_k endpoint L2 >2m',MissCompatible=False,
    CoordinatesNeedGlobalAdapter=True,TargetUniverseIdentical=False,OfficialSplitIdentical=False,
    HiddenTestAnnotationsVerifiedAvailable=False,CurrentDataPristine=False,
    CurrentPredictionServer='VAL leaderboard; official code review/run on hidden TEST for top entries',
    PrimarySources=[
      'https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md',
      'https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py',
      'https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/metrics.py',
      'https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/configs/predict_2020_icra.json'])
p=ROOT/'00_protocol/stage14b_official_protocol_audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n')
print('OFFICIAL_PROTOCOL_AUDIT',out['Status'],'full manifest available=',out['FullTrainValTargetManifestVerified'])
