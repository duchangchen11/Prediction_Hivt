"""Recompute only offline proxies from saved observations and separate labels."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / p) for p in ('00_manifest','06_planning_interface','07_evaluation_preflight')]
from stage13a_common import *
from stage13a_artifacts import load_observation, load_evaluation_only
from stage13a_metrics import evaluate


def main():
    samples = pd.read_csv(ROOT/'06_planning_interface/stage13a_planning_samples_manifest.csv')
    previous = pd.read_csv(ROOT/'07_evaluation_preflight/stage13a_evaluation_feasibility.csv')
    rows=[]
    for sample in samples.itertuples():
        obs,plan,_=load_observation(sample)
        labels=load_evaluation_only(sample)
        plans=[('CV_Ego_EngineeringReference',plan)]
        if labels.ego_future_mask.all():
            plans.append(('RecordedEgoFuture_EvaluationOnly',labels.ego_future_gt))
        for reference,path in plans:
            row,_,_=evaluate(obs,labels,path,reference)
            row.update(SampleOrdinal=sample.SampleOrdinal,SceneToken=sample.SceneToken,SampleToken=sample.SampleToken,Role=sample.Role)
            rows.append(row)
    result=pd.DataFrame(rows)
    old=previous[previous.Reference=='CV_Ego_EngineeringReference'].sort_values('SampleOrdinal')
    new=result[result.Reference=='CV_Ego_EngineeringReference'].sort_values('SampleOrdinal')
    assert np.array_equal(old.PredictedAgentCollisionProxy,new.PredictedAgentCollisionProxy)
    assert np.array_equal(old.PredictedTop1CollisionProxy,new.PredictedTop1CollisionProxy)
    assert np.allclose(old.DrivableAreaViolationFraction,new.DrivableAreaViolationFraction,atol=1e-15,equal_nan=True)
    assert len(result)==84
    dump('07_evaluation_preflight/stage13a_evaluation_feasibility.csv',rows)
    atomic_json(ROOT/'07_evaluation_preflight/stage13a_evaluation_replay_audit.json',dict(Status='PASS',Rows=84,
        FrozenForecastsReplayed=False,CVProxyOutputsUnchanged=True,
        RecordedEgoOrientation='same EvaluationOnly pose heading used for predicted-agent, GT-agent and drivable footprint checks',
        Source='existing separately saved observation and EvaluationOnly labels; no training/forward/model selection'))
    print('STAGE13A_OFFLINE_EVALUATION_REPLAY_PASS',flush=True)


if __name__=='__main__':
    main()
