"""Reload stage-isolated observation artifacts; labels require a separate call."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / s) for s in ('00_manifest', '06_planning_interface')]
from stage13a_common import read_json, np
from stage13a_interface import PlanningObservation, PlanningEvaluationLabels
import shapely


def load_observation(row):
    arrays = np.load(ROOT / row.ObservationCache)
    identity = read_json(ROOT / row.IdentityCache)
    mapping = read_json(ROOT / row.MapCache)
    raster = np.load((ROOT / row.MapCache).with_suffix('.npz'))
    drivable = shapely.from_wkb(bytes.fromhex(mapping['DrivableWKB']))
    shapely.prepare(drivable)
    geometry = dict(
        drivable=drivable, MapBoundaryGeometry=drivable.boundary,
        MapLaneGeometry=[np.array(p) for p in mapping['LaneGeometry']],
        crossing_parts=[shapely.from_wkb(bytes.fromhex(p)) for p in mapping['CrossingWKB']],
        walkway_parts=[shapely.from_wkb(bytes.fromhex(p)) for p in mapping['WalkwayWKB']],
        patch_bounds=np.array(mapping['PatchBounds']), available=not drivable.is_empty,
        **{k: raster[k] for k in raster.files},
    )
    observation = PlanningObservation(
        scene_token=identity['SceneToken'], sample_token=identity['SampleToken'],
        fold=identity['Fold'], t0_timestamp=identity['T0Timestamp'],
        ego_history=arrays['ego_history'], ego_history_heading=arrays['ego_history_heading'],
        history_times=arrays['history_times'], future_times=arrays['future_times'],
        instance_tokens=tuple(identity['InstanceTokens']),
        other_agent_predictions=arrays['other_agent_predictions'],
        other_agent_probabilities=arrays['other_agent_probabilities'],
        other_agent_raw_probabilities=arrays['other_agent_raw_probabilities'],
        other_agent_type=arrays['agent_type'], other_agent_current_position=arrays['current_position'],
        other_agent_current_heading=arrays['current_heading'],
        other_agent_sizes_length_width=arrays['actor_size_length_width'],
        other_agent_size_source=arrays['actor_size_source'], prediction_valid_mask=arrays['prediction_valid_mask'],
        other_agent_history=arrays['history'], other_agent_history_mask=arrays['history_mask'],
        map_geometry=geometry, model_identity=identity['ModelIdentity'],
    ).validate()
    return observation, arrays['cv_ego_reference'], identity


def load_evaluation_only(row):
    arrays = np.load(ROOT / row.EvaluationOnlyCache)
    return PlanningEvaluationLabels(**{k: arrays[k] for k in arrays.files})
