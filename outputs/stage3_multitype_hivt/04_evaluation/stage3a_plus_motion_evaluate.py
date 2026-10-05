"""Motion subsets reuse final-best saved VAL errors; six case trajectories reproduce those VAL batches."""
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3a_plus_common import (CLASSES, SceneDataset, config, read_json, atomic_json, write_csv,
                                sha256, verify_freeze, final_checkpoint)
from stage3_common import model_new, model_input, errors_with_top1
import numpy as np
import torch
from torch_geometric.data import Batch

FIELDS = ('minADE6', 'minFDE6', 'MR6', 'Top1ADE6', 'Top1FDE6')


@torch.no_grad()
def main():
    verify_freeze(); checkpoint, manifest = final_checkpoint()
    actor_path = ROOT / manifest['final_predictions_relative_path']
    assert sha256(actor_path) == manifest['final_predictions_sha256']
    with actor_path.open() as f: actors = list(csv.DictReader(f))
    assert len(actors) == 85027
    assert len({(r['scene_token'], r['sample_token'], r['instance_token'], r['horizon']) for r in actors}) == len(actors)
    for row in actors:
        assert all(np.isfinite(float(row[k])) for k in FIELDS + ('GT_endpoint_displacement_m',))
    full = [r for r in actors if r['horizon'] == 'full_horizon']
    assert len(full) == 54990
    results = []
    for cls, thresholds in (('vehicle', (5,)), ('pedestrian', (1, 5)), ('bicycle', (1, 5))):
        for threshold in thresholds:
            selected = [r for r in full if r['agent_type'] == cls and float(r['GT_endpoint_displacement_m']) > threshold]
            assert selected, (cls, threshold)
            results.append({'Group': cls.capitalize() + f' >{threshold}m', 'AgentType': cls,
                            'GT_endpoint_displacement_gt_m': threshold, 'Count': len(selected),
                            'UniqueInstances': len({r['instance_token'] for r in selected}),
                            'UniqueScenes': len({r['scene_token'] for r in selected}),
                            **{key: float(np.mean([float(r[key]) for r in selected])) for key in FIELDS}})
    write_csv(ROOT / '06_tables/stage3_nontrivial_motion_metrics.csv', results)
    atomic_json(ROOT / '04_evaluation/stage3_nontrivial_motion_metrics.json', {
        'status': 'PASS', 'checkpoint_relative_path': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha256(checkpoint),
        'source_actor_errors_relative_path': str(actor_path.relative_to(ROOT)), 'source_actor_errors_sha256': sha256(actor_path),
        'predictions_reused_without_training_or_reinference_for_subgroup_metrics': True,
        'split': 'official val150', 'horizon': 'full 12-step future only', 'GT_displacement': 'norm(GT[t12]-position[t0]), strict > thresholds',
        'metric_definition': 'minADE6 is ADE of best-FDE mode; MR6 endpoint error>2m; Top1 argmax saved model probability',
        'metrics': results, 'full_horizon_target_count': len(full), 'excluded_partial_future_count': len(actors) - len(full), 'test_used': False})
    for row in results: print('NONTRIVIAL_MOTION', row, flush=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['config'] == config()
    model = model_new(); model.load_state_dict(saved['state_dict']); model.eval()
    ds = SceneDataset('val'); lookup = {(r['scene_token'], r['sample_token']): i for i, r in enumerate(ds.rows)}
    case_manifest = []; bs = config()['batch_size']
    for cls in CLASSES:
        eligible = [r for r in full if r['agent_type'] == cls and float(r['GT_endpoint_displacement_m']) > 5
                    and (cls != 'vehicle' or r['motion_state'] == 'vehicle.moving')]
        assert len(eligible) >= 2
        rank_key = lambda r: (float(r['minFDE6']), r['scene_token'], r['sample_token'], r['instance_token'])
        success = min(eligible, key=rank_key)
        failure = max((r for r in eligible if r['instance_token'] != success['instance_token']), key=rank_key)
        panels = []
        for kind, row in (('success', success), ('failure', failure)):
            index = lookup[row['scene_token'], row['sample_token']]; graph = ds[index]; node = int(row['node_in_graph'])
            assert graph.instance_tokens[node] == row['instance_token'] and graph.future_mask[node].all()
            assert int(graph.agent_type[node]) == CLASSES.index(cls)
            start = index // bs * bs
            graphs = [ds[i] for i in range(start, min(start + bs, len(ds)))]
            data = Batch.from_data_list(graphs).cuda(); output_node = int(data.ptr[index - start]) + node
            out = model(model_input(data)); predictions, errors = errors_with_top1(model, out, data)
            trajectory = predictions[output_node].cpu().numpy()
            gt = graph.positions[node, 5:].numpy(); origin = graph.positions[node, 4].numpy()
            b, t = int(errors['best_mode'][output_node]), int(errors['top1_mode'][output_node])
            be = np.linalg.norm(trajectory[b].astype(np.float64) - gt, axis=1)
            te = np.linalg.norm(trajectory[t].astype(np.float64) - gt, axis=1)
            actual = dict(zip(('minADE6', 'minFDE6', 'Top1ADE6', 'Top1FDE6'), map(float, (be.mean(), be[-1], te.mean(), te[-1]))))
            delta = {key: abs(v - float(row[key])) for key, v in actual.items()}
            assert max(delta.values()) < 1e-4, (cls, kind, delta)
            displacement = float(np.linalg.norm(gt[-1].astype(np.float64) - origin))
            assert displacement > 5 and abs(displacement - float(row['GT_endpoint_displacement_m'])) < 1e-4
            panel = {**row, 'kind': kind, 'history_trajectory_m': graph.positions[node, :5].tolist(),
                     'history_mask': graph.history_mask[node].tolist(), 'GT_trajectory_m': gt.tolist(),
                     'future_mask': graph.future_mask[node].tolist(), 'HiVT_trajectories_m': trajectory.tolist(),
                     'mode_probabilities': out['mode_prob'][output_node].cpu().tolist(),
                     'best_FDE_mode_zero_based': b, 'top1_mode_zero_based': t,
                     'numeric_metrics': {key: float(row[key]) for key in FIELDS},
                     'metric_absolute_differences_m': delta, 'lane_positions_m': graph.lane_positions.tolist(),
                     'lane_vectors_m': graph.lane_vectors.tolist(), 'origin_global_m': graph.origin.tolist(),
                     'ego_yaw_global_rad': float(graph.ego_yaw), 'coordinate_frame': 't0 ego +x forward, +y left; meters',
                     'history_times_seconds': graph.history_times.tolist(), 'future_times_seconds': graph.future_times.tolist(),
                     'reproduced_VAL_batch_start_index': start, 'reproduced_VAL_batch_size': len(graphs),
                     'checkpoint_sha256': sha256(checkpoint), 'GT_endpoint_displacement_recomputed_m': displacement}
            panels.append(panel)
        name = f'stage3a_plus_{cls}_motion_case'
        path = ROOT / '04_evaluation/cases' / (name + '.json')
        atomic_json(path, {'agent_type': cls, 'panels': panels, 'checkpoint_sha256': sha256(checkpoint),
                           'selection': 'lowest/highest minFDE among full-horizon GT displacement>5m; vehicle requires t0 moving; distinct instances',
                           'eligible_actor_windows': len(eligible), 'trajectory_edits': False, 'smoothing': False})
        case_manifest.append({'name': name, 'agent_type': cls, 'source_json': str(path.relative_to(ROOT)), 'source_sha256': sha256(path)})
        print('MOTION_CASE_EXPORT', cls, [(p['kind'], p['minFDE6'], p['Top1FDE6']) for p in panels], flush=True)
    atomic_json(ROOT / '04_evaluation/stage3a_plus_motion_case_manifest.json', {'status': 'PASS', 'figures': case_manifest,
                                                                              'case_count': 6, 'source_checkpoint_sha256': sha256(checkpoint)})
    verify_freeze(); print('MOTION_EVALUATION_AND_CASES=PASS', flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4); main()
