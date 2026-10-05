"""Class gradients at two frozen checkpoints on 100 identical TRAIN batches.

Full scene-window context is retained. The existing original-NLL recovery_loss
is called on one shared forward with only class target masks changed. No loss
formula, optimizer, update, or training loop is introduced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage4ae_common import (ROOT, STAGE3, DIAG, BEST_B, CHECKPOINT_B, CHECKPOINT_C,
    SceneDataset, seed_all, atomic_json, read_json, write_csv, sha256, state_digest,
    verify_frozen, model_input)
from stage4a_evaluate import load_final_model
from stage3b_common import model_new as model_b_new
import torch
from torch_geometric.loader import DataLoader

GROUPS = ('LocalEncoder', 'TypeEmbedding', 'GlobalInteractor', 'Decoder', 'ALL SHARED')
EPS = 1e-12


def load_models():
    summary = read_json(STAGE3 / '03_type_embedding/stage3b_training_summary.json')
    assert summary['status'] == 'COMPLETE' and summary['checkpoint_sha256'] == CHECKPOINT_B
    assert sha256(BEST_B) == CHECKPOINT_B
    saved = torch.load(BEST_B, map_location='cpu', weights_only=False)
    model_b = model_b_new()
    model_b.load_state_dict(saved['state_dict'], strict=True)
    assert state_digest(model_b.state_dict()) == summary['model_state_content_sha256']
    del saved
    model_c, saved, _ = load_final_model()
    del saved
    for model in (model_b, model_c):
        model.eval()
        assert all(not m.training for m in model.modules())
        assert all(p.requires_grad and p.grad is None for p in model.parameters())
    return {'Stage3B': model_b, 'Stage4A': model_c}


def inventory(model):
    names, parameters = zip(*model.named_parameters())
    groups = {name: [] for name in GROUPS[:4]}
    if any('relation_mlp' in name for name in names):
        groups['Relation Module'] = []
    for i, name in enumerate(names):
        if name.startswith('local_encoder.'):
            group = 'LocalEncoder'
        elif name.startswith('type_embedding.'):
            group = 'TypeEmbedding'
        elif name.startswith(('global_interactor.pair_embedding.', 'global_interactor.relation_mlp.')):
            group = 'Relation Module'
        elif name.startswith('global_interactor.'):
            group = 'GlobalInteractor'
        elif name.startswith('decoder.'):
            group = 'Decoder'
        else:
            raise AssertionError('Unclassified parameter: ' + name)
        groups[group].append(i)
    shared = sorted(i for group in GROUPS[:4] for i in groups[group])
    assert sum(parameters[i].numel() for i in shared) == 646001
    assert len(shared) == len(set(shared))
    groups['ALL SHARED'] = shared
    if 'Relation Module' in groups:
        assert sum(parameters[i].numel() for i in groups['Relation Module']) == 1608
    rows = [{'name': name, 'shape': list(p.shape), 'numel': p.numel(),
        'group': next(g for g, indices in groups.items() if g != 'ALL SHARED' and i in indices)}
        for i, (name, p) in enumerate(zip(names, parameters))]
    return names, parameters, groups, rows


def tensor_fingerprint(data):
    digest = hashlib.sha256()
    for name in sorted(data.keys()):
        tensor = data[name]
        if isinstance(tensor, torch.Tensor):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def supervision(data):
    eligible = data.target_mask & data.future_mask.any(-1)
    assert torch.equal(eligible, data.target_mask), 'Targets without future supervision'
    result = {}
    counts, coordinates = [], []
    for t, label in enumerate(('Vehicle', 'Pedestrian', 'Bicycle')):
        target = eligible & (data.agent_type == t)
        counts.append(int(target.sum()))
        coordinates.append(int((data.future_mask & target[:, None]).sum())*2)
        result[label + 'Targets'] = counts[-1]
        result[label + 'ValidCoordinates'] = coordinates[-1]
    assert sum(counts) == int(data.target_mask.sum())
    assert sum(coordinates) == int((data.future_mask & eligible[:, None]).sum())*2
    for label, count, coordinate in zip(('Vehicle', 'Pedestrian', 'Bicycle'), counts, coordinates):
        result[label + 'ActorShare'] = count/sum(counts)
        result[label + 'CoordinateShare'] = coordinate/sum(coordinates)
    return result


def class_loss(model, output, data, t):
    working = data.clone()
    working.target_mask = data.target_mask & (data.agent_type == t)
    assert working.num_nodes == data.num_nodes and torch.equal(working.edge_index, data.edge_index)
    assert torch.equal(working.agent_type, data.agent_type) and torch.equal(working.future_mask, data.future_mask)
    assert working.target_mask.any()
    assert model.reg_loss.reduction == model.cls_loss.reduction == 'mean'
    values = model.recovery_loss(output, working, 'original_nll')
    assert all(torch.isfinite(v) for v in values.values())
    return values


def vector(gradients, parameters, indices):
    # None is represented by a correctly sized zero vector and audited separately.
    return torch.cat([gradients[i].detach().reshape(-1) if gradients[i] is not None
        else torch.zeros_like(parameters[i]).reshape(-1) for i in indices]).double()


def gradient_metrics(v, p, weight_v, weight_p):
    assert torch.isfinite(v).all() and torch.isfinite(p).all()
    norm_v, norm_p = float(torch.linalg.vector_norm(v)), float(torch.linalg.vector_norm(p))
    dot = float(torch.dot(v, p))
    valid = norm_v > 0 and norm_p > 0
    cos = dot/(norm_v*norm_p + EPS) if valid else None
    assert cos is None or -1-1e-10 <= cos <= 1+1e-10
    effective_v, effective_p = weight_v*norm_v, weight_p*norm_p
    return {'Valid': int(valid), 'CosVP': cos, 'DotProduct': dot, 'NormV': norm_v, 'NormP': norm_p,
        'NormRatio': norm_v/(norm_p+EPS), 'WeightV': weight_v, 'WeightP': weight_p,
        'EffectiveV': effective_v, 'EffectiveP': effective_p,
        'EffectiveRatio': effective_v/(effective_p+EPS)}


def run(limit=100):
    assert limit == 100, 'The protocol fixes 100 batches'
    assert not (DIAG / 'stage4ae_gradient_completion.json').exists(), 'Use the completed frozen diagnostic'
    assert read_json(DIAG / 'stage4ae_motion_audit.json')['status'] == 'PASS', 'Part A must pass first'
    verify_frozen()
    seed_all(2022)
    models = load_models()
    inventories = {name: inventory(model) for name, model in models.items()}
    b_shared = {row['name']: row['shape'] for row in inventories['Stage3B'][3]}
    c_shared = {row['name']: row['shape'] for row in inventories['Stage4A'][3] if row['group'] != 'Relation Module'}
    assert b_shared == c_shared
    states = {name: state_digest(model.state_dict()) for name, model in models.items()}
    atomic_json(DIAG / 'stage4ae_gradient_parameter_inventory.json', {
        'status': 'PASS', 'models': {name: inv[3] for name, inv in inventories.items()},
        'shared_name_shape_sets_equal': True, 'shared_parameter_count': 646001,
        'relation_parameter_count': 1608, 'norm_arithmetic_dtype': 'float64', 'eps': EPS,
        'class_loss_reduction': 'original NLL mean over valid class coordinates plus original classification mean over eligible class actors',
        'context_actors_removed': False, 'classification_soft_targets_detached': True,
        'loss_source': 'unchanged models.hivt_loss_recovery.HiVTLossRecovery.recovery_loss(original_nll) and inherited HiVTNuScenesVehicle.loss',
        'loss_source_sha256': {str(p): sha256(ROOT.parents[1] / p) for p in
            [Path('models/hivt_loss_recovery.py'), Path('models/hivt_nuscenes.py'),
             Path('models/hivt_runtime/laplace_nll_loss.py'), Path('models/hivt_runtime/soft_target_cross_entropy_loss.py')]}})
    dataset = SceneDataset('train')
    assert len(dataset) == 16898 and len(dataset.scene_indices) == 700
    generator = torch.Generator().manual_seed(2022)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, generator=generator)
    batch_rows, manifest, pair_rows, loss_checks = [], [], [], []
    start = time.monotonic()
    selected = 0
    max_gpu = 0
    for loader_index, cpu_data in enumerate(loader):
        stats = supervision(cpu_data)
        if not stats['VehicleTargets'] or not stats['PedestrianTargets']:
            continue
        selected += 1
        windows = list(zip(cpu_data.scene_token, cpu_data.sample_token))
        assert cpu_data.num_graphs == 16
        identity_json = json.dumps(windows, separators=(',', ':'))
        identity_sha = hashlib.sha256(identity_json.encode()).hexdigest()
        original_input_sha = tensor_fingerprint(cpu_data)
        record = {'BatchID': selected, 'LoaderBatchIndex': loader_index,
            'WindowIdentitySHA256': identity_sha, 'InputTensorSHA256': original_input_sha,
            'WindowIdentitiesJSON': identity_json, 'WindowCount': 16,
            'ContextActorNodes': cpu_data.num_nodes, 'SceneCount': len(set(cpu_data.scene_token)), **stats}
        manifest.append(record)
        data = cpu_data.cuda()
        assert tensor_fingerprint(data) == original_input_sha
        for model_name, model in models.items():
            assert not model.training and torch.is_grad_enabled()
            names, parameters, groups, _ = inventories[model_name]
            torch.cuda.reset_peak_memory_stats()
            # One forward per model per selected batch; both class losses share it.
            output = model(model_input(data))
            lv, lp = class_loss(model, output, data, 0), class_loss(model, output, data, 1)
            if selected == 1:
                # Reconcile official mixed reduction without adding any gradient pass.
                mixed = model.recovery_loss(output, data, 'original_nll')
                class_values = [lv, lp]
                if stats['BicycleTargets']:
                    class_values.append(class_loss(model, output, data, 2))
                labels = ('Vehicle', 'Pedestrian', 'Bicycle')[:len(class_values)]
                reg = sum(x['regression_loss']*stats[label+'CoordinateShare'] for label, x in zip(labels, class_values))
                cls = sum(x['classification_loss']*stats[label+'ActorShare'] for label, x in zip(labels, class_values))
                assert torch.allclose(reg, mixed['regression_loss'], rtol=1e-6, atol=1e-6)
                assert torch.allclose(cls, mixed['classification_loss'], rtol=1e-6, atol=1e-6)
                loss_checks.append({'Model': model_name, 'BatchID': selected,
                    'class_weighted_regression_matches_official': True,
                    'class_weighted_classification_matches_official': True,
                    'regression_difference': float((reg-mixed['regression_loss']).abs()),
                    'classification_difference': float((cls-mixed['classification_loss']).abs()),
                    'note': 'Regression coordinate weights and classification actor weights differ; the norm proxy is not an exact mixed-gradient decomposition.'})
                del mixed, class_values, reg, cls
            gv = torch.autograd.grad(lv['loss'], parameters, retain_graph=True, create_graph=False, allow_unused=True)
            gp = torch.autograd.grad(lp['loss'], parameters, retain_graph=False, create_graph=False, allow_unused=True)
            losses = {f'Loss{name}': float(values['loss'].detach()) for name, values in (('V', lv), ('P', lp))}
            for group, indices in groups.items():
                none_v, none_p = [names[i] for i in indices if gv[i] is None], [names[i] for i in indices if gp[i] is None]
                values = gradient_metrics(vector(gv, parameters, indices), vector(gp, parameters, indices),
                    stats['VehicleCoordinateShare'], stats['PedestrianCoordinateShare'])
                batch_rows.append({'Model': model_name, 'BatchID': selected,
                    'WindowIdentitySHA256': identity_sha, 'InputTensorSHA256': original_input_sha,
                    **stats, 'ParameterGroup': group, **values, **losses,
                    'ParameterCount': sum(parameters[i].numel() for i in indices),
                    'ParameterTensors': len(indices), 'NoneVTensors': len(none_v), 'NonePTensors': len(none_p),
                    'NoneVParametersJSON': json.dumps(none_v), 'NonePParametersJSON': json.dumps(none_p)})
                if group in GROUPS:
                    assert values['Valid'], ('STOP: zero shared class gradient', model_name, selected, group)
                    assert not none_v and not none_p, ('STOP: unexpected unused shared parameters', model_name, selected, group, none_v, none_p)
            if model_name == 'Stage4A':
                pair_index = names.index('global_interactor.pair_embedding.weight')
                assert gv[pair_index] is not None and gp[pair_index] is not None
                labels = ('V<-V', 'V<-P', 'V<-B', 'P<-V', 'P<-P', 'P<-B', 'B<-V', 'B<-P', 'B<-B')
                for row, pair in enumerate(labels):
                    values = gradient_metrics(gv[pair_index][row].detach().double(), gp[pair_index][row].detach().double(),
                        stats['VehicleCoordinateShare'], stats['PedestrianCoordinateShare'])
                    pair_rows.append({'Model': model_name, 'BatchID': selected, 'Pair': pair, **values})
            max_gpu = max(max_gpu, torch.cuda.max_memory_allocated())
            assert all(p.grad is None for p in parameters), 'autograd.grad must not populate parameter.grad'
            assert state_digest(model.state_dict()) == states[model_name], 'Parameters/buffers changed'
            del output, lv, lp, gv, gp
        assert tensor_fingerprint(data) == original_input_sha
        del data, cpu_data
        if selected % 10 == 0 or selected == 1:
            write_csv(DIAG / 'stage4ae_gradient_batch_manifest.csv', manifest)
            write_csv(DIAG / 'stage4ae_gradient_conflict_batches.csv', batch_rows)
            write_csv(DIAG / 'stage4ae_relation_embedding_row_batches.csv', pair_rows)
            print('STAGE4AE_GRADIENT_BATCH', selected, '/', limit, 'loader_batch', loader_index,
                'elapsed_seconds', round(time.monotonic()-start, 2), 'memory_peak_MiB', round(max_gpu/2**20, 2), flush=True)
        if selected == limit:
            break
    dataset.clear()
    assert selected == limit
    assert len(batch_rows) == 1100 and len(pair_rows) == 900 and len(manifest) == 100
    for model_name in models:
        assert state_digest(models[model_name].state_dict()) == states[model_name]
    atomic_json(DIAG / 'stage4ae_gradient_loss_audit.json', {'status': 'PASS', 'checks': loss_checks})
    atomic_json(DIAG / 'stage4ae_gradient_completion.json', {'status': 'PASS', 'batches': selected,
        'batch_size': 16, 'fixed_loader_order': 'TRAIN index order, shuffle=False, workers=0, seed2022',
        'models': {name: {'checkpoint_sha256': CHECKPOINT_B if name == 'Stage3B' else CHECKPOINT_C,
            'state_before': state, 'state_after': state_digest(models[name].state_dict()),
            'forward_calls': selected, 'class_gradient_calls': 2*selected} for name, state in states.items()},
        'original_NLL_loss_reused': True, 'full_context_graph_retained': True,
        'dropout_disabled_eval': True, 'grad_enabled': True, 'method': 'torch.autograd.grad',
        'optimizer_created': False, 'optimizer_step_called': False, 'backward_called': False,
        'parameter_grad_buffers_populated': False, 'parameters_buffers_changed': False,
        'NaN': 0, 'Inf': 0, 'shared_parameter_None_tensors': 0,
        'manifest_sha256': sha256(DIAG / 'stage4ae_gradient_batch_manifest.csv'),
        'batch_metrics_sha256': sha256(DIAG / 'stage4ae_gradient_conflict_batches.csv'),
        'embedding_row_metrics_sha256': sha256(DIAG / 'stage4ae_relation_embedding_row_batches.csv'),
        'elapsed_seconds': time.monotonic()-start, 'peak_GPU_MiB': max_gpu/2**20,
        'test_used': False, 'formal_model_trained': False})
    verify_frozen()
    print('STAGE4AE_GRADIENT_COMPLETE', flush=True)


if __name__ == '__main__':
    run()
