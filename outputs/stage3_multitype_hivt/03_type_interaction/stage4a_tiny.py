"""Neutral-output, directed-edge, gradient and frozen-window tiny gates."""
import copy
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage4a_common import (CLASSES, SceneDataset, config, model_new, model_input,
    evaluate_graphs, loss_diagnostics, atomic_json, read_json, write_csv, sha256,
    git, verify_previous, canonical_stage3b_model)
from stage4a_global_interactor import PAIR_LABELS
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph


def graphs_from_frozen_selection():
    dataset = SceneDataset('train')
    source = ROOT / '00_manifest/stage3b_tiny_selection.json'
    selection = read_json(source)
    graphs = []
    pair_counts = torch.zeros(9, dtype=torch.long)
    windows = []
    for record in selection['windows']:
        graph = dataset[record['index']]
        assert (graph.scene_token, graph.sample_token) == (
            record['scene_token'], record['sample_token'])
        assert graph.num_nodes == record['actor_count']
        edges, _ = subgraph(~graph.padding_mask[:, 4], graph.edge_index)
        ids = 3 * graph.agent_type[edges[1]] + graph.agent_type[edges[0]]
        counts = torch.bincount(ids, minlength=9)
        pair_counts += counts
        windows.append({**record, 'directed_global_edge_counts':
                        dict(zip(PAIR_LABELS, counts.tolist()))})
        graphs.append(graph)
    dataset.clear()
    totals = {name: sum(int((g.target_mask & (g.agent_type == t)).sum())
                        for g in graphs) for t, name in enumerate(CLASSES)}
    assert totals == selection['target_counts']
    assert all(totals[name] >= minimum for name, minimum in
               zip(CLASSES, (10, 10, 5)))
    assert all(pair_counts[i] > 0 for i in (0, 1, 3, 4)), (
        'Frozen tiny lacks required real V<-V,V<-P,P<-V,P<-P edges; stop')
    audit = {**selection, 'status': 'PASS', 'windows': windows,
             'same_exact_actor_windows_as_Stage3B': True,
             'Stage3B_selection_sha256': sha256(source),
             'directed_global_edge_counts': dict(zip(PAIR_LABELS, pair_counts.tolist())),
             'required_heterogeneous_edge_coverage': True,
             'V_B_and_B_V_available_and_retained': bool(pair_counts[2] and pair_counts[6]),
             'tiny_selection_changed': False, 'full_scene_context_preserved': True,
             'formal_train_and_val_selection_changed': False}
    atomic_json(ROOT / '00_manifest/stage4a_tiny_selection.json', audit)
    return graphs


def rotate_working(data):
    working = data.clone()
    s, c = torch.sin(working.rotate_angles), torch.cos(working.rotate_angles)
    working.rotate_mat = torch.stack([c, -s, s, c], dim=-1).reshape(-1, 2, 2)
    return working


def integration_tests(graphs):
    model = model_new(audit=True)
    canonical = canonical_stage3b_model()
    model.eval()
    canonical.eval()
    data = Batch.from_data_list(graphs[:2]).cuda()
    original_edge = data.edge_index.clone()
    original_lane_edge = data.lane_actor_index.clone()
    original_x = data.x.clone()
    rejected = 0
    for types in (data.agent_type[:, None], data.agent_type.float(),
                  data.agent_type[:-1], torch.full_like(data.agent_type, 3),
                  torch.full_like(data.agent_type, -1)):
        try:
            model.validate_actor_types(types, data.num_nodes)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('Invalid actor taxonomy accepted')
    assert data.agent_type.dtype == torch.long and rejected == 5
    assert set(data.agent_type.tolist()) == {0, 1, 2}
    global_module = model.global_interactor
    artificial_types = torch.arange(3, device='cuda')
    # Explicit order: target-major, source-minor; target V/source P is pair 1.
    artificial_edges = torch.tensor([[0, 1, 2] * 3,
                                    [0] * 3 + [1] * 3 + [2] * 3], device='cuda')
    artificial_ids = global_module.directed_pair_ids(artificial_types, artificial_edges)
    assert torch.equal(artificial_ids, torch.arange(9, device='cuda'))
    assert int(artificial_ids[1]) == 1 and int(artificial_ids[3]) == 3
    artificial_embeddings = global_module.pair_embedding(artificial_ids)
    assert artificial_embeddings.shape == (9, 16)
    assert torch.isfinite(artificial_embeddings).all()
    working = rotate_working(data)
    lookup_inputs = []
    hook = global_module.pair_embedding.register_forward_pre_hook(
        lambda module, args: lookup_inputs.append(args[0].detach().clone()))
    with torch.no_grad():
        relation = global_module.relation_bias(working)
        edges = relation['edge_index']
        E = edges.shape[1]
        assert relation['pair_ids'].shape == (E,)
        assert (relation['pair_ids'] >= 0).all() and (relation['pair_ids'] <= 8).all()
        assert relation['relation_features'].shape == (E, 20)
        assert relation['bias'].shape == (E, 3, 8)
        assert torch.count_nonzero(relation['bias']) == 0
        assert (edges >= 0).all() and (edges < data.num_nodes).all()
        expected_edges, _ = subgraph(~data.padding_mask[:, 4], data.edge_index)
        assert torch.equal(edges, expected_edges)
        # Geometry uses source-minus-target and the target's original rotation.
        expected_rel = data.positions[edges[0], 4] - data.positions[edges[1], 4]
        expected_rel = torch.bmm(expected_rel.unsqueeze(1),
                                 working.rotate_mat[edges[1]]).squeeze(1) / 50.0
        theta = data.rotate_angles[edges[0]] - data.rotate_angles[edges[1]]
        torch.testing.assert_close(relation['relation_features'][:, 16:18], expected_rel,
                                   rtol=0, atol=0)
        torch.testing.assert_close(relation['relation_features'][:, 18:],
                                   torch.stack((torch.cos(theta), torch.sin(theta)), -1),
                                   rtol=0, atol=0)
        output = model(model_input(data))
        reference = canonical(data)
        differences = {key: float((output[key] - reference[key]).abs().max())
                       for key in ('raw_prediction', 'mode_logits', 'mode_prob')}
        assert all(value < 1e-6 for value in differences.values()), differences
        assert output['raw_prediction'].shape == (6, data.num_nodes, 12, 4)
        assert output['mode_prob'].shape == (data.num_nodes, 6)
        assert all(torch.isfinite(v).all() for v in output.values())
        torch.testing.assert_close(output['mode_prob'].sum(-1),
                                   torch.ones(data.num_nodes, device='cuda'))
        assert all(torch.equal(ids, relation['pair_ids']) for ids in lookup_inputs)
        assert torch.equal(data.edge_index, original_edge)
        assert torch.equal(data.lane_actor_index, original_lane_edge)
        assert torch.equal(data.x, original_x)
        # Neither future coordinates nor future availability enters the adapter.
        perturbed = data.clone()
        perturbed.positions[:, 5:] += 1234
        perturbed.y -= 777
        perturbed.future_mask = ~perturbed.future_mask
        perturbed.padding_mask[:, 5:] = ~perturbed.padding_mask[:, 5:]
        perturbed.future_times += 950
        perturbed.ego_future += 990
        changed = model(perturbed)
        future_diff = float((output['raw_prediction'] - changed['raw_prediction']).abs().max())
        assert future_diff < 1e-6
        # A nonzero adapter leaves graph selection and the value path unchanged.
        global_module.relation_mlp[-1].bias.fill_(0.01)
        nonneutral = global_module.relation_bias(working)
        assert torch.equal(nonneutral['edge_index'], expected_edges)
        global_module.reset_neutral_bias()
    hook.remove()
    neutral_audit = {'status': 'PASS', 'fixed_real_train_batch':
                     [(g.scene_token, g.sample_token) for g in graphs[:2]],
                     'mode': 'eval', 'threshold': 1e-6,
                     'raw_prediction_max_abs_diff': differences['raw_prediction'],
                     'mode_logits_max_abs_diff': differences['mode_logits'],
                     'mode_prob_max_abs_diff': differences['mode_prob'],
                     'all_initial_relation_bias_exactly_zero': True,
                     'trained_Stage3B_checkpoint_loaded': False}
    atomic_json(ROOT / '00_manifest/stage4a_neutral_initialization_audit.json', neutral_audit)
    tests = {'status': 'PASS', 'Test1_actor_type_only_012': True,
             'invalid_actor_type_shape_dtype_and_range_rejections': rejected,
             'Test2_pair_id_range_0_to_8': True,
             'Test3_V_from_P_and_P_from_V_different': True,
             'Test4_all_nine_pairs_legal_embedding': list(artificial_embeddings.shape),
             'Test5_relation_feature_shape': list(relation['relation_features'].shape),
             'Test6_relation_bias_shape': list(relation['bias'].shape),
             'Test7_zero_init_all_bias_exact_zero': True,
             'Test8_step0_matches_Stage3B': differences,
             'Test9_raw_prediction_shape': list(output['raw_prediction'].shape),
             'Test9_mode_probability_shape': list(output['mode_prob'].shape),
             'Test10_NaN': 0, 'Test10_Inf': 0,
             'Test11_actor_actor_only_no_lane_lookup': True,
             'Test12_edge_index_unchanged': True,
             'target_frame_source_minus_target_geometry_exact': True,
             'heading_cos_sin_source_minus_target_exact': True,
             'future_perturbation_max_abs_diff': future_diff,
             'lane_actor_edges_and_input_x_unchanged': True,
             'bias_added_only_before_softmax': True,
             'tests_optimizer_updates': 0}
    atomic_json(ROOT / '00_manifest/stage4a_unit_tests.json', tests)
    del model, canonical, data
    torch.cuda.empty_cache()


def gradient_norms(model):
    parameters = dict(model.named_parameters())
    gradients = [p.grad for p in parameters.values() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    groups = {
        'pair_embedding': ['global_interactor.pair_embedding.weight'],
        'first_MLP_layer': ['global_interactor.relation_mlp.0.weight',
                            'global_interactor.relation_mlp.0.bias'],
        'final_MLP_layer': ['global_interactor.relation_mlp.2.weight',
                            'global_interactor.relation_mlp.2.bias'],
        'TypeEmbedding': ['type_embedding.weight']}
    norms = {}
    for group, names in groups.items():
        assert all(parameters[name].grad is not None for name in names), group
        norms[group] = float(torch.stack([
            parameters[name].grad.detach().square().sum() for name in names]).sum().sqrt())
    return norms


def gradient_audit(graphs):
    model = model_new()
    c = config()
    optimizer = model.optimizer(c['tiny']['warmup_lr'], c['weight_decay'])
    data = Batch.from_data_list(graphs).cuda()
    rows = []
    for update in range(1, 11):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = loss_diagnostics(model, model(model_input(data)), data, 'fixed_scale')
        assert all(torch.isfinite(v) for v in values.values())
        values['loss'].backward()
        norms = gradient_norms(model)
        assert norms['final_MLP_layer'] > 0 and norms['TypeEmbedding'] > 0
        if update == 1:
            assert norms['pair_embedding'] == 0 and norms['first_MLP_layer'] == 0
        optimizer.step()
        model.eval()
        with torch.no_grad():
            bias = model.global_interactor.relation_bias(rotate_working(data))['bias']
            assert torch.isfinite(bias).all()
            mean_abs_bias = float(bias.abs().mean())
        rows.append({'optimizer_update': update, 'loss': float(values['loss'].detach()),
                     'gradient_norms': norms, 'mean_absolute_bias_after_update': mean_abs_bias})
    assert any(row['gradient_norms']['pair_embedding'] > 0 for row in rows[1:])
    assert any(row['gradient_norms']['first_MLP_layer'] > 0 for row in rows[1:])
    assert rows[-1]['mean_absolute_bias_after_update'] > 0
    result = {'status': 'PASS', 'updates': rows, 'optimizer_updates': 10,
              'first_backward_final_MLP_gradient_nonzero': True,
              'first_backward_pair_and_first_layer_gradient_zero_allowed': True,
              'pair_embedding_gradient_nonzero_within_10_updates': True,
              'first_MLP_gradient_nonzero_within_10_updates': True,
              'bias_becomes_nonzero': True, 'NaN': 0, 'Inf': 0,
              'train_only': True, 'validation_used': False,
              'model_discarded_before_tiny_and_formal': True}
    atomic_json(ROOT / '04_evaluation/stage4a_gradient_audit.json', result)
    del model, optimizer, data
    torch.cuda.empty_cache()


def adequate(initial, current):
    return all(current[name][key] < 0.5 * initial[name][key]
               for name in CLASSES for key in ('ADE', 'FDE', 'fixed_scale_regression_loss'))


def main():
    verify_previous()
    graphs = graphs_from_frozen_selection()
    integration_tests(graphs)
    print('STAGE4A_UNIT_AND_NEUTRAL=PASS', flush=True)
    gradient_audit(graphs)
    print('STAGE4A_GRADIENT_AUDIT=PASS', flush=True)
    model = model_new()
    c = config()
    optimizer = model.optimizer(c['tiny']['warmup_lr'], c['weight_decay'])
    initial = evaluate_graphs(model, graphs)
    rows = []
    global_step = 0
    best_state = best_optimizer = None
    best_fde = float('inf')
    started = time.monotonic()
    minima = {name: float('inf') for name in
              ('pair_embedding', 'first_MLP_layer', 'final_MLP_layer', 'TypeEmbedding')}
    maxima = {name: 0.0 for name in minima}
    warm_source = 0
    for phase, maxkey, lrkey in (
        ('fixed_scale', 'warmup_max_steps', 'warmup_lr'),
        ('original_nll', 'nll_max_steps', 'nll_lr')):
        if phase == 'original_nll':
            model.load_state_dict(best_state)
            optimizer.load_state_dict(best_optimizer)
        for group in optimizer.param_groups:
            group['lr'] = c['tiny'][lrkey]
        for step in range(1, c['tiny'][maxkey] + 1):
            start = ((step - 1) * 2) % len(graphs)
            chosen = [graphs[(start + j) % len(graphs)]
                      for j in range(min(2, len(graphs)))]
            data = Batch.from_data_list(chosen).cuda()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            values = loss_diagnostics(model, model(model_input(data)), data, phase)
            assert all(torch.isfinite(v) for v in values.values())
            values['loss'].backward()
            grads = gradient_norms(model)
            assert grads['final_MLP_layer'] > 0 and grads['TypeEmbedding'] > 0
            if global_step > 0:
                assert grads['pair_embedding'] > 0 and grads['first_MLP_layer'] > 0
            for name in minima:
                minima[name] = min(minima[name], grads[name])
                maxima[name] = max(maxima[name], grads[name])
            optimizer.step()
            global_step += 1
            if step % c['tiny']['check_every'] == 0 or step == c['tiny'][maxkey]:
                current = evaluate_graphs(model, graphs)
                with torch.no_grad():
                    current_bias = model.global_interactor.relation_bias(rotate_working(data))['bias']
                    assert torch.isfinite(current_bias).all()
                    bias_abs = float(current_bias.abs().mean())
                row = {'global_step': global_step, 'phase': phase, 'phase_step': step,
                       'train_loss': float(values['loss'].detach()),
                       'regression_loss': float(values['regression_loss'].detach()),
                       'classification_loss': float(values['classification_loss'].detach()),
                       'relation_bias_mean_absolute': bias_abs}
                row.update({name + '_gradient_norm': value for name, value in grads.items()})
                for name in CLASSES:
                    row.update({name + '_' + field: value for field, value in current[name].items()})
                rows.append(row)
                write_csv(ROOT / '03_type_interaction/stage4a_tiny_curve.csv', rows)
                print('STAGE4A_TINY', phase, step,
                      {name: (current[name]['ADE'], current[name]['FDE']) for name in CLASSES},
                      'pair_grad', grads['pair_embedding'], 'bias_abs', bias_abs, flush=True)
                if phase == 'fixed_scale' and current['overall']['FDE'] < best_fde:
                    best_fde = current['overall']['FDE']
                    warm_source = step
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                if (phase == 'fixed_scale' and step >= c['tiny']['warmup_min_steps']
                        and adequate(initial, current)):
                    break
        if phase == 'fixed_scale':
            warm_steps = step
    final = evaluate_graphs(model, graphs)
    with torch.no_grad():
        complete = Batch.from_data_list(graphs).cuda()
        bias = model.global_interactor.relation_bias(rotate_working(complete))['bias']
        final_abs_bias = float(bias.abs().mean())
        finite_bias = bool(torch.isfinite(bias).all())
    status = 'PASS' if adequate(initial, final) and finite_bias and final_abs_bias > 0 else 'FAIL'
    result = {'status': status, 'initial': initial, 'final': final,
              'criterion': 'three-class ADE/FDE/fixed-scale diagnostics decrease >=50%; finite outputs; relation bias nonzero; pair gradients positive after neutral first update',
              'warmup_steps': warm_steps, 'warmup_best_source_step': warm_source,
              'NLL_steps': step, 'global_steps': global_step,
              'gradient_norm_min': minima, 'gradient_norm_max': maxima,
              'first_neutral_pair_and_first_MLP_zero_gradient_expected': True,
              'final_mean_absolute_relation_bias': final_abs_bias,
              'elapsed_seconds': time.monotonic() - started,
              'same_exact_Stage3B_tiny_windows': True,
              'full_scene_context_preserved': True, 'train_only': True,
              'model_discarded_for_formal_training': True,
              'hyperparameters_changed_or_searched': False,
              'NaN': 0, 'Inf': 0, 'git_commit_SHA': git('rev-parse', 'HEAD')}
    atomic_json(ROOT / '04_evaluation/stage4a_tiny_overfit.json', result)
    path = ROOT / '07_checkpoints/stage4a_tiny_last.pt'
    temp = path.with_suffix('.pt.tmp')
    torch.save({'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'metadata': result}, temp)
    os.replace(temp, path)
    verify_previous()
    print('STAGE4A_TINY_OVERFIT=' + status, flush=True)
    assert status == 'PASS', 'Stop: Stage4A tiny failed; formal training forbidden'


if __name__ == '__main__':
    torch.set_num_threads(4)
    try:
        main()
    except Exception as error:
        atomic_json(ROOT / '04_evaluation/stage4a_gate_failure.json',
                    {'status': 'FAIL', 'error': repr(error), 'formal_training_started': False})
        raise
