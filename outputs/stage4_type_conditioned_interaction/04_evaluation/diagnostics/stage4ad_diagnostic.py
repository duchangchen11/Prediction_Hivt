"""Read-only Stage4A diagnostic: exact forward interception and VAL sensitivity.

The installed message is generated from the frozen formal method's AST. Only
its additive bias statement is intercepted. All q/k/value operations and the
actual PyG softmax remain the original code. The dropout pre-hook observes the
actual final attention; no approximate attention implementation is introduced.
"""
import argparse
import ast
import csv
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
import types

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
DIAG = ROOT / '04_evaluation/diagnostics'
sys.path[:0] = [str(ROOT / '00_manifest'), str(ROOT / '04_evaluation')]
from stage4a_common import (SceneDataset, atomic_json, sha256, state_digest,
    model_input, config, evaluate, seed_all, verify_previous, read_json, write_csv)
from stage4a_evaluate import (load_final_model, read_membership, read_actors,
    select, summarize, actor_key, MEMBERSHIP, BASE_ACTORS, ACTORS, BEST, INTERACTION)
import stage4a_global_interactor as formal
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax

BASE_COMMIT = '528b9501d66368b7d6a3b754f64bc1c5006f7d13'
CHECKPOINT_SHA = 'ff25266ba6a44630cfec01ae1596ef7de05f0d71f06f9676f6ea7ef327775ecb'
FREEZE = ROOT / '00_manifest/stage4ad_frozen_inputs.json'
EDGE_DTYPE = np.dtype([('batch', '<u2'), ('source', '<i4'), ('target', '<i4'),
    ('pair', 'u1'), ('flags', 'u1'), ('base', '<f4', (8,)),
    ('bias', '<f4', (8,)), ('final', '<f4', (8,))])
NODE_DTYPE = np.dtype([('batch', '<u2'), ('target', '<i4'), ('type', 'u1'),
    ('flags', 'u1'), ('pair_context', '<u2'), ('entropy_base', '<f4', (8,)),
    ('entropy_final', '<f4', (8,)), ('l1_shift', '<f4', (8,)), ('switch', 'u1', (8,))])


def freeze_inputs():
    assert not FREEZE.exists(), 'Do not replace an existing frozen-input ledger'
    verify_previous(shards=True)
    assert sha256(BEST) == CHECKPOINT_SHA
    manifest = read_json(ROOT / '00_manifest/stage4a_artifact_manifest.json')
    for row in manifest['artifacts']:
        assert sha256(ROOT / row['relative_path']) == row['sha256'], row['relative_path']
    files = {}
    for p in ROOT.rglob('stage4a_*'):
        if p.is_file() and not {'__pycache__', 'stage4a_cache'}.intersection(p.parts):
            files[str(p.relative_to(PROJECT))] = sha256(p)
    old = PROJECT / 'outputs/stage2c_trainval_vehicle_baseline/04_evaluation'
    for p in old.glob('stage2c_qualitative_main_case_new*'):
        if p.is_file():
            files[str(p.relative_to(PROJECT))] = sha256(p)
    atomic_json(FREEZE, {'status': 'FROZEN_BEFORE_DIAGNOSTIC', 'base_commit': BASE_COMMIT,
        'checkpoint_sha256': CHECKPOINT_SHA, 'files': files,
        'membership_sha256': sha256(MEMBERSHIP), 'Stage3B_actor_sha256': sha256(BASE_ACTORS),
        'requirements_sha256': sha256(ROOT / '00_manifest/stage4ad_requirements.txt'),
        'no_training': True, 'lambdas': [0, .25, .5, .75, 1]})
    print('STAGE4AD_INPUTS_FROZEN', len(files), flush=True)


def verify_frozen():
    ledger = read_json(FREEZE)
    for name, digest in ledger['files'].items():
        assert sha256(PROJECT / name) == digest, 'Frozen input changed: ' + name
    verify_previous(shards=False)
    return ledger


def exact_message():
    """Transform exactly one formal statement, keeping its function signature."""
    source = textwrap.dedent(inspect.getsource(formal.TypeConditionedGlobalInteractorLayer.message))
    original = ast.parse(source)
    modified = ast.parse(source)
    expected = ast.dump(ast.parse('alpha = alpha + relation_bias').body[0])
    count = 0
    for i, statement in enumerate(modified.body[0].body):
        if ast.dump(statement) == expected:
            modified.body[0].body[i] = ast.parse(
                'alpha = self.stage4ad_observer.logits(self, alpha, relation_bias, index, ptr, size_i)').body[0]
            count += 1
    assert count == 1
    restored = ast.parse(ast.unparse(modified))
    replacement = ast.dump(ast.parse(
        'alpha = self.stage4ad_observer.logits(self, alpha, relation_bias, index, ptr, size_i)').body[0])
    for i, statement in enumerate(restored.body[0].body):
        if ast.dump(statement) == replacement:
            restored.body[0].body[i] = ast.parse('alpha = alpha + relation_bias').body[0]
    assert ast.dump(restored) == ast.dump(original)
    namespace = dict(formal.__dict__)
    exec(compile(ast.fix_missing_locations(modified), '<stage4ad_exact_formal_message>', 'exec'), namespace)
    return namespace['message'], {'single_additive_statement_intercepted': True,
        'restored_AST_equals_formal_AST': True, 'formal_message_source': source,
        'diagnostic_message_source': ast.unparse(modified),
        'q_k_node_k_edge_value_projections_unchanged': True,
        'formal_softmax_and_aggregation_unchanged': True}


class Observer:
    def __init__(self, model, membership, scale, capture=False):
        self.model, self.membership, self.scale = model, membership, float(scale)
        self.capture = capture
        self.batch_id = -1
        self.edge_counts = [0] * 3
        self.node_counts = [0] * 3
        self.pending = {}
        self.layers = list(model.global_interactor.global_interactor_layers)
        self.original = [layer.message for layer in self.layers]
        self.layer_ids = {id(layer): i for i, layer in enumerate(self.layers)}
        function, self.ast_audit = exact_message()
        self.handles = []
        if capture:
            self.edge_handles = [(DIAG / f'stage4ad_raw_L{i+1}_logits.bin').open('wb') for i in range(3)]
            self.node_handles = [(DIAG / f'stage4ad_raw_L{i+1}_attention.bin').open('wb') for i in range(3)]
            self.identity_handle = (DIAG / 'stage4ad_raw_node_identity.csv').open('w', newline='')
            self.identity_writer = csv.writer(self.identity_handle, lineterminator='\n')
            self.identity_writer.writerow(['batch', 'node', 'scene_token', 'sample_token', 'instance_token',
                'node_in_graph', 'target_type', 'flags'])
            self.handles.append(model.global_interactor.register_forward_pre_hook(self.prepare))
        model.global_interactor.record_relation_bias = capture
        for layer in self.layers:
            layer.stage4ad_observer = self
            layer.message = types.MethodType(function, layer)
            if capture:
                self.handles.append(layer.attn_drop.register_forward_pre_hook(
                    lambda module, args, layer=layer: self.attention(layer, args[0])))

    def prepare(self, module, args):
        assert not torch.is_grad_enabled() and not self.model.training
        data = args[0]
        self.batch_id += 1
        self.data = data
        flags = np.zeros(data.num_nodes, dtype=np.uint8)
        ptr = data.ptr.cpu().numpy()
        types_ = data.agent_type.cpu().numpy()
        for graph in range(data.num_graphs):
            for local, token in enumerate(data.instance_tokens[graph]):
                node = int(ptr[graph]) + local
                base = (data.scene_token[graph], data.sample_token[graph], token)
                candidates = [self.membership.get(base + (h,)) for h in ('full_horizon', 'partial_future')]
                rows = [r for r in candidates if r is not None]
                assert len(rows) <= 1
                if rows:
                    row = rows[0]
                    assert int(row['node_in_graph']) == local and int(row['agent_type_id']) == int(types_[node])
                    # bit0: known frozen membership; bit1: full horizon;
                    # bit2: heterogeneous20m; bit3: VP20m.
                    flags[node] = 1 | (2 if row['horizon'] == 'full_horizon' else 0)
                    flags[node] |= 4 if row['Heterogeneous-20m'] == '1' else 0
                    flags[node] |= 8 if row['VP-context-20m'] == '1' else 0
                self.identity_writer.writerow([self.batch_id, node, *base, local, int(types_[node]), int(flags[node])])
        self.flags = flags

    def logits(self, layer, base, bias, index, ptr, size_i):
        assert not torch.is_grad_enabled() and not layer.training
        # At lambda=1 use the identical formal addition, without a multiply.
        final = base + bias if self.scale == 1 else base + self.scale * bias
        if self.capture:
            assert self.scale == 1 and base.shape == bias.shape == final.shape
            observation = self.model.global_interactor.relation_bias_observation
            li = self.layer_ids[id(layer)]
            edge = observation['edge_index']
            assert torch.equal(index, edge[1])
            assert torch.equal(bias, observation['bias'][:, li, :])
            assert torch.equal(observation['pair_ids'],
                3 * self.data.agent_type[edge[1]] + self.data.agent_type[edge[0]])
            assert not torch.any(edge[0] == edge[1])
            assert not torch.any(self.data.padding_mask[edge.flatten(), 4])
            assert torch.equal(self.data.batch[edge[0]], self.data.batch[edge[1]])
            self.pending[li] = (base, bias, final, index, ptr, size_i, observation)
        return final

    def attention(self, layer, alpha_final):
        li = self.layer_ids[id(layer)]
        base, bias, final, index, ptr, size_i, observation = self.pending.pop(li)
        alpha_base = softmax(base, index, ptr, size_i)
        assert base.shape[1] == 8 and all(torch.isfinite(x).all() for x in (base, bias, final, alpha_base, alpha_final))
        nodes, inverse = torch.unique(index, sorted=True, return_inverse=True)
        n = len(nodes)
        sums = torch.zeros((n, 8, 5), device=base.device, dtype=torch.float32)
        entries = torch.stack((alpha_base, alpha_final,
            -alpha_base * torch.log(alpha_base.clamp_min(torch.finfo(alpha_base.dtype).tiny)),
            -alpha_final * torch.log(alpha_final.clamp_min(torch.finfo(alpha_final.dtype).tiny)),
            (alpha_final - alpha_base).abs()), -1)
        sums.index_add_(0, inverse, entries)
        assert torch.allclose(sums[:, :, :2], torch.ones_like(sums[:, :, :2]), atol=2e-6)
        # Deterministic first-edge tie break; each source identity is unique per target.
        top_edges = []
        edge_ids = torch.arange(len(index), device=base.device)[:, None].expand(-1, 8)
        for alpha in (alpha_base, alpha_final):
            maxima = torch.full((n, 8), -torch.inf, device=base.device)
            maxima.scatter_reduce_(0, inverse[:, None].expand(-1, 8), alpha, reduce='amax', include_self=True)
            candidate = torch.where(alpha == maxima[inverse], edge_ids, len(index))
            winners = torch.full((n, 8), len(index), device=base.device, dtype=torch.long)
            winners.scatter_reduce_(0, inverse[:, None].expand(-1, 8), candidate, reduce='amin', include_self=True)
            top_edges.append(observation['edge_index'][0][winners])
        switch = top_edges[0] != top_edges[1]
        source, target = observation['edge_index'].cpu().numpy()
        pair = observation['pair_ids'].cpu().numpy()
        edge_rows = np.empty(len(index), dtype=EDGE_DTYPE)
        edge_rows['batch'], edge_rows['source'], edge_rows['target'] = self.batch_id, source, target
        edge_rows['pair'], edge_rows['flags'] = pair, self.flags[target]
        for name, tensor in (('base', base), ('bias', bias), ('final', final)):
            edge_rows[name] = tensor.detach().cpu().numpy()
        edge_rows.tofile(self.edge_handles[li])
        self.edge_counts[li] += len(edge_rows)
        nodes_ = nodes.cpu().numpy()
        inverse_ = inverse.cpu().numpy()
        context = np.zeros(n, dtype=np.uint16)
        np.bitwise_or.at(context, inverse_, (1 << pair).astype(np.uint16))
        node_rows = np.empty(n, dtype=NODE_DTYPE)
        node_rows['batch'], node_rows['target'] = self.batch_id, nodes_
        node_rows['type'] = self.data.agent_type[nodes].cpu().numpy()
        node_rows['flags'], node_rows['pair_context'] = self.flags[nodes_], context
        node_rows['entropy_base'] = sums[:, :, 2].cpu().numpy()
        node_rows['entropy_final'] = sums[:, :, 3].cpu().numpy()
        node_rows['l1_shift'], node_rows['switch'] = sums[:, :, 4].cpu().numpy(), switch.cpu().numpy()
        node_rows.tofile(self.node_handles[li])
        self.node_counts[li] += n

    def close(self):
        assert not self.pending
        for handle in self.handles:
            handle.remove()
        for layer, original in zip(self.layers, self.original):
            layer.message = original
            del layer.stage4ad_observer
        self.model.global_interactor.record_relation_bias = False
        if self.capture:
            for handle in self.edge_handles + self.node_handles + [self.identity_handle]:
                handle.close()
            atomic_json(DIAG / 'stage4ad_capture_schema.json', {
                'status': 'PASS', 'edge_dtype': EDGE_DTYPE.descr, 'node_dtype': NODE_DTYPE.descr,
                'edge_counts': self.edge_counts, 'node_counts': self.node_counts,
                'batches': self.batch_id + 1, 'lambda': 1,
                'record_unit': 'actual edge/window/layer with eight heads; no added neighbors',
                'attention_unit': 'target with at least one actual incoming edge, per layer/head',
                'flags': {'1': 'frozen membership known', '2': 'full_horizon', '4': 'Heterogeneous-20m', '8': 'VP-context-20m'},
                'pair_context_definition': 'whole-neighborhood entropy on targets with >=1 actual incoming edge of this directed pair; context groups overlap',
                'entropy_log': 'natural', 'top_neighbor_tie_break': 'first edge in original formal edge order',
                'raw_files_local_only': True, 'AST_audit': self.ast_audit})


@torch.no_grad()
def smoke(model, membership):
    dataset = SceneDataset('val')
    batch = next(iter(DataLoader(dataset, batch_size=config()['batch_size'], shuffle=False, num_workers=0))).cuda()
    before = state_digest(model.state_dict())
    original = model(model_input(batch))
    observer = Observer(model, membership, 1, capture=False)
    try:
        modified = model(model_input(batch))
    finally:
        observer.close()
    differences = {}
    for key in ('raw_prediction', 'mode_prob'):
        a, b = original[key], modified[key]
        differences[key] = float((a-b).abs().max())
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (key, differences[key])
    assert state_digest(model.state_dict()) == before
    dataset.clear()
    atomic_json(DIAG / 'stage4ad_instrumentation_audit.json', {
        'status': 'PASS', 'maximum_absolute_output_differences': differences,
        'comparison_atol': 1e-5, 'comparison_rtol': 1e-5,
        'GPU_original_protocol_nondeterministic': True, 'parameters_buffers_unchanged': True,
        'no_grad': True, 'eval_mode': True, 'AST_audit': observer.ast_audit})


def pairing(rows, formal_rows, membership):
    mapping = {actor_key(r): r for r in rows}
    reference = {actor_key(r): r for r in formal_rows}
    assert mapping.keys() == reference.keys() == membership.keys()
    columns = ('node_in_graph', 'agent_type', 'agent_type_id', 'motion_state',
        'valid_future_steps', 'GT_trajectory_sha256', 'future_mask_bits')
    for key, row in mapping.items():
        assert all(row[k] == reference[key][k] for k in columns), key
        assert abs(float(row['GT_endpoint_displacement_m']) - float(reference[key]['GT_endpoint_displacement_m'])) < 1e-6
    return {'paired_actor_windows': len(rows), 'paired_full_horizon': 54990,
        'paired_partial_future': 30037, 'GT_and_masks_and_identities_equal': True}


@torch.no_grad()
def run():
    verify_frozen()
    seed_all(2022)
    model, _, _ = load_final_model()
    membership = read_membership()
    formal_rows = read_actors(ACTORS)
    state_before = state_digest(model.state_dict())
    smoke(model, membership)
    groups = [('Overall', 'overall'), ('Vehicle', 'vehicle'), ('Pedestrian', 'pedestrian'),
        ('Bicycle', 'bicycle'), ('vehicle.moving', 'vehicle.moving'),
        ('Vehicle >5m', 'Vehicle >5m'), ('Pedestrian >5m', 'Pedestrian >5m')]
    groups += [(g, g) for g in INTERACTION]
    sensitivity = []
    for scale in (1, .75, .5, .25, 0):
        start = time.monotonic()
        name = f'{scale:.2f}'.replace('.', 'p')
        actor_path = DIAG / f'stage4ad_lambda_{name}_actor_errors.csv'
        observer = Observer(model, membership, scale, capture=(scale == 1))
        try:
            measured = evaluate(SceneDataset('val'), model, actor_path=actor_path, progress=True)
        finally:
            observer.close()
        assert measured['windows'] == 3603 and len(measured['scenes']) == 150
        rows = read_actors(actor_path)
        pair_audit = pairing(rows, formal_rows, membership)
        summaries = {label: summarize(select(rows, group, membership)) for label, group in groups}
        checks = []
        if scale == 1:
            for label, group in groups:
                ref = summarize(select(formal_rows, group, membership))
                for metric in ('minFDE6', 'MR6', 'Top1FDE6'):
                    tolerance = 1e-4 if metric == 'Top1FDE6' else 1e-6
                    difference = abs(summaries[label][metric] - ref[metric])
                    checks.append({'group': label, 'metric': metric, 'difference': difference,
                        'tolerance': tolerance, 'passed': difference < tolerance})
            assert all(r['passed'] for r in checks), checks
        assert state_digest(model.state_dict()) == state_before
        assert all(p.grad is None for p in model.parameters())
        result = {'Lambda': scale, 'VALScenes': 150, 'VALWindows': 3603, 'FullHorizonActors': 54990}
        for label, values in summaries.items():
            for metric in ('minFDE6', 'MR6', 'Top1FDE6'):
                result[label + '_' + metric] = values[metric]
            result[label + '_Count'] = values['Count']
        sensitivity.append(result)
        write_csv(DIAG / 'stage4ad_bias_scale_sensitivity.csv', sorted(sensitivity, key=lambda r: r['Lambda']))
        atomic_json(DIAG / f'stage4ad_lambda_{name}_evaluation_audit.json', {
            'status': 'PASS', 'lambda': scale, 'metrics': summaries, **pair_audit,
            'VAL_windows': measured['windows'], 'VAL_scenes': len(measured['scenes']),
            'checkpoint_sha256': CHECKPOINT_SHA, 'state_before': state_before,
            'state_after': state_digest(model.state_dict()), 'no_grad': True, 'parameters_changed': False,
            'lambda1_formal_result_reconciliation': checks, 'actor_csv_sha256': sha256(actor_path),
            'elapsed_seconds': time.monotonic()-start, 'test_used': False,
            'checkpoint_selection': False, 'lambda0_is_Stage3B': False})
        print('STAGE4AD_LAMBDA_COMPLETE', scale, {k: v for k, v in result.items() if k in
            ('Overall_minFDE6', 'Vehicle_minFDE6', 'Pedestrian_minFDE6')}, flush=True)
    verify_frozen()
    atomic_json(DIAG / 'stage4ad_inference_completion.json', {'status': 'PASS',
        'lambdas': [0, .25, .5, .75, 1], 'official_VAL_scenes_per_lambda': 150,
        'official_VAL_windows_per_lambda': 3603, 'checkpoint_sha256': CHECKPOINT_SHA,
        'parameters_buffers_unchanged': True, 'optimizer_created': False, 'backward_called': False,
        'all_frozen_Stage4A_files_unchanged': True, 'Stage4A_scientific_conclusion': 'NOT SUPPORTED',
        'Ready_Stage4B': 'NO'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run', 'verify'))
    args = parser.parse_args()
    {'freeze': freeze_inputs, 'run': run, 'verify': verify_frozen}[args.action]()
