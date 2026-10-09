"""Fair synchronized CUDA timings on identical cached inputs for all four heads.

This benchmark excludes feature construction, frozen HiVT prediction, transfer,
R2 routing, and evaluation labels. It is not an end-to-end latency measurement.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_protocol'))
from stage14a_common import *
sys.path.insert(0, str(ROOT / '05_evaluation'))
from stage14a_evaluate import frozen_gate

WARMUP = 5
REPEATS = 20
ACTORS_PER_FOLD = 1024
BENCHMARK_BATCH = 128


@torch.no_grad()
def main():
    frozen_gate()
    seed()
    verify(history=True)
    audit = read_json(ROOT / '05_evaluation/stage14a_identity_audit.json')
    assert audit['Status'] == 'PASS' and audit['Models'] == list(MODELS)
    rows, all_ids = [], {}
    for fold in (1, 2, 3):
        store = Store(fold)
        # Fixed first 1024 original OuterTest actor rows; no error-based selection.
        ids = indices(fold, 'OuterTest')[:ACTORS_PER_FOLD]
        assert len(ids) == ACTORS_PER_FOLD
        all_ids[str(fold)] = ids.tolist()
        inputs = []
        for start in range(0, len(ids), BENCHMARK_BATCH):
            args, _, _ = store.batch(ids[start:start + BENCHMARK_BATCH], part='OuterTest')
            inputs.append(args)
        models, counts, sources = {}, {}, {}
        for name in MODELS:
            if name.startswith('NG-'):
                model = fresh(fold)
                path = cp_path(fold, name[-1])
            else:
                model = SparseGraphReranker('G1', seed=2022 + 100 * (fold - 1)).cuda()
                path = S11B / f'04_checkpoints/fold{fold}/{name[-1]}_best.pt'
            counts[name] = {'TotalParameters': sum(p.numel() for p in model.parameters()),
                            'OptimizationTrainableParameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            saved = torch.load(path, map_location='cpu', weights_only=False)
            assert saved['Fold'] == fold and saved['Model'] == name[-1]
            assert saved['normalization_sha256'] == sha256(store.normpath)
            model.load_state_dict(saved['state_dict'])
            models[name] = model.eval().requires_grad_(False)
            sources[name] = {'Path': str(path.relative_to(PROJECT)), 'SHA256': sha256(path)}
        before = {name: state_sha(model) for name, model in models.items()}
        for _ in range(WARMUP):
            for name in MODELS:
                for args in inputs:
                    models[name](*args)
        torch.cuda.synchronize()
        durations = {name: [] for name in MODELS}
        peaks = {name: [] for name in MODELS}
        # Rotating a fixed order reduces a consistent first-model timing bias.
        for repeat in range(REPEATS):
            order = MODELS[repeat % 4:] + MODELS[:repeat % 4]
            for name in order:
                torch.cuda.synchronize()
                baseline = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for args in inputs:
                    output = models[name](*args)
                torch.cuda.synchronize()
                durations[name].append(time.perf_counter() - started)
                peaks[name].append(max(0, torch.cuda.max_memory_allocated() - baseline))
                assert torch.isfinite(output['mode_logits']).all() and torch.isfinite(output['mode_prob']).all()
                del output
        assert before == {name: state_sha(model) for name, model in models.items()}
        for name in MODELS:
            ms = np.asarray(durations[name]) * 1000. / len(ids)
            rows.append({'Fold': fold, 'Model': name, **counts[name], 'FrozenInferenceTrainableParameters': 0,
                         'BenchmarkActors': len(ids), 'BatchSize': BENCHMARK_BATCH,
                         'WarmupRepeats': WARMUP, 'MeasuredRepeats': REPEATS,
                         'MeanMSPerActor': float(ms.mean()), 'MedianMSPerActor': float(np.median(ms)),
                         'StdMSPerActor': float(ms.std(ddof=1)),
                         'MinMSPerActor': float(ms.min()), 'MaxMSPerActor': float(ms.max()),
                         'PeakIncrementalAllocatedGPUMemoryBytes': max(peaks[name]),
                         'CheckpointSHA256': sources[name]['SHA256'],
                         'Device': torch.cuda.get_device_name(), 'Precision': 'FP32',
                         'Scope': 'identical on-device cached seven-argument input; pure ranking-head forward; full graph retained',
                         'Excluded': 'HiVT/feature construction/transfer/R2 routing/GT/evaluation'})
        print('STAGE14A_EFFICIENCY_FOLD_PASS', fold, flush=True)
        del models, inputs, store
        torch.cuda.empty_cache()
    dump('07_diagnostics/stage14a_efficiency.csv', rows)
    atomic_json(ROOT / '07_diagnostics/stage14a_efficiency_audit.json', {
        'Status': 'PASS', 'Models': list(MODELS), 'Folds': 3, 'SameInputsForEveryModel': True,
        'InputSelection': 'first1024 original OuterTest actor indices per fold, not error-selected',
        'ActorIndices': all_ids, 'BatchSize': BENCHMARK_BATCH, 'WarmupRepeats': WARMUP,
        'MeasuredRepeats': REPEATS, 'CUDA_Synchronized': True, 'GTInTimedForward': False,
        'AllCheckpointsFrozen': True, 'ModelStatesUnchanged': True,
        'IncludesFrozenPredictor': False, 'IncludesFeaturePreparation': False,
        'IncludesR2Routing': False, 'EndToEndLatencyClaim': False,
        'IdentityAuditSHA256': sha256(ROOT / '05_evaluation/stage14a_identity_audit.json')})
    frozen_gate()
    verify(history=True)
    print('STAGE14A_EQUAL_INPUT_EFFICIENCY_PASS', flush=True)


if __name__ == '__main__':
    main()
