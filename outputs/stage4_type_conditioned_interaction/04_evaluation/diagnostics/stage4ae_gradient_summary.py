"""Paired batch bootstrap and descriptive class-gradient pressure summaries."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage4ae_common import (ROOT, DIAG, atomic_json, read_json, read_csv,
    write_csv, sha256, verify_frozen)
import numpy as np


def mean(rows, field):
    return float(np.mean([float(r[field]) for r in rows]))


def main():
    verify_frozen()
    completion = read_json(DIAG / 'stage4ae_gradient_completion.json')
    assert completion['status'] == 'PASS' and completion['batches'] == 100
    batch_path = DIAG / 'stage4ae_gradient_conflict_batches.csv'
    manifest_path = DIAG / 'stage4ae_gradient_batch_manifest.csv'
    rows, manifest = read_csv(batch_path), read_csv(manifest_path)
    assert len(rows) == 1100 and len(manifest) == 100
    assert sha256(batch_path) == completion['batch_metrics_sha256']
    assert sha256(manifest_path) == completion['manifest_sha256']
    identities = {int(r['BatchID']): (r['WindowIdentitySHA256'], r['InputTensorSHA256']) for r in manifest}
    assert set(identities) == set(range(1, 101))
    for r in rows:
        assert (r['WindowIdentitySHA256'], r['InputTensorSHA256']) == identities[int(r['BatchID'])]
        assert not int(r['NoneVTensors']) and not int(r['NonePTensors'])
        assert all(np.isfinite(float(r[k])) for k in ('CosVP', 'NormV', 'NormP', 'DotProduct',
            'NormRatio', 'EffectiveRatio', 'EffectiveV', 'EffectiveP'))
    grouped = {}
    for row in rows:
        grouped.setdefault((row['Model'], row['ParameterGroup']), []).append(row)
    summaries = []
    for (model, group), values in grouped.items():
        valid = [r for r in values if int(r['Valid'])]
        if group != 'Relation Module':
            assert len(valid) >= 95
        cos = np.array([float(r['CosVP']) for r in valid])
        assert len(cos)
        q = np.quantile(cos, [.1, .25, .5, .75, .9])
        summary = {'Model': model, 'ParameterGroup': group, 'ValidBatches': len(valid),
            'MeanCosVP': float(cos.mean()), 'MedianCosVP': float(q[2]), 'P10CosVP': float(q[0]),
            'P25CosVP': float(q[1]), 'P75CosVP': float(q[3]), 'P90CosVP': float(q[4]),
            'ConflictRate': float(np.mean(cos < 0)),
            'StrongConflictRate_0.1': float(np.mean(cos < -.1)),
            'StrongConflictRate_0.25': float(np.mean(cos < -.25)),
            'MeanNormV': mean(valid, 'NormV'), 'MeanNormP': mean(valid, 'NormP'),
            'MeanNormRatio': mean(valid, 'NormRatio'), 'MedianNormRatio': float(np.median([float(r['NormRatio']) for r in valid])),
            'MeanEffectiveV': mean(valid, 'EffectiveV'), 'MeanEffectiveP': mean(valid, 'EffectiveP'),
            'MeanEffectiveRatio': mean(valid, 'EffectiveRatio'),
            'MedianEffectiveRatio': float(np.median([float(r['EffectiveRatio']) for r in valid])),
            'RatioOfMeanEffectiveNorms': mean(valid, 'EffectiveV')/(mean(valid, 'EffectiveP')+1e-12),
            'FractionEffectiveRatioAbove1': float(np.mean([float(r['EffectiveRatio']) > 1 for r in valid])),
            'MeanDotProduct': mean(valid, 'DotProduct'), 'NoneVTensors': 0, 'NonePTensors': 0}
        summaries.append(summary)
    write_csv(ROOT / '06_tables/stage4ae_gradient_conflict_summary.csv', summaries)
    draws = np.random.default_rng(2022).integers(0, 100, size=(1000, 100))
    boot_rows, shifts = [], []
    for group in ('LocalEncoder', 'TypeEmbedding', 'GlobalInteractor', 'Decoder', 'ALL SHARED'):
        b = sorted(grouped[('Stage3B', group)], key=lambda r: int(r['BatchID']))
        c = sorted(grouped[('Stage4A', group)], key=lambda r: int(r['BatchID']))
        assert [r['BatchID'] for r in b] == [r['BatchID'] for r in c] == [str(i) for i in range(1, 101)]
        bv, cv = np.array([float(r['CosVP']) for r in b]), np.array([float(r['CosVP']) for r in c])
        delta = cv-bv
        sampled_b, sampled_c, sampled_delta = bv[draws].mean(1), cv[draws].mean(1), delta[draws].mean(1)
        blo, bhi = np.quantile(sampled_b, [.025, .975])
        clo, chi = np.quantile(sampled_c, [.025, .975])
        dlo, dhi = np.quantile(sampled_delta, [.025, .975])
        result = {'ParameterGroup': group, 'PairedBatches': 100,
            'Stage3BMeanCos': float(bv.mean()), 'Stage4AMeanCos': float(cv.mean()),
            'Stage3BMeanCI95Lower': float(blo), 'Stage3BMeanCI95Upper': float(bhi),
            'Stage4AMeanCI95Lower': float(clo), 'Stage4AMeanCI95Upper': float(chi),
            'DeltaCosMean': float(delta.mean()), 'DeltaCosMedian': float(np.median(delta)),
            'DeltaCosCI95Lower': float(dlo), 'DeltaCosCI95Upper': float(dhi),
            'FractionDeltaCosBelow0': float(np.mean(delta < 0)), 'Replicates': 1000, 'Seed': 2022}
        boot_rows.append(result)
        shifts.extend({'BatchID': int(br['BatchID']), 'ParameterGroup': group,
            'Stage3BCos': float(br['CosVP']), 'Stage4ACos': float(cr['CosVP']),
            'DeltaCos': float(cr['CosVP'])-float(br['CosVP'])} for br, cr in zip(b, c))
    write_csv(ROOT / '06_tables/stage4ae_gradient_shift_bootstrap.csv', boot_rows)
    write_csv(DIAG / 'stage4ae_paired_batch_gradient_shift.csv', shifts)
    atomic_json(DIAG / 'stage4ae_gradient_shift_bootstrap.json', {'status': 'PASS',
        'method': 'paired batch percentile bootstrap of mean class-gradient cosine',
        'replicates': 1000, 'seed': 2022, 'confidence': .95, 'same_resampled_batch_ids_for_both_models': True,
        'groups': boot_rows, 'formal_model_significance': False,
        'limitation': 'Sequential batches and overlapping scene windows are not independent scene clusters; optimization diagnostic only.',
        'batch_source_sha256': sha256(batch_path)})
    coordinates = {label: sum(int(r[label+'ValidCoordinates']) for r in manifest)
        for label in ('Vehicle', 'Pedestrian', 'Bicycle')}
    actors = {label: sum(int(r[label+'Targets']) for r in manifest) for label in coordinates}
    scene_tokens = set()
    windows = []
    import json
    for row in manifest:
        identities_ = json.loads(row['WindowIdentitiesJSON'])
        windows.extend(tuple(v) for v in identities_)
        scene_tokens.update(v[0] for v in identities_)
    assert len(windows) == len(set(windows)) == 1600
    distribution = []
    for label in coordinates:
        distribution.append({'Class': label, 'SupervisedActorWindows': actors[label],
            'ValidFutureCoordinates': coordinates[label], 'PooledActorShare': actors[label]/sum(actors.values()),
            'PooledCoordinateShare': coordinates[label]/sum(coordinates.values()),
            'MeanBatchActorShare': mean(manifest, label+'ActorShare'),
            'MeanBatchCoordinateShare': mean(manifest, label+'CoordinateShare')})
    write_csv(ROOT / '06_tables/stage4ae_supervision_distribution.csv', distribution)
    atomic_json(DIAG / 'stage4ae_supervision_distribution.json', {'status': 'PASS',
        'dataset': 'frozen official TRAIN700', 'sample_scope': 'first 100 jointly Vehicle/Pedestrian-supervised batches in deterministic index order',
        'sampled_scene_count': len(scene_tokens), 'sampled_window_count': 1600,
        'scanned_loader_batch_count': int(manifest[-1]['LoaderBatchIndex'])+1,
        'batches': 100, 'batch_size': 16, 'class_counts': distribution,
        'coordinate_definition': '2 scalar spatial coordinates per valid supervised future timestep',
        'share_reused_exactly_for_both_models': True, 'full_TRAIN700_distribution_estimated': False,
        'proxy': 'class valid-coordinate share times class-mean total-loss gradient norm',
        'proxy_is_exact_mixed_gradient_contribution': False,
        'mean_of_ratios_is_not_ratio_of_means': True, 'manifest_sha256': sha256(manifest_path)})
    pair_rows = read_csv(DIAG / 'stage4ae_relation_embedding_row_batches.csv')
    pair_summary = []
    for pair in ('V<-V', 'V<-P', 'V<-B', 'P<-V', 'P<-P', 'P<-B', 'B<-V', 'B<-P', 'B<-B'):
        cell = [r for r in pair_rows if r['Pair'] == pair]
        valid = [r for r in cell if int(r['Valid'])]
        pair_summary.append({'Pair': pair, 'Batches': len(cell), 'ValidBatches': len(valid),
            'MeanNormV': mean(cell, 'NormV'), 'MeanNormP': mean(cell, 'NormP'),
            'MeanCosVP': mean(valid, 'CosVP') if valid else None,
            'ConflictRate': float(np.mean([float(r['CosVP']) < 0 for r in valid])) if valid else None,
            'ZeroNormVBatches': sum(float(r['NormV']) == 0 for r in cell),
            'ZeroNormPBatches': sum(float(r['NormP']) == 0 for r in cell)})
    write_csv(ROOT / '06_tables/stage4ae_relation_embedding_row_summary.csv', pair_summary)
    atomic_json(DIAG / 'stage4ae_gradient_summary_audit.json', {'status': 'PASS',
        'same_100_batch_identity_and_input_tensor_hashes': True, 'paired_batch_rows': len(rows),
        'primary_shared_groups_valid_batches': 100, 'NaN': 0, 'Inf': 0,
        'summary_rows': len(summaries), 'embedding_row_descriptive_only': True,
        'effective_proxy_not_exact_mixed_gradient_decomposition': True,
        'bootstrap_is_optimization_diagnostic_only': True, 'source_sha256': sha256(batch_path)})
    print('STAGE4AE_GRADIENT_SUMMARY=PASS', flush=True)


if __name__ == '__main__':
    main()
