"""Paired motion-bin audit of frozen actor CSVs; this file performs no inference."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage4ae_common import (DIAG, ROOT, STAGE3, FIELDS, BINS, paired_actors,
    summarize, atomic_json, write_csv, read_json, sha256, BASE_ACTORS, ACTORS,
    MEMBERSHIP, verify_frozen)
import numpy as np


def main():
    verify_frozen()
    old, new, membership, bins = paired_actors()
    scenes = sorted(read_json(ROOT / '04_evaluation/stage4a_metrics.json')['scenes'])
    assert len(scenes) == 150
    scene_index = {token: i for i, token in enumerate(scenes)}
    rng = np.random.default_rng(2022)
    draws = rng.integers(0, 150, size=(1000, 150))
    multiplicities = np.stack([np.bincount(row, minlength=150) for row in draws])
    metrics, cross, ci_rows, attribution = [], [], [], []
    result = {'status': 'PASS', 'method': 'paired scene-cluster percentile bootstrap',
        'official_VAL_scenes': 150, 'replicates': 1000, 'seed': 2022, 'confidence': .95,
        'delta': 'Stage4A - Stage3B', 'weighting': 'equal actor-window within pooled resampled scene clusters',
        'frozen_B_actor_sha256': sha256(BASE_ACTORS), 'frozen_C_actor_sha256': sha256(ACTORS),
        'frozen_membership_sha256': sha256(MEMBERSHIP), 'inference_performed': False,
        'full_horizon_actors': 54990, 'full_horizon_pedestrian': 12002, 'bins': {}}
    for (lower, upper, expected), (name, keys) in zip(BINS, bins.items()):
        b, c = [old[k] for k in keys], [new[k] for k in keys]
        sb, sc = summarize(b), summarize(c)
        row = {'MotionBin': name, 'LowerInclusive_m': lower, 'UpperExclusive_m': upper,
            'Count': len(keys), 'UniqueInstances': sb['UniqueInstances'], 'UniqueScenes': sb['UniqueScenes']}
        for metric in FIELDS:
            row.update({f'Stage3B_{metric}': sb[metric], f'Stage4A_{metric}': sc[metric],
                f'Delta_{metric}': sc[metric]-sb[metric]})
        metrics.append(row)
        counts = np.zeros(150)
        sums = np.zeros((150, 3))
        for key in keys:
            i = scene_index[key[0]]
            counts[i] += 1
            sums[i] += [float(new[key][metric])-float(old[key][metric])
                for metric in ('minFDE6', 'Top1FDE6', 'minADE6')]
        denominator = multiplicities @ counts
        valid = denominator > 0
        assert valid.sum() >= 950, 'Too many empty resampled cells; stop'
        samples = (multiplicities[valid] @ sums) / denominator[valid, None]
        group_ci = {}
        for j, metric in enumerate(('minFDE6', 'Top1FDE6', 'minADE6')):
            point = float(sums[:, j].sum()/len(keys))
            lo, hi = np.quantile(samples[:, j], [.025, .975])
            group_ci[metric] = {'estimate': point, 'CI95': [float(lo), float(hi)],
                'valid_replicates': int(valid.sum()), 'empty_replicates': int((~valid).sum())}
            ci_rows.append({'MotionBin': name, 'Metric': metric, 'Count': len(keys),
                'UniqueInstances': sb['UniqueInstances'], 'UniqueScenes': sb['UniqueScenes'],
                'Delta_C_minus_B': point, 'CI95Lower': float(lo), 'CI95Upper': float(hi),
                'Replicates': 1000, 'ValidReplicates': int(valid.sum()), 'Seed': 2022,
                'ResamplingUnit': 'official VAL scene'})
        result['bins'][name] = {'Count': len(keys), 'UniqueInstances': sb['UniqueInstances'],
            'UniqueScenes': sb['UniqueScenes'], 'metrics': group_ci}
        for flag, label in ((True, 'hetero-20m'), (False, 'non-hetero-20m')):
            cell = [k for k in keys if (membership[k]['Heterogeneous-20m'] == '1') == flag]
            cell_row = {'MotionBin': name, 'Context': label, 'Count': len(cell),
                'UniqueScenes': len({k[0] for k in cell}), 'UniqueInstances': len({k[2] for k in cell})}
            if cell:
                bs, cs = summarize([old[k] for k in cell]), summarize([new[k] for k in cell])
                delta = cs['minFDE6']-bs['minFDE6']
                cell_row.update(Stage3B_FDE=bs['minFDE6'], Stage4A_FDE=cs['minFDE6'], DeltaFDE=delta,
                    Top1FDEDelta=cs['Top1FDE6']-bs['Top1FDE6'],
                    ContributionToPedestrianOverallDelta=delta*len(cell)/12002)
            else:
                cell_row.update(Stage3B_FDE=None, Stage4A_FDE=None, DeltaFDE=None,
                    Top1FDEDelta=None, ContributionToPedestrianOverallDelta=0.)
            cross.append(cell_row)
        attribution.append({'MotionBin': name, 'Count': len(keys), 'PedestrianActorShare': len(keys)/12002,
            'DeltaFDE': row['Delta_minFDE6'],
            'ContributionToPedestrianOverallDelta': row['Delta_minFDE6']*len(keys)/12002})
        print('STAGE4AE_MOTION_BIN', name, row, group_ci['minFDE6'], flush=True)
    total_delta = sum(r['ContributionToPedestrianOverallDelta'] for r in attribution)
    overall_b = summarize([old[k] for keys in bins.values() for k in keys])['minFDE6']
    overall_c = summarize([new[k] for keys in bins.values() for k in keys])['minFDE6']
    assert abs(total_delta-(overall_c-overall_b)) < 1e-12
    write_csv(ROOT / '06_tables/stage4ae_pedestrian_motion_bin_metrics.csv', metrics)
    write_csv(ROOT / '06_tables/stage4ae_pedestrian_motion_bin_bootstrap.csv', ci_rows)
    write_csv(ROOT / '06_tables/stage4ae_pedestrian_motion_context_cross.csv', cross)
    write_csv(ROOT / '06_tables/stage4ae_pedestrian_degradation_attribution.csv', attribution)
    atomic_json(DIAG / 'stage4ae_pedestrian_motion_bin_bootstrap.json', result)
    atomic_json(DIAG / 'stage4ae_motion_audit.json', {'status': 'PASS', 'fixed_counts': [len(k) for k in bins.values()],
        'sum_counts': 12002, 'strict_paired_full_horizon_actors': 54990, 'no_inference': True,
        'membership_recomputed': False, 'pedestrian_overall_B_FDE': overall_b,
        'pedestrian_overall_C_FDE': overall_c, 'overall_delta': total_delta,
        'low_motion_lt5_contribution': sum(r['ContributionToPedestrianOverallDelta'] for r in attribution[:3]),
        'moving_ge5_contribution': sum(r['ContributionToPedestrianOverallDelta'] for r in attribution[3:]),
        'context_metrics_descriptive_only': True, 'small_bins_require_unique_scene_caution': ['1-2m', '10-20m'],
        'test_used': False})


if __name__ == '__main__':
    main()
