"""Registered paired scene inference for the four frozen controlled heads.

No fitting, checkpoint selection, or actor-independent resampling occurs here.
Four Overall Top1FDE contrasts form the preregistered Bonferroni family.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_protocol'))
from stage14a_common import *

COMPARISONS = (('A', 'NG-C', 'NG-A'), ('B', 'G-A', 'NG-A'),
               ('C', 'G-C', 'NG-C'), ('D', 'G-C', 'G-A'))
INTERACTION = '(G-C-G-A)-(NG-C-NG-A)'


def main():
    verify(history=True)
    audit = read_json(ROOT / '05_evaluation/stage14a_identity_audit.json')
    assert audit['Status'] == 'PASS' and audit['Models'] == list(MODELS)
    registration = read_json(PROTOCOL)['bootstrap']
    assert registration['replicates'] == 2000 and registration['seed'] == 2022
    assert registration['family_size'] == 4
    assert registration['adjusted_percentiles'] == [.625, 99.375]
    assert registration['co_primary'] == [f'{new}-{old}' for _, new, old in COMPARISONS]
    dest = ROOT / '05_evaluation/cache'
    for name, digest in audit['cache_files'].items():
        assert sha256(dest / name) == digest
    f = pd.read_csv(dest / 'stage14a_oof_actor_records.csv', dtype={'future_mask_bits': str})
    values = np.load(dest / 'stage14a_oof_metrics.npy', mmap_mode='r')
    assert len(f) == 260151 and audit['Fields'][0] == 'Top1FDE'
    fd = np.asarray(values[:, :, 0])
    assert fd.shape == (len(f), 4)
    scene_order = []
    for fold in (1, 2, 3):
        scenes = sorted(split(fold)['OuterTest'])
        assert len(scenes) == 210
        assert set(f.loc[f.Fold == fold, 'scene_token']) == set(scenes)
        scene_order.extend(scenes)
    assert len(set(scene_order)) == 630
    lookup = {scene: i for i, scene in enumerate(scene_order)}
    codes = f.scene_token.map(lookup).to_numpy()
    rng = np.random.default_rng(2022)
    weights = np.zeros((2000, 630), np.int32)
    for replicate in range(2000):
        for fold in range(3):
            weights[replicate, fold * 210:(fold + 1) * 210] = np.bincount(
                rng.integers(0, 210, 210), minlength=210)
    assert np.all(weights.reshape(2000, 3, 210).sum(-1) == 210)
    weight_path = ROOT / '06_bootstrap/cache/stage14a_scene_bootstrap_weights.npy'
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(weight_path, weights)
    # The paired draw scheme must reproduce the original Stage11B draw matrix.
    historical_weights_path = S11B / '06_bootstrap/stage11b_scene_bootstrap_weights.npy'
    if historical_weights_path.exists():
        assert np.array_equal(weights, np.load(historical_weights_path, mmap_mode='r'))
    rows, foldrows, meanrows, interactionrows = [], [], [], []
    masks = groups(f)
    for group, keep in masks.items():
        count = int(keep.sum())
        assert count > 0
        scene_count = int(len(set(f.loc[keep, 'scene_token'])))
        denominator = np.bincount(codes[keep], minlength=630).astype(np.float64)
        bootstrap_denominator = weights @ denominator
        assert (bootstrap_denominator > 0).all()
        sums = np.stack([np.bincount(codes[keep], weights=fd[keep, j], minlength=630)
                         for j in range(4)], -1)
        means = weights @ sums / bootstrap_denominator[:, None]
        assert np.allclose(sums.sum(0) / denominator.sum(), fd[keep].mean(0), atol=1e-12, rtol=0.)
        for j, name in enumerate(MODELS):
            ci = np.quantile(means[:, j], [.025, .975])
            meanrows.append({'Group': group, 'Model': name, 'Count': count,
                             'SceneCount': scene_count, 'Top1FDE': float(fd[keep, j].mean()),
                             'CI95Lower': float(ci[0]), 'CI95Upper': float(ci[1]),
                             'Units': 'm', 'Interpretation': 'descriptive model mean; not a primary comparison'})
        for label, new, old in COMPARISONS:
            a, b = MODELS.index(new), MODELS.index(old)
            delta = fd[keep, a] - fd[keep, b]
            draws = means[:, a] - means[:, b]
            ci = np.quantile(draws, [.025, .975])
            primary = group == 'Overall'
            adjusted = np.quantile(draws, [.00625, .99375]) if primary else (None, None)
            row = {'ComparisonID': label, 'Comparison': f'{new}-{old}', 'Group': group,
                   'Count': count, 'SceneCount': scene_count, 'DeltaTop1FDE': float(delta.mean()),
                   'CI95Lower': float(ci[0]), 'CI95Upper': float(ci[1]), 'CoPrimary': primary,
                   'BonferroniFamilySize': 4 if primary else None,
                   'AdjustedIndividualCoverage': .9875 if primary else None,
                   'BonferroniCILower': None if adjusted[0] is None else float(adjusted[0]),
                   'BonferroniCIUpper': None if adjusted[1] is None else float(adjusted[1]),
                   'Units': 'm', 'Direction': 'negative favors first model',
                   'Interpretation': 'registered co-primary' if primary else 'exploratory; no multiplicity claim'}
            for fold in (1, 2, 3):
                within = keep & np.asarray(f.Fold == fold)
                assert within.any()
                fold_delta = float((fd[within, a] - fd[within, b]).mean())
                row[f'Fold{fold}DeltaTop1FDE'] = fold_delta
                row[f'Fold{fold}Count'] = int(within.sum())
                foldrows.append({'ComparisonID': label, 'Comparison': f'{new}-{old}',
                                 'Group': group, 'Fold': fold, 'Count': int(within.sum()),
                                 'SceneCount': int(len(set(f.loc[within, 'scene_token']))),
                                 'DeltaTop1FDE': fold_delta, 'Units': 'm'})
            row['NegativeDirectionFolds'] = sum(row[f'Fold{fold}DeltaTop1FDE'] < 0 for fold in (1, 2, 3))
            rows.append(row)
        individual = (fd[keep, 3] - fd[keep, 2]) - (fd[keep, 1] - fd[keep, 0])
        draws = (means[:, 3] - means[:, 2]) - (means[:, 1] - means[:, 0])
        ci = np.quantile(draws, [.025, .975])
        interaction = {'ComparisonID': 'E', 'Comparison': INTERACTION, 'Group': group,
                       'Count': count, 'SceneCount': scene_count,
                       'DeltaTop1FDE': float(individual.mean()),
                       'CI95Lower': float(ci[0]), 'CI95Upper': float(ci[1]),
                       'Units': 'm', 'Interpretation': 'descriptive interaction contrast; no causal or primary inference'}
        for fold in (1, 2, 3):
            within = keep & np.asarray(f.Fold == fold)
            interaction[f'Fold{fold}DeltaTop1FDE'] = float(((fd[within, 3] - fd[within, 2]) -
                                                        (fd[within, 1] - fd[within, 0])).mean())
            interaction[f'Fold{fold}Count'] = int(within.sum())
        interactionrows.append(interaction)
    dump('06_bootstrap/stage14a_bootstrap_ci.csv', rows)
    dump('06_bootstrap/stage14a_fold_comparisons.csv', foldrows)
    dump('06_bootstrap/stage14a_model_mean_ci.csv', meanrows)
    dump('06_bootstrap/stage14a_interaction_ci.csv', interactionrows)
    primary_rows = {row['ComparisonID']: row for row in rows if row['Group'] == 'Overall'}
    def directional_gate(row):
        return {'PointImproves': row['DeltaTop1FDE'] < 0,
                'AdjustedCIExcludesZeroInImprovementDirection': row['BonferroniCIUpper'] < 0,
                'AtLeastTwoNegativeFolds': row['NegativeDirectionFolds'] >= 2,
                'NegativeDirectionFolds': row['NegativeDirectionFolds'],
                'DeltaTop1FDE': row['DeltaTop1FDE'],
                'AdjustedCILower': row['BonferroniCILower'],
                'AdjustedCIUpper': row['BonferroniCIUpper']}
    evidence = {label: directional_gate(row) for label, row in primary_rows.items()}
    for label, item in evidence.items():
        item['RegisteredConditionsAllMet'] = all(item[key] for key in
            ('PointImproves', 'AdjustedCIExcludesZeroInImprovementDirection', 'AtLeastTwoNegativeFolds'))
        item['Comparison'] = primary_rows[label]['Comparison']
    evidence['GraphIncrementRegisteredConditionsMet'] = evidence['C']['RegisteredConditionsAllMet']
    evidence['LossAcrossStructuresRegisteredConditionsMet'] = (
        evidence['A']['RegisteredConditionsAllMet'] and evidence['D']['RegisteredConditionsAllMet'])
    evidence['Scope'] = 'statistical evidence for final report; internal ranking OOF, not independent end-to-end validation'
    atomic_json(ROOT / '06_bootstrap/stage14a_decision_evidence.json', evidence)
    atomic_json(ROOT / '06_bootstrap/stage14a_bootstrap_audit.json', {
        'Status': 'PASS', 'Replicates': 2000, 'Seed': 2022, 'Folds': 3,
        'Unit': 'whole scenes with every actor/window paired across all four models',
        'WithinFoldScenesDrawn': 210, 'SceneOrder': scene_order,
        'ReplicateWeightsSHA256': sha256(weight_path),
        '95DescriptivePercentiles': [2.5, 97.5], 'BonferroniFamilySize': 4,
        'AdjustedIndividualCoverage': .9875, 'AdjustedPercentiles': [.625, 99.375],
        'CoPrimary': [f'{new}-{old} Overall Top1FDE' for _, new, old in COMPARISONS],
        'Interaction': 'descriptive paired 95%CI only', 'OtherGroups': 'exploratory',
        'FoldCheckpointsReselected': False, 'PointEstimatesReproduced': True,
        'ActorIndependentBootstrap': False, 'official_VAL_or_test_used': False,
        'IdentityAuditSHA256': sha256(ROOT / '05_evaluation/stage14a_identity_audit.json'),
        'Evidence': evidence})
    verify(history=True)
    print('STAGE14A_PAIRED_SCENE_BOOTSTRAP_PASS', flush=True)


if __name__ == '__main__':
    main()
