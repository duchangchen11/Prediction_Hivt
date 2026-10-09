"""Unmodified Stage11B switch-cost definitions on unified Stage14A actor rows."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_protocol'))
from stage14a_common import *

COMPARISONS = (('NG-C', 'NG-A'), ('G-A', 'NG-A'), ('G-C', 'NG-C'), ('G-C', 'G-A'))
# Extract the exact historical function without importing the historical script's
# write hook or executing its old result-specific scientific decision logic.
_source = S11B / '06_bootstrap/stage11b_analyze.py'
_tree = ast.parse(_source.read_text())
_body = [node for node in _tree.body if isinstance(node, ast.FunctionDef)
         and node.name == 'switch_statistics']
assert len(_body) == 1
exec(compile(ast.fix_missing_locations(ast.Module(body=_body, type_ignores=[])), str(_source), 'exec'))


def main():
    verify(history=True)
    audit = read_json(ROOT / '05_evaluation/stage14a_identity_audit.json')
    assert audit['Status'] == 'PASS' and audit['Models'] == list(MODELS)
    dest = ROOT / '05_evaluation/cache'
    for name, digest in audit['cache_files'].items():
        assert sha256(dest / name) == digest
    f = pd.read_csv(dest / 'stage14a_oof_actor_records.csv', dtype={'future_mask_bits': str})
    values = np.load(dest / 'stage14a_oof_metrics.npy', mmap_mode='r')
    p = np.load(dest / 'stage14a_oof_probabilities.npy', mmap_mode='r')
    fd = np.asarray(values[:, :, 0])
    modes = p.argmax(-1)
    masks = groups(f)
    rows, foldrows, probabilities = [], [], []
    for group, keep in masks.items():
        assert keep.any()
        for new, old in COMPARISONS:
            a, b = MODELS.index(new), MODELS.index(old)
            delta = fd[keep, a] - fd[keep, b]
            changed = modes[keep, a] != modes[keep, b]
            assert np.all(delta[~changed] == 0.)
            statistics = switch_statistics(delta, changed)
            assert statistics['ImprovedCount'] + statistics['WorsenedCount'] + statistics['NeutralCount'] == int(keep.sum())
            assert np.isclose(statistics['GrossHarm'] - statistics['GrossGain'], delta.sum(), atol=1e-9, rtol=0.)
            rows.append({'Comparison': f'{new}-{old}', 'Group': group, 'Count': int(keep.sum()),
                         'SceneCount': len(set(f.loc[keep, 'scene_token'])), **statistics,
                         'HighCostHarmCount': statistics['Top10HarmCount'],
                         'HighCostHarmSum': statistics['Top10HarmSum'],
                         'HighCostDefinition': 'largest ceil(0.1*positive-harm actors) per comparison/group',
                         'NetFDEDeltaSum': float(delta.sum()),
                         'Units': 'FDE and gain/harm m; count actors/windows',
                         'Interpretation': 'descriptive; subset/tail membership differs between comparisons'})
            for fold in (1, 2, 3):
                within = keep & np.asarray(f.Fold == fold)
                change = modes[within, a] != modes[within, b]
                dd = fd[within, a] - fd[within, b]
                foldrows.append({'Fold': fold, 'Comparison': f'{new}-{old}', 'Group': group,
                                 'Count': int(within.sum()), **switch_statistics(dd, change)})
        for j, model in enumerate(MODELS):
            entropy = np.asarray(values[keep, j, 8])
            maximum = np.asarray(values[keep, j, 9])
            probabilities.append({'Group': group, 'Model': model, 'Count': int(keep.sum()),
                                  'PredictionEntropy': float(entropy.mean()),
                                  'Top1Probability': float(maximum.mean()),
                                  'Top1Top2Margin': float(values[keep, j, 10].mean()),
                                  'OracleModeProbability': float(values[keep, j, 11].mean()),
                                  'SoftCE': float(values[keep, j, 5].mean()),
                                  'EntropyMedian': float(np.median(entropy)),
                                  'Top1ProbabilityMedian': float(np.median(maximum)),
                                  'Top1ProbabilityP90': float(np.quantile(maximum, .9)),
                                  'ProbabilityAbove0p9Fraction': float((maximum > .9).mean()),
                                  'ExpectedRegret': float(values[keep, j, 6].mean()),
                                  'NormalizedExpectedRegret': float(values[keep, j, 7].mean())})
    dump('07_diagnostics/stage14a_mode_switch.csv', rows)
    dump('07_diagnostics/stage14a_mode_switch_by_fold.csv', foldrows)
    dump('07_diagnostics/stage14a_probability_statistics.csv', probabilities)
    # Disjoint observable vehicle-state subsets explain their contribution to the
    # full Vehicle delta, without selecting an advantageous stationary subgroup.
    vehicle = masks['Vehicle']
    membership = np.stack([masks[group] for group in ('MovingVehicle', 'StoppedVehicle', 'ParkedVehicle')])
    assert np.all(membership.sum(0) <= 1)
    other = vehicle & ~membership.any(0)
    contribution = []
    for new, old in COMPARISONS:
        a, b = MODELS.index(new), MODELS.index(old)
        for group, keep in [('MovingVehicle', masks['MovingVehicle']),
                            ('StoppedVehicle', masks['StoppedVehicle']),
                            ('ParkedVehicle', masks['ParkedVehicle']), ('OtherVehicleState', other)]:
            delta = fd[keep, a] - fd[keep, b]
            contribution.append({'Comparison': f'{new}-{old}', 'Group': group,
                                 'Count': int(keep.sum()), 'VehicleDenominator': int(vehicle.sum()),
                                 'MeanDeltaTop1FDE': float(delta.mean()) if len(delta) else None,
                                 'DeltaSum': float(delta.sum()),
                                 'ContributionToVehicleMeanDelta': float(delta.sum() / vehicle.sum()),
                                 'Units': 'm', 'Interpretation': 'additive descriptive attribution by disjoint current vehicle state'})
        pieces = [row['ContributionToVehicleMeanDelta'] for row in contribution if row['Comparison'] == f'{new}-{old}']
        assert np.isclose(sum(pieces), (fd[vehicle, a] - fd[vehicle, b]).mean(), atol=1e-12, rtol=0.)
    dump('07_diagnostics/stage14a_motion_contribution.csv', contribution)
    atomic_json(ROOT / '07_diagnostics/stage14a_mode_switch_audit.json', {
        'Status': 'PASS', 'OriginalDefinitionPath': str(_source.relative_to(PROJECT)),
        'OriginalDefinitionSHA256': sha256(_source), 'OriginalFunctionASTReused': True,
        'IdentityAuditSHA256': sha256(ROOT / '05_evaluation/stage14a_identity_audit.json'),
        'Comparisons': [f'{new}-{old}' for new, old in COMPARISONS],
        'RequiredGroupsPresent': all(group in masks for group in
            ('Overall', 'Vehicle', 'Pedestrian', 'Bicycle', 'MovingVehicle', 'StoppedVehicle', 'ParkedVehicle')),
        'UnchangedModesHaveZeroFDEDelta': True, 'GrossGainHarmNetAccounting': True,
        'VehicleStateContributionsSumToVehicleDelta': True,
        'HighCostDefinition': 'largest10% of strictly positive harms, ceil count; separately defined comparison/group tails',
        'NoCausalClaim': True, 'official_VAL_or_test_used': False})
    verify(history=True)
    print('STAGE14A_MODE_SWITCH_DIAGNOSTICS_PASS', flush=True)


if __name__ == '__main__':
    main()
