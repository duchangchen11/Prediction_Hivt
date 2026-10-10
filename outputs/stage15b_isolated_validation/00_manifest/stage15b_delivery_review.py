"""Post-experiment artifact consistency review; no fitting or new hypothesis tests."""
from pathlib import Path
import csv, json, hashlib, math
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    names = [
        'stage15b_predictor_training_report.md', 'stage15b_predictor_checkpoint_audit.md',
        'stage15b_candidate_provenance.md', 'stage15b_ranking_training_report.md',
        'stage15b_end_to_end_metrics.csv', 'stage15b_ablation_metrics.csv',
        'stage15b_bootstrap_ci.csv', 'stage15b_failure_cases.md', 'stage15b_final_report.md']
    assert all((ROOT/n).is_file() and (ROOT/n).stat().st_size for n in names)
    assert read(ROOT/'00_manifest/stage15b_final_audit.json')['Status'] == 'PASS'
    usage = read(ROOT/'01_data_isolation/stage15b_actual_data_usage.json')
    assert usage['Status'] == 'PASS' and sum(x['OptimizerSteps'] for x in usage['Folds']) == 36500
    for registry in ('stage15b_registration.json', 'stage15b_ranking_registration.json', 'stage15b_evaluation_registration.json'):
        for path, expected in read(ROOT/'00_manifest'/registry)['sources'].items():
            if path == '00_manifest/stage15b_finalize.py':
                correction = read(ROOT/'00_manifest/stage15b_report_formatter_correction.json')
                assert sha(ROOT/correction['PreservedInitialSource']) == expected == correction['InitialSourceSHA256']
                assert sha(ROOT/path) == correction['CurrentSourceSHA256']
            else:
                assert sha(ROOT/path) == expected, path
    with (ROOT/'stage15b_end_to_end_metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    pooled = {(x['Group'],x['Model']):x for x in metrics if x['Fold'] == 'Pooled'}
    for (group, model), row in pooled.items():
        parts = [x for x in metrics if x['Group'] == group and x['Model'] == model and x['Fold'] != 'Pooled']
        count = sum(int(x['Count']) for x in parts)
        assert count == int(row['Count'])
        for field in ('Top1FDE','Top1ADE','minFDE6','HitRate'):
            mean = sum(int(x['Count'])*float(x[field]) for x in parts)/count
            assert math.isclose(mean,float(row[field]),rel_tol=1e-9,abs_tol=1e-9)
    with (ROOT/'stage15b_bootstrap_ci.csv').open() as f:
        primary = [x for x in csv.DictReader(f) if x['CoPrimary'] == 'True']
    decisions = read(ROOT/'09_statistics/stage15b_decisions.json')['Decisions']
    assert len(primary) == len(decisions) == 3
    for x in primary:
        a,b = next((a,b) for a,b in [('G-C','NG-C'),('G-C','Matched-NG-C'),('NG-C','NG-A')] if a+'-'+b == x['Comparison'])
        delta = float(pooled['Overall',a]['Top1FDE'])-float(pooled['Overall',b]['Top1FDE'])
        assert abs(delta-float(x['Delta'])) < 1e-6
        expected = 'SUPPORTED' if float(x['Delta']) < 0 and float(x['BonferroniCIUpper']) < 0 and int(x['NegativeFolds']) >= 2 else 'NOT_SUPPORTED'
        stored = next(v for v in decisions.values() if v['Evidence']['Comparison'] == x['Comparison'])
        assert stored['Decision'] == expected
        assert int(x['Replicates']) == 2000 and int(x['Seed']) == 2022 and int(float(x['FamilySize'])) == 3
    result = {'Status':'PASS','RequiredArtifacts':{n:sha(ROOT/n) for n in names},
        'FrozenFittingAndStatisticalSourcesUnchanged':True,'FormalOptimizerBatchesAudited':36500,
        'WeightedFoldToPooledConsistency':True,'PrimaryDecisionConsistency':True,
        'NoFittingOrNewHypothesisTest':True,'HistoricalSHAReview':'already PASS in final_audit; no redundant rerun'}
    (ROOT/'00_manifest/stage15b_delivery_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DELIVERY_REVIEW_PASS')


if __name__ == '__main__':
    main()
