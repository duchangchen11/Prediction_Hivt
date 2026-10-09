"""Evaluate all four controlled heads only after all six new heads are frozen.

Graph results are historical Stage11B A/C scores, joined by the full actor key.
Future labels are read as separate evaluation arrays and never passed to heads.
The metric operations are the original Stage11B operations, including FP32
probability/cost arithmetic before conversion to FP64 for actor aggregation.
"""
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '00_protocol'))
from stage14a_common import *

FIELDS = ('Top1FDE', 'Top1ADE', 'OracleGap', 'HitRate', 'MRR', 'SoftCE',
          'ExpectedRegret', 'NormalizedExpectedRegret', 'PredictionEntropy',
          'Top1Probability', 'Top1Top2Margin', 'OracleModeProbability',
          'minADEOracle6', 'minFDE6', 'MR6')
KEYS = ('scene_token', 'sample_token', 'instance_token')
INPUT_NAMES = ('local_nodes', 'interaction_edge', 'neighbor_mask', 'map_node',
               'map_edge', 'map_mask', 'original_logits')


def frozen_gate():
    """The gate precedes every read of the historical OuterTest cache."""
    assert os.environ.get('STAGE14A_PHASE') not in ('tiny', 'train')
    gate = read_json(ROOT / '04_checkpoints/stage14a_all_frozen.json')
    assert gate['Status'] == 'FROZEN_ALL_COMPLETE'
    rows = gate['Checkpoints']
    assert len(rows) == 6
    expected = {str(cp_path(fold, name).relative_to(PROJECT))
                for fold in (1, 2, 3) for name in VARIANTS}
    assert {r['Path'] for r in rows} == expected
    for row in rows:
        assert sha256(PROJECT / row['Path']) == row['SHA256'], row['Path']
    return gate


def joined_history(f):
    audit = read_json(OOF / 'complete.json')
    assert audit['Status'] == 'PASS'
    assert audit['models'] == ['R0', 'R2', 'A', 'B', 'C']
    assert audit['fields'] == list(FIELDS)
    for name, digest in audit['cache_files'].items():
        assert sha256(OOF / name) == digest, name
    old = pd.read_csv(OOF / 'stage11b_oof_predictions_actor_records.csv',
                      dtype={'future_mask_bits': str})
    current_index = pd.MultiIndex.from_frame(f[list(KEYS)])
    history_index = pd.MultiIndex.from_frame(old[list(KEYS)])
    assert current_index.is_unique and history_index.is_unique
    assert len(current_index) == len(history_index) == 260151
    join = history_index.get_indexer(current_index)
    assert np.all(join >= 0) and len(np.unique(join)) == len(join)
    for field in ('source_index', 'agent_type', 'agent_type_id',
                  'future_mask_bits', 'GT_displacement', 'motion_state'):
        a, b = f[field].to_numpy(), old.iloc[join][field].to_numpy()
        # CSV serialization of descriptive displacement can round the last bit;
        # model costs are read from unchanged binary arrays and checked below.
        if a.dtype.kind in 'fc':
            assert np.allclose(a, b, atol=1e-12, rtol=0., equal_nan=True), field
        else:
            assert pd.Series(a).equals(pd.Series(b)), field
    return old, join, audit


def actor_metrics(z, p, fd, ad):
    ii = torch.arange(len(fd), device=fd.device)
    best = fd.argmin(-1)
    oracle = fd.min(-1).values
    oracleade = ad.min(-1).values
    costs = fd - oracle[:, None]
    scale = costs.mean(-1).clamp(min=1)
    top = p.argmax(-1)
    order = p.argsort(dim=-1, descending=True, stable=True)
    rank = (order == best[:, None]).long().argmax(-1) + 1
    sortedp = p.sort(-1, descending=True).values
    return torch.stack((
        fd[ii, top].double(), ad[ii, top].double(),
        fd[ii, top].double() - oracle.double(), (top == best).double(),
        1. / rank.double(), objective(z, fd, 'A').double(),
        (p * costs).sum(-1).double(),
        (p * (costs / scale[:, None])).sum(-1).double(),
        -(p * p.clamp(min=1e-30).log()).sum(-1).double(),
        sortedp[:, 0].double(), (sortedp[:, 0] - sortedp[:, 1]).double(),
        p[ii, best].double(), oracleade.double(), oracle.double(),
        (oracle > 2).double()), -1)


@torch.no_grad()
def main():
    frozen = frozen_gate()
    seed()
    verify(history=True, data=True)
    dest = ROOT / '05_evaluation/cache'
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'complete.json').exists(), 'preserve the completed unified evaluation'
    f = frame().copy()
    old, join, oldaudit = joined_history(f)
    oldz = np.load(OOF / 'stage11b_oof_predictions_logits.npy', mmap_mode='r')
    oldp = np.load(OOF / 'stage11b_oof_predictions_probabilities.npy', mmap_mode='r')
    oldv = np.load(OOF / 'stage11b_oof_metrics.npy', mmap_mode='r')
    assert oldz.shape == oldp.shape == (len(f), 5, 6)
    assert oldv.shape == (len(f), 5, len(FIELDS))
    n = len(f)
    filled = np.zeros(n, bool)
    foldid = np.zeros(n, np.int8)
    logits = np.lib.format.open_memmap(dest / 'stage14a_oof_logits.npy', mode='w+',
                                      dtype='float32', shape=(n, 4, 6))
    prob = np.lib.format.open_memmap(dest / 'stage14a_oof_probabilities.npy', mode='w+',
                                    dtype='float32', shape=(n, 4, 6))
    values = np.lib.format.open_memmap(dest / 'stage14a_oof_metrics.npy', mode='w+',
                                      dtype='float64', shape=(n, 4, len(FIELDS)))
    candidate_path = S11A / '01_identity_audit/cache/candidates.npy'
    candidate_sha_before = sha256(candidate_path)
    candidates = np.load(candidate_path, mmap_mode='r')
    foldrows, identity_rows, poison_rows, runtime_rows = [], [], [], []
    bicyclecount = 0
    for fold in (1, 2, 3):
        store = Store(fold)
        ids = indices(fold, 'OuterTest')
        assert not filled[ids].any()
        assert not set(f.iloc[ids].scene_token) & set(split(fold)['InnerTrain'] + split(fold)['InnerDev'])
        assert np.all(old.iloc[join[ids]].Fold.to_numpy() == fold)
        models = {}
        for name in VARIANTS:
            saved = torch.load(cp_path(fold, name), map_location='cpu', weights_only=False)
            assert saved['Fold'] == fold and saved['Model'] == name
            assert saved['normalization_sha256'] == sha256(store.normpath)
            model = fresh(fold)
            model.load_state_dict(saved['state_dict'])
            models['NG-' + name] = model.eval().requires_grad_(False)
        before = {name: state_sha(model) for name, model in models.items()}
        elapsed = {name: 0. for name in models}
        preparation = 0.
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        for start in range(0, len(ids), 128):
            ix = ids[start:start + 128]
            begin = time.monotonic()
            args, fd, ad = store.batch(ix, part='OuterTest')
            torch.cuda.synchronize()
            preparation += time.monotonic() - begin
            assert len(args) == 7 and args[3:6] == (None, None, None)
            assert all(arg is None or arg.data_ptr() not in (fd.data_ptr(), ad.data_ptr()) for arg in args)
            shared = candidates[store.source[ix]]
            shared_sha = array_sha(shared)
            bike = torch.as_tensor(store.types[ix] == 2, device='cuda')
            bicyclecount += int(bike.sum())
            oldix = join[ix]
            r2z = torch.from_numpy(np.array(oldz[oldix, 1], copy=True)).cuda()
            r2p = torch.from_numpy(np.array(oldp[oldix, 1], copy=True)).cuda()
            zz, pp = {}, {}
            for name, model in models.items():
                torch.cuda.synchronize()
                begin = time.monotonic()
                out = model(*args)
                torch.cuda.synchronize()
                elapsed[name] += time.monotonic() - begin
                zz[name], pp[name] = out['mode_logits'].clone(), out['mode_prob'].clone()
                if start == 0:
                    # Changing the separate labels does not touch any of the seven inputs.
                    fd_poison, ad_poison = torch.full_like(fd, float('nan')), torch.full_like(ad, float('nan'))
                    repeated = model(*args)
                    assert torch.equal(repeated['mode_logits'], out['mode_logits'])
                    assert torch.equal(repeated['mode_prob'], out['mode_prob'])
                    assert not torch.isfinite(fd_poison).any() and not torch.isfinite(ad_poison).any()
                    poison_rows.append({'Fold': fold, 'Model': name, 'Actors': len(ix),
                                        'LabelPoison': 'FDE/ADE NaN, kept outside forward input',
                                        'LogitMaxDiff': 0., 'ProbabilityMaxDiff': 0., 'Status': 'PASS'})
                zz[name][bike], pp[name][bike] = r2z[bike], r2p[bike]
            for name, axis in (('G-A', 2), ('G-C', 4)):
                zz[name] = torch.from_numpy(np.array(oldz[oldix, axis], copy=True)).cuda()
                pp[name] = torch.from_numpy(np.array(oldp[oldix, axis], copy=True)).cuda()
                assert torch.equal(zz[name][bike], r2z[bike])
                assert torch.equal(pp[name][bike], r2p[bike])
            for j, name in enumerate(MODELS):
                z, p = zz[name], pp[name]
                assert torch.isfinite(z).all() and torch.isfinite(p).all()
                assert torch.allclose(p.sum(-1), torch.ones(len(ix), device='cuda'), atol=1e-6, rtol=0.)
                assert torch.equal(z[bike], r2z[bike]) and torch.equal(p[bike], r2p[bike])
                v = actor_metrics(z, p, fd, ad)
                logits[ix, j] = z.cpu().numpy()
                prob[ix, j] = p.cpu().numpy()
                values[ix, j] = v.cpu().numpy()
                if name.startswith('G-'):
                    axis = {'G-A': 2, 'G-C': 4}[name]
                    assert np.array_equal(values[ix, j], oldv[oldix, axis]), 'historical actor metrics must reproduce bitwise'
            for j in range(1, 4):
                assert np.array_equal(values[ix, j, 12:], values[ix, 0, 12:])
                assert np.array_equal(logits[ix[bike.cpu().numpy()], j], logits[ix[bike.cpu().numpy()], 0])
                assert np.array_equal(prob[ix[bike.cpu().numpy()], j], prob[ix[bike.cpu().numpy()], 0])
                assert np.array_equal(values[ix[bike.cpu().numpy()], j], values[ix[bike.cpu().numpy()], 0])
            assert array_sha(shared) == shared_sha
            filled[ix], foldid[ix] = True, fold
        assert before == {name: state_sha(model) for name, model in models.items()}
        assert all(not p.requires_grad and p.grad is None for model in models.values() for p in model.parameters())
        for group, mask in groups(f.iloc[ids]).items():
            if not mask.any():
                continue
            for j, name in enumerate(MODELS):
                foldrows.append({'Fold': fold, 'Group': group, 'Model': name,
                                 'Count': int(mask.sum()), **dict(zip(FIELDS, values[ids[mask], j].mean(0)))})
        for name in models:
            runtime_rows.append({'Fold': fold, 'Model': name, 'OOFActors': len(ids),
                                 'HeadForwardSeconds': elapsed[name],
                                 'HeadLatencyMSPerActor': 1000 * elapsed[name] / len(ids),
                                 'FeaturePreparationSeconds': preparation,
                                 'PeakAllocatedGPUMemoryBytes': torch.cuda.max_memory_allocated(),
                                 'TimingScope': 'FP32 cached-input head forward, synchronized CUDA; Bicycle routing and predictor excluded'})
        identity_rows.append({'Fold': fold, 'Scenes': len(set(f.iloc[ids].scene_token)),
                              'Actors': len(ids), 'IdentityJoin': 'PASS',
                              'GraphActorMetricsBitwiseReproduced': True,
                              'StatesUnchanged': before == {name: state_sha(model) for name, model in models.items()}})
        print('STAGE14A_OUTER_FOLD_PASS', fold, len(ids), time.monotonic() - started, flush=True)
        del models, store
        torch.cuda.empty_cache()
    assert filled.all() and np.all(foldid > 0) and len(set(f.scene_token)) == 630
    assert sha256(candidate_path) == candidate_sha_before and bicyclecount == 2980
    f['Fold'], f['Partition'], f['HistoricalOOFRow'] = foldid, 'OuterTest', join
    f.to_csv(dest / 'stage14a_oof_actor_records.csv', index=False)
    rows = []
    for group, mask in groups(f).items():
        assert mask.any(), group
        for j, name in enumerate(MODELS):
            rows.append({'Group': group, 'Model': name, 'Count': int(mask.sum()),
                         **dict(zip(FIELDS, values[mask, j].mean(0)))})
    dump('05_evaluation/stage14a_ablation_metrics.csv', rows)
    dump('05_evaluation/stage14a_fold_metrics.csv', foldrows)
    dump('05_evaluation/stage14a_fold_identity.csv', identity_rows)
    dump('05_evaluation/stage14a_label_input_audit.csv', poison_rows)
    dump('07_diagnostics/stage14a_evaluation_runtime.csv', runtime_rows)
    # Reproduction of historical published CSVs is checked at the displayed precision.
    historical_tables = [S11B / f'05_oof_evaluation/stage11b_oof_{part}_results.csv'
                         for part in ('main', 'type', 'motion')]
    historical_table = pd.concat([pd.read_csv(path) for path in historical_tables])
    lookup = {(r['Group'], r['Model']): r for r in rows}
    for row in historical_table.to_dict('records'):
        if row['Model'] not in ('A', 'C'):
            continue
        reproduced = lookup[row['Group'], 'G-' + row['Model']]
        assert reproduced['Count'] == row['Count']
        for field in FIELDS:
            assert format(reproduced[field], '.12g') == format(row[field], '.12g'), (row['Group'], field)
    fold_lookup = {(row['Fold'], row['Group'], row['Model']): row for row in foldrows}
    for row in pd.read_csv(S11B / '05_oof_evaluation/stage11b_fold_metrics.csv').to_dict('records'):
        if row['Model'] not in ('A', 'C'):
            continue
        reproduced = fold_lookup[row['Fold'], row['Group'], 'G-' + row['Model']]
        assert reproduced['Count'] == row['Count']
        for field in FIELDS:
            assert format(reproduced[field], '.12g') == format(row[field], '.12g'), (row['Fold'], row['Group'], field)
    for array in (logits, prob, values):
        array.flush()
    audit = {'Status': 'PASS', 'Folds': 3, 'OOFScenes': 630, 'OOFActors': n,
             'Fields': list(FIELDS), 'Models': list(MODELS), 'IdentityKeys': list(KEYS),
             'EachActorExactlyOnce': True, 'EachSceneExactlyOneFold': True,
             'AllSixNewCheckpointsFrozenBeforeEvaluation': True,
             'FrozenGateSHA256': sha256(ROOT / '04_checkpoints/stage14a_all_frozen.json'),
             'CandidateIdentity': 'PASS', 'CandidateCoordinateMaxDiff': 0.,
             'CandidatePath': str(candidate_path.relative_to(PROJECT)),
             'CandidateSHA256': candidate_sha_before, 'GeometryAllModelsBitwiseEqual': True,
             'minADEOracle6_minFDE6_MR6_AllModelsBitwiseEqual': True,
             'HistoricalGraphScoresAndActorMetricsBitwiseReproduced': True,
             'HistoricalGraphPublishedMetricReproduction': 'PASS',
             'HistoricalPublishedCSVPrecision': 'original %.12g; exact comparison at those 12 significant digits',
             'HistoricalOOFAuditSHA256': sha256(OOF / 'complete.json'),
             'BicyclePreserved': 'YES', 'BicycleActors': bicyclecount,
             'BicycleScoresProbabilitiesModesMetricsMaxDiff': 0.,
             'BicyclePolicy': 'preregistered corresponding-fold R2 route; not a graph improvement',
             'ForwardInputNames': list(INPUT_NAMES), 'FutureLabelsInForward': False,
             'FrozenModelStatesUnchanged': True, 'RankingOuterTestSeenInCorrespondingFoldTraining': False,
             'PredictorTrainingOverlap': 'separate provenance audit; ranking OOF is not end-to-end independent evaluation',
             'official_VAL_or_test_used': False,
             'Units': {'Top1FDE': 'm', 'Top1ADE': 'm', 'OracleGap': 'm', 'minFDE6': 'm',
                       'minADEOracle6': 'm', 'ExpectedRegret': 'm', 'MR6': 'fraction; minFDE6>2m',
                       'HitRate': 'fraction; Top1 matches lowest-index endpoint-FDE oracle'},
             'EvaluationMask': 'identical historical full 12-step future actor rows; denominator Count actors/windows',
             'cache_files': {path.name: sha256(path) for path in dest.glob('*') if path.suffix in ('.npy', '.csv')}}
    frozen_gate()
    verify(history=True)
    atomic_json(dest / 'complete.json', audit)
    atomic_json(ROOT / '05_evaluation/stage14a_identity_audit.json', audit)
    print('STAGE14A_UNIFIED_OOF_EVALUATION_PASS', flush=True)


if __name__ == '__main__':
    main()
