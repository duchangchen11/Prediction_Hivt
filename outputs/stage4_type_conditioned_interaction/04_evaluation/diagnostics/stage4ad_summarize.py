"""Exact population moments/quantiles of recorded formal logits and attention."""
import argparse
import json
from pathlib import Path
import numpy as np
from stage4ad_diagnostic import (ROOT, DIAG, EDGE_DTYPE, NODE_DTYPE, atomic_json,
    read_json, write_csv, sha256, CHECKPOINT_SHA)

TYPES = ('Vehicle', 'Pedestrian', 'Bicycle')
PAIRS = ('V<-V', 'V<-P', 'V<-B', 'P<-V', 'P<-P', 'P<-B', 'B<-V', 'B<-P', 'B<-B')


def distribution(values, prefix, q=(.1, .5, .9)):
    values = np.asarray(values).reshape(-1)
    assert len(values) and np.isfinite(values).all()
    quantiles = np.quantile(values, q, method='linear')
    return {prefix + 'Mean': float(np.mean(values, dtype=np.float64)),
        prefix + 'Std': float(np.std(values, dtype=np.float64, ddof=0)),
        **{prefix + name: float(v) for name, v in zip(('P10', 'Median', 'P90'), quantiles)}}


def logit_summary(records, masks, head=None):
    arrays = {}
    for name in ('base', 'bias', 'final'):
        arrays[name] = np.concatenate([r[name][mask].reshape(-1) if head is None
            else r[name][mask, head].reshape(-1) for r, mask in zip(records, masks)])
    base, bias, final = (arrays[k] for k in ('base', 'bias', 'final'))
    if not len(base):
        return None
    abs_base, abs_bias = np.abs(base), np.abs(bias)
    mean_base, mean_bias = np.mean(abs_base, dtype=np.float64), np.mean(abs_bias, dtype=np.float64)
    bq, rq = np.quantile(abs_base, [.5, .9]), np.quantile(abs_bias, [.5, .9])
    return {'EdgeCount': sum(int(mask.sum()) for mask in masks), 'LogitCount': len(base),
        'MeanBase': float(np.mean(base, dtype=np.float64)), 'StdBase': float(np.std(base, dtype=np.float64)),
        'MeanAbsBase': float(mean_base), 'MedianAbsBase': float(bq[0]), 'P90AbsBase': float(bq[1]),
        'MeanBias': float(np.mean(bias, dtype=np.float64)), 'StdBias': float(np.std(bias, dtype=np.float64)),
        'MeanAbsBias': float(mean_bias), 'MedianAbsBias': float(rq[0]), 'P90AbsBias': float(rq[1]),
        'MeanAbsFinal': float(np.mean(np.abs(final), dtype=np.float64)),
        'BiasBaseRatio': float(mean_bias / (mean_base + 1e-8))}


def attention_summary(records, masks, head=None):
    arrays = {}
    for name in ('entropy_base', 'entropy_final', 'l1_shift', 'switch'):
        arrays[name] = np.concatenate([r[name][mask].reshape(-1) if head is None
            else r[name][mask, head].reshape(-1) for r, mask in zip(records, masks)])
    if not len(arrays['entropy_base']):
        return {'NodeLayerCount': 0, 'NodeHeadCount': 0}
    hb, hf = arrays['entropy_base'], arrays['entropy_final']
    result = {'NodeLayerCount': sum(int(m.sum()) for m in masks), 'NodeHeadCount': len(hb),
        **distribution(hb, 'EntropyBase'), **distribution(hf, 'EntropyFinal'),
        **distribution(hf - hb, 'DeltaEntropy'), **distribution(arrays['l1_shift'], 'AttentionL1Shift'),
        'TopNeighborSwitchRate': float(np.mean(arrays['switch'], dtype=np.float64))}
    for field in ('EntropyBase', 'EntropyFinal', 'DeltaEntropy', 'AttentionL1Shift'):
        result[field] = result[field + 'Mean']
    return result


def raw_sources():
    return {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
        for p in sorted(DIAG.glob('stage4ad_raw_*')) if p.is_file()}


def main():
    schema = read_json(DIAG / 'stage4ad_capture_schema.json')
    assert schema['status'] == 'PASS' and schema['edge_counts'] == [3058206]*3
    edges = [np.memmap(DIAG / f'stage4ad_raw_L{i+1}_logits.bin', dtype=EDGE_DTYPE, mode='r') for i in range(3)]
    nodes = [np.memmap(DIAG / f'stage4ad_raw_L{i+1}_attention.bin', dtype=NODE_DTYPE, mode='r') for i in range(3)]
    assert [len(r) for r in edges] == schema['edge_counts']
    assert [len(r) for r in nodes] == schema['node_counts']
    # Both observed quantities were written before any parameter changes and
    # must retain the exact float32 addition from the real lambda=1 forward.
    for record in edges:
        for start in range(0, len(record), 100000):
            part = record[start:start+100000]
            assert np.array_equal(part['base'] + part['bias'], part['final'])
    subsets = ('All actual targets', 'Heterogeneous-20m', 'VP-context-20m',
        'Heterogeneous-20m full horizon', 'VP-context-20m full horizon')

    def subset_mask(record, subset):
        flags = record['flags']
        if subset == 'All actual targets':
            return np.ones(len(record), dtype=bool)
        bit = 4 if subset.startswith('Heterogeneous') else 8
        mask = (flags & bit) != 0
        if subset.endswith('full horizon'):
            mask &= (flags & 2) != 0
        return mask

    core, pairs, detail, contexts = [], [], [], []
    for subset in subsets:
        for kind, identifiers in (('Target', range(3)), ('Pair', range(9))):
            # Pair logit and entropy contexts are detailed on all actual targets.
            if kind == 'Pair' and subset != subsets[0]:
                continue
            for ident in identifiers:
                edge_masks = [subset_mask(r, subset) & ((r['pair']//3 == ident)
                    if kind == 'Target' else (r['pair'] == ident)) for r in edges]
                node_masks = [subset_mask(r, subset) & ((r['type'] == ident)
                    if kind == 'Target' else ((r['pair_context'] & (1 << ident)) != 0)) for r in nodes]
                label = TYPES[ident] if kind == 'Target' else PAIRS[ident]
                if not any(m.any() for m in edge_masks):
                    continue
                metadata = {'Subset': subset, 'TargetType': TYPES[ident if kind == 'Target' else ident//3],
                    'Pair': '' if kind == 'Target' else label, 'Group': label}
                for li in (None, 0, 1, 2):
                    selected = list(range(3)) if li is None else [li]
                    er, em = [edges[i] for i in selected], [edge_masks[i] for i in selected]
                    nr, nm = [nodes[i] for i in selected], [node_masks[i] for i in selected]
                    logits = logit_summary(er, em)
                    attention = attention_summary(nr, nm)
                    if logits is None:
                        continue
                    row = {**metadata, 'Layer': 'All' if li is None else li+1, 'Head': 'All',
                        **logits, **attention}
                    contexts.append(row)
                    if subset == subsets[0] and kind == 'Target':
                        core.append(row)
                    if subset == subsets[0] and kind == 'Pair' and li is None:
                        pairs.append(row)
                    if li is not None:
                        for head in range(8):
                            lh = logit_summary(er, em, head)
                            if lh is not None:
                                detail.append({**metadata, 'Layer': li+1, 'Head': head+1,
                                    **lh, **attention_summary(nr, nm, head)})
                print('STAGE4AD_SUMMARY', subset, label, flush=True)
    tables = ROOT / '06_tables'
    write_csv(tables / 'stage4ad_attention_scale_summary.csv', core)
    write_csv(tables / 'stage4ad_pair_scale_summary.csv', pairs)
    write_csv(tables / 'stage4ad_layer_head_statistics.csv', detail)
    write_csv(tables / 'stage4ad_frozen_context_statistics.csv', contexts)
    atomic_json(DIAG / 'stage4ad_statistics_audit.json', {'status': 'PASS',
        'checkpoint_sha256': CHECKPOINT_SHA, 'raw_sources': raw_sources(),
        'original_edge_window_count': schema['edge_counts'][0],
        'unique_target_window_with_neighbor_count': schema['node_counts'][0],
        'layers': 3, 'heads': 8, 'all_group_logit_count': sum(len(r)*8 for r in edges),
        'core_rows': len(core), 'pair_rows': len(pairs), 'head_rows': len(detail), 'context_rows': len(contexts),
        'count_definition': 'EdgeCount counts edge-layer observations; LogitCount adds head multiplicity. NodeLayerCount counts target-layer observations; NodeHeadCount adds head multiplicity.',
        'population_std_ddof': 0, 'quantiles': 'exact np.quantile, linear; no sampling',
        'target_logit_weighting': 'equal edge-layer-head',
        'target_attention_weighting': 'equal target-layer-head among nodes with >=1 actual incoming edge',
        'pair_context_entropy': 'whole incoming neighborhood; pair contexts overlap',
        'frozen_contexts': 'exact registered ledger flags, no recomputed distances or membership',
        'all_targets_includes_unsupervised_context_nodes': True,
        'context_targets': 'only ledger-known targets; combined full/partial and full-only reported separately',
        'overlapping_windows_independent': False, 'bootstrap_or_significance_test': False})


if __name__ == '__main__':
    main()
