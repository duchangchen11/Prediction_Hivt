"""At most 3000 NLL updates from the frozen best, with exact optimizer and cursor restore."""
import csv
import math
import os
from pathlib import Path
import shutil
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3a_plus_common import (CONFIG, CACHE, BEST_MANIFEST, CLASSES, SceneDataset, SceneSampler,
                                config, atomic_json, read_json, sha256, git, write_csv, verify_freeze)
from stage3_common import model_new, model_input, loss_diagnostics, evaluate
import torch
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURVE = ROOT / '03_no_type_baseline/stage3a_plus_nll_extension_curve.csv'
LAST = ROOT / '07_checkpoints/stage3a_plus_last_checkpoint.pt'
NEW_BEST = ROOT / '07_checkpoints/stage3a_plus_best_overall_minfde.pt'
ORIGINAL = ROOT / '07_checkpoints/stage3_best_overall_minfde.pt'
LATEST_ACTORS = CACHE / 'stage3a_plus_latest_actor_errors.csv'
BEST_ACTORS = CACHE / 'stage3a_plus_best_actor_errors.csv'
SOURCE = ROOT / '00_manifest/stage3a_plus_preregistration.json'


def plot_curve(rows, original):
    fig, ax = plt.subplots(figsize=(8.5, 4.3), layout='constrained')
    x = [original['global_step']] + [r['global_step'] for r in rows]
    for name, color in zip(('overall',) + CLASSES, ('#246C9E', '#737B84', '#4A8A77', '#C87B24')):
        start = original['validation_FDE'] if name == 'overall' else original['per_class_full_horizon_metrics'][name]['minFDE6']
        ax.plot(x, [start] + [r['VAL_' + name + '_FDE'] for r in rows], 'o-', ms=4, color=color, label=name)
    ax.axhline(original['validation_FDE'], ls='--', lw=.8, color='#246C9E', alpha=.5)
    ax.set(xlabel='Executed optimizer step (Stage3A + extension)', ylabel='Full-horizon official VAL minFDE@6 (m)')
    ax.grid(alpha=.15); ax.legend(ncol=2, frameon=False)
    stem = CURVE.with_suffix('')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(str(stem) + '.' + ext, dpi=300)
    svg = Path(str(stem) + '.svg'); svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines()) + '\n')
    plt.close(fig)


def save_checkpoint(path, model, optimizer, metadata, state, iterator):
    temp = path.with_suffix('.pt.tmp')
    torch.save({'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                'metadata': metadata, 'extension_state': state, 'iterator': iterator, 'config': config(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}, temp)
    os.replace(temp, path)


def main():
    c = config(); frozen = verify_freeze(); prereg = read_json(SOURCE)
    for name, digest in prereg['source_sha256'].items():
        assert sha256(ROOT / name) == digest, 'Extension source changed after registration'
    assert c['batch_size'] == 16 and c['nll']['lr'] == .0001
    source = torch.load(ORIGINAL, map_location='cpu', weights_only=False)
    original_meta = source['metadata']; original_sha = sha256(ORIGINAL)
    assert original_sha == prereg['original_best_sha256']
    assert source['config'] == c and original_meta['config_sha256'] == sha256(CONFIG)
    assert original_meta['phase'] == 'original_nll' and original_meta['selection_metric'] == 'overall minFDE6'
    if BEST_MANIFEST.exists() and read_json(BEST_MANIFEST)['status'] == 'COMPLETE':
        print('EXTENSION_ALREADY_COMPLETE', flush=True); return
    state = {'extension_step': 0, 'best_FDE': original_meta['validation_FDE'], 'best_step': original_meta['global_step'],
             'bad_validations': 0, 'best_refreshed': False, 'completed': False}
    iterator = dict(source['iterator']); saved = source; rows = []
    if LAST.exists():
        saved = torch.load(LAST, map_location='cpu', weights_only=False)
        state = saved['extension_state']; iterator = saved['iterator']
        if CURVE.exists():
            with CURVE.open() as f:
                rows = [{k: float(v) if v else None for k, v in r.items()} for r in csv.DictReader(f)
                        if int(r['extension_step']) <= state['extension_step']]
    model = model_new(); model.load_state_dict(saved['state_dict'])
    optimizer = model.optimizer(c['nll']['lr'], c['weight_decay'])
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    initial_groups = [{k: v for k, v in group.items() if k != 'params'} for group in source['optimizer_state_dict']['param_groups']]
    assert [{k: v for k, v in group.items() if k != 'params'} for group in optimizer.param_groups] == initial_groups
    torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    train = SceneDataset('train'); val = SceneDataset('val'); sampler = SceneSampler(train, c['seed'])
    steps_per_epoch = math.ceil(len(train) / c['batch_size'])
    print('RESTORED_BEST', original_meta['global_step'], original_meta['validation_FDE'], 'cursor', iterator,
          'extension_step', state['extension_step'], 'LR', optimizer.param_groups[0]['lr'], flush=True)
    original_record = {'relative_path': str(ORIGINAL.relative_to(ROOT)), 'sha256': original_sha,
                       'global_step': original_meta['global_step'], 'overall_minFDE6': original_meta['validation_FDE']}
    manifest = read_json(BEST_MANIFEST) if BEST_MANIFEST.exists() else {'status': 'RUNNING', 'original_best': original_record, 'final_best': original_record,
                'maximum_extra_steps': 3000, 'validation_interval': 500, 'patience': 5,
                'selection': 'full-horizon official VAL overall minFDE6; strict improvement',
                'optimizer_hyperparameters_unchanged': True, 'optimizer_and_rng_restored': True,
                'initial_iterator': source['iterator'], 'training_code_git_commit': git('rev-parse', 'HEAD'),
                'type_embedding': False, 'test_used': False}
    atomic_json(BEST_MANIFEST, manifest)
    started = time.monotonic(); resumed_loader = True
    while state['extension_step'] < 3000 and not state['completed']:
        sampler.epoch = iterator['epoch'] + 100000
        order = list(sampler)
        batches = [order[i:i + c['batch_size']] for i in range(0, len(order), c['batch_size'])]
        start_batch = iterator['next_batch']
        loader = DataLoader(train, batch_sampler=batches[start_batch:], num_workers=0)
        # Recreating a mid-epoch loader must not consume the saved dropout RNG stream.
        rng = torch.get_rng_state(); loader_iterator = iter(loader)
        if resumed_loader: torch.set_rng_state(rng)
        resumed_loader = False
        for batch_index, batch in enumerate(loader_iterator, start=start_batch):
            model.train(); data = batch.cuda(); optimizer.zero_grad(set_to_none=True)
            output = model(model_input(data)); values = loss_diagnostics(model, output, data, 'original_nll')
            assert all(torch.isfinite(v) for v in values.values()), 'Nonfinite loss'
            values['loss'].backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), 'Nonfinite gradient'
            optimizer.step(); state['extension_step'] += 1; iterator['next_batch'] = batch_index + 1
            step = state['extension_step']; global_step = original_meta['global_step'] + step
            if step % 100 == 0: print('EXTENSION_STEP', step, 'global', global_step, flush=True)
            if step % 500 == 0:
                measured = evaluate(val, model, actor_path=LATEST_ACTORS)
                assert len(measured['scenes']) == 150 and measured['windows'] == 3603
                full = measured['metrics']['full_horizon']; fde = full['overall']['minFDE6']
                improved = fde < state['best_FDE']
                state['bad_validations'] = 0 if improved else state['bad_validations'] + 1
                if improved:
                    state.update(best_FDE=fde, best_step=global_step, best_refreshed=True)
                state['completed'] = state['bad_validations'] >= 5 or step >= 3000
                row = {'extension_step': step, 'global_step': global_step, 'learning_rate': optimizer.param_groups[0]['lr'],
                       'best_overall_FDE': state['best_FDE'], 'best_step': state['best_step'],
                       'improved': int(improved), 'consecutive_nonimprovements': state['bad_validations']}
                for name in ('overall',) + CLASSES:
                    assert all(v is not None and math.isfinite(v) for v in full[name].values())
                    for metric, short in (('minADE6', 'ADE'), ('minFDE6', 'FDE'), ('MR6', 'MR')):
                        row['VAL_' + name + '_' + short] = full[name][metric]
                metadata = {**original_meta, 'epoch': iterator['epoch'] + iterator['next_batch'] / steps_per_epoch,
                            'global_step': global_step, 'phase_step': original_meta['phase_step'] + step,
                            'extension_step': step, 'validation_ADE': full['overall']['minADE6'], 'validation_FDE': fde,
                            'validation_MR': full['overall']['MR6'], 'git_commit_SHA': git('rev-parse', 'HEAD'),
                            'extension_original_checkpoint_sha256': original_sha, 'optimizer_state_preserved': True,
                            'per_class_full_horizon_metrics': {name: full[name] for name in CLASSES}}
                measured.update(checkpoint_metadata=metadata, extension_step=step)
                atomic_json(ROOT / f'04_evaluation/stage3a_plus_val_step_{step:05d}.json', measured)
                if improved:
                    save_checkpoint(NEW_BEST, model, optimizer, metadata, dict(state), dict(iterator))
                    shutil.copyfile(LATEST_ACTORS, BEST_ACTORS)
                    atomic_json(ROOT / '04_evaluation/stage3a_plus_best_val_metrics.json', measured)
                    manifest['final_best'] = {'relative_path': str(NEW_BEST.relative_to(ROOT)), 'sha256': sha256(NEW_BEST),
                                              'global_step': global_step, 'overall_minFDE6': fde}
                save_checkpoint(LAST, model, optimizer, metadata, dict(state), dict(iterator))
                manifest.update(executed_extra_steps=step, best_refreshed=state['best_refreshed'],
                                consecutive_nonimprovements=state['bad_validations'],
                                last_checkpoint_sha256=sha256(LAST), final_iterator=dict(iterator))
                atomic_json(BEST_MANIFEST, manifest)
                rows.append(row); write_csv(CURVE, rows); plot_curve(rows, original_meta)
                print('EXTENSION_VALIDATION', row, flush=True)
                if state['completed']: break
        if not state['completed']:
            iterator['epoch'] += 1; iterator['next_batch'] = 0
    manifest.update(status='COMPLETE', converged=state['bad_validations'] >= 5,
                    convergence_definition='five consecutive full VAL evaluations without strict overall minFDE6 improvement',
                    stop_reason='patience_5' if state['bad_validations'] >= 5 else 'extra_steps_3000_limit',
                    elapsed_seconds_this_invocation=time.monotonic() - started,
                    final_predictions_relative_path=str((BEST_ACTORS if state['best_refreshed'] else ROOT / '04_evaluation/stage3_no_type_actor_errors.csv').relative_to(ROOT)),
                    NaN=0, Inf=0)
    manifest['final_predictions_sha256'] = sha256(ROOT / manifest['final_predictions_relative_path'])
    assert sha256(ORIGINAL) == original_sha
    verify_freeze(); atomic_json(BEST_MANIFEST, manifest)
    print('NLL_EXTENSION_COMPLETE', manifest, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(4)
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans', 'font.size': 10})
    main()
