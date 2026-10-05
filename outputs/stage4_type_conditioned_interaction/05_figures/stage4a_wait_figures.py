"""Render real Stage4A figures only after all four evaluation jobs have exited."""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAGE3_ROOT = ROOT.parent / 'stage3_multitype_hivt'
sys.path.insert(0, str(STAGE3_ROOT / '00_manifest'))
sys.path.insert(0, str(STAGE3_ROOT / '04_evaluation'))
sys.path.insert(0, str(ROOT / '00_manifest'))
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3b_common import atomic_json, read_json, sha256

EVALUATION_STATE = ROOT / '00_manifest/stage4a_evaluation_pipeline_state.json'
STATE = ROOT / '00_manifest/stage4a_figures_pipeline_state.json'
SUMMARY = ROOT / '03_type_interaction/stage4a_training_summary.json'
STAGES = ('fresh-val', 'analyze', 'diagnostics', 'efficiency')


def now():
    return datetime.now(timezone.utc).isoformat()


def evaluation_or_training_active():
    process = subprocess.run(['pgrep', '-f', r'[/ ]stage4a_(evaluate|train)\.py( |$)'], capture_output=True, text=True)
    assert process.returncode in (0, 1)
    return process.returncode == 0


def ready():
    if not EVALUATION_STATE.exists() or not SUMMARY.exists():
        return False
    state = read_json(EVALUATION_STATE)
    if state['status'] != 'PASS':
        return False
    assert set(STAGES) <= set(state['completed_stages'])
    assert read_json(SUMMARY)['status'] == 'COMPLETE'
    if evaluation_or_training_active():
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poll-seconds', type=float, default=30.)
    args = parser.parse_args()
    assert 1 <= args.poll_seconds <= 60
    lock_path = ROOT / '02_preprocessed/stage4a_cache/stage4a_figures_pipeline.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'WAITING_FOR_ALL_FOUR_EVALUATION_STAGES', 'started_UTC': now(),
                 'completed_stages': [], 'stages': {}, 'watcher_source_SHA256': sha256(Path(__file__)),
                 'evaluation_process_exited_before_cases': False,
                 'no_training': True, 'manual_visual_QA_required_after_exports': True}
        atomic_json(STATE, state)
        print('STAGE4A_FIGURES_WAITING_FOR_COMPLETE_EVALUATION', now(), flush=True)
        while not ready():
            time.sleep(args.poll_seconds)
        evaluation = read_json(EVALUATION_STATE)
        state.update(status='RUNNING', evaluation_pipeline_SHA256=sha256(EVALUATION_STATE),
                     evaluation_completed_stages=evaluation['completed_stages'],
                     evaluation_process_exited_before_cases=True,
                     checkpoint_SHA256=evaluation['checkpoint_SHA256'])
        atomic_json(STATE, state)
        print('STAGE4A_FIGURES_ALL_FOUR_EVALUATION_STAGES_COMPLETE', now(), flush=True)
        for name, script in (('cases', 'stage4a_plot_cases.py'), ('quantitative', 'stage4a_plot_quantitative.py'),
                             ('export-QA', 'stage4a_check_exports.py')):
            assert not evaluation_or_training_active(), 'Evaluation or training overlaps figure job'
            path = ROOT / '05_figures' / script
            state['active_stage'] = name
            state['stages'][name] = {'started_UTC': now(), 'source_SHA256': sha256(path)}
            atomic_json(STATE, state)
            print('STAGE4A_FIGURES_BEGIN', name, now(), flush=True)
            result = subprocess.run([sys.executable, str(path)], cwd=ROOT.parents[1])
            state['stages'][name].update(completed_UTC=now(), exit_code=result.returncode)
            if result.returncode:
                state.update(status='FAIL', failed_stage=name, completed_UTC=now())
                atomic_json(STATE, state)
                raise SystemExit(result.returncode)
            state['completed_stages'].append(name)
            atomic_json(STATE, state)
            print('STAGE4A_FIGURES_END', name, now(), flush=True)
        state.update(status='PASS', completed_UTC=now(), active_stage=None)
        state['outputs_SHA256'] = {str(path.relative_to(ROOT)): sha256(path)
                                  for directory in ('03_type_interaction', '05_figures', '04_evaluation/cases')
                                  for path in sorted((ROOT / directory).glob('stage4a_*'))
                                  if path.is_file() and path.suffix in ('.png', '.pdf', '.svg', '.json', '.csv')}
        state['outputs_SHA256']['00_manifest/stage4a_export_QA.json'] = sha256(ROOT / '00_manifest/stage4a_export_QA.json')
        atomic_json(STATE, state)
        print('STAGE4A_FIGURES_PIPELINE=PASS', now(), flush=True)


if __name__ == '__main__':
    main()
