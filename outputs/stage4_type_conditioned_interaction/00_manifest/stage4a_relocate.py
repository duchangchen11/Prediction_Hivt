"""One-time byte-preserving migration requested after Stage4A training.

Frozen Stage3 inputs stay in place. Registered source snapshots preserve the
exact code used for training; operational path adapters are applied afterwards.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
OLD = PROJECT / 'outputs/stage3_multitype_hivt'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    audit_path = ROOT / '00_manifest/stage4a_relocation_audit.json'
    assert not audit_path.exists(), 'Migration already recorded'
    training = json.loads((OLD / '03_type_interaction/stage4a_training_summary.json').read_text())
    evaluation = json.loads((OLD / '00_manifest/stage4a_evaluation_pipeline_state.json').read_text())
    assert training['status'] == 'COMPLETE' and evaluation['status'] == 'PASS'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            if args and b'python' in args[0]:
                assert not any(Path(arg.decode()).name in ('stage4a_train.py', 'stage4a_evaluate.py',
                    'stage4a_wait_evaluate.py', 'stage4a_wait_figures.py') for arg in args[1:] if arg), 'Live Stage4A process'
        except (FileNotFoundError, ProcessLookupError):
            pass
    sources = json.loads((OLD / '00_manifest/stage4a_training_sources.json').read_text())
    snapshots = {}
    for name, digest in sources['training_source_sha256'].items():
        source = OLD / name
        assert sha256(source) == digest
        target = ROOT / '00_manifest/stage4a_training_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert sha256(target) == digest
        snapshots[name] = str(target.relative_to(ROOT))
    shutil.copyfile(OLD / '00_manifest/stage4a_requirements.txt',
                    ROOT / '00_manifest/stage4a_requirements_original.txt')
    moved = []
    for source in sorted(OLD.rglob('stage4a_*')):
        if not source.is_file():
            continue
        relative = source.relative_to(OLD)
        target_relative = Path(str(relative).replace('stage3_cache/', 'stage4a_cache/'))
        target = ROOT / target_relative
        assert not target.exists(), target
        before = sha256(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        assert sha256(target) == before
        moved.append({'old_project_path': str((OLD / relative).relative_to(PROJECT)),
            'new_project_path': str(target.relative_to(PROJECT)), 'bytes': target.stat().st_size,
            'SHA256_before_and_after_move': before})
    for name in ('00_manifest', '01_data_audit', '02_preprocessed', '03_type_interaction',
                 '04_evaluation/cases', '05_figures', '06_tables', '07_checkpoints', '08_logs', '09_reports'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    assert not [p for p in OLD.rglob('stage4a_*') if p.is_file()]
    audit = {'status': 'BYTE_PRESERVING_MOVE_PASS_PATH_ADAPTATION_PENDING',
        'user_instruction': 'Stage4 must have its own directory separate from Stage3',
        'old_stage4_root': str(OLD), 'new_stage4_root': str(ROOT),
        'frozen_Stage3_input_root': str(OLD), 'training_completed_before_migration': True,
        'final_evaluation_completed_before_migration': True, 'retraining': False,
        'best_checkpoint_SHA256': training['checkpoint_sha256'], 'registered_source_snapshots': snapshots,
        'files': moved, 'original_requirements_preserved': '00_manifest/stage4a_requirements_original.txt'}
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print('STAGE4A_BYTE_PRESERVING_RELOCATION_PASS', len(moved), 'files')


if __name__ == '__main__':
    main()
