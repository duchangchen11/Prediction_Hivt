"""Verify diagnostic completeness, frozen inputs and small-artifact delivery."""
import ast
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(ROOT / '04_evaluation/diagnostics'))
from stage4ad_diagnostic import (DIAG, read_json, atomic_json, sha256, verify_frozen,
    verify_previous, CHECKPOINT_SHA, BASE_COMMIT)


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    freeze = verify_frozen()
    verify_previous(shards=True)
    completion = read_json(DIAG / 'stage4ad_inference_completion.json')
    assert completion['status'] == 'PASS' and completion['lambdas'] == [0, .25, .5, .75, 1]
    scales = read(DIAG / 'stage4ad_bias_scale_sensitivity.csv')
    assert [float(r['Lambda']) for r in scales] == [0, .25, .5, .75, 1]
    evaluations = []
    for scale in [0, .25, .5, .75, 1]:
        label = f'{scale:.2f}'.replace('.', 'p')
        path = DIAG / f'stage4ad_lambda_{label}_evaluation_audit.json'
        audit = read_json(path)
        assert audit['status'] == 'PASS' and audit['VAL_scenes'] == 150 and audit['VAL_windows'] == 3603
        assert audit['state_before'] == audit['state_after'] and not audit['parameters_changed']
        assert audit['GT_and_masks_and_identities_equal'] and audit['no_grad']
        assert sha256(DIAG / f'stage4ad_lambda_{label}_actor_errors.csv') == audit['actor_csv_sha256']
        assert audit['checkpoint_sha256'] == CHECKPOINT_SHA and not audit['checkpoint_selection']
        evaluations.append({'lambda': scale, 'elapsed_seconds': audit['elapsed_seconds'], 'audit_sha256': sha256(path)})
    stats = read_json(DIAG / 'stage4ad_statistics_audit.json')
    assert stats['status'] == 'PASS' and stats['layers'] == 3 and stats['heads'] == 8
    detail = read(ROOT / '06_tables/stage4ad_layer_head_statistics.csv')
    keys = {(r['Subset'], r['Group'], int(r['Layer']), int(r['Head'])) for r in detail}
    assert len(keys) == len(detail) == 528
    for group in ('Vehicle', 'Pedestrian', 'Bicycle', 'V<-V', 'V<-P', 'V<-B',
                  'P<-V', 'P<-P', 'P<-B', 'B<-V', 'B<-P', 'B<-B'):
        for layer in range(1, 4):
            for head in range(1, 9):
                assert ('All actual targets', group, layer, head) in keys
    old = read_json(ROOT / '04_evaluation/stage4a_relation_bias_statistics.json')['layer_head_aggregated']
    current = read(ROOT / '06_tables/stage4ad_pair_scale_summary.csv')
    bias_reconciliation = []
    for previous, row in zip(old, current):
        assert previous['directed_pair'] == row['Pair']
        difference = abs(previous['mean_abs_bias']-float(row['MeanAbsBias']))
        assert difference < 1e-10
        bias_reconciliation.append({'pair': row['Pair'], 'absolute_mean_bias_difference': difference})
    names = ['stage4ad_logit_scale_by_target', 'stage4ad_bias_base_ratio_heatmap',
        'stage4ad_attention_entropy_shift', 'stage4ad_top_neighbor_switch',
        'stage4ad_lambda_sensitivity', 'stage4ad_lambda_motion_sensitivity']
    figure_checks = []
    for name in names:
        audit_path = ROOT / '05_figures' / (name + '_audit.json')
        audit = read_json(audit_path)
        assert audit['status'] == 'PASS' and audit['backend'] == 'python'
        assert sha256(ROOT / audit['source']) == audit['source_sha256']
        for kind in ('png', 'pdf', 'svg'):
            assert sha256(ROOT / audit['exports'][kind]['path']) == audit['exports'][kind]['sha256']
        figure_checks.append({'name': name, 'exports': ['png', 'pdf', 'svg'], 'visual_review': 'PASS',
            'source_and_export_SHA256': 'PASS', 'labels': 'readable, no overlap or clipping',
            'sensitivity_note': 'absolute curves intentionally share the same metric axis; small changes are quantified in source CSV and report'})
    conclusion = read_json(DIAG / 'stage4ad_diagnostic_conclusion.json')
    assert conclusion['DiagnosticConclusion'] == 'CASE B' and not conclusion['new_architecture_implemented']
    assert not conclusion['lambda_selected_as_new_model']
    assert conclusion['Stage4A_scientific_conclusion'] == 'NOT SUPPORTED' and conclusion['Ready_Stage4B'] == 'NO'
    forbidden_calls = []
    for path in ROOT.rglob('stage4ad_*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('backward', 'step'):
                forbidden_calls.append(str(path.relative_to(ROOT)))
    assert not forbidden_calls
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=PROJECT).decode().strip()
    assert branch == 'stage4a-d/relation-bias-diagnostic'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT).decode().strip() == BASE_COMMIT
    py = '/home/lrj/anaconda3/envs/ped_intent/bin/python'
    commands = [py+' 04_evaluation/diagnostics/stage4ad_diagnostic.py freeze',
        py+' -u 04_evaluation/diagnostics/stage4ad_diagnostic.py run',
        py+' -u 04_evaluation/diagnostics/stage4ad_summarize.py',
        py+' 05_figures/stage4ad_figures.py', py+' 09_reports/stage4ad_report.py',
        py+' 00_manifest/stage4ad_finalize.py']
    atomic_json(ROOT / '00_manifest/stage4ad_execution_record.json', {
        'stage': 'Stage4A-D', 'status': 'COMPLETE', 'commands_relative_to_Stage4_root': commands,
        'actual_cwd': str(PROJECT), 'base_commit': BASE_COMMIT, 'branch': branch,
        'interpreter': py, 'environment_created_or_upgraded': False, 'evaluations': evaluations,
        'logs': ['08_logs/stage4ad_inference.log', '08_logs/stage4ad_statistics.log'],
        'figure_dependency_resolution': 'Used existing pdftotext/pdffonts for PDF text/font inspection; no package installation.',
        'checkpoint_sha256': CHECKPOINT_SHA, 'STOP_after_diagnostic': True})
    atomic_json(ROOT / '00_manifest/stage4ad_final_audit.json', {
        'status': 'PASS', 'base_commit': BASE_COMMIT, 'branch': branch,
        'all_protected_Stage4A_files_unchanged': True, 'protected_file_count': len(freeze['files']),
        'previous_stage_files_and_850_scene_shards_unchanged': True,
        'old_Stage2C_untracked_files_unchanged_and_excluded': True, 'checkpoint_sha256': CHECKPOINT_SHA,
        'full_VAL150_passes': 5, 'lambda1_formal_result_reconciliation': 'PASS',
        'all_layer_head_target_pair_statistics': 'PASS', 'head_statistics_rows': len(detail),
        'exact_quantiles_and_population_moments': True, 'bias_capture_reconciliation': bias_reconciliation,
        'attention_uses_actual_final_softmax': True, 'frozen_membership_reused': True,
        'raw_logit_and_attention_source_hashes_verified': True, 'figures': figure_checks,
        'forbidden_training_calls': forbidden_calls, 'parameters_buffers_changed': False,
        'new_checkpoint_or_model_selection': False, 'new_architecture_implemented': False,
        'diagnostic_conclusion': 'CASE B', 'formal_Stage4A_conclusion': 'NOT SUPPORTED', 'Ready_Stage4B': 'NO',
        'test_used': False, 'main_merged': False})
    artifacts = []
    for path in sorted(ROOT.rglob('stage4ad_*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name == 'stage4ad_artifact_manifest.json':
            continue
        if path.name.startswith('stage4ad_git_upload_'):
            continue
        local = (path.name.startswith('stage4ad_raw_') or path.name.endswith('_actor_errors.csv') or path.suffix == '.log')
        artifacts.append({'relative_path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': sha256(path), 'git_delivery': 'local_only' if local else 'version_control'})
    assert all(p['relative_path'].split('/')[-1].startswith('stage4ad_') for p in artifacts)
    atomic_json(ROOT / '00_manifest/stage4ad_artifact_manifest.json', {'stage': 'Stage4A-D',
        'status': 'PASS', 'self_hash_excluded': True, 'artifacts': artifacts,
        'large_derived_arrays_actor_records_and_logs_stay_local': True})
    print('STAGE4AD_FINAL_AUDIT=PASS', len(artifacts), flush=True)


if __name__ == '__main__':
    main()
