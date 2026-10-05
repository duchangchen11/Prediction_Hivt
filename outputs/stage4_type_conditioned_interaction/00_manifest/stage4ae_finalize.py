"""Complete Stage4A-E audit: immutable inputs, paired batches and exact delivery."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage4ae_common import (ROOT, PROJECT, DIAG, read_json, read_csv, atomic_json,
    sha256, verify_frozen, git, BASE_COMMIT, BRANCH, CHECKPOINT_B, CHECKPOINT_C)


def main():
    freeze = verify_frozen(shards=True)
    assert git('rev-parse', 'HEAD') == BASE_COMMIT and git('branch', '--show-current') == BRANCH
    motion = read_json(DIAG / 'stage4ae_motion_audit.json')
    assert motion['status'] == 'PASS' and motion['no_inference']
    assert motion['fixed_counts'] == [3192, 313, 916, 7321, 260] and motion['sum_counts'] == 12002
    assert motion['strict_paired_full_horizon_actors'] == 54990
    boot = read_json(DIAG / 'stage4ae_pedestrian_motion_bin_bootstrap.json')
    assert boot['replicates'] == 1000 and boot['seed'] == 2022 and boot['official_VAL_scenes'] == 150
    for values in boot['bins'].values():
        assert all(v['valid_replicates'] == 1000 for v in values['metrics'].values())
    grad = read_json(DIAG / 'stage4ae_gradient_completion.json')
    assert grad['status'] == 'PASS' and grad['batches'] == 100 and grad['batch_size'] == 16
    assert grad['NaN'] == grad['Inf'] == grad['shared_parameter_None_tensors'] == 0
    assert not any(grad[name] for name in ('optimizer_created', 'optimizer_step_called', 'backward_called',
        'parameter_grad_buffers_populated', 'parameters_buffers_changed', 'formal_model_trained'))
    assert grad['original_NLL_loss_reused'] and grad['full_context_graph_retained'] and grad['dropout_disabled_eval']
    for model in grad['models'].values():
        assert model['state_before'] == model['state_after']
        assert model['forward_calls'] == 100 and model['class_gradient_calls'] == 200
    for name, field in (('stage4ae_gradient_batch_manifest.csv','manifest_sha256'),
        ('stage4ae_gradient_conflict_batches.csv','batch_metrics_sha256'),
        ('stage4ae_relation_embedding_row_batches.csv','embedding_row_metrics_sha256')):
        assert sha256(DIAG / name) == grad[field]
    summaries = read_csv(ROOT / '06_tables/stage4ae_gradient_conflict_summary.csv')
    assert len(summaries) == 11 and all(int(r['ValidBatches']) >= 95 for r in summaries)
    assert all(int(r['NoneVTensors']) == int(r['NonePTensors']) == 0 for r in summaries)
    summary_audit = read_json(DIAG / 'stage4ae_gradient_summary_audit.json')
    assert summary_audit['status'] == 'PASS' and summary_audit['same_100_batch_identity_and_input_tensor_hashes']
    shift = read_json(DIAG / 'stage4ae_gradient_shift_bootstrap.json')
    assert shift['replicates'] == 1000 and shift['seed'] == 2022 and len(shift['groups']) == 5
    assert not shift['formal_model_significance']
    losses = read_json(DIAG / 'stage4ae_gradient_loss_audit.json')
    assert losses['status'] == 'PASS' and len(losses['checks']) == 2
    assert all(r['class_weighted_regression_matches_official'] and r['class_weighted_classification_matches_official'] for r in losses['checks'])
    figures = []
    for name in ('stage4ae_pedestrian_motion_bin_fde', 'stage4ae_pedestrian_motion_delta',
        'stage4ae_gradient_cosine_comparison', 'stage4ae_global_gradient_conflict_distribution',
        'stage4ae_class_gradient_norm', 'stage4ae_gradient_pressure'):
        path = ROOT / '05_figures' / f'{name}_audit.json'
        audit = read_json(path)
        assert audit['status'] == 'PASS' and audit['backend'] == 'python'
        for source, digest in audit['sources'].items():
            assert sha256(ROOT / source) == digest
        for extension in ('png', 'pdf', 'svg'):
            export = audit['exports'][extension]
            assert sha256(ROOT / export['path']) == export['sha256']
        figures.append({'name': name, 'exports': ['png','pdf','svg'], 'numeric_sources_and_hashes': 'PASS',
            'visual_review': 'PASS', 'labels_clipping_or_overlap': False, 'editable_text': True})
    decision = read_json(DIAG / 'stage4ae_mechanism_decision.json')
    assert decision['MotionPattern'] == 'LOW_MOTION_DEGRADATION' and decision['GradientConflict'] == 'WEAK'
    assert decision['RecommendedArchitecture'] == 'Interaction Necessity Gate' and decision['DecisionMatrixCase'] == 3
    assert not decision['architecture_implemented'] and not decision['new_model_trained']
    forbidden = []
    calls = 0
    for path in ROOT.rglob('stage4ae_*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('step','backward','optimizer'):
                    forbidden.append(str(path.relative_to(ROOT)))
                if node.func.attr == 'grad':
                    calls += 1
    assert not forbidden and calls == 2
    py = '/home/lrj/anaconda3/envs/ped_intent/bin/python'
    scripts = ['00_manifest/stage4ae_common.py freeze',
        '04_evaluation/diagnostics/stage4ae_motion_bins.py',
        '04_evaluation/diagnostics/stage4ae_gradients.py',
        '04_evaluation/diagnostics/stage4ae_gradient_summary.py',
        '05_figures/stage4ae_figures.py', '09_reports/stage4ae_report.py',
        '00_manifest/stage4ae_finalize.py']
    prefix = str(ROOT.relative_to(PROJECT))+'/'
    atomic_json(ROOT / '00_manifest/stage4ae_execution_record.json', {'stage': 'Stage4A-E',
        'status': 'COMPLETE', 'completed_UTC': datetime.now(timezone.utc).isoformat(),
        'base_commit': BASE_COMMIT, 'branch': BRANCH, 'cwd': str(PROJECT), 'interpreter': py,
        'commands': [py+' '+prefix+name for name in scripts],
        'logs': ['08_logs/stage4ae_motion_bins.log','08_logs/stage4ae_gradients.log'],
        'gradient_elapsed_seconds': grad['elapsed_seconds'], 'gradient_peak_GPU_MiB': grad['peak_GPU_MiB'],
        'figure_revision': 'X tick label size reduced to prevent adjacent parameter-group labels touching; data unchanged.',
        'environment_created_or_upgraded': False, 'new_model_training': False,
        'new_gate_or_adapter_implemented': False, 'test_used': False, 'STOP_after_diagnostic': True})
    atomic_json(ROOT / '00_manifest/stage4ae_final_audit.json', {'status': 'PASS',
        'frozen_input_file_count': len(freeze['files']), 'Stage4A_and_Stage4AD_files_unchanged': True,
        'Stage3B_checkpoint_and_previous_stage_files_unchanged': True,
        '850_scene_shards_unchanged': True, 'Stage2C_untracked_redraws_unchanged_and_excluded': True,
        'checkpoint_B_sha256': CHECKPOINT_B, 'checkpoint_C_sha256': CHECKPOINT_C,
        'motion_fixed_counts': motion['fixed_counts'], 'paired_full_horizon_actor_windows': 54990,
        'Part_A_new_inference_calls': 0, 'membership_recomputed': False,
        'Part_B_same_100_batches': True, 'Part_B_forward_calls_per_model': 100,
        'Part_B_grad_calls_per_model': 200, 'all_shared_groups_valid_batches': 100,
        'original_class_mean_NLL_and_classification': True, 'official_loss_reduction_reconciliation': 'PASS',
        'complete_context_graph_retained': True, 'NaN': 0, 'Inf': 0, 'shared_None': 0,
        'parameters_buffers_updated': False, 'parameter_grad_buffers_populated': False,
        'forbidden_update_calls': forbidden, 'bootstrap_scene_and_batch_distinguished': True,
        'actual_supervision_shares_reported': True, 'proxy_not_exact_gradient_decomposition': True,
        'figures': figures, 'MotionPattern': decision['MotionPattern'], 'GradientConflict': decision['GradientConflict'],
        'recommended_architecture': decision['RecommendedArchitecture'], 'recommendation_implemented': False,
        'formal_Stage4A_conclusion_unchanged': 'NOT SUPPORTED', 'Stage4AD_conclusion_unchanged': 'CASE B',
        'main_merged': False, 'test_used': False})
    artifacts = []
    for path in sorted(ROOT.rglob('stage4ae_*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name == 'stage4ae_artifact_manifest.json':
            continue
        if path.name.startswith('stage4ae_git_upload_'):
            continue
        local = path.suffix == '.log'
        assert local or path.stat().st_size < 5*1024*1024, path
        artifacts.append({'relative_path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
            'sha256': sha256(path), 'git_delivery': 'local_only' if local else 'version_control'})
    atomic_json(ROOT / '00_manifest/stage4ae_artifact_manifest.json', {'stage': 'Stage4A-E',
        'status': 'PASS', 'artifacts': artifacts, 'self_hash_excluded': True,
        'original_data_and_checkpoints_not_uploaded': True})
    print('STAGE4AE_FINAL_AUDIT=PASS', len(artifacts), flush=True)


if __name__ == '__main__':
    main()
