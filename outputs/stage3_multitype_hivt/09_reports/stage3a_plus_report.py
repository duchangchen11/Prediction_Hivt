"""Verify bounded continuation, motion subsets, graph counts, figures and the original freeze."""
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage3a_plus_common import (PROJECT, CONFIG, CLASSES, FREEZE, read_json, atomic_json, sha256,
                                verify_freeze, final_checkpoint, plus_manifest, git)
import numpy as np
import torch
from PIL import Image


def main():
    frozen = verify_freeze(data=True); checkpoint, extension = final_checkpoint()
    prereg = read_json(ROOT / '00_manifest/stage3a_plus_preregistration.json')
    for name, digest in prereg['source_sha256'].items(): assert sha256(ROOT / name) == digest
    assert sha256(CONFIG) == prereg['original_config_sha256']
    with (ROOT / '03_no_type_baseline/stage3a_plus_nll_extension_curve.csv').open() as f:
        curve = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    steps = [int(r['extension_step']) for r in curve]
    assert steps == list(range(500, extension['executed_extra_steps'] + 1, 500))
    assert steps[-1] <= 3000 and all(r['learning_rate'] == .0001 for r in curve)
    minimum = extension['original_best']['overall_minFDE6']; bad = 0
    for row in curve:
        improved = row['VAL_overall_FDE'] < minimum
        assert bool(row['improved']) == improved
        bad = 0 if improved else bad + 1
        minimum = min(minimum, row['VAL_overall_FDE'])
        assert row['consecutive_nonimprovements'] == bad
        assert row['best_overall_FDE'] == minimum
    assert abs(minimum - extension['final_best']['overall_minFDE6']) < 1e-10
    assert extension['converged'] == (bad >= 5)
    assert steps[-1] == 3000 or bad >= 5
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert read_json(ROOT / '00_manifest/stage3a_plus_checkpoint_metadata_audit.json')['status'] == 'PASS'
    assert saved['metadata']['global_step'] == extension['final_best']['global_step']
    assert saved['metadata']['config_sha256'] == sha256(CONFIG)
    actor_path = ROOT / extension['final_predictions_relative_path']
    assert sha256(actor_path) == extension['final_predictions_sha256']
    def read_actor_keys(path):
        with path.open() as f:
            return {(r['scene_token'], r['sample_token'], r['instance_token'], r['horizon']):
                    (r['valid_future_steps'], r['agent_type'], r['motion_state'], r['GT_endpoint_displacement_m']) for r in csv.DictReader(f)}
    assert read_actor_keys(actor_path) == read_actor_keys(ROOT / '04_evaluation/stage3_no_type_actor_errors.csv')
    motion = read_json(ROOT / '04_evaluation/stage3_nontrivial_motion_metrics.json')
    assert motion['status'] == 'PASS' and motion['checkpoint_sha256'] == sha256(checkpoint)
    assert len(motion['metrics']) == 5 and all(r['Count'] > 0 and all(np.isfinite(r[k]) for k in ('minADE6', 'minFDE6', 'MR6', 'Top1ADE6', 'Top1FDE6')) for r in motion['metrics'])
    with actor_path.open() as f: actors = list(csv.DictReader(f))
    for row in motion['metrics']:
        selected = [r for r in actors if r['horizon'] == 'full_horizon' and r['agent_type'] == row['AgentType']
                    and float(r['GT_endpoint_displacement_m']) > row['GT_endpoint_displacement_gt_m']]
        assert len(selected) == row['Count']
        assert len({r['instance_token'] for r in selected}) == row['UniqueInstances']
        assert len({r['scene_token'] for r in selected}) == row['UniqueScenes']
        for key in ('minADE6', 'minFDE6', 'MR6', 'Top1ADE6', 'Top1FDE6'):
            assert abs(np.mean([float(r[key]) for r in selected]) - row[key]) < 1e-12
    interaction = read_json(ROOT / '01_data_audit/stage3_interaction_density.json')
    assert interaction['status'] == 'PASS' and interaction['all_shard_hashes_verified']
    assert read_json(ROOT / '00_manifest/stage3a_plus_interaction_consistency_audit.json')['status'] == 'PASS'
    cases = read_json(ROOT / '04_evaluation/stage3a_plus_motion_case_manifest.json')
    assert cases['case_count'] == 6 and len(cases['figures']) == 3
    for item in cases['figures']:
        source = ROOT / item['source_json']; assert sha256(source) == item['source_sha256']
        case = read_json(source); assert case['checkpoint_sha256'] == sha256(checkpoint)
        assert len({p['instance_token'] for p in case['panels']}) == 2
        for panel in case['panels']:
            assert panel['GT_endpoint_displacement_recomputed_m'] > 5
            if item['agent_type'] == 'vehicle': assert panel['motion_state'] == 'vehicle.moving'
        audit = read_json(ROOT / '05_figures' / (item['name'] + '_audit.json'))
        assert audit['status'] == 'PASS' and audit['source_sha256'] == sha256(source)
        for panel in audit['panels']:
            assert max(panel['metric_absolute_differences_m'].values()) < 1e-4
            assert panel['future_marker_count_per_trajectory'] == 12 and panel['endpoint_layout_checks'] == 'PASS'
        for ext, record in audit['exports'].items(): assert sha256(ROOT / record['relative_path']) == record['sha256']
    required_pngs = [ROOT / '03_no_type_baseline/stage3a_plus_nll_extension_curve.png', ROOT / '05_figures/stage3_interaction_density.png']
    required_pngs += [ROOT / '05_figures' / (c['name'] + '.png') for c in cases['figures']]
    for path in required_pngs:
        with Image.open(path) as image: image.verify()
    curve_audit = read_json(ROOT / '03_no_type_baseline/stage3a_plus_nll_extension_curve_audit.json')
    curve_path = ROOT / '03_no_type_baseline/stage3a_plus_nll_extension_curve.csv'
    assert curve_audit['status'] == 'PASS' and sha256(curve_path) == curve_audit['source_CSV_sha256']
    for ext, digest in curve_audit['exports'].items(): assert sha256(curve_path.with_suffix('.' + ext)) == digest
    final_val_path = ROOT / ('04_evaluation/stage3a_plus_best_val_metrics.json' if extension['best_refreshed'] else '03_no_type_baseline/stage3_no_type_val_metrics.json')
    final_val = read_json(final_val_path)['metrics']['full_horizon']
    reference = read_json(PROJECT / 'outputs/stage2c_trainval_vehicle_baseline/04_evaluation/stage2c_val_metrics.json')['metrics']['full_horizon']
    ratios = {'vehicle_overall_FDE': final_val['vehicle']['minFDE6'] / reference['overall']['minFDE6'],
              'vehicle_moving_FDE': final_val['vehicle.moving']['minFDE6'] / reference['vehicle.moving']['minFDE6']}
    ready = extension['converged'] and all(v <= 1.25 for v in ratios.values())
    decision = {'Stage3A_frozen': 'YES' if ready else 'NO', 'Ready_for_Stage3B_Type_Embedding': 'YES' if ready else 'NO',
                'original_Stage3A_files_and_shards_unchanged': True,
                'freeze_definition': 'Decision to declare a converged final scientific baseline; distinct from preservation of original files and saved extension checkpoint.',
                'converged': extension['converged'], 'Stage3B_executed': False,
                'reason': 'Frozen source/data verified; finite motion groups; require five nonimproving validations and vehicle FDE ratios<=1.25, as registered before extension.',
                'vehicle_retention_FDE_ratios_to_Stage2C': ratios, 'final_checkpoint_sha256': sha256(checkpoint)}
    atomic_json(ROOT / '09_reports/stage3a_plus_decision.json', decision)
    original, final = extension['original_best'], extension['final_best']
    lines = ['# Stage3A+ 收尾审计', '',
             '原 Stage3A 文件、数据、模型及结果保持冻结。追加实验恢复正式 best 的 AdamW、随机数和采样游标，使用原始 NLL；所有新 checkpoint 和结果独立保存。未从头训练，未执行 Type Embedding，未使用 test。', '',
             '## 【NLL Extension】', '',
             f"original best = step {original['global_step']}, overall minFDE6 {original['overall_minFDE6']:.9f} m",
             f"new best = step {final['global_step']}, overall minFDE6 {final['overall_minFDE6']:.9f} m",
             f"是否刷新 best = {'YES' if extension['best_refreshed'] else 'NO'}",
             f"converged = {'yes' if extension['converged'] else 'no'}; executed extra steps = {extension['executed_extra_steps']}; stop reason = {extension['stop_reason']}",
             'LR=0.0001、batch=16、原始 NLL、完整 scene-window 与 HiVT64 均不变。每 500 步评价完整 official VAL150，以 full-horizon overall minFDE6 严格改善选 best。收敛在本报告中特指连续 5 次不改善；3000 步上限本身不能证明收敛或达到全局最优。', '',
             f"Final checkpoint: `{final['relative_path']}`; SHA256 `{final['sha256']}`.",
             f"Optimizer/training code commit: `{extension['training_code_git_commit']}`.", '',
             '![NLL extension](../03_no_type_baseline/stage3a_plus_nll_extension_curve.png)', '',
             '## 【Nontrivial Motion】', '',
             '使用最终 best 已保存的 VAL actor 误差，未为分组统计重新训练或重新推理。仅完整 12 步未来；按 norm(GT endpoint − t0 position) 严格大于阈值分组。minADE6 使用 best-FDE mode，MR6 为终点误差大于 2 m；Top1 使用模型概率最大 mode。', '',
             '| Group | Count | minADE6 (m) | minFDE6 (m) | MR6 | Top1ADE6 (m) | Top1FDE6 (m) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in motion['metrics']:
        lines.append(f"| {r['Group']} | {r['Count']} | " + ' | '.join(f"{r[k]:.6f}" for k in ('minADE6', 'minFDE6', 'MR6', 'Top1ADE6', 'Top1FDE6')) + ' |')
    lines += ['', f"完整未来总目标 {motion['full_horizon_target_count']}；排除 partial future {motion['excluded_partial_future_count']}。Count 为 actor-window，阈值组存在嵌套，不能相加。Bicycle 的两组真正运动指标独立列出，不以 overall 代替。", '',
              '[Source table](../06_tables/stage3_nontrivial_motion_metrics.csv)', '', '## 【Interaction Density】', '',
              f"扫描全部 850 frozen shards。可用 graph windows={interaction['graph_window_counts']}；无图 anchors={interaction['anchors_without_graph']}；有图但无监督 target windows={interaction['windows_with_graph_but_no_supervised_targets']}。",
              't0 欧氏距离 <= 10/20/30/50 m，排除目标自身；邻居包含三类当前 context actor，包括非监督 actor，ego 仅为独立坐标参考。Target 使用原 target_mask，含 full 与 partial，未按运动阈值删减。目标到邻居的统计有方向；下面的 pair/window 数对每个 scene-window 的无序参与者对去重，至少一个端点为监督 target。空间邻近表征可用交互 context，不等同于因果交互或 attention 权重。', '',
              '| Pair (train + val) | 20m pairs | 20m windows | 50m pairs | 50m windows |',
              '|---|---:|---:|---:|---:|']
    bicycle_rows = [r for r in motion['metrics'] if r['AgentType'] == 'bicycle']
    # Insert the bicycle sample-coverage note before the interaction section.
    position = lines.index('## 【Interaction Density】')
    lines[position:position] = ['Bicycle 运动子组的独立覆盖：' + '；'.join(
        f">{r['GT_endpoint_displacement_gt_m']}m: {r['Count']} actor-windows / {r['UniqueInstances']} instances / {r['UniqueScenes']} scenes" for r in bicycle_rows) + '。同一 instance 的重叠窗口有相关性，本文只报告描述性指标，未将这些窗口当作独立重复实验。', '']
    for pair, short in (('vehicle-vehicle', 'V-V'), ('vehicle-pedestrian', 'V-P'), ('vehicle-bicycle', 'V-B'),
                        ('pedestrian-pedestrian', 'P-P'), ('pedestrian-bicycle', 'P-B'), ('bicycle-bicycle', 'B-B')):
        a = next(r for r in interaction['pair_counts'] if r['Split'] == 'all' and r['Pair'] == pair and r['Radius_m'] == 20)
        b = next(r for r in interaction['pair_counts'] if r['Split'] == 'all' and r['Pair'] == pair and r['Radius_m'] == 50)
        lines.append(f"| {short} | {a['UniqueTargetIncidentPairs']} | {a['WindowsWithTargetIncidentPair']} | {b['UniqueTargetIncidentPairs']} | {b['WindowsWithTargetIncidentPair']} |")
    lines += ['', '| Target (train + val) | Target count | 20m mean neighbors | 50m mean neighbors |', '|---|---:|---:|---:|']
    pooled = {20: [], 50: []}
    for cls in CLASSES:
        selected = [r for r in interaction['neighbor_statistics'] if r['Split'] == 'all' and r['TargetType'] == cls and r['NeighborType'] == 'all']
        a = next(r for r in selected if r['Radius_m'] == 20); b = next(r for r in selected if r['Radius_m'] == 50)
        pooled[20].append(a); pooled[50].append(b)
        lines.append(f"| {cls} | {a['TargetCount']} | {a['MeanNeighborCount']:.6f} | {b['MeanNeighborCount']:.6f} |")
    means = {radius: sum(r['TotalNeighborCount'] for r in rows) / sum(r['TargetCount'] for r in rows) for radius, rows in pooled.items()}
    lines += ['', f"20m平均邻居数 = {means[20]:.6f}；50m平均邻居数 = {means[50]:.6f}（按 target 数加权的总体均值）。",
              'CSV/JSON 分开提供 train、val、pooled 的全部 4 半径、9 个有向类型组合及每类总邻居的均值/中位数/>=1/>=2 概率；另外提供 all-context 与 target-incident 无序 pair/window 数。', '',
              '![Interaction density](../05_figures/stage3_interaction_density.png)', '', '## Motion cases', '',
              '每类一个 success 和一个 failure，均为完整未来、GT 位移 >5 m；vehicle 还要求 t0 moving。Success/failure 分别按类内 minFDE 最小/最大选择，因此 oracle best 成功并不保证 Top1 成功。两面板包含 History、GT、best 与 Top1，各条未来轨迹 12 个 marker；zoom 自动取三条轨迹最后 6 点加 2.5 m 边距。四项 ADE/FDE 重算误差均 <1e-4 m，无平滑、插值或轨迹改写。']
    for item in cases['figures']:
        lines += ['', f"![{item['agent_type']} motion](../05_figures/{item['name']}.png)",
                  f"[Source JSON](../{item['source_json']})"]
    lines += ['', '## 【Decision】', '', f"Stage3A frozen = {decision['Stage3A_frozen']}",
              f"Ready for Stage3B Type Embedding = {decision['Ready_for_Stage3B_Type_Embedding']}",
              f"Vehicle overall/moving FDE ratios to Stage2C = {ratios}.",
              'Ready 条件在追加训练前记录：冻结检查通过、运动组指标有限、连续 5 次 VAL 不改善，且 vehicle overall/moving FDE 相对 Stage2C 的退化均不超过 25%。该判断不影响 checkpoint selection，也未用于修改实验。若延长训练仍在改善，converged=no 且 Ready=NO，如实保留该限制。',
              '此处 Stage3A frozen 指是否确认当前模型为已收敛的最终科学基线；原 Stage3A 文件和 850 shards 的物理冻结检查为 PASS，旧 checkpoint 完整保留，延长后的 best 也已单独保存。尚未达到连续 5 次不改善，因此本轮不宣告最终科学基线冻结。',
              '所有代码与适合版本管理的小型结果提交到 stage3/multitype-hivt；shards、checkpoint 和大型逐目标 CSV 留在本地并记录 SHA。完成后停止，Type Embedding 未执行，未 merge main。', '',
              '[Execution commands and audit notes](stage3a_plus_execution_commands.md)', '']
    (ROOT / '09_reports/stage3a_plus_final_report.md').write_text('\n'.join(lines))
    atomic_json(ROOT / '00_manifest/stage3a_plus_final_audit.json', {
        'status': 'PASS', 'frozen_files_verified': len(frozen['files']), 'frozen_scene_shards_verified': 850,
        'extension_extra_steps': steps[-1], 'validation_count': len(curve), 'optimizer_LR_batch_loss_model_data_unchanged': True,
        'final_checkpoint_sha256': sha256(checkpoint), 'final_VAL_actor_identity_horizon_motion_displacement_match_original': True,
        'nontrivial_groups_independently_recomputed': 5, 'prediction_panels_numeric_audited': 6,
        'required_PNG_verified': len(required_pngs), 'test_used': False, 'Stage3B_executed': False,
        'decision': decision, 'git_at_audit': git('rev-parse', 'HEAD')})
    verify_freeze(); plus_manifest()
    print('STAGE3A_PLUS_FINAL_AUDIT=PASS', decision, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2); main()
