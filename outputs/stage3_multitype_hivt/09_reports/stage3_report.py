"""Independent final checks and a measured Stage3A report; never start Stage3B."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3_common import (PREVIOUS,CLASSES,GROUPS,HORIZONS,CONFIG,atomic_json,read_json,
                          sha256,git,verify_frozen,update_manifest,write_csv)
import numpy as np
import torch
from PIL import Image


def main():
    verify_frozen()
    prereg=read_json(ROOT/'00_manifest/stage3_formal_preregistration.json')
    for relative,digest in prereg['source_sha256'].items():
        assert sha256(ROOT/relative)==digest, f'Formal optimization source changed: {relative}'
    warm_source=torch.load(ROOT/'07_checkpoints/stage3_warmup_best_overall.pt',map_location='cpu',weights_only=False)
    prereg_commit=warm_source['metadata']['git_commit_SHA']
    manifest_relative=str((ROOT/'00_manifest/stage3_artifact_manifest.json').relative_to(ROOT.parents[1]))
    committed_artifacts=json.loads(git('show',prereg_commit+':'+manifest_relative))
    shard_hashes={r['relative_path']:r['sha256'] for r in committed_artifacts['artifacts']
                  if r['relative_path'].startswith(('02_preprocessed/train/','02_preprocessed/val/')) and r['relative_path'].endswith('.pt')}
    assert len(shard_hashes)==850
    prep=read_json(ROOT/'02_preprocessed/stage3_preprocess_manifest.json')
    assert prep['status']=='COMPLETE' and prep['scene_shards']==850 and prep['failed_shards']==0
    dataset=read_json(ROOT/'01_data_audit/stage3_class_statistics.json')
    stats={(r['split'],r['class']):r for r in dataset['statistics']}
    physical={(s,c,h):0 for s in ('train','val') for c in CLASSES for h in HORIZONS};anchors=0
    for shard in prep['shards']:
        assert sha256(ROOT/shard['relative_path'])==shard_hashes[shard['relative_path']], 'Preprocessed shard changed after preregistration'
        saved=torch.load(ROOT/shard['relative_path'],map_location='cpu',weights_only=False)
        assert saved['input_signature']==prep['input_signature']
        anchors+=len(saved['graphs'])
        for g in saved['graphs']:
            if g is None:continue
            for key,value in g:
                if torch.is_tensor(value) and value.is_floating_point():assert torch.isfinite(value).all(),key
            assert torch.equal(g.target_mask,(g.history_mask.sum(1)>=2)&g.future_mask.any(1))
            for t,c in enumerate(CLASSES):
                full=g.target_mask&g.future_mask.all(1)&(g.agent_type==t)
                partial=g.target_mask&~g.future_mask.all(1)&(g.agent_type==t)
                physical[shard['split'],c,'full_horizon']+=int(full.sum())
                physical[shard['split'],c,'partial_future']+=int(partial.sum())
        del saved
    for s in ('train','val'):
        for c in CLASSES:
            assert physical[s,c,'full_horizon']==stats[s,c]['full_horizon']
            assert physical[s,c,'partial_future']==stats[s,c]['partial_future']
    assert anchors==sum(v['candidate_windows'] for v in prep['splits'].values())
    tiny=read_json(ROOT/'03_no_type_baseline/stage3_tiny_overfit.json');assert tiny['status']=='PASS'
    assert read_json(ROOT/'00_manifest/stage3_input_output_checks.json')['status']=='PASS'
    assert read_json(ROOT/'01_data_audit/stage3_coordinate_checks.json')['status']=='PASS'
    warm=read_json(ROOT/'03_no_type_baseline/stage3_warmup_summary.json');nll=read_json(ROOT/'03_no_type_baseline/stage3_nll_summary.json')
    supervision_steps={c:stats['train',c]['future_valid_count_sum'] for c in CLASSES}
    supervision_fractions={c:n/sum(supervision_steps.values()) for c,n in supervision_steps.items()}
    assert warm['status']==nll['status']=='COMPLETE'
    measured=read_json(ROOT/'03_no_type_baseline/stage3_no_type_val_metrics.json')
    assert not measured['test_used'] and len(measured['scenes'])==150
    for h in HORIZONS:
        for c in CLASSES:
            summary=measured['metrics'][h][c]
            assert summary['count']==physical['val',c,h]
            assert all(v is not None and np.isfinite(v) for k,v in summary.items() if k!='count')
            assert summary['minFDE6']<=summary['Top1FDE6']+1e-5
    checkpoint=ROOT/'07_checkpoints/stage3_best_overall_minfde.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);meta=saved['metadata']
    assert sha256(checkpoint)==measured['checkpoint_sha256']
    assert meta['config_sha256']==sha256(CONFIG) and meta['selection_metric']=='overall minFDE6'
    with open(ROOT/'03_no_type_baseline/stage3_nll_curve.csv') as f:curve=list(csv.DictReader(f))
    assert abs(min(float(r['VAL_overall_FDE']) for r in curve)-meta['validation_FDE'])<1e-8
    assert abs(measured['metrics']['full_horizon']['overall']['minFDE6']-meta['validation_FDE'])<1e-4
    checkpoints=read_json(ROOT/'00_manifest/stage3_checkpoint_manifest.json')['checkpoints']
    for p,record in checkpoints.items():assert sha256(ROOT/p)==record['sha256']
    with open(ROOT/'04_evaluation/stage3_no_type_actor_errors.csv') as f:
        actor_rows=list(csv.DictReader(f))
    assert len(actor_rows)==sum(r['total_target_actor_windows'] for r in dataset['statistics'] if r['split']=='val')
    for r in actor_rows:
        assert all(np.isfinite(float(r[k])) for k in ('minADE6','minFDE6','Top1ADE6','Top1FDE6','NLL'))
    retention=read_json(ROOT/'04_evaluation/stage3_vehicle_retention_audit.json');assert retention['status']=='PASS'
    cases=read_json(ROOT/'04_evaluation/stage3_case_manifest.json');assert len(cases['cases'])==12
    for case in cases['cases']:
        audit=read_json(ROOT/f"05_figures/{case['name']}_audit.json")
        assert audit['status']=='PASS' and max(audit['metric_absolute_differences_m'].values())<1e-4
        layout=audit['layout_checks']
        assert layout['endpoint_labels_in_reserved_side_margins'] and layout['endpoint_text_inside_canvas']
        assert not layout['endpoint_text_overlap_with_inset_title_legend_or_other_labels']
        assert sha256(ROOT/case['source_json'])==audit['source_sha256']
        for ext in ('png','pdf','svg'):
            path=ROOT/f"05_figures/{case['name']}.{ext}"
            assert path.exists() and sha256(path)==audit['exports'][ext]['sha256']
    pngs=list((ROOT/'05_figures').glob('*.png'));assert len(pngs)==15
    for path in pngs:
        with Image.open(path) as image:image.verify()
    full=measured['metrics']['full_horizon'];reference=read_json(PREVIOUS/'04_evaluation/stage2c_val_metrics.json')['metrics']['full_horizon']
    vehicle_ratios={'overall_FDE':full['vehicle']['minFDE6']/reference['overall']['minFDE6'],
                    'moving_FDE':full['vehicle.moving']['minFDE6']/reference['vehicle.moving']['minFDE6']}
    ready=all(v<=1.25 for v in vehicle_ratios.values())
    decision={'Stage3A':'PASS','Ready_for_Type_Embedding':'YES' if ready else 'NO','Stage3B_executed':False,
              'type_information_in_model':False,'test_used':False,'vehicle_FDE_ratios_to_Stage2C':vehicle_ratios,
              'readiness_judgement':'engineering and tiny PASS, finite official VAL per-class metrics; vehicle overall/moving FDE degradation<=25% practical guard; no experiment was changed to meet this guard',
              'vehicle_retention_completed':True,'formal_training_completed':True}
    atomic_json(ROOT/'09_reports/stage3a_decision.json',decision)
    audit={'status':'PASS','physical_scene_shards':850,'all_physical_shards_finite':True,'candidate_anchors':anchors,
           'class_counts_recomputed_from_shards':True,'tiny_three_class_pass':True,'full_VAL_scene_count':150,
           'primary_checkpoint_SHA256':sha256(checkpoint),'primary_config_SHA256':sha256(CONFIG),'checkpoint_manifest_verified':True,
           'formal_optimization_sources_match_preregistration':True,
           'preprocessed_shard_hashes_match_pretraining_commit':prereg_commit,
           'paired_vehicle_actor_windows':retention['exactly_paired_vehicle_actor_windows'],'prediction_case_numeric_audits':12,
           'prediction_endpoint_label_layout_checks':12,
           'figure_png_count':len(pngs),'previous_artifacts_unchanged':True,'test_used':False,'Stage3B_executed':False,
           'git_at_audit':git('rev-parse','HEAD')}
    atomic_json(ROOT/'00_manifest/stage3_final_audit.json',audit)
    report=['# Stage3A Multi-Type HiVT (No Type)','',
            '使用 nuScenes official trainval 的自然类别分布，保留 Stage2C 的 HiVT64 架构与 Protocol 1。三类参与者共享模型，未加入 type embedding、类别加权或 oversampling。Ego 仅提供坐标参考与 context，本轮止于 Stage3A。','',
            '## 【Dataset】','',f"Official scenes: train700 / val150; overlap0; test unused. Candidate/supervised windows: {prep['splits']}",'',
            '| Split | Class | Targets | Full horizon | Partial | Mean valid history | Mean valid future |',
            '|---|---|---:|---:|---:|---:|---:|']
    for r in dataset['statistics']:
        report.append(f"| {r['split']} | {r['class']} | {r['total_target_actor_windows']} | {r['full_horizon']} | {r['partial_future']} | {r['mean_history_valid_length']:.3f} | {r['mean_future_valid_length']:.3f} |")
    report+=['',f"Class ratio counts={dataset['class_ratio_counts']}; normalized to bicycle={dataset['class_ratio_normalized_to_bicycle']}.",
             '表中 Targets 为符合监督条件的 actor-window 数，同一 instance 可以出现在多个窗口。完整 context actor 和无监督 anchor 另有索引；数据未按类别重采样，损失未按类别加权。',
             '', '## 【Tiny Overfit】','', '| Class | Targets | Initial ADE/FDE (m) | Final ADE/FDE (m) | Initial/final fixed-scale regression |', '|---|---:|---|---|---|']
    for c in CLASSES:
        a,b=tiny['initial'][c],tiny['final'][c]
        report.append(f"| {c} | {b['count']} | {a['ADE']:.6f}/{a['FDE']:.6f} | {b['ADE']:.6f}/{b['FDE']:.6f} | {a['fixed_scale_regression_loss']:.6f}/{b['fixed_scale_regression_loss']:.6f} |")
    report+=['',f"TINY_OVERFIT=PASS. {tiny['criterion']}. Train-only complete context; tiny weights discarded for formal initialization.",'',
             '## 【Full Validation】','',
             '主指标仅统计完整 12 步未来。minADE6 使用 best-FDE mode 的 ADE，MR6 为终点误差大于 2 m 的比例；Top1 使用模型概率最大的 mode。独立最小 ADE 另存。NLL 沿用原始损失的 best-summed-L2 mode Laplace 密度，先按 actor 的有效时间与坐标取均值，再对 actor 汇总；连续密度的 NLL 可以为负。','',
             '| Group | Count | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 | NLL |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for c in GROUPS:
        m=full[c];report.append(f"| {c} | {m['count']} | "+' | '.join(f"{m[k]:.6f}" for k in ('minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL'))+' |')
    report+=['','Partial-future 指标单独保存在 [stage3_no_type_partial_results.csv](../06_tables/stage3_no_type_partial_results.csv)，不混入主指标或 checkpoint 选择。','',
             '## 【Vehicle Retention】','',f"Exact paired vehicle actor-windows={retention['exactly_paired_vehicle_actor_windows']}; identities, horizons and real t0 motion states match. No actor deletion or split changes.",'',
             '| Group | Stage2C ADE/FDE/MR | Stage3A ADE/FDE/MR |','|---|---|---|']
    for group,old_group in (('vehicle','overall'),('vehicle.moving','vehicle.moving'),('vehicle.stopped','vehicle.stopped'),('vehicle.parked','vehicle.parked'),('unknown','unknown')):
        a=reference[old_group];b=full[group]
        report.append(f"| {group} | "+'/'.join(f"{a[k]:.6f}" for k in ('minADE6','minFDE6','MR6'))+' | '+'/'.join(f"{b[k]:.6f}" for k in ('minADE6','minFDE6','MR6'))+' |')
    report+=['',f"Vehicle FDE ratios to Stage2C={vehicle_ratios}. Stage2C is a vehicle reference only, not a multi-type overall comparison.",'',
             '## 【Training】','',f"batch size={saved['config']['batch_size']}; steps/epoch={read_json(ROOT/'00_manifest/stage3_training_plan.json')['steps_per_epoch']}; warmup steps={warm['phase_steps']}; NLL steps={nll['phase_steps']}.",
             f"Th={saved['config']['historical_steps']}; Tf={saved['config']['future_steps']}; K={saved['config']['num_modes']}; seed={saved['config']['seed']}; embed_dim={saved['config']['embed_dim']}.",
             f"Primary best global step={meta['global_step']}; NLL phase step={meta['phase_step']}; warm-up source step={meta['warmup_source_step']}; warm-up actually executed={meta['warmup_executed_steps']}.",
             f"Checkpoint SHA256={sha256(checkpoint)}; training-code commit={meta['git_commit_SHA']}; config SHA256={meta['config_sha256']}.",
             '每 500 optimizer steps 完整评价 official VAL。三类 target 均进入原始损失；各类回归损失另作诊断，不改变优化目标。warm-up 与 NLL 的回归损失定义不同，曲线阶段边界已标注。',
             f"全 train 有效 future-step 监督量={supervision_steps}；类别比例={supervision_fractions}。Vehicle 的监督量最多。回归损失沿用每个 batch 的有效时间/坐标均值，分类损失沿用 eligible actor 均值；类内回归均值记录在曲线 CSV 中，不作为类别权重或梯度份额估计。",
             '', '![Train loss](../03_no_type_baseline/stage3_no_type_loss_curve.png)','',
             '![VAL per-class FDE](../03_no_type_baseline/stage3_no_type_val_fde_curve.png)','',
             '## 【Visualization】','',f"Coverage={cases['coverage']}; data-coordinate QA=10 samples/class in3 PNGs; prediction figures=12 PNG/PDF/SVG bundles. Total05_figures PNG count=15.",
             '每张预测图从保存的真实 case JSON 重画，GT、best 与 Top1 各包含 12 个未来 marker，并标记终点。Inset 范围由三条轨迹最后 6 个真实点计算；图前重新计算四项 ADE/FDE，误差均小于 1e-4 m。终点文字置于主图两侧，已检查画布边界及与 inset、标题、图例和其他文字的遮挡。无轨迹平滑、插值或坐标改写。Success 要求明显 GT 位移，vehicle success 为真实 t0 moving；排名案例用于解释误差，整体性能以完整 VAL 表为准。',
             '', '## 【Decision】','',f"Stage3A=PASS; Ready for Type Embedding={decision['Ready_for_Type_Embedding']}; Stage3B executed=False.",
             'Ready 判断依据工程检查和训练前声明的实用条件：vehicle overall 与 moving FDE 相对 Stage2C 的退化均不超过 25%。该条件未用于调整数据、训练或 checkpoint 选择。Stage3B 未执行，type embedding 的收益尚未评价。',
             '', '## Reproducibility','',f"Branch=stage3/multitype-hivt; report-generation commit={git('rev-parse','HEAD')}; final upload verified separately. No merge main.",
             '代码、配置、审计、小型表格与图件纳入版本管理。Scene shards、checkpoint、大型 actor CSV 与传输缓存保留本地，通过 artifact/checkpoint manifest 记录大小和 SHA。',
             '原始数据在预处理时通过只读分区检查；2026-10-05 恢复工作时，原路径未挂载，后续训练与评价使用已冻结的 850 个本地 shards。重新预处理需先恢复配置中的原始数据挂载，详见 resume environment 记录。',
             '[Execution commands and frozen protocol](stage3_execution_commands.md)','']
    (ROOT/'09_reports/stage3a_final_report.md').write_text('\n'.join(report))
    verify_frozen();update_manifest()
    for split,label in (('train','Train'),('val','Val')):
        print(label+': '+', '.join(c+'='+str(stats[split,c]['total_target_actor_windows']) for c in CLASSES),flush=True)
    print('FINAL_AUDIT=PASS; Stage3A=PASS; Ready_for_Type_Embedding='+decision['Ready_for_Type_Embedding'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(4);main()
