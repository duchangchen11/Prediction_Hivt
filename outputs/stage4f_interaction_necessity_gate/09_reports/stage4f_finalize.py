"""Evidence-grounded report, complete technical QA, and local artifact inventory."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'00_manifest'),str(ROOT/'04_evaluation')]
from stage4f_common import (atomic_json,read_json,sha256,verify_previous,git,CONFIG,
    SUMMARY,BEST,CURVE,ACTORS,CLASSES)
from stage4f_evaluate import paired, select_keys, GATES
from PIL import Image
import xml.etree.ElementTree as ET
import subprocess
import numpy as np


def read_csv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def markdown(rows,columns):
    def value(r,c):
        x=r.get(c,'')
        if x is None or x=='':return '—'
        if c in ('Count','Instances','Scenes','parameters'):return str(int(float(x)))
        try:return f'{float(x):.6f}' if c not in ('Group','Model','Comparison','Metric','Population','Feature') else str(x)
        except ValueError:return str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',*['| '+' | '.join(value(r,c) for c in columns)+' |' for r in rows]])


def technical_audit():
    verify_previous(shards=True);maps,membership=paired()
    for name in ('unit_tests','initialization_audit','neutral_initialization_audit','protocol_audit'):
        assert read_json(ROOT/f'00_manifest/stage4f_{name}.json')['status']=='PASS'
    for name in ('gradient_audit','tiny_overfit','checkpoint_resume_audit','target_gate_broadcast_audit','metrics','pairing_audit','fresh_val_reconciliation','gate_statistics','efficiency_audit','qualitative_case_manifest'):
        assert read_json(ROOT/f'04_evaluation/stage4f_{name}.json')['status']=='PASS'
    train=read_json(SUMMARY);assert train['status']=='COMPLETE' and train['warmup_steps']==5000 and train['final_executed_global_step']<=21000
    assert sha256(BEST)==train['checkpoint_sha256'];curve=read_csv(CURVE)
    steps=[int(float(r['global_step'])) for r in curve]
    assert steps==list(range(500,train['final_executed_global_step']+1,500))
    assert train['stop_reason'] in ('patience_5','global21000_budget')
    if train['stop_reason']=='patience_5':assert int(float(curve[-1]['consecutive_nonimprovements']))==5
    for name,digest in read_json(ROOT/'00_manifest/stage4f_training_sources.json')['training_source_sha256'].items():assert sha256(ROOT/name)==digest
    gate=read_json(ROOT/'04_evaluation/stage4f_gate_statistics.json');assert gate['full']==54990 and gate['partial']==30037
    main=read_csv(ROOT/'06_tables/stage4f_main_results.csv')
    fresh=read_json(ROOT/'04_evaluation/stage4f_metrics.json')['metrics']['full_horizon']
    for row in main:
        if row['Model']=='Stage4F':
            expected=fresh[row['Group']];assert int(row['Count'])==expected['count']
            if expected['count']:
                for metric in ('minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL'):
                    assert abs(float(row[metric])-expected[metric])<1e-8
    efficiency=read_csv(ROOT/'06_tables/stage4f_efficiency.csv');timings=read_csv(ROOT/'06_tables/stage4f_efficiency_batch_timings.csv')
    assert len(timings)==500 and [int(r['parameters']) for r in efficiency]==[646001,647609,647786]
    for row in efficiency:
        values=np.array([float(t[row['Model']+'_ms']) for t in timings])
        assert abs(values.mean()-float(row['mean_inference_ms']))<1e-9
        assert abs(np.median(values)-float(row['median_inference_ms']))<1e-9
    cases=read_json(ROOT/'04_evaluation/stage4f_qualitative_case_manifest.json');assert cases['count']>=4
    exports=[]
    for p in sorted((ROOT/'05_figures').glob('stage4f_*.png')):
        with Image.open(p) as im:dimensions=im.size;im.verify()
        svg=p.with_suffix('.svg');pdf=p.with_suffix('.pdf');assert svg.exists() and pdf.exists()
        texts=ET.parse(svg).findall('.//{http://www.w3.org/2000/svg}text');assert texts
        extracted=subprocess.check_output(['pdftotext',str(pdf),'-']).decode();assert extracted.strip()
        fonts=subprocess.check_output(['pdffonts',str(pdf)]).decode();assert 'Type 3' not in fonts
        exports.append({'name':p.name,'PNG_dimensions':dimensions,'SVG_text_nodes':len(texts),'PDF_selectable_text':True,'PDF_no_Type3':True})
    assert len(exports)>=9
    scientific=read_json(ROOT/'04_evaluation/stage4f_scientific_decision.json')
    assert scientific['STOP'] and not scientific['Reliability_executed'] and not scientific['additional_seeds_executed']
    result={'status':'PASS','Stage4F':'PASS','initialization_gradient_tiny_checks':'PASS','fresh_full_VAL_and_pairing':'PASS',
        'official_train700_val150_test_unused':True,'training_sources_unchanged':True,'old_roots_and_scene_shards_unchanged':True,
        'gate_only_history_no_future_GT':True,'attention_topology_frozen_complete_graph':True,'gate_feature_neighbor_radius_m':50,
        'formal_warmup_updates':5000,'formal_NLL_updates':train['NLL_steps'],'best_global_step':train['best_global_step'],
        'stop_reason':train['stop_reason'],'efficiency500paired_verified':True,'figure_exports':exports,
        'artifact_prefix':'stage4f_','new_root_isolated':True,'full':54990,'partial':30037,'total':85027,
        'scientific_decision_separate_from_technical_PASS':True,'no_second_innovation':True}
    atomic_json(ROOT/'00_manifest/stage4f_final_audit.json',result)
    return result


def report():
    train=read_json(SUMMARY);decision=read_json(ROOT/'04_evaluation/stage4f_scientific_decision.json');gates=read_json(ROOT/'04_evaluation/stage4f_gate_statistics.json')
    ablation=read_csv(ROOT/'06_tables/stage4f_three_model_ablation.csv');boot=read_csv(ROOT/'06_tables/stage4f_bootstrap_ci.csv')
    efficiency=read_csv(ROOT/'06_tables/stage4f_efficiency.csv');cases=read_json(ROOT/'04_evaluation/stage4f_qualitative_case_manifest.json')['cases']
    subgroup=read_json(ROOT/'01_data_audit/stage4f_offline_subgroup_registration.json')['counts']
    columns=['Group','Count','Stage3B_minFDE6','Stage4A_minFDE6','Stage4F_minFDE6','D-B_minFDE6','D-C_minFDE6']
    table=lambda groups:markdown([r for r in ablation if r['Group'] in groups],columns)
    g={r['Group']:r for r in gates['groups'] if r['Population']=='full_horizon'}
    eff=read_json(ROOT/'04_evaluation/stage4f_efficiency_audit.json')
    neutral=read_json(ROOT/'00_manifest/stage4f_neutral_initialization_audit.json')
    fullmetrics=read_csv(ROOT/'06_tables/stage4f_main_results.csv')
    lines=['# Stage4F：Interaction Necessity-Gated Type-Conditioned Interaction','',
        '## Research Motivation','',
        '冻结的 Stage4A always-on relation bias 改善 vehicle，却可靠损害 pedestrian；Stage4A-E 将主要损害定位于 pedestrian <5m，global gradient conflict 证据为 WEAK。单一研究问题是：仅根据目标 actor 可观测历史和当前邻域，用 scalar necessity gate 控制额外 relation enhancement，能否保留 moving vehicle 收益并恢复低运动 pedestrian？该假设不是既定事实，下面按预注册规则判定。','',
        '## Method','',
        '保留 LocalEncoder、TypeEmbedding、decoder、原 HiVT update gate、value/aggregation、FFN 和 LayerNorm。复用 Stage4A 的 directed pair_id=3×target_type+source_type，Embedding(9,16)，20→32→24 relation MLP，输出每条边3layers×8heads。仅新增9→16→1 ReLU MLP，经 sigmoid 得到目标 scalar g_i；所有 incoming edges、layers、heads 共享 g_i。Attention logits = base + g_i×b_ij。新增177参数，总647786；相对 Stage3B 新增1785参数。无 gate label、额外监督、正则损失、类别重权或 oversampling。','',
        '九维输入依次为 V/P/B onehot3、log1p recent displacement、log1p history net displacement、log1p history path length、log1p current50m incoming neighbor count、minimum distance/50、heterogeneous fraction。历史路径连接按时间排序的连续有效观测，跳过 padding；不足两点时运动统计为0；无邻居时 count=0、distance/50=1、heterogeneous fraction=0。','',
        '**拓扑审计修正**：冻结实现的 GlobalInteractor 实际为当前有效 actor 完整有向图；50m 截断仅位于 LocalEncoder。为了保持 B/C/D 可比，本轮完全保留该 attention 拓扑，门控邻域特征只统计这些已有边中当前距离≤50m 的邻居；未新增、删除或重连 attention edges。这一口径在正式训练前记录。','',
        '## No Future Leakage Audit','',
        '纯特征函数只接收 positions[:,:5]、padding_mask[:,:5]、agent_type、已有 current edges；不能接收未来 GT、future displacement、未来标签、decoder output 或 prediction error。十二项测试包括手算历史、padding gap、单观测、无邻居、50m边界、真实输入原样保留、未来坐标/标签/mask扰动下输出一致。Pedestrian GT motion bins 仅用于离线评价和案例选择，未进入 gate 或损失。','',
        '## Initialization','',
        f'从 seed2022 canonical Stage3B 和 Stage4A step0 初始化，所有共同参数/缓冲位级一致，max diff=0；relation step0 与 Stage4A 完全一致，未加载任一 trained checkpoint。Relation final weight/bias=0；gate final weight=0、bias=logit(0.1)，初始 g≈0.1。真实 TRAIN batch 在 CPU single-thread eval 上 raw/logit/prob 最大差分别为 {neutral["raw_prediction_max_abs_diff"]}, {neutral["mode_logits_max_abs_diff"]}, {neutral["mode_prob_max_abs_diff"]}，均小于1e-6。','',
        '10次独立优化诊断：第一步 relation final gradient>0，gate gradient=0符合零 relation bias 的链式求导；10步内 relation MLP、pair embedding、gate MLP均获得非零梯度。模型随后丢弃。独立 tiny TRAIN 六完整窗口覆盖11vehicle、12pedestrian、6bicycle targets，历史覆盖moving V、low-motion P、moving P和VP context；1000步后所有三类回归损失下降、关系偏置非零、门控分化、无NaN/Inf。tiny不用于超参数选择，正式训练重新从头初始化。','',
        '## Training','',
        f'Official TRAIN700/VAL150，test未使用；沿用 frozen scene shards，未重新预处理。Th5/Tf12/K6，batch16，embed64，heads8，global3，dropout0.1，LocalEncoder radius50，AdamW weight_decay1e-4。Fixed-scale LR0.001执行5000updates；恢复自己的 warm-up best（global step {train["warmup_best_source_step"]}）model/optimizer/RNG，仅改LR0.0001进入原 learnable-scale Laplace NLL。每500步全VAL，strict overall full-horizon minFDE改善才更新best。NLL执行{train["NLL_steps"]}步，最终global {train["final_executed_global_step"]}，stop reason={train["stop_reason"]}，best global step={train["best_global_step"]}。','',
        f'训练源代码 commit：`{train["training_code_git_commit"]}`。Best checkpoint SHA256：`{train["checkpoint_sha256"]}`。Final指标来自重新加载该checkpoint的fresh完整VAL。CUDA原协议scatter非确定性可能改变近似并列mode的Top1 argmax，因此训练/fresh核对对selection指标保持1e-6、Top1报告允许1e-4；阈值在正式训练前注册，未重新选择checkpoint。','',
        '## Main Results','',table(('overall','vehicle','pedestrian','bicycle')),'',
        'ADE取 best-FDE mode，不是独立最小ADE；FDE取6mode最小endpoint error；MR为endpoint error>2m；Top1使用最高预测概率；NLL沿用原best-summed-L2 mode。所有窗口等权，full与partial分开。三模型所有85027条identity严格相同，其中54990full、30037partial；scene/sample/instance/node/type/motion/mask/GT trajectory SHA均一致。','',
        markdown([r for r in fullmetrics if r['Model']=='Stage4F'],['Group','Count','minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6','NLL']),'',
        '## Vehicle Motion','',table(('vehicle.moving','vehicle.stopped','vehicle.parked','unknown','Vehicle >5m')),'',
        '## Pedestrian Motion Bins','',table(tuple(f'Pedestrian {a}-{b}m' for a,b in ((0,1),(1,2),(2,5),(5,10),(10,20)))),'',
        '固定为左闭右开区间，不因本轮结果更改。Bicycle >1m/>5m与Pedestrian >1m/>5m见 nontrivial_motion_metrics.csv；稀疏bicycle motion组仅作描述，不宣称强改善。','',
        '## Low-Motion Recovery','',
        f'正式训练前注册 Pedestrian <5m：{subgroup["Pedestrian <5m"]["Count"]}actor-windows、{subgroup["Pedestrian <5m"]["Instances"]}instances、{subgroup["Pedestrian <5m"]["Scenes"]}scenes；5–10m：{subgroup["Pedestrian 5-10m"]["Count"]}windows、{subgroup["Pedestrian 5-10m"]["Instances"]}instances、{subgroup["Pedestrian 5-10m"]["Scenes"]}scenes。核心恢复比较是 D−C，同时用D−B约束可靠负迁移。','',table(('Pedestrian <5m','Pedestrian 5-10m')),'',
        '## Interaction Context','',table(('Heterogeneous-20m','Vehicle hetero-20m','Pedestrian hetero-20m','VP-context-20m')),'',
        '完全复用冻结的20m interaction ledger：t0当前有效、不同type、非自身、至少一个邻居；VP仅V/P异类context。未使用本轮误差或未来运动改变group定义。空间context不能证明因果交互。','',
        '## Bootstrap','',
        'Paired scene-cluster percentile bootstrap：150official VAL scenes，1000replicates，seed2022；同一重采样保留整scene的配对actor-window deltas，按window等权池化，95% percentile CI。Primary为Overall D−B minFDE；次要组和bins为预注册/诊断评价，intervals未作多重比较校正。Bootstrap处理scene内窗口相关性，不提供跨训练随机种子的稳定性证据。','',
        markdown([r for r in boot if r['Metric']=='minFDE6' and r['Group'] in ('overall','vehicle','pedestrian','vehicle.moving','Vehicle >5m','Pedestrian <5m','Pedestrian 5-10m','Heterogeneous-20m','VP-context-20m')],['Comparison','Group','Delta','CI95_lower','CI95_upper','Count','Scenes']),'',
        '## Gate Behavior','',
        markdown([r for r in gates['groups'] if r['Population']=='full_horizon'],['Group','Count','mean','std','median','p10','p25','p75','p90','fraction_lt0p1','fraction_gt0p5','fraction_gt0p9']),'',
        f'Full-horizon mean gate：moving/stopped/parked vehicle={g["vehicle.moving"]["mean"]:.6f}/{g["vehicle.stopped"]["mean"]:.6f}/{g["vehicle.parked"]["mean"]:.6f}；P0–1/P5–10={g["Pedestrian 0-1m"]["mean"]:.6f}/{g["Pedestrian 5-10m"]["mean"]:.6f}。这些真实描述统计不强制符合预期，不构成“门控恢复误差”的因果证明。','',
        f'Gate collapse={gates["gate_collapse"]}；定义为all-current-valid actor-window observations中>95%落在g<0.05或g>0.95。全context、full和partial统计分别记录，未人为改变gate。Spearman对六个可观测历史/邻域量计算，见 gate_feature_correlations.csv；log1p及/50为单调变换，不改变对应raw量的rank关联。仅描述相关，不解释为因果，重叠窗口的nominal p-value不作为独立样本显著性证据。','',
        '## Efficiency','',markdown(efficiency,['Model','parameters','mean_inference_ms','median_inference_ms','peak_CUDA_memory_MiB']),'',
        f'同一GPU相同batch，warm20后执行500paired triplets，循环B/C/D全部六种执行顺序；CUDA events计时forward，排除I/O/H2D和外部input clone，所有模型内部clone均保留。Peak memory由各模型单独在GPU、独立warm/reset、遍历相同226unique VAL batches测量。D相对B mean latency overhead={eff["mean_overhead_D_vs_B_percent"]:.3f}%，相对C={eff["mean_overhead_D_vs_C_percent"]:.3f}%。','',
        '## Qualitative','',markdown(cases,['name','case_kind','gate','B_FDE','C_FDE','D_FDE']),'',
        '四个真实例子展示moving vehicle保留收益、low-motion pedestrian恢复、moving pedestrian和真实退化。三列同actor、相同GT/坐标范围；灰圆history、黑方GT、蓝圆Best、橙虚线三角Top1，未来每条12原始marker。每个模式/指标从同一冻结VAL batch replay核对，未插值、平滑、修改轨迹。案例在定量评价锁定后按效果选择，仅作机制说明；总体判断由完整数据bootstrap决定。','',
        '## Scientific Decision','',
        f'Interaction Necessity Gate = **{decision["scientific_decision"]}**。Stage4F = **PASS**（技术完成与科学支持分开）。Ready Reliability Head = **{decision["Ready_Reliability_Head"]}**。','',
        f'判定事实：Overall reliable gain={decision["overall_gain_reliable"]}；Pedestrian reliable harm={decision["pedestrian_reliable_harm"]}；major-class harm guard={decision["major_class_harm_guard"]}；moving vehicle reliable gain={decision["moving_vehicle_gain_reliable"]}；P<5相对C恢复={decision["low_motion_recovery_from_C"]}、可靠恢复={decision["low_motion_recovery_from_C_reliable"]}；P<5相对B无可靠损害={decision["low_motion_D_minus_B_no_reliable_harm"]}。','',
        '遵循单一seed2022筛选协议，没有超参数搜索或额外seed，没有进入Reliability Head、mode re-ranking或任何第二创新。即使Ready=YES，也须后续研究决策再决定稳定性seed实验；本轮在Stage4F完成后STOP。','',
        '## Artifacts','',
        '所有新增代码、报告、图表、checkpoint和本地逐actor CSV位于独立Stage4F根目录，Stage3与旧Stage4只读。逐actor误差/gate、checkpoint、每500步完整scene metrics及日志保留本地，Git只存其SHA/规模和适合版本管理的脚本、最终小指标、表图报告。artifact_manifest.json为完整local inventory，self hash除外。']
    (ROOT/'09_reports/stage4f_final_report.md').write_text('\n'.join(lines)+'\n')


def inventory():
    destination=ROOT/'00_manifest/stage4f_artifact_manifest.json';rows=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p==destination or '__pycache__' in p.parts or 'stage4f_git_upload_' in p.name or p.name.endswith('.tmp'):continue
        assert p.name.startswith('stage4f_'),p
        ignored=subprocess.run(['git','check-ignore','-q',str(p)],stdout=subprocess.DEVNULL).returncode==0
        rows.append({'relative_path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':sha256(p),'git_eligible':not ignored})
    atomic_json(destination,{'stage':'Stage4F','root':str(ROOT),'self_hash_excluded':True,'old_roots_read_only':True,'artifacts':rows})


if __name__=='__main__':
    technical_audit();report();inventory();print('STAGE4F_FINALIZATION_PASS',flush=True)
