"""Evidence-based mixed-mechanism report, final QA and diagnostic-only inventory."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'04_evaluation/diagnostics'))
from stage4fd_common import *
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image

def rows(name):
    with (ROOT/'06_tables'/name).open() as f:return list(csv.DictReader(f))

def table(rr,columns):
    def value(r,k):
        x=r.get(k,'');
        if x in ('',None):return '—'
        if k in ('Group','Weighting','Pair','Measurement','Layer','Head'):return str(x)
        if k in ('TargetCount','EdgeCount'):return str(int(x))
        try:return f'{float(x):.6f}'
        except (ValueError,TypeError):return str(x)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |',*['| '+' | '.join(value(r,k) for k in columns)+' |' for r in rr]])

def main():
    verify();capture=read_json(DIAG/'stage4fd_capture_audit.json');statistics=read_json(DIAG/'stage4fd_statistics_audit.json')
    assert capture['status']==statistics['status']=='PASS'
    assert read_json(DIAG/'stage4fd_instrumentation_audit.json')['status']=='PASS'
    group=rows('stage4fd_effective_strength_by_group.csv');layers=rows('stage4fd_layer_statistics.csv');heads=rows('stage4fd_layer_head_statistics.csv')
    pairs=rows('stage4fd_effective_strength_by_pair.csv');corr=rows('stage4fd_gate_bias_correlations.csv')
    lookup={(r['Group'],r['Weighting']):r for r in group};overall=lookup['Overall','EdgeWeighted']
    exception=[r for r in heads if r['Layer']=='2' and r['Head']=='3' and r['Group'] in ('Overall','Vehicle','Pedestrian')]
    assert len(exception)==6
    all_heads_by_group={}
    for g in ('Overall','Vehicle','Pedestrian'):
        rr=[r for r in heads if r['Group']==g and r['Weighting']=='EdgeWeighted']
        all_heads_by_group[g]={'heads':24,'raw_amplification_above_descriptive_reference1':sum(float(r['RawAmplification'])>1 for r in rr),
            'effective_retention_range':[min(float(r['EffectiveRetention']) for r in rr),max(float(r['EffectiveRetention']) for r in rr)],
            'attention_shift_retention_range':[min(float(r['AttentionShiftRetention']) for r in rr),max(float(r['AttentionShiftRetention']) for r in rr)]}
    decision={'GateCompensation':'MIXED','decision_basis':'Pooled raw and effective amplitudes and attention shifts are strongly reduced under both weightings, but Layer2/Head3 shows raw amplification and substantial attention retention across V/P; V<-B also has a mild opposite raw-amplitude direction. This is localized compensation, not population-wide cancellation of gate shrinkage.',
        'global_compensation_explanation_supported':False,'localized_head_exception_present':True,'localized_head_exception':exception,
        'all24_head_summary':all_heads_by_group,'reference1_is_not_significance_threshold':True,'no_new_threshold_or_causal_inference':True,
        'FrozenStage3B':'SUPPORTED','FrozenStage4A':'NOT SUPPORTED','FrozenStage4F':'NOT SUPPORTED','FrozenReadyReliabilityHead':'NO',
        'RecommendedNextStep':'等待大脑AI根据全局衰减与 Layer2/Head3 局部模式判断；不要立即训练 bounded variant。',
        'Training_executed':False,'counterfactual_selection_executed':False,'new_checkpoint_created':False,'next_model_executed':False,'STOP':True}
    atomic_json(DIAG/'stage4fd_scientific_decision.json',decision)
    amplitude=['Group','Weighting','TargetCount','GateMean','A_MeanAbsRawBias','F_MeanAbsRawBias','F_MeanAbsEffectiveBias','RawAmplification','EffectiveRetention','CompensationIndex']
    attention=['Group','Weighting','A_AttentionL1Shift','F_AttentionL1Shift','AttentionShiftRetention','A_TopNeighborSwitchRate','F_TopNeighborSwitchRate','DeltaSwitchRate','A_DeltaEntropy','F_DeltaEntropy']
    select=lambda groups:[r for r in group if r['Group'] in groups]
    lines=['# Stage4F-D：Effective Interaction Strength / Gate Compensation Audit','',
        '## Question','',
        'Does the relation branch compensate for the necessity gate? 本轮研究 g_i 下降是否被 raw relation amplitude 放大抵消，而不是再次选择 checkpoint 或搜索变体。结论仅描述 seed2022 的两份冻结模型。','',
        '## Frozen Models','',
        *[f'- Stage4{"A" if k=="A" else "F"} checkpoint SHA256：`{d}`。' for k,(p,d) in CHECKPOINTS.items()],
        '',f'基于 commit `{BASE_COMMIT}`，两模型全程 eval/no_grad，参数 requires_grad=False，未调用 backward、optimizer 或任何训练。参数/缓冲位级哈希前后相同，全部 grad=None。Stage3B SUPPORTED、Stage4A/Stage4F NOT SUPPORTED、Ready Reliability Head=NO 均保留。','',
        '## Pairing Audit','',
        f'Official VAL150、3603 windows、batch16且shuffle=False。实际边严格配对 {capture["actual_edges"]:,} 条，对应3layers×8heads共 {capture["edge_head_observations"]:,} 个 edge-layer-head observations；EdgeKey为scene/sample/source-instance/target-instance，顺序哈希 `{capture["ordered_EdgeKey_sha256"]}`。每个batch逐层确认源/目标、pair ID、index和完整边顺序一致；每个scene-window均验证 current-valid complete directed graph 的 n(n−1) 条非自身边，无跨scene边。','',
        'GlobalInteractor 是完整当前有效图。50m只限制LocalEncoder，以及 Stage4F gate 输入中从已有边筛出的近邻统计；本轮没有修改任何 actor、edge、padding、type、GT 或map。两模型从同一原始batch克隆输入，原始所有张量/身份字段的fingerprint保持不变。','',
        f'全current-valid共 {capture["all_current_valid_targets"]:,} actor-windows，full54990、partial30037、context-only{capture["context_only"]}。其中28个无incoming edge，18个为full vehicle target（moving15、parked3）；其gate保留，但edge/attention均值未定义，统计中明确排除。因此主强度表Overall TargetCount={overall["TargetCount"]}，incoming full-target edges={overall["EdgeCount"]}。这未改变原预测评价的54990/30037数量。','',
        '正式 message AST 仅插入观察语句，删除这些语句后与原AST完全一致；原 q/k/value、加法、PyG softmax、dropout、聚合完全保留。固定真实CPU batch插桩前后 raw/logits/prob差均为0。正式VAL中 captured final_A=base_A+b_A、final_F=base_F+g_i b_F，max diff均为0；真实dropout入口alpha与同final的PyG softmax复核最大差小于1e-6，微小差异来自原CUDA scatter的非确定归约。','',
        '## Overall Interaction Strength','',table(select(('Overall','Vehicle','Pedestrian','Bicycle')),amplitude),'',
        'EdgeWeighted对应每条edge/layer/head等权；TargetWeighted先在每target内部平均全部incoming edges和heads/layers，再等权平均target。前者attention及gate也按target degree加权，后者是标准等权target-layer-head平均L1/entropy/switch。两套结果完整保留，不能将edge-weighted gate均值混用为actor均值。所有比值均为对应加权均值的比值，eps=1e-8；CompensationIndex=EffectiveRetention/(GateMean+eps)，1仅为raw不变的描述性参考。','',
        table(select(('Overall','Vehicle','Pedestrian','Bicycle')),['Group','Weighting','A_MeanAbsBase','F_MeanAbsBase','A_NormalizedInteraction','F_NormalizedInteraction','NormalizedInteractionRetention']),'',
        f'Overall EdgeWeighted raw amplification={float(overall["RawAmplification"]):.6f}，effective retention={float(overall["EffectiveRetention"]):.6f}，normalized retention={float(overall["NormalizedInteractionRetention"]):.6f}，attention shift retention={float(overall["AttentionShiftRetention"]):.6f}。因此全局层面并非“raw变大抵消gate”：raw branch本身更弱，gate再进一步减小实际注入量。两个模型的base states独立训练且不同；比值不是相同base模型的因果反事实。','',
        '## Vehicle Motion','',table(select(('vehicle.moving','vehicle.stopped','vehicle.parked')),amplitude),'',
        'Moving vehicle的effective bias大于parked/stopped；差异与可观测运动gate分化方向一致。但其相对Stage4A的effective strength仍明显下降，此处不推断性能改善。','',
        '## Pedestrian Motion','',table(select(tuple(f'Pedestrian {lo}-{hi}m' for lo,hi in ((0,1),(1,2),(2,5),(5,10),(10,20)))),amplitude),'',
        table(select(('Pedestrian <5m','Pedestrian 5-10m')),amplitude),'',
        '低运动P的effective bias比5–10m组低；各组pooled F raw本身也远低于A。GT endpoint bins仅来自冻结actor ledger，分组发生在forward观察之后，没有进入gate、relation branch或任何新模型输入。','',
        '## Directed Type Pairs','',table([r for r in pairs if r['Layer']=='All'],['Pair','EdgeCount','GateMean','A_Raw','F_Raw','F_Effective','RawAmplification','EffectiveRetention','CompensationIndex']),'',
        'Pair=target←source，与pair_id=3×target_type+source_type一致。V←V、V←P、P←V、P←P均显示pooled raw及effective衰减。V←B的raw ratio略大于1，但effective retention仍低于其原A幅度，且该pair只占少量full-target edges；不把稀疏pair证据外推为整体补偿。各pair三layer及24head明细均已保存。','',
        '## Layer Analysis','',table([r for r in layers if r['Group']=='Pedestrian'],['Weighting','Layer','GateMean','RawAmplification','EffectiveRetention','CompensationIndex','NormalizedInteractionRetention','AttentionShiftRetention']),'',
        '三个Pedestrian layer的pooled raw/effective强度均减小，但均值掩盖了局部例外。下面同时显示两种加权的Layer2/Head3：','',
        table(exception,['Group','Weighting','Layer','Head','RawAmplification','EffectiveRetention','CompensationIndex','NormalizedInteractionRetention','AttentionShiftRetention']),'',
        '在Overall/Vehicle/Pedestrian的24个head中，各只有Layer2/Head3的raw amplification大于描述性参考1。该head的Pedestrian raw amplification约2.15，effective retention约0.49，但attention shift retention约0.87；Vehicle attention shift甚至略大于A。这说明“所有head都被同样压小”的叙述不成立。其余head存在不同强度的attention保留，完整24head CSV和图中矩阵保留这些信息，没有选择性省略或平滑。','',
        '## Attention Shift','',table(select(('Overall','Vehicle','Pedestrian','vehicle.moving','vehicle.stopped','vehicle.parked','Pedestrian 0-1m','Pedestrian 1-2m','Pedestrian 2-5m','Pedestrian 5-10m')),attention),'',
        'alpha_final直接取自正式forward的dropout入口；alpha_base用该layer实际base logits和同一完整incoming neighborhood，调用原PyG softmax。L1是sum_j|alpha_final−alpha_base|；neighbor argmax用原edge顺序的first-edge tie break；entropy为自然对数，0log0=0。此参考仅是layer内固定base的注意力观察，没有执行base-only模型rollout、counterfactual性能评价或变体选择。','',
        'Attention entropy下降只描述注意力集中度变化，不能自动解释为更好。绝对logit magnitude还包含softmax不敏感的共同偏移，因此以base归一化与实际attention结果一起判断。','',
        '## Interaction Context','',table(select(('Heterogeneous-20m','non-heterogeneous','Vehicle hetero-20m','Pedestrian hetero-20m','VP-context-20m')),amplitude),'',
        '20m membership完全复用冻结ledger；non-heterogeneous为其full-target补集。完整incoming softmax未截断到20m，也未重新定义组。','',
        '## Gate–Raw Bias Correlation','',table(corr,['Group','Measurement','TargetCount','Spearman_rho']),'',
        '在每target先平均edge/layer/head后做Spearman。Overall/Vehicle gate与raw平均幅度呈负相关，但Pedestrian以及P<5m、P5–10m为正相关；相关方向并不一致，且不能否定F pooled raw远小于A的直接观察。这些窗口有重叠，跨类型/邻域组成也可能影响相关，不作为因果或显著性证明；未事后规定统计阈值。','',
        '## Scientific Decision','',
        '**GateCompensation = MIXED**。不存在一致的全局amplitude compensation证据：pooled raw、effective、base-normalized强度和attention shifts在两种加权下均大幅降低。但Layer2/Head3确有局部raw放大和较高attention保留，少量pair也出现不同方向；所以不将本轮简化成“完全没有任何补偿”。MIXED描述全局衰减与局部补偿共存，而不是证明局部head导致pedestrian性能退化，更不是改写Stage4F的NOT SUPPORTED结论。','',
        '## Recommended Next Step','',decision['RecommendedNextStep'],
        '本轮完成后STOP。未实现bounded modulation、tanh/clip、gate scale/lambda变体、训练、fine-tune、额外seed或Reliability Head。','',
        '## Artifacts','',
        '四张图的底层artist数值逐项与完整精度CSV比对，absolute difference=0，满足<1e-8。图中文字注释保留3位小数（head矩阵2位）、报告表保留6位以便阅读；这些格式化注释不作为完整精度数值，完整数值和逐项比较在CSV/figure audit JSON中保留。','',
        f'本地target_strength.csv共{statistics["target_table"]["row_count"]}行，SHA256 `{statistics["target_table"]["sha256"]}`；schema记录于statistics_audit.json。raw_target_head.bin与identity CSV保留本地且记录SHA，支持重算全部group/layer/head；Git仅上传脚本、requirements、审计JSON、较小汇总CSV、四图PNG/PDF/SVG及报告，原Stage4F结果完全冻结。']
    (ROOT/'09_reports/stage4fd_effective_interaction_strength_report.md').write_text('\n'.join(lines)+'\n')
    exports=[]
    for p in sorted((ROOT/'05_figures').glob('stage4fd_*.png')):
        audit=read_json(p.with_name(p.stem+'_audit.json'));assert audit['status']=='PASS' and audit['max_absolute_difference']<1e-8
        for name,digest in audit['CSV_source_sha256'].items():assert sha256(ROOT/name)==digest
        for name,digest in audit['exports_sha256'].items():assert sha256(ROOT/name)==digest
        with Image.open(p) as im:im.verify()
        assert ET.parse(p.with_suffix('.svg')).findall('.//{http://www.w3.org/2000/svg}text')
        assert subprocess.check_output(['pdftotext',str(p.with_suffix('.pdf')),'-']).decode().strip()
        assert 'Type 3' not in subprocess.check_output(['pdffonts',str(p.with_suffix('.pdf'))]).decode()
        exports.append(p.name)
    visual=read_json(DIAG/'stage4fd_manual_visual_qa.json');assert visual['status']=='PASS' and len(exports)==len(visual['figures'])==4
    for r in visual['figures']:assert sha256(ROOT/r['path'])==r['sha256']
    assert statistics['layer_head_rows']==1008 and len(pairs)==36
    # Diagnostic sources may observe or summarize only; no forbidden actions.
    import ast
    for p in ROOT.rglob('stage4fd_*.py'):
        tree=ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):assert node.func.attr not in ('backward','step','train'),str(p)
    verify()
    atomic_json(DIAG/'stage4fd_final_audit.json',{'status':'PASS','frozen_inputs_unchanged':True,'model_parameters_unchanged':True,'full_VAL_edge_pairing':'PASS',
        'actual_forward_algebra':'PASS','statistics_both_weightings':'PASS','all_24_heads_retained':'PASS','CSV_figure_absolute_difference_max':0.0,
        'four_PNG_PDF_SVG_exports':exports,'manual_visual_QA':'PASS','no_backward_optimizer_train_checkpoint_selection':True,
        'original_scientific_conclusions_unchanged':True,'GateCompensation':decision['GateCompensation'],'STOP':True})
    print('STAGE4FD_FINAL_REPORT_PASS',decision['GateCompensation'],flush=True)
    inventory=[]
    for p in sorted(ROOT.rglob('stage4fd_*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name.startswith('stage4fd_git_upload_') or p.name.endswith('.tmp') or p.name=='stage4fd_artifact_manifest.json':continue
        ignored=subprocess.run(['git','check-ignore','-q',str(p)],stdout=subprocess.DEVNULL).returncode==0
        inventory.append({'relative_path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':sha256(p),'git_eligible':not ignored})
    atomic_json(DIAG/'stage4fd_artifact_manifest.json',{'stage':'Stage4F-D','status':'PASS','self_hash_excluded':True,'base_commit':BASE_COMMIT,'artifacts':inventory})

if __name__=='__main__':main()
