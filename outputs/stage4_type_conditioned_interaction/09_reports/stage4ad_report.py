"""Render the conservative diagnostic assessment from observed source tables."""
import csv
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '04_evaluation/diagnostics'))
from stage4ad_diagnostic import (read_json, atomic_json, sha256, CHECKPOINT_SHA,
    verify_frozen, ROOT, DIAG)


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def table(rows, keys, labels=None):
    labels = labels or keys
    def value(row, key):
        v = row[key]
        try:
            return f'{float(v):.6f}' if key not in ('Layer', 'Lambda', 'EdgeCount') else str(v)
        except (TypeError, ValueError):
            return str(v)
    return '\n'.join(['| ' + ' | '.join(labels) + ' |', '| ' + ' | '.join(['---']*len(keys)) + ' |'] +
        ['| ' + ' | '.join(value(r, k) for k in keys) + ' |' for r in rows])


def main():
    verify_frozen()
    assert read_json(DIAG / 'stage4ad_inference_completion.json')['status'] == 'PASS'
    assert read_json(DIAG / 'stage4ad_statistics_audit.json')['status'] == 'PASS'
    sources = [ROOT / '06_tables' / name for name in ('stage4ad_attention_scale_summary.csv',
        'stage4ad_pair_scale_summary.csv', 'stage4ad_layer_head_statistics.csv',
        'stage4ad_frozen_context_statistics.csv')]
    sources.append(DIAG / 'stage4ad_bias_scale_sensitivity.csv')
    targets = [r for r in read(sources[0]) if r['Layer'] == 'All']
    layers = [r for r in read(sources[0]) if r['Layer'] != 'All' and r['TargetType'] in ('Vehicle', 'Pedestrian')]
    pairs = read(sources[1])
    context = [r for r in read(sources[3]) if r['Layer'] == 'All' and r['Subset'] != 'All actual targets'
        and r['TargetType'] in ('Vehicle', 'Pedestrian')]
    scales = read(sources[4])
    assert [float(r['Lambda']) for r in scales] == [0, .25, .5, .75, 1]
    v, p = targets[:2]
    ratio_difference = (float(p['BiasBaseRatio'])/float(v['BiasBaseRatio'])-1)*100
    l1_difference = (float(p['AttentionL1Shift'])/float(v['AttentionL1Shift'])-1)*100
    switch_difference = (float(p['TopNeighborSwitchRate'])-float(v['TopNeighborSwitchRate']))*100
    baseline_path = ROOT.parent / 'stage3_multitype_hivt/06_tables/stage3b_main_results.csv'
    # Baseline values are independently reconstructed from its frozen actor CSV.
    from stage4a_evaluate import read_actors, BASE_ACTORS, summarize, select
    baseline_rows = read_actors(BASE_ACTORS)
    baseline = {g: summarize(select(baseline_rows, g)) for g in ('overall', 'vehicle', 'pedestrian')}
    delta_positive = float(scales[1]['Pedestrian_minFDE6'])-float(scales[-1]['Pedestrian_minFDE6'])
    harm = float(scales[-1]['Pedestrian_minFDE6'])-baseline['pedestrian']['minFDE6']
    recovered_fraction = -delta_positive/harm*100
    del baseline_rows
    conclusion = {
        'stage': 'Stage4A-D', 'status': 'PASS', 'DiagnosticConclusion': 'CASE B',
        'DiagnosticConclusionLabel': 'Mixed evidence',
        'RecommendedNextArchitecture': '暂不确定；先分析 representation / class imbalance，不直接实施 bias 限幅架构。',
        'qualitative_assessment': 'Positive intermediate scales show small consistent pedestrian recovery, but lambda=0 reverses it; aggregate relative-scale difference is modest and the frozen interaction contexts do not show higher pedestrian L1 shift.',
        'thresholds_preregistered': False, 'thresholds_invented_posthoc': False,
        'lambda_selected_as_new_model': False, 'new_architecture_implemented': False,
        'Stage4A_scientific_conclusion': 'NOT SUPPORTED', 'Ready_Stage4B': 'NO',
        'checkpoint_sha256': CHECKPOINT_SHA, 'target_statistics': targets,
        'sensitivity': scales, 'source_sha256': {str(s.relative_to(ROOT)): sha256(s) for s in sources}}
    atomic_json(DIAG / 'stage4ad_diagnostic_conclusion.json', conclusion)
    text = f'''# Stage4A-D：Relation Bias Scale Diagnostic

**诊断结论：CASE B — Mixed evidence。** 正的中间缩放值带来小幅、一致的 pedestrian minFDE 恢复，但整体尺度差异有限，冻结交互子群不显示更高的 pedestrian L1 重排，完全移除 bias 则使 pedestrian 更差。这组证据支持 checkpoint 对 bias 强度敏感，尚不足以认定“pedestrian bias 过强”是 Stage4A 可靠退化的原因。

Stage4A 正式结果仍为 **lambda=1；技术 PASS；Type-conditioned Interaction = NOT SUPPORTED；Ready Stage4B = NO**。本轮没有训练、优化器、反向传播、参数更新、新 checkpoint 或模型选择。

## 冻结范围与正式 forward 捕获

诊断从 commit `528b9501d66368b7d6a3b754f64bc1c5006f7d13` 创建分支 `stage4a-d/relation-bias-diagnostic`。新增代码和产物统一在 Stage4 独立根目录内，文件均为 `stage4ad_` 前缀；Stage3 数据及模型文件只读引用。checkpoint SHA256 为 `{CHECKPOINT_SHA}`，全部 Stage4A 正式文件及旧 Stage2C 重绘文件在执行前后哈希一致。

捕获函数通过原始 `message` 的 AST 生成，只截取 `alpha = alpha + relation_bias` 这一句。恢复该句后 AST 与原函数完全一致。原 forward 的 q、k_node、k_edge、edge_attr、relation_bias、edge_index、PyG softmax、value 和聚合均保持原样；加法前 alpha 即本轮 base_logit。lambda=1 保留原始加法，不增加乘法。dropout 的前置 hook 捕获**实际最终 softmax 输出**，另用相同 index/ptr/size 计算 base softmax，仅用于对照。

覆盖 official VAL150、3603 scene windows、所有 3 层 × 8 heads、3,058,206 条实际有向边/window 观测，共 73,396,944 个 edge-layer-head logits。每层有 91,064 个至少有一个实际 incoming edge 的 target/window，包含 supervised targets 和 context-only actor nodes；没有新增邻居或改变 50m graph topology。base、bias、final 的 float32 原始数组和节点身份映射留在本地，SHA256 与 dtype schema 已记录。

attention 熵和 L1 使用完整 incoming neighborhood。按 directed pair context 汇总时，选取“至少有一条该 pair 入边”的 target，仍统计其完整邻域分布；各 context 可重叠，不把单类邻居重新归一化。Heterogeneous-20m 和 VP-context-20m 直接读取原注册 ledger，分别提供 full+partial 和 full-only 汇总，未重新定义成员。

## Target 尺度与 attention 变化

{table(targets, ['TargetType','MeanAbsBase','MeanAbsBias','BiasBaseRatio','EntropyBase','EntropyFinal','DeltaEntropy','AttentionL1Shift','TopNeighborSwitchRate'])}

logit 统计对实际 edge-layer-head 等权，attention 统计对实际 target-layer-head 等权。EdgeCount 表示 edge-layer 观测数，LogitCount 再计入 head 数；NodeLayerCount 和 NodeHeadCount 同理。标准差为总体标准差（ddof=0），median/p10/p90 为全量精确线性分位数，没有抽样。表中的 switch rate 为比例。

Pedestrian 的 pooled bias/base ratio 仅比 Vehicle 高 **{ratio_difference:.2f}%**；L1 shift 高 **{l1_difference:.2f}%**，top-neighbor switch 高 **{switch_difference:.2f} 个百分点**。Pedestrian 的 entropy delta 较小，不能称为更强的整体 attention 集中；负 delta 只表示更集中，不表示更好或更坏。

绝对 logits 和 bias 中对同一 target/head 所有邻居相同的偏移会被 softmax 消去，因此绝对均值及比例是尺度诊断，不能单独建立“过强”或损害的因果解释。

## 层间与关键 directed pairs

{table(layers, ['TargetType','Layer','BiasBaseRatio','DeltaEntropy','AttentionL1Shift','TopNeighborSwitchRate'])}

Pedestrian 相对尺度在 Layer 2/3 较高，但 Layer 1 较低。完整 24 个 layer/head 的 base/bias 均值、标准差、绝对值 median/p90、final 绝对均值及 attention 分布统计见 `stage4ad_layer_head_statistics.csv`。

{table(pairs, ['Pair','MeanAbsBase','MeanAbsBias','BiasBaseRatio'])}

P<-V 和 P<-P 的 ratio 分别接近 1.226 和 1.220，均没有超过 V<-V 的 1.234；V<-P 为 0.887。pooled target 差异同时受 pair 组成及层间分布影响，不足以直接推出 pedestrian 专属 bias 过强。

## 冻结交互 context

{table(context, ['Subset','TargetType','BiasBaseRatio','DeltaEntropy','AttentionL1Shift','TopNeighborSwitchRate'])}

这两个冻结交互 context 中，Pedestrian 的 L1 shift 均不高于 Vehicle，full-only 汇总也如此。更高的总体 pedestrian 重排因此不构成稳定、普遍的 context 内证据。Bicycle 的完整统计仍保留，但其稀疏样本仅作描述。

## 同一 checkpoint 的推理 sensitivity

每个 lambda 均完成同一 official VAL150/3603 windows；使用相同 54,990 个 full-horizon actors。另保存并核对 30,037 个 partial-future actors 的身份、GT SHA256、mask、type、motion 和 displacement，未更改评价成员。模型参数及 buffer SHA256 在每次推理后均一致。

{table(scales, ['Lambda','Overall_minFDE6','Vehicle_minFDE6','Pedestrian_minFDE6','Bicycle_minFDE6'], ['lambda','Overall minFDE (m)','Vehicle minFDE (m)','Pedestrian minFDE (m)','Bicycle minFDE (m)'])}

{table(scales, ['Lambda','vehicle.moving_minFDE6','Vehicle >5m_minFDE6','Pedestrian >5m_minFDE6','Overall_MR6','Overall_Top1FDE6'])}

lambda=0.75→0.5→0.25 时，Pedestrian FDE 相对 lambda=1 小幅、连续下降；lambda=0.25 的变化为 **{delta_positive:+.6f} m**，恢复约 **{recovered_fraction:.1f}%** 的原 pedestrian 退化，但仍高于冻结 Stage3B 的 {baseline['pedestrian']['minFDE6']:.6f} m。lambda=0 的 Pedestrian FDE 为 {float(scales[0]['Pedestrian_minFDE6']):.6f} m，反而比 lambda=1 更差，故整个规定区间并非“越小越好”。Vehicle 在全部规定缩放点仍优于 Stage3B 的 {baseline['vehicle']['minFDE6']:.6f} m，优势没有消失。

lambda=0 保留了**在含 bias 条件下训练的 Stage4A shared weights**，不等于 Stage3B，也不是新 baseline。完整 CSV 同时记录各类/运动/交互组 minFDE、MR 和 Top1FDE；MR 定义为 endpoint error >2m，Top1 是最大 mode probability 的预测。lambda=1 的正式指标复核在原 GPU 非确定性协议容差内通过：minFDE/MR 1e-6，Top1FDE 1e-4；主选择指标容差未放宽。

本轮不为 sensitivity 差异新增显著性声明。原 Stage4A 对 Stage3B 的 scene-bootstrap 结论仍有效：Vehicle 可靠改善，Pedestrian 可靠退化；缩放曲线不覆盖正式结论。

## CASE B 的依据与后续建议

1. 存在尺度、重排及 layer 差异，但总体 ratio 仅略高；关键 P pair ratio 并不超过 V<-V，冻结交互 context 内的 pedestrian L1 shift 也没有更高。
2. 正的三个中间缩放点确有一致的小幅 pedestrian 恢复，不能忽略；完全移除 bias 的结果则反向恶化。这说明强度敏感性和保留关系调制的需求并存，尚不能形成充分的“pedestrian 明显过强、降低强度可稳定修复”的证据链。
3. 用户规则没有为“明显高”预设数值阈值，本报告没有事后构造阈值或显著性门槛；基于上述混合证据作保守的 CASE B 判断，不满足 CASE A 的充分支持，也不符合 CASE C 的完全缺乏恢复证据。

**Recommended next architecture：暂不确定；先分析 representation / class imbalance。** 不直接认定 bias 过强，不据此实现 bounded gate 或任何新模型。本轮未实施 class-balanced loss、oversampling、Reliability、Intent、TTC、map semantics、pruning、dynamic topology、额外种子或超参数搜索。

## 产物与复现

- 诊断入口：`04_evaluation/diagnostics/stage4ad_diagnostic.py`。
- 精确统计：`04_evaluation/diagnostics/stage4ad_summarize.py`；4 份 `06_tables/stage4ad_*.csv`。
- sensitivity：`04_evaluation/diagnostics/stage4ad_bias_scale_sensitivity.csv` 与每个 lambda 的评价审计。
- 六张真实数据图：尺度 grouped bars、pair ratio heatmap、entropy shift、neighbor switch、lambda 主曲线、motion 曲线；全部 PNG/PDF/SVG，附源哈希及数值审计。曲线没有平滑。
- 原始 binary logits、attention 和逐 actor CSV 留在本地，不上传数据、checkpoint 或大型数组。
- 命令及执行审计见 `00_manifest/stage4ad_execution_record.json`；最终输入保护和交付检查见 `00_manifest/stage4ad_final_audit.json`。

**STOP。未训练下一模型，未进入 Stage4B，未 merge main。**
'''
    (ROOT / '09_reports/stage4ad_relation_bias_diagnostic_report.md').write_text(text)
    print('STAGE4AD_REPORT=CASE B', flush=True)


if __name__ == '__main__':
    main()
