"""Evidence-grounded mechanism decision and recommendation; no implementation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_manifest'))
from stage4ae_common import (ROOT, DIAG, read_csv, read_json, atomic_json, sha256,
    verify_frozen, CHECKPOINT_B, CHECKPOINT_C, BASE_COMMIT)


def table(rows, keys, labels=None):
    def cell(row, key):
        value = row.get(key)
        if value is None or value == '':
            return '—'
        if key in ('Count', 'UniqueInstances', 'UniqueScenes', 'ValidBatches', 'Batches', 'SupervisedActorWindows', 'ValidFutureCoordinates'):
            return str(value)
        try:
            return f'{float(value):.6f}'
        except (ValueError, TypeError):
            return str(value)
    labels = labels or keys
    return '\n'.join(['| '+' | '.join(labels)+' |', '| '+' | '.join(['---']*len(keys))+' |'] +
        ['| '+' | '.join(cell(r, k) for k in keys)+' |' for r in rows])


def main():
    verify_frozen()
    motion_source = ROOT / '06_tables/stage4ae_pedestrian_motion_bin_metrics.csv'
    ci_source = ROOT / '06_tables/stage4ae_pedestrian_motion_bin_bootstrap.csv'
    context_source = ROOT / '06_tables/stage4ae_pedestrian_motion_context_cross.csv'
    gradient_source = ROOT / '06_tables/stage4ae_gradient_conflict_summary.csv'
    shift_source = ROOT / '06_tables/stage4ae_gradient_shift_bootstrap.csv'
    distribution_source = ROOT / '06_tables/stage4ae_supervision_distribution.csv'
    pair_source = ROOT / '06_tables/stage4ae_relation_embedding_row_summary.csv'
    bins = read_csv(motion_source)
    ci = {r['MotionBin']: r for r in read_csv(ci_source) if r['Metric'] == 'minFDE6'}
    contexts = read_csv(context_source)
    gradients = read_csv(gradient_source)
    shift = read_csv(shift_source)
    distribution = read_csv(distribution_source)
    pair_rows = read_csv(pair_source)
    motion = read_json(DIAG / 'stage4ae_motion_audit.json')
    sample = read_json(DIAG / 'stage4ae_supervision_distribution.json')
    assert motion['status'] == 'PASS'
    assert [int(r['Count']) for r in bins] == [3192, 313, 916, 7321, 260]
    assert all(float(ci[r['MotionBin']]['CI95Lower']) > 0 for r in bins[:3])
    assert float(bins[3]['Delta_minFDE6']) < 0 and motion['moving_ge5_contribution'] < 0
    summary = {(r['Model'], r['ParameterGroup']): r for r in gradients}
    b, c = summary[('Stage3B', 'ALL SHARED')], summary[('Stage4A', 'ALL SHARED')]
    bg, cg = summary[('Stage3B', 'GlobalInteractor')], summary[('Stage4A', 'GlobalInteractor')]
    relation = summary[('Stage4A', 'Relation Module')]
    all_shift = next(r for r in shift if r['ParameterGroup'] == 'ALL SHARED')
    global_shift = next(r for r in shift if r['ParameterGroup'] == 'GlobalInteractor')
    # Keep contradictory primary evidence explicit: this is a qualitative,
    # conservative assessment, not a post-hoc threshold or majority vote.
    assert float(all_shift['DeltaCosMean']) < 0
    assert float(all_shift['DeltaCosCI95Lower']) < 0 < float(all_shift['DeltaCosCI95Upper'])
    assert float(global_shift['DeltaCosMean']) > 0 and float(global_shift['DeltaCosCI95Lower']) > 0
    assert float(cg['ConflictRate']) < float(bg['ConflictRate'])
    architecture = 'Interaction Necessity Gate'
    formula = "h_i' = h_base,i + g_i Δh_rel,i; g_i = sigmoid(MLP(h_i, type_i, m_i^hist, c_i^relative))"
    assessments = [
        {'Evidence': 'ALL SHARED mean/median cosine lower', 'Finding': 'YES',
         'details': f"mean {b['MeanCosVP']}→{c['MeanCosVP']}; median {b['MedianCosVP']}→{c['MedianCosVP']}"},
        {'Evidence': 'ALL SHARED conflict rate higher', 'Finding': 'YES',
         'details': f"{b['ConflictRate']}→{c['ConflictRate']}"},
        {'Evidence': 'GlobalInteractor shared changes in same conflict direction', 'Finding': 'NO',
         'details': f"mean improves by {global_shift['DeltaCosMean']}; CI [{global_shift['DeltaCosCI95Lower']},{global_shift['DeltaCosCI95Upper']}]; conflict decreases"},
        {'Evidence': 'Most paired batch DeltaCos negative', 'Finding': 'WEAK',
         'details': f"ALL SHARED fraction={all_shift['FractionDeltaCosBelow0']}; mean-change CI crosses zero"},
        {'Evidence': 'Vehicle effective shared pressure exceeds Pedestrian', 'Finding': 'YES',
         'details': f"mean-per-batch ratio={c['MeanEffectiveRatio']}; median={c['MedianEffectiveRatio']}; ratio-of-means={c['RatioOfMeanEffectiveNorms']}"}
    ]
    decision = {'stage': 'Stage4A-E', 'status': 'PASS', 'MotionPattern': 'LOW_MOTION_DEGRADATION',
        'low_motion_degradation_evidence': 'YES', 'GradientConflict': 'WEAK',
        'RecommendedArchitecture': architecture, 'DecisionMatrixCase': 3,
        'proposed_formula_only': formula, 'architecture_implemented': False, 'new_model_trained': False,
        'gradient_assessment': assessments, 'posthoc_single_threshold_created': False,
        'ALL_SHARED_cosine_positivity_does_not_exclude_conflicting_batches': True,
        'Stage4A_scientific_conclusion': 'NOT SUPPORTED', 'Stage4AD_conclusion': 'CASE B',
        'motion_bins': bins, 'motion_CI': ci,
        'ALL_SHARED_B': b, 'ALL_SHARED_C': c, 'GlobalInteractor_B': bg, 'GlobalInteractor_C': cg,
        'RelationModule_C': relation, 'supervision_distribution': distribution,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
            (motion_source, ci_source, context_source, gradient_source, shift_source, distribution_source, pair_source)}}
    atomic_json(DIAG / 'stage4ae_mechanism_decision.json', decision)
    motion_rows = [{**r, 'CI95Lower': ci[r['MotionBin']]['CI95Lower'],
        'CI95Upper': ci[r['MotionBin']]['CI95Upper']} for r in bins]
    context_total = []
    for label in ('hetero-20m', 'non-hetero-20m'):
        selected = [r for r in contexts if r['Context'] == label]
        count = sum(int(r['Count']) for r in selected)
        contribution = sum(float(r['ContributionToPedestrianOverallDelta']) for r in selected)
        context_total.append({'Context': label, 'Count': count, 'ActorShare': count/12002,
            'MeanDeltaFDE': contribution*12002/count,
            'ContributionToOverallPedestrianDelta': contribution,
            'FractionOfNetDelta': contribution/motion['overall_delta']})
    write_extra = ROOT / '06_tables/stage4ae_context_degradation_attribution.csv'
    from stage4ae_common import write_csv
    write_csv(write_extra, context_total)
    b_rows = [r for r in gradients if r['Model'] == 'Stage3B']
    c_rows = [r for r in gradients if r['Model'] == 'Stage4A']
    text = f'''# Stage4A-E：Motion-Bin + Class Gradient Conflict Diagnostic

MotionPattern=**LOW_MOTION_DEGRADATION**。GradientConflict=**WEAK**。RecommendedArchitecture=**Interaction Necessity Gate**（决策矩阵 CASE 3，仅建议）。

【Frozen Inputs】

本轮从 `{BASE_COMMIT}` 建立 `stage4a-e/motion-gradient-diagnostic`，新增文件均保存在 Stage4 独立目录并使用 `stage4ae_` 前缀。Stage4A、Stage4A-D、Stage3B checkpoint、误差表、membership 及旧 Stage2C 重绘文件在执行前后保持哈希一致；没有更改旧分支、原始数据或环境，没有 merge main。

Stage3B checkpoint SHA256：`{CHECKPOINT_B}`。
Stage4A checkpoint SHA256：`{CHECKPOINT_C}`。

Part A 直接读取已经通过 pairing audit 的冻结 Stage3B/Stage4A actor errors，严格核对 54,990 个 full-horizon actor-window 的身份、type、GT SHA、future mask、motion state、endpoint displacement；未做任何新 VAL 推理、target 重定义或 membership 计算。12,002 个 full-horizon pedestrian 的固定计数为 **3192+313+916+7321+260=12002**，区间全部采用 `[lower,upper)`。

【Pedestrian Motion-Bin Results】

{table(motion_rows, ['MotionBin','Count','UniqueInstances','UniqueScenes','Stage3B_minFDE6','Stage4A_minFDE6','Delta_minFDE6','CI95Lower','CI95Upper'])}

完整 minADE6、minFDE6、MR6、Top1ADE6、Top1FDE6、NLL 及 C−B 差值见 `stage4ae_pedestrian_motion_bin_metrics.csv`；minFDE、Top1FDE、minADE 的 paired scene-bootstrap CI 见对应 bootstrap CSV/JSON。统一从 official VAL150 场景有放回重采样，1000 replicates，seed2022；每个 replicate 对同一场景中的 B/C 差值配对并保持 actor-window pooling。所有 bin 均有 1000 个有效 replicates。

**Q1：YES。** 0–1m、1–2m、2–5m 的 minFDE 差值都为正，三个 scene-bootstrap 区间都在零以上。低于5m 的加权贡献为 **+{motion['low_motion_lt5_contribution']:.6f} m**，大于净 overall 退化 **+{motion['overall_delta']:.6f} m**，因为 ≥5m 的合计贡献为 **{motion['moving_ge5_contribution']:+.6f} m**、抵消了部分低运动退化。这是 count×delta 的严格加权归因，不是独立 actor 因果归因。

**Q2：5–10m 改善；10–20m 无明确方向。** 5–10m 占 7321 个 actor-windows，FDE {float(bins[3]['Delta_minFDE6']):+.6f} m，CI 上界仅略低于零，改善幅度小。10–20m 的点估计为 +{float(bins[4]['Delta_minFDE6']):.6f} m，但 CI 跨零；260 个 actor-windows 仅来自29个 scene、41个 instance，不能据此声称高位移均改善或可靠退化。1–2m 也只有313个 actor-windows、49个 scene、95个 instance，其正区间作为离线证据保留，避免过强外推。

low-motion degradation evidence=**YES**。本轮主要对 minFDE 作运动模式判断；其他指标不全部同向，例如2–5m minADE 小幅改善、FDE 退化。不得把这一诊断写成所有误差指标均恶化。固定五个 bin 的 CI 未作多重比较校正，结论保持探索性机制诊断范围。

【Motion × Interaction Context】

{table(contexts, ['MotionBin','Context','Count','UniqueScenes','Stage3B_FDE','Stage4A_FDE','DeltaFDE','Top1FDEDelta'])}

{table(context_total, ['Context','Count','ActorShare','MeanDeltaFDE','ContributionToOverallPedestrianDelta','FractionOfNetDelta'])}

**Q3：没有 hetero 专属退化模式。** 在五个固定 bin 的交叉表中，三个低于5m 的 bin 均在 hetero 和 non-hetero context 出现 FDE 正差。hetero 占约81.1% 的 pedestrian 样本，贡献约67.6%的加权净退化；non-hetero 虽只有约18.9%的样本，其人均 delta 更大。5–10m hetero 改善而 non-hetero 小幅变差，也表明 context 与 motion 的关系并非统一。所有交叉 cell 只作描述，不在38个或60个 actor-windows、10个 scene 的 cell 上宣称显著性。

【Stage3B Gradient Conflict】

{table(b_rows, ['ParameterGroup','ValidBatches','MeanCosVP','MedianCosVP','P10CosVP','P25CosVP','P75CosVP','P90CosVP','ConflictRate','StrongConflictRate_0.1','StrongConflictRate_0.25'])}

Stage3B ALL SHARED 平均 cosine={float(b['MeanCosVP']):.6f}、median={float(b['MedianCosVP']):.6f}、ConflictRate={float(b['ConflictRate']):.2%}，说明原模型已经有部分 V/P 方向冲突。cosine 正的总体均值不排除 batch 层面的负向冲突。

【Stage4A Gradient Conflict】

{table(c_rows, ['ParameterGroup','ValidBatches','MeanCosVP','MedianCosVP','P10CosVP','P25CosVP','P75CosVP','P90CosVP','ConflictRate','StrongConflictRate_0.1','StrongConflictRate_0.25'])}

两个 checkpoint 使用完全相同100个符合条件的 TRAIN batches，每批完整16个 scene windows、所有 context actors、同一输入 tensor fingerprint 和 scene/window identity hash。两个模型分别 `eval()`、关闭 dropout，**每模型每批仅 forward 一次**；在同一 output 上调用原 `recovery_loss(original_nll)`，只将 target_mask 设为 V 或 P，再以 `torch.autograd.grad` 提取两类梯度，未使用全局 `no_grad()`。原 Laplace NLL 对该类有效未来坐标取 mean，原 detached soft-target classification 对该类 eligible actor 取 mean，既没有 class sum，也没有删除其他 context actors。

五个 shared 参数组完整覆盖646001个参数，Stage4A GlobalInteractor shared core 明确排除 pair_embedding/relation_mlp；Relation Module 独立1608个参数。所有主要 group 在两模型都有100个非零 V/P gradient batch，None=0、NaN=0、Inf=0。所有参数/buffer state SHA 在每批之后一致；`autograd.grad` 未填充 parameter.grad。没有 optimizer、step 或 backward，未生成新 checkpoint。

【Stage3B vs Stage4A Gradient Shift】

{table(shift, ['ParameterGroup','Stage3BMeanCos','Stage4AMeanCos','DeltaCosMean','DeltaCosMedian','DeltaCosCI95Lower','DeltaCosCI95Upper','FractionDeltaCosBelow0'])}

ALL SHARED mean/median cosine 下降，ConflictRate 14%→33%，强冲突 cos<−0.1 比例10%→25%，cos<−0.25 比例4%→10%。但是 ALL SHARED 同 batch mean DeltaCos 的 CI **[{float(all_shift['DeltaCosCI95Lower']):.6f},{float(all_shift['DeltaCosCI95Upper']):.6f}]** 跨零，只有52%的 batch DeltaCos<0，不能称为稳定、普遍增强。

GlobalInteractor shared mean cosine **{float(bg['MeanCosVP']):.6f}→{float(cg['MeanCosVP']):.6f}**，mean DeltaCos 的优化诊断 CI **[{float(global_shift['DeltaCosCI95Lower']):.6f},{float(global_shift['DeltaCosCI95Upper']):.6f}]** 在零以上，ConflictRate **32%→26%**，方向与“global shared 冲突增强”的假设相反；其 median 略降，分布证据不完全一致。ALL SHARED 的 gradient norm 主要由 Decoder 主导，不能把 ALL SHARED 的变化定位为 GlobalInteractor 独有问题。

1000 paired batch replicates、seed2022、同 batch 索引同时重采样 B/C。这里 batch 不是独立 scene cluster，bootstrap 仅用于优化诊断；不是正式模型显著性结果。

【Dataset Supervision Distribution】

{table(distribution, ['Class','SupervisedActorWindows','ValidFutureCoordinates','PooledActorShare','PooledCoordinateShare','MeanBatchActorShare','MeanBatchCoordinateShare'])}

从 frozen TRAIN700 index 顺序（shuffle=False、workers=0、seed2022）扫描前108个 loader batch，选取前100个同时有 supervised V/P 的 batch；共1600个独特 windows、**{sample['sampled_scene_count']} 个 scene**。未要求 bicycle。valid future coordinates 定义为每个有效未来时间步的两项 x/y scalar coordinates。

实际共同样本的 coordinate shares 为 Vehicle **60.5508%**、Pedestrian **37.2014%**、Bicycle **2.2478%**；没有沿用约73%/25%/1%的先验。pooled share 与 batch share 均值另行报告。这个固定前缀覆盖71/700 scene，不是全 TRAIN700 类别分布的估计，不能推广为整个训练的真实总贡献或历史 optimizer 轨迹。

【Effective Gradient Pressure Proxy】

{table(gradients, ['Model','ParameterGroup','MeanNormV','MeanNormP','MeanNormRatio','MeanEffectiveV','MeanEffectiveP','MeanEffectiveRatio','MedianEffectiveRatio','RatioOfMeanEffectiveNorms'])}

定义每批 w_V=C_V/(C_V+C_P+C_B)、w_P 同理；effective_V=w_V||g_V||，effective_P=w_P||g_P||，effective_ratio=effective_V/(effective_P+1e−12)。class gradient 本身来自类内 mean loss，类样本数量不通过 sum 人为放大 norm。

Stage4A ALL SHARED mean class norms V/P 为 **{float(c['MeanNormV']):.6f}/{float(c['MeanNormP']):.6f}**，mean effective norms 为 **{float(c['MeanEffectiveV']):.6f}/{float(c['MeanEffectiveP']):.6f}**。per-batch ratio 的均值 **{float(c['MeanEffectiveRatio']):.6f}**、median **{float(c['MedianEffectiveRatio']):.6f}**；mean pressures 的 ratio 为 **{float(c['RatioOfMeanEffectiveNorms']):.6f}**。81%的 batch effective_V>effective_P，显示这组诊断样本的 shared pressure proxy 偏向 Vehicle；均值比受低 P norm 的 batch 影响，不能只报告均值比。

**这是 optimization-pressure proxy，不是原始 mixed-loss 梯度的严格代数分解。** 官方回归按 valid coordinate mean，分类按 actor mean，两种权重不同；第一批的拆分 reduction 已与原 mixed regression/classification 数值核对，但把同一 coordinate share 乘上组合 class-loss norm 并不等于精确总梯度贡献，更不能据此证明训练中 Vehicle 因果压制 Pedestrian。

【Relation Module Gradients】

Stage4A Relation Module mean cosine={float(relation['MeanCosVP']):.6f}、median={float(relation['MedianCosVP']):.6f}、ConflictRate={float(relation['ConflictRate']):.2%}。mean V norm={float(relation['MeanNormV']):.6f}、P norm={float(relation['MeanNormP']):.6f}。Relation Module 上 P 的平均 norm 与平均 weighted proxy 反而高于 V；mean proxy ratio={float(relation['MeanEffectiveRatio']):.6f}、median={float(relation['MedianEffectiveRatio']):.6f}、ratio-of-means={float(relation['RatioOfMeanEffectiveNorms']):.6f}，不能把 shared 的 V dominance 推广到 relation 模块。

{table(pair_rows, ['Pair','Batches','ValidBatches','MeanNormV','MeanNormP','MeanCosVP','ConflictRate','ZeroNormVBatches','ZeroNormPBatches'])}

pair_embedding 每行的 V/P 梯度通过完整多层 graph 传播；V loss 不只会影响 V-target pair row。稀有 bicycle pair 的零 row gradient 在表中审计，不视为主要 shared group 缺失；这是训练信号诊断，不是某条交互边或 type pair 的因果解释。Stage3B 没有 Relation Module，没有构造不对应的跨模型比较。

【Mechanism Decision】

MotionPattern=**LOW_MOTION_DEGRADATION**。
GradientConflict=**WEAK**。

{table(assessments, ['Evidence','Finding','details'])}

判定依据是综合证据，不是事后创造一个阈值或按简单多数投票。ALL SHARED 与 pressure 支持部分担忧，但 GlobalInteractor 出现反向变化、ALL SHARED mean shift 的优化区间跨零、DeltaCos 负值仅略多于一半，尚不支持“Stage4A interaction 后共享梯度冲突明显增强”这一完整机制。WEAK 同时保留了冲突率上升的真实证据，没有将其归为完全缺乏支持。

本轮不重新判 Stage4A PASS/FAIL：Stage3B TypeEmbedding=SUPPORTED、Stage4A Type-conditioned additive relation bias=技术PASS/科学NOT SUPPORTED、Stage4A-D=CASE B 均保持原结论。

【Recommended Architecture】

名称=**Interaction Necessity Gate**（给定矩阵 CASE 3）。

核心公式（仅设计建议）：

`h_i' = h_base,i + g_i · Δh_rel,i`

`g_i = sigmoid(MLP(h_i, type_i, m_i^hist, c_i^relative))`

为什么=保持 HiVT interaction 主路径，通过观测历史的 motion magnitude、relative context 与 type 判断是否需要启用 interaction residual。低运动目标可允许 gate 接近0，运动或交互需求较高的目标可允许更高 gate；现有证据支持需求门控方向，但不支持增加复杂类别解耦。`m_i^hist` 必须从已观测历史轨迹/速度构造，`c_i^relative` 只使用预测时可用上下文，**不能把本轮未来 GT displacement bin 当作 inference input**。

本轮未实现 gate、adapter 或新 loss；未训练、重选 checkpoint、调整 class weights、oversampling、Reliability 或 Intent。该建议等待“大脑 AI”审查。

【Limitations】

- gradient conflict 是 **optimization diagnostic，不代表因果证明**；这里只测两个终态 checkpoint 的局部损失几何，未追踪真实训练过程，不等于训练过程中持续的 gradient contribution。
- motion bins 使用 **GT displacement，只用于离线评价，不能作为推理时直接可用的输入**。
- 固定100 qualifying batches 来自71个 TRAIN scene，受 sequential scene/window clustering 和入选条件影响，不代表整个 TRAIN700；未增加随机种子或另选 batch 来改变结论。
- batch bootstrap 不等于 independent scene-cluster statistics，也不产生正式模型显著性结论；motion 交叉 context 保持描述性。
- 1–2m、10–20m 及小 context cell 的 scene/instance 覆盖有限，不能过强泛化；scene-bootstrap 保留窗口重叠依赖，未做多重比较校正。
- 原 GPU scatter 协议非 bitwise deterministic。dropout 已关闭、种子固定、全部身份/输入哈希配对；无额外重复批次、种子或 hyperparameter search。
- 参数完全共享的 ALL SHARED 向量按梯度幅度合成，norm 较大的 Decoder 会主导总体 cosine；Relation Module 另行报告。

复现入口与命令见 `00_manifest/stage4ae_execution_record.json`。所有六张图使用真实 CSV、附源数据哈希，导出 PNG/PDF/SVG；梯度 norm 与 effective proxy 分图，绝无模拟数据。

**STOP。没有实现或训练下一模型，没有 merge main。**
'''
    (ROOT / '09_reports/stage4ae_motion_gradient_diagnostic_report.md').write_text(text)
    print('STAGE4AE_REPORT=LOW_MOTION_DEGRADATION/WEAK/CASE3', flush=True)


if __name__ == '__main__':
    main()
