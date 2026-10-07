"""Evidence-qualified A/B/C/D interpretation, frozen-input audit and complete delivery."""
from pathlib import Path
import sys,ast,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage7c_common import *

def md_table(rows,columns):
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for row in rows:
        def value(k):
            x=row[k]
            if isinstance(x,(float,np.floating)):return f'{x:.6f}' if np.isfinite(x) else 'undefined'
            return str(x)
        lines.append('| '+' | '.join(value(k) for k in columns)+' |')
    return '\n'.join(lines)

def main():
    capture=read_json(ROOT/'01_attention_capture/stage7c_capture_manifest.json')
    integrity=read_json(ROOT/'00_manifest/stage7c_attention_capture_integrity.json')
    assert capture['status']==integrity['status']=='PASS'
    assert read_json(ROOT/'04_perturbation/stage7c_pairing_audit.json')['status']=='PASS'
    assert read_json(ROOT/'04_perturbation/stage7c_perturbation_run_audit.json')['forward_runs']==2
    att=pd.read_csv(ROOT/'06_tables/stage7c_attention_relevance.csv')
    ci=pd.read_csv(ROOT/'06_tables/stage7c_attention_bootstrap_ci.csv')
    corr=pd.read_csv(ROOT/'06_tables/stage7c_attention_error_correlation.csv')
    assoc=pd.read_csv(ROOT/'06_tables/stage7c_attention_error_scene_bootstrap.csv')
    turn=pd.read_csv(ROOT/'06_tables/stage7c_turn_attention.csv')
    ambiguity=pd.read_csv(ROOT/'06_tables/stage7c_turn_ambiguity.csv',dtype={'TurnOptionCount20':str})
    residual=pd.read_csv(ROOT/'06_tables/stage7c_residual_decomposition.csv')
    perturb=pd.read_csv(ROOT/'06_tables/stage7c_perturbation_results.csv')
    perturb_ci=pd.read_csv(ROOT/'06_tables/stage7c_perturbation_bootstrap_ci.csv')
    representation=pd.read_csv(ROOT/'06_tables/stage7c_representation_effect.csv')
    coverage=pd.read_csv(ROOT/'06_tables/stage7c_lane_coverage.csv')
    actors=pd.read_csv(ROOT/'02_relevance_analysis/stage7c_actor_attention.csv')
    def interval(g,m):return ci[(ci.Group==g)&(ci.Metric==m)].iloc[0].to_dict()
    eligible=[]
    for g in ['Overall Vehicle','vehicle.moving','TurningVehicle_GT','GT-left','GT-right']:
        mass=interval(g,'GTRelevantMass2m')
        association=assoc[(assoc.Group==g)&(assoc.X=='DeltaRelevantMass')&(assoc.Y=='DeltaMinFDE')].iloc[0]
        # User A: relevance loss plus a direction-consistent error association.
        # This is descriptive pattern recognition, not a causal mechanism test.
        if mass['CI_upper']<0 and association.CI_upper<0:eligible.append(g)
    assert eligible==['vehicle.moving','GT-left']
    decision={'FailurePattern':'RELEVANCE_MISALLOCATION','SemanticRoute':'CONTINUE',
        'RecommendSemanticAttentionModel':'YES','interpretation_scope':'localized exploratory support for H1; weak associations; not a cause of all Stage7A degradation',
        'A_supporting_groups':eligible,
        'A_support':'Moving/GT-left paired scene CIs show lower relevant mass and small negative relevant-mass/error association. Turning correct mass and top-connector match also decline.',
        'A_counterevidence':'Primary all-Vehicle rho≈0; turning minFDE slightly improves; GT-right geometric relevance and turning nearest rank improve; opposite-turn mass falls; ambiguity error delta is not monotonic.',
        'B_not_established':'Overall Vehicle and Moving relevance do deteriorate; CENTERED is essentially unchanged and not a rescue. Cross-checkpoint V differences also reflect independently trained shared weights.',
        'C_not_established':'ON≈SHUFFLE suggests weak spatial correspondence benefit, but semantic-specific norm1.8195 and mean ratio0.7747 are substantial. Large r0 alone does not demonstrate generic adapter dominance.',
        'D_not_selected':'Localized direction-consistent moving and GT-left association survives scene resampling; evidence sufficient for a limited score-only diagnostic proposal, not a performance claim.',
        'future_proposal':'semantic_l, relative_lane_actor_position, actor_type_i affect attention score only; geometry-only lane V unchanged',
        'proposal_implemented':False,'training_updates':0,'new_checkpoints':0,'Stage7B_executed':False,
        'historical_Stage7A_decision_unchanged':True,'test_used':False,'STOP':True}
    atomic_json(ROOT/'09_reports/stage7c_scientific_decision.json',decision)
    b=att[(att.Group=='Overall Vehicle')&(att.Model=='Stage3B')].iloc[0]
    o=att[(att.Group=='Overall Vehicle')&(att.Model=='Stage7A')].iloc[0]
    t=turn[turn.Group=='TurningVehicle_GT'].iloc[0]
    rr=residual[residual.SemanticGroup=='all'].iloc[0]
    high=ambiguity[ambiguity.TurnOptionCount20=='>=3'].iloc[0]
    fields={'source_Stage7A_commit':SOURCE_COMMIT,'Attention_capture_integrity':'PASS',
        'attention_population':'Overall Vehicle full-horizon targets','attention_population_count':int(b.Count),
        'Stage3B_GTRelevantMass2m':float(b.GTRelevantMass2m),'Stage7A_GTRelevantMass2m':float(o.GTRelevantMass2m),
        'GTRelevantMass2m_Delta_CI':interval('Overall Vehicle','GTRelevantMass2m'),
        'Stage3B_GTNearestAttentionRank':float(b.GTNearestAttentionRank),'Stage7A_GTNearestAttentionRank':float(o.GTNearestAttentionRank),
        'Stage3B_AttentionEntropy':float(b.AttentionEntropy),'Stage7A_AttentionEntropy':float(o.AttentionEntropy),
        'TurningVehicle_count':int(t.Count),'CorrectTurnMass_Delta_CI':interval('TurningVehicle_GT','CorrectTurnMass'),
        'Stage3B_OppositeTurnMass':float(t.Stage3B_OppositeTurnMass),'Stage7A_OppositeTurnMass':float(t.Stage7A_OppositeTurnMass),
        'Stage3B_TopConnectorMatchRate':float(t.Stage3B_TopConnectorMatchRate),'Stage7A_TopConnectorMatchRate':float(t.Stage7A_TopConnectorMatchRate),
        'TurnOptionCount20_ge3_count':int(high.Count),'HighAmbiguity_Delta_minFDE':float(high.DeltaMinFDE),
        'norm_r0':float(rr.mean_norm_r0),'mean_norm_r':float(rr.mean_norm_r),'mean_norm_delta_r':float(rr.mean_norm_delta_r),
        'perturbation_Overall':perturb[perturb.Group=='Overall'].to_dict('records'),
        'Spearman_DeltaRelevantMass_vs_DeltaMinFDE':assoc[(assoc.X=='DeltaRelevantMass')&(assoc.Y=='DeltaMinFDE')].to_dict('records'),
        **{k:decision[k] for k in ['FailurePattern','SemanticRoute','RecommendSemanticAttentionModel','training_updates','new_checkpoints','Stage7B_executed','STOP']}}
    atomic_json(ROOT/'09_reports/stage7c_final_fields.json',fields)
    paired=[]
    for group in att.Group.unique():
        bb=att[(att.Group==group)&(att.Model=='Stage3B')].iloc[0];oo=att[(att.Group==group)&(att.Model=='Stage7A')].iloc[0]
        paired.append({'Group':group,'Count':int(bb.Count),'B Mass2m':bb.GTRelevantMass2m,'A Mass2m':oo.GTRelevantMass2m,
            'B Rank':bb.GTNearestAttentionRank,'A Rank':oo.GTNearestAttentionRank,'B Entropy':bb.AttentionEntropy,'A Entropy':oo.AttentionEntropy})
    uncertainty=ci[ci.Group.isin(['Overall Vehicle','vehicle.moving','Vehicle >5m','TurningVehicle_GT','GT-left','GT-right'])&
        ci.Metric.isin(['GTRelevantMass2m','CorrectTurnMass','OppositeTurnMass','TopConnectorMatchRate'])].to_dict('records')
    errors=[]
    for group,mask in groups(actors).items():
        if not mask.any():continue
        errors.append({'Group':group,'Count':int(mask.sum()),'B minFDE':float(actors.loc[mask,'Stage3B_minFDE6'].mean()),
            'A minFDE':float(actors.loc[mask,'Stage7A_minFDE6'].mean()),'Delta minFDE':float(actors.loc[mask,'DeltaMinFDE'].mean()),
            'B Top1FDE':float(actors.loc[mask,'Stage3B_Top1FDE6'].mean()),'A Top1FDE':float(actors.loc[mask,'Stage7A_Top1FDE6'].mean())})
    cases=read_json(ROOT/'07_cases/stage7c_case_selection.json')['cases'];assert len(cases)==4
    case_rows=[{'Case':c['case'],'Delta minFDE':c['DeltaMinFDE'],'Delta relevant mass':c['DeltaRelevantMass'],
        'Delta correct mass':c['DeltaCorrectTurnMass'] if c['DeltaCorrectTurnMass'] is not None else 'not turning',
        'Delta opposite mass':c['DeltaOppositeTurnMass'] if c['DeltaOppositeTurnMass'] is not None else 'not turning'} for c in cases]
    sections=[]
    def section(title,text):sections.append('## 【'+title+'】\n\n'+text)
    section('Stage7A Failure Question',f'''本阶段只做冻结模型的诊断，不训练、不 fine-tune、不实现新的 attention 模型。
源提交 `{SOURCE_COMMIT}`，独立分支 `{BRANCH}`。
Stage3B checkpoint SHA256 `{BASE_SHA}`；Stage7A SHA256 `{BEST_SHA}`。
原结论保持：Stage3B Overall minFDE6=1.343260461；Stage7A=1.350545814、Top1FDE6=2.749482077。
Stage7A 的 SemanticMap=NOT_SUPPORTED、PaperUsableSemantic=NO、ReadyStage7B=NO 均未修改。
三个问题是 H1 相关性分配不当、H2 K/V value 改变、H3 generic adapter 主导；本报告只评估证据。''')
    section('Attention Capture Integrity',f'''使用原 ALEncoder 的只读 forward hooks：观察 `attn_drop` 的输入，即 softmax 后 alpha；所有观察 hook 返回 None。
两个模型各检查相同100个不同的随机 official VAL batch（seed2022，batch16）。
raw_prediction、mode_logits、mode_prob 的最大差值均为 **0.0**；state_dict 完全不变，结果 **PASS**。
完整捕获仍采用原评价 kernel policy；相对冻结逐 actor 误差的最大数值差约3.05e-5m，均值差绝对值<4e-9m。
这与受控 deterministic CUDA 的 instrumentation 等价检查不同；两者均记录，未更改历史指标。
捕获3603 windows、150 scenes、{capture['captured_edges']:,}条实际保留的 lane-actor edges。
两模型 edge index/edge attributes 完全一致，8-head 每 actor 权重和最大误差 {capture['edge_alpha_sum_max_error']:.9g}。
Stage3B input 显式移除 lane_semantic，解释性 semantic 标签只在 CPU 离线加入。
所有 edge 字段用无损规范化 NPZ 保存，`stage7c_read_edges.py` 可以逐条恢复完整用户字段；大档案只保留本地。''')
    section('GT Future Lane Relevance',f'''所有54990 full-horizon targets按 scene/sample/instance、node、GT SHA、mask和类型配对。
GT future 是12个未来点组成的连续 polyline，计算它与当前 `[lane_position,lane_position+lane_vector]` 真正 segment 的最小2D距离。
已验证 interior crossing、stationary GT、degenerate segment和point-to-segment计算；不使用 start-point 近似。
GT-nearest primary在全当前 graph 的 lane segments 中 argmin；相同最小距离选最小本地 lane index。
未进入50m incoming edge的 lane权重为0；attention rank采用相同权重的平均rank。
辅助 incoming-only nearest 和最佳 co-nearest rank（距离容差1e-8m）单独报告，避免几何覆盖与选择混淆。
{capture['GT_nearest_outside_incoming_edges']}个target的全图最近 lane不在其incoming edges；{capture['zero_edge_full_targets']}个target没有任何incoming edge。
相关mass对空edge求和为0；entropy/topK/incoming-rank无定义时留空，明确排除相应均值/CI分母。
Vehicle entropy分母41974，Vehicle mass/rank分母42332；Turning1663全部有incoming edge。
2m/4m仅辅助距离定义，GT、转向和relevance从未加入模型输入。
详细覆盖和tie敏感性见 `06_tables/stage7c_lane_coverage.csv`、`stage7c_attention_relevance.csv`。''')
    section('Stage3B vs Stage7A Attention',md_table(paired,['Group','Count','B Mass2m','A Mass2m','B Rank','A Rank','B Entropy','A Entropy'])+
        '\n\nB=Stage3B，A=Stage7A。rank越小越靠前；entropy增加只描述分散程度，不预设优劣。\n\n'+
        md_table(uncertainty,['Group','Metric','Count','Delta','CI_lower','CI_upper'])+
        '\n\n1000 paired scene-cluster bootstrap、seed2022、150 official scenes，actor-window sums/counts pooled，percentile95%CI。重叠次要组不做多重比较调整。全部rank/entropy/turn指标CI见 `stage7c_attention_bootstrap_ci.csv`。')
    section('Turning Vehicle',f'''完全复用 Stage7A membership：vehicle，future endpoint displacement>5m，首尾1秒secant>0.5m，|wrapped heading change|>20°。
Count=1663；GT-left=831，GT-right=832，仅offline。
correct-turn mass下降，**opposite-turn mass也下降**，不是总体转向相反方向。straight mass从0.308012增至0.443093。
24个Turning actor没有incoming connector；主match rate把NO_CONNECTOR记为nonmatch，条件match rate另存。
match是离线解释指标，不能称为预测准确率。
GT-left relevant mass显著下降，GT-right反而改善；Turning全体nearest rank改善。因此不能把所有turning结果归为单一坏attention。

'''+md_table(turn[turn.Group.isin(['TurningVehicle_GT','GT-left','GT-right'])].to_dict('records'),
        ['Group','Count','Stage3B_CorrectTurnMass','Stage7A_CorrectTurnMass','Stage3B_OppositeTurnMass','Stage7A_OppositeTurnMass','Stage3B_TopConnectorMatchRate','Stage7A_TopConnectorMatchRate'])+
        '\n\n'+md_table(errors,['Group','Count','B minFDE','A minFDE','Delta minFDE','B Top1FDE','A Top1FDE'])+
        '\n\nTurning minFDE点估计略改善，Top1仍恶化；correct-turn mass变化与minFDE误差变化几乎无相关。这些是H1证据的重要边界。')
    section('Turn Ambiguity',md_table(ambiguity.to_dict('records'),list(ambiguity.columns))+
        '\n\n计数来自所有区域map token的完整centerline到t0 vehicle的严格<20m距离，每个connector token计一次；TurnOptionCount20是非空left/straight/right类别数。\n>=3组Count15905、ΔminFDE=+0.003594m。minFDE/Top1FDE的退化没有随着选项数单调增大：1-option组的delta更大。不得把高歧义组较大的绝对误差解释成Stage7A特有退化。')
    section('Semantic Residual Decomposition',md_table(residual.to_dict('records'),['SemanticGroup','Count','mean_norm_r','mean_norm_r0','mean_norm_delta_r','mean_delta_ratio'])+
        '\n\n按实际VAL segment/window出现次数加权，共2714494次、50个已观察9D pattern。r(0)常量较大，但mean||delta_r||=1.819476、mean ratio=0.774654，不属于很小的semantic-specific成分。ratio可>1，因为常量与特异向量可以抵消；这些norm比例不是方差解释比例、能量占比或物理效应。\n\n'+
        md_table(representation[representation.Group.isin(['Overall Vehicle','vehicle.moving','TurningVehicle_GT'])].to_dict('records'),
        ['Group','Count','Stage3B_MeanValueNorm','Stage7A_MeanValueNorm','MeanValueDeltaNorm','MeanValueCosine','SemanticEffect','GTRelevantSemanticEffect','AttentionWeightedResidualValueNorm','AttentionWeightedSpecificValueNorm'])+
        '\n\n输入lane semantic residual经Stage7A自身lin_v投影的非零norm可以证明分支确实改变representation。跨Stage3B/Stage7A的V差异还包含独立训练后共享权重变化，不能归因于semantic污染。actor effect是alpha加权delta norm，绝不是因果归因。')
    section('ON / ZERO / SHUFFLE / CENTERED',md_table(perturb.to_dict('records'),['Model','Group','Count','minFDE','Top1FDE'])+
        '\n\nON/ZERO及Stage3B均直接复用冻结actor结果并核对完全相同的身份、GT、mask；本轮只新增SHUFFLE/CENTERED各一次完整VAL推理。\nSHUFFLE每graph用seed2022和scene+sample哈希派生固定permutation，仅置换9D rows；分布、geometry、lane-actor edges和actor数据均保持。它是一次OOD诊断，不是正式baseline或等效性检验。\nCENTERED使用同一checkpoint的evaluation-only输出hook减去r(0)；参数和state_dict不变。\n在此ALEncoder，减去同一r0对同actor所有lane的key score加同一常量，softmax理论上抵消该常量，主要留下value/gate路径扰动；不能由此隔离所有semantic-specific K/V效应。\n\n'+
        md_table(perturb_ci[perturb_ci.Group=='Overall'].to_dict('records'),['Model','Metric','Delta','CI_lower','CI_upper'])+
        '\n\n表中delta为setting−ON。ON比ZERO的candidate minFDE更好，但ZERO的Top1更好。SHUFFLE和CENTERED与ON差值很小且CI跨0；这不证明等效，也不证明semantic无信息。CENTERED没有修复Stage7A相对Stage3B的差距。')
    section('Attention-Error Association',md_table(assoc.to_dict('records'),['Group','X','Y','Count','SpearmanR','CI_lower','CI_upper'])+
        '\n\n所有要求的Spearman及原始descriptive p-value见 `stage7c_attention_error_correlation.csv`；p-value不能替代scene uncertainty。因主相关幅度很弱，补充1000次whole-scene重采样，每次重新计算Spearman ranks。\nOverall Vehicle相关近零。Moving及GT-left的relevant-mass/minFDE相关小幅负向且探索性scene CI仍为负，这提供局部方向一致的H1证据；不能泛化为总体预测退化的因果机制。Turning correct-turn变化与误差基本无相关，GT-right/nearest-rank有相反或不确定模式。')
    section('Case Studies',md_table(case_rows,['Case','Delta minFDE','Delta relevant mass','Delta correct mass','Delta opposite mass'])+
        '\n\n四个不同actor均按固定条件选取，完整数值source在 `07_cases/`；匹配GT/map/axis和共用attention色标。PNG300dpi/PDF/SVG在 `05_figures/`。\nCase1相关mass下降且误差增加；Case2 opposite增加的少数例子不能覆盖全体opposite下降事实；Case3 correct增加且改善；Case4相关mass>0.5并提升，但预测恶化。极端案例不是总体比例证据，也不代替bootstrap。控制/crosswalk用token标记保留multi-hot状态，turn用segment颜色，不覆盖turn颜色。')
    section('Interpretation',f'''**FailurePattern = RELEVANCE_MISALLOCATION**，作为最符合用户CASE A规则的局部、探索性诊断，而非确定的退化原因。
Moving和GT-left同时满足GTRelevantMass2m的paired CI<0及attention/error关联的scene CI<0；correct-turn/connector match的下降补充attention分配改变证据。
必须同时保留反证：整体Vehicle相关为−0.000307，Turning nearest rank改善、GT-right几何相关性改善、opposite mass下降、Turning minFDE不变差、ambiguity退化不单调。局部相关幅度小，不能解释全部总体退化。
CASE B的“relevance未变差”前提不成立，CENTERED也未挽救表现。CASE C的ON≈SHUFFLE前提近似成立，但semantic-specific成分不小，r0大不能单独证明generic dominance。
这些限制并不支持宣称新的semantic方法已有效；只足以推荐一个未来的受控、score-only相关性验证。

**SemanticRoute = CONTINUE**

**RecommendSemanticAttentionModel = YES**

建议仅记录，不实现：
`b_il^sem = MLP(semantic_l, relative_lane_actor_position, actor_type_i)`；
`alpha_il = softmax(q_i k_l / sqrt(d) + b_il^sem)`；geometry-only `V_l`保持不变。
本阶段训练updates=0、新checkpoint=0、新attention model实现=0、Stage7B=未运行、test=未使用、main=未merge。
Stage7A原NOT_SUPPORTED/NO/NO结论保持。CONTINUE只表示建议，后续训练仍需新授权。
**STOP。等待审查，不启动任何新模型训练。**''')
    report='# Stage7C Actor-Conditioned Semantic Lane Relevance Audit\n\n'+'\n\n'.join(sections)+'\n'
    (ROOT/'09_reports/stage7c_actor_conditioned_semantic_audit_report.md').write_text(report)
    required=['stage7c_attention_relevance.csv','stage7c_turn_attention.csv','stage7c_turn_ambiguity.csv',
        'stage7c_residual_decomposition.csv','stage7c_attention_error_correlation.csv','stage7c_perturbation_results.csv']
    assert all((ROOT/'06_tables'/n).is_file() for n in required)
    for c in cases:
        name='stage7c_'+c['case'];assert read_json(ROOT/'05_figures'/f'{name}_audit.json')['status']=='PASS'
        assert all((ROOT/'05_figures'/f'{name}.{ext}').is_file() for ext in ['png','pdf','svg'])
    for p in ROOT.rglob('*.py'):ast.parse(p.read_text())
    frozen=verify_frozen(shards=True)
    for item in capture['archives']:
        p=ROOT/item['path'];assert p.stat().st_size==item['bytes'] and sha256(p)==item['sha256']
    assert sha256(BASE)==BASE_SHA and sha256(BEST)==BEST_SHA
    assert not list(ROOT.rglob('*.pt'))
    atomic_json(ROOT/'00_manifest/stage7c_final_protocol_audit.json',{'status':'PASS','frozen_inputs_SHA_checked':frozen,
        'edge_archives_SHA_checked':len(capture['archives']),'captured_edges':capture['captured_edges'],
        'full_horizon_targets':54990,'VAL_scenes':150,'TurningVehicle_GT_count':1663,
        'integrity_random_batches_per_model':100,'capture_output_maxdiff':0.,'required_tables':required,'case_count':4,
        'training_updates':0,'new_checkpoints':0,'new_attention_model_implemented':False,'Stage7B_executed':False,
        'test_used':False,'historical_Stage7A_decision_unchanged':True,'main_merged':False,'FailurePattern':decision['FailurePattern'],'STOP':True})
    print('FINAL PROTOCOL PASS',frozen,decision['FailurePattern'],flush=True)

if __name__=='__main__':main()
