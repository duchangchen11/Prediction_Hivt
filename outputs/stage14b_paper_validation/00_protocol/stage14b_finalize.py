"""Finalize six Stage14B deliverables from frozen, audited artifacts only."""
from stage14b_common import *

def main():
    frozen_history=verify(history=True,data=True)
    historical_counts={k:len(frozen_history[k]) for k in ('historical_files','preserved_untracked','checkpoints')}
    for p in ['01_preflight/stage14b_capacity_preflight.json','01_preflight/stage14b_tiny_audit.json','05_evaluation/stage14b_identity_audit.json','06_statistics/stage14b_capacity_evidence.json','05_evaluation/stage14b_efficiency_audit.json','05_evaluation/stage14b_state_supplement.json']:
        assert read_json(ROOT/p)['Status']=='PASS',p
    review=read_json(ROOT/'00_protocol/stage14b_design_review.json')
    assert review['Status']=='PASS_DESIGN_ONLY_WITH_DOCUMENTED_LIMITS'
    assert review['BlockingDesignErrors']==[] and review['ThisReviewAuthorizesFutureTrainingOrOfficialEvaluation'] is False
    for p,h in review['ReviewedSourceSHA256'].items():assert sha256(PROJECT/p)==h,p
    gate=read_json(ROOT/'04_checkpoints/stage14b_all_frozen.json')
    assert gate['Status']=='FROZEN_ALL_COMPLETE' and gate['Folds']==3 and len(gate['Checkpoints'])==3
    expected_paths={str(cp_path(fold,'C').relative_to(PROJECT)) for fold in (1,2,3)}
    assert {row['Path'] for row in gate['Checkpoints']}==expected_paths
    assert {row['Fold'] for row in gate['Checkpoints']}=={1,2,3}
    assert all(row['Model']=='Matched-NG-C' for row in gate['Checkpoints'])
    assert gate['OuterTestEvaluationPermitted'] is True and gate['FurtherTrainingPermitted'] is False
    assert gate['NewFormalTrainingRuns']==3 and gate['NoOuterTestModelSelection'] is True
    assert all(gate[k] is False for k in ('TinyWeightsUsed','HeadDevUsed','OfficialVALTestUsed','HistoricalGACRetrained'))
    checkpoint_rows={row['Fold']:row for row in gate['Checkpoints']}
    for row in gate['Checkpoints']:
        assert row['Path']==str(cp_path(row['Fold'],'C').relative_to(PROJECT))
        assert sha256(PROJECT/row['Path'])==row['SHA256']
    assert sha256(ROOT/'stage14b_preregistered_plan.md')==read_json(ROOT/'00_protocol/stage14b_plan_registration.json')['PlanSHA256']
    tiny=read_json(ROOT/'01_preflight/stage14b_tiny_audit.json');evidence=read_json(ROOT/'06_statistics/stage14b_capacity_evidence.json');ci=evidence['RegisteredCapacityComparison']
    training=pd.read_csv(ROOT/'03_training/stage14b_training_summary.csv',float_precision='round_trip');assert len(training)==3
    assert set(training.Fold)=={1,2,3} and training.Fold.is_unique
    assert training.Status.eq('COMPLETE').all() and training.Model.eq('C').all()
    assert training.PaperModel.eq('Matched-NG-C').all() and training.Params.eq(24001).all() and training.TrainableParams.eq(24001).all()
    assert training.OuterTestUsed.eq(False).all() and training.TinyWeightsUsed.eq(False).all()
    assert training.FrozenR2Reused.eq(True).all() and training.PredictorGradientCount.eq(0).all()
    assert np.isfinite(training.Seconds).all() and training.Seconds.gt(0).all()
    training=training.sort_values('Fold').reset_index(drop=True)
    fitting_sources=('00_protocol/stage14b_common.py','01_preflight/stage14b_capacity_preflight.py',
                     '02_models/stage14b_matched_nograph.py','03_training/stage14b_train.py')
    verified_fitting_sources={}
    for fold in (1,2,3):
        config_path=ROOT/f'03_training/fold{fold}/C/stage14b_training_config.json'
        c=read_json(config_path);row=checkpoint_rows[fold];summary=training.loc[training.Fold.eq(fold)].iloc[0]
        assert c['Fold']==fold and c['Model']=='C' and c['PaperModel']=='Matched-NG-C'
        assert c['protocol_sha256']==sha256(PROTOCOL)
        assert c['OriginalTrainingSourceSHA256']==sha256(S11B/'03_training/stage11b_train.py')
        assert sha256(config_path)==row['TrainingConfigSHA256']==summary.config_sha256
        assert row['SHA256']==summary.checkpoint_sha256 and row['SelectedEpoch']==summary.SelectedEpoch
        assert row['CheckpointScore']==summary.CheckpointScore
        for key,filename in [('normalization','normalization'),('split','split')]:
            h=sha256(S11B/f'02_splits/stage11b_fold{fold}_{filename}.json')
            assert c[key+'_sha256']==h==row[key.title()+'SHA256']==summary[key+'_sha256']
        # Only these four new sources enter fitting. Other recorded metadata and
        # official-protocol utilities may be finalized while the GPU runs; their
        # training-time snapshots are not claimed to be fitting-source checks.
        assert set(fitting_sources)<=set(c['new_sources_sha256'])
        verified_fitting_sources[fold]={p:c['new_sources_sha256'][p] for p in fitting_sources}
        for p,h in verified_fitting_sources[fold].items():assert sha256(ROOT/p)==h,p
    metrics=pd.read_csv(ROOT/'05_evaluation/stage14b_capacity_metrics.csv');eff=pd.read_csv(ROOT/'05_evaluation/stage14b_efficiency.csv')
    cost=read_json(ROOT/'00_protocol/stage14b_scene_design_cost_audit.json');actual=float(training.Seconds.sum()/3600)
    supplement=read_json(ROOT/'05_evaluation/stage14b_state_supplement.json')
    assert supplement['Count']==4315 and supplement['ModelInferenceRuns']==0 and supplement['OptimizerUpdates']==0
    for source,h in supplement['InputSourceSHA256'].items():assert sha256(PROJECT/source)==h
    assert sha256(ROOT/'05_evaluation/stage14b_state_supplement.csv')==supplement['OutputMetricsCSVSHA256']
    modelmeans=eff.groupby('Model').MeanMSPerActor.mean().to_dict()
    updatedcost=[]
    for x in cost['Schemes']:
        updatedcost.append(dict(Scheme=x['Scheme'],HiVTRuns=x['HiVTRuns'],RankingRuns=18,Historical15HeadHours=x['FifteenRankingRunsMeasuredHistoricalAnchorHours'],MeasuredMatchedThreeHeadHours=actual,
            PlanningObservedScheduleAnchorHours=x['KnownComponentLowerHistoricalScheduleAnchorHours']+actual,
            PlanningFullPredictorBudgetAnchorHours=x['KnownComponentUpperHistoricalBudgetAnchorHours']+actual,
            Limits='historical/frozen-candidate observed wall times, not guaranteed future costs; excludes graph cache I/O/construction/diagnostics'))
    atomic_json(ROOT/'00_protocol/stage14b_updated_resource_estimate.json',dict(Status='MEASURED_MATCHED_COST_PLUS_HISTORICAL_PLANNING_ANCHORS',Schemes=updatedcost,MeasuredMatchedGPUWallHours=actual,MeasuredMatchedTimeScope='sum of formal head training wall times including InnerDev evaluation, I/O and saves; single-device reservation anchor, not pure active CUDA time',FormalHiVTJobsExecuted=0))
    cap='''# Stage14B Capacity-Matched NoGraph 对照

固定一次结构：原自身15→64→64节点编码、LayerNorm64、64→32→1评分头，加自身64→128→64 ReLU residual adapter。实际forward读取 `local_nodes[:,0]` 和原logits，真实邻居、interaction edges、map输入不参与计算；无空参数或借用G-C权重。源码见 [模型](02_models/stage14b_matched_nograph.py)、[结构检查](01_preflight/stage14b_model_audit.py)、[真实输入及梯度检查](01_preflight/stage14b_capacity_preflight.py)。

| 模型 | 实际可训练参数 | 相对G-C差额 |
| --- | ---: | ---: |
| NG-C | 7425 | 16641 |
| G-C | 24066 | 0 |
| Matched-NG-C | 24001 | 65，0.2701%% |

新增16576参数均在真实目标路径参与计算，四个adapter参数张量有实质非零C梯度。共享最终标量bias对softmax共同平移不识别，G-C也含这个相同遗留维度；不为凑数加入额外无效参数。所有输入保持原字节、step0与原logits一致；非零评分探针邻居/边/地图改变差0，目标改变有响应；重新构造10个原始InnerTrain window并污染未来GT/mask后forward差0。证据见 [preflight](01_preflight/stage14b_capacity_preflight.json)、[model integrity](01_preflight/stage14b_model_integrity.json)、[原始GT重放](01_preflight/stage14b_gt_poison_windows.csv)。

结构、参数、输入输出、概率归一、共享初始化、有限梯度、未来GT隔离全部PASS。原Fold1 128个target、seed2022、300次更新tiny C降幅 %.6f%%（门槛80%%），未保存tiny checkpoint，不复用tiny权重。见 [tiny审计](01_preflight/stage14b_tiny_audit.json)。

全部检查通过后按授权执行3个小头训练；没有训练HiVT、G-C或历史模型。与原NG-C/G-C共享候选、折分、仅InnerTrain标准化、归一化C、AdamW、FP32、128×8、carry、每epoch样本顺序与严格InnerDev选择规则，原协议不变。完整原AST训练体只改变新artifact前缀、实际参数断言，并添加历史order hash核验。每折仍有Bicycle监督目标，最终Bicycle固定R2路由。见 [runner](03_training/stage14b_train.py)、[训练汇总](03_training/stage14b_training_summary.csv)。

三fold选中epoch=%s，执行epoch=%s，实测三头wall=%.6f小时。全部三checkpoint先冻结再统一读OuterTest，SHA清单见 [冻结门](04_checkpoints/stage14b_all_frozen.json)。本阶段不是端到端独立验证；历史候选生成器见过全部TRAIN700，不能通过新头消除此事实。

## 同身份统一评价

'''%(100*tiny['Result']['Reduction'],training.SelectedEpoch.tolist(),training.ExecutedEpochs.tolist(),actual)
    selected=metrics[metrics.Group.isin(['Overall','Vehicle','Pedestrian','MovingVehicle'])]
    cap+='| Group | Model | Count | minFDE6 | Top1ADE | Top1FDE | HitRate |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
    for x in selected.to_dict('records'):cap+=f"| {x['Group']} | {x['Model']} | {x['Count']} | {x['minFDE6']:.6f} | {x['Top1ADE']:.6f} | {x['Top1FDE']:.6f} | {x['HitRate']:.6f} |\n"
    cap+='\n### 预登记的车辆状态透明补充\n\nOtherVehicleState采用Vehicle扣除互斥的Moving/Stopped/Parked，4315目标、207scene，原t0 motion_state均unknown。既有main groups没有这项，因此另从冻结输出离线汇总；不改分组函数、主表、主要比较或任何训练/forward。完整状态表如下。\n\n| Group | Model | Count | minFDE6 | Top1ADE | Top1FDE | HitRate |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |\n'
    state_rows=list(supplement['ModelMetrics'])
    for group in supplement['ExistingStoppedParkedTransparency']:
        state_rows.extend(dict(Group=group['Group'],Count=group['Count'],**row) for row in group['ModelMetrics'])
    for x in state_rows:cap+=f"| {x['Group']} | {x['Model']} | {x['Count']} | {x['minFDE6']:.6f} | {x['Top1ADE']:.6f} | {x['Top1FDE']:.6f} | {x['HitRate']:.6f} |\n"
    cap+='\nOther G-C−Matched Top1FDE=-0.155732m，探索性95%CI[-0.252722,-0.055661]。Parked G-C−Matched点差+0.001480m，95%CI[-0.001002,+0.004114]；G-C−NG-C点差+0.002050m，95%CI[-0.000186,+0.004484]。Parked有点退化，区间跨0，不能写成显著损害或全组均改善。Stopped G-C−Matched点差-0.015444m，探索性95%CI[-0.032228,-0.000610]。补充仅用原配对2000/seed2022权重，均探索性，不升格主要推断。[状态补充CSV](05_evaluation/stage14b_state_supplement.csv)、[来源/CI审计](05_evaluation/stage14b_state_supplement.json)。\n'
    cap+=f"\n新主要容量比较 G-C−Matched-NG-C Overall Top1FDE={ci['DeltaTop1FDE']:.6f} m；描述95%CI=[{ci['CI95Lower']:.6f}, {ci['CI95Upper']:.6f}]，family 3 调整98.3333%CI=[{ci['BonferroniCILower']:.6f}, {ci['BonferroniCIUpper']:.6f}]，负向fold={ci['NegativeFolds']}/3。`CapacityAlternativeNotSufficient={evidence['CapacityAlternativeNotSufficient']}`。见 [原配对bootstrap](06_statistics/stage14b_bootstrap_ci.csv)、[判定](06_statistics/stage14b_capacity_evidence.json)。两个历史主要contrast重新计算统一的 family 3 CI 并明确标为已知；仅容量contrast是新增比较，不替换Stage14A family 4 原判定。\n"
    if evidence['CapacityAlternativeNotSufficient']=='SUPPORTED':cap+='\n在这套历史固定候选与训练规则下，目标自身MLP增加近似相同参数量不足以复现G-C表现。该结果减轻“仅多参数即可解释”这一替代解释，仍不证明消息的因果效果，也不消除深度、归纳偏置和优化差异。\n'
    else:cap+='\n登记条件不足以排除容量替代解释；不能继续宣称图收益已独立于参数量成立。报告匹配模型、点估计、完整CI与fold方向，不改结构/seed/预算以寻求正面结果。\n'
    cap+='\nHitRate与误差幅度并非同一指标：Overall G-C HitRate=0.420121低于Matched-NG-C的0.421390，Vehicle亦为0.451991低于0.456796，尽管对应Top1FDE更低。不能将主要Top1FDE改善概括为所有指标全面改善；完整表保留此差异。\n'
    cap+='\n| Model | cached CUDA head mean ms/actor |\n| --- | ---: |\n'
    for name in MODELS:cap+=f'| {name} | {modelmeans[name]:.6f} |\n'
    cap+='\n同一RTX3080/FP32、128batch、固定1024target池、20warm-up与20轮转重复/三fold，只包含预加载设备输入的头forward，排除HiVT、特征构造、传输、路由；不是整体FPS。近似参数匹配不等于FLOPs/延迟匹配。见 [效率CSV](05_evaluation/stage14b_efficiency.csv)、[scope审计](05_evaluation/stage14b_efficiency_audit.json)。候选/oracle几何及Bicycle输出全部逐bit一致；完整身份/GT分离检查见 [评价审计](05_evaluation/stage14b_identity_audit.json)。\n'
    (ROOT/'stage14b_capacity_control.md').write_text(cap)
    a,b=updatedcost
    report=f'''# Stage14B 最终报告

Stage14B完成方法审计、700场景来源/隔离设计、基线/官方协议审计和预登记。容量模型全部工程与tiny检查PASS后，执行了获授权的3个冻结候选小头对照并冻结；HiVT重训次数=0，未启动Stage15。历史{historical_counts['historical_files']}个tracked文件、{historical_counts['preserved_untracked']}个未提交Stage2C文件、{historical_counts['checkpoints']}个旧checkpoint及冻结输入SHA全部保持。

## 审查问题的直接回答

1. **两个方法在代码中真实存在：YES。** G1具有候选节点与候选对attention消息、残差模式评分；C实际为归一化expected regret。生成器仍含类型embedding和运动条件decoder，R2含概率加权邻居摘要，必须按实际组件写论文。源码位置见 [method audit](stage14b_method_audit.md)。存在实现不等于已经验证文献新颖性或独立泛化。
2. **参数量增加不足以解释G-C优势：{evidence['CapacityAlternativeNotSufficient']}（近似容量对照范围内）。** NG-C7425/G-C24066原本有容量混杂。Matched-NG-C24001仅用目标自身特征，G-C−Matched Overall Top1FDE={ci['DeltaTop1FDE']:.6f}m、family3 CI=[{ci['BonferroniCILower']:.6f},{ci['BonferroniCIUpper']:.6f}]、负向fold={ci['NegativeFolds']}/3；按预登记规则解释，不将非显著写为等价，也不声称因果。完整表见 [capacity control](stage14b_capacity_control.md)。
3. **预测器与排序器场景隔离：** 每fold随机初始化Stage5A结构，训练378、开发42，完全排除对应Outer210，隔离历史HeadDev70；各方法用同fold共同候选，重提特征/仅InnerTrain标准化，全部头重新训练并先冻结再Outer评价。旧全700预测器及旧头不能冒充隔离模型。旧入口硬编码700/VAL150，未来须新目录中的fold/seed入口与逐batch来源检查，不改旧文件。
4. **推荐基础A。** 它解决当前最关键的预测器训练重叠且成本较低。B额外审计训练候选OOF，但存在252/378模型输出分布差异、重复Dev选择与更复杂来源证明；不预设其更优，不根据Outer结果切换。详见 [evaluation protocol](stage14b_evaluation_protocol.md)。
5. **需要重训多少次HiVT：** A=3；B=12（每Outer3个内层+1最终）。两方案完整比较均另需18个小头训练（NG-A/NG-C/G-A/G-C/Matched-NG-C/R2每fold）。共同生成器原分数R0无需头训练；真正OriginalHiVT生成器baseline需额外3次预测器，未包含A/B数字。[命名/seed澄清](00_protocol/stage14b_design_clarifications.md)
6. **当前数据不满足全新独立确认要求。** 700TRAIN全部被旧预测器训练，630池与70开发池有研究复用；150VAL用于历史选择和开发。本地注册850没有已核实pristine确认池。新隔离CV可检验拟合隔离条件下的内部端到端性能，不能把旧开发scene改称新独立数据；hiddenTEST本地目标/标签可用性未证实。
7. **现有基线支持内部机制消融，尚不足支持公平官方/SOTA或独立端到端优越性。** R0、R2、NG/G×A/C及matched有用；当前“original mode scores”属于增强Stage5A，不能冒充原HiVT网络。若主张与外部预测模型公平竞争，须统一split/target/horizon/K/map/metric并重新运行合适基线，禁止直接混入不同协议论文数字。[baseline audit](stage14b_baseline_audit.md)
8. **正式资源预估：** RTX3080单设备历史墙钟锚点，加入本次三matched头实测 {actual:.3f} h，A约 {a['PlanningObservedScheduleAnchorHours']:.2f}–{a['PlanningFullPredictorBudgetAnchorHours']:.2f} h，B约 {b['PlanningObservedScheduleAnchorHours']:.2f}–{b['PlanningFullPredictorBudgetAnchorHours']:.2f} h；不是CI或承诺，另需缓存I/O、图构造、诊断和缓冲，也不含真正OriginalHiVT额外3次。当前设备显存实测 10240 MiB（10 GiB），同一设备已完成历史训练；Stage5A历史 1735.690918 MiB 峰值仅是forward测量，不能代表训练峰值或保证未来训练适配，未来fold入口仍须验证训练显存。本阶段小头的CUDA显存也不能替代完整预测器训练测量。见 [历史效率表（报告第175行）](../stage5a_motion_aware_decoder/09_reports/stage5a_final_report.md)。现有shard约 3.784 GiB 只引用；三fold合并目标缓存说明性估计 17.067 GiB，额外上下文/临时/optimizer空间待实施核验。历史代理及更新实测分别见 [设计成本](00_protocol/stage14b_scene_design_cost_audit.json)、[更新预算](00_protocol/stage14b_updated_resource_estimate.json)。
9. **下一阶段正式启动条件：NOT_YET。** 方法/低成本控制和设计可审查；仍需大脑AI审查与下一阶段授权、独立于旧入口的fold训练接口及GT/来源小检查、资源确认。若要求官方或pristine确认，还缺完整官方targets、global导出/官方指标adapter及未开发确认数据/流程。本阶段结束并STOP，不实施这些后续工作。

## 容量对照与冻结事实

新matched三fold选中epoch={training.SelectedEpoch.tolist()}，执行epoch={training.ExecutedEpochs.tolist()}；仅InnerDev选择。tiny降幅{100*tiny['Result']['Reduction']:.2f}%，新增adapter梯度有效，未来GT/raw-mask污染forward差0，邻居扰动差0。全部3checkpoint冻结后评价260151个唯一scene/sample/instance target，630scene；共同oracle几何和Bicycle R2逐bit一致。统计保持2000配对整scenebootstrap、seed2022、family3；历史两contrast已知，新增capacity结果未用于结构/预算/选checkpoint。原Stage14A判定不改。补充如实报告Parked的点退化、OtherVehicleState4315目标，以及HitRate与Top1FDE的差异；不概括为全部分组/指标均改善。

## 官方协议结论

`OfficialProtocolCompatibility=PARTIAL_NEEDS_ADAPTER_AND_PROTOCOL_RESET`。本机devkit确认challenge train500/train_val200/val150，但完整prediction_scenes.json未找到。6秒2Hz12点、2秒历史可对接；还需globalXY、指定token、概率top1/5、官方全时域Miss≥2m和OffRoadRate。项目endpointMR6>2m及自定义full-target池不可替代官方定义。VAL leaderboard与hiddenTEST后续流程不意味着旧VAL独立。依据：[官方prediction README](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md)、[split](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py)、[metrics](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/metrics.py)、[本机核验](00_protocol/stage14b_official_protocol_audit.json)。

## 交付与停止

六文件：[method audit](stage14b_method_audit.md)、[evaluation protocol](stage14b_evaluation_protocol.md)、[capacity control](stage14b_capacity_control.md)、[baseline audit](stage14b_baseline_audit.md)、[preregistered plan](stage14b_preregistered_plan.md)、本报告。原计划SHA保留，方法命名/未来seed澄清单独存档，参数数量与实际有效路径分开审计。

GitBranch=`stage14b/paper-validation-design`。基准commit=`{BASE}`，预登记本地commit=`461e446`、远端commit=`00998ed1fc12a0273f293781e2837bf37aefbef0`；最终commit由该分支HEAD核验（避免报告自嵌最终SHA循环）。所有新增内容位于 `outputs/stage14b_paper_validation/`，大型候选/模型/日志留本机，不上传原始数据、环境或凭据；不merge main。

Recommendation=`A_FIRST_AFTER_REVIEW_AND_ISOLATION_PREFLIGHT`。当前没有训练HiVT、覆盖历史checkpoint或启动Stage15。提交和push后STOP，等待大脑AI审查。
'''
    (ROOT/'stage14b_final_report.md').write_text(report)
    required=[ROOT/f'stage14b_{n}.md' for n in ['method_audit','evaluation_protocol','capacity_control','baseline_audit','preregistered_plan','final_report']]
    assert all(p.exists() for p in required)
    atomic_json(ROOT/'00_protocol/stage14b_final_audit.json',dict(Status='PASS',MethodAudit='PASS',CapacityControl='PASS',CapacityAlternativeNotSufficient=evidence['CapacityAlternativeNotSufficient'],EvaluationProtocol='DESIGN_PASS_A_RECOMMENDED',FormalHiVTRunsExecuted=0,MatchedHeadRunsExecuted=3,EstimatedHiVTRuns=dict(A=3,B=12),FutureRankingRuns=18,PristineDataVerified=False,OfficialProtocolCompatibility='PARTIAL_NEEDS_ADAPTER_AND_PROTOCOL_RESET',HistoricalFilesUnchanged=True,OldCheckpointCount=historical_counts['checkpoints'],HistoricalTrackedFileCount=historical_counts['historical_files'],PreservedUntrackedCount=historical_counts['preserved_untracked'],ExactThreeCheckpointPathsVerified=True,CheckpointSHA256={row['Path']:row['SHA256'] for row in gate['Checkpoints']},VerifiedFittingSourceSHA256=verified_fitting_sources,FittingSourceVerificationScope='four new imported fitting sources plus byte-frozen historical sources; not every metadata utility snapshot in training config',DesignReviewStatus=review['Status'],DesignReviewSourcesVerified=True,RequiredReportSHA256={p.name:sha256(p) for p in required},STOP=True,Stage15Started=False))
    print('STAGE14B_FINALIZE_PASS',evidence['CapacityAlternativeNotSufficient'],flush=True)
if __name__=='__main__':main()
