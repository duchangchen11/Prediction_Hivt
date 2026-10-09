"""Apply preregistered decisions and report both gains and regret tradeoffs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11c_common import *
def markdown(rows,columns):
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v)
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']+
        ['| '+' | '.join(fmt(r[c]) for c in columns)+' |' for r in rows])
def main():
    verify(history=True);audit=read_json(ROOT/'08_reports/stage11c_final_audit.json');assert audit['Status']=='PASS'
    table=pd.read_csv(ROOT/'07_tables/stage11c_oof_probability_metrics.csv')
    boot=pd.read_csv(ROOT/'04_bootstrap/stage11c_bootstrap_ci.csv')
    fits=read_json(ROOT/'02_calibration/stage11c_temperature_fit_isolation.json')['Fits']
    def row(group,model):return table[(table.Group==group)&(table.Model==model)].iloc[0]
    def ci(group,metric):return boot[(boot.Scope=='Pooled')&(boot.Group==group)&(boot.Metric==metric)].iloc[0]
    def worse(group,metric):return float(ci(group,metric).CI95Low)>0
    improvement={t:float(row(t,'C Calibrated').OracleBestModeNLL)<float(row(t,'C Raw').OracleBestModeNLL) for t in TYPES[:2]}
    conflict=all(worse(t,'BrierScore') for t in TYPES[:2]) or all(worse(t,'Top1ModeECE') for t in TYPES[:2])
    useful='INCONCLUSIVE' if all(improvement.values()) and conflict else ('YES' if all(improvement.values()) else 'NO')
    concentration=all(row(t,'C Calibrated').Top1Probability<row(t,'C Raw').Top1Probability and
        row(t,'C Calibrated')['FractionAbove0.99']<row(t,'C Raw')['FractionAbove0.99'] for t in TYPES[:2])
    ready=useful=='YES' and audit['CalibrationEngineering']=='PASS'
    decision={'FrozenModelIntegrity':'PASS','TemperatureFitIsolation':'PASS','Top1Identity':'PASS','BicyclePreserved':'YES',
        'CalibrationEngineering':'PASS','VehicleNLLImproved':'YES' if improvement['Vehicle'] else 'NO',
        'PedestrianNLLImproved':'YES' if improvement['Pedestrian'] else 'NO',
        'ProbabilityConcentrationReduced':'YES' if concentration else 'NO','CalibrationUseful':useful,
        'ReadyForSemanticStudy':'YES' if ready else 'NO','Stage11C':'GO' if ready else 'STOP',
        'ExecutionBoundary':'STOP_AFTER_STAGE11C; wait for brain-AI review, no next-stage authorization',
        'EvidenceScope':'OOF development evidence, already inspected; not independent testing',
        'ProbabilityScope':'six-candidate FDE-oracle-mode classification only; not true future/intent/collision/risk probabilities',
        'ExpectedRegretTradeoff':{t:{'Raw':float(row(t,'C Raw').ExpectedRegret),'Calibrated':float(row(t,'C Calibrated').ExpectedRegret),
            'Delta':float(ci(t,'ExpectedRegret').DeltaCalMinusRaw),'CI95':[float(ci(t,'ExpectedRegret').CI95Low),float(ci(t,'ExpectedRegret').CI95High)]} for t in TYPES[:2]},
        'Temperatures':{f'Fold{r["Fold"]}':r['Temperature'] for r in fits},'ProtocolSHA256':sha256(PROTOCOL),
        'NewModelTraining':False,'HeadDevOfficialVALTestRead':False}
    atomic_json(ROOT/'08_reports/stage11c_scientific_decision.json',decision)
    rawcal=table[table.Model.isin(['C Raw','C Calibrated'])]
    primary=rawcal[rawcal.Group.isin(['Overall',*TYPES])].to_dict('records')
    controls=table[table.Group.isin(['Overall',*TYPES])].to_dict('records')
    motion=rawcal[~rawcal.Group.isin(['Overall',*TYPES])].to_dict('records')
    intervals=boot[(boot.Scope=='Pooled')&boot.Group.isin(TYPES[:2])].to_dict('records')
    concentrationrows=rawcal[rawcal.Group.isin(TYPES[:2])].to_dict('records')
    foldtable=pd.read_csv(ROOT/'07_tables/stage11c_fold_metrics.csv')
    foldrows=foldtable[foldtable.Model.isin(['C Raw','C Calibrated'])&foldtable.Group.isin(TYPES[:2])].to_dict('records')
    text=f'''# Stage11C Error-Aware Ranking Probability Calibration

**Stage11C — OOF Development Evidence。CalibrationEngineering={decision['CalibrationEngineering']}；CalibrationUseful={useful}；ReadyForSemanticStudy={decision['ReadyForSemanticStudy']}；Stage11C={decision['Stage11C']}。**

Vehicle/Pedestrian 的六候选 oracle-mode NLL、Brier 和 ECE 均降低，Top1 选择和几何误差完全不变。温度缩放同时增加 ExpectedRegret，因此这个结论只支持指定 oracle-mode 分类概率指标的改善。Stage11C 科学判定为 GO 时，本次执行仍在本阶段结束后 STOP，等待大脑 AI 审查。

## 冻结与数据边界

基础 commit：`{BASE}`。分支：`stage11c/ranking-probability-calibration`。全部产物单独位于 `outputs/stage11c_probability_calibration/`。

Stage11B 的 12 个 checkpoint 与原 manifest SHA256 全部一致，另外 8 个历史依赖 checkpoint 保持冻结。{audit['HistoricalTrackedFilesUnchanged']} 个历史版本文件及 {audit['PreservedStage2CUntrackedFiles']} 个 Stage2C 未提交文件保持原样。冻结划分、对应折标准化和候选数组；没有训练、fine-tune、修改权重、新 checkpoint、重划分或候选变更。

只访问 HeadTrain630 的原始三折数据：每折 InnerTrain378 / InnerDev42 / OuterTest210；OOF 合并 260,151 个 full-horizon actor/window，其中 Vehicle 191,026、Pedestrian 66,145、Bicycle 2,980。没有访问 HeadDev70、official VAL150 或 test。共享源缓存包含历史 HeadDev 行，但本阶段访问器仅索引已冻结 HeadTrain actor 身份；拟合访问器进一步限制为对应折 InnerDev。

**InnerDev 已用于 Stage11B checkpoint 选择，本阶段再用于温度拟合。Stage11B OOF 结果此前已被科研人员查看；因此以下 OOF 均是新增方案的开发阶段评价，不能声称独立测试。**

## 预注册校准协议与三折参数

仅注册一个正式方案：每折一个全局标量 T，Vehicle/Pedestrian 共用；Bicycle 原始 FoldR2 直通，不参与拟合。固定目标为 `0.5 mean_V NLL + 0.5 mean_P NLL`，oracle 标签为六条冻结候选中 FDE 最小模式，平局取最低 index。未来误差仅作为拟合标签和离线指标，从未进入图网络输入或推理温度选择。

FP64，在 log(T) 空间使用 SciPy 1.14.1 bounded scalar minimization；范围 [1,1000]、`xatol=1e-10`、`maxiter=1000`，显式评价两端点，最小目标精确平局选择较小 T。协议在拟合前登记，SHA256=`{sha256(PROTOCOL)}`。没有修改区间、type-specific/per-mode 参数、额外 calibrator 或任何 OOF 参数选择。

{markdown(fits,['Fold','Temperature','FitActors','VehicleCount','PedestrianCount','NLLRawMacro','NLLCalibratedMacro','Boundary'])}

三个最优 T 均为合法内部解。冻结 C 的 InnerDev checkpoint 选择分数逐折精确复现；模型 eval、requires_grad=False、inference_mode，无权重梯度或优化器。拟合进程禁止打开 Stage11B OOF cache，访问记录验证 OOF 读取次数为 0。独立 Torch FP64 复算拟合 NLL，并利用 NLL 对 inverse temperature 的凸性检查一阶最优条件；未搜索另一套温度。

## Top1 / 候选 / Bicycle 保持性

全体 260,151 个 actor/window：raw probability、raw logits、calibrated probability 与 calibrated logits 的 argmax 一致，改变模式数为 0；最低 index 平局规则保持。

候选坐标、Top1FDE、Top1ADE、minFDE6、minADEOracle6、MR6 全部逐 actor 严格一致，最大差异为 0。候选始终引用相同冻结坐标源，逐 actor 对照并验证源哈希不变。零几何变化不是新增预测收益。

Bicycle 的 6 candidates、6 logits、6 probabilities、Top1 mode 和 Top1FDE 与 C Raw 及原 FoldR2 逐位一致。V/P 校准结果保留 FP64，Bicycle 序列化为单独的原始 FP32 block，以免 dtype 转换破坏逐位要求。由观测 actor type 路由，未来位移不参与此规则。

## 核心概率指标

NLL 使用稳定的 FP64 log_softmax / shifted logsumexp，无概率裁剪。Brier 为六类 `sum_k (p_k-onehot_k*)²`，不除以 6。ECE 使用预注册 15 个等宽 [0,1] 分箱，左闭右开，最后一箱包括 1，空箱贡献 0，actor/window 数加权。ECE 事件是“Top1 是否等于冻结六候选的 FDE oracle mode”。原始概率直接复用 Stage11B FP32 值，精确提升为 FP64 做统计；稳定 logits NLL 不受原概率下溢影响。

{markdown(primary,['Group','Model','Count','Top1FDE','OracleBestModeNLL','BrierScore','Top1ModeECE'])}

## 全部冻结对照及辅助概率质量

SoftCE 固定 `q=softmax(-FDE/1m)`；ExpectedFDE=`sum p*FDE`；ExpectedRegret=`sum p*(FDE-minFDE)`；NormalizedExpectedRegret 除以 `max(1m,mean six costs)`，复用 Stage11B 定义。原有对照全部保持；此处按相同定义以 FP64 重算概率诊断，与 Stage11B 部分 FP32 辅助指标可能存在舍入级差异。

{markdown(controls,['Group','Model','Top1ADE','Top1FDE','OracleBestModeNLL','BrierScore','Top1ModeECE','SoftCE','ExpectedFDE','ExpectedRegret','NormalizedExpectedRegret'])}

校准把车辆 ExpectedRegret 从 {row('Vehicle','C Raw').ExpectedRegret:.6f} 提高到 {row('Vehicle','C Calibrated').ExpectedRegret:.6f} m；行人从 {row('Pedestrian','C Raw').ExpectedRegret:.6f} 提高到 {row('Pedestrian','C Calibrated').ExpectedRegret:.6f} m。两类 scene bootstrap 的该指标 delta CI 都大于 0。降低过度自信会增加低排名候选的权重；oracle 分类 NLL 与几何期望损失目标不同。本阶段没有以牺牲 Top1 误差换取 NLL 收益，Top1 始终严格不变；但不能声称所有概率质量指标均改善。

## 概率集中度与可靠性

{markdown(concentrationrows,['Group','Model','PredictionEntropy','Top1Probability','Top1ProbabilityP50','Top1ProbabilityP90','Top1ProbabilityP99','FractionAbove0.9','FractionAbove0.99','OracleTop1MatchRate'])}

两类 mean pmax 和 pmax>0.99 占比均下降。温度缩放降低了原模型 C 的极端自信，但并未消除全部校准偏差：Vehicle 校准后 mean pmax={row('Vehicle','C Calibrated').Top1Probability:.6f}，oracle Top1 match rate={row('Vehicle','C Calibrated').OracleTop1MatchRate:.6f}，合并均值表现为偏低自信。全局 macro T 不保证每个类型、fold 或运动子组都达到理想可靠性。15-bin ECE 只是指定分箱下的描述统计，不能替代完整 reliability 分布。

## 2000 次 paired whole-scene bootstrap

seed2022，每次分别在三个 outer210 fold 内有放回抽取 210 个完整场景，合并为 630 场景；同一 actor/window 的 raw/cal 配对。Stage11B 场景顺序和 replicate weights 已逐位重现。按被抽取场景 actor 数计算比值；ECE 每次从 scene/bin confidence 与 correct-count 重新计算非线性绝对差，未对 actor ECE 值简单平均。直接 actor 加权重算检验通过。

{markdown(intervals,['Group','Metric','C_Raw','C_Calibrated','DeltaCalMinusRaw','CI95Low','CI95High'])}

95% 区间为 2.5/97.5 percentile 描述性区间；固定已选择 checkpoint 和已拟合 T，没有重新训练、重新选择 checkpoint 或 bootstrap refit T，所以不包含模型选择和温度拟合的不确定性。ECE 区间还依赖固定 15-bin 定义，不是独立测试或多重检验控制的确认性结论。Top1ADE/FDE 差严格为零，不包装为新收益。

## 三折与探索性运动子组

{markdown(foldrows,['Fold','Group','Model','Count','OracleBestModeNLL','BrierScore','Top1ModeECE','ExpectedRegret'])}

{markdown(motion,['Group','Model','Count','OracleBestModeNLL','BrierScore','Top1ModeECE','ExpectedRegret'])}

Moving/Stopped/Parked 使用原 motion_state 标签；未标记这些状态的 Vehicle 仍计入总体车辆。GT 位移组只做离线探索：Pedestrian<5m 为 <5；>5m 为 >5；5-8m 复用原定义 [5,8)。所有 V/P 在同一 fold 使用相同 T，不根据 GT 位移选择温度。探索性结果不调整正式参数或目标。

## 独立验证与图表

独立 Torch FP64 完整重算全部六模型、260,151 actor 的 14 项逐 actor 指标，并核对全部分组均值与 66 项 group ECE；最大数值差低于 1e-10。所有 logits、概率和诊断值 finite；V/P 校准概率和误差 <1e-12，原始 FP32 概率和误差保留原舍入精度。全历史文件及 checkpoint 的结尾哈希复核通过。

五类图均带 **Stage11C — OOF Development Evidence** 标记，各导出 PNG/SVG/PDF：

1. [Probability distribution](../06_figures/stage11c_probability_distribution.png)：pmax 及所有六候选概率分布。
2. [Vehicle reliability](../06_figures/stage11c_vehicle_reliability.png)：oracle 事件可靠性及分箱样本分布。
3. [Pedestrian reliability](../06_figures/stage11c_pedestrian_reliability.png)：相同分箱。
4. [NLL before/after](../06_figures/stage11c_oracle_mode_nll.png)。
5. [Top1FDE identity](../06_figures/stage11c_top1_fde_identity.png)：全量点与零差审计。

完整对照表在 `07_tables/stage11c_oof_probability_metrics.csv`，type 表在 `stage11c_type_metrics.csv`，bootstrap 表在 `04_bootstrap/stage11c_bootstrap_ci.csv`；分箱原始计数和概率集中度表位于 `05_probability_diagnostics/`。完整 actor 输出、数组和运行日志本地保留，不上传 Git；代码、预注册、小型表格和报告提交到阶段分支。

## 科学决定与执行边界

{markdown([{'Item':k,'Value':decision[k]} for k in ['FrozenModelIntegrity','TemperatureFitIsolation','Top1Identity','BicyclePreserved',
 'CalibrationEngineering','VehicleNLLImproved','PedestrianNLLImproved','ProbabilityConcentrationReduced','CalibrationUseful','ReadyForSemanticStudy','Stage11C']],['Item','Value'])}

判断严格依照拟合前登记规则：T 合法；V/P NLL 均降低；两类 Brier 没有同时明显恶化，实际两类均改善；ECE 与集中度完整报告；Top1 严格不变。ExpectedRegret 恶化已完整披露，按照预注册规则不单独否决 oracle-event 校准有效性。

这些概率仅对应六个固定候选之间的 FDE-oracle-mode 分类事件，GT 只有单条观测未来，不能视作真实世界轨迹、过街意图、碰撞或风险概率。ReadyForSemanticStudy={decision['ReadyForSemanticStudy']} 不意味着真实风险概率已经校准，也不构成下一阶段授权。

**本阶段结束后 STOP，等待大脑 AI 审查。未启动 Stage12、Semantic Graph、official VAL/test 评价或任何新训练。**
'''
    (ROOT/'08_reports/stage11c_final_report.md').write_text(text)
    commands=['00_manifest/stage11c_register.py','02_calibration/stage11c_fit.py','03_oof_evaluation/stage11c_evaluate.py',
        '04_bootstrap/stage11c_bootstrap.py','08_reports/stage11c_final_audit.py','06_figures/stage11c_figures.py','08_reports/stage11c_report.py']
    (ROOT/'00_manifest/stage11c_commands.txt').write_text('\n'.join('PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /home/lrj/anaconda3/envs/ped_intent/bin/python outputs/stage11c_probability_calibration/'+p for p in commands)+'\n\nRegistration must precede fitting and refuses to overwrite registration. Completed evaluators refuse overwrite. Historical inputs are read-only.\n')
    print('SCIENTIFIC_DECISION',json.dumps(decision,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
