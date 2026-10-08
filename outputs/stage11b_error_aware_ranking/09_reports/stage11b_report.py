"""Render the final report and experiment figures from frozen OOF tables."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'00_manifest'))
from stage11b_common import *

def markdown(df,columns=None):
    if columns is not None:df=df[columns]
    def cell(v):
        if pd.isna(v):return '—'
        if isinstance(v,(float,np.floating)):return f'{v:.6f}'
        return str(v)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(cell,row))+' |' for row in df.itertuples(index=False,name=None))

def figures(results,ci,mean_ci):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    colors={'R0':'#9a9a9a','R2':'#444444','A':'#3978a8','B':'#d99036','C':'#418966'}
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,group in zip(axes,('Overall','Vehicle','Pedestrian')):
        for j,name in enumerate(MODELS):
            r=results[(results.Group==group)&(results.Model==name)].iloc[0];c=mean_ci[(mean_ci.Group==group)&(mean_ci.Model==name)].iloc[0]
            ax.bar(j,r.Top1FDE,color=colors[name]);ax.errorbar(j,r.Top1FDE,yerr=[[r.Top1FDE-c.CI95Lower],[c.CI95Upper-r.Top1FDE]],color='black',capsize=3,fmt='none')
        ax.set(xticks=range(5),xticklabels=MODELS,title=group,ylabel='OOF Top1FDE (m)')
    for ext in ('png','svg','pdf'):fig.savefig(ROOT/f'08_figures/stage11b_oof_top1fde.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.4),layout='constrained')
    r=ci[(ci.Comparison=='C-A')&ci.Group.isin(['Vehicle','Pedestrian','Overall'])].set_index('Group')
    for j,g in enumerate(('Vehicle','Pedestrian','Overall')):
        c=r.loc[g];ax.plot([c.CI95Lower,c.CI95Upper],[j,j],lw=4,color=colors['C'],label='95% descriptive' if j==0 else None)
        if g!='Overall':ax.plot([c.BonferroniCILower,c.BonferroniCIUpper],[j,j],lw=1.5,color='black',label='97.5% individual Bonferroni' if j==0 else None)
        ax.plot(c.DeltaTop1FDE,j,'o',color='black')
    ax.axvline(0,color='#999999',lw=1);ax.set(yticks=range(3),yticklabels=['Vehicle','Pedestrian','Overall'],xlabel='C minus A Top1FDE (m); negative means improvement')
    ax.legend(loc='best',fontsize=8)
    for ext in ('png','svg','pdf'):fig.savefig(ROOT/f'08_figures/stage11b_primary_deltas.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for fold,ax in enumerate(axes,1):
        for name in VARIANTS:
            curve=pd.read_csv(ROOT/f'03_training/fold{fold}/{name}/stage11b_training_curve.csv');ax.plot(curve.Epoch,curve.DevSelectionScore,color=colors[name],label=name)
            row=curve.iloc[curve.DevSelectionScore.argmin()];ax.plot(row.Epoch,row.DevSelectionScore,'o',color=colors[name],ms=4)
        ax.set(title=f'Fold {fold}',xlabel='Epoch',ylabel='Fixed InnerDev selection score');ax.legend()
    for ext in ('png','svg','pdf'):fig.savefig(ROOT/f'08_figures/stage11b_selection_curves.{ext}',dpi=180)
    plt.close(fig)

def main():
    verify(history=True);audit=read_json(ROOT/'09_reports/stage11b_final_audit.json');assert audit['Status']=='PASS'
    decision=read_json(ROOT/'09_reports/stage11b_scientific_decision.json');assert decision['EngineeringGate']=='PASS'
    pre=read_json(ROOT/'01_preflight/stage11b_data_and_poison_audit.json');tiny=read_json(ROOT/'01_preflight/stage11b_tiny_audit.json')
    result=pd.concat([pd.read_csv(ROOT/f'05_oof_evaluation/stage11b_oof_{name}_results.csv') for name in ('main','type','motion')],ignore_index=True)
    ci=pd.read_csv(ROOT/'06_bootstrap/stage11b_bootstrap_ci.csv');mean_ci=pd.read_csv(ROOT/'06_bootstrap/stage11b_model_mean_ci.csv')
    switch=pd.read_csv(ROOT/'07_diagnostics/stage11b_mode_switch_cost.csv');prob=pd.read_csv(ROOT/'07_diagnostics/stage11b_probability_statistics.csv')
    training=pd.read_csv(ROOT/'03_training/stage11b_training_summary.csv');cp=pd.read_csv(ROOT/'04_checkpoints/stage11b_checkpoint_manifest.csv')
    eff=pd.read_csv(ROOT/'07_diagnostics/stage11b_efficiency.csv');tinytable=pd.read_csv(ROOT/'01_preflight/stage11b_tiny_results.csv')
    def subset(df,group):return df[df.Group==group]
    def metric(group,model,key='Top1FDE'):return float(result[(result.Group==group)&(result.Model==model)].iloc[0][key])
    def point(comp,group):return float(ci[(ci.Comparison==comp)&(ci.Group==group)].iloc[0].DeltaTop1FDE)
    sections=[]
    def add(title,text):sections.append(f'## 【{title}】\n\n{text}')
    add('Research Question',f"固定 Stage5A 的六条候选和 Stage8 G1 架构，仅改变 ranking objective，检验车辆高代价错误切换、行人错误切换以及 SoftCE 与 Top1FDE 不对齐的问题。主要比较在训练前登记为 C−A 的 Vehicle / Pedestrian Top1FDE；Overall 作为方向检查。\n\n实际结论：**ErrorAwareRanking={decision['ErrorAwareRanking']}；HardCE={decision['HardCE']}；Stage11B={decision['Stage11B']}**。本阶段已完成后停止，等待大脑AI审查。")
    add('Stage11A Findings','Stage11A 冻结结论保持：LossRankingMismatch=CONFIRMED，CostlyModeSwitchProblem=CONFIRMED，CandidateGeometryBottleneck=PARTIAL，SoftTargetAmbiguity=NOT_CONFIRMED。此前车辆错误切换的平均 harm 高于平均 gain，行人错误切换次数增加；这些发现提出本阶段假设，并不预定本阶段结果。Stage8 AgentGraph=SUPPORTED / FSCG=NOT_SUPPORTED / SemanticGraph=NOT_SUPPORTED，Stage9 TAFIG=NOT_SUPPORTED，Stage10A=STOP，均未改写。')
    add('Controlled Variables','复用原始 Stage8 `SparseGraphReranker("G1")`：Node15D、Edge17D、hidden64、1层消息传递、50m、nearest8、K6、Th5、Tf12、24,066 参数。最终分数是 original Stage5A logits + graph delta；未使用 Stage9 R2 residual、semantic、adapter 或 gate。每折 A/B/C 初始 state_dict 相同，最后评分层零初始化；折1/2/3种子2022/2122/2222。候选、feature、训练 actor、同一 epoch 的 carry batch 顺序、AdamW、FP32、预算和选择指标一致，正式变量只有 loss。\n\nStage5A checkpoint SHA256：`'+PREDICTOR_SHA+'`。预测器 eval、requires_grad=False、梯度数0；完整 forward cache 来源和 bitwise replay 已审计。模型训练仅消耗冻结 cache。')
    add('Fold-specific R2','每折独立新建原 Stage6A R2：19→32→1、673参数、ReLU、最后层零初始化、原 SoftCE loss、seed2022、AdamW lr1e−3 / weight_decay1e−4、FP32、batch1024、末尾不足batch正常更新、eval4096、max50 / patience5。原 CPU Generator(seed2022+epoch) randperm 和 Overall Top1FDE 严格改善选择协议保留。仅改变该折 scene 分区和 normalization。历史 R2 权重与旧 HeadTrain630 normalization 均未加载；折 R2 按 InnerDev Overall Top1FDE 选择，第一轮才具备 checkpoint 资格。')
    splitrows=[]
    for fold in (1,2,3):
        for part in ('InnerTrain','InnerDev','OuterTest'):
            ids=indices(fold,part);f=frame().iloc[ids];splitrows.append({'Fold':fold,'Partition':part,'Scenes':len(split(fold)[part]),'Actors':len(ids),**{t:int((f.agent_type==t).sum()) for t in TYPES}})
    add('3-fold Nested Data Split','仅使用 HeadTrain630。按 SHA256(`2022|outer|scene_token`) 排序，连续三个210-scene block作OuterTest；各折其余420按 SHA256(`2022|inner|foldN|scene_token`) 排序，前42为InnerDev，其余378为InnerTrain。均加scene-token词典序tie规则。全部划分在拟合前登记；三分区互斥，630场景各做一次OuterTest，同scene所有window/actor/sample始终同分区。\n\n'+markdown(pd.DataFrame(splitrows))+'\n\n历史 HeadDev70 的1703个window跳过；official VAL/test从未参与本阶段训练、选择、OOF或归一化。')
    add('Normalization','各折 Graph连续feature和R2连续feature均仅由该折InnerTrain378拟合。使用原实现的valid-neighbor masks、population std(ddof=0)+1e−6；type/binary字段不变，missing-neighbor convention保留。R2 raw features从仅含可观测信息的历史窗口重新计算，未反归一化旧630统计量。fit scene列表、actor索引SHA及统计量保存于`../02_splits/`。100窗口GT poison：GT置NaN并扰动future/target masks，固定既定actor索引，raw graph/R2输入逐位不变；166次Bicycle-neighbor出现保留。')
    add('A SoftCE','`q=softmax(−FDE/1m)`；`L_A=mean_i[−Σ_k q_ik log p_ik]`。这是在本次CV协议重新训练的G1受控对照，不是历史G1 checkpoint。GT FDE只用于监督/offline。')
    add('B HardCE','`best=argmin_k FDE_k`，tie选择最低mode index；`L_B=mean_i[−log p_i,best]`。B是预先登记的secondary对照；其结果不改变C−A的两项主要比较。')
    add('C Normalized Error-Aware Ranking','`c_ik=FDE_ik−min_j FDE_ij`，`scale_i=max(1m, mean_k c_ik)`，`L_C=mean_i[Σ_k p_ik c_ik/scale_i]`。这是**normalized formulation**，与Stage11A未归一化的原始expected regret不同。scale、1m下限、无clip、无额外类别权重在训练前固定；只控制actor loss尺度，不保证类别平衡。FDE/c/scale都detach，绝不进入node/edge/ranker/inference gate。')
    add('Tiny Overfit','折1 InnerTrain 固定128 actor、每模型300 optimizer updates、同初始状态、tiny权重全部丢弃。A只按高于q entropy floor的excess计算下降，B/C按自身初始loss相对下降；三者都finite、梯度finite且非零、预测器梯度0。\n\n'+markdown(tinytable,['Variant','Status','Updates','InitialLoss','FinalLoss','EntropyFloor','Reduction','Threshold','PredictorGradientCount'])+'\n\n全部门槛在正式CV前通过，没有调参救结果。')
    add('Training','严格执行折1 R2→A→B→C，然后折2、折3。ABC：AdamW lr1e−3、weight_decay1e−4，FP32，无AMP，micro128、累积8、effective1024、max50、patience5。训练剩余项carry至下一epoch；停止时pending occurrences保存在last.pt，所有distinct有效actor均已参与至少一次optimizer update，包括Bicycle target。不同方案执行epoch数可因统一早停规则而不同；最大预算与共用epoch的batch顺序相同。\n\n'+markdown(training, ['Fold','Model','SelectedEpoch','ExecutedEpochs','CheckpointScore','Seconds','MeanEpochSeconds','Params','AllDistinctTrainingActorsUsed'])+'\n\n总训练用时 '+f"{training.Seconds.sum():.2f}"+' 秒。A/B/C的每epoch TrainLoss、DevLoss、类型/Overall Top1FDE、S_dev、entropy、max probability、gradient norm完整保存。C额外保存raw/normalized regret和scale分布。各模型启动时额外收集可用脚本SHA；评估辅助脚本可在训练期间新增，实际training/common/tiny源码SHA全程相同，不构成训练变量。')
    add('Checkpoint Selection','先固定每折R2最佳InnerDev Vehicle/Pedestrian Top1FDE。ABC统一选择 `S_dev=.5*VehicleDevFDE/R2VehicleDevFDE+.5*PedestrianDevFDE/R2PedestrianDevFDE` 的最小值；严格改善，第一epoch起选择。没有按各自loss选择，没有根据OuterTest重选。审计从完整curve重算S_dev及argmin，并核对12个best权重SHA。**全部三折12个checkpoint冻结并完成训练审计后，才开始任何OuterTest evaluation**。\n\n'+markdown(cp,['Fold','Model','SelectedEpoch','CheckpointScore','SHA256'])+'\n\n完整manifest还包含split、normalization及training-config SHA。')
    add('OOF Identity',f"合并仅包含630个OuterTest scene、{audit['OOFActors']}个full-horizon actor-window：Vehicle191026 / Pedestrian66145 / Bicycle2980。actor_id及source_index一一对应，没有InnerDev预测混入。每fold相应ranking训练/选择没有接触其OuterTest。共享candidate坐标maxdiff=0，minADEOracle6/minFDE6/MR6在五模型逐位一致；MR6定义oracle minFDE>2m。原R0分数、FoldR2、ABC均使用同一冻结候选。缓存prediction/logit/prob/metric/identity文件保存本地，SHA公开；不上传大数组。")
    for title,group in [('Overall Results','Overall'),('Vehicle Results','Vehicle'),('Pedestrian Results','Pedestrian')]:
        text=markdown(subset(result,group),['Model','Count','Top1FDE','Top1ADE','OracleGap','HitRate','MRR','SoftCE','ExpectedRegret','NormalizedExpectedRegret'])
        text+='\n\n'+markdown(ci[(ci.Group==group)&ci.Comparison.isin(['C-A','C-R2','B-A','C-B'])],['Comparison','DeltaTop1FDE','CI95Lower','CI95Upper','BonferroniCILower','BonferroniCIUpper'])
        if group=='Pedestrian':
            oracle=metric(group,'C','minFDE6');top=metric(group,'C');text+=f'\n\n冻结候选oracle minFDE={oracle:.6f}m，C Top1FDE={top:.6f}m；oracle为Top1FDE的{100*oracle/top:.2f}%，残余排序gap={top-oracle:.6f}m。这只描述候选条件下的误差构成，不给PedestrianCandidateRefinement新的训练授权。'
            any_gain=any(metric(group,m)<metric(group,'R2') for m in VARIANTS)
            text+='\n\n'+('本次至少一个ranking objective的行人点估计优于FoldR2；因此不能把结果解释为所有排序目标均失败，是否值得确认仍按预登记主要比较和R2 guard判断。' if any_gain else '本次三个ranking objective的行人点估计均未优于FoldR2。Pedestrian Candidate Refinement可作为大脑AI审查的候选研究方向；oracle/gap的绝对数值仍需结合场景难度判断，当前证据不能证明候选生成优化一定有效。')
        if group=='Overall':text+='\n\n![OOF Top1FDE](../08_figures/stage11b_oof_top1fde.png)'
        add(title,text)
    add('Bicycle Results','Bicycle仍有完整观测、六条候选和六个最终概率，参与graph neighbor、ABC训练loss和Overall。最终ABC按提前固定路由使用相同fold R2；全部2980个OOF Bicycle的candidate/logits/probability/selected mode及metric逐位一致。Bicycle稳定来自该部署策略，不能解释为graph本身学会了避免负迁移。\n\n'+markdown(subset(result,'Bicycle'),['Model','Count','Top1FDE','Top1ADE','OracleGap','HitRate','MRR']))
    add('MovingVehicle',markdown(result[result.Group.isin(['MovingVehicle','StoppedVehicle','ParkedVehicle','Vehicle>5m'])],['Group','Model','Count','Top1FDE','Top1ADE','OracleGap'])+'\n\n以上motion组属于exploratory；没有用于split、checkpoint选择或推理路由。')
    add('Pedestrian Motion Groups',markdown(result[result.Group.isin(['Pedestrian<5m','Pedestrian>5m','Pedestrian5-8m'])],['Group','Model','Count','Top1FDE','Top1ADE','OracleGap'])+'\n\nGT位移仅离线诊断。<5m和>5m严格不含等于5m；5–8m采用[5,8)并复用Stage11A分组。未将GT未来运动组输入推理。')
    add('High-Cost Switch Analysis','delta定义new−reference；负为改善。changed按mode index变化；gain/harm按严格FDE差值正负，不用结果筛阈值。mean gain/harm是条件均值，gross是组内绝对总和；net delta按全部actor平均。top10取最大的ceil(10%×worsened_count)个正harm。\n\n'+markdown(switch[switch.Group.isin(TYPES[:2])],['Group','Comparison','ChangedCount','ImprovedCount','WorsenedCount','MeanGain','MeanHarm','GrossGain','GrossHarm','NetFDEDelta','P90Harm','P95Harm','P99Harm','Top10HarmShare'])+f"\n\n相对A−R2，C−R2车辆gross harm减少{decision['VehicleGrossHarmReduction']:.6f}m（{100*decision['VehicleGrossHarmReductionFraction']:.2f}%）；top10正harm总量减少{decision['VehicleTop10HarmSumReduction']:.6f}m（{100*decision['VehicleTop10HarmSumReductionFraction']:.2f}%）。行人worsened次数减少{decision['PedestrianWrongSwitchCountReduction']}（{100*decision['PedestrianWrongSwitchCountReductionFraction']:.2f}%）。负的“减少”表示增加。CostlySwitchReduced要求gross和tail sum都下降；PedestrianWrongSwitchReduced只评价频率，不等价于净FDE改善或错误严重度下降。")
    add('Probability Analysis',markdown(prob[prob.Group.isin(['Vehicle','Pedestrian'])],['Group','Model','PredictionEntropy','Top1Probability','Top1Top2Margin','OracleModeProbability','ProbabilityAbove0p9Fraction','ProbabilityAbove0p99Fraction','ExpectedRegret','NormalizedExpectedRegret'])+'\n\n0.9/0.99阈值在OOF计算前固定于统计代码，仅描述高置信概率占比，不是筛选模型、推理gate或校准声明。B和C是否呈现更低entropy/更高pmax必须从表读出；高置信并不证明选择正确，结合HitRate、oracle-mode probability和Top1FDE解释。没有概率NaN/Inf或归一化失败。')
    probability_notes=[]
    for group in TYPES[:2]:
        a=prob[(prob.Group==group)&(prob.Model=='A')].iloc[0]
        for name in ('B','C'):
            r=prob[(prob.Group==group)&(prob.Model==name)].iloc[0]
            concentrated=r.PredictionEntropy<a.PredictionEntropy and r.Top1Probability>a.Top1Probability
            probability_notes.append(f'{group} {name}：相对A'+('输出更集中' if concentrated else '未呈现entropy下降且pmax上升的一致集中模式')+f'；entropy={r.PredictionEntropy:.6f}，mean pmax={r.Top1Probability:.6f}，pmax>0.99占比={100*r.ProbabilityAbove0p99Fraction:.2f}%，HitRate={metric(group,name,"HitRate"):.6f}，oracle-mode probability={r.OracleModeProbability:.6f}。')
    sections[-1]+='\n\n'+'\n\n'.join(probability_notes)+'\n\n概率接近one-hot而HitRate远低于1，说明尖锐输出并不等于确定性正确；这是概率质量的描述性风险，未做正式校准检验，也不能从单一统计量宣称因果或完整概率塌缩。'
    add('Bootstrap','2000 paired scene-cluster replicates，seed2022。每replicate在每折210个Outer scene内有放回抽210次，全部scene actor/window一起赋权，并合并三折actor-window平均。五模型、所有比较和group共用同组draws。模型固定，不重新训练/选择。报告percentile95%区间[2.5,97.5]；C−A的Vehicle/Pedestrian是两个共同primary，同时提供97.5% individual区间[1.25,98.75]以作Bonferroni family95%覆盖调整。Secondary/motion不伪装成primary。权重文件被独立seed replay核对。\n\n'+markdown(ci[(ci.Comparison.isin(['C-A','C-R2']))&ci.Group.isin(['Overall','Vehicle','Pedestrian'])],['Comparison','Group','Count','DeltaTop1FDE','CI95Lower','CI95Upper','BonferroniCILower','BonferroniCIUpper'])+'\n\n![Primary differences](../08_figures/stage11b_primary_deltas.png)')
    add('Limitations','假设来自Stage11A；历史HeadDev与official VAL已经用于此前研究开发。本阶段没有再用它们，但冻结Stage5A可能在历史训练中使用过这些scene。因此OOF是冻结候选条件下排序模型的内部泛化证据，**不是 fully independent end-to-end test**。三个fold的数据、模型和训练集有依赖；bootstrap是在固定训练结果上的scene采样不确定性，不涵盖训练seed/训练集变动。只有各折一套固定seed，无追加loss搜索。期望归一化regret目标不必然改善离散Top1选择；尺度归一化也不保证类别公平。GT未来组仅用于离线解释，不能据此改推理。\n\n推理时延仅含缓存feature准备/排序，不含冻结候选生成器；synchronized batch128的观测含首批启动开销。各fold评估GPU峰值包含同时加载的R2/ABC，不应当作单模型独占峰值；训练峰值逐模型记录。实验效率表：\n\n'+markdown(eff,['Fold','Model','Params','TrainingSeconds','MeanEpochSeconds','HeadLatencyMSPerActor','ApproxRankingWithFeaturePreparationMSPerActor','TrainPeakGPUMemoryBytes','EvalPeakGPUMemoryBytes']))
    add('Scientific Decision','以下结论按训练前登记规则计算，没有根据结果更换primary、loss scale或结构。HardCE仅是secondary开发证据；即使比C更好，也等待大脑AI决定是否进一步验证。\n\n'+ '\n'.join(f'- **{k}={decision[k]}**' for k in ('LossImplementation','CVIsolation','CandidateIdentity','BicyclePreserved','HardCE','ErrorAwareRanking','VehicleImproved','PedestrianImproved','OverallImproved','CostlySwitchReduced','PedestrianWrongSwitchReduced','Stage11B','ReadyForFurtherConfirmation'))+'\n\n![Selection curves](../08_figures/stage11b_selection_curves.png)\n\n完成Stage11B后STOP：没有official VAL/test评价、没有Stage11C、没有新loss/seed搜索、没有网络结构修改。代码、小型报告和图表推送独立分支，未merge；checkpoint/缓存/actor级大表仅保存在本地。')
    report='# Stage11B Controlled Error-Aware Ranking Objective Study\n\n基础commit：`'+BASE+'`；分支：`stage11b/controlled-ranking-objectives`。\n\n'+'\n\n'.join(sections)+'\n'
    (ROOT/'09_reports/stage11b_final_report.md').write_text(report)
    figures(result,ci,mean_ci)
    atomic_json(ROOT/'08_figures/stage11b_figure_manifest.json',{'Conclusion':decision['ErrorAwareRanking'],'FigureSources':'frozen OOF and selected InnerDev logs, no re-fitting',
        'Files':{p.name:sha256(p) for p in sorted((ROOT/'08_figures').glob('stage11b_*')) if p.suffix in ('.png','.svg','.pdf')},
        'Intervals':'OOF means95 descriptive; C-A primary97.5 individual Bonferroni','Export':['PNG','SVG','PDF']})
    verify(history=True);print('REPORT_AND_FIGURES_COMPLETE',flush=True)

if __name__=='__main__':main()
