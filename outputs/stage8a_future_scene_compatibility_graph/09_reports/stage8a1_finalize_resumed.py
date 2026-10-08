"""Evidence-first resumed report; first-attempt STOP report remains immutable."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage8a1_common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(ROOT/'04_bootstrap'))
from stage8a1_analyze import masks

def md(df,columns=None):
    if columns is not None:df=df[columns]
    cols=list(df.columns);out='|'+ '|'.join(cols)+'|\n|'+'|'.join(['---']*len(cols))+'|\n'
    for row in df.itertuples(index=False,name=None):
        out+='|'+'|'.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+'|\n'
    return out

def main():
    verify();frozen=read_json(FROZEN);decision=read_json(ROOT/'09_reports/stage8a1_scientific_decision.json')
    complete=read_json(ROOT/'03_evaluation/stage8a1_complete.json');identity=read_json(ROOT/'03_evaluation/stage8a1_candidate_identity.json')
    assert complete['status']==identity['status']=='PASS'
    for row in frozen['checkpoints']:assert sha256(PROJECT/row['Path'])==row['SHA256']
    history=read_json(ROOT/'00_manifest/stage8a1_resume_frozen_history.json')
    for p,h in history['files'].items():assert sha256(PROJECT/p)==h,p
    reg=read_json(ROOT/'00_manifest/stage8a1_resume_registration.json')
    for p,h in reg['new_sources_sha256'].items():assert sha256(ROOT/'01_training/00_code'/p)==h,p
    precision_reg=read_json(ROOT/'00_manifest/stage8a1_resume_precision_registration.json')
    for p,h in precision_reg['new_sources_sha256'].items():assert sha256(ROOT/'01_training/00_code'/p)==h,p
    assert read_json(ROOT/'07_cases/stage8a1_case_manifest.json')['status']=='PASS'
    assert read_json(ROOT/'08_efficiency/stage8a1_efficiency_audit.json')['status']=='PASS'
    main=pd.read_csv(ROOT/'06_tables/stage8a1_main_results.csv');groups=pd.read_csv(ROOT/'06_tables/stage8a1_group_results.csv')
    sem=pd.read_csv(ROOT/'06_tables/stage8a1_semantic_group_results.csv');mapping=pd.read_csv(ROOT/'06_tables/stage8a1_map_availability_results.csv')
    ci=pd.read_csv(ROOT/'06_tables/stage8a1_bootstrap_ci.csv');mode=pd.read_csv(ROOT/'06_tables/stage8a1_mode_change_analysis.csv')
    att=pd.read_csv(ROOT/'06_tables/stage8a1_attention_diagnostics.csv');eff=pd.read_csv(ROOT/'06_tables/stage8a1_efficiency.csv')
    amended=pd.read_csv(ROOT/'06_tables/stage8a1_amended_tiny_gate_results.csv');norm=pd.read_csv(ROOT/'06_tables/stage8a1_normalization_audit.csv')
    configs={v:read_json(ROOT/'01_training'/v/'formal/training_config.json') for v in PARAMS}
    summaries={v:read_json(ROOT/'01_training'/v/'formal/stage8a1_summary.json') for v in PARAMS}
    actors=pd.read_csv(ROOT/'03_evaluation/stage8a1_actor_results.csv');assert len(actors)==54990
    charts=[]
    fig,axs=plt.subplots(1,3,figsize=(11,3),sharey=True)
    for ax,v in zip(axs,PARAMS):
        curve=pd.read_csv(ROOT/'01_training'/v/'formal/training_curve.csv')
        ax.plot(curve.epoch,curve.train_loss,label='HeadTrain');ax.plot(curve.epoch,curve.dev_loss,label='HeadDev')
        ax.axvline(summaries[v]['best_epoch'],ls=':',color='black',label='Selected min dev loss');ax.set_title(v);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axs[0].set_ylabel('Soft ranking loss');axs[-1].legend(fontsize=7);fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(ROOT/'05_diagnostics'/f'stage8a1_formal_training_curves.{suffix}',dpi=160)
    plt.close(fig)
    svg=ROOT/'05_diagnostics/stage8a1_formal_training_curves.svg';svg.write_text('\n'.join(s.rstrip() for s in svg.read_text().splitlines())+'\n')
    trows=[{'Variant':v,'Epochs':s['executed_epochs'],'SelectedEpoch':s['best_epoch'],
        'HeadDevLoss':s['selected']['dev_loss'],'HeadDevTop1FDE_record_only':s['selected']['dev_Top1FDE'],
        'Microbatch':s['microbatch'],'Accumulation':s['accumulation'],'AMP':s['AMP'],'EffectiveBatch':s['effective_batch']} for v,s in summaries.items()]
    write_csv(ROOT/'06_tables/stage8a1_formal_training_summary.csv',trows)
    def pair(n,b,g='Overall'):return ci[(ci.New==n)&(ci.Baseline==b)&(ci.Group==g)&(ci.Metric=='Top1FDE')]
    def pairtext(n,b):
        row=pair(n,b).iloc[0];return f"{n}−{b}: ΔTop1FDE={row.Delta:.6f}m,95% scene CI[{row.CI_lower:.6f},{row.CI_upper:.6f}]。"
    def modeltable(v):return md(main[main.Model==v])
    rationale=[]
    for v in ('G3','G1'):
        pedestrian=pair(v,'R2','Pedestrian').iloc[0]
        if pedestrian.CI_lower>0:
            rationale.append(f"{v}−R2 的 Overall Top1FDE 虽可靠改善，但 Pedestrian Δ={pedestrian.Delta:.6f}m，95% scene CI[{pedestrian.CI_lower:.6f},{pedestrian.CI_upper:.6f}]，属于可靠恶化。"+
                ('这违反 FSCG 的预注册 Vehicle/Pedestrian 无可靠恶化条件，因此 FSCG=NOT_SUPPORTED。' if v=='G3' else '这也不满足推荐 G1 的预注册条件，因此 RecommendedFinalModel=R2。'))
    semantic=pair('G3','G1').iloc[0]
    if semantic.CI_lower<=0<=semantic.CI_upper:
        rationale.append(f"G3−G1 的 Overall Δ={semantic.Delta:.6f}m，95% scene CI[{semantic.CI_lower:.6f},{semantic.CI_upper:.6f}] 包含0；同时未满足冻结的 targeted semantic 支持条件，SemanticContributionToJoint=NOT_SUPPORTED。")
    columns=['Group','Model','Count','Top1ADE','Top1FDE','OracleGap','HitRate']
    state='FSCG 未满足预定支持条件，按授权STOP；不调模型。' if decision['FSCG']=='NOT_SUPPORTED' else 'FSCG 满足预定支持条件；本阶段仍STOP，不进入下一研究阶段。'
    empty=mapping[mapping.Group=='ZeroMap']
    sections=[('Research Question','是否 learned mode-mode 与 sparse semantic mode-map 可以独立/联合改善固定候选轨迹的ranking，并可靠超过冻结R2？主比较为G3−R2。全部科学判定沿用修正前冻结规则；本次只修正不可实现的工程tiny判据。'),
      ('Frozen Inputs','Stage5A predictor SHA `88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`；R2 SHA `e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314`。predictor eval、requires_grad=False、gradient0。历史commit `2a79b0d687192002537073147d93658bf4c6423e` 全部跟踪文件保持原样，包括原STOP与FAIL证据；见 resume_frozen_history。'),
      ('Graph Specification','0C GraphSpecFrozen=YES。lane10m、polygon2m、原三类quota、neighbor<=50m/max8、K6、Th5/Tf12、hidden64、温度1m与15/17/18/11D特征均不变。G1/G2/G3参数24066/20546/37187。interaction value input遵循冻结raw15+raw15+17=47D；attention input145D。真实map edges为空时exact zero64D，padding不进map softmax，无NULL/fallback/筛样。'),
      ('Training Protocol','630 HeadTrain/70 HeadDev，沿用Stage6A seed2022 scene split；targets260151/29934。AdamW lr1e-3、weight_decay1e-4、effective batch1024、max50epochs、patience5；连续epoch流末尾余数carry，严格不使用小于1024的优化步骤。G1→G2→G3均从原独立冻结初始化开始，未复用tiny/G1/G2权重。仅根据HeadDev ranking loss strict minimum选择。\n\n'+md(pd.DataFrame(trows))+'\n![Training curves](../05_diagnostics/stage8a1_formal_training_curves.png)\n\n'+
        '\n'.join(v+': '+json.dumps(summaries[v]['fallbacks'],ensure_ascii=False) for v in PARAMS)+'\n\nFP16工程错误只触发允许的FP32回退，不修改冻结实现。'),
      ('Normalization',f'HeadTrain-only连续列mean/std；one-hot/binary/valid flags原值，padding/无效方向保留0。实际normalized mean最大绝对值{norm.Mean.abs().max():.3g}，非constant std接近1。HeadDev/VAL不参与。GT只用于ranking label及离线评价；observable图先构建后join标签，adapter future-poison16 TRAIN windows与冻结0C200-window审计均PASS。'),
      ('Tiny Overfit','历史first attempt=STOP_TINY_GATE_FAILURE；failure source=INFEASIBLE_TINY_CRITERION，分类PROTOCOL_BUG。原20% total-loss gate低于H(q)=1.294783592224121，原FAIL不改写。新的ExcessLossReduction>=0.90及finite/nonzero/gradient0等附加检查，仅重判已有fixed128 seed2022/300-update FP32结果。tiny重复次数0、新增tiny更新0；修正发生在正式训练与officialVAL之前。\n\n'+md(amended[['Variant','HistoricalStatus','InitialExcessLoss','FinalExcessLoss','ExcessLossReduction','AmendedStatus']])),
      ('Checkpoint Selection','选择指标只有HeadDev ranking loss，Top1ADE/FDE/OracleGap仅记录。全部训练结束后冻结SHA，随后才第一次统一VAL；freeze后无训练/覆盖。\n\n'+md(pd.DataFrame(frozen['checkpoints']))),
      ('Candidate Identity','PASS。同一次fresh冻结predictor pipeline中R0/R2/G1/G2/G3共享一个候选tensor；全部fresh window raw/ego SHA与冻结cache逐位一致，GT/masks/identities也逐位核对。模型不输出/修改geometry。minADE6/minFDE6/MR6最大差0，候选maxdiff0；attention hook不改变forward。54990 full targets，150 scenes，TurningVehicle_GT1663。正式成功VAL pass为1次；首批16windows曾因availability索引错误在保存任何batch结果前终止，失败registration与修复TRAIN审计另存，未用于训练或模型选择。保留历史metric convention：minADE6是minFDE-selected mode的ADE，MR6为minFDE>2m。'),
      ('Main Results',md(main)+'\n主指标Top1FDE，单位m；Count是actor-window targets，Params是新增ranking head参数，不包含共同冻结predictor。'),
      ('R0','重新运行Stage5A original ranking；本阶段正式结果来自统一VAL，不抄历史CSV。\n\n'+modeltable('R0')),
      ('R2','重新加载SHA冻结R2与其原HeadTrain normalization，在同一fresh VAL pipeline重新计算hand-crafted interaction features与probability。\n\n'+modeltable('R2')),
      ('G1 Future Interaction Graph',pairtext('G1','R0')+'\n\n'+modeltable('G1')),
      ('G2 Semantic Compatibility Graph',pairtext('G2','R0')+'\n\n'+modeltable('G2')),
      ('G3 Joint FSCG',pairtext('G3','R2')+'\n\n'+modeltable('G3')),
      ('G1 vs R2',pairtext('G1','R2')+'\nLearnedGraphAdvantageOverR2='+decision['LearnedGraphAdvantageOverR2']),
      ('G3 vs R2',pairtext('G3','R2')+'\nFSCG='+decision['FSCG']+'\n\n'+pairtext('G3','G1')+'\n'+pairtext('G3','G2'))]
    for title,g in (('Vehicle','Vehicle'),('Pedestrian','Pedestrian'),('Bicycle','Bicycle'),('Moving Vehicle','MovingVehicle')):
        included=('Vehicle','StoppedVehicle','ParkedVehicle','Vehicle>5m') if g=='Vehicle' else ('Pedestrian','Pedestrian<5m','Pedestrian>5m') if g=='Pedestrian' else (g,)
        sections.append((title,md(groups[groups.Group.isin(included)],columns)+'\n\n'+md(pair('G3','R2',g))))
    sections.extend([
      ('Semantic-sensitive Groups','历史t0到完整centerline strict<20m memberships离线identity join；turning endpoint>5m、first/last1s secants>0.5m、abs angle>20°，保留正left/负right。GT group不进模型。\n\n'+md(sem,columns)),
      ('Map Availability','固定actor-level定义：any of6 modes有真实map edge为MapNonEmpty；all6无edge为ZeroMap。RouteCenterlineAvailable仅V/B且任意mode有lane/connector；PedSemanticSpecificAvailable仅P且任意mode有crossing/walkway。分组独立于新ranking，未用于训练筛样。\n\n'+md(mapping,columns)+'\n\nZeroMap的G2/G3改变只能由shared node scorer/interaction与representation产生，不能归因于semantic map branch。'),
      ('Bootstrap','paired whole-scene cluster：150scenes、1000draws、seed2022；相同resampling indices用于所有6 comparisons和groups。actor sums/counts随whole scene一起重复，percentile2.5/97.5%，Δ=new−baseline。FDE/ADE负向改善，HitRate正向改善。secondary subgroup CIs未作multiplicity correction，groups重叠，不能视为独立replications。\n\n'+md(ci[(ci.Group=='Overall')&(ci.Metric=='Top1FDE')])),
      ('Mode Change Analysis',md(mode)+'\nHitRate以minFDE candidate index作ranking oracle；不代表轨迹行为预测准确率。'),
      ('Interaction Diagnostics',md(att[att.Branch!='map'])+'\n在各模型选中top1 mode上计算attention mass；pair方向为target query→neighbor context，邻居保留原6mode。future-min-distance bins预注册为0/2/5/10/20/50/∞m。零邻居actors贡献0，均值保留它们。仅描述权重分布，无因果解释。'),
      ('Semantic Diagnostics',md(att[att.Branch=='map'])+'\nG2/G3各自top1 mode的真实entity attention；按V/P/B与六类entity分组。空map mode贡献0。每个actor top-attention实体存本地sidecar，完整alpha存本地batch结果。Attention不是semantic causal attribution。'),
      ('Cases','aggregate evaluation与scientific decision完成后按固定类别选4个distinct actor instances：R2错/G1选FDE-best；R2错/G2或G3选FDE-best且有semantic context；G1错/G3选FDE-best；G3恶化failure。每图展示GT、6candidates、5model概率/top1、neighbor modes及selected map entities，同一geometry轴。Case2只描述正确ranking与道路语义context共现，不能证明“因为语义”。\n\n'+'\n\n'.join(f'![Case{i}](../07_cases/stage8a1_case{i}.png)' for i in range(1,5))),
      ('Efficiency',md(eff)+'\nA：RTX3080 batch128、100warmup/500measured；CUDA已准备normalized输入，不含geometry feature/retrieval。B：live CPU retrieval/graph/features/normalization，128targets，2warmup/10timings，HDmap index已加载，排除predictor/CPU→CUDA。C：原16完整scene-window predictor batch，实际target数见表；100/500。单位不同，不能把offline cache或A宣称为完整online latency。'),
      ('Limitations','Single fixed seed/model run，VAL150及相关overlapping secondary groups；未作多seed或multiplicity-adjusted subgroup确证。attention/cases为descriptive。候选geometry固定，所有改善限ranking；不能claim更强oracle geometry。t0→future预测图可能对输入候选误差敏感，diagnostic不建立因果。Checkpoint/cache/raw maps/完整actor表仅本地，未上传数据或凭据。历史停止记录未覆盖。'),
      ('Scientific Decision','\n'.join(k+' = '+v for k,v in decision.items())+'\n\n'+'\n\n'.join(rationale)+'\n\n'+state+'\n只有AgentGraph与SemanticContributionToJoint受支持且FSCG支持/targeted支持时，才满足重新讨论联合innovation的预定条件。本报告不因G3胜R0自动宣布创新。')])
    output=ROOT/'09_reports/stage8a1_resumed_final_report.md'
    output.write_text('\n\n'.join('【'+title+'】\n\n'+body for title,body in sections)+'\n')
    atomic_json(ROOT/'09_reports/stage8a1_resumed_final_audit.json',{'status':'COMPLETE','historical_files_unchanged':len(history['files']),
        'HistoricalAttempt':'STOP_TINY_GATE_FAILURE','HistoricalFailureSource':'INFEASIBLE_TINY_CRITERION','TinyGateAmended':'YES',
        'TinyGate':'PASS','tiny_repeated':False,'new_tiny_updates':0,'successful_official_VAL_passes':1,'failed_preflight_scene_windows':16,'VAL_full_targets':54990,'VAL_scenes':150,
        'candidate_identity':'PASS','frozen_checkpoints_sha256':{r['Variant']:r['SHA256'] for r in frozen['checkpoints']},
        'predictor_gradient_count':0,'history_graph_selector_training_hyperparameters_unchanged':True,
        'formal_training_sources_unchanged':True,'scientific_decision':decision,'final_report_sha256':sha256(output),'STOP':True})
    print('FINAL_RESUMED_REPORT_PASS',decision,flush=True)

if __name__=='__main__':main()
