"""Evidence-grounded Stage9A report; no untouched-VAL or innovation claim."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage9a_common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def md(df):
    cols=list(df.columns);out='|'+'|'.join(cols)+'|\n|'+'|'.join(['---']*len(cols))+'|\n'
    for row in df.itertuples(index=False,name=None):out+='|'+'|'.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+'|\n'
    return out

def main():
    verify();history=read_json(ROOT/'00_manifest/stage9a_frozen_history.json')
    for p,h in history['tracked_history'].items():assert sha256(PROJECT/p)==h,p
    reg=read_json(REG)
    for p,h in reg['training_sources'].items():assert sha256(ROOT/'01_training/00_code'/p)==h,p
    freeze=read_json(FROZEN)
    for row in freeze['checkpoints']:assert sha256(PROJECT/row['Path'])==row['SHA256']
    complete=read_json(ROOT/'03_evaluation/stage9a_complete.json');identity=read_json(ROOT/'03_evaluation/stage9a_candidate_identity.json')
    assert complete['status']==identity['status']=='PASS' and complete['Official_VAL_role']=='development-reuse evidence'
    assert complete['actor_csv_sha256']==sha256(ROOT/'03_evaluation/stage9a_actor_results.csv')
    assert read_json(ROOT/'05_diagnostics/stage9a_macro_gradient_audit.json')['status']=='PASS'
    assert read_json(ROOT/'08_efficiency/stage9a_efficiency_audit.json')['status']=='PASS'
    def table(name):return pd.read_csv(ROOT/'06_tables'/f'stage9a_{name}.csv')
    main=table('main_results');types=table('type_results');motion=table('motion_results');context=table('interaction_context_results');ci=table('bootstrap_ci');gates=table('gate_statistics');mode=table('mode_change');training=table('training_summary');eff=table('efficiency')
    decision=read_json(ROOT/'09_reports/stage9a_scientific_decision.json');ratios=read_json(ROOT/'05_diagnostics/stage9a_retention_recovery.json')
    case=read_json(ROOT/'07_cases/stage9a_case_manifest.json');tiny=read_json(ROOT/'01_training/stage9a_tiny_gate.json');assert tiny['status']=='PASS'
    def pair(n,b,g='Overall'):return ci[(ci.New==n)&(ci.Baseline==b)&(ci.Group==g)&(ci.Metric=='Top1FDE')]
    def pairtext(n,b,g='Overall'):
        x=pair(n,b,g).iloc[0];return f'{n}−{b} {g}: Δ={x.Delta:.6f}m，95% scene CI[{x.CI_lower:.6f},{x.CI_upper:.6f}]。'
    columns=['Group','Model','Count','Top1ADE','Top1FDE','OracleGap','HitRate','BestModeRank','MRR']
    fig,axs=plt.subplots(1,3,figsize=(11,3),sharey=True)
    for ax,v in zip(axs,VARIANTS):
        curve=pd.read_csv(ROOT/'01_training'/v/'formal/stage9a_training_curve.csv');selected=read_json(ROOT/'01_training'/v/'formal/stage9a_training_summary.json')['selected_epoch']
        assert int(curve.loc[curve.DevSelectionScore.idxmin(),'epoch'])==selected
        ax.plot(curve.epoch,curve.DevSelectionScore,label='DevSelectionScore');ax.plot(curve.epoch,curve.OverallDevSoftCE,label='Dev micro');ax.plot(curve.epoch,curve.MacroTypeDevSoftCE,label='Dev macro')
        ax.axvline(selected,color='black',ls=':',label='Selected epoch');ax.set_title(v);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    axs[0].set_ylabel('Dev ranking objective');axs[-1].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(ROOT/'05_diagnostics'/f'stage9a_training_curves.{ext}',dpi=160)
    plt.close(fig);svg=ROOT/'05_diagnostics/stage9a_training_curves.svg';svg.write_text('\n'.join(x.rstrip() for x in svg.read_text().splitlines())+'\n')
    macroaudit=read_json(ROOT/'05_diagnostics/stage9a_macro_gradient_audit.json')
    findings=pairtext('T3','R2')+'\n\n'+pairtext('T3','R2','Pedestrian')+'\n'+pairtext('T3','R2','Bicycle')
    findings+='\n\nT3 Overall未可靠改善，Pedestrian与Bicycle均可靠恶化；Pedestrian point delta也超过+.01m。Vehicle收益保留比为负，未满足>=.70条件，因此TAFIG与PaperUsableTAFIG均不受支持。'
    findings+='\n\n'+pairtext('T2','T1','Pedestrian')+' Pedestrian仅点估计改善且CI包含0，因此TypeAdaptation=WEAK_SUPPORTED。'
    findings+='\n\n'+pairtext('T3','T2','Pedestrian')+' 虽然Pedestrian改善，'+pairtext('T3','T2')+pairtext('T3','T2','Vehicle')+' Overall和Vehicle可靠恶化，故BalancedTraining=NOT_SUPPORTED。'
    proof='Official VAL is no longer an untouched confirmation set for the Stage9 hypothesis.\n\nOfficial VAL result = development-reuse evidence。Stage9设计受Stage8 VAL结果启发，不能称为独立test-like confirmation。Stage9模型选择、normalization和early stopping仅用HeadTrain/HeadDev。'
    sections=[('Research Question','在冻结R2 reliability上，通过target-type低秩adapter和actor-level learned gate使用future interaction残差，能否保留Vehicle收益并避免Pedestrian/Bicycle负迁移？\n\n'+proof),
      ('Why Stage8 G1 Failed on Heterogeneous Types','历史Stage8发现类型间冲突；新阶段重新评价G1/G3 reference，不修改Stage8结论。Vehicle收益与Pedestrian/Bicycle退化为development hypothesis，不能据此声称因果机制。SemanticGraph、SemanticContributionToJoint、FSCG历史NOT_SUPPORTED均保留。'),
      ('Frozen R2 Base','Stage5A predictor SHA `88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`。R2 SHA `'+R2SHA+'`。两者eval/requires_grad=False，gradient=0；本阶段不修改candidate geometry。R0/R2/G1/G3/T1/T2/T3同一次fresh pipeline评价，未抄历史表。'),
      ('TA-FIG Architecture','l_TA(i,k)=l_R2(i,k)+g_i·delta_graph(i,k)。只复用G1 15D mode nodes、17D individual directed mode-mode edges、current-valid/self-excluded/<=50m/nearest max8、K6/Tf12、1层message passing。Node15→64→64，raw47→64→64 message，hidden145→64→1 attention，LayerNorm(h_node+m_int)，64→32→1 score。Stage9无map branch或semantic inputs；G3的semantic计算仅用于独立冻结历史reference，不传入T1/T2/T3。\n\nParams: T1=24066，T2/T3=27515，新增均<40k。'),
      ('Type Adapter','三个target-type adapter共享graph backbone后各自64→8→64，ReLU，上投影weight/bias zero-init；h_type=h+A_type(h)。T2/T3共同adapter初始化逐位相同，seed2224；没有独立三个GNN。'),
      ('Reliability Gate','8→16→1/ReLU/Sigmoid，seed2225，actor的6mode使用同一个gate。输入只有V/P/B one-hot、历史recent/net/path motion、selected neighbor_count/8、nearest current neighbor_distance/50；无neighbor时distance=1，clip[0,1]，context不z-score。历史motion复用Stage8 HeadTrain-only node columns5/6/7统计。无manual type prior或后处理gate。'),
      ('Balanced Objective','T1/T2 micro SoftCE；T3 L=.5 micro+.5 present-type macro。q=softmax(-FDE/1m)。macro按完整1024target batch中的存在类型计算，通过exact type count权重积累128×8microbatch，不按microbatch各自macro平均。HeadTrain样本不重采样。\n\nT3 exact first formal batch half-macro gradient norm='+f"{macroaudit['half_macro_gradient_norm']:.6g}"+'，finite/nonzero；诊断本身optimizer更新0。每epoch type counts/loss/contribution及micro/macro见训练曲线。'),
      ('No Leakage','TRAIN16-window GT=NaN、future/target masks翻转、semantic字段NaN后，observable graph/gate逐位不变。future graph仅使用冻结预测candidate；GT仅作ranking监督和离线评价。HeadTrain630/HeadDev70严格复用Stage6A split；Stage8 normalization仅读取future node/interaction统计，未读semantic arrays。TRAIN cache identity join逐位核对FDE与original logits；fresh R2输入验证maxdiff<1e-6。未用test。'),
      ('Tiny Overfit','固定128 HeadTrain targets/seed2022/300 FP32 AdamW updates，各variant独立。T3 entropy floor按同一balanced权重计算；不是使用micro entropy代替macro下界。ExcessLossReduction>=90%。所有predictor/R2 grad0、graph grads finite/nonzero、候选不需梯度。tiny权重丢弃；正式训练重置原初始化。\n\n'+md(pd.DataFrame(tiny['results'])[['Variant','status','InitialLoss','FinalLoss','EntropyFloor','InitialExcessLoss','FinalExcessLoss','ExcessLossReduction','Updates']])+'\n\n'+md(table('neutral_identity'))),
      ('Formal Training','T1→T2→T3；每个从独立冻结初始化开始，无warm-start。AdamW1e-3/weight_decay1e-4，effective1024，max50/patience5，FP32，epoch余数carry且无短optimizer batch。无AMP、无variant搜索。\n\n'+md(training)+'\n\n![Training curves](../05_diagnostics/stage9a_training_curves.png)'),
      ('Checkpoint Selection','三者都只按DevSelectionScore=.5 OverallDevSoftCE+.5 MacroTypeDevSoftCE的strict minimum选checkpoint。Dev macro按整个HeadDev全体targets汇总，各存在类型平均。Top1FDE仅记录。全部freeze SHA后才跑Stage9 secondary VAL。\n\n'+md(pd.DataFrame(freeze['checkpoints']))),
      ('Candidate Identity','PASS。fresh Stage5A预测所有VAL source windows与冻结cache raw/ego逐位相同；七个ranking共享一个候选tensor。candidate/minADE6/minFDE6/MR6 maxdiff=0，54990 full targets/150 scenes。沿用原指标约定：minADE6是minFDE-selected mode的ADE，MR6=minFDE>2m；不把ranking收益称为candidate geometry改善。'),
      ('Main Results',md(main)+'\n\n'+pairtext('T3','R2')+'\n\n'+proof),
      ('R2',md(main[main.Model=='R2'])+'\n冻结baseline，Stage9始终锚定其logits。'),
      ('T1 Shared Residual Graph',md(main[main.Model=='T1'])+'\n\n'+pairtext('T1','R2')+'\nResidualGraph='+decision['ResidualGraph']),
      ('T2 Type-Adaptive Graph',md(main[main.Model=='T2'])+'\n\n'+pairtext('T2','T1','Pedestrian')+'\nTypeAdaptation='+decision['TypeAdaptation']),
      ('T3 TA-FIG',md(main[main.Model=='T3'])+'\n\n'+pairtext('T3','T2','Pedestrian')+'\nBalancedTraining='+decision['BalancedTraining'])]
    for title,g in (('Vehicle','Vehicle'),('Pedestrian','Pedestrian'),('Bicycle','Bicycle'),('Moving Vehicle','MovingVehicle')):
        df=motion if g=='MovingVehicle' else types;sections.append((title,md(df[df.Group==g][columns])+'\n\n'+pairtext('T3','R2',g)))
    sections.extend([
      ('Interaction Context','evaluation前冻结count0/1-2/3-5/6-8，距离[0,5)/[5,10)/[10,20)/[20,50]m。无邻居target归Neighbor0，distance bins排除；不改变模型输入或筛样。\n\n'+md(context[columns])+'\n\n其余motion/GT-displacement分组见 [motion table](../06_tables/stage9a_motion_results.csv)。'),
      ('Bootstrap','paired whole-scene cluster，150scene/1000replicates/seed2022；全pair/group共享resampling indices，actor sums与counts随完整scene一起重复，percentile95% CI。Δ=new−baseline，负FDE改善。除指定六比较外，T2−T1和T3−G3用于要求的机制/reference比较，未加variant。subgroup CIs未作multiplicity correction，groups重叠，属于development reuse secondary evidence。\n\n'+md(ci[(ci.Group=='Overall')&(ci.Metric=='Top1FDE')])+'\n\n'+md(ci[(ci.New=='T3')&(ci.Baseline=='R2')&(ci.Metric=='Top1FDE')&(ci.Group.isin(['Vehicle','Pedestrian','Bicycle','MovingVehicle','Vehicle>5m','Pedestrian<5m','Pedestrian>5m']))])),
      ('Vehicle Benefit Retention',f"VehicleRetentionRatio={ratios['VehicleRetentionRatio']}；冻结target>=.70。R2−G1与R2−T3转为正向improvement后相除；非正分母不宣布保留成功。VehicleBenefitRetained="+decision['VehicleBenefitRetained']),
      ('Pedestrian Recovery',f"G1PedDegradation={ratios['G1PedDegradation']:.6f}m；T3PedDegradation={ratios['T3PedDegradation']:.6f}m；RecoveryRatio={ratios['PedestrianRecoveryRatio']}。>=.8仅描述性工程目标。负迁移resolved的冻结条件为T3−R2 point<=+.01m且CI不可靠恶化；不由单一recovery ratio决定统计支持。PedestrianNegativeTransferResolved="+decision['PedestrianNegativeTransferResolved']),
      ('Gate Diagnostics',md(gates)+'\nT1 gate固定1，T2/T3来自各自训练网络。分布只描述残差使用强度；gate大小不能证明某类型更依赖交互，也未据gate结果做手工覆写。'),
      ('Mode Change',md(mode)+'\nmode是否FDE-best只作ranking oracle；不是行为预测accuracy。'),
      ('Cases','aggregate decision之后按冻结条件选distinct instances；largest requested-category FDE effect，lexical identity tie-break；case3 high interaction=Neighbor6-8，high gate=Vehicle gate top quartile。cases不能代替bootstrap或证明因果。\n\n'+'\n\n'.join('!['+x['case']+'](../07_cases/stage9a_'+x['case']+'.png)' for x in case['cases'])+'\n\nUnavailable categories: '+json.dumps(case['unavailable'],ensure_ascii=False)),
      ('Efficiency',md(eff)+'\n100warmup/500measured，batch128，RTX3080 FP32。Stage9 combined rows包含冻结R2 forward与新graph residual；另外分开报告graph residual only与gate only。Params列为新graph参数，FrozenR2Params单列673个冻结base参数。prepared normalized CUDA输入，不包含frozen predictor、graph construction或transfer。gate-only耗时已经包含在graph和combined rows中，不能重复相加；不测semantic retrieval，Stage9无map graph。'),
      ('Limitations',proof+'\n\n单seed、单训练run；HeadDev Bicycle158 targets导致macro selection可能较噪。bootstrap表征scene采样不确定性，不能消除已用VAL启发假设的适应性。type adapter与gate联合消融仅T1−T2，不归因于某一模块。T3−T2隔离固定平衡目标。连续的gate不是可信度校准或因果机制；R2 fallback只有gate=0才精确，不能称为自动保证无负迁移。full-horizon actor-window单位、case选择偏倚与overlapping groups均需保留。数据、checkpoint、完整actor表/cache仅本地。'),
      ('Scientific Decision','\n'.join(k+' = '+v for k,v in decision.items())+'\n\nSTRONG_PLUS='+str(ratios['STRONG_PLUS'])+'\n\n'+findings+'\n\n判据按00_manifest/stage9a_protocol.json执行，未见结果后修改。PaperUsableTAFIG=YES也仅表示值得进一步确认的development evidence；成功必须由大脑AI设计Stage9B TRAIN700 frozen scene-level CV，本阶段不执行Stage9B。失败STOP，不调gate/rank/radius/loss/seed、不加semantic、不重训predictor。')])
    output=ROOT/'09_reports/stage9a_final_report.md';output.write_text('\n\n'.join('【'+title+'】\n\n'+body for title,body in sections)+'\n')
    atomic_json(ROOT/'09_reports/stage9a_final_audit.json',{'status':'COMPLETE','historical_files_unchanged':len(history['tracked_history']),'candidate_identity':'PASS','tiny_all_PASS':True,'formal_checkpoints_frozen':True,'training_sources_unchanged':True,'Stage9_semantic_inputs':False,'predictor_R2_gradients':0,'Official_VAL_role':'development-reuse evidence','Stage9B_executed':False,'test_used':False,'scientific_decision':decision,'final_report_sha256':sha256(output),'STOP':True})
    print('STAGE9_COMPLETE_STOP',decision,flush=True)

if __name__=='__main__':main()
