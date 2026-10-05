"""Evidence-grounded Stage4A decision/report; stops before Stage4B."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'00_manifest'))
from stage4a_common import SUMMARY,PREREG,CONFIG,read_json,atomic_json,sha256

def rows(name):
    with (ROOT/'06_tables'/name).open() as handle:return list(csv.DictReader(handle))

def f(value):return f'{float(value):.6f}'

def table(report,header,data):
    lines=['| '+' | '.join(header)+' |','| '+' | '.join('---' for _ in header)+' |']
    lines.extend('| '+' | '.join(str(v) for v in row)+' |' for row in data)
    report.append('\n'.join(lines))

def decide(ci,prereg):
    groups=ci['groups'];overall=groups['overall']['minFDE6'];lower,upper=overall['CI95_m']
    harmed=[c for c in prereg['major_class_degradation_guard']['classes'] if groups[c]['minFDE6']['CI95_m'][0]>0]
    if harmed:return 'NOT SUPPORTED','Preregistered reliable major-class degradation guard triggered: '+', '.join(harmed),harmed
    if lower>0:return 'NOT SUPPORTED','Overall deltaFDE has its CI95 entirely above0.',harmed
    if overall['estimate_m']<0 and upper<0:
        return 'SUPPORTED','Overall deltaFDE<0 and CI95 upper<0; no reliable major-class degradation.',harmed
    improved=[g for g in prereg['scientific_decision']['interaction_relevant_groups']
              if groups[g]['minFDE6']['estimate_m']<0 and groups[g]['minFDE6']['CI95_m'][1]<0]
    if lower<=0<=upper and improved:
        return 'PARTIAL','Overall CI95 includes0; reliable improvement in '+', '.join(improved)+'; no reliable overall/major-class harm.',harmed
    return 'NOT SUPPORTED','Neither primary overall nor preregistered interaction-relevant groups have reliable improvement.',harmed

def main():
    training=read_json(SUMMARY);prereg=read_json(PREREG)
    audit=read_json(ROOT/'00_manifest/stage4a_final_protocol_audit.json');assert audit['status']=='PASS'
    visual=read_json(ROOT/'00_manifest/stage4a_visual_QA.json');assert visual['status']=='PASS'
    exports=read_json(ROOT/'00_manifest/stage4a_export_QA.json');assert exports['status']=='PASS'
    ci=read_json(ROOT/'04_evaluation/stage4a_bootstrap_ci.json');assert ci['status']=='PASS'
    pairing=read_json(ROOT/'04_evaluation/stage4a_pairing_audit.json');assert pairing['status']=='PASS'
    efficiency=read_json(ROOT/'04_evaluation/stage4a_efficiency_audit.json');assert efficiency['status']=='PASS'
    cases=read_json(ROOT/'04_evaluation/stage4a_qualitative_case_manifest.json');assert cases['status']=='PASS'
    init=read_json(ROOT/'00_manifest/stage4a_initialization_audit.json')
    fresh=read_json(ROOT/'04_evaluation/stage4a_metrics.json');assert fresh['status']=='PASS'
    atomic_json(ROOT/'07_checkpoints/stage4a_best_checkpoint_manifest.json',{**training,
        'phase':fresh['checkpoint_metadata']['phase'],'global_step':training['best_global_step'],
        'parameter_count':init['total_parameter_count'],'additional_parameter_count':init['additional_parameter_count'],
        'shared_parameter_count':init['shared_parameter_count'],
        'relative_parameter_increase_percent':init['relative_parameter_increase_percent'],
        'final_fresh_full_horizon_metrics':fresh['metrics']['full_horizon']})
    neutral=read_json(ROOT/'00_manifest/stage4a_neutral_initialization_audit.json')
    subgroup=read_json(ROOT/'04_evaluation/stage4a_interaction_subgroup_audit.json')
    ablation=rows('stage4a_type_interaction_ablation.csv')
    motion=rows('stage4a_nontrivial_motion_metrics.csv')
    support,reason,harmed=decide(ci,prereg);ready='YES' if support in ('SUPPORTED','PARTIAL') and not harmed else 'NO'
    decision={'status':'PASS','Type_conditioned_Interaction':support,'Reason':reason,
        'Stage4A':'PASS','Ready_Stage4B':ready,'major_class_harm_guard_triggered':harmed,
        'technical_PASS_independent_of_effect_direction':True,'preregistered_rules_sha256':sha256(PREREG),
        'primary_endpoint':prereg['primary_endpoint'],'training':training,'ablation':ablation,
        'bootstrap':ci['groups'],'efficiency':efficiency,'Stage4B_executed':False,
        'Ready_does_not_authorize_Stage4B_execution':True}
    atomic_json(ROOT/'04_evaluation/stage4a_scientific_decision.json',decision)
    report=['# Stage4A：Type-Conditioned Interaction','【Research Question】',
        '在固定节点级TypeEmbedding之后，有向source-target类型条件关系bias是否进一步改善交互表示和预测误差？本轮仅检验这个变量。',
        '【Model Change】','Type pair definition=pair_id=3*target_type+source_type；PyG edge[0]=source j、edge[1]=target i；V0/P1/B2。',
        '9有向关系=V<-V、V<-P、V<-B、P<-V、P<-P、P<-B、B<-V、B<-P、B<-B，V<-P与P<-V不同。',
        'Pair embedding dim=16；nn.Embedding(9,16)。',
        'Relation feature=[E_pair, target-frame(source−target) position/50m, cos(source_angle−target_angle), sin(source_angle−target_angle)]，20维。',
        'Bias MLP=Linear(20,32)→ReLU→Linear(32,24)；reshape[E,3,8]；最后weight/bias均zero-init。',
        'Insertion position=原agent-agent attention score之后、softmax之前加bias；保留原rel_embed、q/k/v、gate、FFN、LayerNorm、projection及value aggregation。',
        '未直接修改runtime GlobalInteractor。LocalEncoder、TypeEmbedding、Decoder的结构及loss定义不变，Stage4A全部参数从头训练；拓扑、节点、map及数据冻结，无lane-node relation bias。',
        f"Stage3B params={init['Stage3B_shared_parameter_count']}；Stage4A params={init['total_parameter_count']}；Additional params={init['additional_parameter_count']}；relative increase={init['relative_parameter_increase_percent']:.6f}%。",
        '【Initialization Audit】',f"shared diff={init['max_shared_parameter_diff']}；shared state SHA256={init['shared_state_SHA256']}。",
        f"step0 raw prediction diff={neutral['raw_prediction_max_abs_diff']:.9g}；logit diff={neutral['mode_logits_max_abs_diff']:.9g}；prob diff={neutral['mode_prob_max_abs_diff']:.9g}，均<1e-6。",
        'canonical Stage3B seed2022 step0逐name/shape完全一致；从头训练，无训练后Stage3B权重加载。12unit tests、future扰动不变性、10update梯度、原六TRAIN-window tiny均PASS，tiny模型丢弃。',
        '新增relation模块中首反传仅final MLP有非零梯度符合zero-init设计；pair embedding与first MLP在10update内非零。',
        '【Training】',f"warmup steps={training['warmup_steps']}；NLL steps={training['NLL_steps']}；executed global step={training['final_executed_global_step']}。",
        f"best global step={training['best_global_step']}；stop reason={training['stop_reason']}；converged_by_patience={training['converged_by_patience']}；stopped_by_budget={training['stopped_by_budget']}。",
        f"恢复本模型自己的warmup best step={training['warmup_best_source_step']}的model/AdamW/RNG；仅LR0.001→0.0001并依冻结Stage3B重置NLL sampler cursor。",
        'warmup固定5000，NLL最多16000/global最多21000；每500完整official VAL150；post-update originalNLL overall full-horizon minFDE6 strict improvement；patience5。',
        '只增加五项配置；复制YAML历史预算字段保留，实际执行预算与冻结Stage3B注册/执行一致。seed2022/batch16/Th5/Tf12/K6/embed64/heads8/global3/dropout0.1/radius50/weight_decay1e-4。',
        'official train700/val150，850shard逐SHA核验；VAL3603有监督window，test unused，未重新预处理，ego仅坐标参考。',
        '训练中断后从本模型NLL global5500 checkpoint恢复model、AdamW、CPU/CUDA RNG与sampler cursor，未保存的5600–5700更新重放。原协议未启用确定性算法，GPU重放不保证逐位一致；恢复记录见 ../00_manifest/stage4a_resume_record.json。',
        f"训练代码commit={training['training_code_git_commit']}；config SHA256={sha256(CONFIG)}；best checkpoint SHA256={training['checkpoint_sha256']}。",
        '【Main Results】','所有正式数字来自最终best checkpoint重新完整VAL推理，NaN/Inf=0。Delta=C(Stage4A)−B(Stage3B)，负值更好；meters。minADE6是best-FDE mode的ADE；MR为endpoint>2m；Top1为argmax概率；原LaplaceNLL定义不变。']
    table(report,['Group','Count','B ADE','C ADE','ΔADE','B FDE','C FDE','ΔFDE','B MR','C MR','B Top1FDE','C Top1FDE'],
        [[r['Group'],r['Count']]+[f(r[k]) for k in ('B_minADE','C_minADE','Delta_ADE','B_minFDE','C_minFDE','Delta_FDE','B_MR','C_MR','B_Top1FDE','C_Top1FDE')] for r in ablation if r['Group'] in ('Overall','Vehicle','Pedestrian','Bicycle','vehicle.moving','vehicle.stopped','vehicle.parked','unknown')])
    report.extend(['【Nontrivial Motion】','严格GT endpoint displacement阈值；Count为重叠actor-window，unique实例/scene另报。'])
    table(report,['Group','Count','Instances','Scenes','minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6'],
        [[r[k] for k in ('Group','Count','UniqueInstances','UniqueScenes')]+[f(r[k]) for k in ('minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6')] for r in motion])
    table(report,['Group','B FDE','C FDE','ΔFDE'],[[r['Group']]+[f(r[k]) for k in ('B_minFDE','C_minFDE','Delta_FDE')] for r in ablation if ' >' in r['Group']])
    report.append('Bicycle >5m只有132个actor-window、12unique实例、11scene；仅描述性解释，不作强改善或因果论断。')
    report.extend(['【Interaction-sensitive Results】','20m阈值及membership在任何正式optimizer update之前固定；只用t0 positions/types/current valid distinct actors，无未来GT或预测筛组。Heterogeneous-20m含不同type邻居；VP-context限定vehicle/pedestrian双向。',
        f"membership ledger SHA256={subgroup['membership_sha256']}；全85027actor身份已登记，完整GT/mask未用于membership。"])
    table(report,['Group','Count','Instances','Scenes','B FDE','C FDE','ΔFDE'],
        [[r['Group']]+[subgroup['counts']['full_horizon'][r['Group']][k] for k in ('Count','UniqueInstances','UniqueScenes')]+[f(r[k]) for k in ('B_minFDE','C_minFDE','Delta_FDE')] for r in ablation if r['Group'] in subgroup['counts']['full_horizon']])
    report.extend(['【Bootstrap】','严格配对full54990/partial30037/total85027；scene/sample/instance/node/type/motion/全部12步GT哈希及future mask均一致。',
        'paired scene-cluster percentile bootstrap：150official VAL scenes、1000replicates、seed2022；按scene重采样并pool窗口delta，所有组同draws。CI仅衡量固定两模型的scene采样变异，不涵盖训练seed变异；辅助子组未经多重比较校正。'])
    table(report,['Group','Metric','Δ C−B (m)','95%CI (m)'],
        [[group,metric,f(v['estimate_m']),f"[{f(v['CI95_m'][0])}, {f(v['CI95_m'][1])}]"] for group,metrics in ci['groups'].items() for metric,v in metrics.items()])
    report.extend(['【Relation Bias Analysis】','9directedpairs分别按3layer×8head记录edge observation count、mean、std、mean absolute bias；heatmap聚合layer/head的mean|bias|并保留source CSV。',
        '详见 ../06_tables/stage4a_relation_bias_statistics.csv 和 ../04_evaluation/stage4a_relation_bias_statistics.json。bias/attention仅模型内部诊断，不代表因果影响或某交通类别物理上更重要。',
        ])
    pair_aggregate=rows('stage4a_relation_bias_pair_aggregate.csv')
    table(report,['Directed pair','Edge observations','Mean bias','Std bias','Mean absolute bias'],
        [[r['directed_pair'],r['edge_observation_count']]+[f(r[k]) if r[k] else 'NA' for k in ('mean_bias','std_bias','mean_abs_bias')] for r in pair_aggregate])
    report.extend(['【Efficiency】','相同GPU、真实VAL输入、batch16、evalmode与预热；500个配对timed batch observations，另隔离各模型测CUDA peak memory；source timings及统计口径见efficiency audit。'])
    table(report,['Model','Params','Mean inference (ms)','Median inference (ms)','Peak CUDA allocated (MiB)'],
        [[name,v.get('parameter_count',v.get('parameters','see audit')),f(v['mean_inference_ms']),f(v.get('median_inference_ms',v.get('median_ms'))),f(v['peak_CUDA_memory_MiB'])] for name,v in efficiency['models'].items()])
    report.extend([f"Inference overhead={efficiency['overhead_percent']:.6f}%；additional parameters={init['additional_parameter_count']}（{init['relative_parameter_increase_percent']:.6f}%）。",
        '计时范围、个体window/batch数、warm-up与峰值显存定义均以audit为准，参数量与实际运行开销分别报告。',
        '【Qualitative Cases】','同actor/scene/GT，左右Stage3B/Stage4A同axis/equalaspect；每条future显示12markers；仅绘History/GT/Best-FDE/Top1。案例误差对sourceCSV重新计算<1e-4，inset只按最后6个真实点bbox必要时启用。',
        'vehicle/pedestrian improvement来自预注册heterogeneous20m群组，degradation真实展示局限；V-P neighbor仅淡色history/GT。t0 spatial interaction context不等价于真实因果交互；案例有目的选取不能代表整体频率。'])
    for item in cases.get('figures',[]):
        report.append(f"{item.get('case_kind',item.get('kind','case'))}={item['name']}；source=../{item['source_json']}。")
    report.extend(['【Scientific Decision】',f'Type-conditioned Interaction={support}。',f'Reason={reason}',
        '预注册主要类别异常退化守门=任一V/P/B overall FDE配对CI95 lower>0；B仅保守守门，不作强改善结论。',
        f'Stage4A=PASS；Ready for Stage4B Reliability={ready}。',
        'PASS仅表示实现、初始化、训练、冻结、配对、效率、图与报告流程正确，与效果方向独立。Ready需要SUPPORTED/PARTIAL且无overall/majorclass可靠退化。',
        '本轮STOP，未执行Stage4B Reliability Head、Intent、TTC、future compatibility、mode reranking、校准、新图拓扑、额外seed、超参搜索、class-balanced loss、oversampling或main合并。',
        '【Audit and Artifacts】',f"frozen protected files={audit['frozen_prior_files_verified']}；frozen scene shards={audit['frozen_scene_shards_verified']}；实际完整VAL点={audit['all_actual500step_VAL_points_verified']}。",
        'Stage3A/Stage3B结果与原runtime文件保持SHA一致；旧Stage2C五个未跟踪重绘文件保留且不提交。大checkpoint、actorCSV、membership ledger、frozen数据不上传Git；小代码/报告/真实图表上传stage4a/type-conditioned-interaction。'])
    (ROOT/'09_reports/stage4a_final_report.md').write_text('\n\n'.join(report)+'\n')
    print('STAGE4A_FINAL_REPORT',support,'PASS','Ready_Stage4B',ready,flush=True)
    manifest=ROOT/'00_manifest/stage4a_artifact_manifest.json';artifacts=[]
    for p in sorted(ROOT.rglob('stage4a_*')):
        if not p.is_file() or p==manifest or '__pycache__' in p.parts or p.suffix in ('.pyc','.tmp') or 'stage3_cache' in p.parts:continue
        local=p.suffix in ('.pt','.pth','.ckpt') or p.name in ('stage4a_actor_errors.csv','stage4a_interaction_membership.csv')
        artifacts.append({'relative_path':str(p.relative_to(ROOT)),'file_size':p.stat().st_size,
            'sha256':sha256(p),'created_by_stage':'Stage4A','git_delivery':'local_only' if local else 'version_control'})
    atomic_json(manifest,{'stage':'Stage4A','status':'PASS','root':str(ROOT),'self_hash_excluded':True,'artifacts':artifacts})

if __name__=='__main__':main()
