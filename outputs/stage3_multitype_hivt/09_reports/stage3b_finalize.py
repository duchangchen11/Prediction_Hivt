"""Evidence-first final Stage3B decision and report; never starts another stage."""
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'00_manifest'))
from stage3b_common import read_json,atomic_json,sha256,SUMMARY,PREREG,PROJECT,git

def csv_rows(name):
    with (ROOT/'06_tables'/name).open() as f:return list(csv.DictReader(f))

def f(value):return f'{float(value):.6f}'

def decide(ci):
    overall=ci['groups']['overall']['minFDE6'];lower,upper=overall['CI95_m']
    if overall['estimate_m']<0 and upper<0:return 'SUPPORTED','Overall delta FDE <0 and scene-bootstrap CI95 entirely <0.'
    subgroup=any(ci['groups'][group]['minFDE6']['estimate_m']<0 and
        ci['groups'][group]['minFDE6']['CI95_m'][1]<0 for group in ('vehicle','pedestrian'))
    if lower<=0<=upper and subgroup:return 'PARTIAL','Overall CI95 includes0; Vehicle or Pedestrian FDE has CI95 entirely <0, without significant overall harm.'
    return 'NOT SUPPORTED',('Overall delta FDE CI95 entirely >0.' if lower>0 else 'No reliable primary overall or main-class FDE improvement under the preregistered rule.')

def main():
    training=read_json(SUMMARY);prereg=read_json(PREREG)
    audit=read_json(ROOT/'00_manifest/stage3b_final_protocol_audit.json');assert audit['status']=='PASS'
    ci=read_json(ROOT/'04_evaluation/stage3b_bootstrap_ci.json');assert ci['status']=='PASS'
    visual=read_json(ROOT/'00_manifest/stage3b_visual_QA.json');assert visual['status']=='PASS'
    pairing=read_json(ROOT/'04_evaluation/stage3b_pairing_audit.json');assert pairing['status']=='PASS'
    metrics=read_json(ROOT/'04_evaluation/stage3b_type_embedding_metrics.json');assert metrics['status']=='PASS'
    cases=read_json(ROOT/'04_evaluation/stage3b_qualitative_case_manifest.json');assert cases['status']=='PASS'
    embedding=read_json(ROOT/'04_evaluation/stage3b_type_embedding_vectors.json')
    ablation=csv_rows('stage3b_type_embedding_ablation.csv');motion=csv_rows('stage3b_nontrivial_motion_metrics.csv')
    main_results=csv_rows('stage3b_type_embedding_main_results.csv')
    support,reason=decide(ci);stage='PASS';ready='YES' if support in ('SUPPORTED','PARTIAL') else 'NO'
    decision={'status':'PASS','TypeEmbedding':support,'Reason':reason,'Stage3B':stage,'Ready_Stage4':ready,
        'primary_endpoint':'Overall full-horizon minFDE6; Type-NoType','preregistered_rules_sha256':sha256(PREREG),
        'technical_PASS_independent_of_effect_direction':True,'Stage4_executed':False,
        'Ready_does_not_authorize_Stage4_execution':True,'training':training,
        'ablation':ablation,'nontrivial_motion':motion,'bootstrap':ci['groups']}
    atomic_json(ROOT/'04_evaluation/stage3b_scientific_decision.json',decision)
    report=['# Stage3B：Multi-Type HiVT + Type Embedding',
        '本轮仅检验 additive Type Embedding。全部正式比较来自相同冻结 official VAL，单次 seed2022 训练；结论限于本数据、协议和种子。',
        '【Model Change】',
        'Type embedding dim=64；nn.Embedding(3,64)，0 vehicle / 1 pedestrian / 2 bicycle。',
        'Insertion point=LocalEncoder 输出之后、GlobalInteractor 之前。',
        'Fusion=local actor embedding + E_type(agent_type)；decoder 使用增强后的 local representation。',
        'Additional parameter count=192；公共参数645809、总参数646001。',
        '共享 step0 初始化逐参数完全一致，max_abs_diff=0；未加载训练过的 No-Type 权重。A-A、TemporalEncoder、A-L、GlobalInteractor 公式和 decoder 保持原实现。',
        '【Training】',
        f"Warm-up steps={training['warmup_steps']}；NLL steps={training['NLL_steps']}；executed global step={training['final_executed_global_step']}。",
        f"Best global step={training['best_global_step']}；Stop reason={training['stop_reason']}。",
        f"converged_by_patience={training['converged_by_patience']}；stopped_by_budget={training['stopped_by_budget']}。",
        f"warm-up 执行5000更新，NLL恢复本模型 warm-up best step={training['warmup_best_source_step']} 的模型、AdamW 与 RNG；仅按原 Protocol1 将 LR 从0.001切换0.0001，NLL sampler cursor重置。",
        'batch16、Th5、Tf12、K6、embed64；每500更新完整150scene VAL，按 full-horizon overall minFDE6 严格改善选择 post-update NLL best；最大 global21000，patience5。',
        'frozen YAML 除 type_embedding=True 完全一致。原 YAML 的历史 max_steps 字段不改写，执行预算在训练前单独登记为5000+16000，与冻结 Stage3A 实际总预算一致。',
        '官方 train700 / val150；VAL3603有监督窗口（候选3619，空监督16）；test unused。相同850scene frozen shards，未重新预处理。',
        '曲线未经平滑；loss点是前500个真实batch更新的均值，VAL点均为当时完整 official VAL。',
        f"训练代码commit={training['training_code_git_commit']}；分支=stage3b/type-embedding；No-Type frozen commit={prereg['frozen_NoType_commit']}。",
        '【Main Results】',
        '指标为 meters；Delta=Type−No-Type，负值有利于 Type。minADE6沿用 best-FDE mode 的ADE，另存独立minADE诊断；MR6为endpoint error>2m；Top1为最高概率mode；NLL沿用original best-summed-L2 mode Laplace定义。',
        '| Group | Count | No-Type ADE | Type ADE | ΔADE | No-Type FDE | Type FDE | ΔFDE | No-Type MR | Type MR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in ablation[:5]:
        report.append('| '+' | '.join([r['Group'],r['Count']]+[f(r[k]) for k in ('NoType_minADE6','Type_minADE6','Delta_ADE','NoType_minFDE6','Type_minFDE6','Delta_FDE','NoType_MR6','Type_MR6')])+' |')
    report.extend(['完整主表含vehicle.stopped、vehicle.parked、unknown、Top1ADE6和NLL，见 ../06_tables/stage3b_type_embedding_main_results.csv。',
        '【Top1 Ranking】','| Group | No-Type Top1FDE6 | Type Top1FDE6 | ΔTop1FDE |','|---|---:|---:|---:|'])
    for r in ablation:
        report.append('| '+' | '.join([r['Group']]+[f(r[k]) for k in ('NoType_Top1FDE6','Type_Top1FDE6','Delta_Top1FDE')])+' |')
    oracle=float(ablation[0]['Delta_FDE']);ranking=float(ablation[0]['Delta_Top1FDE'])
    report.append(f'Overall oracle ΔFDE={oracle:.6f} m；Top1 ΔFDE={ranking:.6f} m。分别衡量候选轨迹覆盖误差与最高概率模式误差；Top1改善不替代预登记的primary oracle FDE支持判断。')
    top_interval=ci['groups']['overall']['Top1FDE6']['CI95_m']
    if top_interval[1]<0:interpretation='Top1FDE可靠下降，支持固定两模型下的最高概率模式误差改善。'
    elif top_interval[0]>0:interpretation='Top1FDE可靠上升，本轮未改善最高概率模式误差。'
    elif ranking<0:interpretation='Top1FDE描述性下降，但scene-cluster CI跨0，未获得可靠模式排序改善证据。'
    else:interpretation='Top1FDE未描述性改善，且scene-cluster CI跨0。'
    report.append(interpretation+' 该结论基于Top1与oracle误差对照，不能将Top1变化直接视为oracle coverage变化。')
    report.extend(['【Nontrivial Motion】','阈值严格按 GT endpoint displacement 分组，Count 为可能重叠的 actor-window；UniqueInstances / UniqueScenes 单独统计。',
        '| Group | Count | Instances | Scenes | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in motion:
        report.append('| '+' | '.join([r[k] for k in ('Group','Count','UniqueInstances','UniqueScenes')]+[f(r[k]) for k in ('minADE6','minFDE6','MR6','Top1ADE6','Top1FDE6')])+' |')
    report.append('Bicycle >5m：limited unique bicycle instances/scenes；132个重叠actor-window只有12个独立instances、11个scenes，仅描述性比较，不作强显著性结论。')
    report.extend(['有效运动FDE配对消融：','| Group | No-Type FDE | Type FDE | ΔFDE |','|---|---:|---:|---:|'])
    for r in ablation[5:]:
        report.append('| '+' | '.join([r['Group']]+[f(r[k]) for k in ('NoType_minFDE6','Type_minFDE6','Delta_FDE')])+' |')
    report.extend(['【Paired Bootstrap】',
        '严格配对full-horizon54990、partial30037，共85027。identity、node/type/motion/valid steps一致；GT float32完整trajectory SHA和future mask bits由未变更的No-Type冻结graph底账核对。No-Type旧CSV无逐时刻GT，因此不冒称CSV自带完整GT，底账来源与150VAL shard hashes保留。',
        'paired scene-cluster percentile bootstrap：150个official VAL scenes、1000 replicates、seed2022；每次重采样完整scene后pool全部actor-window delta，以窗口等权聚合。所有组使用同一scene draws；95%CI为2.5/97.5百分位。',
        'CI描述固定两模型在scene采样下的配对差异，未包括不同训练seed的变异；一个primary endpoint，其余预登记亚组为辅助证据，未作多重比较校正。',
        '| Group | Metric | Δ Type−No-Type (m) | Scene-cluster 95%CI (m) |','|---|---|---:|---|'])
    for group,values in ci['groups'].items():
        for metric,value in values.items():
            report.append(f"| {group} | {metric} | {f(value['estimate_m'])} | [{f(value['CI95_m'][0])}, {f(value['CI95_m'][1])}] |")
    report.extend(['【Qualitative】','案例按预登记改善/退化排序有目的选择，不能代表总体效果；同actor、同GT、同坐标范围，原VAL16-window batch重放，八项误差均与source CSV核验<1e-4。'])
    for item in cases['figures']:
        report.append(f"{item['case_kind']}={item['name']}；class={item['agent_type']}，ΔFDE={f(item['delta_FDE_m'])} m；source ../{item['source_json']}。")
    for item in cases['omitted_cases']:report.append(f"未生成 {item['kind']}：{item['reason']}")
    report.append('V-P图为t0<=20m且同步future GT最近距离<=5m的接近候选，不证明模型因果交互机制。所有future轨迹显示12个原始marker；inset仅用于转弯、GT接近或重合，bbox由两模型GT/Best/Top1最后6点共同计算。')
    report.extend(['【Embedding Auxiliary Analysis】',
        '三类训练后type vectors仅展示L2 norm与cosine；无t-SNE，不将embedding距离解释为物理语义。',
        '| Type | L2 norm |','|---|---:|'])
    for name,value in embedding['L2_norm'].items():report.append(f'| {name} | {f(value)} |')
    for i,j in ((0,1),(0,2),(1,2)):
        report.append(f"cosine({embedding['class_order'][i]}, {embedding['class_order'][j]})={f(embedding['cosine_similarity_matrix'][i][j])}。")
    report.extend(['【Scientific Decision】',f'Type Embedding={support}。',f'Reason={reason}',f'Stage3B={stage}。',
        f'Ready for Stage4 Type-aware Dynamic Interaction={ready}。',
        'Stage3B PASS 表示完整单变量协议、配对和真实证据通过验收，独立于效果正负。Ready仅按训练前保守规则生成，必须交用户审查；本轮在Stage3B结束，没有执行Stage4、type-aware动态交互、类别重权、Intent Head或main合并。',
        f"Overall FDE变化为{f(ablation[0]['Delta_FDE'])} m（{float(ablation[0]['Relative_FDE_change'])*100:.3f}%）。统计支持对应固定两模型的primary整体指标，不等价于所有运动类型均可靠改善。",
        f"vehicle.moving ΔFDE CI95={ci['groups']['vehicle.moving']['minFDE6']['CI95_m']}；Vehicle >5m ΔFDE CI95={ci['groups']['Vehicle >5m']['minFDE6']['CI95_m']}。其运动亚组解释依据这些CI是否跨0，不能用vehicle overall代替。",
        f"Pedestrian >5m ΔFDE CI95={ci['groups']['Pedestrian >5m']['minFDE6']['CI95_m']}；这是有效运动目标的单独证据。",
        f"Bicycle >5m 的ADE变化{f(ablation[-1]['Delta_ADE'])} m，FDE变化{f(ablation[-1]['Delta_FDE'])} m，Top1FDE变化{f(ablation[-1]['Delta_Top1FDE'])} m；结果并非各指标一致改善，且仅有12instances/11scenes，仍限描述性。",
        '【Audit and Artifacts】',
        f"原有受保护文件{audit['frozen_prior_files_verified']}份与scene shards{audit['frozen_scene_shards_verified']}份SHA全部未变更；核心训练源码SHA与预登记完全一致；{audit['all_actual500step_VAL_points_verified']}个完整VAL点核验通过。",
        '图件来自真实CSV/JSON，Python导出PNG300dpi/PDF/SVG，SVG/PDF保留可编辑文字；source与export hashes、markers、bbox和视觉审计归档。',
        '大型checkpoint、逐actor错误CSV、GT底账与原始shards仅本地保存，不上传Git。GitHub仅提交本阶段新代码、配置、报告、汇总表、小型source JSON、图件和运行日志，Stage2C未提交重绘文件原样保留。',
        f"Final checkpoint SHA256={training['checkpoint_sha256']}。",
        f"No-Type checkpoint SHA256={prereg['NoType_checkpoint_sha256']}。",
        f"Type config SHA256={training['config_sha256']}。",
        '本轮无额外训练seed或重复超参数实验，结论需在未来独立实验中检验泛化。'])
    lines=[]
    for paragraph in report:
        if lines and not (paragraph.startswith('|') and lines[-1].startswith('|')):lines.append('')
        lines.append(paragraph)
    (ROOT/'09_reports/stage3b_final_report.md').write_text('\n'.join(lines)+'\n')
    # Flush the completion line before hashing the redirected run log.
    print('STAGE3B_FINAL_REPORT=PASS','TypeEmbedding='+support,'Ready_Stage4='+ready,flush=True)
    artifacts=[];manifest=ROOT/'00_manifest/stage3b_artifact_manifest.json'
    for path in sorted(ROOT.rglob('stage3b_*')):
        if not path.is_file() or path==manifest or '__pycache__' in path.parts or path.suffix in ('.pyc','.tmp'):continue
        rel=str(path.relative_to(ROOT));local=path.suffix in ('.pt','.pth','.ckpt') or 'stage3_cache' in path.parts or path.name=='stage3b_type_embedding_actor_errors.csv'
        artifacts.append({'relative_path':rel,'file_size':path.stat().st_size,'sha256':sha256(path),
            'created_by_stage':'Stage3B','git_delivery':'local_only' if local else 'version_control'})
    atomic_json(manifest,{'stage':'Stage3B','status':'PASS','root':str(ROOT),'self_hash_excluded':True,'artifacts':artifacts})

if __name__=='__main__':main()
