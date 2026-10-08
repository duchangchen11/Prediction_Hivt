"""Document actual tiny-gate stop, without inventing formal VAL results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'01_training/00_code'))
from stage8a1_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    verify();stop=read_json(ROOT/'01_training/stage8a1_stop_gate.json');assert stop['status']=='STOP_BEFORE_FORMAL_TRAINING'
    feasibility=read_json(ROOT/'01_training/stage8a1_tiny_feasibility.json');rows=[]
    fig,ax=plt.subplots(figsize=(7,4))
    for v in PARAMS:
        tiny=stop['tiny'][v];root=ROOT/'01_training'/v
        rows.append({'Variant':v,'Status':tiny['status'],'Updates':300,'Precision':'FP32','Targets':128,
            'InitialLoss':tiny['initial_loss'],'FinalLoss':tiny['final_loss'],'RelativeDecrease':tiny['relative_decrease'],
            'RequiredDecrease':.2,'EntropyLowerBound':feasibility['soft_target_entropy_lower_bound'],
            'RequiredLossBelow':feasibility['required_final_loss_strictly_below'],'FinalKLAboveEntropy':tiny['final_loss']-feasibility['soft_target_entropy_lower_bound'],
            'NonzeroDeltaAbsMax':tiny['delta_abs_max'],'PredictorGradientCount':0,'FormalUpdates':0})
        with (root/'stage8a1_tiny_curve.csv').open() as f:curve=list(csv.DictReader(f))
        ax.plot([0]+[int(r['Update']) for r in curve],[tiny['initial_loss']]+[float(r['Loss']) for r in curve],label=v,lw=1.5)
        atomic_json(root/'normalization_manifest.json',read_json(NORM))
        atomic_json(root/'training_config.json',{'status':'FORMAL_TRAINING_NOT_STARTED_TINY_GATE_FAILED',
            'variant':v,'seed':2022,'tiny_FP32_updates':300,'planned_optimizer':'AdamW','planned_lr':.001,'planned_weight_decay':.0001,
            'planned_max_epochs':50,'planned_patience':5,'planned_microbatch':128,'planned_accumulation':8,'planned_effective_batch':1024,
            'planned_checkpoint_selection':'minimum HeadDev ranking loss','normalization_sha256':sha256(NORM),
            'initial_state_sha256':read_json(ROOT/'00_manifest/stage8a1_registration.json')['initial_state_sha256'][v]})
    write_csv(ROOT/'06_tables/stage8a1_tiny_gate_results.csv',rows)
    ax.axhline(feasibility['soft_target_entropy_lower_bound'],color='black',ls='--',label='Soft-target entropy lower bound')
    ax.axhline(feasibility['required_final_loss_strictly_below'],color='#cc3311',ls=':',label='Required20% gate (<1.250974)')
    ax.set_xlabel('Optimizer update');ax.set_ylabel('Soft ranking cross-entropy');ax.set_title('Stage8A-1 fixed128-target tiny gates');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(ROOT/'05_diagnostics/stage8a1_tiny_gate.png',dpi=160);fig.savefig(ROOT/'05_diagnostics/stage8a1_tiny_gate.svg');plt.close(fig)
    svg=ROOT/'05_diagnostics/stage8a1_tiny_gate.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    frozen=read_json(C/'00_manifest/stage8a0c_frozen_inputs.json');checked=0
    for p,h in frozen['files'].items():assert sha256(PROJECT/p)==h,p;checked+=1
    assert not FROZEN.exists()
    assert not any((ROOT/'01_training'/v/'best_dev_loss.pt').exists() for v in PARAMS)
    assert not (ROOT/'03_evaluation/stage8a1_val_registration.json').exists()
    atomic_json(ROOT/'09_reports/stage8a1_final_audit.json',{'status':'STOP_TINY_GATE_FAILURE','historical_frozen_files_verified':checked,
        'graph_spec_sha256':sha256(SPEC),'training_dependency_sources_unchanged':True,'full_training_updates':0,
        'formal_checkpoints':0,'official_VAL_passes':0,'test_used':False,'data_or_environment_downloads':0,
        'architecture_selector_hyperparameters_unchanged':True,'predictor_gradient_count':0,'tiny_weights_discarded':True,
        'tiny_candidate_identity':'PASS on1024 real HeadTrain targets pervariant','formal_VAL_candidate_identity':'NOT_EVALUATED',
        'next_stage_authorized':False})
    sections=[('Research Question','比较 learned mode-mode、semantic mode-map 与 joint FSCG 的 ranking 效果，并以冻结 R2 为强基线。本阶段在 tiny 门槛停止，研究问题尚未得到正式评估。'),
        ('Frozen Inputs',f'基准 commit `{BASE}`。Stage5A SHA `88fe3feb59b7e830ec1e40aa917484386e0d2799d0f6116f410adda2d97fdce7`；R2 SHA `e3de457d635cd30c629ce316d73a87e419a3ded8abbe0e1f2601f1071527e314`。历史冻结文件复核 {checked} 个均未变化。'),
        ('Graph Specification','0C GraphSpecFrozen=YES；原 selector/radius/quota/features/neighbor/hidden/temperature 未修改。参数 G1=24066、G2=20546、G3=37187。原 message 是 raw15+raw15+17=47D，attention 是 hidden64+hidden64+17=145D；遵循冻结规范与参数数量。无 map edges 时仍为 exact zero64D。'),
        ('Training Protocol','沿用 seed2022 的630 HeadTrain/70 HeadDev scenes；full targets 分别260151/29934。仅完成三个 FP32 tiny，各固定同一128 targets、300 AdamW 更新、lr1e-3、weight_decay1e-4。G1 FAIL 后没有再优化 G1；按逐 variant 的 tiny 要求完成 G2/G3，随后全部 STOP。未全量训练，未改 sample、loss、temperature、seed 或模型。全量训练代码已准备，但未经过实际全量/VAL运行验证。'),
        ('Normalization','只统计 HeadTrain valid packed contexts；node continuous3..14；interaction0..10；map tangent15/16仅 lane-valid；map edge0..7/9，heading仅 lane-valid、inside fraction仅 polygon-valid。one-hot/binary/valid flags不标准化，padding/无效方向保留0。实测列均值接近0、std接近1，见 normalization audit；HeadDev和VAL不参与。新adapter16 TRAIN-window GT NaN/future-mask/target-mask poison逐位一致；冻结0C另有200-window检查。'),
        ('Tiny Overfit','\n\n|Variant|Initial|Final|Decrease|Gate|\n|---|---:|---:|---:|---|\n'+''.join(f"|{r['Variant']}|{r['InitialLoss']:.9f}|{r['FinalLoss']:.9f}|{100*r['RelativeDecrease']:.4f}%|FAIL|\n" for r in rows)+
            f"\n该样本 soft-target entropy H(q)={feasibility['soft_target_entropy_lower_bound']:.9f}；要求 L<0.8L0={feasibility['required_final_loss_strictly_below']:.9f}。由 L=H(q)+KL(q||p)>=H(q)，理论最大下降仅{100*feasibility['maximum_mathematically_possible_relative_reduction']:.4f}%，故固定样本的20%门槛不可满足。不能因此自行放宽门槛或改样本。loss已接近下界，但正式 Tiny 判定仍是FAIL。三个 delta 非零/finite，graph gradients finite/nonzero，predictor gradient=0。\n\n![Tiny curves](../05_diagnostics/stage8a1_tiny_gate.png)"),
        ('Checkpoint Selection','未执行 HeadDev checkpoint selection；没有 best_dev_loss.pt、last.pt 或正式 checkpoint manifest。未以VAL或HeadDev FDE选择任何模型。'),
        ('Candidate Identity','训练前1024真实 HeadTrain targets、每个 variant 的 logit/probability maxdiff=0，candidate maxdiff=0，attention instrumentation maxdiff=0。此 PASS 仅适用于训练前 neutral audit；正式VAL CandidateIdentity未评估。')]
    pending=['Main Results','R0','R2','G1 Future Interaction Graph','G2 Semantic Compatibility Graph','G3 Joint FSCG','G1 vs R2','G3 vs R2',
        'Vehicle','Pedestrian','Bicycle','Moving Vehicle','Semantic-sensitive Groups','Map Availability','Bootstrap','Mode Change Analysis','Interaction Diagnostics','Semantic Diagnostics','Cases','Efficiency']
    for title in pending:
        sections.append((title,'未执行：tiny gate FAIL，正式训练与checkpoint freeze均未进入；未生成正式指标、CI或案例/效率结论。历史CSV不充当本阶段评估结果。'))
    sections.append(('Limitations','本阶段目前只有工程、normalization、neutral identity、gradient 与tiny证据。不能据此判断 G1/G2/G3 科学有效性或 FSCG 相对R2优势。20%总SoftCE下降门槛在该固定样本上低于数学下界，是当前阻断原因；不擅自替换为 excess-loss/KL 门槛，不进行失败驱动重采样。缓存约16GB仅本地，未上传；评价/诊断代码只有静态语法检查，尚无正式运行验证。'))
    sections.append(('Scientific Decision','Stage8A-1 status=STOP_BEFORE_FORMAL_TRAINING。AgentGraph、LearnedGraphAdvantageOverR2、SemanticGraph、SemanticContributionToJoint、FSCG、JointComplementarity、PaperUsableFSCG、RecommendedFinalModel均为NOT_EVALUATED，不能填入未经评估的SUPPORTED/NOT_SUPPORTED。现有冻结R2结论保持。本次STOP，等待大脑AI审查；不开始下一阶段。'))
    (ROOT/'09_reports/stage8a1_final_report.md').write_text('\n\n'.join('【'+title+'】\n\n'+body for title,body in sections)+'\n')
    print('FINAL_STOP_AUDIT_PASS',checked,flush=True)

if __name__=='__main__':main()
