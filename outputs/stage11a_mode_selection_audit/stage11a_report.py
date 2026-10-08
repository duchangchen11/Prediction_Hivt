"""Build the evidence-grounded report from verified frozen diagnostic tables."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'01_identity_audit'))
from stage11a_common import *

def table(frame,columns):
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in frame.iterrows():
        values=[]
        for c in columns:
            v=r[c];values.append(f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v))
        rows.append('| '+' | '.join(values)+' |')
    return '\n'.join(rows)

def main():
    verify();qa=read_json(ROOT/'01_identity_audit/stage11a_final_integrity.json');assert qa['Status']=='PASS'
    decision=read_json(ROOT/'09_reports/stage11a_decisions.json');evidence=decision['PedestrianMotionEvidence']
    m=pd.read_csv(ROOT/'05_loss_alignment/stage11a_all_model_group_metrics.csv')
    s=pd.read_csv(ROOT/'02_switch_regret/stage11a_switch_summary.csv')
    align=pd.read_csv(ROOT/'05_loss_alignment/stage11a_loss_regret_alignment.csv')
    ped=pd.read_csv(ROOT/'03_pedestrian_analysis/stage11a_pedestrian_motion.csv')
    soft=pd.read_csv(ROOT/'04_soft_label_analysis/stage11a_soft_label_groups.csv')
    geom=pd.read_csv(ROOT/'06_candidate_geometry/stage11a_geometry_groups.csv')
    train=pd.read_csv(ROOT/'07_training_distribution/stage11a_training_loss_gradient.csv')
    cases=pd.read_csv(ROOT/'08_cases/stage11a_case_selection.csv')
    md=m[m.Partition=='HeadDev'];sd=s[s.Group.isin(TYPES)];sf=soft[(soft.Partition=='HeadDev')&soft.Group.isin(TYPES)]
    metriccols=['Model','Count','SoftCE','Top1FDE','Top1ADE','OracleFDE','OracleGap','HitRate']
    sections=[f'''# Stage11A Mode Selection Regret and Loss Alignment Audit

本阶段完成冻结推理和离线诊断。主要证据是 SoftCE 与 Top1 模式选择的有限变化并不一致：DualExpert 在行人组降低 SoftCE，同时增加 Top1FDE。车辆切换的高代价尾部与行人错误切换次数增加是不同问题。候选质量也限制可达误差，不能将全部行人误差归因于排序。

基础 commit：`f6d80e233bd1fffcc0734d02a5aa6b55e9b21cca`。分支：`stage11a/mode-selection-loss-audit`。全部新增文件位于本阶段独立目录。

## Scope and evidence status

只使用官方 TRAIN700 中既有 HeadTrain630 / HeadDev70 划分，分别 260,151 / 29,934 个 full-horizon target。候选 K=6，历史 5 步，未来 12 步，0.5s 采样。未读取 official VAL 或 test；未训练、未修改模型/loss/checkpoint，optimizer update=0，新 checkpoint=0。原始历史文件和 5 个 Stage2C 未跟踪重绘文件保持原样。

本报告中的 **CONFIRMED** 表示直接测得的数值或冻结 checkpoint 在已使用 HeadDev 上的关联；**STRONGLY_SUGGESTED** 表示支持某种机制的描述性证据；**UNRESOLVED** 表示因果归因、历史共享参数梯度主导性或新目标的泛化收益仍未确定。不能把本报告的 CONFIRMED 当作独立统计确认或因果结论。

阈值和决策规则在统计前记录于 [protocol](../01_identity_audit/stage11a_protocol.json)，SHA256=`{qa['protocol_sha256']}`。本地输入、checkpoint 和既有报告哈希记录于 [frozen history](../01_identity_audit/stage11a_frozen_history.json)。HeadDev 已反复用于历史 checkpoint 选择和提出研究假设，登记规则不能消除这一复用影响。

## Code and label audit — CONFIRMED / PASS

SoftCE target 严格为 `q=softmax(-FDE/1m)`；Top1 为概率 argmax，BestMode 为 FDE argmin，平局取最小 mode index。所有 target 的 12 步未来均有效，最后有效 index=11；逐 target FDE/ADE 与冻结缓存逐位相同。GT/future mask/target mask 仅用于监督对象筛选、标签及离线评价，未作为模型特征。GT 与 mask poison 检查保持已选对象固定，对 observable 输入无影响。

一个公共候选 tensor 为所有 head 提供相同几何；全部 HeadDev 与 Stage10 predictor replay 逐位一致。DualExpert 的 Bicycle 输出与 R2 逐位一致。VehicleExpert/PedestrianExpert 仅在各自类型上定义；下表的专家指标不会假冒全类型 Overall 指标。

R0/R2/G1/G3 以及 T2/T3/专家/DualExpert 共 54 个历史 HeadDev 指标在 1e-6 容差内重现。HeadDev 已有 R0/R2/G1/Dual 输出直接复用；缺少的 G3/T2/T3 分数和 HeadTrain 分数由冻结 eval 模型生成。HeadTrain R2 的公共 batch128 与历史 batch512 anchor logits 最大差 9.5367431640625e-7；此差异不涉及几何，HeadDev 主诊断仍复用既有输出。

最终 57 项一致性检查 PASS；独立 SoftCE 重算最大差 {qa['independent_SoftCE_maxdiff']:.3g}，概率和最大偏差 {qa['probability_sum_maxdiff']:.3g}；16,898 个 TRAIN window 的 raw/ego predictor SHA 全部通过。输出 logits 解析梯度由孤立 logits 的 autograd 独立验证，冻结模型参数没有执行 backward。

详见 [code/labels](../01_identity_audit/stage11a_code_label_audit.csv)、[historical reproduction](../01_identity_audit/stage11a_historical_reproduction.csv)、[final checks](../01_identity_audit/stage11a_final_checks.csv)。

## Frozen HeadDev model comparison

所有距离指标单位 m；SoftCE 是 nats；两者未相减形成指标。OracleFDE 是相同六个候选的 minFDE，与模型分数无关。

{table(md[(md.Group=='Overall')&md.Model.isin(FULL_MODELS)],metriccols)}

行人组，PedestrianExpert 与 DualExpert 的行人路径相同：

{table(md[md.Group=='Pedestrian'],metriccols)}

Overall 的 G1/G3 改善不意味着所有类型改善。行人的排序恶化在 G1/G3/T2/T3/Stage10 上均有体现；这里只诊断冻结模型，不修改 Stage8/9/10 历史结论。

## Mode switch cost — CONFIRMED

正 delta 表示相对 R2 更差；net_delta_FDE 使用全组 actor 分母，gain/harm 的条件均值仅使用改善/恶化的切换。每条 switch 的记录包含身份、旧/新/oracle mode、三种 FDE、delta 及变化标记；大表留在本地。

{table(sd,['Group','Comparison','Count','changed_count','improved_count','worsened_count','mean_gain','mean_harm','median_gain','median_harm','net_delta_FDE','top10_harm_share'])}

DualExpert 车辆改善 8,974 次、恶化 5,515 次；恶化次数较少，但平均 harm=2.081183m，高于平均 gain=1.255953m，gross harm=11,477.725345m 超过 gross gain=11,270.921774m。最大的 552 次恶化（正 harm 中最高10%）贡献62.289033% gross harm；该尾部只占全部车辆2.43%，足以抵消大量小改善。G1/G3 车辆也有尾部，但其总收益仍超过总损失。

行人机制不同：DualExpert 恶化 2,618 次、改善 1,774 次，平均 harm=0.514124m、gain=0.476335m，gross harm/gain=1,345.975886/845.018289m。最高10% harm 占45.966546%，不足以独立满足登记的50%尾部标准；这里同时存在更多错误切换及略高的单次损失，不能全部称为“少数极端值”。G1/G3 行人同样是恶化次数较多。Bicycle 的 G1/G3 各有少量样本且退化，Dual 路由完全不变。

完整 p90/p95/p99、1%/5%/10% tail 和零代价切换见 [switch summary](../02_switch_regret/stage11a_switch_summary.csv)。统计给出代价分解，不进行删除尾部或调阈值的模型选择。

## Pedestrian motion — CONFIRMED association, limited observable signal

GT 位移仅用于离线分组。区间左闭右开，8m 归入8m+；精确 >5m 筛选单独保留以复现历史。历史 recent speed 是最后两个有效观测位移除以实际时间间隔，net displacement/path length 仅来自历史有效点；不足2点单列，不当作静止。

{table(ped[ped.Group.str.startswith('Pedestrian/GTDisplacement/')],['Group','Model','Count','Top1FDE','OracleFDE','OracleGap','mode_change_rate','switch_harm','SoftCE','DeltaTop1FDE'])}

DualExpert 的 >5m 行人4,804个，delta={evidence['GTGreater5mMeanDelta']:+.6f}m；<=5m delta={evidence['GTAtMost5mMeanDelta']:+.6f}m；>5m 贡献 {evidence['GTGreater5mGrossHarmShare']*100:.2f}% gross harm。退化主要集中5–8m（+0.240948m），8m+反而小幅改善（−0.010847m）。因此这是特定运动分组的集中退化，不支持“运动越大就越差”的单调解释。

{table(ped[ped.Group.str.startswith('Pedestrian/HistoryRecentSpeed/')],['Group','Model','Count','Top1FDE','OracleFDE','OracleGap','mode_change_rate','switch_harm','SoftCE','DeltaTop1FDE'])}

按历史速度 >=1m/s 对 <1m/s，Dual delta 分别 {evidence['ObservedSpeedAtLeast1mpsMeanDelta']:+.6f} / {evidence['ObservedSpeedBelow1mpsMeanDelta']:+.6f}m，仅差约0.001068m；满足登记的方向条件，但该可观测分组证据很弱。0.5–1m/s 组退化较大，>=2m/s 组改善，不能声称高速度普遍更差。历史 net/path 分组完整见 [pedestrian motion](../03_pedestrian_analysis/stage11a_pedestrian_motion.csv)。因果性运动偏置及稳健性 **UNRESOLVED**。

## Soft target quality — global ambiguity NOT_CONFIRMED

温度固定1m，无搜索、无新标签用于训练。每个 actor 计算 best/second/worst FDE、两种 gap、qmax、熵/标准化熵、q_best/q_second/margin；数组保存在本地缓存，字段见 schema。

{table(sf,['Group','Count','best_FDE','second_best_FDE','worst_FDE','best_second_gap','max_q','normalized_entropy','q_margin','DiffuseTargetRate'])}

行人平均标准化熵0.842872、qmax0.324819、best-second FDE gap0.237791m，存在软目标接近的情形，但同时满足 H/log6>=0.9 与 q_margin<=0.05 的行人仅 {decision['HeadDevPedestrianDiffuseTargetRate']*100:.2f}%，未达到登记的多数50%标准；HeadTrain 对应24.50%。5–8m/8m+ 局部 diffuse 比例约39.64%/37.21%，保留作为局部现象，不能宣布固定1m是主要根因或最佳/最差温度。标签平坦也可能反映候选接近，因果归因 **UNRESOLVED**。

见 [soft label groups](../04_soft_label_analysis/stage11a_soft_label_groups.csv)。

## SoftCE vs ranking regret — CONFIRMED finite-change mismatch

{table(align[(align.Group.isin(['Vehicle','Pedestrian']))&align.Model.isin(SWITCH)],['Group','Comparison','Count','DeltaSoftCE','DeltaRegret_m','CEDown_RegretUp_Count','CEDown_RegretUp_Rate','CEDown_RegretUp_ConditionalRate'])}

DualExpert 行人 CE 降低0.011280nats、Regret 却增加0.070987m；915个 actor（全部行人12.96585%，CE下降对象中的23.86542%）表现 CE↓/Regret↑。G1/G3 行人也满足这一条件。Dual 车辆同样 CE↓/Regret↑，2,134个 actor（9.39302%）具有逐样本反向变化。这直接确认冻结输出的损失与选择指标变化不同步，**不等于确认训练目标造成了退化**。

对固定 actor 的 q，SoftCE=H(q)+KL(q||p)，其无约束最优 p=q，argmax q 与 minFDE mode 一致。因此并非数学最优解方向相反；问题是共享模型、有限拟合/概率变化、argmax 的不连续与错误选择的不同距离代价。相关系数/p 值仅描述性，窗口相关且 HeadDev 反复选择，不作为独立显著性检验。

完整类型及 GT/历史运动分组见 [loss-regret alignment](../05_loss_alignment/stage11a_loss_regret_alignment.csv)，全部模型每组 ADE/FDE/CE/Regret 见 [model metrics](../05_loss_alignment/stage11a_all_model_group_metrics.csv)。

## Candidate geometry — PARTIAL bottleneck

在统计前登记 endpoint<=0.25m 和全轨迹 RMS<=0.25m 两种近重复标准；exact duplicate 要求完整轨迹 bitwise 一致。每个 actor 保存全部15个无序 endpoint pair 距离、最小/平均距离、RMS spread/diameter。minADEOracle 是 ADE 最佳候选，与 FDE 最佳候选允许不同。

{table(geom[(geom.Partition=='HeadDev')&geom.Group.isin(TYPES)],['Group','Count','minADEOracle','minFDE','MinEndpointPairDistance','MeanEndpointPairDistance','EndpointSpreadRMS','NearDuplicateEndpointRate','NearDuplicateTrajectoryRate','ExactDuplicateTrajectoryRate'])}

Pedestrian oracleFDE=1.002098m，R2 Top1FDE=1.426775m，Regret=0.424677m。Oracle 占总 Top1FDE 的 {decision['PedestrianR2OracleFraction']*100:.2f}%，按登记20%/80%规则判为 PARTIAL；即使选择器每次知道真实最佳 mode，也只能消除当前 R2 约29.76%的总终点误差，不能修复候选本身的约70.24%。这是 oracle 分解，不是因果解释或新方法预期收益。

同时69.66%行人有 oracle<=1m 的好候选；其中存在选错情形。按已登记 oracle<=1m 且 Regret>0.5m，全行人中 R2 为11.8889%、Dual 为14.2979%。所以“有好候选但排序错”和“候选本身不足”都存在。Ped endpoint近重复21.99%、轨迹近重复25.92%，exact duplicate=0；较高的车辆近重复率需结合静态/运动分组，不能直接宣布全模型 mode collapse。

见 [geometry groups](../06_candidate_geometry/stage11a_geometry_groups.csv) 与 [geometry schema](../06_candidate_geometry/stage11a_geometry_schema.json)。

## HeadTrain distribution and gradients — static domination UNRESOLVED

以下为冻结 checkpoint 在 HeadTrain 的横截面统计，不是原训练每一步梯度的重建。gradient 精确为 dSoftCE/dlogit=p−q；记录 L1/L2/平方L2、有效micro batch1024的逐输出梯度缩放。平方L2 share 表示独立输出 logits 空间的梯度能量份额，不能代替共享参数梯度，后者还取决于 Jacobian 与向量抵消。

{table(train[(train.Model=='R2')&train.Group.isin(['Overall','Vehicle','Pedestrian','Bicycle','MovingVehicle','StoppedVehicle','ParkedVehicle','OtherOrUnknownVehicleState','ObservableStaticLowError','Complement'])],['Group','Count','PopulationFraction','MeanSoftCE','SoftCEContributionShare','MeanTargetEntropy','MeanExcessCE_KL','ExcessCEContributionShare','LogitGradSquaredL2Share'])}

Vehicle/Pedestrian/Bicycle HeadTrain 数量191,026/66,145/2,980。Vehicle moving/stopped/parked=41,728/23,044/121,939，另4,315 unknown 保留，不强归类。Vehicle 占73.43%样本、约85.60% R2 输出梯度能量，但其 moving 子组贡献76.44%全训练输出梯度能量，parked 仅3.15%。仅按数量不能认定 parked 导致主导。

观测静态低误差筛选（历史>=2点、recent speed<0.2m/s、oracleFDE<=1m）共150,936个，占58.02%；R2 原始 CE份额53.11%，扣除不可优化的目标熵后 KL/excess CE份额4.14%，输出梯度能量仅2.60%。G1/Dual 的相应梯度能量也仅3.18%/3.37%。大的 CE总量主要包含 H(q) 常量地板，不能直接当成优化驱动力。当前冻结输出统计不支持静态样本主导的简单解释；原训练过程中共享参数梯度、Jacobian与优化路径仍 **UNRESOLVED**，本阶段不训练来验证。

所有模型、车辆状态、行人历史运动组的 loss分位数和梯度统计见 [training distribution](../07_training_distribution/stage11a_training_loss_gradient.csv)，[gradient validation](../07_training_distribution/stage11a_gradient_audit.json)。

## Fixed case studies — illustrative, not parameter selection

从三个登记比较的联合池按有符号 delta 排序，先10 Pedestrian harm、再10 Vehicle harm、最后10 success；全30个 instance唯一，平局按identity/model排序。成功池最大收益均来自车辆，未为获得行人成功例改选。每图包含 GT、六候选、R2/G1/Stage10 Dual 分数与 logits、selected/oracle、15pair endpoint 差异；若触发比较是G3还显示G3。颜色与表格mode一致，黑线GT，虚线oracle，所有轨迹同轴且等比例。

Case02：G1 从R2 oracle候选0.287911m切到6.778095m，增加6.490184m，直接展示好候选被误排。Case07：G3增加4.704707m且oracle0.266089m，也有明确可用候选。Case10：oracle本身8.282680m，体现即使改排序仍有几何误差。Case11：车辆harm增加30.969259m；Case21：成功切换减少34.421583m，但oracle仍20.869108m，说明“改善”与“预测已好”不同。

数值原表见 [selection](../08_cases/stage11a_case_selection.csv)、[scores](../08_cases/stage11a_case_mode_scores.csv)、[endpoint pairs](../08_cases/stage11a_case_endpoint_pairs.csv)。图中三位小数用于阅读，CSV保留较高精度，模型选择仍使用原始完整分数。
''']
    case_table=['| Case | Category | Comparison | delta FDE (m) | Figure |','| --- | --- | --- | --- | --- |']
    for r in cases.itertuples():case_table.append(f'| {r.CaseNumber:02d} | {r.CaseCategory} | {r.Comparison} | {r.delta_FDE:+.6f} | [PNG](../08_cases/stage11a_case_{r.CaseNumber:02d}.png) |')
    sections.append('\n'.join(case_table))
    sections.append('''
## Exactly three candidate objectives — literature and risks

仅提出以下A/B/C三个目标；本阶段没有实现或训练B/C。文献已核对原始来源，不把其他领域结果当成本工程收益证明。OpenAlex三条精确检索HTTP429，CVF fetch受限，随后使用作者论文/arXiv和ACL原文；失败与支持范围记录于 [reference audit](literature/stage11a_verified_references.json)，可导入 [BibTeX](literature/stage11a_references.bib)。

### A. Original SoftCE

`L_A=-sum_k q_k log p_k`, `q=softmax(-FDE/1m)`。文献依据为严格 proper logarithmic score 与 KL 的关系；此理论针对给定目标分布，本工程的 FDE Gibbs q 属于具体设计，不意味着真实未来概率已校准。[Gneiting & Raftery, JASA 2007, Section3 Example3](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)

自身推导：`L_A=H(q)+KL(q||p)`，固定q最优p=q，最佳mode一致。与Top1FDE关系：有限的CE下降可以改善其他mode的概率拟合，同时让argmax转向更差mode；分类概率误差不会直接等同于选错后的米级代价。风险：局部近似候选使target margin小；共享拟合和概率变化受训练分布影响；用CE选checkpoint可能错过Top1改进。优点是当前协议已有完整证据，保留为未来受控对照。

### B. Hard Best-Mode CE

`L_B=-log p[argmin_k FDE_k]`。文献启发来自CoverNet的候选轨迹分类与最近候选CE训练；其原文最近候选采用平均逐点距离，**不是本工程拟议的FDE-best**。本提案将标签准则换为minFDE，是方法启发的适配，不能说直接复现其loss。[Phan-Minh et al., CoverNet, CVPR 2020, Section4.3](https://arxiv.org/html/1911.10298)

理论依据：one-hot标签对oracle mode给出明确分类监督；固定actor无约束最优将概率集中到best mode。与Top1FDE关系：强化最佳mode分类，但标准CE对所有错误类没有米级代价区分，在不确定未来/共享模型下最佳类别频率不必等价于最低期望距离。风险：近tie时硬标签跳变、单一GT与多模态不确定性、概率校准和多样性下降、坏候选coverage无法修复。无法据此宣布优于SoftCE。

### C. Error-Aware Ranking

`c_k=FDE_k-minFDE`, `L_C=sum_k p_k*c_k`。文献依据为minimum-risk训练中的posterior加权任务损失原则；原论文研究机器翻译，此处仅借鉴期望代价，不是轨迹预测实证。[Shen et al., ACL 2016, Section3 Eqs7–8](https://aclanthology.org/P16-1159.pdf)

自身推导：固定candidates/GT标签时，减minFDE仅减与logits无关的常量；`dL_C/dz_k=p_k*(c_k-E_p[c])`，高代价mode与低代价mode受到不同方向/幅度压力。与Top1FDE关系：直接对应概率加权FDE regret，**仍不是argmax Top1FDE本身**，期望代价下降不保证当前argmax误差单调下降。风险：极端FDE/噪声标签的梯度、概率集中及校准损失、单GT不确定性、共享优化的折衷、无法修复候选几何；冻结geometry的成本应保持监督标签属性。这里只给数学建议，没有替换工程loss或执行B/C。

## Interpretation with evidence levels

**CONFIRMED**：身份/候选/指标实现正确；多个冻结头出现CE下降与Regret上升；Dual车辆存在高代价切换尾部；行人>5m集中承担gross harm，但GT5–8m/8m+行为不同；大多数行人不满足已登记全局diffuse定义；候选oracle与选择regret均贡献误差；当前静态低误差样本的输出梯度能量较低。

**STRONGLY_SUGGESTED**：距离代价与有限概率变化的区别值得作为下一次受控loss比较的主问题。车辆尾部和行人多数错误切换提示要分别检查频率与代价，不能用一个“negative transfer”故事覆盖全部类型。

**UNRESOLVED**：SoftCE是否因果导致退化；任何新目标是否提高独立数据上的Top1FDE；共享参数和原训练全程梯度的组主导性；1m温度是否不合适；运动分组现象在其他checkpoint/scene上的稳定性。HeadDev复用、相关窗口和单一GT限制解释；本阶段未进行新训练、温度搜索或official VAL确认。

## Registered final decisions
''')
    for key in ('LossRankingMismatch','CostlyModeSwitchProblem','PedestrianMotionBias','SoftTargetAmbiguity','CandidateGeometryBottleneck','RecommendedNextObjective','ReadyForControlledLossExperiment'):
        sections.append(f'`{key} = {decision[key]}`\n')
    sections.append('推荐 ErrorAwareRanking 仅表示值得审查下一次受控对照，Ready=YES不构成Stage11B执行授权。HardCE也保留为用户指定的三个候选之一，不预先声称谁更好。Stage11A 完成后 STOP，等待大脑AI审查。')
    report=ROOT/'09_reports/stage11a_root_cause_report.md';report.write_text('\n\n'.join(sections)+'\n')
    atomic_json(ROOT/'09_reports/stage11a_report_audit.json',{'Status':'PASS','report_sha256':sha256(report),
        'final_integrity_status':qa['Status'],'decisions_sha256':sha256(ROOT/'09_reports/stage11a_decisions.json'),
        'original_references':3,'candidate_objectives':3,'cases':30,'evidence_levels':['CONFIRMED','STRONGLY_SUGGESTED','UNRESOLVED'],
        'Stage11B_started':False,'official_VAL_test_used':False})
    print('STAGE11_REPORT_WRITTEN',report,flush=True)

if __name__=='__main__':main()
