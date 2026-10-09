"""Verify the engineering evidence and write the conditional feasibility decision."""
from pathlib import Path
import sys, ast

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / s) for s in ('00_manifest', '06_planning_interface', '07_evaluation_preflight')]
from stage13a_common import *
from stage13a_artifacts import load_observation, load_evaluation_only
from stage13a_interface import cv_reference
from stage13a_metrics import numerical_audit

SECTIONS = ['Research Objective', 'Frozen Prediction System', 'nuScenes Ego Pose Source',
            'History/Future Time Horizon', 'Timestamp Alignment', 'Coordinate Alignment',
            'Prediction Identity Join', 'Observation-Only Forecasting', 'Prediction Coverage',
            'Missing Actor Handling', 'HD Map Compatibility', 'PlanningObservation Interface',
            'EvaluationOnly Labels', 'GT Leakage Audit', 'Open-loop Planning Feasibility',
            'Collision and Road Evaluation Limits', 'Route Information', 'Failure Cases',
            'Chapter 3 to Chapter 5 Connection', 'Scientific Decision']


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(str(v) for v in r)+' |' for r in rows])


def main():
    seed()
    frozen = verify(history=True, data=True)
    protocol = read_json(PROTOCOL)
    assert protocol['FrozenHistorySHA256']==sha256(FREEZE)
    assert protocol['RequirementsSHA256']==sha256(ROOT/'00_manifest/stage13a_requirements.txt')
    audits = {name: read_json(ROOT/path) for name, path in dict(
        small='06_planning_interface/stage13a_small_preflight.json',
        observation='04_observation_only_inference/stage13a_observation_only_audit.json',
        poison='06_planning_interface/stage13a_gt_poison_audit.json',
        coordinate='02_ego_trajectory/stage13a_coordinate_audit.json',
        mapping='05_map_alignment/stage13a_map_alignment.json',
        dataset='01_dataset_audit/stage13a_dataset_audit.json',
        pose='02_ego_trajectory/stage13a_ego_pose_audit.json',
        timing='02_ego_trajectory/stage13a_time_audit.json',
        cases='08_cases/stage13a_case_audit.json',
        metrics='07_evaluation_preflight/stage13a_metric_unit_audit.json').items()}
    for name, audit in audits.items():
        assert audit['Status']=='PASS' or (name=='pose' and audit['Status']=='PASS_WITH_SOURCE_LIMIT'), name
    windows = pd.read_csv(ROOT/'01_dataset_audit/cache/stage13a_all_windows.csv')
    samples = pd.read_csv(ROOT/'06_planning_interface/stage13a_planning_samples_manifest.csv')
    coverage = pd.read_csv(ROOT/'03_prediction_alignment/stage13a_prediction_coverage.csv')
    smallcover = pd.read_csv(ROOT/'04_observation_only_inference/stage13a_small_prediction_coverage.csv')
    join = pd.read_csv(ROOT/'03_prediction_alignment/stage13a_prediction_join.csv')
    poison = pd.read_csv(ROOT/'06_planning_interface/stage13a_gt_poison_details.csv')
    cases = pd.read_csv(ROOT/'08_cases/stage13a_case_manifest.csv')
    evaluation = pd.read_csv(ROOT/'07_evaluation_preflight/stage13a_evaluation_feasibility.csv')
    timestamps = pd.read_csv(ROOT/'02_ego_trajectory/stage13a_timestamp_alignment.csv')
    fixed = read_json(ROOT/'01_dataset_audit/stage13a_fixed_samples.json')
    assert set(windows.SceneToken)==set(scene_folds()) and windows.SceneToken.nunique()==630
    assert not windows.duplicated(['SceneToken', 'SampleToken']).any()
    assert len(samples)==len(poison)==48 and samples.SceneToken.nunique()==12
    assert samples.CompleteEgoEvaluation.sum()==36
    assert set(zip(samples.SceneToken, samples.SampleToken))=={(r['SceneToken'], r['SampleToken']) for r in fixed['Samples']}
    assert (poison.PredictionMaxDiff==0).all() and (poison.PlanningMaxDiff==0).all()
    assert poison.SourceFutureMetadataPoison.sum()==12
    assert not join.duplicated(['SceneToken', 'SampleToken', 'InstanceToken']).any()
    assert len(join)==samples.CurrentActors.sum()==smallcover.CurrentActors.sum()
    assert join.HistoricalOOFJoined.sum()==samples.HistoricalOOFJoined.sum()
    assert int(windows.HistoricalOOFActors.sum())==260151
    assert int(windows.EgoHistoryComplete.sum())==audits['dataset']['CompleteEgoHistoryWindows']
    assert int(windows.OpenLoopEgoEvaluationValid.sum())==audits['dataset']['OpenLoopEgoEvaluationWindows']
    assert audits['coordinate']['EgoAndActorFP64RoundtripMaxM']<1e-5 and audits['coordinate']['MapFP64RoundtripMaxM']<1e-5
    assert np.allclose(global_to_ego([[0,1],[-1,0]],[0,0],np.pi/2), [[1,0],[0,1]], atol=1e-14)
    assert np.allclose(ego_to_global([[1,0],[0,1]],[0,0],np.pi/2), [[0,1],[-1,0]], atol=1e-14)
    assert np.allclose(global_heading_to_ego(np.pi, np.pi/2), np.pi/2)
    assert np.allclose(wrap_angle([3*np.pi,-3*np.pi]), [-np.pi,-np.pi])
    assert (timestamps.SampleMinusLidarUS==timestamps.SampleTimestampUS-timestamps.LidarTimestampUS).all()
    assert np.isfinite(join.loc[join.HistoricalOOFJoined,'ReplaySingleGraphCandidateMaxDiffM']).all()
    cache_replay = []; graph_sources = {}
    graph_index = pd.read_csv(S3/'02_preprocessed/stage3_train_index.csv')
    legacy_graphs = {(r.scene_token,r.sample_token):r for r in graph_index.itertuples()}
    for row in samples.itertuples():
        obs, plan, identity = load_observation(row)
        assert obs.scene_token==row.SceneToken and obs.sample_token==row.SampleToken and obs.fold==scene_folds()[row.SceneToken]
        assert len(obs.instance_tokens)==row.CurrentActors
        expected = obs.other_agent_history_mask.sum(-1)>=2
        assert np.array_equal(obs.prediction_valid_mask, expected) and obs.other_agent_history_mask[:,-1].all()
        assert np.isfinite(obs.other_agent_predictions).all() and np.isfinite(plan).all()
        assert obs.other_agent_top1_predictions.shape==(row.CurrentActors,12,2)
        assert np.array_equal(cv_reference(obs), plan)
        assert np.array_equal(obs.other_agent_probabilities.argmax(-1),obs.other_agent_raw_probabilities.argmax(-1))
        bike = obs.other_agent_type==2
        assert np.array_equal(obs.other_agent_raw_probabilities[bike],obs.other_agent_probabilities[bike])
        assert identity['ModelIdentity']['Temperature']==TEMPERATURES[obs.fold]
        assert obs.route_information is None
        assert obs.map_geometry['available'] and obs.map_geometry['MapDrivableMask'].shape==(301,301)
        labels = load_evaluation_only(row)
        if labels.ego_future_mask.all():
            assert np.array_equal((labels.future_sample_timestamps_us-obs.t0_timestamp)/1e6,obs.future_times)
            assert identity['FutureTimeSource']=='recorded keyframe query timestamps'
            original = legacy_graphs[(row.SceneToken,row.SampleToken)]
            path = S3/original.file_path
            graph_sources[str(path.relative_to(PROJECT))]=sha256(path)
        else:
            assert not labels.ego_future_mask.any() and not labels.agent_future_mask.any()
            assert identity['FutureTimeSource']=='externally requested nominal0.5s query'
        for table_name, token in identity['PastMetadataReads']:
            assert table_name in ('sample_annotation','sample_data','ego_pose')
        cache_replay.append(dict(SampleOrdinal=row.SampleOrdinal, Identity='PASS', Fold='PASS',
                                 CVReplay='BITWISE_EQUAL', FutureQuerySchedule='PASS', ActorEligibility='PASS',
                                 BicycleProbabilityCopy='BITWISE_EQUAL', RouteInput=None))
    for row in coverage.itertuples():
        column = row.AgentType
        h = windows.EgoHistoryComplete
        assert row.CurrentActors==windows.loc[h,column+'Current'].sum()
        assert row.HistoryEligibleActors==windows.loc[h,column+'Eligible'].sum()
        assert row.MissingHistoryActors==row.CurrentActors-row.HistoryEligibleActors
    for row in cases.itertuples():
        assert row.SameAxesAndEgoFrame and row.FiveFigureFamilies and row.GTLabel=='EvaluationOnly'
        for p in (row.PNG,row.PDF,row.SVG):
            assert (ROOT/p).is_file() and 1000<(ROOT/p).stat().st_size<10_000_000
    assert audits['cases']['FixedScenesRepresented']==12 and audits['cases']['MissingFutureCases']>=1
    assert len(evaluation)==84 and (evaluation.Reference=='CV_Ego_EngineeringReference').sum()==48
    assert np.isfinite(evaluation[['MeanSpeedMPS','MaxAccelerationMPS2','MaxJerkProxyMPS3']].to_numpy()).all()
    assert ((evaluation.GTInterpolatedPairCoverage>=0)&(evaluation.GTInterpolatedPairCoverage<=1)).all()
    assert ((evaluation.DrivableEvaluationCoverage>=0)&(evaluation.DrivableEvaluationCoverage<=1)).all()
    metric_units = numerical_audit()
    assert metric_units['Status']=='PASS'
    atomic_json(ROOT/'07_evaluation_preflight/stage13a_metric_unit_audit.json',metric_units)
    # Static guard for accidental training entry points in this stage's code.
    forbidden_calls = {'backward','train','step','Adam','AdamW','SGD','Trainer','fit','save_checkpoint'}
    for path in ROOT.rglob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ''
                assert name not in forbidden_calls, (path,name)
    assert not list(ROOT.rglob('*.pt')) and not list(ROOT.rglob('*.pth'))
    verify(history=True,data=True)
    atomic_json(ROOT/'03_prediction_alignment/stage13a_legacy_graph_source_hashes.json',graph_sources)
    dump('06_planning_interface/stage13a_artifact_replay_audit.csv',cache_replay)
    dataset = audits['dataset']; timing = audits['timing']; pose = audits['pose']; coordinate = audits['coordinate']
    current = int(smallcover.CurrentActors.sum()); eligible = int(smallcover.HistoryEligible.sum())
    retained_missing = int(smallcover.MissingHistory.sum()); gt_incomplete = int(smallcover.FutureIncompleteEligible.sum())
    discrepancy = distribution(join.loc[join.HistoricalOOFJoined,'ReplaySingleGraphCandidateMaxDiffM'].values)
    issues = [
        'No supplied route/goal/high-level command; legal route input design remains necessary.',
        f'Original trainval volume unmounted; original ego_pose.timestamp verified for only {pose["KnownKeyframes"]}/{pose["KnownKeyframes"]+pose["UnknownKeyframes"]} HeadTrain keyframes. Restore original metadata for full source verification.',
        'Current actor annotation size unavailable outside small raw subset; class-prior footprints and approximate ego dimensions require disclosure or original-size recovery.',
        'Insufficient actor history and incomplete future GT remain unknown-risk/unknown-evaluation; no claim of complete traffic coverage.',
        'Source about 2Hz, nonreactive forecasts, no route-conditioned or closed-loop planning validation; interpolated footprints remain proxies.'
    ]
    decision = dict(
        Stage='Stage13A',Stage13A='CONDITIONAL_GO',ReadyForPlanningDevelopment='CONDITIONAL',
        EngineeringScope='observation-only frozen forecast to ego planning/evaluation interfaces',
        EgoTrajectoryExtraction='PASS',TimestampAlignment='PASS',OriginalPoseTimestampVerification='PARTIAL',
        CoordinateAlignment='PASS',PredictionIdentityJoin='PASS',ObservationOnlyInference='PASS',
        GTLeakage='PASS',GTLeakageTestInterpretation='no future-label dependency detected',
        MapCompatibility='PASS',PlanningInterface='PASS',PlanningSampleCount=48,
        CompleteOpenLoopEvaluationSampleCount=36,InferenceOnlyMissingFutureSampleCount=12,
        FullMetadataScenes=630,FullMetadataForwardReplay=False,
        VehiclePredictionCoverage=float(coverage.loc[coverage.AgentType=='Vehicle','MetadataPredictionCoverage'].iloc[0]),
        PedestrianPredictionCoverage=float(coverage.loc[coverage.AgentType=='Pedestrian','MetadataPredictionCoverage'].iloc[0]),
        BicyclePredictionCoverage=float(coverage.loc[coverage.AgentType=='Bicycle','MetadataPredictionCoverage'].iloc[0]),
        CoverageDefinition='history eligibility / all supported t0 actors with5 ego history frames; full630 metadata, actual replay fixed48 samples',
        RouteInformation='UNAVAILABLE',OpenLoopEvaluationFeasibility='PASS_WITH_LIMITS',
        BlockingIssues=issues,NoTraining=True,NoOptimizer=True,NoNewCheckpoint=True,
        HistoricalDecisionsUnchanged=True,EndToEndIndependentValidation=False,ClosedLoopSafetyClaim=False,
        Stage13BImplemented=False,NextStageAuthorized=False,AfterCompletion='STOP_FOR_BRAIN_AI_REVIEW',
        Remediation=[
            dict(Module='ego metadata and footprints',Action='restore original local trainval mount; join source timestamps and sizes by tokens',NewData='no new dataset; existing original metadata required',ExtraPlanningFramework=False),
            dict(Module='route/command interface',Action='design lawful inference-time route input; never use recorded GT endpoint',NewData='possibly route annotations or a separately authorized route-generation protocol',ExtraPlanningFramework='not required for current audit; future design pending review'),
            dict(Module='missing-actor/evaluation masks',Action='retain conservative unknown-risk status, publish coverage; any fallback requires future authorization',NewData='not required for honest present audit',ExtraPlanningFramework=False)],
        FrozenCheckpoints=len(frozen['checkpoints']),FrozenTrackedHistoricalFiles=len(frozen['historical_files']),
        PreservedUntrackedFiles=len(frozen['preserved_untracked']),CandidateReplaySingleGraphMaxDiffM=discrepancy['Max'])
    atomic_json(ROOT/'10_reports/stage13a_scientific_decision.json',decision)
    cov_table = table(['Class','Current t0 occurrences','History eligible','Metadata coverage','Missing history','Old OOF','Eligible absent OOF'],
        [[r.AgentType,r.CurrentActors,r.HistoryEligibleActors,f'{r.MetadataPredictionCoverage:.4%}',r.MissingHistoryActors,r.HistoricalOOFActors,r.EligibleButAbsentOOF] for r in coverage.itertuples()])
    win_table = table(['HeadTrain630 metadata quantity','Count'],[[key,dataset[key]] for key in [
        'Scenes','TotalT0Windows','CompleteEgoHistoryWindows','AnyEgoFutureGTWindows','CompleteEgoFutureGTWindows',
        'ObservationInputWindows','HistoryAndAnyPredictionWindows','VehiclePredictionWindows','PedestrianPredictionWindows',
        'BicyclePredictionWindows','MapAvailableWindows','OpenLoopEgoEvaluationWindows']])
    content = [
        '本阶段审计第三章冻结预测系统能否向第五章自车规划提供合法、同帧同时间、无未来GT依赖的工程接口。只做数据提取、冻结forward、静态地图对接、CV参考与离线评价预检；没有训练、优化器、新checkpoint、official VAL/test评价或规划模型选择。结论为 CONDITIONAL_GO，属于工程可行性。',
        f'基线提交 `{BASE}`；分支 `stage13a/prediction-to-ego-planning-audit`。冻结 Stage5A 六模态候选生成器、原三折 Stage11B C、原三折 R2、原归一化与 Stage11C 温度。推理的7个模型全部eval、requires_grad=False，参数字节前后相同且无梯度；另对历史{len(frozen["checkpoints"])}个checkpoint及{len(frozen["historical_files"])}个已跟踪文件验SHA，{len(frozen["preserved_untracked"])}个旧Stage2C未提交文件保留。Vehicle/Pedestrian使用对应折C，可提供已有T缩放；Bicycle直接保留对应折R2概率。Stage12B的NOT_SUPPORTED/STOP及所有历史结论保持。HeadTrain630沿用原OuterTest每折210，没有重划分，HeadDev/VAL/test未用于预测重放、评价或模型选择。Stage5A历史训练可能见过本场景，不能称为独立端到端泛化评估。',
        f'通过 sample.data[LIDAR_TOP] → keyframe sample_data.ego_pose_token → ego_pose 提取位置与四元数yaw。完整历史Stage2C SQLite只读，原始volume当前未挂载；SQLite含850 scenes/34149 samples，本阶段仅处理HeadTrain630。可用原始JSON仅5 scenes，其中4属HeadTrain；原始pose timestamp可直接验证{pose["KnownKeyframes"]}个keyframe，另{pose["UnknownKeyframes"]}个未知。SQLite保留pose token/translation/rotation，丢弃timestamp；也丢弃annotation size。未知项从不补造，不拿sample timestamp冒充pose timestamp。逐token核对有原始记录的translation/rotation和sensor channel，记录字段来源。恢复原始本地volume是完整来源核验的后续条件。',
        'Th=5、Tf=12、K=6，源数据约2Hz、未来约6s。历史以真实sample相对时间保留。36个可评价窗口使用记录的12个future sample时间作为外部查询时间表；模型仍是按keyframe索引预测，未升级为连续时间模型。另12个末帧演示采用明确标注的外部0.5s名义查询表，没有未来记录，也没有假装已与未来GT核对。',
        f'TimestampAlignment=PASS（keyframe/sample与预测索引）；OriginalPoseTimestampVerification=PARTIAL。scene/sample/t0/fold及12个relative query times逐样本核对，36窗口与旧图future_times逐项相等。sample−LIDAR偏差（微秒）为 `{json.dumps(timing["SampleMinusLidarUS"])}`；真实采样间隔（秒）为 `{json.dumps(timing["SamplingIntervalSeconds"])}`。有原始pose timestamp的记录其pose−LIDAR偏差为 `{json.dumps(pose["PoseMinusLidarUS"])}`。pose采用匹配LIDAR_TOP的ego_pose，未对未知获取时刻做无记录的强制修正。',
        f'全部数据在t0 ego平面坐标：原点为t0 LIDAR关联pose位置，+x前进、+y左侧；yaw为逆时针弧度，距离米。复用preprocessing.coordinates的FP64 global_to_ego/ego_to_global。固定样本ego/actor roundtrip最大{coordinate["EgoAndActorFP64RoundtripMaxM"]:.3g}m，地图{coordinate["MapFP64RoundtripMaxM"]:.3g}m；全630 ego最大{dataset["CoordinateFP64MaxErrorM"]:.3g}m，均<1e-5m。另验证90°前/左方向、角度归一化与反变换。36窗口历史actor与旧Stage3图、ego GT与原build_window按FP32字节相等。',
        f'用唯一(scene_token,sample_token,instance_token)键加入Stage11B OOF，不能靠数组位置拼接。全630 metadata复现原260151个OOF targets；48窗口实际匹配{int(join.HistoricalOOFJoined.sum())}个，fold、旧候选source_index与6个几何SHA及frame origin/yaw均核对。Mode A明确是full-horizon离线子集。Mode B单图forward与旧批量候选的最大差为{discrepancy["Max"]:.9g}m（分布 `{json.dumps(discrepancy)}`），该数作为数值重放差异公开，未宣称逐bit候选相同；同一新接口的未来污染对照则逐bit相同。',
        f'ObservationOnlyInference=PASS。PastOnly facade仅允许读取t0及此前4帧的annotations、LIDAR sample_data及ego_pose；越界未来token读取会失败。用历史重建原Stage3 x、history positions、rotation、BOS、lane geometry与当前全actor交互图，未用Dataset的未来target/full-horizon筛选来挑actor。graph未使用future字段以中性placeholder提供并专门污染；C/R2特征复用原数值定义。固定12 scenes跨4地图，48窗口实际forward当前{current}个支持类别actor，{eligible}个历史有效，{retained_missing}个缺历史的行仍保留，{gt_incomplete}个有效行没有完整未来GT。12个末帧窗口无未来GT仍可生成有限预测与CV。',
        win_table+'\n\n'+cov_table+'\n\n覆盖率分母为有5帧ego历史的t0全部支持类别actor出现次数，包含场景最后12帧；是元数据历史资格比例，不能声称630场景全部模型forward已经通过。实际forward限定固定48窗口。上述map可用数是源文件/区域可用性；实际48窗口均验证drivable geometry/raster存在。无可预测actor场景比例='+f'{dataset["NoPredictableActorsSceneFraction"]:.4%}，历史窗口比例='+f'{dataset["NoPredictableActorsHistoryWindowFraction"]:.4%}。全窗口细表仅存本地cache，Git提交汇总与630 scene级表。',
        f'所有t0支持类实例均进入接口；至少2/5历史有效才PredictionValidMask=True。缺历史{retained_missing}行有有限forward输出但不能作为有效预测或“无风险”，评估显式保留UnknownActorCount。OOF缺失原因分为历史不足、未来不完整/录制外；缺失未来GT不改变预测资格。未支持taxonomy的actor不在冻结模型输入范围，现有覆盖率是支持三类条件下覆盖，不能声称全部交通对象覆盖。未来缺失段要求两端均有效才能评估，缺失不按空场景处理。',
        f'MapCompatibility=PASS。复用已冻结4个地图JSON及既有SparseSemanticIndex，无地图修复、semantic训练或路线GT推断。每个t0提取±150m patch内drivable area、边界、lane、crosswalk/walkway，均转t0 frame，1m raster [301,301]；48窗口都存在可用drivable union。历史源中已被既有索引排除的无效component为 `{json.dumps(audits["mapping"]["ExistingInvalidSourceComponentsExcluded"])}`，本阶段未改变该处理。超出patch的footprint标未知评价覆盖，不自动称为offroad；map道路连通性不等于导航路线。',
        'PlanningObservation是独立dataclass，包含ego历史/heading、实际history times及外部future queries，其他参与者[N,6,12,2]预测、raw/温度缩放概率、当前type/position/heading、尺寸及来源、valid mask、history及其mask、唯一instance键、scene/sample/fold/timestamp、静态地图、冻结模型identity。route_information=None。cv_reference只接受该类型，输出[12,2]，用最后两帧ego位置与真实dt得到恒速参考。完整PlanningSample或EvaluationLabels传入即TypeError。CV是验证接口的无训练参考，没有避障、目标导航或新的planner模块。',
        'PlanningEvaluationLabels单独保存ego future GT/heading/mask、other agent future GT/heading/mask、未来sample/LIDAR/source-pose时间；永不传给FrozenForecaster或CV函数。PlanningSample只用于离线打包Observation与EvaluationOnly。Recorded Ego Future只在评价/图示中作为参考，36完整窗口可检验shape、frame与零自比误差；12末帧未来标签全无效。保存到独立 *_observation.npz 与 *_evaluation_only.npz，本地大缓存不上Git。',
        'GTLeakage=PASS表示未检出依赖。48窗口分别把评价未来浮点标签替换NaN并改mask、把模型未来placeholder positions变NaN及padding/target/future/full-horizon masks污染；Observation identities/features、预测、CV均相同，预测最大差=0、CV最大差=0。额外每scene一个窗口（共12）直接把原metadata中t0之后ego translations和actor translations/quaternions替换NaN，再完整重建Observation并forward，仍逐bit相同。过去token白名单阻断未来数据读取。冻结7模型参数/eval/无梯度及全历史SHA再次核验。',
        'OpenLoopEvaluationFeasibility=PASS_WITH_LIMITS。48窗口均可运行CV与预测碰撞/drivable/motion预检，其中36有完整ego GT；Recorded Ego Future仅这36能评价，因此84条reference预检记录。EgoADE按有效GT步平均，EgoFDE仅12步完整时输出；末帧无GT保持缺失。速度使用连续位置/真实dt，acceleration使用速度中点时间，jerk proxy使用acceleration中点时间。数值审计验证非均匀采样下2m/s恒速→零acceleration/jerk、旋转矩形分离/重叠、keyframe间穿越与缺失mask。统计用于验证接口，不汇总或宣传CV优于真实驾驶。',
        '碰撞是两辆有方向矩形的4轴SAT重叠proxy，包含t0到12个future keyframe之间≤0.05s线性位置/最短角度插值；插值增加检查点，不增加传感器真实观测。预测heading采用轨迹切向，停滞保留当前heading；GT actor采用实际annotation heading，记录自车参考采用可用pose heading，CV采用切向。actor当前原始尺寸可得时用annotation width/length，否则V/P/B长宽priors分别4.5×1.8、0.6×0.6、1.8×0.6m；ego 4.8×2.0m为明确近似。All6、Top1与模式加权overlap分开，后者为未校准的加权重叠/受影响actor得分，不是真实碰撞概率。drivable检查插值ego完整footprint与原始union的覆盖，patch外为未知；motion jerk是~2Hz代理。非反应式预测对自车改变轨迹不会作出环境响应；不支持闭环、交互式反事实安全结论。',
        'RouteInformation=UNAVAILABLE。现有sample/ego_pose/log及HD Map没有提供可直接使用的预定路线、合法目标或高层驾驶指令；地图lane connectivity和记录future endpoint都不能替代推理时已知的route。没有把未来GT终点作为输入。后续若要目标驱动规划，必须另行设计合法的route/command数据或生成协议，等待大脑AI审查授权。当前CV仅验证无route的接口。',
        f'案例从固定12scene池按已注册当前density/预测-CV footprint潜在冲突取样，未根据GT ADE/FDE或成功效果选案例。实际展示车辆较多{audits["cases"]["VehicleRichScenes"]}个scene、行人较多{audits["cases"]["PedestrianRichScenes"]}个、潜在预测冲突{audits["cases"]["PotentialConflictScenes"]}个；所有12固定scene均代表，共{audits["cases"]["Cases"]}张五面板案例，其中包含4地图末帧未来GT缺失。每案例5族图使用相同axis/t0 frame、GT标EvaluationOnly。为读图只显示当前最近12actor，列表保存在case manifest；预测和risk计算仍包含全actor。源pose timestamp缺失、尺寸来源近似和历史不足的unknown状态均披露，没有伪造错位/冲突案例。',
        '第三章提供history-only冻结多参与者六候选与模式概率；第五章接口可消费EgoHistory、OtherAgentPredictions、StaticMap及可选的合法route，输出EgoFutureTrajectory，评价对象独立消费ego/actorGT及map。现阶段完成接口衔接和反泄漏证据。尚无可用route，需恢复源metadata时间戳/尺寸核验并保留mask覆盖限制；现有接口无需额外规划框架，后续具体算法或训练框架由审查决定。没有实现Stage13B、Diffusion或新的语义结构。',
        'Stage13A=CONDITIONAL_GO；ReadyForPlanningDevelopment=CONDITIONAL。关键工程gate均PASS，但完整源pose时间核验、精确footprint、route与评价覆盖有条件。\n\n'+
        '\n'.join('- '+issue for issue in issues)+'\n\n预计修改ego元数据/尺寸join、合法route接口与unknown-risk评价策略；恢复已有原始metadata不需新数据集，route可能需要新增标注或单独授权协议。该结论不授权进入下一阶段。提交并push本分支，不merge main；完成后STOP，等待大脑AI检查报告。'
    ]
    assert len(content)==len(SECTIONS)
    report = '# Stage13A Prediction-to-Ego-Planning Feasibility Audit\n\n'
    report += '\n\n'.join('## 【'+name+'】\n\n'+text for name,text in zip(SECTIONS,content))+'\n'
    (ROOT/'10_reports/stage13a_final_report.md').write_text(report)
    verification = dict(Status='PASS',RequiredReportSections=len(SECTIONS),FixedSampleCacheReplay=48,
        HeadTrainSceneIsolation=630,HistoricalOOFIdentityMatches=260151,NoTrainingCalls=True,NoNewCheckpoints=True,
        HistoricalSHAIntegrity=True,OriginalDataSHAIntegrity=True,PreservedStage2CFiles=True,
        ProbabilityAndFrameChecks=True,UnknownMasksAndCoverageRetained=True,
        CandidateReplayDifferences=discrepancy,PlanningDecision='CONDITIONAL_GO',SourceTimestampVerification='PARTIAL',
        SourceStage3GraphFiles=len(graph_sources),ScientificDecisionSHA256=sha256(ROOT/'10_reports/stage13a_scientific_decision.json'),
        ReportSHA256=sha256(ROOT/'10_reports/stage13a_final_report.md'))
    atomic_json(ROOT/'10_reports/stage13a_final_verification.json',verification)
    print('STAGE13A_FINAL_VERIFICATION_PASS',json.dumps(decision,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
