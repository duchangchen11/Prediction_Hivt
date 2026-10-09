# Stage13A Prediction-to-Ego-Planning Feasibility Audit

## 【Research Objective】

本阶段审计第三章冻结预测系统能否向第五章自车规划提供合法、同帧同时间、无未来GT依赖的工程接口。只做数据提取、冻结forward、静态地图对接、CV参考与离线评价预检；没有训练、优化器、新checkpoint、official VAL/test评价或规划模型选择。结论为 CONDITIONAL_GO，属于工程可行性。

## 【Frozen Prediction System】

基线提交 `020bf774e33b3269a463bbc2d03e0c6f9185a1d1`；分支 `stage13a/prediction-to-ego-planning-audit`。冻结 Stage5A 六模态候选生成器、原三折 Stage11B C、原三折 R2、原归一化与 Stage11C 温度。推理的7个模型全部eval、requires_grad=False，参数字节前后相同且无梯度；另对历史29个checkpoint及2894个已跟踪文件验SHA，5个旧Stage2C未提交文件保留。Vehicle/Pedestrian使用对应折C，可提供已有T缩放；Bicycle直接保留对应折R2概率。Stage12B的NOT_SUPPORTED/STOP及所有历史结论保持。HeadTrain630沿用原OuterTest每折210，没有重划分，HeadDev/VAL/test未用于预测重放、评价或模型选择。Stage5A历史训练可能见过本场景，不能称为独立端到端泛化评估。

## 【nuScenes Ego Pose Source】

通过 sample.data[LIDAR_TOP] → keyframe sample_data.ego_pose_token → ego_pose 提取位置与四元数yaw。完整历史Stage2C SQLite只读，原始volume当前未挂载；SQLite含850 scenes/34149 samples，本阶段仅处理HeadTrain630。可用原始JSON仅5 scenes，其中4属HeadTrain；原始pose timestamp可直接验证162个keyframe，另25145个未知。SQLite保留pose token/translation/rotation，丢弃timestamp；也丢弃annotation size。未知项从不补造，不拿sample timestamp冒充pose timestamp。逐token核对有原始记录的translation/rotation和sensor channel，记录字段来源。恢复原始本地volume是完整来源核验的后续条件。

## 【History/Future Time Horizon】

Th=5、Tf=12、K=6，源数据约2Hz、未来约6s。历史以真实sample相对时间保留。36个可评价窗口使用记录的12个future sample时间作为外部查询时间表；模型仍是按keyframe索引预测，未升级为连续时间模型。另12个末帧演示采用明确标注的外部0.5s名义查询表，没有未来记录，也没有假装已与未来GT核对。

## 【Timestamp Alignment】

TimestampAlignment=PASS（keyframe/sample与预测索引）；OriginalPoseTimestampVerification=PARTIAL。scene/sample/t0/fold及12个relative query times逐样本核对，36窗口与旧图future_times逐项相等。sample−LIDAR偏差（微秒）为 `{"Count": 25307, "Min": 0.0, "Mean": 0.0, "Median": 0.0, "P95": 0.0, "Max": 0.0}`；真实采样间隔（秒）为 `{"Count": 24677, "Min": 0.394377, "Mean": 0.4989543034809742, "Median": 0.499883, "P95": 0.549643, "Max": 0.750945}`。有原始pose timestamp的记录其pose−LIDAR偏差为 `{"Count": 162, "Min": 0.0, "Mean": 0.0, "Median": 0.0, "P95": 0.0, "Max": 0.0}`。pose采用匹配LIDAR_TOP的ego_pose，未对未知获取时刻做无记录的强制修正。

## 【Coordinate Alignment】

全部数据在t0 ego平面坐标：原点为t0 LIDAR关联pose位置，+x前进、+y左侧；yaw为逆时针弧度，距离米。复用preprocessing.coordinates的FP64 global_to_ego/ego_to_global。固定样本ego/actor roundtrip最大2.84e-14m，地图1.36e-12m；全630 ego最大0m，均<1e-5m。另验证90°前/左方向、角度归一化与反变换。36窗口历史actor与旧Stage3图、ego GT与原build_window按FP32字节相等。

## 【Prediction Identity Join】

用唯一(scene_token,sample_token,instance_token)键加入Stage11B OOF，不能靠数组位置拼接。全630 metadata复现原260151个OOF targets；48窗口实际匹配920个，fold、旧候选source_index与6个几何SHA及frame origin/yaw均核对。Mode A明确是full-horizon离线子集。Mode B单图forward与旧批量候选的最大差为2.28881836e-05m（分布 `{"Count": 920, "Min": 4.76837158203125e-07, "Mean": 3.6433986995531165e-06, "Median": 3.814697265625e-06, "P95": 7.62939453125e-06, "Max": 2.288818359375e-05}`），该数作为数值重放差异公开，未宣称逐bit候选相同；同一新接口的未来污染对照则逐bit相同。

## 【Observation-Only Forecasting】

ObservationOnlyInference=PASS。PastOnly facade仅允许读取t0及此前4帧的annotations、LIDAR sample_data及ego_pose；越界未来token读取会失败。用历史重建原Stage3 x、history positions、rotation、BOS、lane geometry与当前全actor交互图，未用Dataset的未来target/full-horizon筛选来挑actor。graph未使用future字段以中性placeholder提供并专门污染；C/R2特征复用原数值定义。固定12 scenes跨4地图，48窗口实际forward当前2023个支持类别actor，1948个历史有效，75个缺历史的行仍保留，1028个有效行没有完整未来GT。12个末帧窗口无未来GT仍可生成有限预测与CV。

## 【Prediction Coverage】

| HeadTrain630 metadata quantity | Count |
| --- | --- |
| Scenes | 630 |
| TotalT0Windows | 25307 |
| CompleteEgoHistoryWindows | 22787 |
| AnyEgoFutureGTWindows | 24677 |
| CompleteEgoFutureGTWindows | 17747 |
| ObservationInputWindows | 22787 |
| HistoryAndAnyPredictionWindows | 22742 |
| VehiclePredictionWindows | 22651 |
| PedestrianPredictionWindows | 18602 |
| BicyclePredictionWindows | 5194 |
| MapAvailableWindows | 25307 |
| OpenLoopEgoEvaluationWindows | 15227 |

| Class | Current t0 occurrences | History eligible | Metadata coverage | Missing history | Old OOF | Eligible absent OOF |
| --- | --- | --- | --- | --- | --- | --- |
| Vehicle | 443435 | 429071 | 96.7607% | 14364 | 191026 | 238045 |
| Pedestrian | 155283 | 150534 | 96.9417% | 4749 | 66145 | 84389 |
| Bicycle | 8147 | 7806 | 95.8144% | 341 | 2980 | 4826 |

覆盖率分母为有5帧ego历史的t0全部支持类别actor出现次数，包含场景最后12帧；是元数据历史资格比例，不能声称630场景全部模型forward已经通过。实际forward限定固定48窗口。上述map可用数是源文件/区域可用性；实际48窗口均验证drivable geometry/raster存在。无可预测actor场景比例=0.0000%，历史窗口比例=0.1975%。全窗口细表仅存本地cache，Git提交汇总与630 scene级表。

## 【Missing Actor Handling】

所有t0支持类实例均进入接口；至少2/5历史有效才PredictionValidMask=True。缺历史75行有有限forward输出但不能作为有效预测或“无风险”，评估显式保留UnknownActorCount。OOF缺失原因分为历史不足、未来不完整/录制外；缺失未来GT不改变预测资格。未支持taxonomy的actor不在冻结模型输入范围，现有覆盖率是支持三类条件下覆盖，不能声称全部交通对象覆盖。未来缺失段要求两端均有效才能评估，缺失不按空场景处理。

## 【HD Map Compatibility】

MapCompatibility=PASS。复用已冻结4个地图JSON及既有SparseSemanticIndex，无地图修复、semantic训练或路线GT推断。每个t0提取±150m patch内drivable area、边界、lane、crosswalk/walkway，均转t0 frame，1m raster [301,301]；48窗口都存在可用drivable union。历史源中已被既有索引排除的无效component为 `[["boston-seaport", "walkway", "15de5628-2b89-4cc1-acb3-f5751dcbba5f", "b7ada811-352c-4501-8162-cfdba0a8d640"]]`，本阶段未改变该处理。超出patch的footprint标未知评价覆盖，不自动称为offroad；map道路连通性不等于导航路线。

## 【PlanningObservation Interface】

PlanningObservation是独立dataclass，包含ego历史/heading、实际history times及外部future queries，其他参与者[N,6,12,2]预测、raw/温度缩放概率、当前type/position/heading、尺寸及来源、valid mask、history及其mask、唯一instance键、scene/sample/fold/timestamp、静态地图、冻结模型identity。route_information=None。cv_reference只接受该类型，输出[12,2]，用最后两帧ego位置与真实dt得到恒速参考。完整PlanningSample或EvaluationLabels传入即TypeError。CV是验证接口的无训练参考，没有避障、目标导航或新的planner模块。

## 【EvaluationOnly Labels】

PlanningEvaluationLabels单独保存ego future GT/heading/mask、other agent future GT/heading/mask、未来sample/LIDAR/source-pose时间；永不传给FrozenForecaster或CV函数。PlanningSample只用于离线打包Observation与EvaluationOnly。Recorded Ego Future只在评价/图示中作为参考，36完整窗口可检验shape、frame与零自比误差；12末帧未来标签全无效。保存到独立 *_observation.npz 与 *_evaluation_only.npz，本地大缓存不上Git。

## 【GT Leakage Audit】

GTLeakage=PASS表示未检出依赖。48窗口分别把评价未来浮点标签替换NaN并改mask、把模型未来placeholder positions变NaN及padding/target/future/full-horizon masks污染；Observation identities/features、预测、CV均相同，预测最大差=0、CV最大差=0。额外每scene一个窗口（共12）直接把原metadata中t0之后ego translations和actor translations/quaternions替换NaN，再完整重建Observation并forward，仍逐bit相同。过去token白名单阻断未来数据读取。冻结7模型参数/eval/无梯度及全历史SHA再次核验。

## 【Open-loop Planning Feasibility】

OpenLoopEvaluationFeasibility=PASS_WITH_LIMITS。48窗口均可运行CV与预测碰撞/drivable/motion预检，其中36有完整ego GT；Recorded Ego Future仅这36能评价，因此84条reference预检记录。EgoADE按有效GT步平均，EgoFDE仅12步完整时输出；末帧无GT保持缺失。速度使用连续位置/真实dt，acceleration使用速度中点时间，jerk proxy使用acceleration中点时间。数值审计验证非均匀采样下2m/s恒速→零acceleration/jerk、旋转矩形分离/重叠、keyframe间穿越与缺失mask。统计用于验证接口，不汇总或宣传CV优于真实驾驶。

## 【Collision and Road Evaluation Limits】

碰撞是两辆有方向矩形的4轴SAT重叠proxy，包含t0到12个future keyframe之间≤0.05s线性位置/最短角度插值；插值增加检查点，不增加传感器真实观测。预测heading采用轨迹切向，停滞保留当前heading；GT actor采用实际annotation heading，记录自车参考采用可用pose heading，CV采用切向。actor当前原始尺寸可得时用annotation width/length，否则V/P/B长宽priors分别4.5×1.8、0.6×0.6、1.8×0.6m；ego 4.8×2.0m为明确近似。All6、Top1与模式加权overlap分开，后者为未校准的加权重叠/受影响actor得分，不是真实碰撞概率。drivable检查插值ego完整footprint与原始union的覆盖，patch外为未知；motion jerk是~2Hz代理。非反应式预测对自车改变轨迹不会作出环境响应；不支持闭环、交互式反事实安全结论。

## 【Route Information】

RouteInformation=UNAVAILABLE。现有sample/ego_pose/log及HD Map没有提供可直接使用的预定路线、合法目标或高层驾驶指令；地图lane connectivity和记录future endpoint都不能替代推理时已知的route。没有把未来GT终点作为输入。后续若要目标驱动规划，必须另行设计合法的route/command数据或生成协议，等待大脑AI审查授权。当前CV仅验证无route的接口。

## 【Failure Cases】

案例从固定12scene池按已注册当前density/预测-CV footprint潜在冲突取样，未根据GT ADE/FDE或成功效果选案例。实际展示车辆较多3个scene、行人较多3个、潜在预测冲突3个；所有12固定scene均代表，共19张五面板案例，其中包含4地图末帧未来GT缺失。每案例5族图使用相同axis/t0 frame、GT标EvaluationOnly。为读图只显示当前最近12actor，列表保存在case manifest；预测和risk计算仍包含全actor。源pose timestamp缺失、尺寸来源近似和历史不足的unknown状态均披露，没有伪造错位/冲突案例。

## 【Chapter 3 to Chapter 5 Connection】

第三章提供history-only冻结多参与者六候选与模式概率；第五章接口可消费EgoHistory、OtherAgentPredictions、StaticMap及可选的合法route，输出EgoFutureTrajectory，评价对象独立消费ego/actorGT及map。现阶段完成接口衔接和反泄漏证据。尚无可用route，需恢复源metadata时间戳/尺寸核验并保留mask覆盖限制；现有接口无需额外规划框架，后续具体算法或训练框架由审查决定。没有实现Stage13B、Diffusion或新的语义结构。

## 【Scientific Decision】

Stage13A=CONDITIONAL_GO；ReadyForPlanningDevelopment=CONDITIONAL。关键工程gate均PASS，但完整源pose时间核验、精确footprint、route与评价覆盖有条件。

- No supplied route/goal/high-level command; legal route input design remains necessary.
- Original trainval volume unmounted; original ego_pose.timestamp verified for only 162/25307 HeadTrain keyframes. Restore original metadata for full source verification.
- Current actor annotation size unavailable outside small raw subset; class-prior footprints and approximate ego dimensions require disclosure or original-size recovery.
- Insufficient actor history and incomplete future GT remain unknown-risk/unknown-evaluation; no claim of complete traffic coverage.
- Source about 2Hz, nonreactive forecasts, no route-conditioned or closed-loop planning validation; interpolated footprints remain proxies.

预计修改ego元数据/尺寸join、合法route接口与unknown-risk评价策略；恢复已有原始metadata不需新数据集，route可能需要新增标注或单独授权协议。该结论不授权进入下一阶段。提交并push本分支，不merge main；完成后STOP，等待大脑AI检查报告。
