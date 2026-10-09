# Stage14B Capacity-Matched NoGraph 对照

固定一次结构：原自身15→64→64节点编码、LayerNorm64、64→32→1评分头，加自身64→128→64 ReLU residual adapter。实际forward读取 `local_nodes[:,0]` 和原logits，真实邻居、interaction edges、map输入不参与计算；无空参数或借用G-C权重。源码见 [模型](02_models/stage14b_matched_nograph.py)、[结构检查](01_preflight/stage14b_model_audit.py)、[真实输入及梯度检查](01_preflight/stage14b_capacity_preflight.py)。

| 模型 | 实际可训练参数 | 相对G-C差额 |
| --- | ---: | ---: |
| NG-C | 7425 | 16641 |
| G-C | 24066 | 0 |
| Matched-NG-C | 24001 | 65，0.2701% |

新增16576参数均在真实目标路径参与计算，四个adapter参数张量有实质非零C梯度。共享最终标量bias对softmax共同平移不识别，G-C也含这个相同遗留维度；不为凑数加入额外无效参数。所有输入保持原字节、step0与原logits一致；非零评分探针邻居/边/地图改变差0，目标改变有响应；重新构造10个原始InnerTrain window并污染未来GT/mask后forward差0。证据见 [preflight](01_preflight/stage14b_capacity_preflight.json)、[model integrity](01_preflight/stage14b_model_integrity.json)、[原始GT重放](01_preflight/stage14b_gt_poison_windows.csv)。

结构、参数、输入输出、概率归一、共享初始化、有限梯度、未来GT隔离全部PASS。原Fold1 128个target、seed2022、300次更新tiny C降幅 83.401239%（门槛80%），未保存tiny checkpoint，不复用tiny权重。见 [tiny审计](01_preflight/stage14b_tiny_audit.json)。

全部检查通过后按授权执行3个小头训练；没有训练HiVT、G-C或历史模型。与原NG-C/G-C共享候选、折分、仅InnerTrain标准化、归一化C、AdamW、FP32、128×8、carry、每epoch样本顺序与严格InnerDev选择规则，原协议不变。完整原AST训练体只改变新artifact前缀、实际参数断言，并添加历史order hash核验。每折仍有Bicycle监督目标，最终Bicycle固定R2路由。见 [runner](03_training/stage14b_train.py)、[训练汇总](03_training/stage14b_training_summary.csv)。

三fold选中epoch=[15, 14, 2]，执行epoch=[20, 19, 7]，实测三头wall=0.411193小时。全部三checkpoint先冻结再统一读OuterTest，SHA清单见 [冻结门](04_checkpoints/stage14b_all_frozen.json)。本阶段不是端到端独立验证；历史候选生成器见过全部TRAIN700，不能通过新头消除此事实。

## 同身份统一评价

| Group | Model | Count | minFDE6 | Top1ADE | Top1FDE | HitRate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Overall | NG-A | 260151 | 1.258135 | 1.073364 | 2.380943 | 0.385914 |
| Overall | NG-C | 260151 | 1.258135 | 0.994880 | 2.254734 | 0.419872 |
| Overall | G-A | 260151 | 1.258135 | 1.089466 | 2.384061 | 0.385069 |
| Overall | G-C | 260151 | 1.258135 | 0.973628 | 2.199512 | 0.420121 |
| Overall | Matched-NG-C | 260151 | 1.258135 | 0.995979 | 2.258354 | 0.421390 |
| Vehicle | NG-A | 191026 | 1.381594 | 1.225630 | 2.748780 | 0.416299 |
| Vehicle | NG-C | 191026 | 1.381594 | 1.123940 | 2.587120 | 0.455598 |
| Vehicle | G-A | 191026 | 1.381594 | 1.244684 | 2.751081 | 0.410766 |
| Vehicle | G-C | 191026 | 1.381594 | 1.096568 | 2.516273 | 0.451991 |
| Vehicle | Matched-NG-C | 191026 | 1.381594 | 1.125898 | 2.593195 | 0.456796 |
| Pedestrian | NG-A | 66145 | 0.907948 | 0.645696 | 1.347066 | 0.296545 |
| Pedestrian | NG-C | 66145 | 0.907948 | 0.630692 | 1.317552 | 0.316607 |
| Pedestrian | G-A | 66145 | 0.907948 | 0.653994 | 1.352684 | 0.309199 |
| Pedestrian | G-C | 66145 | 0.907948 | 0.626157 | 1.304968 | 0.328007 |
| Pedestrian | Matched-NG-C | 66145 | 0.907948 | 0.629361 | 1.314246 | 0.319117 |
| MovingVehicle | NG-A | 41728 | 5.327164 | 4.781322 | 10.899423 | 0.220859 |
| MovingVehicle | NG-C | 41728 | 5.327164 | 4.344736 | 10.208203 | 0.207175 |
| MovingVehicle | G-A | 41728 | 5.327164 | 4.860093 | 10.905267 | 0.250216 |
| MovingVehicle | G-C | 41728 | 5.327164 | 4.227418 | 9.906479 | 0.224238 |
| MovingVehicle | Matched-NG-C | 41728 | 5.327164 | 4.354228 | 10.238310 | 0.203844 |

### 预登记的车辆状态透明补充

OtherVehicleState采用Vehicle扣除互斥的Moving/Stopped/Parked，4315目标、207scene，原t0 motion_state均unknown。既有main groups没有这项，因此另从冻结输出离线汇总；不改分组函数、主表、主要比较或任何训练/forward。完整状态表如下。

| Group | Model | Count | minFDE6 | Top1ADE | Top1FDE | HitRate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OtherVehicleState | NG-A | 4315 | 1.793360 | 1.353440 | 3.117549 | 0.434994 |
| OtherVehicleState | NG-C | 4315 | 1.793360 | 1.302642 | 3.060851 | 0.454461 |
| OtherVehicleState | G-A | 4315 | 1.793360 | 1.365033 | 3.072165 | 0.412051 |
| OtherVehicleState | G-C | 4315 | 1.793360 | 1.225889 | 2.865023 | 0.463036 |
| OtherVehicleState | Matched-NG-C | 4315 | 1.793360 | 1.285001 | 3.020755 | 0.469293 |
| StoppedVehicle | NG-A | 23044 | 0.811419 | 0.488668 | 1.255360 | 0.458601 |
| StoppedVehicle | NG-C | 23044 | 0.811419 | 0.484244 | 1.252048 | 0.476740 |
| StoppedVehicle | G-A | 23044 | 0.811419 | 0.484530 | 1.238519 | 0.406613 |
| StoppedVehicle | G-C | 23044 | 0.811419 | 0.478716 | 1.236939 | 0.466455 |
| StoppedVehicle | Matched-NG-C | 23044 | 0.811419 | 0.484716 | 1.252383 | 0.479387 |
| ParkedVehicle | NG-A | 121939 | 0.124584 | 0.143606 | 0.228774 | 0.474524 |
| ParkedVehicle | NG-C | 121939 | 0.124584 | 0.136337 | 0.214693 | 0.536654 |
| ParkedVehicle | G-A | 121939 | 0.124584 | 0.146872 | 0.235169 | 0.466446 |
| ParkedVehicle | G-C | 121939 | 0.124584 | 0.137365 | 0.216743 | 0.526804 |
| ParkedVehicle | Matched-NG-C | 121939 | 0.124584 | 0.136691 | 0.215263 | 0.538646 |

Other G-C−Matched Top1FDE=-0.155732m，探索性95%CI[-0.252722,-0.055661]。Parked G-C−Matched点差+0.001480m，95%CI[-0.001002,+0.004114]；G-C−NG-C点差+0.002050m，95%CI[-0.000186,+0.004484]。Parked有点退化，区间跨0，不能写成显著损害或全组均改善。Stopped G-C−Matched点差-0.015444m，探索性95%CI[-0.032228,-0.000610]。补充仅用原配对2000/seed2022权重，均探索性，不升格主要推断。[状态补充CSV](05_evaluation/stage14b_state_supplement.csv)、[来源/CI审计](05_evaluation/stage14b_state_supplement.json)。

新主要容量比较 G-C−Matched-NG-C Overall Top1FDE=-0.058842 m；描述95%CI=[-0.074483, -0.044475]，family 3 调整98.3333%CI=[-0.078546, -0.041768]，负向fold=3/3。`CapacityAlternativeNotSufficient=SUPPORTED`。见 [原配对bootstrap](06_statistics/stage14b_bootstrap_ci.csv)、[判定](06_statistics/stage14b_capacity_evidence.json)。两个历史主要contrast重新计算统一的 family 3 CI 并明确标为已知；仅容量contrast是新增比较，不替换Stage14A family 4 原判定。

在这套历史固定候选与训练规则下，目标自身MLP增加近似相同参数量不足以复现G-C表现。该结果减轻“仅多参数即可解释”这一替代解释，仍不证明消息的因果效果，也不消除深度、归纳偏置和优化差异。

HitRate与误差幅度并非同一指标：Overall G-C HitRate=0.420121低于Matched-NG-C的0.421390，Vehicle亦为0.451991低于0.456796，尽管对应Top1FDE更低。不能将主要Top1FDE改善概括为所有指标全面改善；完整表保留此差异。

| Model | cached CUDA head mean ms/actor |
| --- | ---: |
| NG-A | 0.004744 |
| NG-C | 0.004755 |
| G-A | 0.010234 |
| G-C | 0.010222 |
| Matched-NG-C | 0.006492 |

同一RTX3080/FP32、128batch、固定1024target池、20warm-up与20轮转重复/三fold，只包含预加载设备输入的头forward，排除HiVT、特征构造、传输、路由；不是整体FPS。近似参数匹配不等于FLOPs/延迟匹配。见 [效率CSV](05_evaluation/stage14b_efficiency.csv)、[scope审计](05_evaluation/stage14b_efficiency_audit.json)。候选/oracle几何及Bicycle输出全部逐bit一致；完整身份/GT分离检查见 [评价审计](05_evaluation/stage14b_identity_audit.json)。
