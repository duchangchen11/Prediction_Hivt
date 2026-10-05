# Stage3B：Multi-Type HiVT + Type Embedding

本轮仅检验 additive Type Embedding。全部正式比较来自相同冻结 official VAL，单次 seed2022 训练；结论限于本数据、协议和种子。

【Model Change】

Type embedding dim=64；nn.Embedding(3,64)，0 vehicle / 1 pedestrian / 2 bicycle。

Insertion point=LocalEncoder 输出之后、GlobalInteractor 之前。

Fusion=local actor embedding + E_type(agent_type)；decoder 使用增强后的 local representation。

Additional parameter count=192；公共参数645809、总参数646001。

共享 step0 初始化逐参数完全一致，max_abs_diff=0；未加载训练过的 No-Type 权重。A-A、TemporalEncoder、A-L、GlobalInteractor 公式和 decoder 保持原实现。

【Training】

Warm-up steps=5000；NLL steps=10500；executed global step=15500。

Best global step=13000；Stop reason=patience_5。

converged_by_patience=True；stopped_by_budget=False。

warm-up 执行5000更新，NLL恢复本模型 warm-up best step=3500 的模型、AdamW 与 RNG；仅按原 Protocol1 将 LR 从0.001切换0.0001，NLL sampler cursor重置。

batch16、Th5、Tf12、K6、embed64；每500更新完整150scene VAL，按 full-horizon overall minFDE6 严格改善选择 post-update NLL best；最大 global21000，patience5。

frozen YAML 除 type_embedding=True 完全一致。原 YAML 的历史 max_steps 字段不改写，执行预算在训练前单独登记为5000+16000，与冻结 Stage3A 实际总预算一致。

官方 train700 / val150；VAL3603有监督窗口（候选3619，空监督16）；test unused。相同850scene frozen shards，未重新预处理。

曲线未经平滑；loss点是前500个真实batch更新的均值，VAL点均为当时完整 official VAL。

训练代码commit=0358a4434c50d6690b5ea4bae61012ccf3b24654；分支=stage3b/type-embedding；No-Type frozen commit=baf8b59b44047657dca0a64460dd4005c323af76。

【Main Results】

指标为 meters；Delta=Type−No-Type，负值有利于 Type。minADE6沿用 best-FDE mode 的ADE，另存独立minADE诊断；MR6为endpoint error>2m；Top1为最高概率mode；NLL沿用original best-summed-L2 mode Laplace定义。

| Group | Count | No-Type ADE | Type ADE | ΔADE | No-Type FDE | Type FDE | ΔFDE | No-Type MR | Type MR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 54990 | 0.670959 | 0.666406 | -0.004553 | 1.364500 | 1.343260 | -0.021239 | 0.166121 | 0.163102 |
| Vehicle | 42332 | 0.711687 | 0.708347 | -0.003340 | 1.449723 | 1.431914 | -0.017809 | 0.167037 | 0.165761 |
| Pedestrian | 12002 | 0.534771 | 0.525708 | -0.009063 | 1.077623 | 1.043875 | -0.033748 | 0.163723 | 0.154891 |
| Bicycle | 656 | 0.534423 | 0.534082 | -0.000341 | 1.113615 | 1.099908 | -0.013707 | 0.150915 | 0.141768 |
| vehicle.moving | 10461 | 2.314935 | 2.328189 | 0.013254 | 4.826872 | 4.791532 | -0.035340 | 0.600803 | 0.597075 |

完整主表含vehicle.stopped、vehicle.parked、unknown、Top1ADE6和NLL，见 ../06_tables/stage3b_type_embedding_main_results.csv。

【Top1 Ranking】

| Group | No-Type Top1FDE6 | Type Top1FDE6 | ΔTop1FDE |
|---|---:|---:|---:|
| Overall | 2.723308 | 2.687984 | -0.035324 |
| Vehicle | 3.094630 | 3.050323 | -0.044306 |
| Pedestrian | 1.466291 | 1.457706 | -0.008585 |
| Bicycle | 1.759772 | 1.814865 | 0.055093 |
| vehicle.moving | 10.799044 | 10.734859 | -0.064186 |
| Vehicle >5m | 12.193457 | 12.105361 | -0.088097 |
| Pedestrian >5m | 1.750384 | 1.743370 | -0.007014 |
| Bicycle >5m | 7.452160 | 7.759411 | 0.307251 |

Overall oracle ΔFDE=-0.021239 m；Top1 ΔFDE=-0.035324 m。分别衡量候选轨迹覆盖误差与最高概率模式误差；Top1改善不替代预登记的primary oracle FDE支持判断。

Top1FDE可靠下降，支持固定两模型下的最高概率模式误差改善。 该结论基于Top1与oracle误差对照，不能将Top1变化直接视为oracle coverage变化。

【Nontrivial Motion】

阈值严格按 GT endpoint displacement 分组，Count 为可能重叠的 actor-window；UniqueInstances / UniqueScenes 单独统计。

| Group | Count | Instances | Scenes | minADE6 | minFDE6 | MR6 | Top1ADE6 | Top1FDE6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vehicle >5m | 9744 | 853 | 137 | 2.647145 | 5.579887 | 0.676416 | 5.177695 | 12.105361 |
| Pedestrian >1m | 8810 | 825 | 113 | 0.671712 | 1.361468 | 0.210443 | 0.879428 | 1.862172 |
| Pedestrian >5m | 7581 | 716 | 110 | 0.659635 | 1.340389 | 0.205646 | 0.838939 | 1.743370 |
| Bicycle >1m | 155 | 15 | 13 | 1.935424 | 4.255760 | 0.593548 | 3.043487 | 6.993361 |
| Bicycle >5m | 132 | 12 | 11 | 2.117075 | 4.625799 | 0.598485 | 3.369153 | 7.759411 |

Bicycle >5m：limited unique bicycle instances/scenes；132个重叠actor-window只有12个独立instances、11个scenes，仅描述性比较，不作强显著性结论。

有效运动FDE配对消融：

| Group | No-Type FDE | Type FDE | ΔFDE |
|---|---:|---:|---:|
| Vehicle >5m | 5.630972 | 5.579887 | -0.051085 |
| Pedestrian >5m | 1.372662 | 1.340389 | -0.032273 |
| Bicycle >5m | 4.673726 | 4.625799 | -0.047927 |

【Paired Bootstrap】

严格配对full-horizon54990、partial30037，共85027。identity、node/type/motion/valid steps一致；GT float32完整trajectory SHA和future mask bits由未变更的No-Type冻结graph底账核对。No-Type旧CSV无逐时刻GT，因此不冒称CSV自带完整GT，底账来源与150VAL shard hashes保留。

paired scene-cluster percentile bootstrap：150个official VAL scenes、1000 replicates、seed2022；每次重采样完整scene后pool全部actor-window delta，以窗口等权聚合。所有组使用同一scene draws；95%CI为2.5/97.5百分位。

CI描述固定两模型在scene采样下的配对差异，未包括不同训练seed的变异；一个primary endpoint，其余预登记亚组为辅助证据，未作多重比较校正。

| Group | Metric | Δ Type−No-Type (m) | Scene-cluster 95%CI (m) |
|---|---|---:|---|
| overall | minADE6 | -0.004553 | [-0.008803, -0.000444] |
| overall | minFDE6 | -0.021239 | [-0.034236, -0.010796] |
| overall | Top1FDE6 | -0.035324 | [-0.064410, -0.008999] |
| vehicle | minADE6 | -0.003340 | [-0.008784, 0.001701] |
| vehicle | minFDE6 | -0.017809 | [-0.033765, -0.004626] |
| pedestrian | minADE6 | -0.009063 | [-0.012720, -0.005179] |
| pedestrian | minFDE6 | -0.033748 | [-0.042389, -0.025997] |
| vehicle.moving | minFDE6 | -0.035340 | [-0.096121, 0.018674] |
| Vehicle >5m | minFDE6 | -0.051085 | [-0.121162, 0.004746] |
| Pedestrian >5m | minFDE6 | -0.032273 | [-0.040993, -0.023804] |
| bicycle | minADE6 | -0.000341 | [-0.015813, 0.019053] |
| bicycle | minFDE6 | -0.013707 | [-0.048454, 0.025442] |

【Qualitative】

案例按预登记改善/退化排序有目的选择，不能代表总体效果；同actor、同GT、同坐标范围，原VAL16-window batch重放，八项误差均与source CSV核验<1e-4。

improvement=stage3b_vehicle_improvement_case_001；class=vehicle，ΔFDE=-9.076766 m；source ../04_evaluation/cases/stage3b_vehicle_improvement_case_001.json。

improvement=stage3b_pedestrian_improvement_case_001；class=pedestrian，ΔFDE=-2.231484 m；source ../04_evaluation/cases/stage3b_pedestrian_improvement_case_001.json。

degradation=stage3b_degradation_case_001；class=vehicle，ΔFDE=8.935171 m；source ../04_evaluation/cases/stage3b_degradation_case_001.json。

V-P GT proximity=stage3b_vp_interaction_case_001；class=vehicle，ΔFDE=-4.457641 m；source ../04_evaluation/cases/stage3b_vp_interaction_case_001.json。

V-P图为t0<=20m且同步future GT最近距离<=5m的接近候选，不证明模型因果交互机制。所有future轨迹显示12个原始marker；inset仅用于转弯、GT接近或重合，bbox由两模型GT/Best/Top1最后6点共同计算。

【Embedding Auxiliary Analysis】

三类训练后type vectors仅展示L2 norm与cosine；无t-SNE，不将embedding距离解释为物理语义。

| Type | L2 norm |
|---|---:|
| vehicle | 8.170973 |
| pedestrian | 8.845589 |
| bicycle | 8.357695 |

cosine(vehicle, pedestrian)=-0.043218。

cosine(vehicle, bicycle)=0.201644。

cosine(pedestrian, bicycle)=0.007144。

【Scientific Decision】

Type Embedding=SUPPORTED。

Reason=Overall delta FDE <0 and scene-bootstrap CI95 entirely <0.

Stage3B=PASS。

Ready for Stage4 Type-aware Dynamic Interaction=YES。

Stage3B PASS 表示完整单变量协议、配对和真实证据通过验收，独立于效果正负。Ready仅按训练前保守规则生成，必须交用户审查；本轮在Stage3B结束，没有执行Stage4、type-aware动态交互、类别重权、Intent Head或main合并。

Overall FDE变化为-0.021239 m（-1.557%）。统计支持对应固定两模型的primary整体指标，不等价于所有运动类型均可靠改善。

vehicle.moving ΔFDE CI95=[-0.09612054591443679, 0.018673794592280846]；Vehicle >5m ΔFDE CI95=[-0.12116189192090221, 0.0047459714872683445]。其运动亚组解释依据这些CI是否跨0，不能用vehicle overall代替。

Pedestrian >5m ΔFDE CI95=[-0.04099298472725452, -0.023803655433449998]；这是有效运动目标的单独证据。

Bicycle >5m 的ADE变化0.018007 m，FDE变化-0.047927 m，Top1FDE变化0.307251 m；结果并非各指标一致改善，且仅有12instances/11scenes，仍限描述性。

【Audit and Artifacts】

原有受保护文件846份与scene shards850份SHA全部未变更；核心训练源码SHA与预登记完全一致；31个完整VAL点核验通过。

图件来自真实CSV/JSON，Python导出PNG300dpi/PDF/SVG，SVG/PDF保留可编辑文字；source与export hashes、markers、bbox和视觉审计归档。

大型checkpoint、逐actor错误CSV、GT底账与原始shards仅本地保存，不上传Git。GitHub仅提交本阶段新代码、配置、报告、汇总表、小型source JSON、图件和运行日志，Stage2C未提交重绘文件原样保留。

Final checkpoint SHA256=461dd9fc8ccc4a03bb72e6a92f3e328e34e1fefb6ebf890ff87eedffaa718547。

No-Type checkpoint SHA256=e432906f8938f5e37e9f16816a90dac29734762d6e1cde51376aa2b7ef7648f8。

Type config SHA256=5afcf18547930f57af16eb7770709333a663365693b948079a945bde9bb3fa91。

本轮无额外训练seed或重复超参数实验，结论需在未来独立实验中检验泛化。
