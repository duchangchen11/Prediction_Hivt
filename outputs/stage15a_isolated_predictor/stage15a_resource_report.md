# Stage15A GPU与存储资源

GPUResource = PASS；StorageResource = PASS_WITH_SHARED_STREAMING。设备NVIDIA GeForce RTX 3080，标称10.00GiB、当前CUDA可见9.654GiB，原torch/CUDA环境不升级。本阶段仅工程 preflight；未启动完整三折预测器训练，未运行 OuterTest forward、性能评价或模型选择。30次更新均为带 `PREFLIGHT_ONLY_NOT_ELIGIBLE_FOR_FORMAL_TRAINING` 标记的检查，不可作为正式模型或下一阶段初始化。

| Fold | Seed | TrainScenes | DevScenes | OuterExcluded | HeadDevExcluded | TrainWindows | DevWindows | OptimizerChecks | ForwardMean_s | StepMean_s | RecoveryMaxDiff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2022 | 378 | 42 | 210 | 70 | 9137 | 1017 | 13 | 0.049511 | 0.343626 | 0.0 |
| 2 | 2122 | 378 | 42 | 210 | 70 | 9103 | 1018 | 9 | 0.063590 | 0.529917 | 0.0 |
| 3 | 2222 | 378 | 42 | 210 | 70 | 9104 | 1011 | 8 | 0.062628 | 0.433024 | 0.0 |

15次预先转移的eval forward（3fold×5）均值0.058576s/16window batch，排除shard I/O、H2D和排序特征构造。18次primary完整train step均值0.435523s，中位0.399312s，范围0.179032–0.759281s；包含shard校验/读取、collation、H2D、forward、backward、AdamW及同步。额外5次step benchmark均值0.409253s，计入30次实际更新。

元数据选出的Fold2 InnerTrain最大16个actor-count窗口共1633演员，完整训练step成功，峰值allocated4.520GiB、reserved4.904GiB。该批次选择只依赖t0 actor_count，不读取GT error或Outer；它提供batch16的显存证据，不代表全程穷尽所有地图/边密度组合。

每个模型+AdamW+RNG检查checkpoint 8.198–8.207MB；12个共98.429MB。实际小样本候选/label缓存总2.530MB，均留本地不上传。详见 [原始计时与资源JSON](06_resources/stage15a_resource_summary.json)、[各fold汇总CSV](06_resources/stage15a_resource_summary.csv)。磁盘本次audit剩余43.192GiB。

正式方案A为3次新HiVT＋18头；从本次不同fold少量step均值延伸，3×11500步的较短历史schedule代理约3.29小时，3×21000上限的较慢fold代理约9.27小时（预测器train step部分）。这些不是收敛承诺或时间置信区间。上限126次完整Dev42评价的forward-only代理0.131小时，还要加开发shard I/O/保存。历史18头实测代理1.963小时。

保留Stage14B完整已知组件墙钟预算8.41–13.70 GPU小时作为排程参考；未来新候选分布/early stop、缓存构造/I/O和诊断未完整计时，不能把少量forward推为总FPS或保证耗时。建议单RTX3080顺序运行，安排约1–2天设备可用窗口，仍只执行登记21000更新上限；不为排程增加任何模型训练预算。

三折合并630场景/780453 target occurrence dense目标缓存代理17.067GiB；加新R2约0.334GiB、全当前上下文旧格式体积代理2.496GiB、身份记录0.186GiB、CP预算0.5GiB、bounded scratch2GiB，预计新增峰值22.582GiB，另留8GiB运行余量。当前余量满足该设计。

存储结论以共同候选被六种方法复用、streaming写一次合并缓存、避免永久保留完整chunk和merged双副本为条件。原scene shard只引用；不删除历史。若未来按旧chunk+merged双套直接保留，将增加dense缓存近一倍，当前余量不能承诺。正式缓存实现和执行前磁盘空余需再次核对，本阶段没有构造大缓存。
