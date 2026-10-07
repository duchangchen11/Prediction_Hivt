# Stage8A-0 执行计划

本阶段所有新增代码、计划、审计记录和小型报告均位于本目录。基础 commit：35b7811ce2ec0c7dd3b252995b49724e5971b4be；分支：stage8a/future-scene-compatibility-graph。

当前只授权 Stage8A-0；完成后 STOP，等待审查。不得训练 G1/G2/G3，不进行 tiny overfit，不运行 Stage8A-1。后续完整要求保存在 stage8a_requirements.txt，不能将未来训练要求视为当前授权。

1. 冻结历史代码、Stage5A/R2 checkpoint、Stage6A 候选缓存与 Stage7A 地图语义文件；检查原有未提交文件保持原样。
2. 随机 seed2022 抽取 100 TRAIN 和 100 VAL windows，以原 batch 结构重新 forward 冻结 Stage5A，逐 actor 验证候选、logit、probability、身份、type、历史与 mask 完全相同。
3. 使用既有 ego origin/yaw 将候选从 t0 ego frame 转到地图全局坐标，验证 roundtrip、坐标轴方向、地图 segment 对齐、局部/全局检索等价。
4. 从 Stage7A 完整中心线按候选未来 polyline 距离检索。固定 radius=10m、TopM=8 distinct tokens；无覆盖时使用同一区域最近一个 token，并记录 fallback。所有 current-valid 候选均统计覆盖；另外记录 full-horizon ranking target 子集，防止隐去 context/partial 候选。
5. 构建 15D Actor-Mode node、最多8个 Stage6A 同定义邻居、逐 mode-mode 关系和 mode-map 关系。GT 不进入 ObservableWindow 或任何图构造函数；只记录推理可计算字段。图缓存引用原始候选，不复制或修改候选几何。
6. 通过数值单元检查、真实窗口 GT 污染/删除测试、GT 字段访问保护、Stage6A 邻居逐元素比较和 semantic metadata 一致性检查，审计 GT 泄漏与边界。
7. 全 TRAIN/VAL 汇总 map coverage、邻居/图规模分布，估计缓存、单 target 和 batch 内存；用固定1层 hidden64结构计算 G1/G2/G3 参数估计，无训练。
8. 若 Vehicle fallback>5% 或地图坐标不可信，判定 NOT_READY。否则按所有完整性 gate 判定 READY。撰写 Stage8A-0 报告，提交并 push 专用分支，不 merge main，然后 STOP。

固定常数：K=6，Tf=12，future horizon=6s，neighbors<=8，current distance<=50m，map tokens<=8，map radius=10m，hidden=64，未来监督温度=1m。HeadTrain630 / HeadDev70 原 scene split只引用，不重新划分；official VAL150 不用于训练或模型选择。

预计特征：Node=15；Mode-Mode edge=17（11几何/概率项+两端type onehot6）；Map node=13（冻结semantic9+局部tangent sin/cos+lane/connector标记2）；Mode-Map edge=17（8几何项+冻结semantic9），保持map node中的语义字段。邻居和地图mask显式保留；无邻居 message 置0。几何数值定义在实现/特征字典中固定，不据VAL结果调整。
