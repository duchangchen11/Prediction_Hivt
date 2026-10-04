# Stage 2：nuScenes Vehicle-only HiVT Baseline

**当前状态：Tiny Overfit FAIL。已按用户要求停止后续训练，mini baseline未运行，Stage3=NO。**

## HiVT 来源

repo=https://github.com/ZikangZhou/HiVT
commit=6876656ce7671982ebdc29113aaaa028c2931518
原始dataset=Argoverse1.1；默认history/future=20/30；K=6；Apache-2.0。

核心结构保持：YES。HiVT-64、4层temporal encoder、3层global interactor、8 attention heads、原始MLP decoder和LaplaceNLL+soft-target mode CE。

修改位置：models/hivt_runtime（相对import、PyG batching签名、PyTorch causal keyword）；models/hivt_nuscenes.py（原生训练wrapper、clone避免累积旋转）；datasets/nuscenes_hivt_vehicle_dataset.py与lane adapter。无创新模块。

## 环境

env=ped_intent；torch=2.5.1+cu124；cuda runtime=12.4；CUDA available=True；GPU=RTX3080 10GB。
torch_geometric=2.6.1；torch_scatter=2.1.2+pt25cu124；torch_sparse=0.6.18+pt25cu124。
lightning=未安装，原生PyTorch loop无需它。现有包未变更，无新环境。

## trainval

TRAJECTORY_METADATA_READY=YES；HD_MAP_READY=YES；LIDAR_REQUIRED_FOR_STAGE2=NO；readiness=PASS。

源目录=/media/lrj/54926A1D926A0438/nuscenes-trainval。因本机16GB内存，流式读取源metadata并抽取5个完整scene，之后通过NuScenes构造器读取缓存。850个源scene中的随机5个scene链、sample链、annotation双向链、ego pose、map location检查通过，四个HD Map API均可读。没有点云文件读取/下载，没有宣称全量传感器数据验收。

## 数据

Th=5；Tf=12；vehicle definition=vehicle.*排除vehicle.bicycle（包含motorcycle）。ego仅作为坐标/context字段，不作预测target。

train windows=146；val windows=48；test windows=50。沿用Stage1场景6/2/2划分，无交集；未在mini test上做模型实验。

lane representation=lane/connector centerline有向segments，2m间距；radius=50m/actor（upstream默认）。intersection/turn/control输入为未指定的零常量，占位而非真实语义。当前t0 ego坐标不变，agent内部旋转使用历史运动heading（原始规则），另保留annotation heading。

历史displacement保持upstream定义，不加入time embedding；实际timestamp保留用于CV与horizon确认。100个随机窗口检查及所有244个窗口检查通过；NaN=0、Inf=0、index范围与batch通过。

full-horizon定义=全部12个未来关键帧均有效；实际窗口未来长度范围5.749241–6.100788s，约6s，不宣称严格6.000s。partial-future单列。

## Tiny Overfit

windows=16；epochs=240；lr=5e-4，cosine退火到0；batch size=4；seed=2022。

initial loss=10.805600；final loss=1.576870（均为eval mode的全tiny确定性评估，训练step loss另见CSV）。
initial regression loss=8.896064；final regression loss=0.017860。
initial classification loss=1.909536；final classification loss=1.559010。

| full-horizon 236 actor-windows | initial | final |
|---|---:|---:|
| minADE_K | 7.700448 | 5.571721 |
| minFDE_K | 12.866848 | 11.274157 |
| MR_K | 0.275424 | 0.275424 |

OVERFIT=FAIL。预先固定gate要求minADE<1m、minFDE<2m且降到初始的25%以内；未修改gate来追认成功。

5个初始预测、5个最终预测、5个最差移动目标案例和训练曲线已保存。Checkpoint仅在本机。

## Mini Baseline

未运行。Tiny gate失败后禁止启动mini训练；mini的正式CV vs HiVT验证对比、验证成功/失败案例均未生成。

为诊断而在同一组tiny训练窗口、同一236个full-horizon actor-window上比较（不能替代mini验证结果）：

| Method | K | ADE / minADE (m) | FDE / minFDE (m) | MR |
|---|---:|---:|---:|---:|
| CV | 1 | 1.403184 | 3.194129 | 0.275424 |
| HiVT tiny | 6 | 5.571721 | 11.274157 | 0.275424 |

minADE_K采用官方HiVT/Argoverse的best-final-FDE mode后计算ADE；独立各mode minimum-ADE另存independent_minADE_K，禁止混用。MR threshold=2m；MR_K定义=min_k(endpoint displacement)>2m；strict >而非>=；保留全部K=6。full/partial分别计数与汇总。

## trainval smoke test

5个正式scene图读取。batch size1：5个vehicle节点；batch size2：19个vehicle节点。
forward=PASS；loss finite=PASS；backward=PASS；gradient finite=PASS；optimizer step=PASS（参数实际变化）。
pred_traj=[N_vehicle,6,12,2]；mode_prob=[N_vehicle,6]且各row和为1。

## 问题

- 移动目标65个full-horizon actor-window的mean minADE/minFDE=20.059934/40.694905m；静止/小位移171个为0.064505/0.090831m。主要错误集中于移动目标。
- GT相对位移最大108.760704m，预测位移最大仅8.666229m，出现严重位移低估。NLL下降不能据此认为轨迹记忆成功。
- 坐标反变换后的误差与agent局部误差最大差1.52587891e-05m；未发现rotation反变换不一致。shape/mask/batch/source/loss一致性检查均通过，但这些检查不能证明没有任何其他训练问题。
- 训练schedule最后LR接近0；优化不足只是待检验假设，未把它当作已证实唯一原因。下一步优先检查location/scale head梯度、multimodal选择与学习率，再做tiny-only复验。
- 未来为12个实测关键帧，未重采样到严格6.000s；mini不是论文最终性能，需后续正式数据/划分/时序协议。

## Git

branch=stage2/vehicle-hivt-baseline；保留独立audit、vehicle adapter、lane adapter、tiny验收里程碑。最终远程SHA见聊天汇报与本机outputs/debug/stage2_git_final.txt。raw data、processed graphs、cache与checkpoint不上传。

## 结论

**Stage3 Multi-Type HiVT=NO。**

当前卡点是tiny overfit失败；mini训练及正式对比尚未执行。按照用户的失败停止规则，本轮停止在诊断与结果保存，没有扩大训练或加入新模块。
