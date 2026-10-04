# Stage 2B: K=6 Loss Recovery & Baseline Finalization

工作目录 /home/lrj/Prediction_Hivt；分支 stage2/vehicle-hivt-baseline。复用 ped_intent，没有安装/升级依赖。
冻结前提交 `d8b93ae7fb89f9abaefe49d08e987975f21bc5c5`；原 tiny、moving audit 和 moving_* 报告共 175 个原文件 SHA256 保持相同。新结果仅写入 k6_loss_recovery/ 与 k6_* 报告。
K=6、Th=5、Tf=12、约2s/6s、HiVT-64、原 encoder/global interactor/MLP decoder、dropout=.1、weight_decay=1e-4、地图半径50m/间距2m 均保持不变。不删除 stopped/parked，不重采样，不加入创新模块。

## 【A K6 Fixed Scale】

epochs=580；ADE=0.123128m；FDE=0.950630m；MR=0.000000；**PASS**。
同一 scene-0655 moving actor，GT endpoint 49.5348m；51 vehicle nodes + 666 lane segments 全部保留，只有该车参与监督。
regression=mean(|y−μ|)，保留原 best-sum-L2 mode selection 和 detached-soft-target mode classification。b=1，regression 不含 log(2) 常数；NLL 另记录。固定 lr=.001，seed2022，上限1000，每10 epoch评估。严格 gate ADE<.5/FDE<1，未放宽。
初始 ADE/FDE=26.388250/48.824348m；初始模型 SHA256=bd8bff1b8ca27d589f5c659d646838e390a5786d07ee43f29249600016527644。
最后 train backward head gradients：loc=3.1175，scale=0，pi=0.0102688。
![A predictions](../stage2/k6_loss_recovery/A_fixed_scale/predictions.png)

## 【Mode Audit】

| Mode | Endpoint local xy (m) | Endpoint ego xy (m) | Displacement (m) | Probability | sum L2 (m) | FDE (m) |
|---|---|---|---:|---:|---:|---:|
| 0 | 48.361740/0.042718 | 11.416306/0.544708 | 48.361759 | 0.149119 | 4.700878 | 1.179667 |
| 1 | 48.584293/-0.062936 | 11.639618/0.440667 | 48.584335 | 0.199898 | 1.477532 | 0.950624 |
| 2 | 48.419426/0.024756 | 11.474121/0.527164 | 48.419434 | 0.154931 | 4.207389 | 1.120459 |
| 3 | 48.497162/0.031347 | 11.551807/0.534317 | 48.497169 | 0.163309 | 3.585509 | 1.043809 |
| 4 | 48.459789/0.001878 | 11.514648/0.504578 | 48.459789 | 0.158531 | 4.032969 | 1.078251 |
| 5 | 48.515244/0.018968 | 11.569977/0.522069 | 48.515244 | 0.174213 | 2.901119 | 1.024534 |

best training mode=1；best FDE mode=1；一致=True。
MODE_COLLAPSE=NO；最大 mode pairwise trajectory distance=0.597049m；预先定义 collapse tolerance=.001m。单目标 collapse 可以接受，没有因此修改模型。
全部6个 mode 的完整12步 local/ego trajectory 与 probability 保存于 k6_mode_audit.json。

## 【B Warm-up】

warm-up epochs=580；ADE=0.129300m；FDE=1.028159m；PASS。
B 从 fresh seed2022 初始化，未使用 A final checkpoint。fixed-scale + classification、LR=.001，gate ADE<1/FDE<2，上限700。
raw pre-ELU scale head mean/min/max=0.084006/-1.718286/2.014637；processed raw scale mean/max=1.233169/3.015636。该分支此时不参与 regression。
warmup_checkpoint.pt 保存 weights、AdamW state 及 Torch CPU/CUDA RNG。

## 【Original NLL restored】

epochs=300；ADE=0.014022m；FDE=0.013048m；MR=0.000000；NLL=-1.828557；scale mean/max=0.116970/0.698733；**PASS**。
从 B warmup_checkpoint 继续，保留 AdamW state，LR降至1e-4，不用 scheduler，运行完整300 epochs。恢复原 free-scale LaplaceNLL 与原 mode classification。
最终 gate 预注册为 ADE<1/FDE<2，且相对 warm-up 的 ADE 增量≤.5m、FDE 增量≤1m；finite loss/gradients 必须通过。该增量定义在任何 B 结果前写入配置，用于量化‘不能明显退化’。

## 【C bounded scale】

NOT_RUN：只有 A PASS 且 B FAIL 才允许执行。

## 【Selected Protocol】

protocol=1；name=HiVT-NuScenes-Vehicle-Baseline。
K=6 fresh fixed-scale warm-up→原 Laplace NLL 严格通过，保留原 probabilistic loss；仅 optimization warm-up 为 nuScenes 约6s任务适配。
architecture unchanged；保留多模态与 mode probability。

## 【Full Tiny】

NOT_RUN：只有最终 K=6 protocol 确定后允许重跑。

## 【CV comparison】

NOT_RUN


## 【Mini】

NOT_RUN：Full Tiny 未 PASS 时禁止启动。

## 【Decision】

Stage2 baseline=FAIL；Allow Stage3 discussion=NO。本轮没有执行 Stage 3，也不 merge main。
Stop reason=None。

命令记录见 docs/stage2b_execution_commands.md；各实验 config/metrics/curve/gradient logs/figures 全部保留。Weights、optimizer states 和完整冻结副本仅保存在本机。
