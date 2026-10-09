# Stage15A 模型与GT输入完整性

PredictorInitialization = PASS；GTLeakageAudit = PASS；CheckpointRecovery = PASS。实际模型仍为 Stage5A `HiVTMotionAwareDecoder`：原HiVT-64 local/global backbone、3×64类型embedding、两个64→16→64 motion residual experts、6→16→2 router、原K6 decoder与原pi评分。总参数和requires_grad参数均650403，无新增预测/语义/规划模块。[stage15a_common.py:72](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/00_manifest/stage15a_common.py#L72)复用原 fresh No-Type→Type→Motion构造及neutral初始化，fold1 seed2022与字面历史Stage5A fresh step0状态完全一致；三fold各自重复初始化一致且三seed状态不同。初始化没有任何checkpoint加载，shared state仅来自本次随机构造的内存对象。

| Fold | Seed | Parameters | FreshStateSHA256 |
| --- | --- | --- | --- |
| 1 | 2022 | 650403 | 7b824f0e51d34da3445848abfa2fb349c7a3abcf45ae59ac951def86a21cbbb4 |
| 2 | 2122 | 650403 | e7e40e1bacfd6ab86cf6385b43fd7129e87fc9157ecd60b5814400542886a006 |
| 3 | 2222 | 650403 | 6f5e99f5ac938ac890e5288b0eee4e3b05e3aa171dc471622023e782a8a93fbf |

[stage15a_common.py:103](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/00_manifest/stage15a_common.py#L103)实际构造只含历史观测、几何lane/actor边及类型的Data，positions/padding截为5历史帧，物理移除y、future_mask、target_mask、full_horizon_mask、ego_future及身份/未来标签。监督只在forward完成后进入原recovery_loss。完整原容器forward仅用于参考数值对照，实际入口始终调用allowlist。

普通参考与新allowlist的raw_prediction/mode_logits/mode_prob均maxdiff0；GT trajectory、y、未来padding/mask、target mask、ego future、未来time置NaN或翻转后实际forward仍maxdiff0。独立手工fixed-scale与原NLL loss重算均maxdiff0。Loss和原训练公式未改变，scale仍由原decoder预测。

raw为[K6,N,T12,4]，对应logits/prob为[N,6]；候选转换为[N,6,12,2]的t0-ego米坐标：local row vector@rotation^T+current_position。逐模式配对、prob归一化、显式坐标换算和原数据身份保持。恢复后原NLL checkpoint产生K6且与其state SHA绑定；没有用GT选择模型的推理输入或邻居。示例证据：[Fold1检查](03_checks/stage15a_fold1_checks.json)、[Fold2检查](03_checks/stage15a_fold2_checks.json)、[Fold3检查](03_checks/stage15a_fold3_checks.json)。

原mask监督与empty-window过滤完全保留；它们决定监督资格，不删除完整window中没有未来标签的当前参与者。历史5/12关键帧真实采样间隔存在jitter，审计按原timestamps精确对账而不修改数据。名义6s自定义协议不能据此直接等同官方固定时间点协议。

历史3206个受版本管理文件、5个Stage2C未提交文件、38个历史checkpoint逐项SHA保持。新12个checkpoint均仅供preflight，[清单](04_checkpoints/stage15a_preflight_checkpoint_manifest.json)记录SHA与约8.2MB大小，模型权重不上传Git。
