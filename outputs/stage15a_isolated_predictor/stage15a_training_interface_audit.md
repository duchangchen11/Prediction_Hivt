# Stage15A 训练入口审计

TrainingInterface = PASS。新增入口显式接收 `--fold --seed --training-scenes --development-scenes --output-dir`；三折seed固定2022/2122/2222。默认describe只核对配置与元数据，preflight最多运行已登记的小样本步骤，formal明确拒绝。CLI三折describe均exit0，formal拒绝均exit1且没有启动优化。代码：[stage15a_train.py:261](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/02_training/stage15a_train.py#L261)。

冻结配置：batch16、FP32、AdamW，原 named-module decay/no_decay 分区分别wd0.0001/0；fixed-scale warm-up5000步/lr0.001，原NLL最多16000步/lr0.0001，总上限21000，开发每500步、NLL连续5次无改善停止。选择只用42开发scene的full-horizon Overall minFDE6，严格改善；最终只用NLL best。完整循环已实现并由Stage15A关闭的授权gate保护：[stage15a_train.py:116](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/02_training/stage15a_train.py#L116)。本阶段没有改变500步验证间隔，也没有用早期小样本指标代替正式选择。

实测每fold4次warm+2次NLL primary检查，6次replay对照、5次训练计时和1次压力更新，合计30。开发4窗的probe在正式500步日程之外，checkpoint标记仅供检查；其 best_step=5002是NLL协议索引，不能解释为已执行5002更新。

[stage15a_train.py:28](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/02_training/stage15a_train.py#L28) 保存模型、AdamW、phase/严格best/坏验证次数、scene sampler epoch+cursor以及CPU/CUDA/Python/NumPy RNG；[stage15a_train.py:39](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/02_training/stage15a_train.py#L39) 严格绑定fold、seed、列表SHA、输出目录和fitting源码SHA，禁止历史模型或其他fold替代。三折恢复后下一步模型、AdamW moments、loss/梯度、batch身份、游标和全部RNG逐位一致，maxdiff0。NLL checkpoint也完成严格load与候选forward核验。

[stage15a_train.py:94](https://github.com/duchangchen11/Prediction_Hivt/blob/stage15a/isolated-predictor-preflight/outputs/stage15a_isolated_predictor/02_training/stage15a_train.py#L94) 从本实验自己的warm best恢复模型/优化器/RNG，仅将LR0.001改为0.0001，AdamW moments与其他group选项保持；NLL sampler游标重置，epoch偏移100000。原SceneSampler算法逐scene/窗shuffle复用；新按步重建loader只模拟原每新epoch一次CPU base-seed抽样，防止恢复时额外消耗RNG。确定性算法与 `CUBLAS_WORKSPACE_CONFIG=:4096:8`用于可复现检查，batch、损失、模型和预算保持原定义。

边界模拟验证严格tie不改善、NLL patience5、warm固定5000及21000总上限；这些模拟不训练。有限梯度、type embedding、所有expert/router参数在4步内都收到非零梯度。任何隔离、非有限或恢复失败即退出；没有历史模型填补、自动换seed/batch或额外训练预算。

[冻结协议](00_manifest/stage15a_protocol.json)、[初始源码登记](00_manifest/stage15a_code_registration.json)、[CLI检查](02_training/stage15a_cli_smoke.json)。正式启动仍需大脑AI审查与单独授权；下一阶段必须登记自己的formal输出/源码并从随机初始化开始，不能打开本阶段gate后复用这些tiny权重。
