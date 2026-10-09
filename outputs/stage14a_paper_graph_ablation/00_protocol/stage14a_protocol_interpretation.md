# 冻结协议字段说明

`stage14a_protocol.json` 在拟合前冻结，SHA256 由 `stage14a_registration.json` 锁定，随后不改动。为完整沿用 Stage11B 控制变量，保留了部分历史配置说明。

本次实际执行范围以这些明确的 Stage14A 字段为准：

- `formal_variants`：NG-A、NG-C，三折共六次新正式训练。
- `reused_variants`：G-A/G-C 直接复用 Stage11B A/C。
- `bootstrap.co_primary`：四个 Overall Top1FDE 受控比较。
- `decisions`：原样应用预登记点估计、校正 CI 及折间方向条件。

`R2_protocol` 记录的是历史 Stage11B R2 训练配置，包含当时“historical R2 checkpoint never loaded”的描述；Stage14A 不执行该训练，而是加载三个既有冻结 fold R2 checkpoint，用于原定 Bicycle 路由和 InnerDev 相对评分参考。

历史 `B` 损失定义、`secondary` 中的 B vs A / C vs B / C vs R2 / A vs R2 列表只保留作为原 Stage11B 模板来源说明。本阶段不训练或评价新增 B 模型、不新增这些比较；实际统计仅执行 `bootstrap.co_primary` 四比较及预登记描述性交互。

CSV 的 `ComparisonID=A/B/C/D` 对应用户指定的四个受控对比标签，不能与历史模型的 A/B/C 损失变体混淆。模型轴始终为 NG-A、NG-C、G-A、G-C。

所有解释均不改变冻结协议、损失代码或判定门槛，不构成事后选择新比较。
