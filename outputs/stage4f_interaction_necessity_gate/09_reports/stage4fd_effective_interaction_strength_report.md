# Stage4F-D：Effective Interaction Strength / Gate Compensation Audit

## Question

Does the relation branch compensate for the necessity gate? 本轮研究 g_i 下降是否被 raw relation amplitude 放大抵消，而不是再次选择 checkpoint 或搜索变体。结论仅描述 seed2022 的两份冻结模型。

## Frozen Models

- Stage4A checkpoint SHA256：`ff25266ba6a44630cfec01ae1596ef7de05f0d71f06f9676f6ea7ef327775ecb`。
- Stage4F checkpoint SHA256：`89f2bc886bd5e1cc2a78a05e07c9761d3317f4677237a458ea35450e4f56e110`。

基于 commit `514f3809e837626b6976b0f5b17be65d465ca9e6`，两模型全程 eval/no_grad，参数 requires_grad=False，未调用 backward、optimizer 或任何训练。参数/缓冲位级哈希前后相同，全部 grad=None。Stage3B SUPPORTED、Stage4A/Stage4F NOT SUPPORTED、Ready Reliability Head=NO 均保留。

## Pairing Audit

Official VAL150、3603 windows、batch16且shuffle=False。实际边严格配对 3,058,206 条，对应3layers×8heads共 73,396,944 个 edge-layer-head observations；EdgeKey为scene/sample/source-instance/target-instance，顺序哈希 `5de4d5e961027808bad820f578f25b5cfb5a3006f9110168024fd30d991c565a`。每个batch逐层确认源/目标、pair ID、index和完整边顺序一致；每个scene-window均验证 current-valid complete directed graph 的 n(n−1) 条非自身边，无跨scene边。

GlobalInteractor 是完整当前有效图。50m只限制LocalEncoder，以及 Stage4F gate 输入中从已有边筛出的近邻统计；本轮没有修改任何 actor、edge、padding、type、GT 或map。两模型从同一原始batch克隆输入，原始所有张量/身份字段的fingerprint保持不变。

全current-valid共 91,092 actor-windows，full54990、partial30037、context-only6065。其中28个无incoming edge，18个为full vehicle target（moving15、parked3）；其gate保留，但edge/attention均值未定义，统计中明确排除。因此主强度表Overall TargetCount=54972，incoming full-target edges=1803580。这未改变原预测评价的54990/30037数量。

正式 message AST 仅插入观察语句，删除这些语句后与原AST完全一致；原 q/k/value、加法、PyG softmax、dropout、聚合完全保留。固定真实CPU batch插桩前后 raw/logits/prob差均为0。正式VAL中 captured final_A=base_A+b_A、final_F=base_F+g_i b_F，max diff均为0；真实dropout入口alpha与同final的PyG softmax复核最大差小于1e-6，微小差异来自原CUDA scatter的非确定归约。

## Overall Interaction Strength

| Group | Weighting | TargetCount | GateMean | A_MeanAbsRawBias | F_MeanAbsRawBias | F_MeanAbsEffectiveBias | RawAmplification | EffectiveRetention | CompensationIndex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | EdgeWeighted | 54972 | 0.205159 | 9.491454 | 1.239667 | 0.251896 | 0.130609 | 0.026539 | 0.129359 |
| Overall | TargetWeighted | 54972 | 0.207092 | 9.473241 | 1.239284 | 0.254702 | 0.130819 | 0.026886 | 0.129828 |
| Vehicle | EdgeWeighted | 42314 | 0.198330 | 9.988014 | 1.271999 | 0.249701 | 0.127353 | 0.025000 | 0.126053 |
| Vehicle | TargetWeighted | 42314 | 0.201518 | 9.966036 | 1.273221 | 0.254456 | 0.127756 | 0.025532 | 0.126700 |
| Pedestrian | EdgeWeighted | 12002 | 0.227803 | 8.154476 | 1.168172 | 0.266482 | 0.143255 | 0.032679 | 0.143453 |
| Pedestrian | TargetWeighted | 12002 | 0.227093 | 8.158694 | 1.168066 | 0.265610 | 0.143168 | 0.032555 | 0.143358 |
| Bicycle | EdgeWeighted | 656 | 0.207370 | 1.702871 | 0.349307 | 0.072606 | 0.205128 | 0.042637 | 0.205609 |
| Bicycle | TargetWeighted | 656 | 0.200720 | 1.737109 | 0.353253 | 0.071010 | 0.203357 | 0.040879 | 0.203659 |

EdgeWeighted对应每条edge/layer/head等权；TargetWeighted先在每target内部平均全部incoming edges和heads/layers，再等权平均target。前者attention及gate也按target degree加权，后者是标准等权target-layer-head平均L1/entropy/switch。两套结果完整保留，不能将edge-weighted gate均值混用为actor均值。所有比值均为对应加权均值的比值，eps=1e-8；CompensationIndex=EffectiveRetention/(GateMean+eps)，1仅为raw不变的描述性参考。

| Group | Weighting | A_MeanAbsBase | F_MeanAbsBase | A_NormalizedInteraction | F_NormalizedInteraction | NormalizedInteractionRetention |
| --- | --- | --- | --- | --- | --- | --- |
| Overall | EdgeWeighted | 8.241723 | 7.041164 | 1.151635 | 0.035775 | 0.031064 |
| Overall | TargetWeighted | 8.324470 | 7.028902 | 1.137999 | 0.036236 | 0.031842 |
| Vehicle | EdgeWeighted | 8.696755 | 6.634274 | 1.148476 | 0.037638 | 0.032772 |
| Vehicle | TargetWeighted | 8.786078 | 6.638985 | 1.134299 | 0.038327 | 0.033790 |
| Pedestrian | EdgeWeighted | 6.715225 | 8.417418 | 1.214327 | 0.031658 | 0.026071 |
| Pedestrian | TargetWeighted | 6.689244 | 8.428803 | 1.219674 | 0.031512 | 0.025837 |
| Bicycle | EdgeWeighted | 8.526460 | 6.509272 | 0.199716 | 0.011154 | 0.055850 |
| Bicycle | TargetWeighted | 8.466977 | 6.567519 | 0.205163 | 0.010812 | 0.052701 |

Overall EdgeWeighted raw amplification=0.130609，effective retention=0.026539，normalized retention=0.031064，attention shift retention=0.078227。因此全局层面并非“raw变大抵消gate”：raw branch本身更弱，gate再进一步减小实际注入量。两个模型的base states独立训练且不同；比值不是相同base模型的因果反事实。

## Vehicle Motion

| Group | Weighting | TargetCount | GateMean | A_MeanAbsRawBias | F_MeanAbsRawBias | F_MeanAbsEffectiveBias | RawAmplification | EffectiveRetention | CompensationIndex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vehicle.moving | EdgeWeighted | 10446 | 0.308582 | 9.776810 | 1.229700 | 0.376698 | 0.125777 | 0.038530 | 0.124861 |
| vehicle.moving | TargetWeighted | 10446 | 0.308153 | 9.809137 | 1.246104 | 0.381607 | 0.127035 | 0.038903 | 0.126246 |
| vehicle.stopped | EdgeWeighted | 5599 | 0.171336 | 9.675364 | 1.197964 | 0.202605 | 0.123816 | 0.020940 | 0.122218 |
| vehicle.stopped | TargetWeighted | 5599 | 0.166808 | 9.761327 | 1.226702 | 0.202837 | 0.125670 | 0.020780 | 0.124573 |
| vehicle.parked | EdgeWeighted | 25195 | 0.167735 | 10.123887 | 1.302390 | 0.217987 | 0.128645 | 0.021532 | 0.128369 |
| vehicle.parked | TargetWeighted | 25195 | 0.165105 | 10.087715 | 1.297522 | 0.213776 | 0.128624 | 0.021192 | 0.128353 |

Moving vehicle的effective bias大于parked/stopped；差异与可观测运动gate分化方向一致。但其相对Stage4A的effective strength仍明显下降，此处不推断性能改善。

## Pedestrian Motion

| Group | Weighting | TargetCount | GateMean | A_MeanAbsRawBias | F_MeanAbsRawBias | F_MeanAbsEffectiveBias | RawAmplification | EffectiveRetention | CompensationIndex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pedestrian 0-1m | EdgeWeighted | 3192 | 0.183735 | 8.087778 | 1.157667 | 0.212876 | 0.143138 | 0.026321 | 0.143254 |
| Pedestrian 0-1m | TargetWeighted | 3192 | 0.181326 | 8.089779 | 1.159089 | 0.210387 | 0.143278 | 0.026006 | 0.143424 |
| Pedestrian 1-2m | EdgeWeighted | 313 | 0.205314 | 8.183515 | 1.177684 | 0.242045 | 0.143909 | 0.029577 | 0.144058 |
| Pedestrian 1-2m | TargetWeighted | 313 | 0.200742 | 8.177361 | 1.176564 | 0.236362 | 0.143881 | 0.028904 | 0.143988 |
| Pedestrian 2-5m | EdgeWeighted | 916 | 0.219913 | 8.221713 | 1.179391 | 0.259502 | 0.143448 | 0.031563 | 0.143525 |
| Pedestrian 2-5m | TargetWeighted | 916 | 0.217466 | 8.209736 | 1.176227 | 0.255903 | 0.143272 | 0.031171 | 0.143336 |
| Pedestrian 5-10m | EdgeWeighted | 7321 | 0.250087 | 8.178793 | 1.171292 | 0.293146 | 0.143211 | 0.035842 | 0.143319 |
| Pedestrian 5-10m | TargetWeighted | 7321 | 0.247918 | 8.183730 | 1.170389 | 0.290393 | 0.143014 | 0.035484 | 0.143129 |
| Pedestrian 10-20m | EdgeWeighted | 260 | 0.268952 | 8.106476 | 1.172838 | 0.315673 | 0.144679 | 0.038941 | 0.144788 |
| Pedestrian 10-20m | TargetWeighted | 260 | 0.268230 | 8.097498 | 1.173856 | 0.315179 | 0.144965 | 0.038923 | 0.145111 |

| Group | Weighting | TargetCount | GateMean | A_MeanAbsRawBias | F_MeanAbsRawBias | F_MeanAbsEffectiveBias | RawAmplification | EffectiveRetention | CompensationIndex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pedestrian 5-10m | EdgeWeighted | 7321 | 0.250087 | 8.178793 | 1.171292 | 0.293146 | 0.143211 | 0.035842 | 0.143319 |
| Pedestrian 5-10m | TargetWeighted | 7321 | 0.247918 | 8.183730 | 1.170389 | 0.290393 | 0.143014 | 0.035484 | 0.143129 |
| Pedestrian <5m | EdgeWeighted | 4421 | 0.192272 | 8.120412 | 1.163251 | 0.223967 | 0.143250 | 0.027581 | 0.143446 |
| Pedestrian <5m | TargetWeighted | 4421 | 0.190189 | 8.120834 | 1.163877 | 0.221656 | 0.143320 | 0.027295 | 0.143514 |

低运动P的effective bias比5–10m组低；各组pooled F raw本身也远低于A。GT endpoint bins仅来自冻结actor ledger，分组发生在forward观察之后，没有进入gate、relation branch或任何新模型输入。

## Directed Type Pairs

| Pair | EdgeCount | GateMean | A_Raw | F_Raw | F_Effective | RawAmplification | EffectiveRetention | CompensationIndex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V<-V | 1100370 | 0.194725 | 10.698100 | 1.452647 | 0.282929 | 0.135786 | 0.026447 | 0.135815 |
| V<-P | 258536 | 0.212906 | 7.515421 | 0.537870 | 0.114479 | 0.071569 | 0.015233 | 0.071546 |
| V<-B | 15189 | 0.211402 | 0.632338 | 0.680754 | 0.144154 | 1.076567 | 0.227970 | 1.078373 |
| P<-V | 215392 | 0.232730 | 8.555508 | 1.232706 | 0.286936 | 0.144083 | 0.033538 | 0.144108 |
| P<-P | 190592 | 0.222336 | 7.980119 | 1.132746 | 0.251930 | 0.141946 | 0.031570 | 0.141991 |
| P<-B | 6744 | 0.224968 | 0.273715 | 0.108214 | 0.024448 | 0.395351 | 0.089318 | 0.397023 |
| B<-V | 10917 | 0.209788 | 2.242637 | 0.397438 | 0.083337 | 0.177219 | 0.037160 | 0.177132 |
| B<-P | 5216 | 0.204892 | 0.682756 | 0.249443 | 0.051159 | 0.365347 | 0.074930 | 0.365706 |
| B<-B | 624 | 0.185792 | 0.786663 | 0.342000 | 0.064139 | 0.434748 | 0.081533 | 0.438839 |

Pair=target←source，与pair_id=3×target_type+source_type一致。V←V、V←P、P←V、P←P均显示pooled raw及effective衰减。V←B的raw ratio略大于1，但effective retention仍低于其原A幅度，且该pair只占少量full-target edges；不把稀疏pair证据外推为整体补偿。各pair三layer及24head明细均已保存。

## Layer Analysis

| Weighting | Layer | GateMean | RawAmplification | EffectiveRetention | CompensationIndex | NormalizedInteractionRetention | AttentionShiftRetention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EdgeWeighted | 1 | 0.227803 | 0.145391 | 0.033178 | 0.145643 | 0.023340 | 0.065403 |
| EdgeWeighted | 2 | 0.227803 | 0.171181 | 0.038994 | 0.171175 | 0.028488 | 0.120830 |
| EdgeWeighted | 3 | 0.227803 | 0.116067 | 0.026515 | 0.116393 | 0.040666 | 0.143784 |
| TargetWeighted | 1 | 0.227093 | 0.146130 | 0.033239 | 0.146369 | 0.023453 | 0.061212 |
| TargetWeighted | 2 | 0.227093 | 0.169774 | 0.038555 | 0.169775 | 0.027751 | 0.113495 |
| TargetWeighted | 3 | 0.227093 | 0.116296 | 0.026479 | 0.116602 | 0.040060 | 0.138724 |

三个Pedestrian layer的pooled raw/effective强度均减小，但均值掩盖了局部例外。下面同时显示两种加权的Layer2/Head3：

| Group | Weighting | Layer | Head | RawAmplification | EffectiveRetention | CompensationIndex | NormalizedInteractionRetention | AttentionShiftRetention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | EdgeWeighted | 2 | 3 | 1.814367 | 0.368053 | 1.793988 | 0.815544 | 0.974930 |
| Overall | TargetWeighted | 2 | 3 | 1.816641 | 0.372821 | 1.800267 | 0.828009 | 1.015954 |
| Vehicle | EdgeWeighted | 2 | 3 | 1.747769 | 0.343093 | 1.729905 | 0.928813 | 1.029735 |
| Vehicle | TargetWeighted | 2 | 3 | 1.749842 | 0.349664 | 1.735150 | 0.940920 | 1.085950 |
| Pedestrian | EdgeWeighted | 2 | 3 | 2.151976 | 0.490817 | 2.154563 | 0.452167 | 0.870118 |
| Pedestrian | TargetWeighted | 2 | 3 | 2.189361 | 0.497763 | 2.191890 | 0.448273 | 0.868670 |

在Overall/Vehicle/Pedestrian的24个head中，各只有Layer2/Head3的raw amplification大于描述性参考1。该head的Pedestrian raw amplification约2.15，effective retention约0.49，但attention shift retention约0.87；Vehicle attention shift甚至略大于A。这说明“所有head都被同样压小”的叙述不成立。其余head存在不同强度的attention保留，完整24head CSV和图中矩阵保留这些信息，没有选择性省略或平滑。

## Attention Shift

| Group | Weighting | A_AttentionL1Shift | F_AttentionL1Shift | AttentionShiftRetention | A_TopNeighborSwitchRate | F_TopNeighborSwitchRate | DeltaSwitchRate | A_DeltaEntropy | F_DeltaEntropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | EdgeWeighted | 0.763067 | 0.059692 | 0.078227 | 0.438118 | 0.050949 | -0.387169 | -0.453331 | -0.006748 |
| Overall | TargetWeighted | 0.740998 | 0.057404 | 0.077469 | 0.423076 | 0.048406 | -0.374670 | -0.416033 | -0.006397 |
| Vehicle | EdgeWeighted | 0.752829 | 0.050837 | 0.067528 | 0.431616 | 0.043880 | -0.387735 | -0.485278 | -0.005925 |
| Vehicle | TargetWeighted | 0.730449 | 0.050212 | 0.068742 | 0.415924 | 0.042821 | -0.373103 | -0.442795 | -0.005871 |
| Pedestrian | EdgeWeighted | 0.815237 | 0.091143 | 0.111800 | 0.466124 | 0.076080 | -0.390044 | -0.361250 | -0.009738 |
| Pedestrian | TargetWeighted | 0.801373 | 0.085334 | 0.106485 | 0.456827 | 0.070197 | -0.386630 | -0.338718 | -0.008580 |
| vehicle.moving | EdgeWeighted | 0.588878 | 0.083948 | 0.142555 | 0.338961 | 0.072257 | -0.266704 | -0.351817 | -0.012236 |
| vehicle.moving | TargetWeighted | 0.549812 | 0.078419 | 0.142629 | 0.314315 | 0.066162 | -0.248153 | -0.308385 | -0.011271 |
| vehicle.stopped | EdgeWeighted | 0.802837 | 0.051663 | 0.064351 | 0.459646 | 0.043209 | -0.416437 | -0.419727 | -0.005732 |
| vehicle.stopped | TargetWeighted | 0.785304 | 0.046875 | 0.059690 | 0.446575 | 0.039003 | -0.407573 | -0.401469 | -0.004943 |
| vehicle.parked | EdgeWeighted | 0.796607 | 0.039634 | 0.049754 | 0.456496 | 0.034631 | -0.421864 | -0.542630 | -0.003931 |
| vehicle.parked | TargetWeighted | 0.794826 | 0.039135 | 0.049237 | 0.452370 | 0.034008 | -0.418362 | -0.510616 | -0.003875 |
| Pedestrian 0-1m | EdgeWeighted | 0.793016 | 0.081287 | 0.102503 | 0.447194 | 0.070562 | -0.376632 | -0.348876 | -0.005744 |
| Pedestrian 0-1m | TargetWeighted | 0.776486 | 0.075735 | 0.097535 | 0.438675 | 0.064354 | -0.374321 | -0.322984 | -0.004333 |
| Pedestrian 1-2m | EdgeWeighted | 0.791235 | 0.085448 | 0.107993 | 0.456742 | 0.078045 | -0.378696 | -0.356782 | -0.007353 |
| Pedestrian 1-2m | TargetWeighted | 0.757573 | 0.076814 | 0.101394 | 0.434904 | 0.067359 | -0.367545 | -0.329976 | -0.006670 |
| Pedestrian 2-5m | EdgeWeighted | 0.809110 | 0.089912 | 0.111125 | 0.475394 | 0.073716 | -0.401678 | -0.354523 | -0.008175 |
| Pedestrian 2-5m | TargetWeighted | 0.799561 | 0.085014 | 0.106326 | 0.466749 | 0.068550 | -0.398199 | -0.343027 | -0.007622 |
| Pedestrian 5-10m | EdgeWeighted | 0.826878 | 0.096257 | 0.116411 | 0.474146 | 0.078929 | -0.395216 | -0.367244 | -0.012047 |
| Pedestrian 5-10m | TargetWeighted | 0.813233 | 0.089871 | 0.110511 | 0.463808 | 0.073123 | -0.390685 | -0.345226 | -0.010652 |

alpha_final直接取自正式forward的dropout入口；alpha_base用该layer实际base logits和同一完整incoming neighborhood，调用原PyG softmax。L1是sum_j|alpha_final−alpha_base|；neighbor argmax用原edge顺序的first-edge tie break；entropy为自然对数，0log0=0。此参考仅是layer内固定base的注意力观察，没有执行base-only模型rollout、counterfactual性能评价或变体选择。

Attention entropy下降只描述注意力集中度变化，不能自动解释为更好。绝对logit magnitude还包含softmax不敏感的共同偏移，因此以base归一化与实际attention结果一起判断。

## Interaction Context

| Group | Weighting | TargetCount | GateMean | A_MeanAbsRawBias | F_MeanAbsRawBias | F_MeanAbsEffectiveBias | RawAmplification | EffectiveRetention | CompensationIndex |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Heterogeneous-20m | EdgeWeighted | 25113 | 0.219424 | 8.853156 | 1.152080 | 0.251382 | 0.130132 | 0.028395 | 0.129406 |
| Heterogeneous-20m | TargetWeighted | 25113 | 0.218834 | 8.849112 | 1.153510 | 0.251403 | 0.130353 | 0.028410 | 0.129824 |
| non-heterogeneous | EdgeWeighted | 29859 | 0.192001 | 10.080230 | 1.320458 | 0.252370 | 0.130995 | 0.025036 | 0.130396 |
| non-heterogeneous | TargetWeighted | 29859 | 0.197216 | 9.998168 | 1.311424 | 0.257476 | 0.131166 | 0.025752 | 0.130579 |
| Vehicle hetero-20m | EdgeWeighted | 14813 | 0.213432 | 9.564949 | 1.164323 | 0.245527 | 0.121728 | 0.025669 | 0.120270 |
| Vehicle hetero-20m | TargetWeighted | 14813 | 0.213401 | 9.567235 | 1.173112 | 0.247986 | 0.122618 | 0.025920 | 0.121463 |
| Pedestrian hetero-20m | EdgeWeighted | 9732 | 0.228311 | 8.162063 | 1.169790 | 0.267381 | 0.143320 | 0.032759 | 0.143484 |
| Pedestrian hetero-20m | TargetWeighted | 9732 | 0.227921 | 8.170453 | 1.170365 | 0.267041 | 0.143244 | 0.032684 | 0.143399 |
| VP-context-20m | EdgeWeighted | 23209 | 0.220375 | 8.962132 | 1.162495 | 0.254652 | 0.129712 | 0.028414 | 0.128936 |
| VP-context-20m | TargetWeighted | 23209 | 0.219993 | 8.998260 | 1.167703 | 0.255661 | 0.129770 | 0.028412 | 0.129151 |

20m membership完全复用冻结ledger；non-heterogeneous为其full-target补集。完整incoming softmax未截断到20m，也未重新定义组。

## Gate–Raw Bias Correlation

| Group | Measurement | TargetCount | Spearman_rho |
| --- | --- | --- | --- |
| Overall | F_raw | 54972 | -0.298430 |
| Overall | F_effective | 54972 | 0.747801 |
| Vehicle | F_raw | 42314 | -0.213439 |
| Vehicle | F_effective | 42314 | 0.664451 |
| Pedestrian | F_raw | 12002 | 0.167611 |
| Pedestrian | F_effective | 12002 | 0.967139 |
| Pedestrian <5m | F_raw | 4421 | 0.215747 |
| Pedestrian <5m | F_effective | 4421 | 0.922800 |
| Pedestrian 5-10m | F_raw | 7321 | 0.231949 |
| Pedestrian 5-10m | F_effective | 7321 | 0.922400 |

在每target先平均edge/layer/head后做Spearman。Overall/Vehicle gate与raw平均幅度呈负相关，但Pedestrian以及P<5m、P5–10m为正相关；相关方向并不一致，且不能否定F pooled raw远小于A的直接观察。这些窗口有重叠，跨类型/邻域组成也可能影响相关，不作为因果或显著性证明；未事后规定统计阈值。

## Scientific Decision

**GateCompensation = MIXED**。不存在一致的全局amplitude compensation证据：pooled raw、effective、base-normalized强度和attention shifts在两种加权下均大幅降低。但Layer2/Head3确有局部raw放大和较高attention保留，少量pair也出现不同方向；所以不将本轮简化成“完全没有任何补偿”。MIXED描述全局衰减与局部补偿共存，而不是证明局部head导致pedestrian性能退化，更不是改写Stage4F的NOT SUPPORTED结论。

## Recommended Next Step

等待大脑AI根据全局衰减与 Layer2/Head3 局部模式判断；不要立即训练 bounded variant。
本轮完成后STOP。未实现bounded modulation、tanh/clip、gate scale/lambda变体、训练、fine-tune、额外seed或Reliability Head。

## Artifacts

四张图的底层artist数值逐项与完整精度CSV比对，absolute difference=0，满足<1e-8。图中文字注释保留3位小数（head矩阵2位）、报告表保留6位以便阅读；这些格式化注释不作为完整精度数值，完整数值和逐项比较在CSV/figure audit JSON中保留。

本地target_strength.csv共91092行，SHA256 `763bb984b34940fe023a7da8b71bca8cf8e3d312f982299420a0400fa3bf5ba8`；schema记录于statistics_audit.json。raw_target_head.bin与identity CSV保留本地且记录SHA，支持重算全部group/layer/head；Git仅上传脚本、requirements、审计JSON、较小汇总CSV、四图PNG/PDF/SVG及报告，原Stage4F结果完全冻结。
