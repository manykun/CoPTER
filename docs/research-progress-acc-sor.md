# CoPTER：ACC 有效性、灾难性遗忘与 SOR 研究进展

> 更新日期：2026-07-24
> 实验平台：256 主机三层 Fat-Tree，10 Gbps 主机链路，40 Gbps
> Agg-Core 链路，32 机柜、2 Pod，每 Pod 16 个 ToR-Agg 组
> 当前结论适用范围：单流量种子下的机制证据，不构成跨随机种子的统计结论

## 1. 研究目标

本阶段围绕三个问题展开：

1. 静态 ECN 参数在 throughput、mixed、incast 三种场景中的表现如何；
2. ACC 能否相对论文静态参数改善 FCT，同时保持流完成率；
3. 当流量任务从 A 切换到 B 时，普通 ACC 是否发生灾难性遗忘，SOR
   是否能改善稳定性与可塑性之间的权衡。

场景含义：

- `throughput`：WebSearch/长流主导，主要观察吞吐和长流完成；
- `mixed`：CacheFollower/长短流混合；
- `incast`：突发多对一流量，重点观察队列、尾时延和 PFC。

## 2. 已完成的实验基础设施

已完成以下工程工作：

- 在 `/mnt/sdb1/xuduokun` 建立项目、conda 环境和实验输出目录，避免系统盘空间不足；
- 构建 ns-3.33、ns3-gym 和 CoPTER 仿真程序；
- 修复 ns3-gym 在 socket 正常关闭时返回 `observation=None, done=False`
  导致 agent 误报失败的问题；
- 建立固定 flow、manifest、冻结贪心评估、checkpoint、resume 和自动分析流程；
- 静态参数、ACC、SOR 使用相同 flow 文件和 common-flow 指标；
- 将训练预算由“相同 episode 数”改为“相同 optimizer update 数”；
- 在任务切换时仅重置 phase-local epsilon，不重置网络和 replay；
- 增加 acquisition、completion safety、forgetting 和 SOR 对比门槛。

## 3. 早期固定动作敏感性实验

三组固定动作：

- aggressive：`(kmin,kmax,pmax)=(0,0,9)`；
- balanced：`(2,1,4)`；
- permissive：`(5,3,0)`。

| 场景 | 动作 | 完成率 | p95 FCT (µs) | p99 slowdown | 旧奖励 |
|---|---|---:|---:|---:|---:|
| throughput | aggressive | 0.9864 | 662.46 | 6.116 | 0.7129 |
| throughput | balanced | 0.9868 | 645.13 | 6.486 | 0.7074 |
| throughput | permissive | 0.9864 | 564.25 | 6.156 | 0.6938 |
| mixed | aggressive | 0.9883 | 1520.13 | 22.824 | 0.7361 |
| mixed | balanced | 0.9886 | 1510.82 | 22.469 | 0.7348 |
| mixed | permissive | 0.9886 | 1421.33 | 22.260 | 0.7309 |
| incast | aggressive | 0.9002 | 1117.30 | 46.348 | 0.4959 |
| incast | balanced | 0.9231 | 1095.91 | 46.442 | 0.4975 |
| incast | permissive | 0.9079 | 1118.18 | 47.277 | 0.5012 |

主要发现：

- 参数变化确实影响 FCT，因此 ACC 动作不是“对仿真无效”；
- throughput 和 mixed 的 p95 更偏向 permissive；
- incast 的完成率更偏向 balanced；
- 旧奖励对动作的排序与 p95/完成率排序不一致；
- 首轮 sensitivity gate 失败的核心不是缺少动作效应，而是奖励信号弱且方向失配。

## 4. 论文静态参数基线

静态配置：

- SECN_1：`[5 KB, 200 KB, 1%] @ 10 Gbps`；
- SECN_2：`[100 KB, 400 KB, 20%] @ 25 Gbps`；
- switch buffer：400 KB；
- simulator stop time：4.00 s。

| 场景 | 静态方法 | 完成率 | p95 FCT (µs) | p99 slowdown |
|---|---|---:|---:|---:|
| throughput | SECN_1 | 0.9986 | 664.03 | 5.034 |
| throughput | SECN_2 | 0.9983 | 671.14 | 5.174 |
| mixed | SECN_1 | 0.9931 | 2158.81 | 31.430 |
| mixed | SECN_2 | 0.9890 | 2245.11 | 32.899 |
| incast | SECN_1 | 0.8779 | 3067.22 | 66.893 |
| incast | SECN_2 | 0.8201 | 3381.91 | 75.951 |

基线 gate 的 `PASS` 表示两套论文配置在三个场景中都完成测量，不表示
SECN_1/SECN_2 的网络性能达到某个绝对标准。

针对 incast 完成率偏低已完成排查：

- 将停止时间从 2.25 s 延长到 2.50 s，结果基本不变；
- 最后流完成时间约为 2.10 s，2.25 s 后无新完成流；
- PFC pause/resume 数量相等，仿真结束时无端口残留 pause；
- 四组 incast/mixed 静态实验均为 0 drops。

因此未完成流不能简单归因于仿真过早结束、残留 PFC pause 或丢包。

## 5. ACC 单任务有效性实验

运行：`experiments/acc_validation/acc_study_v2`，seed 1。

| 场景 | ACC 完成率 | ACC p95 FCT (µs) | ACC p99 slowdown | ACC reward |
|---|---:|---:|---:|---:|
| throughput | 0.9864 | 598.32 | 7.011 | 0.7000 |
| mixed | 0.9880 | 1454.79 | 22.534 | 0.7335 |
| incast | 0.9068 | 1285.57 | 54.424 | 0.5005 |

以每个场景 p95 更好的静态配置为参照：

| 场景 | 静态最佳 p95 | ACC p95 | p95 改善 | 静态完成率 | ACC 完成率 | 完成率变化 |
|---|---:|---:|---:|---:|---:|---:|
| throughput | 664.03 | 598.32 | 9.90% | 0.9986 | 0.9864 | -1.22 pp |
| mixed | 2158.81 | 1454.79 | 32.61% | 0.9931 | 0.9880 | -0.51 pp |
| incast | 3067.22 | 1285.57 | 58.09% | 0.8779 | 0.9068 | +2.89 pp |

结论：

- mixed 和 incast 通过 effectiveness gate；
- throughput 虽然 p95 改善 9.90%，但完成率下降 1.22 个百分点，超过
  1% 容忍范围，因此失败；
- 总体 effectiveness gate 通过，证明 ACC 在至少两个场景中具备实际收益；
- 后续持续学习实验优先选 mixed 与 incast。

## 6. 首次 ACC 持续学习实验：自然流量切换

运行：`sor_mixed_incast_s1`，课程 `mixed → incast`，每任务 30
episodes，旧奖励 `0.50,0.30,0.20`。

| 比较 | reward before | reward after | reward 变化 | p95 before | p95 after | p95 变化 | completion 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed 习得 | 0.7328 | 0.7347 | +0.26% | 1482.69 | 1502.98 | -1.37% | -0.01 pp |
| incast 习得 | 0.4942 | 0.4947 | +0.10% | 1214.37 | 1182.82 | +2.60% | -5.32 pp |
| mixed 遗忘 | 0.7347 | 0.7355 | +0.11% | 1503.00 | 1455.12 | +3.19% | +0.05 pp |

这里“p95 变化”为正表示改善、负表示恶化。

训练诊断：

| 阶段 | episodes | global updates | env steps | epsilon 起止 | rollout reward 起止 |
|---|---:|---:|---:|---:|---:|
| train A / mixed | 30 | 19 → 570 | 154 → 4620 | 0.941 → 0.050 | 0.7356 → 0.7363 |
| train B / incast | 30 | 582 → 930 | 4714 → 7440 | 0.050 → 0.050 | 0.4997 → 0.4937 |

主要结论：

- 网络确实发生了更新，不能解释为“代码没有训练”；
- 相同 30 episodes 给 mixed 约 570 updates、给 incast 约 360
  updates，计算预算不公平；
- B 阶段 epsilon 从一开始就是 0.05，缺少新任务探索；
- ACC 没有安全习得 incast，完成率下降 5.32 个百分点；
- mixed 在训练 B 后反而改善，因此没有灾难性遗忘；
- 该实验是重要的负结果：在旧协议下不能证明遗忘，也不能继续证明 SOR
  的抗遗忘能力。

据此完成的协议修正：

- 每任务固定 600 optimizer updates；
- A、B 各自使用独立 epsilon 衰减时钟；
- A/B acquisition 未通过即停止；
- completion 下降超过 1 个百分点即判定不安全；
- ACC 未证明遗忘时禁止启动 SOR。

## 7. 任务级线性奖励冲突筛选

校准运行：`controlled_mixed_incast_s1`。

旧设计给 mixed 使用 `0.25,0.55,0.20`，给 incast 使用
`0.70,0.15,0.15`。

| 场景 | 动作 | 完成率 | p95 FCT (µs) | reward |
|---|---|---:|---:|---:|
| mixed | aggressive | 0.9883 | 1519.78 | 0.7772 |
| mixed | balanced | 0.9886 | 1512.35 | 0.7740 |
| mixed | permissive | 0.9886 | 1430.52 | 0.7670 |
| incast | aggressive | 0.9002 | 1222.25 | 0.4702 |
| incast | balanced | 0.9231 | 1355.45 | 0.4726 |
| incast | permissive | 0.9079 | 1356.68 | 0.4887 |

筛选结果：

| 场景 | reward 最佳 | reward spread | p95 spread | 完成率安全 | gate |
|---|---|---:|---:|---:|---:|
| mixed | aggressive | 1.31% | 5.87% | 是 | FAIL |
| incast | permissive | 3.78% | 9.91% | 否 | FAIL |

该结果证明：

- 场景有真实动作敏感性，p95 spread 分别为 5.87% 和 9.91%；
- mixed 的 reward 最优动作却具有最差 p95；
- incast 的 reward 最优动作相对最高完成率低 1.52 个百分点；
- 当前 reward 会引导智能体朝错误或不安全的方向学习。

## 8. 奖励分量诊断

| 场景/动作 | throughput | queue reward | ECN reward | avg queue | peak queue |
|---|---:|---:|---:|---:|---:|
| mixed/aggressive | 0.4177 | 0.8715 | 0.6145 | 0.0095 | 0.0330 |
| mixed/balanced | 0.4187 | 0.8666 | 0.6080 | 0.0101 | 0.0346 |
| mixed/permissive | 0.4283 | 0.8494 | 0.5993 | 0.0128 | 0.0405 |
| incast/aggressive | 0.1708 | 0.7837 | 0.5529 | 0.0613 | 0.0966 |
| incast/balanced | 0.1730 | 0.7822 | 0.5485 | 0.0582 | 0.0930 |
| incast/permissive | 0.1811 | 0.7708 | 0.5451 | 0.0658 | 0.1038 |

两个场景中的三个线性分量具有相同排序：

- throughput：permissive > balanced > aggressive；
- queue/ECN reward：aggressive > balanced > permissive。

对非负且和为 1 的线性权重进行网格搜索后，不存在一组公共线性权重能让
mixed 偏向 permissive、同时让 incast 偏向 balanced/aggressive。
incast 虽存在约 `0.555 throughput + 0.445 queue` 使 balanced 略优的
狭窄区域，但领先幅度仅约 0.14%，不足以形成可靠学习信号。

因此失败原因不是某一组权重“没调好”，而是线性奖励表达能力不足。

## 9. 本次奖励函数修改

新增公共奖励模式 `tail_safe`：

```text
q = 0.3 × avg_queue + 0.7 × peak_queue
e = 0.3 × avg_ecn   + 0.7 × peak_ecn

R_raw = throughput - 5 × q² - 5 × e²
R = clip(R_raw, -1, 1)
```

设计原则：

- 低负载时二次惩罚较弱，允许 mixed 保持吞吐；
- 突发队列/ECN 增大时惩罚快速增强，促使 incast 选择安全动作；
- A、B 使用完全相同的奖励，任务边界只改变流量分布；
- 保留旧 `weighted` 模式，确保历史实验仍可复现。

基于已有聚合数据的离线估计：

| 场景 | aggressive | balanced | permissive | 预计最优 |
|---|---:|---:|---:|---|
| mixed | 0.4103 | 0.4126 | 0.4220 | permissive |
| incast | 0.0775 | 0.0875 | 0.0846 | balanced |

这是近似估计，因为 `E[q²]` 不等于 `E[q]²`；必须通过新的固定动作 screen
验证逐步计算后的真实 reward。该验证不是盲目重复排查，而是新奖励进入正式
训练前的注册验收。

同时完成两项分析修正：

- 固定动作 p95 改为三种动作共同完成流的 common-flow p95；
- 注册主奖励改为 all-congested-port mean；top-30% reward 仅作诊断，
  因为它在 incast 中丢弃持续拥塞端口后会反转动作排序；
- reward spread 门槛注册为 2%，系统性能敏感性要求 common-flow p95
  spread ≥ 5% 或 completion spread ≥ 2 个百分点。

新 screen 的逐步平方成本进一步表明，`λq=5, λe=5` 下：

| 场景/动作 | top-30% reward | all-congested reward |
|---|---:|---:|
| mixed/aggressive | 0.7040 | 0.3759 |
| mixed/balanced | 0.7152 | 0.3794 |
| mixed/permissive | 0.7268 | 0.3870 |
| incast/aggressive | 0.0961 | -0.0926 |
| incast/balanced | 0.0958 | -0.0844 |
| incast/permissive | 0.1006 | -0.0941 |

mixed 在两种聚合方式下均选择 permissive；incast 的 top-30% 指标错误地
选择 permissive，而 all-congested 指标选择完成率最高的 balanced。由于
ACC/SOR replay 覆盖所有端口，all-congested 更接近实际优化总体。

离线搜索得到约 `λq=30, λe=18` 的更大动作间隔，但 incast 原始奖励约为
`-2.4~-2.7`，会被当前 `clip(-1,1)` 全部截成 `-1`，反而完全消除学习信号。
因此不采用该组大系数，保留未饱和且方向正确的 `5,5`。

## 10. 当前研究结论

已经能够支持的结论：

1. 静态论文参数已在三个场景中完整复现；
2. ACC 在 mixed 和 incast 上相对静态配置具有明显 p95 收益；
3. throughput 的收益伴随超过容忍度的完成率损失；
4. ECN 参数对三种流量场景的 FCT/完成率确实有影响；
5. 首次持续学习实验未观察到灾难性遗忘，主要受到奖励失配、探索不公平和
   update 预算不公平影响；
6. 三分量线性奖励无法表达期望的跨场景安全控制冲突；
7. 已实现公共非线性 tail-safe 奖励和更严格的持续学习实验协议。

目前尚不能支持的结论：

- 尚不能宣称 ACC 在自然流量切换下存在灾难性遗忘；
- 尚不能宣称 SOR 已经有效缓解灾难性遗忘；
- 单 seed 不能支持统计泛化结论；
- 新奖励的离线估计不能代替固定动作和训练验证。

## 11. 后续目标与决策门槛

### 目标 1：验证新奖励

在 mixed/incast 上各运行 aggressive、balanced、permissive：

- reward spread ≥ 2%；
- common-flow p95 spread ≥ 5%，或 completion spread ≥ 2 个百分点；
- reward 最优动作完成率距离最高完成率不超过 1 个百分点；
- reward 最优动作必须等于完成率安全集合中的 common-flow p95 最优动作；
- mixed 与 incast 的 reward 最优动作不同。

若失败：停止训练，只调整奖励结构，不放宽完成率门槛。

### 目标 2：证明 ACC 习得两个任务

- 每任务 600 optimizer updates；
- 每个任务 epsilon 从 1.0 独立衰减至 0.05；
- reward 改善 ≥ 2% 或 common-flow p95 改善 ≥ 5%；
- completion 下降 ≤ 1 个百分点。

### 目标 3：检测 ACC 灾难性遗忘

比较 `A after A` 与 `A after B`：

- old-task reward 下降 ≥ 10%；
- old-task common-flow p95 同时恶化 ≥ 10%。

两个条件必须同时成立。

### 目标 4：验证 SOR

仅在 ACC 已习得 A/B 且发生注册遗忘后启动 SOR：

- SOR 也必须习得 A/B；
- 遗忘分数相对 ACC 至少下降 30%；
- task-B p95 不得比 ACC 差 5% 以上；
- task-B completion 不得比 ACC 低 1 个百分点以上。

## 12. 汇报建议

建议将研究进展表述为：

> 已完成 CoPTER/ACC 的环境复现、论文静态基线和三场景 ACC 有效性验证。
> ACC 在 mixed、incast 上显示出明显尾时延收益，但首次 mixed→incast
> 持续学习实验没有产生灾难性遗忘。进一步固定动作实验发现，根因并非控制
> 参数无效，而是原线性奖励与端到端 FCT/完成率方向不一致，并且相同 episode
> 不能保证相同训练更新预算。当前已将协议修正为等 optimizer updates、任务级
> epsilon 重置、common-flow 冻结评估和分阶段安全门槛，并设计了一个 A/B 共用
> 的二次队列/ECN tail-safe 奖励。下一步先通过固定动作 gate 验证新奖励，再
> 依次验证 ACC 习得、灾难性遗忘和 SOR 的抗遗忘效果。
