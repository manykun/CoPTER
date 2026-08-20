# CoPTER：ACC 有效性、灾难性遗忘与 SOR 研究进展

> 更新日期：2026-08-16
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

## 10. 修正协议下的 ACC 持续学习结果

后续正式持续学习实验使用课程 `mixed → incast`、单一公共 `tail_safe`
奖励、固定 flow、冻结贪心评估和每任务 600 次 optimizer update。这里的
正式 reward 是 `rollout_all_congested_mean`，即所有拥塞端口—时间步样本的
平均奖励；`rollout_mean_reward`（top-30% 拥塞端口）仅作为诊断指标。

### 10.1 Global replay 开启

运行目录：`experiments/continual_validation/tailsafe_mixed_incast_s1`。

| 比较 | reward before | reward after | reward 变化 | p95 before (µs) | p95 after (µs) | p95 变化 | completion 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed 习得 | 0.377825 | 0.379439 | +0.4271% | 1489.99 | 1471.50 | 改善 1.2406% | +0.0099 pp |
| incast 习得 | -0.093350 | -0.093529 | -0.1913% | 1385.53 | 1295.24 | 改善 6.5168% | +0.4756 pp |
| 返回 mixed 后的遗忘 | 0.379439 | 0.378924 | -0.1355% | 1466.04 | 1471.17 | 恶化 0.3503% | -0.0099 pp |

结果表明，ACC 在 B 任务 incast 上改善了 p95 和完成率；继续训练 incast 后
重新切回 mixed，旧任务只出现轻微退化，没有观察到严重的全局灾难性遗忘。

### 10.2 关闭 Global replay 的单因素消融

运行目录：
`experiments/continual_validation/tailsafe_mixed_incast_localonly_s1`。
该实验设置 `shared_replay=false`，保留每端口 local FIFO replay，仅关闭跨端口
global replay；flow、seed、奖励、网络宽度和每任务 update 预算均与 Global-on
实验一致。

| 比较 | reward before | reward after | reward 变化 | p95 before (µs) | p95 after (µs) | p95 变化 | completion 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed 习得 | 0.377825 | 0.381450 | +0.9596% | 1489.67 | 1471.01 | 改善 1.2522% | +0.0198 pp |
| incast 习得 | -0.092023 | -0.083161 | +9.6299% | 1294.90 | 1198.70 | 改善 7.4295% | -0.0310 pp |
| 返回 mixed 后的遗忘 | 0.381450 | 0.381051 | -0.1049% | 1454.13 | 1486.75 | 恶化 2.2429% | -0.0298 pp |

Global-on 与 Local-only 的旧任务退化对比：

| mixed 旧任务指标 | Global-on | Local-only | Local-only 相对变化 |
|---|---:|---:|---:|
| reward 下降 | 0.1355% | 0.1049% | 少下降 0.0307 pp |
| p95 恶化 | 0.3503% | 2.2429% | 多恶化 1.8927 pp |
| completion 下降 | 0.0099 pp | 0.0298 pp | 多下降 0.0198 pp |
| p95 遗忘倍数 | 1× | 约 6.4× | 明显放大 |
| completion 回退倍数 | 1× | 约 3× | 明显放大 |

关闭 global replay 后，incast 适应能力更强，但 mixed 的尾部 FCT 和完成率
保留变差。这反映了可塑性—稳定性权衡：global replay 对平均 reward 的影响
不明显，却显著限制了旧任务尾部性能回退。

## 11. 神经网络参数漂移分析

ACC 为每个受控端口维护独立的 `TripleHeadACC`，本实验匹配到 448 组端口
网络。参数漂移比较同一端口 `after_a` 与 `after_b` 的 policy network，主要
指标为相对 L2：

```text
||theta_after_b - theta_after_a||2 / ||theta_after_a||2
```

### 11.1 分层漂移

| 网络区域 | Global-on relative L2 | Local-only relative L2 | Local-only 相对变化 |
|---|---:|---:|---:|
| 全网络 | 0.3363 | 0.3398 | +1.06% |
| shared trunk | 0.3424 | 0.3457 | +0.96% |
| `k_min` head | 0.2850 | 0.2843 | -0.25% |
| `k_max` head | 0.3296 | 0.3007 | -8.76% |
| `p_max` head | 0.2427 | 0.2648 | +9.09% |

两组实验的全网络余弦相似度分别为 0.9439 和 0.9428，整体方向接近。
Local-only 的全网络平均绝对参数变化反而更低（0.01625 vs 0.01814），说明
其差异不是全体参数均匀增大，而更可能集中在部分网络和参数方向。

### 11.2 端口漂移长尾

| 端口漂移指标 | Global-on | Local-only | Local-only 相对变化 |
|---|---:|---:|---:|
| 中位数 | 0.3402 | 0.3243 | -4.68% |
| p95 | 0.4223 | 0.4808 | +13.86% |
| 最大值 | 0.5409 | 0.9723 | +79.75% |

Local-only 最大漂移端口为 371（0.9723），随后为 191、78、404、380、88、
316、319、149 和 379。关闭 global replay 后，普通端口的中位漂移没有增加，
但最高漂移的少数端口形成明显长尾；同时 `p_max` head 漂移增加 9.09%。该现象
与 mixed p95 恶化从 0.35% 放大到 2.24% 在方向上一致。

权重均值、标准差和分位数等边缘分布在两组实验间仍较相似，因此仅观察参数
直方图不足以识别遗忘。当前结果属于“参数长尾漂移与尾部性能恶化”的相关性
证据，尚未通过参数恢复/注入实验建立因果关系。

## 12. 高活跃端口的局部 Reward 观测

端口使用 `after_a/mixed` 的发送速率和拥塞程度预先选择，避免依据
`after_b` 结果事后挑选。选择的五个端口为 185、323、229、316 和 166。
两次冻结评估均使用原实验 `tail_safe` 配置：`lambda_q=5.0`、
`lambda_e=5.0`。

| port | after A / mixed | after B / mixed | 相对变化 | 解释 |
|---:|---:|---:|---:|---|
| 185 | 0.964744 | 0.974122 | +0.97% | 小幅改善 |
| 323 | 0.850225 | 0.830264 | -2.35% | 可观测的局部退化 |
| 229 | -0.999832 | -0.999778 | +0.01% | reward 下限饱和，无法判断 |
| 316 | -0.999724 | -0.999774 | -0.01% | reward 下限饱和，无法判断 |
| 166 | -0.994848 | -0.996033 | -0.12% | 接近下限饱和，区分能力很弱 |

同两次评估的全体拥塞端口 reward 为 0.381450 和 0.381051，下降 0.1049%，
与主持续学习报告完全一致。五个端口中，323 出现 2.35% 的有效局部退化，185
改善 0.97%；其余三个端口因 `clip(-1, 1)` 进入奖励下限，不能把细小数值变化
解释为遗忘。

当前 `watch_ports_reward` 是拥塞期间的 EMA，而不是严格算术平均。因此该结果
只能支持“存在局部端口退化案例”，不能证明多数高活跃端口发生 reward 遗忘。
正式端口机制分析还需记录每端口未裁剪 `raw_reward`、算术平均、样本数、队列、
ECN、吞吐和动作直方图。

曾使用候选 `lambda_q=29.75`、`lambda_e=18.25` 进行过一次端口测评；该参数
与原训练实验不一致，并造成 reward 大量饱和，相关结果已排除，不纳入结论。

## 13. SOR 实验状态

早期 `smoke_mixed_incast` 仅用于检查流程，其 `FAIL` 不能作为科研证据。正式
SOR 后台任务已运行结束：

```text
bash scripts/continual_validation/run_continual.sh \
  --stage sor ... --report-only --resume
```

当前尚未取得正式 SOR 分析表，包括 Task A/B 习得、返回 Task A 后的遗忘、
与 Global-on/Local-only ACC 的 p95 和 completion 对照。因此目前只能确认 SOR
进程完成，不能宣称 SOR 已经缓解遗忘。

## 14. 当前研究结论

### 14.1 已有证据支持

1. 两套论文静态参数已在 throughput、mixed、incast 三个场景中完整复现；
2. ECN 动作在三个场景中具有真实的 FCT/完成率敏感性；
3. ACC 相对最佳静态参数将 mixed 和 incast 的 p95 分别改善 32.61% 和
   58.09%；
4. throughput 的 p95 虽改善 9.90%，但完成率下降 1.22 pp，超过容忍范围；
5. 原线性奖励与端到端性能排序失配，公共二次 `tail_safe` 奖励能表达 mixed
   偏向 permissive、incast 偏向 balanced 的方向；
6. Global-on ACC 在 `mixed → incast` 后只出现轻微旧任务退化；
7. 关闭 global replay 后，mixed p95 退化从 0.35% 放大到 2.24%，completion
   回退约放大三倍；
8. Local-only 的全网平均参数漂移仅小幅增加，但端口漂移 p95 增加 13.86%、
   最大值增加 79.75%，说明参数覆盖集中在少数端口；
9. 当前证据更符合少数关键端口和尾部流上的局部遗忘，而不是全网络平均
   reward 的严重下降。

### 14.2 尚不能支持

- 不能宣称已经观察到严重且广泛的灾难性遗忘；
- 不能宣称多数高活跃端口的 reward 都发生下降；
- 参数长尾漂移与 mixed p95 恶化尚未通过参数恢复/注入证明因果关系；
- 尚不能宣称 SOR 有效缓解遗忘；
- 单 seed 结果不能支持跨随机流量的统计泛化。

## 15. 下一步工作

1. 提取正式 SOR 报告，与 Global-on 和 Local-only ACC 使用同一表格对照；
2. 为观察端口记录未裁剪 raw reward、算术平均、拥塞样本数、队列、ECN、
   吞吐和动作直方图，解决 reward 下限饱和与 EMA 偏置；
3. 使用已有检查点构造参数干预：在 Local-only `after_b` 中恢复漂移最高端口的
   `after_a` 参数，并以随机端口恢复作为对照；
4. 如果恢复高漂移端口能显著恢复 mixed p95，而随机恢复无效，则建立
   “关键端口参数漂移 → 策略变化 → 尾部性能遗忘”的因果证据；
5. 若计算预算允许，再增加流量 seed，用于统计泛化；当前单 seed 继续定位为
   机制证据。

## 16. 汇报建议

建议将当前进展表述为：

> 已完成 CoPTER/ACC 环境复现、论文静态基线和三场景 ACC 有效性验证。
> ACC 在 mixed、incast 上相对静态配置获得显著尾时延收益。修正持续学习协议
> 后，开启 global replay 的 ACC 在 mixed→incast 切换中仅产生轻微旧任务退化；
> 关闭 global replay 后，新任务适应能力增强，但 mixed p95 退化由 0.35% 放大
> 至 2.24%，少数端口网络的参数漂移长尾也明显增强。全网平均 reward 仅下降
> 约 0.1%，端口级结果尚未显示广泛 reward 退化，因此当前证据支持“少数关键
> 端口上的局部尾部遗忘”，而不是严重的全网灾难性遗忘。正式 SOR 运行已经
> 结束，下一步需提取结果并通过参数恢复干预验证遗忘机制和 SOR 效果。

## 17. 动作离散度与物理范围实验更新

端口 323 的冻结单端口插值实验表明，离散网格之间存在当前动作集合没有表达的
更优区域。中心点为 `(Kmin=63 KB, Kmax=120 KB, Pmax=0.50)`：

| 候选点 | mixed p95 (us) | 相对中心 | 端口 reward | incast p95 (us) | 相对中心 |
|---|---:|---:|---:|---:|---:|
| 中心 | 1486.06 | 0.00% | 0.6070 | 1262.74 | 0.00% |
| Kmin=57 KB | 1447.81 | -2.57% | 0.6414 | 1262.74 | 0.00% |
| Kmax=110 KB | 1445.59 | -2.72% | 0.6326 | 1262.74 | 0.00% |
| Pmax=0.55 | 1468.94 | -1.15% | 0.6295 | 1262.74 | 0.00% |

这说明细化或连续化具有实验依据，但单个端口的结果不足以直接修改全局动作
空间。新加入的物理范围实验采用冻结 `after_b` 模型，在 ns-3 中只覆盖一个
端口的物理 ECN 参数，不改变其他端口的网格映射，也不更新网络或 replay。

实验按以下顺序收集机制证据：

1. 端口 323 在 mixed 上扫描 14 个中心、已有优选点和扩大范围点；
2. 按完成率安全条件筛选 p95 最优三个点，并在 incast 上检查副作用；
3. 将胜出参数放到高漂移端口 191 复现，只有出现拥塞样本时才计为第二端口
   证据；
4. 在无拥塞端口 371 运行相同参数，作为路径特异性的负对照；
5. 只有第二个拥塞端口方向一致，才考虑重构全局动作范围或连续动作算法。

对应入口为
`scripts/continual_validation/run_physical_range_sweep.sh`，输出报告为
`PHYSICAL_RANGE_REPORT.md`。该阶段不设置 PASS/FAIL 门槛，结论同时依据 p95、
完成率、目标端口拥塞 reward 和新任务代价。

## 18. 当前实验转向：先修正 ACC 动作空间，再构造受控遗忘

扩大范围的单端口复测显示，端口最大队列只有 24.56 KB 时，旧动作空间最小
`Kmin=32 KB` 不会触发 ECN；降低到 16/32 KB 虽改变了队列，但并未稳定改善
端到端 p95。继续逐点手调无法形成可靠的灾难性遗忘证据。因此当前停止 SOR
比较和单端口阈值搜索，转入预注册的 ACC 动作空间实验。

代码新增 `multiscale` 动作空间：将合法 `(Kmin,Kmax)` 作为一个 Profile 头，
另设 Pmax 头。40 Gbps Profile 覆盖 `(8,24)` 至 `(80,160)` KB，共 9 个
阈值对和 7 个 Pmax，既覆盖低队列端口，又避免独立动作头产生非法组合。

新任务 `samepath_steady` 与 `samepath_burst` 使用相同的 640 条源、目的、大小
流，只改变周期内到达时间的分散或聚集。实验按“冻结任务生成 → 五动作静态冲突
筛选 → ACC A→B 顺序训练”执行。只有先观察到不同最优动作和可测性能敏感性，
才开展长训练。完整命令和结论边界见
`docs/acc-forgetting-multiscale-experiment.md`。
