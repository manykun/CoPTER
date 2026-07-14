# SOR（Self-Organizing Replay）方案原理分析报告

> 范围：`/root/paddlejob/workspace/yangziwen/CoPTER/sor/`
> 主要参考：`SOR_submission_version.docx`（论文草稿） + 代码实现（`sor_copter.py`、`sor_agent.py`、`sor_agent_helper.py`、`sor_replay.py`、`backbone_sor.py`）

---

## 1. 提出背景与问题定义

### 1.1 应用场景
SOR 部署在数据中心拥塞控制（DCQCN 类参数自适应）任务上：
- 由 NS-3 模拟环境驱动（见 `sor_copter.py` 中的 `NetworkHelper(ns3_socket=...)`），controller 周期性地为每个端口选择一组拥塞控制参数 `(K_min, K_max, P_max)`（动作维度 6×4×10，见 `SORACC.action_values`，`sor_agent.py:38-43`）。
- 状态由队列长度、队列梯度、ECN 比率、发送速率、利用率、短期吞吐方差等组成（论文式(2)）。
- 业务流量（incast / random / all-to-all / all-reduce 等）随时间持续切换、混合，**没有显式的 task label，也没有固定 task boundary**。

### 1.2 核心挑战：DDQN 的灾难性遗忘
将其建模为非稳态 MDP `M_t = (S, A, P_t, R_t, γ)`（论文式(1)）。在 DDQN + 普通 replay buffer 下：
- 最近的样本会在 buffer 中占主导，旧 regime 的覆盖率快速衰减；
- Q 网络偏向当前流量模式，对突发/同步类（incast、all-reduce barrier）旧模式遗忘严重；
- 即使切回旧模式，恢复（recovery）速度差。

SOR 的目标即：**在不依赖 workload label / task boundary 的前提下，结构化地组织经验，主动重放正在被遗忘的 regime。**

---

## 2. 方案总体架构

SOR 把 replay 从一个扁平 FIFO 队列，重构成由 4 个在线模块组成的「结构化记忆系统」：

```
              ┌────────────────────────────────────────────────────┐
   s_t  ──►   │  ① 表征学习 f_θ（编码器，与策略网络共享）            │
              └────────────┬───────────────────────────────────────┘
                           ▼ z_t
              ┌─────────────────────────┐
              │  ② 在线 regime 发现       │   原型集 C = {μ_1,…,μ_K}
              │  （增量原型，无标签）      │   阈值 δ → 新 prototype
              └─────────────┬─────────────┘
                            ▼ c_t
              ┌─────────────────────────────────────────────────┐
              │  ③ 结构化 Replay Buffer                           │
              │     B = B_cluster ∪ B_boundary ∪ B_recent       │
              └─────────────┬─────────────────────────────────────┘
                            ▼
              ┌─────────────────────────────┐      ┌──────────────────────┐
              │  ④ Drift / 遗忘检测         │ ──►  │  ⑤ 自适应采样          │
              │  D_k = λ1·KL + λ2·Δδ̄ + λ3·Δr̄ │     │  P(i)∝exp(u_i/T)      │
              └─────────────────────────────┘      └──────────┬───────────┘
                                                              ▼
                                                ┌─────────────────────────┐
                                                │  ⑥ DDQN 训练（TD+一致性+正则）│
                                                └─────────────────────────┘
```

代码实现的对应关系：

| 论文模块 | 代码对应 |
|---|---|
| 编码器 `f_θ` | `SORTripleHeadACC.encoder`（`backbone_sor.py:8-17`），与策略 head 共享主干 |
| 原型管理（regime discovery） | `PrototypeManager`（`sor_replay.py:46-104`） |
| 结构化记忆 | `StructuredSORReplayBuffer`（`sor_replay.py:177-317`） |
| Drift 跟踪 | `DriftTracker`（`sor_replay.py:107-174`） |
| 自适应采样 | `StructuredSORReplayBuffer.sample` + `_score`（`sor_replay.py:216-228, 299-308`） |
| DDQN + 一致性 + 漂移正则 | `SORACC.train_model`（`sor_agent.py:92-147`） |

---

## 3. 核心机制详解

### 3.1 表征学习与潜在 regime 发现

- **编码**：`z_t = f_θ(s_t)`（论文式(3)）。代码中 `state_dim=18 → 32 → 64 → 64 → embedding_dim=32`，再分支到三个动作 head。**编码器和 Q-head 共享主干**，因此表征同时被 TD loss 和 prototype 一致性 loss 共同塑形——“相似动力学的状态在 latent 空间相邻”。

- **增量原型 + 距离阈值**（论文式(4)(5)，代码 `PrototypeManager.assign/update`）：
  - 对新嵌入 `z_t`，找最近原型 `c_t = argmin_k ||z_t-μ_k||₂`；
  - 若最小距离 > 阈值 `δ`（默认 `prototype_distance=1.0`）且未达 `max_clusters=32`，**自动开新 cluster**；
  - 已有 cluster 用 EMA 更新：`μ_k ← (1-η)μ_k + η z_t`（默认 `η=0.05`）。
  - 实现完全无监督、无 task label，且不预设 regime 数量上限以外的语义。

### 3.2 三段式结构化 Replay Buffer

论文式(8)：`B = B_cluster ∪ B_boundary ∪ B_recent`，对应 `StructuredSORReplayBuffer.__init__`：

| 子缓冲区 | 作用 | 默认容量 |
|---|---|---|
| `cluster_memory[k]`（每个 cluster 一个 deque） | 保存属于 cluster k 的转移，保证旧 regime 不被新数据冲掉 | `rb_size=1000` / cluster |
| `boundary_memory` | 收集“regime 切换瞬间”的转移（高价值、易遗忘） | `boundary_size=20000` |
| `recent_memory` | 保存最新若干样本，支持快速适应新流量 | `recent_size=2000` |

**Boundary 检测**（论文式(7)，代码 `_is_boundary`，`sor_replay.py:282-286`）：
```
Δ_t = ||s_t - s_{t-1}||₂ + |r_t - r_{t-1}|
boundary if Δ_t > τ_b   # 默认 boundary_threshold=0.5
```
这个简单的不连续性度量同时利用状态和 reward 的突跳，恰好能捕捉 incast 起爆、ECN 标记跳变、all-reduce 同步爆发等场景。

### 3.3 Drift-Aware 遗忘检测

`DriftTracker` 对每个 cluster `k` 同时维护 **reference 分布**（首次窗口稳定后冻结）和 **current 分布**（滚动窗口 `stats_window=256`），追踪三类统计：嵌入均值、TD 误差均值、reward 均值。

漂移分数（论文式(9)）：

```
D_k = λ_emb · ||μ_emb^cur − μ_emb^ref||
    + λ_td  · |δ̄_k^cur − δ̄_k^ref|
    + λ_r   · |r̄_k^cur − r̄_k^ref|
```

代码权重默认 `drift_embedding_weight=0.3, drift_td_weight=0.4, drift_reward_weight=0.3`（`sor_replay.py:131-135`）。

> 注：论文中第一项写作 `D_KL(P^cur || P^ref)`；代码用 **嵌入均值的 L2 距离**作为简化代理（更稳定、计算成本低）。语义上等价 —— 都是衡量该 regime 当前是否偏离它的“参考状态”。
>
> `D_k` 越大，说明该 cluster 当前已经被模型“跑偏 / 遗忘”，应当多回放。

### 3.4 自适应采样：四因子打分 + softmax

每个 transition 的采样得分（论文式(10)，代码 `_score`）：

```
u_i = α·|δ_i|              # TD error，PER 风格的难样本优先
    + β·U(c_i)              # 1/(recent_hits + 1)，欠采样补偿
    + γ·D(c_i)              # 当前 cluster 漂移分数（正在被遗忘的优先）
    + ρ·B_i                 # boundary 样本加权
```

默认权重：`alpha_td=1.0, beta_under_sample=0.2, gamma_drift=0.5, rho_boundary=0.5`。

采样概率（论文式(11)）：`P(i) = softmax(u_i / T)`，温度 `T=1.0`。
代码采用「**所有子 buffer 去重后构造候选池 → softmax → 无放回采样**」，避免某个 cluster 在三个子缓冲区中重复出现导致的偏置。

> 与 PER 的关键区别：PER 只用 TD 误差，会持续聚焦“当前最难”的样本，对旧 regime 的覆盖反而更差；SOR 把「难样本」「欠采样」「漂移大」「regime 边界」四种证据**并联**，自然地保证遗忘 regime 的回放频率被拉起来。

### 3.5 训练目标：TD + 一致性 + 漂移正则

代码 `SORACC.train_model`（`sor_agent.py:92-147`）实现论文式(13)–(16)：

1. **DDQN TD loss**（论文式(12)(13)）：online 网选动作，target 网估值；动作是三元组（K_min/K_max/P_max），Q 值为三个 head 的和（`_gather_q`）。

2. **表征一致性 loss**（论文式(14)）：
   ```
   L_cons = mean‖ z_t − μ_{c_t} ‖²
   ```
   把每个样本的嵌入拉向其 cluster 原型，让 cluster 几何更紧致、可分。`λ_cons=0.01`。

3. **高漂移 cluster 的 Q 正则**（论文式(15)）：维护一个慢更新的 `reference_net`（每 `ref_update_interval=256` 步从 policy 同步一次）；对 `D_k > drift_reg_threshold(=0.5)` 的样本，惩罚当前 Q 与参考 Q 的差：
   ```
   L_reg = mean over high-drift samples of ‖ Q_θ − Q_ref ‖²
   ```
   `λ_reg=0.001`。**作用**：在该 regime 漂移严重、即将被遗忘时，把 Q 拉回最近一次稳定的参考，缓解灾难性遗忘。

最终：`L = L_TD + λ_cons · L_cons + λ_reg · L_reg`。

---

## 4. 完整训练循环（与代码对照）

`Algorithm 1` 在代码中分布在两处：环境交互在 `sor_copter.py:114-149`，单步训练在 `SORAgentHelper.train` + `SORACC.train_model`。

```text
for each step t:
  s_t ← network_helper                                             # sor_copter.py:125
  z_t = encoder(s_t)                                               # SORACC.encode_state
  a_t = ε-greedy on Q_θ(s_t)                                       # decide()
  r_t, s_{t+1} ← env.step(a_t)                                     # configurator + monitor
  c_t = PrototypeManager.assign(z_t); update prototype             # rb.push
  if Δ_t > τ_b: store in B_boundary
  store in B_cluster[c_t] and B_recent
  DriftTracker.update(c_t, z_t, r_t, td)                           # update D_k
  if t % train_intervals == 0:
      sync(): 从 global_memory 用 SOR 采样补每端口的本地 buffer
      batch = rb.sample(N)                                          # softmax(u_i/T)
      L = L_TD + λ_cons L_cons + λ_reg L_reg                        # train_model
      θ ← θ − ∇L
      每 target_update_interval 步更新 target_net
      每 ref_update_interval 步更新 reference_net
```

每个端口 (port) 拥有独立 SORACC + 本地 `StructuredSORReplayBuffer`，并额外维护一个全局结构化 buffer `global_memory`，`sync()` 把全局采样下沉到各端口（`sor_agent_helper.py:130-135`），让多端口经验共享同时保留个体差异。

---

## 5. 设计亮点 & 与已有方法对比

| 方法 | 是否需要 task label | 是否处理 regime 切换 | 采样策略 |
|---|---|---|---|
| 普通 DQN + FIFO replay | 否 | 否，旧数据被覆盖 | 均匀 |
| PER（Prioritized Replay） | 否 | 弱（仅靠 TD 难度） | TD 误差 |
| EWC / 知识蒸馏类持续学习 | **需要** task 边界 | 是 | — |
| **SOR** | **不需要** | 主动检测漂移并补偿 | TD + 欠采样 + 漂移 + boundary |

突出的工程价值：
1. **完全无标签**：在数据中心这种 workload 边界本来就模糊的场景下尤其关键。
2. **结构化记忆 + 边界缓冲**：让“流量切换瞬间”成为一等公民，恢复速度（recovery time）显著优于扁平 buffer。
3. **drift-aware 软正则**：既不会像硬 EWC 那样把网络冻死，也不会像普通 DDQN 那样无约束漂移；只在“快忘了”的 cluster 才上正则。

---

## 6. 评估协议

按 `SOR_submission_version.docx` 第 9 节，评估在多种数据中心 workload 混合下进行（incast / random / web / Hadoop A2A / DL/Storage all-reduce），且**顺序随机化或混杂**，禁止固定 task schedule。指标包括：
- 任务级：吞吐、FCT、丢包；
- 持续学习级：average forgetting、worst-case forgetting、backward transfer；
- **recovery time**（regime 切换后恢复时间）—— 这是网络场景下最实用的指标。

工程侧产物（`eval_models_sor/` 目录）：
- 每端口的结构化 buffer pickle：`*_sor_rb_port{N}.pkl`；
- 全局结构化 buffer：`*_sor_global_rb.pkl`；
- 每端口三套权重：`*_SORACC_{N}_policy.pt / target.pt / reference.pt`（注意有 reference_net 单独保存，对应论文 L_reg 的慢更新 ref）。

---

## 7. 关键超参速查（默认值）

| 参数 | 默认 | 说明 |
|---|---|---|
| `prototype_distance δ` | 1.0 | 新建 cluster 的距离阈值 |
| `prototype_eta η` | 0.05 | 原型 EMA 更新率 |
| `max_clusters K` | 32 | 最大 cluster 数 |
| `boundary_threshold τ_b` | 0.5 | regime 切换检测阈值 |
| `recent_size` | 2000 | recent 缓冲区容量 |
| `boundary_size` | 20000 | boundary 缓冲区容量 |
| `rb_size`（每 cluster） | 1000 | cluster 子缓冲容量 |
| `α_td / β_us / γ_drift / ρ_boundary` | 1.0/0.2/0.5/0.5 | 采样四因子权重 |
| `temperature T` | 1.0 | softmax 采样温度 |
| `λ_cons / λ_reg` | 0.01 / 0.001 | 一致性 / 漂移正则系数 |
| `drift_reg_threshold` | 0.5 | 触发 L_reg 的 D_k 阈值 |
| `ref_update_interval` | 256 步 | reference_net 慢同步周期 |
| `stats_window` | 256 | DriftTracker 当前窗口长度 |

---

## 8. 小结

SOR 的本质是把「持续学习」嵌进「replay buffer 设计」里：

1. **用共享编码器把状态压到 latent 空间**，用增量原型在线发现 regime；
2. **用三段式记忆**显式分别保护「每个旧 regime 的代表样本」「regime 切换瞬间」「最新样本」；
3. **用 drift score 显式估计哪些 regime 正在被遗忘**，并通过 (i) 采样概率提升、(ii) Q 参考正则两个通道把它“拉回来”；
4. 全程 label-free，复杂度可控，与 DDQN 训练目标只是加了两项轻量正则即可对接。

对数据中心这种「workload 隐式且持续混合」的场景，SOR 把扁平 replay 的根本性缺陷（被最近样本主导）从机制上消除，是相对 PER 等先验工作有清晰增益点的设计。
