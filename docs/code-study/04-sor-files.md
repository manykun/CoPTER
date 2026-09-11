# 4. SOR：持续学习与抗遗忘代码逐文件讲解

## 4.1 SOR 在系统中的位置

SOR 没有修改 ns-3 里的 DCQCN、ECN 或 PFC。它复用 ACC 的：

- 六维 observation 和 18 维 state；
- 6/4/10 三头动作空间；
- Python reward；
- ns3-gym 交互；
- 每端口一个 Agent 的结构。

它主要替换三部分：

1. 网络显式暴露 32 维 embedding；
2. 普通 FIFO replay 变成结构化 replay；
3. DDQN loss 增加原型一致性和参考网络正则化。

目标是处理 A→B→A 一类非平稳流量，使学习 B 时不完全覆盖 A 的知识。

## 4.2 [`sor/backbone_sor.py`](../../sor/backbone_sor.py)

定义 `SORTripleHeadACC`：

```text
18 → 32 → 64 → 64 → embedding(32)
                         ├─ Kmin head: 6
                         ├─ Kmax head: 4
                         └─ Pmax head: 10
```

- `encode(x)` 只返回 embedding，供聚类。
- `forward(x, return_embedding=True)` 同时返回三个 Q head 和 embedding，供训练一致性损失。

网络容量与普通 ACC 接近，所以 ACC/SOR 差异主要来自 replay 与正则化，不是简单通过更深网络获得优势。

## 4.3 [`sor/sor_replay.py`](../../sor/sor_replay.py)

这是 SOR 的核心文件。

### `Transition`

除普通 `(s,a,r,s')` 外，还保存：

- `embedding`：样本进入 replay 时的编码。
- `cluster_id`：原型聚类编号。
- `boundary`：是否是状态切换边界。
- `td_error`：当前优先级的一部分。
- `sample_count`：被采样次数。

embedding 是写入时的快照；网络更新后不会自动重编码旧样本。因此 prototype 空间会混合不同训练阶段的 encoder 表示。

### `SORReplayConfig`

默认参数包括：

```text
per-cluster capacity = 1000
global capacity = 100000（全局池使用）
recent memory = 2000
boundary memory = 20000
max clusters = 32
prototype distance threshold = 1.0
prototype EMA eta = 0.05
boundary threshold = 0.5
```

采样权重系数：TD=1.0、欠采样=.2、漂移=.5、边界=.5；TD error 截断到 10，softmax temperature=1，并混入 1% 均匀概率。

### `PrototypeManager`

`assign()` 找与 embedding 欧氏距离最近的 prototype：

- 无 prototype：创建 cluster 0。
- 最近距离大于阈值且未到 32 个：创建新 cluster。
- 否则：加入最近 cluster。

`update()` 用：

```text
prototype = (1-eta)*old + eta*embedding
```

更新中心。它不是 K-means 的全量重算，适合在线流式数据，但 cluster 编号和结果依赖样本到达顺序。

### `DriftTracker`

每 cluster 保存最近 256 条 embedding、TD error 和 reward。至少积累 32 条后建立 reference summary，之后计算：

```text
drift = 0.3 * ||embedding_mean_now - reference||
      + 0.4 * |td_mean_now - reference|
      + 0.3 * |reward_mean_now - reference|
```

reference 建立后不会随时间更新，因而衡量的是相对早期分布的偏移，而不是自适应变化点检测。

### `StructuredSORReplayBuffer.push()`

依次完成：编码转数组、cluster 分配与更新、边界判断、设置初始 TD、写入 cluster memory、容量淘汰、写 recent/boundary memory、更新 drift。

新样本如果没有 TD error，就继承当前最大 TD，避免在尚未训练前立即被低分淘汰。

### 边界判断

```text
||state - last_state||₂ + |reward - last_reward| > 0.5
```

这里的 `last_state` 是“上一次 push 到这个 buffer 的样本”，不是按端口和时间戳分组。对全局池而言，不同端口连续 push 会互相比较，因此可能把端口差异误判成流量切换边界。

### `push_existing()`

全局池向各本地池广播时直接复用已有 Transition，不重新聚类、不更新 prototype/drift、不重新判断边界。这样速度快，但本地池可能得到一个 cluster ID，却没有对应的本地 prototype；训练一致性损失会回退到零向量或使用语义不一致的中心。

同时，同一个 Transition 对象被广播给多个端口池。某端口更新其 `td_error/sample_count` 时，其他池看到的对象也会改变，形成隐式共享可变状态。

### `sample()` 与 `_score()`

候选集合是 cluster/recent/boundary memory 的对象去重并集。每条经验得分：

```text
score = clipped_td
      + .2/(recent_cluster_hits+1)
      + .5*cluster_drift
      + .5*boundary
```

经 temperature softmax 和 1% uniform mix 后无放回采样。欠采样项按近期“cluster 被采样次数”计算，不是按 cluster 总容量。

### 容量淘汰

- 本地池：每 cluster 超容量时删除该 cluster 最低分样本。
- 全局池：总量超限时从当前最大 cluster 中删除最低分样本。
- boundary memory 单独超限时也按最低分删除。

从最大 cluster 淘汰有助于保护稀有旧场景，但不是全局严格最低分淘汰。

### 持久化

`state_dict/save/load` 保存 prototype、drift、cluster/recent/boundary memories 和统计状态，使用 pickle。只能加载可信文件；pickle 不适合接收未知来源数据。

## 4.4 [`sor/sor_agent.py`](../../sor/sor_agent.py)

定义单端口 `SORACC`。

### 三个网络

- `policy_net`：每个训练 step 更新。
- `target_net`：DDQN 目标估值，按 target interval 更新。
- `reference_net`：用于抗遗忘正则，每 256 个本 Agent 训练 step 从 policy 硬同步。

三个网络初始相同并在 CPU 运行。

### `encode_state()`

用 policy encoder 生成 NumPy embedding。record 阶段调用，因此写入 replay 的 embedding 与当时的 policy 参数绑定。

### `select_action()`

动作值与 ACC 完全一致，epsilon-greedy 逻辑也一致，保证可以做相对公平的控制器对比。

### `train_model()`

#### TD loss

与 ACC 相同：policy 选下一动作，target 估值，三个 head Q 相加。

#### prototype consistency loss

取每条样本 cluster 对应 prototype，最小化当前 embedding 与 prototype 的均方差：

```text
loss_cons = mean((embedding_current - prototype_saved)^2)
```

prototype 缺失时使用零向量，这会把 embedding 拉向 0，未必是期望行为。

#### reference regularization

只对 drift score 大于阈值的样本，约束 policy 三个 head 的输出不要偏离 reference network：

```text
loss_reg = mean_head MSE(Q_policy, Q_reference)
```

总损失默认：

```text
loss = loss_td + 0.01*loss_cons + 0.001*loss_reg
```

训练后返回 TD errors 给 replay 更新优先级。

### checkpoint

保存 policy、target、reference 三个 `.pt`。与 ACC 的单 checkpoint 格式不相同，比较脚本必须使用各自 AgentHelper 加载。

## 4.5 [`sor/sor_agent_helper.py`](../../sor/sor_agent_helper.py)

把 SORACC 扩展到所有端口。

### 内存结构

- 每端口一个 `SORACC`。
- 每端口一个 `StructuredSORReplayBuffer`。
- 一个容量更大的全局 `StructuredSORReplayBuffer`。

全局池增加 `global_total_cap`，并扩大 boundary/recent memory。

### `record()`

先用对应端口 Agent 编码 state，再分别 push 到本地池和全局池。两次 push 会独立分配 cluster，所以本地/全局 cluster ID 不保证语义一致。

### `sync()/maybe_sync()`

每 8 次调用才执行一次 sync。全局池只 sample 一次，然后把相同 Transition 对象广播给所有本地池，减少复杂度，但产生前述 prototype 与对象共享问题。

### `train()`

逐端口从结构化池采样，传入该本地池 prototype/drift，训练后把 TD errors 写回。任何端口经验不足会跳过。target 更新仍依据主循环 `current_step`。

### `load()/save()`

加载三个网络和全局 replay；本地 replay 冷启动，依赖后续 sync 重建。保存时每 epoch 保存模型和轻量 train state，但全局大 pickle 默认每 5 epoch 才写一次，降低 I/O 卡顿。

这意味着崩溃恢复时最近最多 4 个 epoch 的 replay 变化可能丢失，而模型/train state 可能更新得更近，二者不是严格事务一致。

## 4.6 [`sor/sor_copter.py`](../../sor/sor_copter.py)

SOR 主入口，整体复制 ACC 主循环并替换为 `SORAgentHelper`。

### `build_parser()`

除 ACC 参数外，增加 recent/boundary 容量、cluster、prototype、boundary、采样权重、consistency/reg、reference update、sync cadence 和 buffer save cadence。

### `main()`

初始化 `NetworkHelper`、构造 `SORReplayConfig` 和 Helper，随后执行 static steps、决策、monitor、record、maybe_sync、train 和持久化。

与 ACC 主入口长期复制会产生行为漂移：例如奖励聚合口径、日志字段、调试参数可能只在一边更新。更好的重构是共享一个 runner，通过 AgentHelper 接口注入 ACC/SOR。

## 4.7 [`sor/test_sor_replay.py`](../../sor/test_sor_replay.py)

包含 8 个无 pytest fixture 的直接断言测试：

1. push/sample/persist；
2. 极端 TD 下采样不崩溃；
3. 新样本不会先被淘汰；
4. 基于 score 的淘汰；
5. 更严格的淘汰顺序；
6. `push_existing` 不重算 prototype；
7. push 后候选 cache 失效；
8. 采样概率合法。

它很好地覆盖单 buffer 数值稳定性，但没有覆盖：多端口全局同步、共享 Transition 副作用、cluster prototype 一致性、A-B-A 端到端遗忘。

## 4.8 [`sor/sor_training_config.yaml`](../../sor/sor_training_config.yaml)

SOR 连续训练默认配置。前半部分复用 ACC 字段，`sor:` 段映射到 `SORReplayConfig` 和 SORACC loss 参数。默认场景文件在当前分支缺失，`switch_buffer=10000` 也需与实际 `.conf` 对齐。

## 4.9 [`sor/acc_eval_config.yaml`](../../sor/acc_eval_config.yaml)

ACC 对比配置：120 epochs、模型目录 `eval_models_acc_baseline`、TensorBoard `tb_logs_acc`。其场景 `acc_Hadoop_Shuffle.conf` 不在当前分支，所以不能原样运行。

## 4.10 [`sor/sor_eval_config.yaml`](../../sor/sor_eval_config.yaml)

SOR 对比配置：1 epoch，并包含完整 SOR 参数。ACC 为 120、SOR 为 1 的默认 epoch 数并不公平；实际比较调度器可能覆盖 `TOTAL_EPOCHS_OVERRIDE`，必须检查最终运行日志。

## 4.11 [`sor/acc_curriculum_config.yaml`](../../sor/acc_curriculum_config.yaml)

ACC A-B-A/遗忘实验配置。`NS3_CONF_OVERRIDE` 在每阶段切场景。epsilon 计划为 800 env steps，注释意图是让 phase 1 内探索结束，使后续任务覆盖旧策略更明显。

但该文件 `switch_buffer=100`，而其他多数配置为 10000/400/80；如果 `.conf` 不一致，会触发队列归一化偏差。

## 4.12 [`sor/sor_curriculum_config.yaml`](../../sor/sor_curriculum_config.yaml)

SOR curriculum 配置。epsilon decay 为 4000，与 ACC 的 800 不一致，尽管注释写“必须匹配 ACC”。若用它们比较遗忘，差异可能来自探索计划而不只是 SOR。

## 4.13 [`sor/acc_watch_hadoop_config.yaml`](../../sor/acc_watch_hadoop_config.yaml)

单场景 ACC 学习曲线配置，专门配合 watch ports 和固定动作基线。`switch_buffer=80`，注释明确要求和场景 Buffer KB 对齐，是这些 YAML 中单位说明最清楚的一份。

## 4.14 [`sor/run_sor_train.sh`](../../sor/run_sor_train.sh)

读取 SOR YAML，展开所有 SOR 参数并循环启动 `sor_copter.py`。支持 greedy eval/tag、TensorBoard、收敛判断和跨 epoch 状态。它只启动 Agent，需配合 ns-3 侧脚本。

## 4.15 [`sor/run_acc_sor_ns3_compare.sh`](../../sor/run_acc_sor_ns3_compare.sh)

旧对比脚本：检查/创建 Conda 环境，先后台启动 ns-3、固定 sleep 5 秒，再启动 Agent，最后调用 `compare_acc_sor.py`。

这个启动顺序与当前 OpenGym 实际握手不完全稳健；更新版脚本采用“Agent 先监听，ns-3 后连接”。文件还硬编码 `/root/paddlejob/...` 和 Conda 路径。

## 4.16 [`sor/compare_acc_sor.py`](../../sor/compare_acc_sor.py)

读取两种方法的 metrics JSONL，计算记录数、最后 reward/loss、窗口均值等摘要并输出对比。它比较的是训练/rollout 指标，不替代物理 FCT/slowdown/queue 对比。

## 4.17 [`sor/compare_forgetting.py`](../../sor/compare_forgetting.py)

解析带 `train_pN_task/eval_pN_task` 标签的 JSONL：

- 按 phase/task 构造 eval table；
- 计算 forgetting；
- 计算 backward transfer；
- 计算重访任务 recovery；
- 可选绘图。

它是 A-B-A 评价的算法层工具。输入标签格式不匹配时会丢记录，运行前先检查 metrics 的 `eval_tag`。

## 4.18 [`sor/0526/plot_mean_reward.py`](../../sor/0526/plot_mean_reward.py)

历史日期目录下的奖励绘图脚本，读取旧日志/指标并绘制 mean reward。路径和文件命名可能对应 05-26 版本，不属于当前统一实验流水线。

## 4.19 [`sor/SOR_方案原理分析.md`](../../sor/SOR_方案原理分析.md)

项目已有的 SOR 设计说明，适合先了解动机、cluster、boundary、drift 和正则化。但其中部分 state/reward 描述与当前共享 `NetworkHelper` 已不完全一致；应把它视为设计文档，用本章和执行代码核对实现。

## 4.20 两个 `*_train_state.json`

- [`sor/eval_sor_train_state.json`](../../sor/eval_sor_train_state.json)
- [`sor/sor_models/sor_m3_smoke_train_state.json`](../../sor/sor_models/sor_m3_smoke_train_state.json)

当前都是空文件，属于实验残留占位。加载时 JSON 解析会失败并回退默认状态，不提供可用 checkpoint 信息，建议从版本控制移除并由运行时生成。

## 4.21 SOR 正式实验前建议修复顺序

1. 全局 replay 的 boundary 按端口分别维护 last state/reward。
2. sync 时复制 Transition，避免跨端口共享可变对象。
3. 明确全局 cluster 与本地 cluster 的映射；同步 prototype/drift，或在本地重新 assign。
4. prototype 缺失时跳过 consistency loss，而不是使用零向量。
5. ACC/SOR 共用完全相同的 Buffer、epsilon、epoch、seed 和 rollout 聚合代码。
6. 增加端到端 A-B-A 小测试，验证 SOR 指标计算不是只测 replay 内部行为。
