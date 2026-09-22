# 1. 系统架构、运行链路与核心概念

## 1.1 项目要解决的问题

DCQCN 使用三个重要 ECN 参数控制拥塞反馈：

- `Kmin`：队列低于该阈值时通常不标记 ECN。
- `Kmax`：队列达到或超过该阈值时按最大概率标记。
- `Pmax`：`Kmin` 到 `Kmax` 区间内的最大标记概率尺度。

静态参数难以同时适应吞吐型流量、短流、突发 incast 和多租户混合流量。CoPTER 让每个交换机端口拥有一个智能体，根据该端口近期状态动态选择三个参数。

## 1.2 分层结构

```text
实验调度层
run_training.sh / scripts_exp / curriculum scripts
        │
        ├── 启动 ns-3 进程
        └── 启动 Python Agent 进程

Python 学习层
copter.py / agent_helper.py / agent.py / network_helper.py
        │
        └── ns3-gym socket

C++ 仿真层
copter-sim.cc / SwitchNode / SwitchMmu / RdmaHw / QbbNetDevice
        │
        └── topology + flow + conf

数据与评估层
FCT / queue / rate / throughput / PFC / TensorBoard / JSONL
```

## 1.3 一次训练 step 的完整过程

### 第一步：ns-3 采样端口状态

[`copter-sim.cc`](../../ns-3.33/scratch/copter-sim.cc) 每 500 微秒触发一次 OpenGym 状态通知。对每个交换机数据端口产生六维观测：

```text
o_t = [
  peak_queue_occupancy,
  tx_rate_norm,
  ecn_rate_norm,
  kmin_norm,
  kmax_norm,
  pmax
]
```

队列值使用上一个观察窗口内的峰值，而不是只使用采样瞬间的队列，因此能够捕获短突发。读完峰值后，C++ 会重置该端口的峰值统计。

### 第二步：Python 组成状态

[`network_helper.py`](../../copter/network_helper.py) 为每个端口保留最近四次 observation。训练使用前三帧到后三帧的滑动转换：

```text
state_t      = [obs(t-2), obs(t-1), obs(t)]
next_state_t = [obs(t-1), obs(t),   obs(t+1)]
```

每帧 6 维，因此状态为 18 维。

### 第三步：每端口智能体选择动作

[`AgentHelper.decide()`](../../copter/agent_helper.py) 遍历所有端口 Agent。ACC 网络输出三个 Q 向量：

```text
Q_kmin: 6 个值
Q_kmax: 4 个值
Q_pmax: 10 个值
```

epsilon-greedy 决定随机探索或分别取三个 head 的 `argmax`。

### 第四步：动作回传并映射为物理参数

Python 返回归一化动作：

```text
[kmin_norm, kmax_norm, pmax]
```

C++ 根据端口速率缩放阈值范围。25 Gbit/s 基准范围为：

```text
Kmin: 20,000 ～ 50,000 bytes
Kmax: 50,000 ～ 100,000 bytes
Pmax: 0.1 ～ 1.0
```

随后调用 `SwitchMmu::ConfigEcn()`，新参数从后续数据包开始生效。

### 第五步：Python 计算奖励

C++ 的 OpenGym reward 固定返回 0；真正的每端口奖励在 Python 计算：

```text
reward = 0.50 * throughput_reward
       + 0.30 * queue_reward
       + 0.20 * ecn_reward
```

其中：

```text
throughput_reward = clamp(avg_tx_rate, 0, 1)

combined_queue = 0.7 * peak_queue + 0.3 * avg_queue
queue_reward = exp(-6 * combined_queue)

ecn_reward 在 ECN rate≈0.03 时最高；过高时指数衰减
```

统计窗口长度为 32 个 step。主循环的 rollout 指标只聚合拥塞端口，避免大量空闲端口把奖励平均成几乎不变化的常数。

### 第六步：记录和训练

在线模式下，每个端口记录：

```text
(state, action_index_tuple, reward, next_state)
```

普通 ACC 使用均匀随机 replay；SOR 使用结构化 replay。达到训练间隔后从经验池采样并进行 DDQN 更新，定期把 policy 参数硬复制到 target network。

## 1.4 ACC 的 Double DQN

当前动作的预测值是三个 head 中被选动作的 Q 值之和：

```text
Q(s,a) = Q_kmin(s,a_kmin)
       + Q_kmax(s,a_kmax)
       + Q_pmax(s,a_pmax)
```

下一状态由 policy network 选动作、target network 估值：

```text
a* = argmax Q_policy(s', ·)
y  = r + gamma * Q_target(s', a*)
```

损失使用 Smooth L1，学习率 `1e-3`，`gamma=0.95`，梯度范数裁剪为 10。

当前 transition 没有保存 `done`，目标值也没有终止状态 mask。实验以进程结束作为 episode 结束，因此最后 transition 仍可能 bootstrap；这是需要明确验证的实现细节。

## 1.5 每端口多智能体结构

拓扑中有多少交换机数据端口，就创建多少 Agent。每个 Agent 有独立的：

- policy network；
- target network；
- 本地 replay buffer；
- checkpoint。

同时存在一个共享 replay buffer：本地经验上传到共享池，共享池再抽样下发给各端口。这有利于端口之间迁移经验，但也可能让空闲端口或不同角色端口的分布互相污染。

## 1.6 ACC、CoPTER、SOR 的继承关系

```text
共同环境：ns-3 + NetworkHelper + 18维状态 + 3个动作参数

ACC
 ├── 小型三头网络
 ├── 均匀 replay
 └── DDQN loss

CoPTER
 ├── 更大型三头网络
 ├── 端口级 fmap 先验
 └── 当前 Kmin/Kmax 选择主要由 fmap 决定

SOR-ACC
 ├── ACC 风格三头网络 + embedding
 ├── prototype cluster
 ├── recent/boundary/cluster memories
 ├── drift-aware sampling
 └── consistency + reference regularization
```

## 1.7 配置文件的三类参数

### ns-3 `.conf`

决定物理仿真：拓扑、流量、拥塞控制模式、Buffer、静态 ECN 初值、输出文件和 OpenGym 动作范围。

### Agent YAML

决定训练：episode 数、实验名、模型目录、epsilon、训练间隔、TensorBoard、SOR 参数等。

### 启动脚本参数

决定调度：端口、配置路径、并行实验、断点续训、是否 greedy eval、固定动作等。脚本参数可能覆盖 YAML 或 `.conf`，排错时必须记录最终值。

## 1.8 关键单位

| 位置                          | 变量                              | 实际单位                                              |
| ----------------------------- | --------------------------------- | ----------------------------------------------------- |
| `.conf`                     | `BUFFER_SIZE`                   | KB；C++ 乘以 1024                                     |
| C++ OpenGym 范围              | `OPENGYM_*_KMIN/KMAX`           | bytes                                                 |
| `SwitchMmu::ConfigEcn` 参数 | Kmin/Kmax                         | KB；内部再转 bytes                                    |
| Python`--switch_buffer`     | 当前代码按“相对 400 的数值”使用 | 帮助文本写 bytes，但实验配置通常传 KB，存在语义不一致 |
| flow 文件                     | flow size                         | bytes                                                 |
| flow 文件                     | start time                        | seconds                                               |
| FCT 输出                      | 时间字段                          | ns                                                    |

## 1.9 当前最重要的实现风险

1. **队列二次归一化**：C++ 已用 `peak_bytes / actual_buffer_bytes`，Python 又乘 `switch_buffer_size / 400`。只有传入 400 时保持原值；其他 Buffer 可能被放大、截断到 1。
2. **CoPTER 融合退化**：当前 `fusion = 0*q_norm + 1*f_norm`，Kmin/Kmax 实际不使用在线 Q 矩阵。
3. **ACC checkpoint 方法重复定义**：`ACC` 类前后各定义一次 `save_model/load_model`，Python 只使用后面的定义；阅读前半部分时容易误判文件格式。
4. **经验池记录所有端口**：rollout reward 已过滤拥塞端口，但 replay 仍可包含大量空闲端口经验。
5. **启动脚本残留硬编码路径和缺失场景**：多份脚本仍指向 `/root/paddlejob/...` 或仓库中不存在的 `.conf`，不能直接在 `/mnt/sdb1/xuduokun/...` 使用。
6. **M3 实现缺失**：`m3-ml/` 不是可运行模型目录，不能把 `m3_256hosts.conf` 等同于运行 M3。

## 1.10 一次实验应该保存什么

为了让结果可复现，每次实验至少保存：

- Git commit SHA；
- `.conf` 和 YAML 的副本；
- seed、动作范围、Buffer、epsilon schedule；
- Agent/NS3 日志；
- 模型、optimizer、replay 和训练状态；
- FCT、队列、ECN、rate、throughput、PFC 输出；
- 分析脚本输出和图表。

当前 `scripts_exp/common.sh` 的原子写入、锁、状态文件和归档思路比早期直接循环脚本更适合作为正式实验入口。
