# 2. ACC 与 CoPTER：`copter/` 逐文件讲解

## 2.1 [`copter/structures.py`](../../copter/structures.py)

这是 Python 学习端最适合首先阅读的文件，只定义数据结构，不执行训练。

### `NetworkHelperParameters`

保存环境接口维度：

- `port_states=6`：每端口一次 observation 有 6 个数。
- `port_actions=3`：每端口一次 action 有 3 个数。
- `state_observations=3`：一个 state 拼接 3 次 observation。
- `switch_buffer_size`：传给队列归一化逻辑。

### `AgentHelperParameters`

保存多 Agent 和 replay 超参数，包含本地/共享 buffer 容量、上传/下载样本数、训练 batch、target 更新间隔、epsilon 起止值、衰减步数和指标窗口。

### `DCQCNParameters`

一个动作的值对象：

```text
k_min_norm, k_max_norm, p_max
```

它保存的是将发送给 ns-3 的归一化值，不是最终字节阈值。

### `PortObservation`

一个端口的一帧六维观测：队列、发送速率、ECN 速率和三个当前参数。`to_list()` 保证拼接 state 时字段顺序固定。

### `AgentParameters`

保存单个神经网络的输入/输出维度、学习率和折扣因子。还包含 CoPTER 的 Kmin/Kmax 分辨率字段。

## 2.2 [`copter/backbone.py`](../../copter/backbone.py)

只定义 PyTorch 网络，不包含动作选择或训练目标。

### `DualHeadNN`

早期只控制 Kmin/Kmax 的两头网络，结构为 `state→32→64→64→32`。当前主路径不使用它，但 `agent.py` 仍导入，属于历史兼容代码。

### `TripleHeadACC`

ACC 当前网络：

```text
18 → 32 → 64 → 64 → 32
                    ├─ 6: Kmin Q
                    ├─ 4: Kmax Q
                    └─10: Pmax Q
```

三个 head 共享同一特征提取器。优点是参数量小、每端口创建一个网络仍能在 CPU 运行；缺点是端口行为复杂时 32 维瓶颈可能不足。

### `TripleHeadCoPTER`

CoPTER 使用更大网络：

```text
18 → 64 → 128 → 128 → 64 → 三个 head
```

它与 ACC 的核心差别不只在网络深度，还在 `agent.py` 中的 fmap 引导和动作离散集合。

### 文件中的注释代码

中间保留了多个被注释的旧网络版本。它们不会执行，做实验时不要通过取消零散注释来切网络；更可靠的做法是新增明确命名的 backbone，并在配置中选择。

## 2.3 [`copter/network_helper.py`](../../copter/network_helper.py)

这是 C++ 环境和 Python 算法之间的适配层，也是奖励是否有效的关键文件。

### `__init__()`

主要工作：

1. 用 `ns3env.Ns3Env(port=..., startSim=False)` 连接已由脚本启动的 ns-3。
2. 调用 `reset()` 建立 OpenGym 握手。
3. 从 observation shape 推导端口数量：`n_port = total_observation_dim / 6`。
4. 为每端口创建 observation 历史和长度为 32 的 queue/ECN/txrate 窗口。
5. 创建扁平 action 数组和端口动作位图。

### `configurator()`

把一个端口的 `DCQCNParameters` 写进总 action 数组的对应切片，并将位图置 1。`monitor()` 在非初始 step 会断言所有端口都已配置，防止部分端口沿用未知动作。

### `monitor()`

`current_step==0` 时执行一次特殊初始化：先 reset，再发送全零动作，以取得第一帧状态和 `info` 中的端口标识。端口标识格式为：

```text
switch_id-connected_node_id
```

CoPTER 用它查找对应的 `*_fmap.txt`。

后续 step 直接发送当前 action，并解析返回 observation。每端口构造 `PortObservation` 后写入历史和统计窗口。

### 队列归一化问题

当前代码执行：

```python
min(1, obs_queue * switch_buffer_size / 400)
```

但 C++ 的 `obs_queue` 已经是 `peak_queue_bytes / switch_buffer_bytes`。因此这段代码实质上又按 400KB 基准缩放一次：

- Buffer 参数为 400：保持不变。
- 参数为 80：队列被缩小为 0.2 倍。
- 参数为 10000：放大 25 倍并很容易截断到 1。

这是 ACC 有效性实验中应最先修复或用 A/B 实验验证的部分。另一个问题是命令行帮助写“bytes”，实际 YAML 和 `.conf` 大多按 KB 传值。

### `get_port_last_state_list()` / `get_port_current_state_list()`

前者拼历史索引 `[0,1,2]`，后者拼 `[1,2,3]`，形成 replay transition 的前后状态。调用前必须已有 4 帧，所以主循环保留至少 4 个 static step。

### `get_port_current_reward()`

从 32-step 窗口计算：

- 平均/峰值队列；
- 平均/峰值 ECN；
- 平均发送速率。

当前奖励强调吞吐率，并通过指数函数强惩罚队列。ECN 的理想点设为 0.03；零 ECN 仍得到 0.6 的 ECN 子奖励，因为无标记是否合理要同时看队列。

### 拥塞端口辅助方法

- `get_port_congestion_score()`：`10*peak_queue + 5*peak_ecn + tx_rate`。
- `is_port_active()`：有流量即可。
- `is_port_congested()`：峰值队列超过 0.005 或发生 ECN。
- `get_topk_congested_ports()`：按 congestion score 排序。

这些方法只影响 rollout 指标聚合；`AgentHelper.record()` 仍可把所有端口经验写入 replay。

## 2.4 [`copter/agent.py`](../../copter/agent.py)

同时实现抽象 Agent、ACC 和 CoPTER。

### `Agent`

只规定保存、加载、选动作、训练、更新 target 的接口。它没有使用 Python `abc`，直接调用基类方法会抛 `NotImplementedError`。

### `ACC.__init__()`

在 CPU 上创建 policy/target 两个 `TripleHeadACC`、Adam 优化器和 Smooth L1 loss。每个端口都独立创建一套。

### `ACC.select_action()`

离散动作集合：

```text
Kmin = [0, .0949, .2259, .4066, .6560, 1]
Kmax = [0, .25, .5, 1]
Pmax = [.1, .2, ..., 1]
```

随机分支对三个 head 独立均匀抽样；贪心分支分别取三个 Q 向量的 argmax。

### `ACC.train_model()`

训练 batch 的数组形状：

```text
states      [B,18]
actions     [B,3]
rewards     [B]
next_states [B,18]
```

当前动作 Q 值由三个 head gather 后相加。下一动作由 policy 选、target 评估，属于 Double DQN。没有 `done` mask，也没有 importance sampling 权重。

### ACC checkpoint 的重复定义

类中前面定义过保存为 `name_policy.pt/name_target.pt` 的方法，后面又定义保存为单个 `model_dir/name` checkpoint 的同名方法。Python 后定义覆盖前定义，实际生效的是后者，里面包含：

- policy、target；
- optimizer；
- Python/NumPy/Torch RNG 状态；
- 格式版本。

因此不能根据文件前半段推断实际模型文件名。

### `CoPTER`

CoPTER 使用 `TripleHeadCoPTER`，动作网格为 5 个 Kmin、10 个 Kmax、10 个 Pmax。在线探索时随机选择；利用时：

1. 将 Kmin 和 Kmax head 相加为二维 `q_matrix`。
2. 读取该端口的 `f_matrix`。
3. 分别做 min-max 归一化。
4. 计算融合矩阵并选最大位置。
5. Pmax 仍从 Q head 的 argmax 选择。

当前实际语句是：

```text
fusion = 0 * q_norm + 1 * f_norm
```

所以 Kmin/Kmax 完全由 fmap 决定，只有 Pmax 是网络决策。注释写“简化为只基于 Q”，与实际代码不一致；阅读和论文描述必须以执行语句为准。

CoPTER 的 `train_model()` 只在 `online=True` 时更新。其 checkpoint 仍采用前后两个 `.pt` 文件，和 ACC 后定义的单 checkpoint 格式不一致。

### 文件尾测试代码

`if __name__ == '__main__'` 是早期手工测试，不是项目入口。正式入口是 `copter/copter.py`。

## 2.5 [`copter/agent_helper.py`](../../copter/agent_helper.py)

这个文件把“单端口 Agent”扩展为“全拓扑多端口 Agent 系统”。

### 默认超参数

```text
ACC:    state=18, heads=6/4/10, lr=1e-3, gamma=.95
CoPTER: state=18, heads=5/10/10, lr=1e-3, gamma=.95
```

### `ReplayBuffer`

基于 `deque(maxlen=capacity)` 的普通均匀经验池：

- `push()`：追加单条 transition。
- `push_batch()`：追加多条。
- `sample()`：`random.sample` 无放回采样，并转 NumPy 数组。
- 容量满后自动删除最老经验。

### `AgentHelper.__init__()`

根据 `node_number` 创建 Agent 和本地 replay 池。这里的 `node_number` 实际是端口数量，不是网络节点数量，命名略有误导。

ACC 直接创建 agent；CoPTER 初始使用全 1 fmap，并在拿到真实端口标识后延迟加载。还初始化共享 replay、epsilon/global step、TensorBoard 指针和断点续训元数据。

### `get_current_epsilon()`

epsilon 按全局环境 step 线性衰减，并跨 ns-3 episode 从 train state 恢复。它不是按 `train()` 调用次数衰减。

### `decide()`

逐端口调用 `select_action()`。CoPTER 首次遇到端口时通过 `_load_fmap()` 加载：

```text
<fmap_dir>/<switch_id>-<connected_node_id>_fmap.txt
```

加载失败回退为全 1 矩阵。当前代码在端口标识缺失时 `continue`，这会使当次 `paras/actions` 少一个元素，是潜在的端口错位风险。

### `record()`

从 `NetworkHelper` 取得每端口奖励，并把 transition 写入该端口本地 replay。虽然注释提到 CoPTER fmap 奖励，相关参数已被注释，当前 ACC 和 CoPTER 使用同一个网络奖励。

### `sync()`

先从每个本地池抽样上传至共享池，再从共享池为每个端口抽样下发。代码用 `rb_pool[0]` 的长度决定所有端口上传数量；如果端口池长度差异很大，可能对后续端口请求过多样本。

### `train()`

对经验足够的每个端口独立训练，按 `target_update_interval` 硬更新 target。收集 batch mean reward 和 loss，写 TensorBoard，并按间隔保存所有 replay 和训练状态。

### `load()/save()`

恢复或保存：

- 每端口模型；
- 每端口 replay pickle；
- shared replay pickle；
- `*_train_state.json`；
- epsilon、global env/train step、epoch、run_id、phase、config hash。

`load()` 最后执行 `epoch += 1`，所以每次新的 Agent 进程代表一个新 epoch。

### `append_epoch_metrics()`

向 `*_metrics.jsonl` 追加一行，供 TensorBoard、收敛检查、ACC/SOR 对比和遗忘分析读取。

## 2.6 [`copter/copter.py`](../../copter/copter.py)

ACC/CoPTER 的真正 Python 入口。

### 命令行参数组

- 连接：`--ns3_socket`。
- 实验：`--exp_name --mode --model_dir --fmap_dir`。
- 训练：`--online --static_steps --train_intervals --switch_buffer`。
- 探索：`--epsilon_start/end/decay_steps`。
- 恢复/评估：`--resume --checkpoint --eval_greedy --eval_tag`。
- 调试：`--force_action --watch_ports --max_steps`。
- 记录：TensorBoard、seed、run_id、phase。

### 初始化阶段

固定 Python/NumPy/Torch seed，解析固定动作和观察端口，初始化日志、`NetworkHelper`、`AgentHelper`、checkpoint 和 TensorBoard。

### 主循环的三个阶段

1. `step=0`：OpenGym 初始化。
2. `step<max(4,static_steps)`：把环境当前参数原样发回，积累足够 observation 历史。
3. 后续：取所有端口 state、决策、执行、计算奖励、记录 transition、同步和训练。

`--force_action` 会绕过策略，对所有端口强制使用同一个动作索引，是验证“参数是否真正影响 FCT/奖励”的重要工具。

### rollout 指标

主循环记录拥塞端口平均奖励、top 30% 奖励、中位数、拥塞端口数和 watch port EMA。这里的 top 30% 是“奖励最高的 30% 拥塞端口”，能排除结构性卡死端口，但也可能过于乐观，正式报告应同时展示均值和分位数。

### 退出与 `finally`

到达 ns-3 game over 或 `max_steps` 后，主循环退出。在线训练会保存模型、buffer、train state 和 epoch metrics；greedy eval 不应污染训练状态。最后关闭 TensorBoard 和 OpenGym 环境。

## 2.7 [`copter/load_training_config.py`](../../copter/load_training_config.py)

把嵌套 YAML 展平为 shell 可 `eval` 的赋值语句：

```text
convergence.reward_window → CFG_convergence_reward_window
sor.max_clusters          → CFG_sor_max_clusters
```

值通过 `shlex.quote` 转义。`run_train.sh` 和 `simulation/run_train.sh` 依赖它在两个终端读取同一份配置。风险是调用方使用 `eval`；配置文件必须来自可信仓库，不能接收不受信任内容。

## 2.8 [`copter/training_config.yaml`](../../copter/training_config.yaml)

ACC 连续训练默认配置。分为：总体调度、ns-3 路径、Agent、epsilon、持久化、收敛、TensorBoard。

当前默认 `ns3_conf=simulation/mix/copter_Hadoop_Shuffle.conf`，该文件不在当前分支；`switch_buffer=10000` 也与 `m3_256hosts.conf` 的 400KB 不一致。直接使用前应改为实际存在场景，并保持 Buffer 参数一致。

## 2.9 [`copter/run_train.sh`](../../copter/run_train.sh)

Agent 侧多 epoch 启动器。读取 YAML 后循环启动 `python copter.py`，维护跨进程 checkpoint，并可根据 metrics 最近两个窗口的奖励差做早停。

它通常与 [`simulation/run_train.sh`](../../simulation/run_train.sh) 在两个终端配对；更现代的比较脚本会由父进程协调二者，避免手工启动顺序问题。

## 2.10 [`copter/run_100.sh`](../../copter/run_100.sh)

早期重复启动 Agent 的脚本，包含固定次数、固定路径和固定 fmap 目录。注释显示 fmap 原本来自仓库外的 M3/Parsimon 结果目录。它更像历史实验记录，不适合作为新实验入口。

## 2.11 [`copter/scripts/run_acc_sor_ns3_compare.sh`](../../copter/scripts/run_acc_sor_ns3_compare.sh)

ACC 与 SOR 对比调度器的一个副本。它按 epoch：

1. 先启动 Agent 并等待端口监听；
2. 再启动 ns-3；
3. 等待两端退出；
4. 等端口释放；
5. 最后运行对比分析。

比 `sor/run_acc_sor_ns3_compare.sh` 的“先 ns-3、sleep、再 Agent”更稳健。仓库还有 `scripts/run_acc_sor_ns3_compare.sh` 的近似副本，后续应收敛成一个权威版本。

## 2.12 [`copter/fmaps/default.fmap`](../../copter/fmaps/default.fmap)

示例二维参数性能表。第一行通常是 Kmax 网格，第一列是 Kmin 网格，内部数值表示离线性能/偏好。当前内容极小，只适合格式演示，不是 5×10 的正式端口 fmap。

## 2.13 [`copter/fmaps/fmaps.py`](../../copter/fmaps/fmaps.py)

读取带行列坐标的 fmap，进行均值/标准差变换和坐标归一化，再构建 `RegularGridInterpolator(method='cubic')`，返回可查询任意归一化 `(kmin,kmax)` 的函数。

文件尾示例调用了不存在的 `copter_import()`，应为 `import_fmaps()`；因此直接运行会报 `NameError`。主训练路径没有调用这个示例函数，而是由 `AgentHelper._load_fmap()` 用 `np.loadtxt` 直接加载矩阵。

## 2.14 [`copter/delete_model.py`](../../copter/delete_model.py)

按前缀删除模型目录中的文件，用于清理旧实验。文件内带固定路径/前缀示例，执行前必须确认目标目录。它不是训练依赖，且删除操作不可恢复，推荐以后改成显式命令行参数和 `--dry-run`。

## 2.15 [`copter/copter.old`](../../copter/copter.old)

旧版主循环备份，用于比较历史实现。缺少当前分支的 seed、断点恢复、TensorBoard、固定动作、拥塞端口聚合和严格持久化。它不会被入口脚本调用，不应在此继续开发。

## 2.16 根目录 [`copter.py`](../../copter.py)

早期算法草稿，存在未实现的 `pass`/不完整逻辑。文件名与 `copter/copter.py` 相同但位置不同，容易误运行。正式命令应在 `copter/` 目录执行 `python copter.py`，或通过根目录 `run_training.sh` 间接调用。

## 2.17 本模块修改时的检查清单

- 修改 observation 字段时，同时更新 C++ 维度、`PortObservation`、state_dim 和网络输入。
- 修改动作数量时，同时更新动作数组、head 维度、`--force_action` 映射和 SOR。
- 修改 Buffer 时，先统一 `.conf`、YAML、命令行和 Python 归一化单位。
- 修改 reward 时，先运行固定动作 sanity check，再开始长训练。
- 修改 checkpoint 格式时，提供版本迁移，不要只改一个 Agent 类。
- 训练/评估必须使用不同输出目录或确保 `eval_greedy` 不写回训练状态。
