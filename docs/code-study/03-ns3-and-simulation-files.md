# 3. ns-3、OpenGym 与 `simulation/` 逐文件讲解

## 3.1 [`ns-3.33/scratch/copter-sim.cc`](../../ns-3.33/scratch/copter-sim.cc)

这是整个仿真的 C++ 入口，也是 Python Agent 的环境实现。文件可分为六部分阅读。

### 全局配置与 OpenGym 参数

文件顶部定义拥塞控制、流量、输出、Buffer、监控和 OpenGym 默认值。关键默认值：

```text
OpenGym port = 5555
environment step = 0.0005 s
monitor interval = 0.0001 s
state/port = 6
action/port = 3
Kmin range@25G = 20KB～50KB
Kmax range@25G = 50KB～100KB
```

`.conf` 可以用 `OPENGYM_MIN/MAX_KMIN/KMAX` 覆盖范围。

### OpenGym 回调

#### `MyGetObservationSpace()`

返回 `[n_ports*6]` 的连续 Box，声明范围 `[0,1]`。实际 tx rate 统计偶尔可能超过 1，Python reward 侧会 clamp，但 observation space 声明与实际值可能短暂不一致。

#### `MyGetActionSpace()`

返回 `[n_ports*3]` 连续 Box。Python 的 Agent 实际只在离散候选值中选择，然后以连续值发送。

#### `MyGetGameOver()`

用 `opengym_end_time` 判断结束。该时间在读 flow 时更新为最早/当前逻辑决定的 flow 边界，和 `SIMULATOR_STOP_TIME` 不是同一个概念。函数内存在未使用的 `flag` 和注释掉的替代逻辑，建议后续简化并明确 episode 终止定义。

#### `MyGetObservation()`

遍历每个 SwitchNode 的端口 1..N-1；端口 0 通常不作为数据端口。每端口执行：

1. 找到链路对端 node ID，生成 `switch-connectedNode` 标识。
2. 从 MMU 读取窗口峰值队列和当前队列，取二者最大值。
3. 用交换机总 Buffer 字节数归一化队列。
4. 重置端口峰值。
5. 用端口线速归一化 tx bytes rate 和 ECN-marked bytes rate。
6. 重置 rate 统计窗口。
7. 读取当前 Kmin/Kmax/Pmax，并按该端口速率范围归一化。

注意队列分母是交换机总 Buffer，不是某端口独占 Buffer；因此“queue occupancy”更准确地说是该端口队列相对总共享 Buffer 的比例。

#### `MyGetReward()`

固定返回 0。理由是一个 observation 包含所有端口，而每端口应有独立 reward，所以奖励移到 Python 计算。

#### `MyGetExtraInfo()`

把端口标识列表编码为 JSON。变量名声称只返回一次，但当前条件被注释，函数每次都可返回列表。Python 主要在初始化时解析。

#### `MyExecuteActions()`

按端口顺序每 3 个数读取动作，依据端口速率缩放 Kmin/Kmax，再调用：

```text
SwitchMmu::ConfigEcn(port, kminKB, kmaxKB, pmax)
```

动作扁平化顺序必须与 observation 遍历顺序完全一致。

### 网络监控

`ScheduleNetworkMonitor()` 以 100 微秒周期记录队列、发送速率和吞吐率。它与 OpenGym observation 的 500 微秒周期不同，主要用于离线分析，不直接作为 state。

### 拓扑、路由和流

- `ReadFlowInput()/ScheduleFlowInputs()`：逐条读取 flow 并创建 RDMA 应用。
- `CalculateRoute()/CalculateRoutes()/SetRoutingEntries()`：计算可用路径和 ECMP next hops。
- `qp_finish()`：流完成时写 FCT。
- `get_pfc()`：记录 pause/resume 事件。
- `TakeDownLink()`：可选链路故障。

### `main()` 配置读取

程序用第一位置参数作为 `.conf`，同时通过 `--port` 读取 OpenGym socket。配置解析器逐个 key 读取，未知或缺字段不会像成熟配置库那样提供强校验，因此拼写错误可能静默导致默认值。

### `main()` 构建与运行

1. 读取 topology/flow/trace。
2. 创建 server Node 和 SwitchNode。
3. 建 QBB links、IP 和 PFC trace。
4. 为每个交换机端口配置静态 ECN、headroom 和 Buffer。
5. 在 host 安装 RdmaHw/RdmaDriver。
6. 计算路由、RTT、BDP。
7. 调度 flows 和监控。
8. 若 `ENABLE_COPTER=1`，注册 OpenGym 回调并从约 2.001 秒开始交互。
9. `Simulator::Run()` 到 stop time。

`BUFFER_SIZE` 使用 `buffer_size*1024`，所以 `.conf` 单位为 KB。注释显示旧实现曾乘 `1024*1024`，当前改法是为了和 M3 实验口径一致。

## 3.2 交换机与 MMU

### [`switch-node.h`](../../ns-3.33/src/point-to-point/model/switch-node.h) / [`switch-node.cc`](../../ns-3.33/src/point-to-point/model/switch-node.cc)

`SwitchNode` 继承 ns-3 `Node`，负责：

- 基于 ECMP hash 选择输出端口；
- 入队/出队和 PFC pause/resume；
- 在出队时调用 MMU 判断是否 ECN 标记；
- HPCC/INT/PINT 相关交换机遥测；
- 为 OpenGym 累积端口总发送 bytes 和 ECN bytes。

`UpdatePortRate/UpdatePortEcnRate` 更新窗口统计；`GetPortRate/GetPortEcnRate` 用 bytes 与时间差计算 bit/s；`ResetRateStats` 开启下一观察窗口。

### [`switch-mmu.h`](../../ns-3.33/src/point-to-point/model/switch-mmu.h) / [`switch-mmu.cc`](../../ns-3.33/src/point-to-point/model/switch-mmu.cc)

`SwitchMmu` 是 ACC 动作最终生效的位置。它维护：

- 每端口/队列 ingress、egress、headroom bytes；
- shared Buffer 使用量；
- PFC pause 状态和阈值；
- 每端口 `kmin/kmax/pmax`；
- 每端口观察窗口峰值队列。

`ShouldSendCN()` 实现 ECN 曲线：低于 Kmin 不标记，高于 Kmax 按最大概率/确定逻辑标记，中间按队列位置线性增加概率。`ConfigEcn()` 把 KB 参数乘 1000 存成 bytes。

`UpdateEgressAdmission()` 在入队时更新峰值；`ResetPeakBytes/GetPeakBytes` 为 OpenGym 提供 burst-sensitive queue observation。

## 3.3 RDMA 与拥塞控制文件

### [`rdma-hw.h`](../../ns-3.33/src/point-to-point/model/rdma-hw.h) / [`rdma-hw.cc`](../../ns-3.33/src/point-to-point/model/rdma-hw.cc)

模拟 host NIC/RDMA 拥塞控制。负责 Queue Pair 创建、收发 ACK/NACK/CNP、重传、速率调整和完成通知。文件同时实现多种 `CC_MODE`：DCQCN、HPCC、TIMELY、DCTCP、HPCC-PINT 等。

ACC 实验的关键路径是：交换机根据动态 Kmin/Kmax/Pmax 标 ECN，接收端反馈 CNP/ACK，发送端 `RdmaHw` 的 DCQCN 逻辑更新 alpha 和发送速率。ACC 不是直接改发送速率，而是改变交换机产生拥塞反馈的方式。

### [`rdma-driver.h/.cc`](../../ns-3.33/src/point-to-point/model/rdma-driver.h)

把 `RdmaHw` 聚合到 host Node，对外提供创建 Queue Pair 和完成回调，是应用层与硬件模型的桥梁。

### [`rdma-queue-pair.h/.cc`](../../ns-3.33/src/point-to-point/model/rdma-queue-pair.h)

保存单条 RDMA flow 的发送序号、窗口、速率、RTT、DCQCN alpha/阶段以及不同 CC 算法的状态。FCT 完成条件最终依赖 QP 的发送与确认进度。

### [`rdma-client.h/.cc`](../../ns-3.33/src/applications/model/rdma-client.h)

应用模型：在指定时间发起一条指定大小的 RDMA flow，并在 QP 完成时通知仿真。对应 helper 文件负责从 `copter-sim.cc` 方便地安装应用属性。

## 3.4 QBB、PFC 与队列

### [`qbb-net-device.h/.cc`](../../ns-3.33/src/point-to-point/model/qbb-net-device.h)

实现支持 Priority Flow Control 的点到点设备：多优先级队列、pause/resume、RDMA Queue Pair 调度、发送完成和链路 down。交换机和 host 都通过它连接 QBB channel。

### [`qbb-channel.h/.cc`](../../ns-3.33/src/point-to-point/model/qbb-channel.h)

模拟链路传播延迟和两端设备连接。

### [`qbb-header.h/.cc`](../../ns-3.33/src/point-to-point/model/qbb-header.h)

定义 QBB/PFC/ACK/CNP 等控制头部字段和序列化。

### [`qbb-helper.h/.cc`](../../ns-3.33/src/point-to-point/helper/qbb-helper.h)

在两个 Node 之间批量创建 QbbNetDevice/Channel、设置速率和延迟，并支持 trace。

### [`broadcom-egress-queue.h/.cc`](../../ns-3.33/src/network/utils/broadcom-egress-queue.h)

交换机多队列实现，跟踪各优先级和总 bytes，按轮询策略选择非暂停队列。其队列字节统计是 MMU/监控的重要基础。

### [`int-header.h/.cc`](../../ns-3.33/src/network/utils/int-header.h)

实现 HPCC 所需的 in-band telemetry：hop 时间、发送 bytes、队列、线速或 PINT 编码。DCQCN 模式不依赖完整 INT，但同一仿真器通过 `CC_MODE` 共用该文件。

## 3.5 ns3-gym/OpenGym 文件

### [`contrib/opengym/model/messages.proto`](../../ns-3.33/contrib/opengym/model/messages.proto)

定义 C++ 与 Python 交换的 Protocol Buffers 消息：空间描述、状态、动作、reward、done 和 info。修改协议后必须重新生成并重编译两端。

### `opengym_interface.h/.cc`

OpenGym C++ 主接口。保存回调，使用 ZeroMQ 与 Python 交互，在 `NotifyCurrentState()` 时收集 observation/reward/done/info，发送后阻塞等待 action，再调用 `MyExecuteActions()`。

### `opengym_env.h/.cc`

定义环境抽象和 callback 类型，使 scratch 程序可以提供 observation/action/reward 等实现。

### `spaces.h/.cc` 与 `container.h/.cc`

把 Box/Discrete/Tuple/Dict 空间及其数据容器序列化到 protobuf。CoPTER 只使用一维 Box。

### [`ns3gym/ns3env.py`](../../ns-3.33/contrib/opengym/model/ns3gym/ns3gym/ns3env.py)

Python Gym wrapper。`reset()` 和 `step(action)` 把 ZeroMQ 消息转换为 Gym 风格 `(obs,reward,done,info)`。本项目设置 `startSim=False`，所以它不负责启动 waf/ns-3，只连接外部进程。

### `ns3gym/start_sim.py`

ns3-gym 自带的仿真启动辅助。本项目当前统一由 shell 调度 ns-3，通常不经过它。

### `ns3gym/setup.py/requirements.txt/MANIFEST.in`

Python 包安装元数据。`ns3gym==0.1.0` 不在 PyPI，必须从该本地目录安装，这也是原始 environment YAML 创建失败的原因。

### `contrib/opengym/examples/`、`test/`、`doc/`

上游 ns3-gym 示例、测试和文档，不参与 CoPTER 运行。学习 ZeroMQ/Gym 回调模式时可参考，修改 CoPTER 算法无需逐个阅读。

## 3.6 构建文件

### [`build_ns3_copter.sh`](../../build_ns3_copter.sh)

进入 `ns-3.33`，临时移动系统 protobuf include，执行 waf configure/build，再安装本地 ns3gym。它要求 sudo 并假定 `/usr/local/include/google/protobuf` 存在；任何中途失败都可能来不及恢复目录。正式环境建议使用 trap、Conda 内 protoc/include 和无 sudo 构建。

### [`ns-3.33/copter_safe_build.sh`](../../ns-3.33/copter_safe_build.sh)

位于 ns-3 目录的另一份安全构建尝试。用途与根脚本相近，需检查相对路径和 protobuf 版本后再选择一个作为权威入口。

### `ns-3.33/waf`、`wscript`、`Makefile`

ns-3.33 上游构建系统。`wscript` 决定模块、scratch 程序和 contrib/opengym 是否参与编译。正常实验只运行 waf，不需要修改这些文件。

## 3.7 [`simulation/mix/m3_256hosts.conf`](../../simulation/mix/m3_256hosts.conf)

当前仓库唯一完整存在、可作为 smoke test 的 CoPTER 配置。关键内容：

```text
ENABLE_COPTER=1
ENABLE_QCN=1
USE_DYNAMIC_PFC_THRESHOLD=1
TOPOLOGY_FILE=mix/256hosts.topo
FLOW_FILE=mix/256hosts.flow
SIMULATOR_STOP_TIME=4.0
CC_MODE=1
BUFFER_SIZE=400 KB
```

`KMIN/KMAX/PMAX_MAP` 给每种链路速率设置仿真启动时的静态参数；Python Agent 动作生效后会动态覆盖端口参数。

输出文件名中部分仍含 `16hosts`，属于复制配置后的命名残留，不代表实际拓扑。

## 3.8 topology、flow 与 trace 文件

### [`256hosts.topo`](../../simulation/mix/256hosts.topo)

首行：

```text
node_count switch_count link_count
```

第二行列出所有 switch node ID；后续每行：

```text
src_node dst_node data_rate delay error_rate
```

该文件实际首行为 `308 52 352`：256 台 host 加 52 个交换机节点。

### [`16hosts.topo`](../../simulation/mix/16hosts.topo)

小规模同格式拓扑，适合调试，但当前没有与它配套的完整 `.conf`。

### [`256hosts.flow`](../../simulation/mix/256hosts.flow)

首行是 flow 数量，后续每行：

```text
src dst priority_group destination_port size_bytes start_time_seconds
```

当前有 7622 条 flow，主要从约 2 秒开始，以配合 OpenGym 从 2 秒附近启动。

### [`16hosts.flow`](../../simulation/mix/16hosts.flow)

小拓扑 flow 列表，同样格式。

### [`default.trace`](../../simulation/mix/default.trace)

首行是要 trace 的节点数量，后续是 node ID。`ENABLE_TRACE=0` 时不会产生完整 packet trace，但文件仍需能被配置打开。

## 3.9 simulation 启动脚本

### [`simulation/run-copter-sim.sh`](../../simulation/run-copter-sim.sh)

单次 ns-3 包装器：检查 `.conf`，设置 `LD_LIBRARY_PATH/NS_LOG`，执行 `build/scratch/copter-sim conf --port ...`。必须从 `simulation/` 工作目录运行，因为 `.conf` 内 topology/flow 是相对路径。

### [`simulation/run-copter-agent.sh`](../../simulation/run-copter-agent.sh)

只有一行 `python ../copter/copter.py`，不传配置，属于早期最小启动器，不适合正式实验。

### [`simulation/run_train.sh`](../../simulation/run_train.sh)

ns-3 侧连续训练启动器，与 Agent YAML 共用配置。它将相对路径基于仓库根目录展开，但切回 `simulation/` 运行 binary，保证 `.conf` 内相对文件正确。每 epoch 启动一次全新 ns-3。

### [`simulation/run-ns3-secn.sh`](../../simulation/run-ns3-secn.sh)

批量运行指定 SECN/DCQCN 静态配置并保存输出。当前引用的 `acc_mixed_1_DCQCN_SECN.conf` 不在分支，属于历史脚本。

### [`simulation/run-ns3-secn-2.sh`](../../simulation/run-ns3-secn-2.sh)

同时运行 DCQCN 与 HPCC 的长场景配置；引用文件同样缺失。

### [`simulation/run_100.sh`](../../simulation/run_100.sh)

实际循环 20 次，但提示文本写 100 次；硬编码 Hadoop 配置和服务器路径。只适合追溯旧实验。

## 3.10 simulation 输出文件

### `simulation/output/fct_*.txt`

每行是一条完成 flow。常见字段包含源/目的 IP、端口、PG、flow size、开始时间、完成时间/耗时和理想完成时间。不同分析脚本通常用实际 FCT 除以理想 FCT 得 slowdown。

### `simulation/output/pfc_*.txt`

每行记录时间、node/device、queue 和 pause/resume 类型，用于统计 PFC 次数和持续行为。

### `simulation/output/qlen_*.txt`

按时间块记录交换机端口/队列 bytes。它是旧 `monitor_buffer()` 输出，与 `ENABLE_MONITOR` 的端口级 qlen monitor 格式可能不同，分析前要确认生产函数。

### `simulation/output/mix_*.tr`

ns-3 自定义二进制 trace，不是文本文件；终端看到乱码是正常现象，应使用仓库 trace 解析工具或禁用 trace，不能用 `cat` 判断损坏。

这些 output 是样例/历史结果，不应作为新实验输入。正式实验脚本应把每次输出归档到独立 run directory，避免后一个 epoch 覆盖前一个。

## 3.11 checkpoint 文件

### [`simulation/checkpoint/policy.pt`](../../simulation/checkpoint/policy.pt)

### [`simulation/checkpoint/target.pt`](../../simulation/checkpoint/target.pt)

旧版 PyTorch 权重样例。当前 ACC 使用每端口且带实验名前缀/单文件 checkpoint 的新格式，这两个文件不会被 `AgentHelper` 自动读取。没有配套模型结构、commit 和训练配置时，不应把它们当可信预训练模型。

## 3.12 simulation 分析脚本

### [`simulation/scripts/interim_fct_compare.py`](../../simulation/scripts/interim_fct_compare.py)

在长训练尚未结束时读取已有 FCT，按指定 flow key 比较统计量，帮助快速发现某一方法明显异常。

### [`simulation/scripts/sanity_fct_compare.py`](../../simulation/scripts/sanity_fct_compare.py)

面向固定好/坏动作 sanity run，计算平均值和分位数，验证动作改变是否在物理 FCT 上产生可观察差异。

## 3.13 `ns-3.33/` 其余目录如何看

| 目录 | 内容 | 本项目何时需要深入 |
|---|---|---|
| `src/core` | 事件、时间、对象、属性 | 调试 Simulator 调度/生命周期 |
| `src/network` | Packet、Queue、NetDevice | 修改队列或 packet header |
| `src/internet` | IPv4、路由、传输头 | 修改路由或协议解析 |
| `src/point-to-point` | 链路设备；本项目 RDMA/QBB 核心在此 | 修改 ECN/PFC/RDMA 时 |
| `src/applications` | 应用/流模型 | 修改 flow 发起和完成逻辑 |
| `bindings/python` | ns-3 官方 Python bindings | CoPTER 通过 ns3-gym，通常无需改 |
| `examples` | 上游示例 | 学习 ns-3 API 时 |
| `test`/各模块 `test` | 上游单元测试 | 修改底层模块后做回归 |
| `doc` | ns-3 手册与模型文档 | 查 API 和模型语义 |
| `utils`/`waf-tools` | 构建和开发工具 | 处理编译系统问题 |
