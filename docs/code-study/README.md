# CoPTER 代码学习手册

> 对应仓库：`manykun/CoPTER`
> 对应分支：`exp/acc-validation`
> 对应提交：`29ba79f8a5ebe94b1e0845594626b6cf47b5ee90`

这套文档面向准备运行、修改和设计 CoPTER 实验的读者。它不仅说明“文件是做什么的”，还说明文件之间怎样调用、数据怎样流动、ACC/M3/CoPTER/SOR 分别处于系统的哪一层，以及当前实现中需要优先验证的风险点。

## 文档导航

1. [系统架构、运行链路与核心概念](01-architecture-and-dataflow.md)
2. [ACC 与 CoPTER：`copter/` 逐文件讲解](02-acc-and-copter-files.md)
3. [ns-3、OpenGym 与 `simulation/` 逐文件讲解](03-ns3-and-simulation-files.md)
4. [SOR：持续学习与抗遗忘代码逐文件讲解](04-sor-files.md)
5. [实验脚本、流量生成与分析工具逐文件讲解](05-experiments-traffic-analysis-files.md)
6. [M3、环境文件、数据文件与完整文件索引](06-m3-environment-and-file-index.md)

## 覆盖范围

仓库当前有 3887 个 Git 跟踪文件，其中 3755 个位于 `ns-3.33/`。绝大多数是未针对本项目单独编写的 ns-3.33 标准模块、示例、测试和文档。逐一解释这几千个第三方文件既不能帮助理解 CoPTER，也会掩盖真正的控制链路。

本手册采用以下范围：

- 逐文件覆盖仓库根目录、`copter/`、`sor/`、`simulation/`、`scripts/`、`scripts_exp/`、`tools/`、`m3-ml/` 中的项目文件。
- 对模型权重、结果文件、CDF 数据和大规模 flow/topology 文件解释其格式、生产者和消费者，不逐行复述数据。
- 对 `ns-3.33/` 详细解释 CoPTER 直接调用或修改的仿真入口、OpenGym、交换机 MMU、RDMA、QBB/PFC/ECN 文件。
- 其余标准 ns-3 文件按目录说明职责，并给出何时需要深入阅读。

## 建议学习路线

第一次阅读建议按下列顺序，而不是从 1511 行的 C++ 仿真入口直接开始：

1. 阅读第 1 章，理解一次 `env.step()` 的完整闭环。
2. 阅读 [`copter/structures.py`](../../copter/structures.py)，记住状态、动作和超参数的数据结构。
3. 阅读 [`copter/network_helper.py`](../../copter/network_helper.py)，理解观测和奖励。
4. 阅读 [`copter/backbone.py`](../../copter/backbone.py) 与 [`copter/agent.py`](../../copter/agent.py)，理解 ACC 的 DDQN。
5. 阅读 [`copter/agent_helper.py`](../../copter/agent_helper.py)，理解“每端口一个 Agent”和共享经验池。
6. 阅读 [`copter/copter.py`](../../copter/copter.py)，把前面部分串成训练主循环。
7. 再进入 [`ns-3.33/scratch/copter-sim.cc`](../../ns-3.33/scratch/copter-sim.cc)，理解 C++ 如何产生观测、接收动作。
8. 最后阅读 `sor/`，对比普通 replay 与结构化 replay。

## 四个名称的准确含义

| 名称 | 在本仓库中的准确定位 |
|---|---|
| ACC | 基于三头 Double DQN 的在线 DCQCN 参数控制器 |
| CoPTER | 更大的三头网络，加上路径/端口级 `fmap` 引导 |
| M3 | 外部流级性能预测系统；仓库只保留扩展设想和 M3 风格实验资产，没有完整 M3 实现 |
| SOR | 在 ACC 上加入聚类、漂移检测、边界记忆和正则化的持续学习版本 |

## 阅读代码时始终记住的两个“端”

- **C++ 仿真端**：真实执行队列、ECN/PFC、DCQCN、数据包和流；把观测发给 Python，并落实 Python 动作。
- **Python 智能体端**：把连续观测组成状态，选择 `Kmin/Kmax/Pmax`，计算奖励，保存经验并训练网络。

它们通过 `ns3-gym/OpenGym` 的 ZeroMQ/Protocol Buffers 消息相连。任何实验异常都应先判断发生在“物理仿真端”“接口与归一化”“学习算法”还是“实验调度”层。
