# 6. M3、环境文件、数据文件与完整文件索引

## 6.1 仓库里的三个 “m3”

### Conda 环境名 `m3`

终端提示符 `(m3)` 只说明当前激活了 `m3_environment.yml` 创建的 Python 环境。ACC、SOR、ns3gym 都可以在这个环境中运行，它不表示 M3 模型正在执行。

### 外部研究系统 M3

M3 是面向数据中心 flow-level 性能预测的机器学习系统，目标包括预测 FCT/slowdown，并跨拥塞控制参数、路由和 workload 进行估计。它可以作为离线参数评估或路径级先验的来源。

### 本仓库的 M3 资产

当前仓库包含：

- M3 风格 256-host topology、flow 和参数范围；
- `m3-ml/README` 中的扩展设想；
- CoPTER 对端口级 fmap 的加载逻辑；
- 旧脚本中指向外部 M3/Parsimon fmap 的路径注释。

当前仓库不包含完整 M3 数据预处理、模型结构、训练入口、checkpoint 和推理程序。所以 `m3_256hosts.conf` 的含义是“在 M3 风格场景中运行 ns-3+ACC/CoPTER/SOR”，不是“运行 M3 算法”。

## 6.2 [`m3-ml/README`](../../m3-ml/README)

只有两行有效信息：计划扩展 M3，使其支持混合/多租户 workload，并根据每条路径差异化流量模式预测最佳参数。它是研究方向说明，不是可执行模块。

若要真正补齐 M3，需要新增：

1. 路径级训练样本定义；
2. topology/flow 到 feature 的转换；
3. label：FCT、slowdown 或最佳 Kmin/Kmax/Pmax；
4. train/validation/test 划分；
5. 模型训练和推理；
6. 输出到 CoPTER `fmap_dir` 的文件格式；
7. 与 ACC 在线策略的融合和消融实验。

## 6.3 [`m3_environment.yml`](../../m3_environment.yml)

当前可创建环境版本：Python 3.9、PyTorch 2.0.1、NumPy 1.24.3、SciPy 1.10.1、Gym、PyYAML、Loguru、TensorBoard、Matplotlib/Pandas 等，并固定 Python protobuf 3.20.3。

关键点：

- 文件中没有 `ns3gym==0.1.0`，因为它不在 PyPI。
- 创建环境后还要运行：

  ```bash
  python -m pip install ./ns-3.33/contrib/opengym/model/ns3gym
  ```

- CUDA 11.7 的 PyTorch 相关 wheel 占用空间较大，而代码强制使用 CPU；若不需要 GPU，可改为 CPU-only PyTorch，显著降低安装空间。
- 文件含 2026 年版本的包，重建时依赖镜像是否仍提供这些构建；更稳妥的长期方案是区分“核心运行依赖”和“精确 lock”。

## 6.4 [`m3_environment.yml.original`](../../m3_environment.yml.original)

与当前文件唯一实质区别是包含：

```text
ns3gym==0.1.0
```

Conda 创建环境时 pip 会去包索引查找，因不存在而失败。保留它有助于记录原始错误，但不应再用于创建环境。

## 6.5 [`README.md`](../../README.md)

上游 README 只给出 ZMQ/protobuf 安装和 waf/ns3gym 构建命令，内容不足以运行当前分支。`apt install protobuf-compiler==3.20.3` 也未必匹配 Ubuntu 软件源版本语法。建议以后把本手册入口加入 README，并用 Conda 环境中的 protoc/include 替代移动系统头文件。

## 6.6 [`.gitignore`](../../.gitignore)

忽略日志、模型、flow/conf/output、TensorBoard、实验目录和大多数分析产物，同时通过反向规则保留部分源码/默认 fmap。因为 `.conf/.flow/.pt` 默认忽略，新增必要实验输入时需要 `git add -f` 或调整规则，否则容易出现“本地能跑、服务器 pull 后文件缺失”。

它还忽略整个 `simulation/mix`，但已有跟踪文件仍继续被 Git 管理；只影响后续新增文件。

## 6.7 文件类型与生命周期

| 类型 | 例子 | 应否手工修改 | 是否建议提交 |
|---|---|---:|---:|
| 源码 | `.py/.cc/.h/.sh` | 是 | 是 |
| 模板配置 | `.yaml/.conf/.json` | 是 | 小而必要的应提交 |
| topology/flow | `.topo/.flow` | 生成或审查 | 可复现的小/标准场景提交 |
| checkpoint | `.pt/.pkl` | 否 | 通常不提交；用制品存储 |
| 原始结果 | `.fct/.qlen/.pfc/.throughput` | 否 | 通常归档到实验目录 |
| 图表 | `.png/.pdf` | 否 | 论文最终图可提交，临时图忽略 |
| train state/metrics | `.json/.jsonl` | 否 | 作为 run artifact 保存 |

## 6.8 项目文件完整索引

以下索引覆盖 `ns-3.33` 之外的全部 Git 跟踪文件，并指出详细说明所在章节。标准 ns-3 的 3755 个文件按第 3 章末尾的模块目录说明处理。

### 根目录

| 文件 | 说明 |
|---|---|
| `.gitignore` | 生成物和实验数据忽略规则；见本章 6.6 |
| `README.md` | 极简依赖/构建说明；见 6.5 |
| `build_ns3_copter.sh` | protobuf/ns-3/ns3gym 构建；见第 3 章 |
| `copter.py` | 不完整早期草稿，不是入口；见第 2 章 |
| `m3_environment.yml` | 当前 Conda 环境；见 6.3 |
| `m3_environment.yml.original` | 含无效 PyPI ns3gym 的原始环境；见 6.4 |
| `run_copter_oneclick.sh` | TensorBoard+训练包装；见第 5 章 |
| `run_training.sh` | 单/多实验 episode 调度；见第 5 章 |
| `run_stage0_sensitivity.sh` | 固定动作敏感性 gate；见第 5 章 |
| `run_stage1_acc_effectiveness.sh` | 根目录 A-B-A 调度版本；见第 5 章 |
| `run_full_experiment.sh` | 根目录 stage 总调度；见第 5 章 |
| `status_experiment.sh` | 实验状态查看；见第 5 章 |

### `copter/`

| 文件 | 说明 |
|---|---|
| `agent.py` | ACC/CoPTER 动作、DDQN、checkpoint；第 2 章 2.4 |
| `agent_helper.py` | 多端口 Agent、replay、同步、持久化；2.5 |
| `backbone.py` | ACC/CoPTER 神经网络；2.2 |
| `copter.py` | 正式 ACC/CoPTER 主入口；2.6 |
| `copter.old` | 旧主循环备份；2.15 |
| `delete_model.py` | 按前缀清理旧模型；2.14 |
| `load_training_config.py` | YAML 展平为 shell 变量；2.7 |
| `network_helper.py` | OpenGym、状态、奖励；2.3 |
| `structures.py` | dataclass 参数/状态/动作；2.1 |
| `training_config.yaml` | ACC 连续训练配置；2.8 |
| `run_train.sh` | ACC Agent 多 epoch 启动器；2.9 |
| `run_100.sh` | 旧硬编码重复运行器；2.10 |
| `scripts/run_acc_sor_ns3_compare.sh` | Agent-first 对比调度副本；2.11 |
| `fmaps/default.fmap` | fmap 格式样例；2.12 |
| `fmaps/fmaps.py` | fmap 插值工具；2.13 |

### `m3-ml/`

| 文件 | 说明 |
|---|---|
| `README` | M3 混合 workload 扩展设想；6.2 |

### `simulation/`

| 文件 | 说明 |
|---|---|
| `run-copter-agent.sh` | 一行式早期 Agent 启动；第 3 章 |
| `run-copter-sim.sh` | 单次 ns-3 wrapper；第 3 章 |
| `run_train.sh` | ns-3 多 epoch 启动器；第 3 章 |
| `run-ns3-secn.sh` | 旧 SECN 静态批量脚本；第 3 章 |
| `run-ns3-secn-2.sh` | 旧 DCQCN/HPCC 批量脚本；第 3 章 |
| `run_100.sh` | 实际 20 次的旧循环脚本；第 3 章 |
| `scripts/interim_fct_compare.py` | 中期公共 flow FCT 对比；第 3/5 章 |
| `scripts/sanity_fct_compare.py` | 固定动作公共 flow 对比；第 3/5 章 |
| `mix/m3_256hosts.conf` | 256-host 可用仿真配置；第 3 章 |
| `mix/256hosts.topo` | 308 节点/52 交换机拓扑；第 3 章 |
| `mix/16hosts.topo` | 小型调试拓扑；第 3 章 |
| `mix/256hosts.flow` | 7622 条 flow 输入；第 3 章 |
| `mix/16hosts.flow` | 小型 flow 输入；第 3 章 |
| `mix/default.trace` | trace node 列表；第 3 章 |
| `checkpoint/policy.pt` | 旧 policy 样例；第 3 章 |
| `checkpoint/target.pt` | 旧 target 样例；第 3 章 |
| `output/fct_16hosts_topo_16hosts_flow_dcqcn.txt` | 历史小场景 FCT；第 3 章 |
| `output/fct_256hosts_flow_dcqcn.txt` | 历史 256-host FCT；第 3 章 |
| `output/pfc_16hosts_topo_16hosts_flow_dcqcn.txt` | 历史 PFC 事件；第 3 章 |
| `output/pfc_256hosts_flow_dcqcn.txt` | 空/历史 PFC 输出；第 3 章 |
| `output/qlen_16hosts_topo_16hosts_flow_dcqcn.txt` | 历史队列输出；第 3 章 |
| `output/mix_16hosts_topo_16hosts_flow_dcqcn.tr` | 空/历史二进制 trace；第 3 章 |
| `output/mix_256hosts_flow_dcqcn.tr` | 二进制 trace；第 3 章 |

### `sor/`

| 文件 | 说明 |
|---|---|
| `backbone_sor.py` | SOR encoder+三头网络；第 4 章 |
| `sor_replay.py` | 原型、漂移、结构化 replay；第 4 章 |
| `sor_agent.py` | SORACC 三网络和复合损失；第 4 章 |
| `sor_agent_helper.py` | 多端口 SOR 与全局同步；第 4 章 |
| `sor_copter.py` | SOR 主入口；第 4 章 |
| `test_sor_replay.py` | 8 个 replay 测试；第 4 章 |
| `run_sor_train.sh` | SOR Agent 启动器；第 4 章 |
| `run_acc_sor_ns3_compare.sh` | 旧 ACC/SOR 比较器；第 4 章 |
| `compare_acc_sor.py` | metrics 摘要对比；第 4 章 |
| `compare_forgetting.py` | phase/task 遗忘指标；第 4 章 |
| `SOR_方案原理分析.md` | 原有 SOR 设计文档；第 4 章 |
| `0526/plot_mean_reward.py` | 历史奖励绘图；第 4 章 |
| `sor_training_config.yaml` | SOR 默认训练配置；第 4 章 |
| `acc_eval_config.yaml` | ACC 对比配置；第 4 章 |
| `sor_eval_config.yaml` | SOR 对比配置；第 4 章 |
| `acc_curriculum_config.yaml` | ACC 遗忘配置；第 4 章 |
| `sor_curriculum_config.yaml` | SOR 遗忘配置；第 4 章 |
| `acc_watch_hadoop_config.yaml` | 单场景 watch-port 配置；第 4 章 |
| `eval_sor_train_state.json` | 空状态残留；第 4 章 |
| `sor_models/sor_m3_smoke_train_state.json` | 空 smoke 状态残留；第 4 章 |

### `scripts/`

| 文件 | 说明 |
|---|---|
| `analyze_fct.py` | 轻量 FCT/slowdown 汇总；第 5 章 |
| `analyze_forgetting.py` | ACC curriculum reward 诊断；第 5 章 |
| `reward_sensitivity_check.sh` | 固定动作 reward/FCT 检查；第 5 章 |
| `run_acc_watch_baseline.sh` | ACC 学习曲线和静态基线；第 5 章 |
| `run_forgetting_curriculum.sh` | curriculum 兼容包装；第 5 章 |
| `run_acc_sor_ns3_compare.sh` | 当前较稳健的对比调度；第 5 章 |

### `scripts_exp/`

| 文件 | 说明 |
|---|---|
| `common.sh` | 正式实验共享函数；第 5 章 |
| `static_runner.sh` | 静态策略 runner；第 5 章 |
| `acc_train_runner.sh` | ACC 训练 runner；第 5 章 |
| `acc_eval_runner.sh` | ACC greedy eval runner；第 5 章 |
| `sor_eval_runner.sh` | SOR greedy eval runner；第 5 章 |
| `curriculum_runner.sh` | 多阶段 runner；第 5 章 |
| `run_stage0_sensitivity.sh` | stage0 gate 调度；第 5 章 |
| `run_stage1_acc_effectiveness.sh` | stage1 ACC 有效性；第 5 章 |
| `run_stage2_forgetting.sh` | stage2 遗忘/SOR；第 5 章 |
| `run_full_experiment.sh` | 三阶段总调度；第 5 章 |
| `run_acc_scenA_train.sh` | 场景 A 单独训练；第 5 章 |
| `run_force_action_sanity.sh` | 三组固定动作；第 5 章 |
| `run_sanity_good.sh` | good 动作；第 5 章 |
| `run_sanity_rest.sh` | bad/aggressive 动作；第 5 章 |
| `reanalyze_stage0.sh` | 重分析已有 stage0；第 5 章 |
| `check_stage0_gate.py` | 公共 flow/gate 工具库；第 5 章 |
| `analyze_acc_effectiveness.py` | ACC 有效性判定；第 5 章 |
| `analyze_forgetting.py` | ACC/SOR 遗忘判定；第 5 章 |
| `compare_sanity_fct.py` | 固定动作比较旧版；第 5 章 |
| `diag_obs_stats.py` | observation/reward 分量诊断；第 5 章 |
| `generate_experiment_report.py` | Markdown 报告生成；第 5 章 |
| `status_experiment.sh` | run 状态显示；第 5 章 |

### `tools/traffic/`

| 文件组 | 说明 |
|---|---|
| `TraGen.py` | CDF+load 的多模式 flow 生成器；第 5 章 |
| `TraGen_random.py` | 随机 CDF/去重版本；第 5 章 |
| `scenA_throughput_config.json` | Hadoop 吞吐场景；第 5 章 |
| `scenB_incast_config.json` | AliStorage incast 场景；第 5 章 |
| `.gitignore` | 生成结果忽略；第 5 章 |
| `pattern/default.config` | 混合 workload JSON；第 5 章 |
| `pattern/pattern_1.json` | 空占位；第 5 章 |
| `pattern/README` | 极简 schema 说明；第 5 章 |
| `pattern/AliStorage.txt` | AliStorage flow-size CDF；第 5 章 |
| `pattern/GoogleRPC.txt` | Google RPC CDF；第 5 章 |
| `pattern/Hadoop.txt` | Hadoop CDF；第 5 章 |
| `pattern/VL2_CDF.txt` | VL2 CDF；第 5 章 |
| `pattern/WebSearch.txt` | WebSearch CDF；第 5 章 |
| `pattern/WebServer.txt` | WebServer CDF；第 5 章 |
| `pattern/cachefollower-all.txt` | CacheFollower CDF；第 5 章 |
| `pattern/hadoop-all.txt` | 另一 Hadoop CDF；第 5 章 |
| `pattern/flow_sizes.py` | CDF 绘图；第 5 章 |
| `pattern/flow_size_cdf.pdf` | 生成的 CDF 图；第 5 章 |
| `result/default_flows.json` | 超大历史 flow 元数据；第 5 章 |

### `tools/analysis/`

| 文件 | 说明 |
|---|---|
| `analysis_ecn.py` | Kmin/Kmax FCT 热图；第 5 章 |
| `analysis_fct.py` | 实际 FCT 统计；第 5 章 |
| `analysis_fct_slowdown.py` | slowdown 统计；第 5 章 |
| `analysis_throughput.py` | throughput 原始数据统计/图；第 5 章 |
| `com_fct.py` | 多方法大型 FCT 图组；第 5 章 |
| `fct_time.py` | FCT 随时间；第 5 章 |
| `normalized_fct.py` | summary 归一化图；第 5 章 |
| `queue_time.py` | 通用队列时间序列；第 5 章 |
| `queue_time_incast.py` | incast/热点队列；第 5 章 |
| `rate_time.py` | tx/ECN rate 时间序列；第 5 章 |
| `run_curriculum_fct_compare.py` | curriculum FCT 聚合；第 5 章 |
| `run_sor_acc_analysis.py` | ACC/SOR 综合图组；第 5 章 |
| `thesis_com_fct.py` | 论文排版 FCT 图；第 5 章 |
| `throughput_plots.py` | 手填 summary 吞吐柱图；第 5 章 |

## 6.9 从文档回到代码的实践顺序

建议边读边完成以下小练习：

1. 在 `m3_256hosts.conf` 中标出 topology、flow、Buffer、输出和三个静态 ECN map。
2. 手算一个 25G 端口归一化 Kmin=.5 对应的实际 bytes。
3. 在 `network_helper.py` 手算 queue=.1、tx=.8、ECN=.03 的 reward。
4. 在 `agent.py` 追踪一个 batch 的 tensor shape。
5. 用 `--force_action` 对两组动作跑一次 smoke test，检查 C++ 日志中的 ConfigEcn 是否不同。
6. 用公共完成 flow 比较 FCT，而不是直接比较两个文件的独立均值。
7. 运行 `sor/test_sor_replay.py`，再新增一个跨端口 sync 测试。

完成这些练习后，再开始修改奖励、动作范围或网络深度，会比直接进行长训练更可靠。
