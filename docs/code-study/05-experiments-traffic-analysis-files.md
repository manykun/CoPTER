# 5. 实验脚本、流量生成与分析工具逐文件讲解

## 5.1 根目录运行与实验脚本

### [`run_training.sh`](../../run_training.sh)

当前最直观的 ACC/CoPTER 单场景入口。它激活 `m3` 环境，校验 config/binary，按 episode 启动 ns-3，再启动 Agent，等待两端退出并从 `*_train_state.json` 的 `epoch` 恢复进度。

支持单实验和 `config:exp` 列表并行，自动分配 OpenGym 端口。默认 config 指向当前分支不存在的 CacheFollower 文件；默认 Buffer=10000、epsilon decay=100 也不适合 `m3_256hosts.conf`，必须显式覆盖。脚本的 state 文件名和 epoch 字段已与当前 AgentHelper 对齐，这是提交 `29ba79f` 的服务器修复之一。

### [`run_copter_oneclick.sh`](../../run_copter_oneclick.sh)

在 `run_training.sh` 外再包一层 TensorBoard：检测目标端口是否已有 TensorBoard，必要时后台启动，再透传其余训练参数。服务器上需要通过 SSH port forwarding 访问 `127.0.0.1:6006`，开放 `0.0.0.0` 前应考虑组内网络权限。

### [`run_stage0_sensitivity.sh`](../../run_stage0_sensitivity.sh)

正式实验的“动作敏感性门槛”：对同一场景运行至少两组固定动作，归档 FCT，用 `scripts/analyze_fct.py` 比较。如果不同动作的相对差异低于阈值则失败，不继续宣称 ACC 可学习。

脚本硬编码旧服务器根路径和缺失 `acc_<TASK>.conf`，思路正确但需路径参数化。

### [`run_stage1_acc_effectiveness.sh`](../../run_stage1_acc_effectiveness.sh)

当前内容实际执行 ACC/SOR 的 A-B-A 三阶段 curriculum，而文件名仍叫 stage1 effectiveness。每阶段训练当前 task，并在已见任务上 greedy eval，归档 phase 输出。命名与功能不一致，且依赖缺失场景。

### [`run_full_experiment.sh`](../../run_full_experiment.sh)

根目录总调度器：创建 run directory、文件锁和原子 status，依次运行 stage0、stage1；stage0 未通过就阻止后续。支持 `--resume/--smoke/--seed`。它与 `scripts_exp/run_full_experiment.sh` 是不同版本，根版本目前只跑两个阶段。

### [`status_experiment.sh`](../../status_experiment.sh)

查询实验目录的 status、done 标志、进程和最近日志，用于长实验巡检。它依赖既定 run directory 结构，不能用于 `run_training.sh` 的简单日志目录。

### [`build_ns3_copter.sh`](../../build_ns3_copter.sh)

构建和安装脚本已在第 3 章说明。它是环境准备，不是实验算法。

## 5.2 `scripts/`：当前实验辅助脚本

### [`scripts/analyze_fct.py`](../../scripts/analyze_fct.py)

轻量、可复用的 FCT 比较器。输入 `label=path`，输出 flow 数、平均/P50/P95/P99 FCT、平均/P99 slowdown、小流和大流指标，可写 JSON。它逐文件独立统计，不自动取多个 run 的公共完成 flow；固定动作公平比较时应使用公共 flow 分析器。

### [`scripts/analyze_forgetting.py`](../../scripts/analyze_forgetting.py)

读取 ACC curriculum metrics JSONL，按 `eval_tag` 汇总阶段内学习和阶段间 greedy eval，打印 reward 矩阵与下降量。它主要诊断 ACC 是否遗忘，没有同时读取 FCT，也没有自动对照 SOR。

### [`scripts/reward_sensitivity_check.sh`](../../scripts/reward_sensitivity_check.sh)

对多组 `--force_action` 启动完整 ns-3+Agent，保存日志/FCT，再比较 reward 与 FCT。包含端口释放、子进程清理和 preflight，是验证奖励函数的实用入口。仍有硬编码配置/路径时，应先改为当前仓库根目录和 `m3_256hosts.conf`。

### [`scripts/run_acc_watch_baseline.sh`](../../scripts/run_acc_watch_baseline.sh)

单场景训练 ACC，并追踪固定拥塞端口；训练完成后运行 greedy policy 和 early/mid/late 三组固定动作。目的是回答“RL 是否真的提高 reward，且是否超过固定基线”。使用 `acc_watch_hadoop_config.yaml`，依赖缺失 Hadoop 场景。

### [`scripts/run_forgetting_curriculum.sh`](../../scripts/run_forgetting_curriculum.sh)

只有少量包装代码，把实际工作交给其他 curriculum 调度器，属于兼容入口。

### [`scripts/run_acc_sor_ns3_compare.sh`](../../scripts/run_acc_sor_ns3_compare.sh)

较完整的 ACC/SOR 对比调度器：检查环境和配置，先启动 Agent、等待 socket 监听，再启动 ns-3；逐 epoch 等待退出与端口释放，最后调用比较脚本。当前仍硬编码服务器/Conda 路径，但并发控制比旧版可靠。

## 5.3 `scripts_exp/common.sh`

[`scripts_exp/common.sh`](../../scripts_exp/common.sh) 是正式实验框架的共享函数库：

- `atomic_write/set_status/mark_done`：原子记录状态。
- `acquire_lock`：防止同一个 run 被重复启动。
- `cleanup_children`：退出时清理子进程。
- `require_runtime`：检查 binary、Conda、配置和输出目录。
- `wait_for_port/wait_port_free`：协调 OpenGym socket。
- `parse_common_args`：统一 `--run-dir/--seed/--resume/--smoke`。
- `write_manifest/render_config`：保存 Git SHA、参数并从模板生成本次 YAML。
- `fct_path_from_conf`：从 `.conf` 找输出路径。
- `run_ns3_agent_once`：执行一次 ns-3+Agent。
- `run_train_or_eval`：按 method/task/mode 组织训练或 greedy eval。

这是最值得保留和扩展的实验基础设施，但顶部默认 `ROOT` 和场景命名仍要迁移到用户服务器目录。

## 5.4 `scripts_exp/` runner 文件

### [`static_runner.sh`](../../scripts_exp/static_runner.sh)

给定 run directory/场景/静态动作，调用公共函数执行静态或固定动作基线并归档输出。

### [`acc_train_runner.sh`](../../scripts_exp/acc_train_runner.sh)

为独立 run directory 渲染 ACC 配置，循环指定 epochs 训练，支持 resume。它是 stage 脚本与底层 Agent/ns-3 的适配器。

### [`acc_eval_runner.sh`](../../scripts_exp/acc_eval_runner.sh)

加载已训练 ACC checkpoint，执行 greedy eval，不记录/训练 replay，并把 FCT/metrics 归档到指定步骤目录。

### [`sor_eval_runner.sh`](../../scripts_exp/sor_eval_runner.sh)

SOR 版 greedy eval runner，使用 SOR 配置和 checkpoint。应确保 eval 不覆盖训练 global replay。

### [`curriculum_runner.sh`](../../scripts_exp/curriculum_runner.sh)

按阶段任务列表调用训练/评估 runner，负责 curriculum 的顺序与 resume 状态。

## 5.5 `scripts_exp/` stage 文件

### [`run_stage0_sensitivity.sh`](../../scripts_exp/run_stage0_sensitivity.sh)

调用多组静态/固定参数实验，再执行 `check_stage0_gate.py`。相对根版本，它更明确地使用 run directory 和结构化输出。

### [`run_stage1_acc_effectiveness.sh`](../../scripts_exp/run_stage1_acc_effectiveness.sh)

训练 ACC，在 greedy eval 中与多个静态策略比较，调用 `analyze_acc_effectiveness.py` 判定 reward 是否提升以及 FCT 是否接近最佳静态参数。

### [`run_stage2_forgetting.sh`](../../scripts_exp/run_stage2_forgetting.sh)

运行 ACC 与 SOR 的非平稳 curriculum，保存 before/after metrics 和 FCT，再调用 `analyze_forgetting.py` 判断 ACC 是否发生遗忘且 SOR 是否更轻。

### [`run_full_experiment.sh`](../../scripts_exp/run_full_experiment.sh)

按 stage0→stage1→stage2 顺序运行，并在每个 gate 失败时停止，最后生成 Markdown report。它是比根目录同名脚本更完整的三阶段版本。

### [`run_acc_scenA_train.sh`](../../scripts_exp/run_acc_scenA_train.sh)

只训练场景 A 的 ACC，用于先建立可学习 baseline。脚本包含服务器路径和特定配置命名，需配合生成的 scenA `.conf`。

## 5.6 `scripts_exp/` sanity 文件

### [`run_force_action_sanity.sh`](../../scripts_exp/run_force_action_sanity.sh)

依次运行 `good=(2,1,1)`、`bad=(5,3,0)`、`aggr=(0,0,9)`，归档 Agent 日志和 FCT。动作标签只是人工假设，必须由 FCT 验证，不能先验认定 good。

### [`run_sanity_good.sh`](../../scripts_exp/run_sanity_good.sh)

只运行 good 动作，适合先确认管线可用。

### [`run_sanity_rest.sh`](../../scripts_exp/run_sanity_rest.sh)

运行 bad/aggressive 两组，通常接在 good 完成后。

### [`compare_sanity_fct.py`](../../scripts_exp/compare_sanity_fct.py)

按六字段 flow key 取三组共同完成流，比较 FCT/slowdown。路径写死为旧服务器，功能上与 `simulation/scripts/sanity_fct_compare.py` 重复。

### [`diag_obs_stats.py`](../../scripts_exp/diag_obs_stats.py)

从 Agent 日志正则提取 queue/tx/ECN，统计拥塞子集并近似重算 reward 分量，用于判断是哪一项缺乏区分度。其公式权重仍写 `.45/.30/.25`，而当前 `NetworkHelper` 是 `.30/.50/.20`，结果已过时，需同步后再用。

### [`reanalyze_stage0.sh`](../../scripts_exp/reanalyze_stage0.sh)

不重跑仿真，只对已有 stage0 输出用新阈值/脚本重新分析，适合调整 gate；不能修复原始数据缺失或场景不一致。

## 5.7 `scripts_exp/` 分析文件

### [`check_stage0_gate.py`](../../scripts_exp/check_stage0_gate.py)

通用输入校验和 stage0 判定库。能读 JSON/JSONL，校验 flow 声明数量，按共同 flow 比较 FCT，输出 JSON/CSV/可选图。其他两个分析器复用它的 `field/number/load_fct_files/write_outputs`。

### [`analyze_acc_effectiveness.py`](../../scripts_exp/analyze_acc_effectiveness.py)

通过三个条件判定 ACC 有效：训练 reward 相对 baseline 提升；greedy FCT 在最佳静态策略容差内；并至少显著优于一个非最佳静态策略。退出码 0/1 可直接成为实验 gate。

### [`analyze_forgetting.py`](../../scripts_exp/analyze_forgetting.py)

同时使用 before/after reward 和公共 flow FCT。只有 reward 明显下降且 FCT 恶化才判定遗忘，再要求 SOR forgetting score 小于 ACC。

### [`generate_experiment_report.py`](../../scripts_exp/generate_experiment_report.py)

读取三个 stage 的 JSON，生成 `experiment_report.md`，并根据总 PASS/FAIL 返回退出码。当前预期目录名与不同版本 stage 脚本可能不一致，正式运行前需统一路径。

### [`status_experiment.sh`](../../scripts_exp/status_experiment.sh)

轻量显示 run status、`.done` 和步骤目录，供监控正式流水线。

## 5.8 流量生成代码

### [`tools/traffic/TraGen.py`](../../tools/traffic/TraGen.py)

读取 `--flow-groups JSON`，为每个 group 按 CDF 和 offered load 生成 flow。支持：

- `poisson_random`：随机源到随机目的的泊松到达。
- `poisson_incast`：大量源集中到少数目的。
- `all_reduce`：按轮次/环或分组产生集合通信流。
- `all_to_all`：源目的全互连。

`CustomRand` 校验并插值 CDF；`translate_bandwidth` 解析 G/M/K 单位；泊松间隔由目标 load、链路带宽和平均 flow size 决定。输出按开始时间排序：

- `result/<config>.flow`：ns-3 输入。
- `result/<config>_flows.json`：便于检查和其他工具使用。

脚本依赖当前工作目录中的 `pattern/` 和 `result/`，建议从 `tools/traffic/` 运行并显式固定随机 seed。

### [`tools/traffic/TraGen_random.py`](../../tools/traffic/TraGen_random.py)

在 TraGen 基础上增加随机选择 CDF、避免同源同一开始时间重复等逻辑，用于构造混合/多租户 workload。功能大量重复，后续应抽共享库而不是维护两份生成器。

### [`scenA_throughput_config.json`](../../tools/traffic/scenA_throughput_config.json)

场景 A：host 0–127 发往 128–255，Hadoop CDF，`poisson_random`，load=.8，从 2.0 秒开始持续 .05 秒，偏吞吐型。

### [`scenB_incast_config.json`](../../tools/traffic/scenB_incast_config.json)

场景 B：128 个源发往目的集合中的 4 个随机 incast 目标，AliStorage CDF，load=.05，同样从 2 秒开始，用于与场景 A 构成分布变化。

### [`pattern/default.config`](../../tools/traffic/pattern/default.config)

多 group 混合 workload 示例，按不同时段叠加 WebServer、WebSearch、Hadoop 等 CDF。扩展名虽为 `.config`，内容是 JSON。

### [`pattern/pattern_1.json`](../../tools/traffic/pattern/pattern_1.json)

空占位文件，不能直接作为 TraGen 输入。

### [`pattern/README`](../../tools/traffic/pattern/README)

只有一句“traffic generation configuration”说明，尚未记录 schema；本手册上面的字段解释可作为补充。

### CDF 文本文件

- [`AliStorage.txt`](../../tools/traffic/pattern/AliStorage.txt)
- [`GoogleRPC.txt`](../../tools/traffic/pattern/GoogleRPC.txt)
- [`Hadoop.txt`](../../tools/traffic/pattern/Hadoop.txt)
- [`VL2_CDF.txt`](../../tools/traffic/pattern/VL2_CDF.txt)
- [`WebSearch.txt`](../../tools/traffic/pattern/WebSearch.txt)
- [`WebServer.txt`](../../tools/traffic/pattern/WebServer.txt)
- [`cachefollower-all.txt`](../../tools/traffic/pattern/cachefollower-all.txt)
- [`hadoop-all.txt`](../../tools/traffic/pattern/hadoop-all.txt)

每行通常是 `flow_size cumulative_probability`。不同数据文件的 flow size 单位必须由生成器/来源确认，不能只凭文件名推断。`CustomRand.testCdf()` 要求概率单调、终点接近 1。

### [`pattern/flow_sizes.py`](../../tools/traffic/pattern/flow_sizes.py)

用固定硬编码路径读取多个 CDF，绘制对数 x 轴论文风格 CDF 图并保存 `flow_size_cdf.pdf`。它不是生成器依赖；路径 `/home/ame/...` 需参数化。

### [`pattern/flow_size_cdf.pdf`](../../tools/traffic/pattern/flow_size_cdf.pdf)

上一个脚本生成的静态图表资产。

### [`result/default_flows.json`](../../tools/traffic/result/default_flows.json)

约 311 万行的历史生成结果，只用于记录每条 flow 的 id/src/dst/size/start。文件非常大，不应在训练时整体加载；对应 `.flow` 未跟踪时它不能直接被 ns-3 使用。

### [`tools/traffic/.gitignore`](../../tools/traffic/.gitignore)

控制生成结果是否进入 Git。大型临时 flow/JSON 应默认忽略，只提交小型可复现实验输入或生成配置。

## 5.9 通用 FCT 分析工具

### [`tools/analysis/analysis_fct.py`](../../tools/analysis/analysis_fct.py)

解析原始 FCT，计算总体和按 flow-size 百分位分桶的 avg/P50/P95/P99 实际 FCT，支持小流/大流分组。部分入口函数使用硬编码文件列表。

### [`analysis_fct_slowdown.py`](../../tools/analysis/analysis_fct_slowdown.py)

与上一个文件结构相同，但指标是 `max(1,actual_fct/ideal_fct)`。Slowdown 更适合跨 flow size 比较。

### [`fct_time.py`](../../tools/analysis/fct_time.py)

把每条 flow 放到 `start + FCT/2` 的时间 bucket，绘制 FCT/slowdown 随仿真时间变化，观察流量阶段切换。

### [`com_fct.py`](../../tools/analysis/com_fct.py)

生成多方法论文级综合图：整体、百分比/分位、flow-size group 对比。包含大量固定路径、方法名称映射和超大字号，当前更像某篇论文实验的定制脚本。

### [`thesis_com_fct.py`](../../tools/analysis/thesis_com_fct.py)

`com_fct.py` 的论文排版版本，调整字体、刻度、外框和图尺寸，同样硬编码数据集路径。当前第 326 行把位置参数 `ax_relative_pdf` 放在关键字参数 `palette=...` 之后，Python 会报 `SyntaxError: positional argument follows keyword argument`；该文件在修正调用参数顺序前不能运行。

### [`normalized_fct.py`](../../tools/analysis/normalized_fct.py)

从 summary 文本中用正则提取 overall FCT，按指定 baseline 归一化并绘图。它读取的是分析报告，不是 raw FCT。

## 5.10 队列、速率和吞吐工具

### [`queue_time.py`](../../tools/analysis/queue_time.py)

解析五列端口队列 monitor：switch、buffer、connected node、queue bytes、time。可平滑时间序列、按交换机/端口绘图并与 baseline 比较。

### [`queue_time_incast.py`](../../tools/analysis/queue_time_incast.py)

针对 incast 扩展：同时组织端口级和交换机总 occupancy，画单端口、总占用、总队列和 baseline 对比，更适合发现热点汇聚端口。

### [`rate_time.py`](../../tools/analysis/rate_time.py)

解析六列 rate monitor：switch、port、line rate、tx rate、ECN rate、time。生成时间序列、相对 baseline、归一化柱状图，并可批量处理方法目录。

### [`analysis_throughput.py`](../../tools/analysis/analysis_throughput.py)

读取 throughput monitor 到 DataFrame，统计均值/峰值/P95/P99，绘制时间序列、分布、热点交换机和相对线速。顶部路径硬编码旧实验。

### [`throughput_plots.py`](../../tools/analysis/throughput_plots.py)

不读取原始数据，使用脚本内手工填写的 summary 数值画柱状图。适合论文排版，不适合自动实验流水线。

## 5.11 综合分析工具

### [`analysis_ecn.py`](../../tools/analysis/analysis_ecn.py)

从参数扫描 CSV 提取文件名中的 Kmin/Kmax，使用 cubic grid interpolation 画平均/尾部 FCT 热图，可用于生成 fmap 或观察参数敏感区域。输入路径和 CSV schema 固定。

### [`run_sor_acc_analysis.py`](../../tools/analysis/run_sor_acc_analysis.py)

把 FCT slowdown、FCT 对比、时间变化、队列、rate 和 throughput 六类分析整合成一个 ACC/SOR Hadoop 报告。近千行代码复制了多个工具逻辑，并硬编码 `/root/paddlejob/...`；优点是一键出完整图，缺点是难复用和维护。

### [`run_curriculum_fct_compare.py`](../../tools/analysis/run_curriculum_fct_compare.py)

聚合 curriculum 每 phase/task 的 FCT，支持 `flat` 最近一次结果和 `archived` 完整归档布局，可生成 ACC/SOR phase×task heatmap。文档明确指出如果仿真反复覆盖同名输出，就无法恢复历史结果。

## 5.12 选择分析脚本的建议

| 需求 | 首选文件 |
|---|---|
| 快速比较若干 FCT | `scripts/analyze_fct.py` |
| 固定动作公平比较 | `simulation/scripts/sanity_fct_compare.py` |
| 自动 gate | `scripts_exp/check_stage0_gate.py` |
| ACC 是否有效 | `scripts_exp/analyze_acc_effectiveness.py` |
| ACC/SOR 是否遗忘 | `scripts_exp/analyze_forgetting.py` + `sor/compare_forgetting.py` |
| 队列热点 | `queue_time_incast.py` |
| Tx/ECN 时间变化 | `rate_time.py` |
| 完整 ACC/SOR 图组 | 参数化后的 `run_sor_acc_analysis.py` |

正式论文结果应优先使用可参数化、取公共 flow、保存 JSON/CSV 且记录输入路径的脚本；硬编码路径的绘图脚本更适合作为样式参考。
