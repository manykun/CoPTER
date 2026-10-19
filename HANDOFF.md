# CoPTER / SOR 实验交接

整理日期：2026-10-19（Asia/Shanghai）。面向在服务器上直接使用 Codex 的后续工作。

## 1. 接手位置与当前状态

**当前任务：用 WebSearch 流量并行进行两组 ACC 在线训练，各 50 epoch，比较 `acc_paper` 与 `acc_3816bc6_aligned` 奖励。用户明确取消本轮冻结评估。**

代码已提交并通过 SSH 推送到 `manykun/CoPTER` 的 `exp/acc-validation` 分支。服务器最近确认的 HEAD 是 `1079f56`。本对话没有直接登录服务器执行命令；服务器状态来自用户粘贴的日志。

最近一次用户操作：在 `CoPTER/ns-3.33/` 下运行相对路径 `scripts/continual_validation/run_acc_websearch_rewards.sh`，报文件不存在。已告知先 `cd /mnt/sdb1/xuduokun/projects/CoPTER` 再执行。**尚未收到 smoke 成功、50 epoch 正式训练已启动或新实验结果的确认。** Protobuf 最后采用的 `-DNDEBUG` 构建方案也尚无成功日志，应先检查实际产物。

### 路径映射（重要）

| 内容 | 本机路径 | 服务器路径 |
|---|---|---|
| 活跃仓库根目录 | `/Users/manykun/Documents/Codex/2026-07-15/wya/work/CoPTER-acc-validation` | `/mnt/sdb1/xuduokun/projects/CoPTER` |
| 本机外层工作目录 | `/Users/manykun/Documents/Codex/2026-07-15/wya` | 无直接对应，不是服务器项目根目录 |
| 服务器用户与环境 | 本机用户 manykun | `xuduokun@server-103`，Conda 环境 `m3`，Python 3.9.25 |
| 用户提供的结果与附件 | `/Users/manykun/Downloads/`、`/Users/manykun/.codex/attachments/` | 不保证存在，需在服务器原实验目录寻找或另行传输 |

本文后续路径默认相对于仓库根目录。**任何 `/Users/manykun/...` 路径必须在服务器重新对应，不能直接用于服务器命令。** 本机原始附件和本机生成的报告不应被误认为已在 Git 中或已上传服务器。

## 2. 任务目标、用户要求与偏好

- 长期目标：完成 SOR 抗灾难性遗忘论文实验，先证明 ACC 确实有有效获取能力，再评估迁移后的遗忘和 SOR 效果。
- 当前问题：长期看到 TD loss 下降、部分 reward 上升，但 FCT、吞吐等物理指标只小幅震荡。不能据此声称模型有效或存在灾难性遗忘。
- 用户强调不能只比较训练开始/结束数值；需要完整曲线、分段趋势和波动范围，也不能只靠改流量场景解释或解决学习问题。
- 当前明确选择 WebSearch，区别于 WebServer；两组 ACC 并行，50 epoch，无冻结评估，不同时加入 SOR。
- 本轮比较论文奖励与 Git 提交 `3816bc6` 的历史奖励，不是此前的 `weighted` 奖励；勿把三者混淆。
- 记录每 epoch 的 reward、loss、FCT、完成率、吞吐、queue、ECN、PFC，以及 Q、epsilon、动作和奖励分项等诊断信息，输出 CSV 和图表。
- reward 要兼顾吞吐与队列/延迟，不接受仅以尾延迟为目标；不同奖励公式的绝对 reward 不可直接比较。
- global buffer 指跨端口全局 replay；switch buffer 指交换机缓冲区，二者不同。当前关闭全局 replay，交换机 buffer 为 400 KiB。
- epsilon 使用跨 epoch/任务连续的全局调度；target 按 global optimizer update 同步，不用每 episode 重置的 current_step。
- 提供可直接复制的服务器命令，尤其明确当前工作目录，避免占位符、复制 Markdown URL 或把两条命令粘在一起。
- Git 默认使用 SSH。不要假定服务器存在名为 `fork` 的 remote；可直接使用 `git@github.com:manykun/CoPTER.git`。
- 需要修改、提交或推送时，保护已有用户文件和结果，不覆盖无关改动，不重置工作树。
- 首先阅读根目录 `AGENTS.md`；这是现有项目规则文件，不是 `AGENT.md`。

## 3. 已完成的历史工作和主要结论

以下用于解释当前实验方向，不代表本轮 WebSearch 已得到结果。

1. 已做 global/local replay 对照、全网/单端口干预、参数范围扩大、细密插值、多种受控冲突场景和真实流量迁移、ABA/CDC，以及 reward × traffic 筛查。
2. 多次任务对搜索没有选出有效冲突：不同任务常选择相同 reward-best 动作，或动作不敏感、完成率不足。用户曾明确要求停止持续修改场景寻找有利任务对。
3. 多个单端口插值结果：A 场景队列很浅、ECN/PFC 基本不触发，reward 和物理指标近乎平坦。说明动作变化不自动产生收益。
4. 真实 WebServer→CacheFollower 的一次全网 frozen-A 比较：reward -0.48%、p95 FCT +3.64%、completion 下降约 0.0032 pp。只能作为轻微退化观察；单 seed、获取不足和自然训练漂移等因素未充分排除，不能称为显著灾难性遗忘。
5. ACC target 同步此前使用 episode-local 计数存在问题，后来改为 global_train_step 每 100 次更新同步；加入 Q/TD 诊断后，用户一次报告的 Q abs max 约 0.58/1.02，未超过 expected bound 20、warning 40。loss 峰值/下降不等于策略表现。
6. 历史 Hadoop 固定动作敏感性测试（同报告 common flows=4882）：

   | 策略 | Mean FCT us | p95 FCT us |
   |---|---:|---:|
   | fixed high_gentle | 113.197 | 375.276 |
   | frozen epoch 0 | 126.470 | 464.090 |
   | frozen epoch 100 | 125.853 | 458.962 |

   初末模型参数 hash 不同、frozen 动作 TV=0.2491；模型确实变化，环境也有可优化空间，但学习没有有效利用。不能再简单归因于“参数完全没更新”。大量空闲样本下 q>=Kmin 很低，也不代表所有瓶颈都没有压力。
7. 历史 4种流量 × 3种奖励、50 epoch 矩阵的 frozen final-vs-init mean FCT 变化如下（历史结果，本轮不再加 frozen）：

   | 流量 | tail5 | tail1 | weighted |
   |---|---:|---:|---:|
   | WebServer | -0.34% | +2.75% | +6.23% |
   | CacheFollower | +0.37% | -3.76% | +1.36% |
   | Hadoop | -0.71% | -2.22% | -0.66% |
   | AliStorage | +2.00% | -1.59% | +0.20% |

   tail5 是 queue/ECN lambda=5/5；tail1 是1/5；weighted 是0.5/0.3/0.2。没有得到跨场景稳定的明显获取效果。重复相同 seed/流量测试的是可重复性，不是独立 seed 显著性。
8. 历史 Hadoop queue-lambda 两组决策实验：候选 final 相对 control mean FCT -1.51%，候选自己相对初始化 -2.22%，未达到当时预注册的 -3% 筛查条件；不能挑选 p95 单项宣布成功。

历史目录线索：`experiments/aba/acc_reward_traffic_matrix_s1_*`、`acc_credit_hadoop_s1_*`、`acc_queue_decision_s1_*`；早期真实迁移基线 `experiments/continual_validation/real_web_cache_target100_s1`。各报告的共同流交集可能不同，不能跨报告直接混合绝对 FCT。

## 4. 当前两组实验的关键决定

### 4.1 固定配置

| 项目 | 设置 |
|---|---|
| Task | `realistic_websearch`，单任务获取 |
| 两组名称 | `acc_paper`、`acc_3816bc6_aligned` |
| 算法与预算 | ACC；正式各50 epoch；smoke各2 epoch |
| 并行 | jobs=2；端口8200、8201，运行前检查占用 |
| 流量 | WebSearch CDF；Poisson；配置负载60%；源0–127、目的128–255 |
| 生成参数 | host bandwidth 10 Gbps；2.0s开始；流量持续30ms；seed=1 |
| buffer | 400 KiB（代码和报告明确 KiB=1024 bytes） |
| 动作空间 | `multiscale`，9个联合阈值profile × 7个Pmax；本轮保持一致 |
| 网络/学习参数 | hidden 32,64,64,32；lr=0.001；gamma=0.95 |
| replay | local FIFO，每端口1000；global replay关闭；相同50% congestion-stratified目标 |
| epsilon | start=1、end=0.05、global decay=2500 |
| target同步 | 每100 global optimizer updates |
| 动作周期 | 现有500us；不在本轮同时改变 |
| 冻结评估 | **无，用户明确取消** |

两组共用相同流量输入和 seed，模型初始化过程相同；模型、日志、端口和输出目录独立。尚未从服务器结果核实初始参数 hash 是否一致。

### 4.2 两种奖励

令 U 为本动作区间发送字节数×8 / 区间秒数 / 链路bps，L为本动作区间时间平均队列字节数。

- `acc_paper`：`r = 0.7*U + 0.3*D(L)`。阈值 `20*1024*2^n bytes, n=0..9`，取使阈值>=L的最小n，`D=1-n/10`。高于最后阈值时显式将n封顶9（得分0.1），这是实现的边界约定。0.7/0.3是论文给出的推荐示例权重，不是唯一允许值。
- `acc_3816bc6_aligned`：来自 `3816bc6dd293ae7432ca6e4d0437efc7baceb9ae` 的公式；`q=min(1,L/(400*1024))`，对 `[2^i/1024, i=0..10]` 使用 bisect_left，得分 `[1,.9,.8,...,0]`；`r=.5*U+.5*D`。
- 历史公式的物理阈值是约0.391、0.781、1.563、3.125…400 KiB。旧代码注释提及1024KB，但不能据注释把实际阈值解释为1、2、4…1024KB。
- 旧提交输入是瞬时队列；本轮历史公式改用与论文组相同的区间平均队列，因此命名 `aligned`，不能宣称完整复现旧实现。
- 两者均无独立 ECN 奖励项。保留现有 weighted、tail_safe 供旧实验使用，但本入口只跑上述两组。
- U目前不裁剪以避免悄悄改变历史公式；大于1时应检查字节计数边界/诊断，不能掩盖。

**原因：** 当前旧 reward 的4步/32步滚动统计混合了多个动作后的观测，每500us换动作时相当于把约2ms/16ms窗口归给最新动作。且 ns-3 返回的是区间峰值，对几个峰值求平均不等于平均队列。现在在动作应用时重置区间积分，入/出队事件累计 byte×ns，到下一观测才结算；吞吐用发送总字节计数差。状态历史仍保留原来的峰值队列特征，奖励从独立 telemetry 读取。Python拒绝缺失、重叠或不连续的 reward interval。

论文组浅队列<=20KiB时 queue score 恒1；必须检查 queue-bin 分布是否几乎全在首档，不能预先承诺论文奖励一定让FCT下降。

参考：ACC论文 https://conferences.sigcomm.org/sigcomm/2021/files/papers/3452296.3472927.pdf

## 5. 已提交工作与主要文件

远程：`git@github.com:manykun/CoPTER.git`；分支：`exp/acc-validation`。

| 提交 | 内容 |
|---|---|
| `04de5c6` | WebSearch两组入口、奖励公式、ns-3区间统计、逐epoch分组FCT与报告 |
| `1079f56` | OpenGym统一protoc/header/library，添加版本及生成消息编译/链接/运行检查 |
| `76acc59` | 更早的矩阵 frozen 评估独立端口范围，与当前无冻结方案不同 |

本机 remotes：`fork`是manykun/CoPTER的SSH；`origin`是arduino-ctrl/CoPTER的HTTPS。服务器 remote名称可能不同，须先查看。

主要文件（仓库相对路径）：

- `AGENTS.md`：项目规则。
- `scripts/continual_validation/run_acc_websearch_rewards.sh`、同名`.py`：本轮唯一总入口；stage `prepare/smoke/all/run/analyze`，没有 frozen stage。
- `scripts/continual_validation/run_acc_reward_action_pilot.py`：两组variant和共同配置，旧pilot继续兼容。
- `scripts/continual_validation/run_aba.py`：epoch调度、输入哈希、独立输出、完整恢复、完成标记及清理。已修正single-task smoke不应额外跑B/A2。
- `simulation/run_100.sh` → `simulation/run-copter-sim.sh`：单epoch ns-3执行。
- `copter/run_100.sh` → `copter/copter.py`：单epoch ACC执行；调度器不直接绕过这两个入口。
- `copter/acc_rewards.py`：两种奖励及interval telemetry校验。
- `copter/network_helper.py`：读取ns-3信息，独立结算本动作reward。
- `copter/port_metrics.py`：watch trace、奖励分项、queue-bin计数。
- `copter/agent_helper.py`、`agent.py`、`structures.py`：现有replay、DDQN训练、参数/全局计数。
- `ns-3.33/src/point-to-point/model/action-queue-integral.h`、`switch-mmu.h/.cc`：事件积分平均队列。
- `ns-3.33/scratch/copter-sim.cc`：动作开始重置积分/字节基线；通过extraInfo的`reward_interval` v1发送时间戳、mean_queue_bytes、tx_utilization。
- `ns-3.33/src/point-to-point/wscript`：注册新header。
- `ns-3.33/contrib/opengym/wscript`：Protobuf版本、路径、链接和真实生成消息probe。
- `tools/analysis/metrics_core.py`：权威统计公式，含按流大小FCT。
- `tools/analysis/analyze_aba_epoch.py`：每epoch CSV/JSON。
- `tools/analysis/analyze_acc_reward_action_pilot.py`：全曲线、10-epoch平滑及分段表、分组FCT、reward分项和Q/epsilon图。
- `docs/acc-websearch-reward-comparison.md`：运行说明。**目前其构建命令尚未包含后续对话发现的 `-DNDEBUG`，服务器应优先使用本文下一节命令；后续可同步修正文档。**

流量来源：

- `tools/traffic/acc_validation/realistic_websearch.json`
- `tools/traffic/pattern/WebSearch.txt`
- `tools/traffic/TraGen.py`
- `simulation/mix/256hosts.topo`
- conf模板借用 `simulation/mix/aba/webserver_cachefollower/task_a_webserver.conf` 的网络设置，FLOW_FILE会替换为真正WebSearch flow，任务标签也是WebSearch。
- 生成的规范输入：`simulation/mix/aba/websearch_rewards_s1/{task_websearch.flow,task_websearch.conf,manifest.json}`。manifest记录生成配置、CDF、生成器、topo、模板和flow/conf哈希。未完成生成或哈希改变时拒绝静默复用。

## 6. 验证结果与边界

本机已完成：

- 5项interval reward测试：与历史公式对照、论文边界、时间戳拒绝、避免旧窗口/峰值污染、C++事件积分fixture（编译执行）。
- 24项权威指标fixture、scheduler/恢复保护、两组配置一致性和流大小边界测试。
- 2项port metrics测试；合计31项通过。
- 2项Protobuf配置回归测试通过，测试模拟配置调用、拒绝版本混用、参数顺序等；**不等于真实服务器链接成功**。
- Bash语法、Python编译及`git diff --check`通过。
- 本机prepare成功生成seed1的1568条WebSearch flows；重复prepare验证哈希和协议一致。生成目录/结果被忽略，不随Git自动传输；服务器入口会自行准备。
- 本机没有完整ns-3构建产物及m3/Torch训练环境，未执行真实两组仿真smoke，也未生成本轮真实训练曲线。

测试命令（从服务器仓库根目录运行）：

```bash
python -m unittest discover -s copter -p test_acc_interval_rewards.py
python -m unittest discover -s copter -p test_port_metrics.py
python -m unittest tools.analysis.tests.test_metrics_core tools.analysis.tests.test_aba_scheduler tools.analysis.tests.test_websearch_reward_pilot
python ns-3.33/contrib/opengym/test/test_protobuf_configure.py
```

## 7. 服务器构建排障结论（继续工作必读）

已知服务器安装：

- `/usr/bin/protoc` =3.6.1；`/usr/include/google/protobuf/stubs/common.h`宏3006001。
- `/usr/local/bin/protoc` =3.20.3；`pkg-config protobuf`=3.20.3，指向`/usr/local/include`和`/usr/local/lib`；header宏3020003。
- 编译器默认选`/usr/local/include`，GCC7.5.0。

排障经历与证据：

1. 原opengym脚本硬编码`/usr/bin/protoc`，3.6.1生成代码遇3.20.3头文件，出现MutableNoArena等不存在。
2. 手动用3.20.3生成C++后，旧Waf缓存仍让OpenGym依赖`libprotobuf.so.17`，ldd定位到`/lib/x86_64-linux-gnu/`。最终链接命令没有`-L/usr/local/lib`，带旧的前置`-lprotobuf`。仅设LIBRARY_PATH不足以覆盖该缓存。
3. `1079f56`修复选路并增加probe；重新configure日志证实protoc/header/library路径均已到3.20.3，基础`GOOGLE_PROTOBUF_VERIFY_VERSION`检查通过。
4. 生成消息probe仍报 `InternalMetadata::~InternalMetadata()`未定义；详细log中链接命令已有`-L/usr/local/lib`、rpath。此时不是继续猜旧缓存。
5. 查证Protobuf3.20.3的`metadata_lite.h`：析构函数定义受`NDEBUG`影响，匹配官方ABI问题 https://github.com/protocolbuffers/protobuf/issues/9947 。最终建议定义`-DNDEBUG`重配、重编译；未收到成功确认。

**`-DNDEBUG`尚未写入已推送代码或运行说明，只是最后提供给用户的环境构建命令。** 它关闭标准C/C++ assert；当前ns-3 Debug的`NS_ASSERT`受独立`NS3_ASSERT_ENABLE`控制，仍保留。不要直接修改生成消息的函数名，不要运行旧build脚本中的`sudo mv /usr/local/include/google/protobuf`操作，不需要卸载系统库。

完整构建命令（失败立即停止；保留日志；不删除实验结果）：

```bash
bash <<'SH'
set -euo pipefail
cd /mnt/sdb1/xuduokun/projects/CoPTER/ns-3.33
export PATH="/usr/local/bin:$PATH"
export PROTOC=/usr/local/bin/protoc
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib64/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="/usr/local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CXXFLAGS="${CXXFLAGS:-} -DNDEBUG"
./waf configure --enable-examples --enable-mpi 2>&1 | tee /tmp/copter-configure.log
./waf build --targets=copter-sim -j4 2>&1 | tee /tmp/copter-build.log
ldd build/scratch/copter-sim | grep protobuf
SH
```

环境export在上述子shell外不保留；Waf会保存编译配置和RPATH。若仍有运行时库搜索问题，在启动实验的shell也显式设置相同LD_LIBRARY_PATH。不要在失败的configure后继续build旧缓存。

应看到`Checking generated OpenGym messages compile, link and run : yes`，构建成功，实际加载的Protobuf路径是`/usr/local/lib/`。详细失败信息看`ns-3.33/build/config.log`；不是只看摘要 `/tmp/copter-configure.log`。

可用于核实而不改变产物的命令：

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER/ns-3.33
readelf -d build/lib/libns3.33-opengym-debug.so | grep -E 'NEEDED|RPATH|RUNPATH'
ldd build/lib/libns3.33-opengym-debug.so | grep protobuf
nm -D -C /usr/local/lib/libprotobuf.so | grep -E 'InternalMetadata::~InternalMetadata|EpsCopyOutputStream::(WriteRawFallback|WriteStringMaybeAliasedOutline)'
tail -n 180 build/config.log
```

## 8. 下一步实际操作

1. 在服务器读取`AGENTS.md`和本文，检查`git status`/HEAD、现有进程、端口、实验目录和最新日志，确认是否已经启动，不重复运行同一arm。
2. 如尚未成功构建，使用第7节含`-DNDEBUG`的命令完成构建验证。确认两个配置probe通过，并验证最终ELF依赖。
3. 回到仓库根目录，运行第6节测试，再启动smoke：

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
bash scripts/continual_validation/run_acc_websearch_rewards.sh \
  --stage smoke --run-id-prefix acc_websearch_rewards_s1 \
  --epochs 50 --seed 1 --port-base 8200 --jobs 2
```

4. 确认两组各完成2 epoch、指标可分析、无interval校验失败、模型/日志/输出隔离。smoke目录含`_smoke_`，不要拿它的模型当正式初始化。若已存在失败/部分目录，先核实完整状态，再决定`--resume`；不要盲目删除或跳过。
5. 正式并行训练50 epoch（从头初始化，无冻结）：

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
mkdir -p experiments/aba
nohup bash scripts/continual_validation/run_acc_websearch_rewards.sh \
  --stage all --run-id-prefix acc_websearch_rewards_s1 \
  --epochs 50 --seed 1 --port-base 8200 --jobs 2 \
  > experiments/aba/acc_websearch_rewards_s1_driver.log 2>&1 &
echo $!
```

6. 查看日志：`tail -f experiments/aba/acc_websearch_rewards_s1_driver.log`。完成后检查汇总报告和图形；两组应各50条epoch记录，无重复/缺失。
7. 需要恢复时在相同正式命令增加`--resume`，保持seed、预算、端口、输入和profile一致。模型必须与最后完整epoch匹配；若进程已保存本epoch模型但没有完成证明，调度器会拒绝含糊恢复，应分析原始输出和checkpoint而非冷启动。
8. 分析重点：完整FCT轨迹与10-epoch分段、完成率/吞吐代价、分组FCT、reward分项和队列档位、Q和动作变化。若依旧只有loss下降，需要进一步查学习信号、动作收益与DDQN实现；不能先宣称SOR有效。下一轮是否扩展seed/SOR，依据实际结果与用户决定。

## 9. 结果、配置和日志位置

前缀默认`acc_websearch_rewards_s1`，ARM取`acc_paper`或`acc_3816bc6_aligned`。

```text
simulation/mix/aba/websearch_rewards_s1/
  task_websearch.flow
  task_websearch.conf
  manifest.json
  runtime/<run_id>/...                 # 实际epoch conf，按调度器生成路径核实

experiments/aba/<prefix>_<ARM>/
  pilot_config.yaml
  protocol.json
  state.json
  acc/models/                         # policy、target、optimizer、replay、RNG、计数等
  acc/checkpoints/a1/                  # 单任务阶段结束边界
  acc/logs/a1/epoch_XXXX_{ns3,agent}.log
  summaries/acc_epoch_metrics.csv
  summaries/acc_epoch_ports.csv

simulation/output/aba/<prefix>_<ARM>/acc/
  epoch_metrics.csv
  epoch_ports.csv
  a1/epoch_XXXX/
    result.fct / result.queue / result.rate / result.pfc / ...
    watch_trace.jsonl
    fct_steps.csv
    epoch_metrics.json
    epoch_ports.csv
    analysis_provenance.json
    complete.json

experiments/aba/<prefix>_comparison/
  REPORT.md
  epoch_metrics.csv
  figures/reward_loss.png
  figures/fct_curves.png
  figures/network_curves.png
  figures/fct_{short,medium,long}.png
  figures/reward_components.png
  figures/q_epsilon.png
  figures/replay_signal.png
```

raw文件只在分析成功后依既有策略清理，单任务a1末轮完整保留；并非所有epoch的巨大fct/queue/rate文件永久存在。其统计、watch trace、日志和完成标记要保留。不要依赖被清理的raw重新统计所有历史epoch。

权威统计：FCT文件第7/8列为actual/ideal ns，输出us；分位数线性插值位置`(n-1)*q`；slowdown=`max(1,actual/ideal)`；completion=有效完成流数/input.flow声明总数。每epoch使用该轮全部有效完成流；不平均step p95得到epoch p95。短流<=100KiB，中流(100KiB,1MiB)，长流>=1MiB；每组保留完成数量。ECN字段需区分“标记字节速率/链路带宽”与“被标记包占总包比例”，不能混称；PFC映射缺失是n/a而非0。

## 10. 尚未解决和交接边界

- 服务器的最终构建、联调smoke和正式启动状态待确认；不能把提供了命令当作已完成。
- 当前WebSearch方案尚无训练结果；不保证FCT一定下降，更没有新的SOR/遗忘结论。
- 构建文档需补入已确认的3.20.3/NDEBUG兼容说明；如固化进构建逻辑，须显式记录宏语义并验证，不能跳过失败probe。
- 本轮仅修复reward与窗口；其他历史实现差异（例如状态维度、采样方式与论文不同）仍存在，因此不能称为完整ACC论文复现。旧对话提及transition终止标记等问题，本轮没有据此实施修改；后续应读代码核实，不要依据旧断言盲改episode终止语义。
- 正式训练含epsilon探索，单seed线上曲线不足以做冻结策略或统计显著性结论；用户当前仍明确不做冻结评估。
- 当前Git忽略experiments、simulation/output和大部分生成flow/conf；上传代码不代表上传了结果或本机输入。
- 本机有用户既有未跟踪文件：`analysis_outputs/`、`docs/realistic-webserver-cachefollower-global-epsilon-results.md`、`docs/figures/realistic_acc_global_eps_*.png`、`scripts/continual_validation/plot_realistic_global_epsilon.py`、`tools/analysis/analyze_acc_matrix_fct_by_size.py`。这些未包含在本轮提交中，勿擅自删除/覆盖/打包进新提交。服务器工作树需另行检查。
- 用户随后要求将本文件通过Git交接。服务器应拉取`manykun/CoPTER`的`exp/acc-validation`分支，并实际确认根目录存在`HANDOFF.md`；远程推送完成不代表服务器工作树已经更新。

给服务器Codex的起始指令可用：

> 先阅读AGENTS.md与HANDOFF.md，检查Git、进程、端口和最近日志。继续WebSearch的acc_paper与acc_3816bc6_aligned两组ACC实验，每组50epoch、两组并行、不做冻结评估。先确认含-DNDEBUG的Protobuf构建是否成功，再完成真实smoke，通过后推进正式实验。保护已有结果和checkpoint，按权威工具逐epoch统计并汇报完整曲线；不要把loss下降当成FCT改善或遗忘证据。
