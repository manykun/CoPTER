# ACC 有效性验证实验手册

本文档用于回答两个不同的问题：

1. ACC 参数是否真的能改变网络性能？
2. DDQN 是否能学到比固定参数更好的 ACC 策略？

主实验先建立 SECN_1/SECN_2 静态基线，再训练和冻结评估 ACC。固定动作灵敏度、奖励、buffer、动作范围和网络容量属于主实验失败后的消融，不插入主实验流程。

## 1. 本次代码修改

- 修复 `copter/network_helper.py` 中队列占用的重复归一化。ns-3 已按实际 `BUFFER_SIZE` 输出 `[0,1]` 占用率，Python 不再乘 `buffer/400`。
- 将 ACC/SOR 的 Kmin、Kmax、Pmax 离散动作空间集中到 `copter/structures.py`，固定动作和学习策略使用相同映射，并增加索引越界检查。
- epoch 指标增加吞吐、队列、ECN 奖励分量、平均/峰值队列、平均/峰值 ECN 和动作直方图。
- `run_training.sh` 支持单次运行、greedy 评估、固定动作、种子、奖励权重、网络宽度和最大步数；修复 greedy 评估因不更新 epoch 而无限重复的问题。
- `TraGen.py` 支持种子、输出目录、紧凑主机范围和关闭大 JSON 输出。
- 新增三个流量场景、配置生成器、分阶段实验脚本和无第三方依赖的结果分析器。
- ACC 网络宽度可通过命令行配置，用于最后的网络容量消融；默认仍为 `32,64,64,32`。
- 新增论文静态专家基线：SECN_1=`[5 KB,200 KB,1%]`、SECN_2=`[100 KB,400 KB,20%]`。静态基线关闭 OpenGym，直接运行 ns-3 至 `SIMULATOR_STOP_TIME`。

## 2. 三个实验场景

| 场景 | 流量特征 | 主要观察目标 |
|---|---|---|
| `throughput` | 128 对 128、Hadoop CDF、80% 随机负载 | ACC 是否能保持吞吐并降低排队 |
| `incast` | 128 个源汇聚到 4 个目的、AliStorage CDF | Kmin/Kmax/Pmax 是否能抑制突发队列和尾延迟 |
| `mixed` | Hadoop、WebSearch 与 incast 叠加 | 策略在异构负载下是否仍有效 |

每个场景默认使用种子 `1 2 3`。比较方法时，流量场景、种子、buffer 和 ns-3 配置必须完全一致。

论文对比使用以下静态专家配置。数值原样应用于 10/40 Gbps 端口；`@10 Gbps`、`@25 Gbps` 是来源论文的实验条件，不在缺乏依据时自动缩放：

| 名称 | Kmin | Kmax | Pmax | 来源 |
|---|---:|---:|---:|---|
| `secn1` | 5 KB | 200 KB | 0.01 | DCQCN @ 10 Gbps |
| `secn2` | 100 KB | 400 KB | 0.20 | HPCC @ 25 Gbps |

三个场景的最后流注入时间不晚于 `2.08s`。静态基线与 ACC 默认都在 `4.00s` 停止，保证完成率处于同一观察窗口。完成率是有限观察窗口下的结果指标，不再用统一的 99% 阈值否决整组基线；p95/p99 只在待比较方法共同完成的流上计算，避免“困难流未完成反而让尾延迟更低”的幸存者偏差。

```bash
bash scripts/acc_validation/run_validation.sh \
  --stage prepare --run-id paper_baseline_s1 \
  --seeds "1" --scenarios "incast mixed" \
  --baseline-stop-time 2.50

bash scripts/acc_validation/run_validation.sh \
  --stage baseline --run-id paper_baseline_s1 \
  --seeds "1" --scenarios "incast mixed" \
  --baseline-stop-time 2.50
```

`prepare` 和 `baseline` 必须传入相同停止时间。每次运行复制的 `input.conf` 与 `summary.csv` 都会记录实际停止时间。

以下三组只用于动作执行链路的灵敏度检查，不属于论文静态基线：

| 名称 | 索引 | 默认物理含义 |
|---|---|---|
| `aggressive` | `0,0,9` | Kmin/Kmax 取各自下界、Pmax=1.0，最积极标记 |
| `balanced` | `2,1,4` | 中间阈值、Pmax=0.5 |
| `permissive` | `5,3,0` | Kmin/Kmax 取各自上界、Pmax=0.1，最宽松标记 |

配置中的 K 阈值以 25 Gb/s 端口为参考；C++ 会按端口速率乘以 `port_rate/25Gbps`。因此日志中的实际字节值才是最终下发值。

## 3. 服务器准备

在服务器执行：

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
git fetch origin
git switch exp/acc-validation
git pull --ff-only origin exp/acc-validation
conda activate m3

test -x ns-3.33/build/scratch/copter-sim \
  && echo "NS3 binary OK" \
  || echo "NS3 binary missing"

python -c "import torch, numpy, ns3gym; print(torch.__version__)"
df -h /mnt/sdb1 /
git status --short
```

预期 `git status --short` 为空，二进制检查为 `NS3 binary OK`。实验、模型和生成流量均位于 `/mnt/sdb1` 下，不会继续挤占根分区。

## 4. 先做烟雾测试

烟雾测试只使用 `throughput` 的种子 1，并将流量确定性抽样到 2000 条。先运行两组论文静态基线：

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
conda activate m3

bash scripts/acc_validation/run_validation.sh \
  --stage prepare \
  --run-id smoke_acc \
  --smoke

bash scripts/acc_validation/run_validation.sh \
  --stage baseline \
  --run-id smoke_acc \
  --smoke

python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/smoke_acc \
  --stage baseline \
  --gate
```

查看：

```bash
less experiments/acc_validation/smoke_acc/REPORT.md
find experiments/acc_validation/smoke_acc -maxdepth 5 -type f | sort
tail -n 80 experiments/acc_validation/smoke_acc/logs/baseline_throughput_secn1_s1.log
```

烟雾测试的目的只是确认流程可运行；单个种子不能形成实验结论。

## 5. 主实验：静态参数与 ACC

使用全新的 `run-id=acc_study_v2`。先完成 seed 1 的静态、训练和评估闭环，再向相同目录追加 seed 2、3；seed 1 不重跑，且属于最终统计的一部分。

每一批 seed 都严格按照以下顺序执行：

```bash
# 将 SEEDS 设为 "1"；seed 1 结束后改为 "2 3"
SEEDS="1"

bash scripts/acc_validation/run_validation.sh \
  --stage prepare --run-id acc_study_v2 \
  --seeds "${SEEDS}" --scenarios "throughput mixed incast" \
  --buffer-kb 400 --kmin-range 20000,50000 --kmax-range 50000,100000 \
  --baseline-stop-time 4.00

bash scripts/acc_validation/run_validation.sh \
  --stage baseline --run-id acc_study_v2 \
  --seeds "${SEEDS}" --scenarios "throughput mixed incast" \
  --buffer-kb 400 --kmin-range 20000,50000 --kmax-range 50000,100000 \
  --baseline-stop-time 4.00

bash scripts/acc_validation/run_validation.sh \
  --stage train --run-id acc_study_v2 \
  --seeds "${SEEDS}" --scenarios "throughput mixed incast" \
  --buffer-kb 400 --episodes 50 --eps-decay 5000 \
  --acc-hidden-dims "32,64,64,32" --reward-weights "0.50,0.30,0.20" \
  --kmin-range 20000,50000 --kmax-range 50000,100000 \
  --baseline-stop-time 4.00

bash scripts/acc_validation/run_validation.sh \
  --stage eval --run-id acc_study_v2 \
  --seeds "${SEEDS}" --scenarios "throughput mixed incast" \
  --buffer-kb 400 --acc-hidden-dims "32,64,64,32" \
  --reward-weights "0.50,0.30,0.20" \
  --kmin-range 20000,50000 --kmax-range 50000,100000 \
  --baseline-stop-time 4.00
```

seed 1 完成后先运行一次分析。若流程正常，再令 `SEEDS="2 3"` 原样执行上述四步，最后重新分析同一目录：

```bash
python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/acc_study_v2 \
  --stage effectiveness \
  --improvement-threshold 0.05 \
  --completion-tolerance 0.01 \
  --gate
```

默认有效性门槛为：在 SECN_1、SECN_2、greedy ACC 共同完成的流上，ACC 的 p95 FCT 至少比一组静态基线低 5%，并且不超过较优静态基线的 105%；同时 ACC 的全流完成率最多比完成率较高的静态基线低 1 个百分点。至少三分之二场景、每个场景多数种子达到门槛。

若静态结果和 ACC 结果使用不同的 `run-id`，通过 `--baseline-run-dir` 复用已经完成的静态实验，不要重新运行：

```bash
python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/acc_main_s1 \
  --baseline-run-dir experiments/acc_validation/paper_baseline_s1 \
  --stage effectiveness \
  --completion-tolerance 0.01 \
  --gate
```

主要结果文件：

```text
experiments/acc_validation/acc_study_v2/
├── baseline/           # 论文 SECN_1/SECN_2 静态专家基线
├── eval/               # 冻结策略的评估结果
├── models/             # 训练状态、replay buffer 和模型
├── summary.csv         # 可用于画图的逐场景逐种子汇总
├── analysis.json       # 机器可读门槛结果
└── REPORT.md           # 人工阅读报告
```

## 6. 主实验失败后的消融

每次只改变一个因素，并使用新的 `run-id`。不要同时改 buffer、奖励和网络，否则无法归因。

### 6.1 Buffer 消融

依次完整执行 `prepare → sensitivity → analyze`：

```bash
bash scripts/acc_validation/run_validation.sh --stage prepare --run-id accbuf_200 --buffer-kb 200
bash scripts/acc_validation/run_validation.sh --stage sensitivity --run-id accbuf_200 --buffer-kb 200
python scripts/acc_validation/analyze_validation.py --run-dir experiments/acc_validation/accbuf_200 --stage sensitivity

bash scripts/acc_validation/run_validation.sh --stage prepare --run-id accbuf_400 --buffer-kb 400
bash scripts/acc_validation/run_validation.sh --stage sensitivity --run-id accbuf_400 --buffer-kb 400
python scripts/acc_validation/analyze_validation.py --run-dir experiments/acc_validation/accbuf_400 --stage sensitivity

bash scripts/acc_validation/run_validation.sh --stage prepare --run-id accbuf_800 --buffer-kb 800
bash scripts/acc_validation/run_validation.sh --stage sensitivity --run-id accbuf_800 --buffer-kb 800
python scripts/acc_validation/analyze_validation.py --run-dir experiments/acc_validation/accbuf_800 --stage sensitivity
```

注意：生成配置会覆盖 `simulation/mix/acc_validation/*.conf`，因此一个 buffer 的 sensitivity 必须紧跟该 buffer 的 prepare。原始输入配置会复制到各自实验目录，便于追溯。

### 6.2 Kmin/Kmax 范围消融

默认范围是 Kmin `20–50 KB`、Kmax `50–100 KB`。可测试更宽且保持 Kmin 小于 Kmax 的范围：

```bash
bash scripts/acc_validation/run_validation.sh \
  --stage prepare \
  --run-id accrange_wide \
  --kmin-range 10000,40000 \
  --kmax-range 60000,160000

bash scripts/acc_validation/run_validation.sh \
  --stage sensitivity \
  --run-id accrange_wide \
  --kmin-range 10000,40000 \
  --kmax-range 60000,160000

python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/accrange_wide \
  --stage sensitivity
```

Pmax 的固定动作已覆盖 0.1、0.5 和 1.0。若 K 范围改变后 FCT 仍无差异，应从 agent 日志和 ns-3 日志确认动作是否实际进入交换机。

### 6.3 奖励函数消融

当前权重为吞吐/队列/ECN=`0.50,0.30,0.20`。仅当固定动作能改变 FCT、但奖励排序与 FCT 排序不一致时，再测试吞吐优先权重：

```bash
bash scripts/acc_validation/run_validation.sh \
  --stage prepare \
  --run-id accreward_721

bash scripts/acc_validation/run_validation.sh \
  --stage sensitivity \
  --run-id accreward_721 \
  --reward-weights 0.70,0.20,0.10

bash scripts/acc_validation/run_validation.sh \
  --stage train \
  --run-id accreward_721 \
  --reward-weights 0.70,0.20,0.10 \
  --episodes 50 \
  --eps-decay 5000

bash scripts/acc_validation/run_validation.sh \
  --stage eval \
  --run-id accreward_721 \
  --reward-weights 0.70,0.20,0.10

python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/accreward_721 \
  --stage all
```

训练和评估必须使用完全相同的奖励权重及网络结构。

### 6.4 DDQN 网络容量消融

默认 ACC 网络已经有四个隐藏层 `32,64,64,32`，不能仅凭观察结果断言“网络太浅”。在状态/动作/奖励链路通过后，可比较：

```bash
# 较小网络
bash scripts/acc_validation/run_validation.sh \
  --stage train --run-id accnet_small \
  --acc-hidden-dims 64,64 --episodes 50

# 默认网络
bash scripts/acc_validation/run_validation.sh \
  --stage train --run-id accnet_default \
  --acc-hidden-dims 32,64,64,32 --episodes 50

# 更大网络
bash scripts/acc_validation/run_validation.sh \
  --stage train --run-id accnet_large \
  --acc-hidden-dims 64,128,128,64 --episodes 50
```

随后分别使用相同 `run-id` 和相同 `--acc-hidden-dims` 执行 `--stage eval`。网络容量不改变静态专家配置，可复用 `acc_study_v2` 的 baseline 结果：

```bash
python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/accnet_large \
  --baseline-run-dir experiments/acc_validation/acc_study_v2 \
  --stage effectiveness
```

复用基线的前提是场景、seed、buffer、流量输入和仿真版本完全一致。最终比较三个种子的 p95/p99 FCT，而不是只比较训练奖励。

## 7. 推荐结论模板

最终报告至少回答：

1. 固定动作是否在至少两个场景产生大于 5% 的 p95 FCT 差异？
2. 奖励排序是否与更低 FCT 一致？
3. greedy ACC 是否优于至少一组论文静态基线，并接近 `SECN_1/SECN_2` 中的较优者？
4. 结论是否在三个随机种子上稳定，而非由单次运行造成？
5. buffer、K 阈值范围、奖励权重和网络容量中，哪个单因素改变了结论？

若第一问失败，应报告“当前场景或执行链路下 ACC 参数不可辨识”，而不是报告“DDQN 学习失败”。
