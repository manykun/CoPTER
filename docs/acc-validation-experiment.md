# ACC 有效性验证实验手册

本文档用于回答两个不同的问题：

1. ACC 参数是否真的能改变网络性能？
2. DDQN 是否能学到比固定参数更好的 ACC 策略？

必须先回答问题 1，再开始长时间训练。若不同固定动作的 FCT 几乎相同，DDQN 无论加深多少层都没有可学习的控制信号。

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

三个场景的最后流注入时间不晚于 `2.08s`。静态基线默认在 `2.25s` 停止，提供至少 170 ms 的排空窗口，避免 SECN/PFC 事件在无业务区间一直计算到 `4.0s`。分析器仍要求 flow completion ratio 至少为 99%；若未达到，必须增加停止时间后重跑，不能比较被截断的 FCT。

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

## 5. 第一阶段：建立论文基线并验证动作是否有效

生成 3 场景 × 3 种子的流量及 ns-3 配置：

```bash
bash scripts/acc_validation/run_validation.sh \
  --stage prepare \
  --run-id accval_v1 \
  --seeds "1 2 3" \
  --scenarios "throughput incast mixed" \
  --buffer-kb 400
```

先运行 18 次论文静态基线实验（3 场景 × 3 种子 × 2 配置）：

```bash
bash scripts/acc_validation/run_validation.sh \
  --stage baseline \
  --run-id accval_v1 \
  --seeds "1 2 3" \
  --scenarios "throughput incast mixed" \
  --buffer-kb 400

python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/accval_v1 \
  --stage baseline \
  --gate
```

再运行 27 次人工动作 sweep，验证 ACC 动作是否真正影响网络：

```bash
bash scripts/acc_validation/run_validation.sh \
  --stage sensitivity \
  --run-id accval_v1 \
  --seeds "1 2 3" \
  --scenarios "throughput incast mixed" \
  --buffer-kb 400
```

分析并启用门槛检查：

```bash
python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/accval_v1 \
  --stage sensitivity \
  --sensitivity-threshold 0.05 \
  --gate
```

判定标准：

- 每种动作的 flow completion ratio 至少为 99%。否则优先检查仿真停止时间、流量强度和 buffer，不能只比较已完成流。
- 同一场景和种子内，三种固定动作的 p95 FCT 相对跨度至少为 5%。至少三分之二的场景通过，才认为 ACC 执行链路具有可控性。
- `reward_vs_lower_fct_spearman` 越接近 1，代表奖励越高时 p95 FCT 越低。若 FCT 有明显差异但该值接近 0 或负数，优先修奖励，不要加深网络。
- 若 FCT 和奖励都没有差异，检查 C++ 是否收到动作、Kmin/Kmax 的物理范围、Pmax、拥塞程度及监控时间窗。

## 6. 第二阶段：训练并冻结评估

只有固定动作灵敏度通过后，才训练 DDQN：

```bash
bash scripts/acc_validation/run_validation.sh \
  --stage train \
  --run-id accval_v1 \
  --seeds "1 2 3" \
  --scenarios "throughput incast mixed" \
  --buffer-kb 400 \
  --episodes 50 \
  --eps-decay 5000
```

训练完成后，以 `epsilon=0`、不写 replay buffer、不更新网络的方式评估：

```bash
bash scripts/acc_validation/run_validation.sh \
  --stage eval \
  --run-id accval_v1 \
  --seeds "1 2 3" \
  --scenarios "throughput incast mixed" \
  --buffer-kb 400

python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/accval_v1 \
  --stage all \
  --gate
```

默认有效性门槛为：greedy ACC 的 p95 FCT 至少比 `SECN_1/SECN_2` 中一组低 5%，并且不超过两组论文基线中较优者的 105%；ACC 和两组基线完成率均须达到 99%。至少三分之二场景、每个场景多数种子达到门槛。

主要结果文件：

```text
experiments/acc_validation/accval_v1/
├── sensitivity/        # 三组固定动作的原始 FCT、队列、速率和奖励指标
├── baseline/           # 论文 SECN_1/SECN_2 静态专家基线
├── eval/               # 冻结策略的评估结果
├── models/             # 训练状态、replay buffer 和模型
├── summary.csv         # 可用于画图的逐场景逐种子汇总
├── analysis.json       # 机器可读门槛结果
└── REPORT.md           # 人工阅读报告
```

## 7. 逐项定位“ACC 无效”的原因

每次只改变一个因素，并使用新的 `run-id`。不要同时改 buffer、奖励和网络，否则无法归因。

### 7.1 Buffer 消融

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

### 7.2 Kmin/Kmax 范围消融

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

### 7.3 奖励函数消融

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

### 7.4 DDQN 网络容量消融

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

随后分别使用相同 `run-id` 和相同 `--acc-hidden-dims` 执行 `--stage eval`。网络容量不改变静态专家配置，可复用 `accval_v1` 的 sensitivity 和 baseline 结果：

```bash
python scripts/acc_validation/analyze_validation.py \
  --run-dir experiments/acc_validation/accnet_large \
  --baseline-run-dir experiments/acc_validation/accval_v1 \
  --stage effectiveness
```

复用基线的前提是场景、seed、buffer、流量输入和仿真版本完全一致。最终比较三个种子的 p95/p99 FCT，而不是只比较训练奖励。

## 8. 推荐结论模板

最终报告至少回答：

1. 固定动作是否在至少两个场景产生大于 5% 的 p95 FCT 差异？
2. 奖励排序是否与更低 FCT 一致？
3. greedy ACC 是否优于至少一组论文静态基线，并接近 `SECN_1/SECN_2` 中的较优者？
4. 结论是否在三个随机种子上稳定，而非由单次运行造成？
5. buffer、K 阈值范围、奖励权重和网络容量中，哪个单因素改变了结论？

若第一问失败，应报告“当前场景或执行链路下 ACC 参数不可辨识”，而不是报告“DDQN 学习失败”。
