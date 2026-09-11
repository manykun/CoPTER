# ACC 灾难性遗忘：时序突发度校准实验

## 1. 为什么先校准

`samepath_steady_stress → samepath_burst_stress` 的首次固定动作筛选表明：

- steady 在端口 323 上没有拥塞样本；
- burst 在端口 323 上持续拥塞，但不同动作的 reward 和 p95 完全相同；
- 两个极端分别落入“无控制信号区”和“动作失效的饱和区”。

粗粒度校准进一步发现 `spread=0.10` 仍对应约 400 μs 的铺开窗口，而
`spread=0` 直接缩到约 0.5 μs，两者之间仍有约 800 倍空档。下一轮保持
1920 条流的 `(src,dst,pg,dport,size)` 多重集合完全不变，测试五个微突发档位：

| 场景 | spread_fraction |
|---|---:|
| samepath_spread002_stress | 0.020（约80 μs） |
| samepath_spread001_stress | 0.010（约40 μs） |
| samepath_spread0005_stress | 0.005（约20 μs） |
| samepath_spread0002_stress | 0.002（约8 μs） |
| samepath_spread0001_stress | 0.001（约4 μs） |

每档只运行 `low_strong`、`mid`、`high_gentle` 三个固定动作，共 15 次冻结
仿真，不更新网络或 replay。校准同时观察全部448个端口，自动寻找在 A/B 中
都达到活跃和拥塞要求的共享端口，不再只假设端口323是瓶颈。

## 2. 运行校准

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
cd /mnt/sdb1/xuduokun/projects/CoPTER

CAL_ID=acc_microspread_calibration_s1
mkdir -p experiments/continual_validation

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage all \
  --run-id "$CAL_ID" \
  --seed 1 \
  --buffer-kb 400 \
  --spread-profile micro \
  --watch-ports all \
  --port 6256 \
  --reward-profile tail_safe \
  --reward-queue-lambda 5.0 \
  --reward-ecn-lambda 5.0 \
  --reward-weights "0.50,0.30,0.20" \
  --acc-hidden-dims "32,64,64,32" \
  > "experiments/continual_validation/${CAL_ID}_driver.log" 2>&1 &

echo $! > "experiments/continual_validation/${CAL_ID}_driver.pid"
tail -f "experiments/continual_validation/${CAL_ID}_driver.log"
```

如果进程中断，复用已完成的候选/动作，不会重跑：

```bash
CAL_ID=acc_microspread_calibration_s1

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage run \
  --run-id "$CAL_ID" \
  --seed 1 \
  --buffer-kb 400 \
  --spread-profile micro \
  --watch-ports all \
  --port 6256 \
  --reward-profile tail_safe \
  --reward-queue-lambda 5.0 \
  --reward-ecn-lambda 5.0 \
  --reward-weights "0.50,0.30,0.20" \
  --acc-hidden-dims "32,64,64,32" \
  --resume \
  > "experiments/continual_validation/${CAL_ID}_resume.log" 2>&1 &
```

恢复运行完成后单独分析：

```bash
CAL_ID=acc_microspread_calibration_s1

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage analyze \
  --run-id "$CAL_ID" \
  --seed 1 \
  --buffer-kb 400 \
  --spread-profile micro \
  --watch-ports all \
  --port 6256 \
  --reward-profile tail_safe \
  --reward-queue-lambda 5.0 \
  --reward-ecn-lambda 5.0 \
  --reward-weights "0.50,0.30,0.20" \
  --acc-hidden-dims "32,64,64,32"
```

## 3. 查看校准结果

```bash
CAL_DIR="experiments/continual_validation/${CAL_ID}/calibration"

cat "${CAL_DIR}/CALIBRATION_REPORT.md"
column -s, -t < "${CAL_DIR}/calibration_summary.csv" | less -S
```

推荐条件已经注册为：

- 所有固定动作的完成率均不低于 90%；
- reward spread 不低于 2%，p95 spread 不低于 5%；
- 至少一个相同端口在两个场景中都活跃 50 步、拥塞 20 步；
- reward 最优动作和完成率安全的 p95 最优动作一致；
- A、B 选择不同动作；
- B 动作施加到 A、A 动作施加到 B 时，p95 代价都不低于 3%。

若生成了推荐文件：

```bash
source "${CAL_DIR}/recommended_pair.env"
echo "Task A: ${RECOMMENDED_TASK_A}"
echo "Task B: ${RECOMMENDED_TASK_B}"
echo "Shared ports: ${RECOMMENDED_WATCH_PORTS}"
```

如果报告为 `Recommended pair: NONE`，停止长训练，并保留报告。若完成率仍高且
端口不拥塞，可继续缩小时间范围；若端口已拥塞但完成率过低且动作完全等价，
先降低 fan-in。完成降载校准后仍无冲突，再改变共享瓶颈的空间路径。

### 64 源微突发无推荐任务对后的降载校准

若 `micro` 报告呈现完成率低于 90%、共享端口已拥塞但三种动作 spread 为 0，
说明 64 源 fan-in 已进入 PFC/瞬时 incast 主导的动作失效区。此时不要继续缩短
spread，也不要开始长训练。改用 32 源的 `micro32` 档位，使每个场景保持相同的
960 条流身份，只改变到达铺开时间；分析阶段只将物理端点均为交换机的链路计作
共享就绪端口。

```bash
CAL_ID=acc_micro32_calibration_s1

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage all \
  --run-id "$CAL_ID" \
  --seed 1 \
  --buffer-kb 400 \
  --spread-profile micro32 \
  --watch-ports all \
  --port 6256 \
  --reward-profile tail_safe \
  --reward-queue-lambda 5.0 \
  --reward-ecn-lambda 5.0 \
  --reward-weights "0.50,0.30,0.20" \
  --acc-hidden-dims "32,64,64,32" \
  > "experiments/continual_validation/${CAL_ID}_driver.log" 2>&1 &

echo $! > "experiments/continual_validation/${CAL_ID}_driver.pid"
tail -f "experiments/continual_validation/${CAL_ID}_driver.log"
```

该配置运行 4 个 spread × 3 个固定动作，共 12 次冻结仿真。只有报告产生
`recommended_pair.env` 后，才进入后续 ACC A→B 训练。

### 32 源恢复可控性后的 48 源中点校准

若 `micro32` 的完成率恢复到 90% 以上并出现动作敏感性，但所有场景的 reward
最优动作仍相同，则在32和64源之间先测试48源。三个场景都包含0–47号发送端、
接收端128和相同的1440条流身份，仅将 spread 设置为0.05、0.02和0.01。

```bash
CAL_ID=acc_micro48_calibration_s1
CAL_ARGS=(
  --run-id "$CAL_ID"
  --seed 1
  --buffer-kb 400
  --spread-profile micro48
  --watch-ports all
  --port 6256
  --reward-profile tail_safe
  --reward-queue-lambda 5.0
  --reward-ecn-lambda 5.0
  --reward-weights "0.50,0.30,0.20"
  --acc-hidden-dims "32,64,64,32"
)

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage prepare "${CAL_ARGS[@]}"

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage run "${CAL_ARGS[@]}" \
  > "experiments/continual_validation/${CAL_ID}_driver.log" 2>&1 &

echo $! > "experiments/continual_validation/${CAL_ID}_driver.pid"
tail -f "experiments/continual_validation/${CAL_ID}_driver.log"
```

运行结束后执行：

```bash
bash scripts/continual_validation/run_spread_calibration.sh \
  --stage analyze "${CAL_ARGS[@]}"

CAL_DIR="experiments/continual_validation/${CAL_ID}/calibration"
cat "${CAL_DIR}/CALIBRATION_REPORT.md"
column -s, -t < "${CAL_DIR}/port_action_summary.csv" | less -S
```

本轮最多运行3个 spread × 3个动作，共9次冻结仿真。如果三个场景仍全部选择
`high_gentle`且完成率不低于90%，下一轮提高到56源；若完成率低于90%并且动作
差异消失，则降低到40源。不要在同一个校准任务对中混用不同 fan-in。

### 时间/fan-in校准无策略翻转后的CDF任务冲突

如果32、48和64源实验都从同一个reward最优动作直接进入动作失效区，则停止
继续插值，改为固定路径和期望字节负载、改变流大小分布：

- Task A `samepath32_hadoop_long`：Hadoop长流分布；
- Task B `samepath32_alistorage_short`：AliStorage短流分布；
- 两者均使用0–31号源、目的端128和相同的1.6699秒窗口；
- Hadoop周期为0.055665秒，AliStorage周期为0.004秒；周期按CDF均值成比例
  设置，因此每个源的期望字节速率相同；
- 两边启用分层CDF分位数采样，降低有限样本下重尾流量的实际字节率偏差；
- 两个任务的端点支持相同，但流大小与流数量按设计不同。

先运行两个任务 × 三个动作，共六次冻结仿真：

```bash
CAL_ID=acc_cdf32_calibration_s1
CAL_ARGS=(
  --run-id "$CAL_ID"
  --seed 1
  --buffer-kb 400
  --spread-profile cdf32
  --watch-ports all
  --port 6256
  --reward-profile tail_safe
  --reward-queue-lambda 5.0
  --reward-ecn-lambda 5.0
  --reward-weights "0.50,0.30,0.20"
  --acc-hidden-dims "32,64,64,32"
)

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage prepare "${CAL_ARGS[@]}"

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage run "${CAL_ARGS[@]}" \
  > "experiments/continual_validation/${CAL_ID}_driver.log" 2>&1 &

echo $! > "experiments/continual_validation/${CAL_ID}_driver.pid"
tail -f "experiments/continual_validation/${CAL_ID}_driver.log"
```

完成后：

```bash
bash scripts/continual_validation/run_spread_calibration.sh \
  --stage analyze "${CAL_ARGS[@]}"

CAL_DIR="experiments/continual_validation/${CAL_ID}/calibration"
cat "${CAL_DIR}/CALIBRATION_REPORT.md"
```

该模式预期显示`Same flow-identity multiset: False`，但必须显示
`Same endpoint support: True`和`Pair mode: workload-shift`。这不是校验失败：
正式遗忘比较仍分别使用冻结的Task A文件做after-A/after-B对照。

若生成`recommended_pair.env`，正式连续训练应使用：

```bash
--task-pair-mode workload-shift
```

#### 在原CDF运行上增量补齐3×3动作网格

三动作对角筛选没有找到策略冲突时，不重新生成流量，也不重复已有动作。使用
同一个run-id并显式传入`--resume`，将manifest从`diagonal`安全升级为
`factorial3`。运行器会复用`low_strong`、`mid`和`high_gentle`，每个任务只补
六个动作：

```bash
CAL_ID=acc_cdf32_calibration_s1
CAL_ARGS=(
  --run-id "$CAL_ID"
  --seed 1
  --buffer-kb 400
  --spread-profile cdf32
  --action-grid factorial3
  --watch-ports all
  --port 6256
  --reward-profile tail_safe
  --reward-queue-lambda 5.0
  --reward-ecn-lambda 5.0
  --reward-weights "0.50,0.30,0.20"
  --acc-hidden-dims "32,64,64,32"
)

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage run "${CAL_ARGS[@]}" --resume \
  > "experiments/continual_validation/${CAL_ID}_factorial3.log" 2>&1 &

echo $! > "experiments/continual_validation/${CAL_ID}_factorial3.pid"
tail -f "experiments/continual_validation/${CAL_ID}_factorial3.log"
```

完成后确认应有18个结果，再重新分析：

```bash
CAL_DIR="experiments/continual_validation/${CAL_ID}/calibration"

find "${CAL_DIR}/runs" -name metrics.json | wc -l

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage analyze "${CAL_ARGS[@]}" --resume

cat "${CAL_DIR}/CALIBRATION_REPORT.md"
column -s, -t < "${CAL_DIR}/calibration_summary.csv" | less -S
```

### 3.6 空间与时间联合冲突：均衡持续流 → 热点同步突发

仅改变CDF的任务仍可能选择同一个动作。`conflict32`进一步构造一个最大对比、
但端点支持严格受控的任务对：

- Task A `samepath32_balanced_steady`：32个源以25%周期spread向4个接收端
  均匀发送；
- Task B `samepath32_hotspot_burst`：相同源、接收端、CDF、周期、持续时间和
  期望总字节率，流量在周期起点同步，并以`13:1:1:1`集中到一个热点接收端；
- 每个源在最初4个周期显式覆盖全部4个目的端，因此两个任务的
  source/destination endpoint support完全相同；
- 两边都使用分层AliStorage CDF采样，期望聚合流量为12 Gbps；变化只来自
  空间偏斜和到达同步；
- 推荐使用同一个tail-safe目标的`queue_lambda=30`、`ecn_lambda=18.5`，放大
  热点突发的队列/ECN代价，而不是为两个任务使用不同reward。

先运行两个任务 × 九个动作：

```bash
CAL_ID=acc_conflict32_calibration_s1
CAL_ARGS=(
  --run-id "$CAL_ID"
  --seed 1
  --buffer-kb 400
  --spread-profile conflict32
  --action-grid factorial3
  --watch-ports all
  --port 6256
  --reward-profile tail_safe
  --reward-queue-lambda 30.0
  --reward-ecn-lambda 18.5
  --reward-weights "0.50,0.30,0.20"
  --acc-hidden-dims "32,64,64,32"
)

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage prepare "${CAL_ARGS[@]}"

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage run "${CAL_ARGS[@]}" \
  > "experiments/continual_validation/${CAL_ID}_driver.log" 2>&1 &

echo $! > "experiments/continual_validation/${CAL_ID}_driver.pid"
tail -f "experiments/continual_validation/${CAL_ID}_driver.log"
```

完成后运行分析：

```bash
CAL_DIR="experiments/continual_validation/${CAL_ID}/calibration"

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage analyze "${CAL_ARGS[@]}"

cat "${CAL_DIR}/CALIBRATION_REPORT.md"
column -s, -t < "${CAL_DIR}/port_action_summary.csv" | less -S
```

除网络级动作与p95双向代价外，报告新增`Shared-port reward conflicts`。只有同一
交换机端口在A/B任务中选择不同reward最优动作，并且互换动作在两个方向上都
至少损失3% reward，才会进入`RECOMMENDED_WATCH_PORTS`。这避免把“端口只是
活跃/拥塞”误当成“端口存在可学习的任务冲突”。

### 3.7 降低热点强度：复用Task A，只补跑三档Task B

`13:1:1:1`热点任务的最低完成率约为80%，过载掩盖了动作差异。下一轮固定
Task A、拓扑、CDF、总发送周期、总流数和reward，只将Task B的目的端权重依次
降为`9:1:1:1`、`7:1:1:1`和`5:1:1:1`。分析仅比较第一个候选Task A与三档
Task B，不会把两个热点任务错误地选成A/B任务对。

沿用上一轮`acc_conflict32_calibration_s1`，准备阶段会安全地把manifest升级为
`conflict32grid`，保留已冻结的Task A及其九个动作结果。旧的13:1结果继续留在
磁盘中作为审计记录，但不参与本轮分析；服务器只需新增3个任务 × 9个动作，
共27次仿真。

```bash
CAL_ID=acc_conflict32_calibration_s1
CAL_ARGS=(
  --run-id "$CAL_ID"
  --seed 1
  --buffer-kb 400
  --spread-profile conflict32grid
  --action-grid factorial3
  --watch-ports all
  --port 6256
  --reward-profile tail_safe
  --reward-queue-lambda 30.0
  --reward-ecn-lambda 18.5
  --reward-weights "0.50,0.30,0.20"
  --acc-hidden-dims "32,64,64,32"
)

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage prepare "${CAL_ARGS[@]}" --resume

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage run "${CAL_ARGS[@]}" --resume \
  > "experiments/continual_validation/${CAL_ID}_hotspot_grid.log" 2>&1 &

echo $! > "experiments/continual_validation/${CAL_ID}_hotspot_grid.pid"
tail -f "experiments/continual_validation/${CAL_ID}_hotspot_grid.log"
```

进度只统计当前manifest中的四个任务，期望最终得到36个结果（复用A的9个，
新增B的27个）：

```bash
CAL_DIR="experiments/continual_validation/${CAL_ID}/calibration"

for scenario in \
  samepath32_balanced_steady \
  samepath32_hotspot9_burst \
  samepath32_hotspot7_burst \
  samepath32_hotspot5_burst; do
  find "${CAL_DIR}/runs/${scenario}" -name metrics.json 2>/dev/null
done | wc -l

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage analyze "${CAL_ARGS[@]}" --resume

cat "${CAL_DIR}/CALIBRATION_REPORT.md"
if test -f "${CAL_DIR}/recommended_pair.env"; then
  cat "${CAL_DIR}/recommended_pair.env"
fi
```

选择顺序是：先满足完成率不低于90%，再检查网络级reward/p95敏感性、Task A与
Task B的双向p95代价，最后检查共享交换机端口上的reward最优动作是否改变。
reward最优动作与单独的p95最优动作不要求完全相同，因为pair gate已经直接测量
两个reward最优动作互换时的p95代价；这也避免上一轮Task A永久阻断后续候选。
三档均不满足时，应转向调整到达结构或负载，而不是直接启动长时间ACC训练。

## 4. 使用推荐任务对做正式筛选

以下命令只有在 `recommended_pair.env` 存在时执行：

```bash
source "${CAL_DIR}/recommended_pair.env"

CAL_PAIR_MODE="$(python - "${CAL_DIR}/calibration_manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    mode = json.load(handle).get("pair_mode", "timing-only")
print("same-flows" if mode == "timing-only" else mode)
PY
)"

RUN_ID=acc_forgetting_calibrated_s1
COMMON_ARGS=(
  --run-id "$RUN_ID"
  --task-a "$RECOMMENDED_TASK_A"
  --task-b "$RECOMMENDED_TASK_B"
  --seed 1
  --buffer-kb 400
  --action-space multiscale
  --task-pair-mode "$CAL_PAIR_MODE"
  --reward-profile tail_safe
  --reward-queue-lambda 5.0
  --reward-ecn-lambda 5.0
  --reward-weights "0.50,0.30,0.20"
  --shared-replay false
  --updates-task-a 900
  --updates-task-b 1800
  --phase-epochs 180
  --eps-decay 2500
  --task-b-eps-start 0.50
  --task-b-eps-decay 6000
  --acc-hidden-dims "32,64,64,32"
  --screen-watch-ports "$RECOMMENDED_WATCH_PORTS"
  --screen-min-active-samples 50
  --screen-min-congested-samples 20
  --screen-min-old-task-p95-penalty 0.03
  --port 6156
)

bash scripts/continual_validation/run_continual.sh \
  --stage prepare "${COMMON_ARGS[@]}"

bash scripts/continual_validation/run_continual.sh \
  --stage screen "${COMMON_ARGS[@]}" --report-only

cat "experiments/continual_validation/${RUN_ID}/screen/REPORT.md"
```

完整的五动作筛选满足条件后，复用输出应用 gate：

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage screen "${COMMON_ARGS[@]}"
```

## 5. 正式运行 ACC A→B

```bash
nohup bash scripts/continual_validation/run_continual.sh \
  --stage acc "${COMMON_ARGS[@]}" --report-only \
  > "experiments/continual_validation/${RUN_ID}/acc_driver.log" 2>&1 &

echo $! > "experiments/continual_validation/${RUN_ID}/acc_driver.pid"
tail -f "experiments/continual_validation/${RUN_ID}/acc_driver.log"
```

完成后查看：

```bash
cat "experiments/continual_validation/${RUN_ID}/CONTINUAL_REPORT.md"
```
