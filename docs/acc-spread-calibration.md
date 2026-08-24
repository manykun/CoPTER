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

## 4. 使用推荐任务对做正式筛选

以下命令只有在 `recommended_pair.env` 存在时执行：

```bash
source "${CAL_DIR}/recommended_pair.env"

RUN_ID=acc_forgetting_calibrated_s1
COMMON_ARGS=(
  --run-id "$RUN_ID"
  --task-a "$RECOMMENDED_TASK_A"
  --task-b "$RECOMMENDED_TASK_B"
  --seed 1
  --buffer-kb 400
  --action-space multiscale
  --task-pair-mode same-flows
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
