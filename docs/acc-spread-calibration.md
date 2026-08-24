# ACC 灾难性遗忘：时序突发度校准实验

## 1. 为什么先校准

`samepath_steady_stress → samepath_burst_stress` 的首次固定动作筛选表明：

- steady 在端口 323 上没有拥塞样本；
- burst 在端口 323 上持续拥塞，但不同动作的 reward 和 p95 完全相同；
- 两个极端分别落入“无控制信号区”和“动作失效的饱和区”。

因此不能直接开始长训练。本实验保持 1920 条流的
`(src,dst,pg,dport,size)` 多重集合完全不变，只测试四个中间到达分散度：

| 场景 | spread_fraction |
|---|---:|
| samepath_spread070_stress | 0.70 |
| samepath_spread050_stress | 0.50 |
| samepath_spread030_stress | 0.30 |
| samepath_spread010_stress | 0.10 |

每档只运行 `low_strong`、`mid`、`high_gentle` 三个固定动作，共 12 次冻结
仿真，不更新网络或 replay。

## 2. 运行校准

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
cd /mnt/sdb1/xuduokun/projects/CoPTER

CAL_ID=acc_spread_calibration_s1
mkdir -p experiments/continual_validation

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage all \
  --run-id "$CAL_ID" \
  --seed 1 \
  --buffer-kb 400 \
  --watch-port 323 \
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
CAL_ID=acc_spread_calibration_s1

nohup bash scripts/continual_validation/run_spread_calibration.sh \
  --stage run \
  --run-id "$CAL_ID" \
  --seed 1 \
  --buffer-kb 400 \
  --watch-port 323 \
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
CAL_ID=acc_spread_calibration_s1

bash scripts/continual_validation/run_spread_calibration.sh \
  --stage analyze \
  --run-id "$CAL_ID" \
  --seed 1 \
  --buffer-kb 400 \
  --watch-port 323 \
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
- 端口 323 至少活跃 50 步、拥塞 20 步；
- reward 最优动作和完成率安全的 p95 最优动作一致；
- A、B 选择不同动作；
- B 动作施加到 A、A 动作施加到 B 时，p95 代价都不低于 3%。

若生成了推荐文件：

```bash
source "${CAL_DIR}/recommended_pair.env"
echo "Task A: ${RECOMMENDED_TASK_A}"
echo "Task B: ${RECOMMENDED_TASK_B}"
```

如果报告为 `Recommended pair: NONE`，停止长训练，并保留报告。此时说明仅调整
时间突发度仍不能构造动作冲突，下一步应改变共享瓶颈的空间路径，而不是继续
增加训练轮数。

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
  --screen-watch-ports "323"
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
