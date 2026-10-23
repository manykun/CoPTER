# ACC单端口局部遗忘实验

本实验不再搜索新任务对。固定使用完成率稳定的任务：

- Task A：`samepath32_balanced_steady`
- Task B：`samepath32_hotspot5_burst`
- 主端口：323
- 初始负对照端口：321
- 后续复现端口：320、345、346、347

实验分为ACC持续训练和冻结单端口连续插值两部分。训练阶段记录每端口DDQN
loss；冻结阶段记录reward、队列、ECN、PFC和全网FCT。冻结阶段没有优化器更新，
因此不报告loss。

## 1. 更新和编译

`copter-sim.cc`的PFC输出增加了第六列`peer_node_id`，用于把PFC事件精确映射到
OpenGym的`switch-peer`端口，因此更新后必须重新编译ns-3。

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
conda activate /mnt/sdb1/xuduokun/conda/envs/m3

git fetch git@github.com:manykun/CoPTER.git exp/acc-validation
git merge --ff-only FETCH_HEAD

bash build_ns3_copter.sh
```

## 2. ACC局部遗忘训练

关闭global replay以放大端口局部干扰。该设置是明确的消融实验，不代表ACC默认
配置。使用`--report-only`保留所有测量，不受原网络级任务对gate阻断。

```bash
RUN_ID=acc_local_b5_s1
WATCH_PORTS="323,321,320,345,346,347"

COMMON_ARGS=(
  --run-id "$RUN_ID"
  --task-a samepath32_balanced_steady
  --task-b samepath32_hotspot5_burst
  --task-pair-mode workload-shift
  --seed 1
  --buffer-kb 400
  --action-space multiscale
  --shared-replay false
  --updates-task-a 900
  --updates-task-b 1800
  --phase-epochs 180
  --eps-decay 2500
  --task-b-eps-start 0.50
  --task-b-eps-decay 6000
  --acc-hidden-dims "32,64,64,32"
  --reward-profile tail_safe
  --reward-queue-lambda 30.0
  --reward-ecn-lambda 18.5
  --reward-weights "0.50,0.30,0.20"
  --screen-watch-ports "$WATCH_PORTS"
  --screen-min-active-samples 50
  --screen-min-congested-samples 20
  --port 6456
)

bash scripts/continual_validation/run_continual.sh \
  --stage prepare "${COMMON_ARGS[@]}"

bash scripts/continual_validation/run_continual.sh \
  --stage screen "${COMMON_ARGS[@]}" --report-only
```

正式训练建议后台执行：

```bash
nohup bash scripts/continual_validation/run_continual.sh \
  --stage acc "${COMMON_ARGS[@]}" --report-only \
  > "experiments/continual_validation/${RUN_ID}_acc.log" 2>&1 &

echo $! > "experiments/continual_validation/${RUN_ID}_acc.pid"
tail -f "experiments/continual_validation/${RUN_ID}_acc.log"
```

中断后使用完全相同的参数恢复：

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage acc "${COMMON_ARGS[@]}" --report-only --resume
```

训练完成后生成端口报告：

```bash
python scripts/continual_validation/analyze_port_continual.py \
  --run-dir "experiments/continual_validation/${RUN_ID}" \
  --ports "$WATCH_PORTS"

cat "experiments/continual_validation/${RUN_ID}/PORT_CONTINUAL_REPORT.md"
ls "experiments/continual_validation/${RUN_ID}"/port_training_p*.png
```

详细表格：

```bash
column -s, -t < \
  "experiments/continual_validation/${RUN_ID}/PORT_TRAINING_SUMMARY.csv" | less -S

column -s, -t < \
  "experiments/continual_validation/${RUN_ID}/PORT_EVAL_SUMMARY.csv" | less -S
```

报告分别比较after A与after B冻结回测Task A时的：

- 拥塞期间端口reward和未裁剪tail-safe reward；
- 队列均值、p95、峰值；
- ECN平均值、p95、峰值和非零比例；
- 精确物理端口的PFC pause次数、总时长、duty和最长pause；
- 每端口训练loss首值、末值、中位数、峰值；
- 全网完成率和p95 FCT。

## 3. 0.1步长单端口连续路径扫描

插值端点自动取自既有校准结果中端口323在Task A和Task B上的拥塞reward最优
动作。after-B网络冻结，其他447个端口保持greedy，只覆盖被测端口。相同路径也
施加到端口321作为负对照。

```bash
PATH_ARGS=(
  --base-run-id acc_local_b5_s1
  --calibration-run-id acc_conflict32_calibration_s1
  --endpoint-port 323
  --target-ports "323,321"
  --alpha-min 0.0
  --alpha-max 1.0
  --alpha-step 0.1
  --link-gbps 40
  --port 6556
)

bash scripts/continual_validation/run_port_path_sweep.sh \
  --stage prepare "${PATH_ARGS[@]}"

nohup bash scripts/continual_validation/run_port_path_sweep.sh \
  --stage sweep "${PATH_ARGS[@]}" \
  > "experiments/continual_validation/acc_local_b5_s1_port_path.log" 2>&1 &

echo $! > \
  "experiments/continual_validation/acc_local_b5_s1_port_path.pid"

tail -f \
  "experiments/continual_validation/acc_local_b5_s1_port_path.log"
```

初始扫描为2个端口 × 11个参数点 × 2个任务，共44次冻结仿真。完成后：

```bash
bash scripts/continual_validation/run_port_path_sweep.sh \
  --stage analyze "${PATH_ARGS[@]}"

REPORT="experiments/continual_validation/acc_local_b5_s1/port_path_sweep_p323"
cat "${REPORT}/PORT_PATH_REPORT.md"
column -s, -t < "${REPORT}/port_path_summary.csv" | less -S
ls "${REPORT}"/port_path_p*.png
```

`port_training_p*.png`展示每个端口跨A→B训练阶段的batch reward与DDQN TD-loss；
`port_path_p*.png`展示冻结策略下reward、队列p95、ECN标记率和PFC pause duty随
插值系数alpha的变化。

## 4. 只在曲线交叉附近细化

假设初始报告显示交叉区间为`alpha=0.4～0.6`，改用0.02步长：

```bash
REFINE_ARGS=(
  --base-run-id acc_local_b5_s1
  --calibration-run-id acc_conflict32_calibration_s1
  --endpoint-port 323
  --target-ports "323,321"
  --alpha-min 0.4
  --alpha-max 0.6
  --alpha-step 0.02
  --link-gbps 40
  --port 6556
)

bash scripts/continual_validation/run_port_path_sweep.sh \
  --stage prepare "${REFINE_ARGS[@]}"

bash scripts/continual_validation/run_port_path_sweep.sh \
  --stage sweep "${REFINE_ARGS[@]}"

bash scripts/continual_validation/run_port_path_sweep.sh \
  --stage analyze "${REFINE_ARGS[@]}"
```

已有相同alpha结果会自动复用。若323在A/B上的reward最优alpha不同，并伴随队列、
ECN或PFC变化，而321基本稳定，则支持真实的单端口任务冲突。若两端口曲线都近似
水平或同方向变化，应停止细化；这说明动作离散度不是遗忘不明显的主要原因。

## 5. 复现端口

323出现明确冲突后，再将`--endpoint-port`和`--target-ports`分别改为320、345、
346或347。每次使用不同socket端口并保留独立输出目录。至少两个目标端口呈现相同
方向，才能把结论从个例提升为路径局部机制证据。
