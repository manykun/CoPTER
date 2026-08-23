# ACC 多尺度动作空间与灾难性遗忘实验

## 1. 目的

本实验先解决两个已经由端口数据暴露的问题，再进行 ACC 顺序学习：

1. 原 `Kmin/Kmax` 独立动作头可能组合出不合理阈值；
2. 40 Gbps 端口的实测峰值队列可低至 24.56 KB，而旧动作空间的最小
   `Kmin` 为 32 KB，这些端口不会产生 ECN 动作差异。

实验不是通过反复调整阈值追求某次退化，而是预先固定动作空间、流量和评价
协议。只有静态动作筛选先证明两个任务对动作的偏好不同，才进入 A→B 训练。

## 2. 新动作空间

`--action-space multiscale` 使用两个离散输出头：

- Profile 头：选择一个合法 `(Kmin,Kmax)` 配对；
- Pmax 头：选择标记概率。

配置基准范围固定为 `Kmin=[5,50] KB`、`Kmax=[15,100] KB`。ns-3 按链路
速率缩放后，40 Gbps 物理 Profile 为：

| Profile | Kmin (KB) | Kmax (KB) |
|---:|---:|---:|
| 0 | 8 | 24 |
| 1 | 16 | 32 |
| 2 | 20 | 40 |
| 3 | 24 | 48 |
| 4 | 32 | 64 |
| 5 | 48 | 96 |
| 6 | 57 | 110 |
| 7 | 63 | 120 |
| 8 | 80 | 160 |

Pmax 为 `0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00`。因此共有 63 个
合法动作，模型不会产生 `Kmin>=Kmax`。

## 3. 受控任务

正式压力实验使用任务 A `samepath_steady_stress` 和任务 B
`samepath_burst_stress`。两者使用相同 seed 时具有完全相同的 1920 个
`(src,dst,pg,dport,size)` 流，只改变开始时间：

- steady：每个周期内将 64 个 incast 流均匀铺开；
- burst：将同一批 64 个流聚集在周期起点附近。

因此 A→B 的主要处理变量是流量时间突发性，而不是路径、流大小分布或流数。
`--task-pair-mode same-flows` 会在 prepare 阶段自动按多重集合校验流身份，并
生成 `TASK_PAIR_REPORT.md`；不再依赖人工抽查。

训练前还要求端口 323（物理链路 `288-295`）在两个任务中均活跃并出现拥塞，
同时要求“任务 B 偏好动作”施加到任务 A 时至少带来 3% 的 p95 代价。这样长
训练只在共享瓶颈和方向性动作冲突都已被实测确认后启动。

## 4. 服务器实验顺序

在项目根目录执行：

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3

RUN_ID=acc_forgetting_stress_s1
COMMON_ARGS=(
  --run-id "$RUN_ID"
  --task-a samepath_steady_stress
  --task-b samepath_burst_stress
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
)
```

### 阶段 1：生成冻结任务

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage prepare "${COMMON_ARGS[@]}"
```

查看自动生成的任务对校验：

```bash
cat "experiments/continual_validation/${RUN_ID}/TASK_PAIR_REPORT.md"
```

预期为 1920/1920、流身份相同、时间分布不同；burst 的 1 us 窗口峰值流数
和 interarrival CV 应显著高于 steady。

### 阶段 2：固定动作冲突筛选

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage screen "${COMMON_ARGS[@]}" --report-only

cat "experiments/continual_validation/${RUN_ID}/screen/REPORT.md"
```

筛选固定测试 5 个代表动作，共 2 个任务 × 5 次冻结仿真。进入训练前至少检查：

- 两任务的最佳 reward 动作不同；
- p95 或完成率随动作发生可测变化；
- reward 最优动作没有靠牺牲大量完成率取胜。
- 端口 323 在两个任务的固定动作测试中均满足活跃/拥塞样本要求；
- B 偏好动作在 A 上造成至少 3% 的 p95 代价。

若描述性结果满足以上条件，用相同输出重新应用注册 gate（已有 10 次仿真结果
会被复用，不会重跑）：

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage screen "${COMMON_ARGS[@]}"
```

若 gate 失败，应保留完整负结果并停止长训练；此时说明任务冲突或共享路径仍
不足，而不是 ACC 没有遗忘。

### 阶段 3：ACC 顺序训练

只有阶段 2 确认存在任务冲突后运行：

```bash
nohup bash scripts/continual_validation/run_continual.sh \
  --stage acc "${COMMON_ARGS[@]}" --report-only \
  > "experiments/continual_validation/${RUN_ID}/acc_driver.log" 2>&1 &

tail -f "experiments/continual_validation/${RUN_ID}/acc_driver.log"
```

结束后查看：

```bash
cat "experiments/continual_validation/${RUN_ID}/CONTINUAL_REPORT.md"
```

核心比较为同一冻结 A 流量上的 `after_a` 与 `after_b`：reward 下降、共同完成流
p95 上升和完成率下降分别报告，不用单一阈值隐藏原始效果量。这里保留
`--report-only`，避免旧的通用 acquisition gate 在压力场景中提前终止 B 阶段；
静态冲突筛选仍已在上一阶段单独应用 gate。

其中 A 使用 900 次优化更新，B 使用额外 1800 次更新。B 阶段将 phase-local
epsilon 重置到 0.50，并在 6000 个环境步内衰减到 0.05；全局模型、训练步和
本地 replay 连续保留。此非对称预算用于增加新任务覆盖，但没有改变两个任务
的奖励函数、动作空间或评估流量。

## 5. 结论边界

本轮 seed=1 用于机制构造和代码验证。即使观察到明显遗忘，也只能表述为受控
基准上的机制证据；在增加独立 seed 前不能宣称统计泛化。静态冲突筛选、任务
习得和旧任务退化三者必须同时成立，才能将退化解释为灾难性遗忘。
