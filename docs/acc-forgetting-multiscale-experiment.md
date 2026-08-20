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

任务 A 为 `samepath_steady`，任务 B 为 `samepath_burst`。两者使用相同 seed
时具有完全相同的 640 个 `(src,dst,size)` 流，只改变开始时间：

- steady：每个周期内将 32 个 incast 流均匀铺开；
- burst：将同一批流聚集在周期起点附近。

因此 A→B 的主要处理变量是流量时间突发性，而不是路径、流大小分布或流数。

## 4. 服务器实验顺序

在项目根目录执行：

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3

RUN_ID=acc_multiscale_samepath_s1
COMMON_ARGS=(
  --run-id "$RUN_ID"
  --task-a samepath_steady
  --task-b samepath_burst
  --seed 1
  --buffer-kb 400
  --action-space multiscale
  --reward-profile tail_safe
  --reward-queue-lambda 5.0
  --reward-ecn-lambda 5.0
  --reward-weights "0.50,0.30,0.20"
  --shared-replay false
  --updates-per-task 900
  --phase-epochs 80
  --eps-decay 2500
  --acc-hidden-dims "32,64,64,32"
  --report-only
)
```

### 阶段 1：生成冻结任务

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage prepare "${COMMON_ARGS[@]}"
```

核对两个任务除了时间列之外具有相同流集合：

```bash
python - "$RUN_ID" <<'PY'
import sys
from pathlib import Path

root = Path("experiments/continual_validation") / sys.argv[1] / "tasks"

def flows(task):
    rows = (root / task / "input.flow").read_text().splitlines()[1:]
    return sorted(tuple(row.split()[:5]) for row in rows)

a = flows("samepath_steady")
b = flows("samepath_burst")
print("flows A/B:", len(a), len(b))
print("same src/dst/size multiset:", a == b)
PY
```

预期输出为 `640 640` 和 `True`。

### 阶段 2：固定动作冲突筛选

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage screen "${COMMON_ARGS[@]}"

cat "experiments/continual_validation/${RUN_ID}/screen/REPORT.md"
```

筛选固定测试 5 个代表动作，共 2 个任务 × 5 次冻结仿真。进入训练前至少检查：

- 两任务的最佳 reward 动作不同；
- p95 或完成率随动作发生可测变化；
- reward 最优动作没有靠牺牲大量完成率取胜。

若两个任务仍选择同一个最优动作，应先保留完整负结果并停止长训练；此时说明
任务冲突仍不足，而不是 ACC 没有遗忘。

### 阶段 3：ACC 顺序训练

只有阶段 2 确认存在任务冲突后运行：

```bash
nohup bash scripts/continual_validation/run_continual.sh \
  --stage acc "${COMMON_ARGS[@]}" \
  > "experiments/continual_validation/${RUN_ID}/acc_driver.log" 2>&1 &

tail -f "experiments/continual_validation/${RUN_ID}/acc_driver.log"
```

结束后查看：

```bash
cat "experiments/continual_validation/${RUN_ID}/CONTINUAL_REPORT.md"
```

核心比较为同一冻结 A 流量上的 `after_a` 与 `after_b`：reward 下降、共同完成流
p95 上升和完成率下降分别报告，不用单一阈值隐藏原始效果量。

## 5. 结论边界

本轮 seed=1 用于机制构造和代码验证。即使观察到明显遗忘，也只能表述为受控
基准上的机制证据；在增加独立 seed 前不能宣称统计泛化。静态冲突筛选、任务
习得和旧任务退化三者必须同时成立，才能将退化解释为灾难性遗忘。
