# ACC 学习、奖励一致性与遗忘：短预算实验

## 三个问题分别如何验证

| 问题 | 主要证据 | 不能作为充分证据 |
|---|---|---|
| A/B 是否学好了，是否接近平台 | 稀疏检查点的冻结 greedy FCT、完成率、reward；与初始化和预先固定参数比较；相邻检查点增益 | loss 小；只挑最好一轮；随机初始化比训练模型差 |
| 奖励是否误导方向 | 相同轨迹用 weighted 与 tail-safe 双评分；两套模型从相同随机初始化分别训练；与 FCT、完成率方向对照 | 两个不同公式的绝对 reward 高低；ECN 越少一定越好 |
| 切换是否导致遗忘、能否恢复 | 在相同 A 流量测 after-A、AB、AA、return-A；比较同任务动作分布；测有限预算内恢复 | A 和 B 训练 reward 直接相减；loss 尖峰；权重漂移；有限预算未恢复等于永不可恢复 |

## 固定条件与预算

- 基于 `real_web_cache_target100_s1` 的 manifest 和两份原始任务输入，绝不读取旧训练权重作为新初始化。
- WebServer→CacheFollower；seed 1；交换机 buffer 400 KB（继承源 manifest，不随场景改）；multiscale 动作空间。
- 关闭 global replay，保留正常 local replay；epsilon 全程 global；target 更新周期继承原 manifest（该基准为 100 optimizer updates）。
- 两套模型唯一主动更改的学习目标为奖励：tail_safe 与 weighted(0.50,0.30,0.20)。lambda、其余训练配置保持不变。
- 初始网络哈希、空回放、零训练计数都检查。AA/AB 复制同一个 after-A 的完整 policy/target/optimizer/RNG/replay；return 复制完整 after-B。
- 固定参考为所有端口 action `4,3`。它不是 SECN，也不是最优参数；它是预注册的中间档基准。normalized Kmin/Kmax 与 Pmax 由现有 multiscale 映射解释，物理 KB 依赖配置和链路速率，报告不猜测物理值。

| 阶段 | 更新节点 | 冻结测评 | 每套奖励训练开销 |
|---|---|---|---:|
| acquire | A: 0、100、300、600 | A 每个节点；600 时也测 B | 600 |
| continue | AA: +100、+300 | A | 300 |
| continue | AB: +100、+300 | A、B | 300 |
| continue | return A: +50、+100 | A、B | 100 |

两套奖励合计：2600 updates，32 次冻结模拟（含共享固定参考的两次）；acquire 先只做 1200 updates、12 次冻结模拟。
不做相同 seed 的三遍重复，不搜索任务对。通过 `--jobs 2` 并行两套奖励；每套内部按依赖顺序执行，独立 socket、配置输出、模型和日志。
并行只减少可能的墙钟时间，不减少总计算量。若服务器 CPU/RAM 不够，`--jobs 1` 更合适；resume 可改变 jobs。

## 执行

在服务器项目根目录，用 tmux 保持会话：

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
git pull --ff-only origin exp/acc-validation

BASE=experiments/continual_validation/real_web_cache_target100_s1
RUN=experiments/continual_validation/real_web_cache_learning_study_s1
test -f "$BASE/manifest.json" || exit 1
df -h .
free -h
ss -ltnp | grep -E ':(7156|7157)([[:space:]]|$)'

python scripts/continual_validation/run_learning_study.py \
  --base-run-dir "$BASE" --output-dir "$RUN" \
  --stage acquire --jobs 2 --port 7156

cat "$RUN/acquisition_report/REPORT.md"
```

端口检查有占用时先选择另一对连续端口，并在后续命令保持相同 `--port`。无匹配时 grep 返回 1 是正常的空结果，不是启动失败。
保留完整 local replay 快照会占用磁盘；不要在空间不足时运行，不自动删除旧结果。

### acquire 后怎样决策

1. 两套奖励分别检查 A 初始→600 是否改善物理表现（不牺牲完成率），再看 100→300、300→600 是否逐渐平台。
2. reward 提升但 FCT 变差时，查看双评分与 reward_components、物理端口数据；不要称为网络改善。
3. 若仍有明显学习趋势，先延长 A，不急于将“尚未学好”误判为遗忘。若平台但性能不佳，说明当前预算/优化目标下学习不足，不能用后续遗忘证明其已学会。
4. 当前流程不自动挑选最有利奖励、端口、检查点或删掉负结果。

如需延长 A，可复用已完成训练（必须在任何 continue 分支创建之前）：

```bash
python scripts/continual_validation/run_learning_study.py \
  --base-run-dir "$BASE" --output-dir "$RUN" \
  --stage acquire --jobs 2 --port 7156 --resume --extend-a \
  --a-points 100,300,600,900,1200
```

延长后所有命令都需带相同 `--a-points 100,300,600,900,1200`；旧协议保存在 protocol_before_A_600.json，旧节点保留。禁止在看过 AB 结果后更换 after-A 起点。

### 任务切换与恢复

默认未延长时：

```bash
python scripts/continual_validation/run_learning_study.py \
  --base-run-dir "$BASE" --output-dir "$RUN" \
  --stage continue --jobs 2 --port 7156

cat "$RUN/study_report/REPORT.md"
```

若已事先确定执行全部节点，可以一开始用 `--stage all`，但不会自动判断 A 是否学好。
中断恢复：原命令增加 `--resume`；不会按超时猜测失败后自动重启。缺失 replay、改变输入、预算不匹配、同输出重复启动都会拒绝。

只重新分析，不训练：

```bash
python scripts/continual_validation/run_learning_study.py \
  --base-run-dir "$BASE" --output-dir "$RUN" --stage analyze --port 7156
```

## 输出与读法

| 文件（位于 acquisition_report 或 study_report） | 内容 |
|---|---|
| REPORT.md、comparisons.csv | A/B 学习、相邻节点增益、同预算 AB−AA、return-A 残留退化与恢复比例 |
| dual_scores.csv | 同一轨迹/同一行人群的两种奖励；跨检查点人群仍可能变化 |
| action_changes.csv | 同任务动作分布 TVD、主导动作占比、归一化 Kmin/Kmax、Pmax；含样本数 |
| frozen_curves.png | 固定任务冻结 reward、p95 曲线，不能跨奖励公式比较数值高低 |
| attribution_report/training_pPORT.png | batch reward、在线全网 reward、TD loss；不是冻结 reward |
| attribution_report/network.csv | 同任务共同完成流 FCT 与全 offered completion、匹配流数 |
| attribution_report/physical_ports.csv | 端口 queue、ECN、PFC；缺失 PFC 保留 n/a |

TVD 是边际动作分布差异，不是同一状态集合上的模型差异；状态访问分布会随模型变化。0 为分布相同，1 为不相交，不自动意味着显著遗忘。
返回 A 的恢复比例仅在此前该指标确实退化时定义。reward 恢复而 FCT 未恢复时分别报告。AA 本身的退化不应全部归因于任务切换。
未在 100 updates 内恢复只能说明有限预算下恢复不足；不能声称永久不可恢复。单 seed 只支持机制探索，不支持跨流量分布统计泛化。各 episode 仍重启模拟器，不能称为物理队列不中断的在线切换。
最终全网结论依赖网络指标；六个端口不能替代全网，也不要求每个端口同时改善。

## 开发验证

`python -m unittest test_learning_study`（在 scripts/continual_validation 目录）测试模拟调用的完整 acquire→AA/AB→return、完整回放复制、精确预算、初始化、resume、缺失矩阵与数值语义。
这是 simulator-free 测试，不意味着服务器上的 NS-3 实验已经执行或已经证明遗忘。
