# 原weighted与tail-safe奖励的两阶段对照

先在同一冻结轨迹上评分，再决定是否训练。原公式指当前代码中的
`weighted = 0.50*throughput + 0.30*queue + 0.20*ecn`，其中queue/ecn是
原有变换后的奖励分量，不是原始队列长度/ECN率。本轮不恢复旧target同步错误、
不重置任务边界epsilon，也不启用global replay。

## 1. 更新与离线双评分

在服务器项目根目录执行；不需要新仿真，不改已有实验。

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
git pull --ff-only origin exp/acc-validation
BASE=experiments/continual_validation/real_web_cache_target100_s1
SOURCE="${BASE}_aa_ab_reset_v1/attribution_report"
SCORES="${BASE}_reward_rescore_v1"
NEW="${BASE}_reward_pair_v1"

python scripts/continual_validation/rescore_rewards.py \
  --report-dir "$SOURCE" --output-dir "$SCORES" \
  --recorded-profile tail_safe --weights 0.50,0.30,0.20
cat "$SCORES/REPORT.md"
```

输出dual_scores.csv，包含同一端口/样本群的两种reward及全网p95、完成率背景。
查看321、347时，同时对照SOURCE下physical_ports.csv的队列和ECN；CSV中的
network_p95_fct_us始终是全网流指标，不是该端口的FCT。
原reward是否更好应看评价方向是否更贴合FCT、队列和完成率，不能比较两种reward
绝对大小，也不能按是否放大遗忘选择奖励。离线评分不推断重新训练后的策略性能。

历史tail-safe记录可直接使用真实reward均值。未来weighted轨迹额外记录
tail_safe_clipped的样本均值，两种公式可在同一轨迹上准确比较。旧weighted数据
若缺少该值，保留n/a；不能用clip(mean(raw))替代mean(clip(raw))。

## 2. 从头训练匹配的奖励对照

若离线结果有比较价值，再在tmux/保持连接的终端执行：

```bash
df -h .
ss -ltnp | grep ':7056'
```

7056无监听且磁盘空间充足时：

```bash
python scripts/continual_validation/run_reward_ablation.py \
  --base-run-dir "$BASE" --output-dir "$NEW" \
  --updates-per-task 600 --port 7056 \
  --ports 323,321,320,345,346,347
```

两组均为WebServer→CacheFollower，复用原manifest及完全相同的流文件/仿真输入；
**不加载原先训练过的模型或replay**。同seed初始化，两组初始policy/target哈希必须
相同，训练首次启动计数为0且replay为空。不一致即报错；首次训练只跑一个episode
先检查初始化，再继续预算。初始评估不更新模型。

唯一主动改变的训练超参数为reward_profile：tail_safe vs weighted。
lambda沿用manifest，weighted系数固定0.50/0.30/0.20。hidden dimensions、动作空间、
target更新周期、global epsilon、replay容量和更新预算一致。
训练期间每组保留自己的replay跨A→B，绝不混入另一奖励公式的旧经验。

每种reward训练A 600次、B 600次，共2400次更新；initial/after-A/after-B各评估A/B，
两组合计12次冻结仿真（单seed，不重复无波动的相同seed评估）。两组串行使用7056。
完整after-A/after-B快照包含本地replay，磁盘开销显著高于只保留权重；没有自动删除。
每阶段仿真进程重启，队列不跨episode继承，这不是不中断的在线流量切换。

中断后重复runner命令追加`--resume`，所有参数必须相同。已有完整冻结节点校验后
跳过；缺失早期快照且训练已超预算时停止。不要同时启动第二个进程写同一NEW目录。

## 3. 统一业务指标与双reward评分

训练完成会自动生成attribution_report。再执行：

```bash
python scripts/continual_validation/rescore_rewards.py \
  --report-dir "$NEW/attribution_report" \
  --output-dir "$NEW/dual_reward_report" \
  --weights 0.50,0.30,0.20
cat "$NEW/attribution_report/REPORT.md"
cat "$NEW/dual_reward_report/REPORT.md"
```

每个任务的FCT使用两组所有节点共同完成流的统一交集，完成率按全部提供流计算。
attribution_report中的native reward属于各自公式，不可直接横比；dual_reward_report
为每条轨迹同时给出weighted和tail-safe评价。

预先报告四类结果，不按结果挑选单一有利指标：

1. A习得：各组initial→after-A，在A上评价。
2. B习得：各组after-A→after-B，在B上评价。
3. A保留：各组after-A→after-B，在A上评价。
4. 全网FCT/完成率与全部六个观测端口的队列、ECN、PFC，以及batch reward/loss。

训练曲线不是冻结评估；weighted若让reward更好而FCT更差，不认定其优于tail-safe。
选定奖励后再让ACC/SOR共同使用，不为两种算法分别选择最有利奖励。

## 验证

```bash
python -m unittest discover -s scripts/continual_validation -p test_reward_ablation.py -v
python -m unittest discover -s copter -p test_port_metrics.py -v
```

本地测试使用mock仿真验证初始化、预算、隔离、续跑与评分数学；不代表ns-3正式运行。
