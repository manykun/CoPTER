# ACC奖励、复现波动与训练漂移归因实验

目的：解释batch reward、冻结reward与FCT为什么不一致；比较继续训练A和切换B，
而不是搜索更容易出现遗忘的任务对。保持现有奖励、global epsilon、target同步周期、
动作空间、网络结构不变，关闭global replay。所有输出使用独立目录，不修改原实验。

## 1. 更新与只读分析（不训练、不运行仿真）

服务器项目根目录执行：

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
git pull --ff-only origin exp/acc-validation
BASE=experiments/continual_validation/real_web_cache_target100_s1
AUDIT="${BASE}_reward_audit_v1"
REPEAT="${BASE}_repeat_v1"
CONTROL="${BASE}_aa_ab_reset_v1"
test -f "$BASE/manifest.json" || { echo '原实验目录错误'; exit 1; }
df -h .
python scripts/continual_validation/analyze_acc_attribution.py \
  --run-dir "$BASE" --output-dir "$AUDIT"
cat "$AUDIT/REPORT.md"
```

远程名如果是fork，替换git pull中的origin；本地Mac推送到fork，服务器原有origin可以
指向同一仓库。不要改变源实验目录中的模型或流文件。

重点输出：

| 文件 | 用途 |
|---|---|
| training.csv / training_p*.png | replay batch reward、在线全网rollout reward、TD loss分开画 |
| reward_components.csv | 吞吐项、队列/ECN负贡献、raw、clip影响、各统计群体的样本数 |
| physical_ports.csv | 逐端口队列、ECN、PFC和样本数，缺失不作零 |
| network.csv / frozen.png | 冻结A/B的reward、统一共同流p95、完成率 |

当前全网reward实际上是“拥塞端口样本 + 无拥塞时top-K回退”，不是固定端口集。
新版本补充回退样本计数，不改变奖励算法；旧数据没有该计数时保留空值。
逐端口all/active/congested三组分别统计，不在拥塞样本为空时自动替换为活跃样本。
all固定的是端口身份，不能保证策略改变后的状态分布相同。
raw由平均throughput减去lambda乘**平均平方成本**重构，不能把平均队列再平方。
clip(mean(raw))不等于mean(clip(raw))；CSV明确区分二者。

## 2. 原after-A/after-B检查点重复冻结评估

不训练，共2检查点×2任务×3次=12次仿真。保持同流文件、配置seed及检查点RNG状态；
此处衡量同条件重复运行波动，不是3个独立流量seed。
socket 6956必须空闲；按顺序运行，不与其他实验抢占端口。

```bash
python scripts/continual_validation/run_acc_attribution.py \
  --stage repeat --base-run-dir "$BASE" --output-dir "$REPEAT" \
  --ports 323,321,320,345,346,347 --repeats 3 --port 6956
python scripts/continual_validation/analyze_acc_attribution.py --run-dir "$REPEAT"
cat "$REPEAT/attribution_report/REPORT.md"
```

repeat_summary.csv包含每个模型/任务的均值、标准差、最小最大值。范围不是置信区间；
完全一致也不意味着跨流量seed稳定。评估会校验模型、训练状态、replay文件哈希未变。
若变化量与重复范围接近，不应将它当作强遗忘证据；不要为了得到有利结果筛选重复次数。

## 3. 同after-A起点的A→A与A→B对照

**重要：旧版after-A快照通常不含本地replay，而after-B的live replay不能冒充after-A。**
脚本默认`--replay-policy require`，缺少匹配replay就停止。

以下针对现有旧快照采用显式`reset`：两分支都从完全相同的after-A网络、target、
优化器及随机状态出发，**同时清空本地replay**；global epsilon和更新计数保持连续。
这是匹配的冷replay消融，不是原始连续replay实验的无缝续跑。不能直接拿它与原AB
相减归因；应比较本轮AA和AB两条分支。若不接受重置，先取得真实完整after-A replay，
然后使用新目录和`--replay-policy require`，绝不能补入after-B replay。

两分支各训练600次更新，在0/100/300/600冻结A、B并各重复3次；共1200次训练更新、
48次冻结仿真。统计单位为优化器更新，不是epoch。预算预先固定，不按效果提前选点。

```bash
(
  set -e
  python scripts/continual_validation/run_acc_attribution.py \
    --stage control --base-run-dir "$BASE" --output-dir "$CONTROL" \
    --replay-policy reset --updates 100,300,600 \
    --ports 323,321,320,345,346,347 --repeats 3 --port 6956
  python scripts/continual_validation/analyze_acc_attribution.py --run-dir "$CONTROL"
) > "${CONTROL}_driver.log" 2>&1 &
echo "PID=$!"
tail -f "${CONTROL}_driver.log"
```

此后台示例应在保持连接的终端或tmux中运行。`Ctrl-C`退出tail不会停止后台实验。
需要断线运行时将上述两条Python命令放入tmux会话；不要把tail是否退出当作任务状态。
开始前用`ss -ltnp | grep ':6956'`检查监听；若有输出，换空闲端口并记录，勿杀无关进程。

## 4. 查看归因结果

```bash
cat "$CONTROL/attribution_report/REPORT.md"
```

control_effects.csv计算同一重复编号下：

`额外变化 = (AB终点 − AB起点) − (AA终点 − AA起点)`，全部在冻结A上比较。

- reward额外变化为负、p95额外变化为正：支持切换B带来额外退化。
- AA也同样退化：不能把原有回退全部归因于跨任务遗忘。
- reward改善但p95恶化：检查奖励分量与业务目标是否失配。
- 同时检查B习得、完成率和端口样本口径，不以ECN增加单独判坏，不以loss下降判成功。
- 报告不输出PASS/FAIL、不自动改奖励，不将同seed重复冒充统计泛化。

### 中断恢复

重复原runner命令并追加`--resume`（所有参数不变）；已完成的冻结节点校验后跳过，
训练按已保存global_train_step继续。若源文件或协议改变则拒绝续跑。缺失早期训练快照
且当前已超出该预算时停止，不伪造中间结果。分析脚本要求完整评估矩阵。

## 本地验证

```bash
python -m unittest discover -s scripts/continual_validation -p test_acc_attribution.py -v
```

这些是mock模拟器测试，不代表服务器ns-3实验已经运行。
