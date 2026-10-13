# ACC A→B→A 短程恢复实验

从稳定版ACC实验的 after_b 快照开始，额外训练A累计20、50、100次更新。
每个节点冻结评估A和B；return_0重新评估复制的after-B模型，作为本轮起点。
原实验目录不修改。独立目录保存模型、local replay、优化器状态、训练日志和快照。
原持续学习脚本的阶段快照省略.pkl文件；若after_b不含replay，本脚本会验证acc/models的
训练步数、环境步数、epoch、epsilon、phase、端口数、replay条数及所有网络文件哈希
与after_b一致后，复制其本地replay。验证不通过则停止，不会静默使用空replay。
replay来源和复制文件哈希写入replay_provenance.json。复制期间请勿在原实验目录训练。
epsilon继续全局衰减，target同步周期沿用原manifest；每次仿真重新启动，网络队列不会跨仿真保留。

在服务器项目根目录执行（BASE必须指向本次稳定版实验，不要按修改时间自动选择）：

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
git pull --ff-only origin exp/acc-validation
# 若服务器远程名为fork，将上一条的origin替换为fork。
BASE="experiments/continual_validation/${RUN_ID}"
test -f "$BASE/Q_DIAGNOSTICS.md" && cat "$BASE/Q_DIAGNOSTICS.md"
OUT="${BASE}_return_a_v1"
python scripts/continual_validation/run_return_a.py \
  --base-run-dir "$BASE" --output-dir "$OUT" \
  --updates 20,50,100 --ports 323,321,320,345,346,347 --port 6856
cat "$OUT/REPORT.md"
```

RUN_ID使用上次稳定版实验的真实名称。启动前确认Q诊断与目标实验一致；6856需要为空闲端口。
不需要重跑prepare、A训练或B训练。已有输出目录会拒绝覆盖；中断后保留现场，当前版本不支持断点续跑。

仅重新分析已完成结果：

```bash
python scripts/continual_validation/run_return_a.py \
  --stage analyze --base-run-dir "$BASE" --output-dir "$OUT" \
  --updates 20,50,100 --ports 323,321,320,345,346,347 --port 6856
```

输出REPORT.md、network.csv、ports.csv、analysis.json、recovery.png和training_p*.png。
图表需要matplotlib；缺少时仍输出数值报告。
network.csv在每个任务内部统一所有节点的共同完成流集合；完成率始终按全部提供流计算。
ports.csv包含拥塞/活跃reward、样本数、队列、ECN和PFC，缺失PFC保持为空。
checkpoint目录保存各个恢复节点，可用于后续恢复代价或SOR比较。

首要问题是A恢复多少、需要多少更新、是否同时损害B。训练batch reward不能代替冻结评估。
本实验说明再学习能力，不能单独证明灾难性遗忘，也不能替代A→A对照。
