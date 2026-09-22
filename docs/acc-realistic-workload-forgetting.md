# ACC现实业务分布持续学习实验

该实验固定256主机拓扑、60% offered load、泊松到达、buffer、奖励函数和训练预算，
只切换测量得到的流大小CDF：WebServer、CacheFollower和WebSearch。它验证自然业务
分布变化是否产生ACC灾难性遗忘，不根据结果搜索或筛选任务对。

`run_realistic_acc.sh`固定关闭ACC跨端口global replay：

```text
--shared-replay false
```

这里关闭的是global replay buffer，不是交换机buffer。交换机buffer保持400 KB。

实验同时使用`--epsilon-schedule global`。epsilon在任务A中随累计环境步数
衰减，切换到任务B后继续同一条曲线，不重新回到1.0。这对应控制器无法
提前获知“新任务边界”的现实流量切换。原来的`phase`模式仍保留，作为
“任务边界重启探索”消融。

## 1. 单个任务转换

可选转换：

```text
webserver-cachefollower
cachefollower-websearch
websearch-webserver
```

先运行WebServer到CacheFollower：

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
conda activate /mnt/sdb1/xuduokun/conda/envs/m3

PAIR=webserver-cachefollower
RUN_ID=real_webserver_cachefollower_global_eps_s1

bash scripts/continual_validation/run_realistic_acc.sh \
  --stage prepare --pair "$PAIR" --run-id "$RUN_ID"

bash scripts/continual_validation/run_realistic_acc.sh \
  --stage screen --pair "$PAIR" --run-id "$RUN_ID"
```

screen仅作描述性测量，不会因为没有发现预设冲突而阻断现实工作负载实验。

后台训练：

```bash
nohup bash scripts/continual_validation/run_realistic_acc.sh \
  --stage acc --pair "$PAIR" --run-id "$RUN_ID" \
  > "experiments/continual_validation/${RUN_ID}_driver.log" 2>&1 &

echo $! > "experiments/continual_validation/${RUN_ID}_driver.pid"
tail -f "experiments/continual_validation/${RUN_ID}_driver.log"
```

中断后恢复：

```bash
bash scripts/continual_validation/run_realistic_acc.sh \
  --stage acc --pair "$PAIR" --run-id "$RUN_ID" --resume
```

## 2. 端口训练与冻结回测报告

```bash
bash scripts/continual_validation/run_realistic_acc.sh \
  --stage analyze --pair "$PAIR" --run-id "$RUN_ID"

cat "experiments/continual_validation/${RUN_ID}/PORT_CONTINUAL_REPORT.md"
```

## 3. 训练后连续插值

插值端点直接来自本轮screen中端口323在两个现实业务上的reward最优固定动作，不再依赖
旧的校准任务。after-B模型保持冻结，其余端口保持greedy，只覆盖323和负对照321：

```bash
nohup bash scripts/continual_validation/run_realistic_acc.sh \
  --stage interpolate --pair "$PAIR" --run-id "$RUN_ID" \
  > "experiments/continual_validation/${RUN_ID}_interpolation.log" 2>&1 &

tail -f "experiments/continual_validation/${RUN_ID}_interpolation.log"
```

结果：

```bash
REPORT="experiments/continual_validation/${RUN_ID}/port_path_sweep_p323"
cat "${REPORT}/PORT_PATH_REPORT.md"
column -s, -t < "${REPORT}/port_path_summary.csv" | less -S
```

报告使用1%相对reward容差识别平坦曲线和并列最优区间。只有两个任务都动作敏感且最优
alpha区间不重叠时，才报告端口策略冲突。如果两个screen最优动作相同，脚本将路径
标记为degenerate并只运行一个点，避免重复执行完全相同的冻结仿真。

## 4. 三个预注册转换

WebServer→CacheFollower完成后，使用新的run-id依次运行：

```bash
bash scripts/continual_validation/run_realistic_acc.sh \
  --stage all --pair cachefollower-websearch \
  --run-id real_cachefollower_websearch_s1

bash scripts/continual_validation/run_realistic_acc.sh \
  --stage all --pair websearch-webserver \
  --run-id real_websearch_webserver_s1
```

三个转换均应报告，不能只保留遗忘最大的转换。单seed用于机制探索；正式统计结论仍需
对最终预注册配置进行独立训练重复。
