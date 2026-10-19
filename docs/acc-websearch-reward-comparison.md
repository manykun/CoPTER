# WebSearch ACC reward comparison

Two concurrent online-only arms, 50 epochs, no frozen evaluation:

| Arm | Reward | Queue statistic |
|---|---|---|
| acc_paper | 0.7 utilization + 0.3 D(L); thresholds 20*2^n KiB, n=0..9 | Time mean in this action interval |
| acc_3816bc6_aligned | 0.5 utilization + 0.5 D(L); bisect 2^i/1024 against min(1,L/(400 KiB)), i=0..10 | Same interval mean |

The historical formula is from commit `3816bc6dd293ae7432ca6e4d0437efc7baceb9ae`.
Its original input was instantaneous queue depth; this experiment aligns both
arms to action-interval time means. Neither formula includes ECN reward.
Paper scores above the final threshold are capped at 0.1 and documented as an
implementation boundary convention. Byte units are KiB = 1024 bytes.

Shared controls: WebSearch CDF, 60% offered load, Poisson arrivals, sources
0–127 to receivers 128–255, 10-Gbps host links, start 2 s, duration 30 ms,
400-KiB buffer, multiscale actions, local FIFO replay capacity 1000,
50% congestion-stratified minibatch target, seed 1, hidden layers 32/64/64/32,
learning rate 0.001, gamma 0.95, global epsilon decay 2500, target synchronization
every 100 global training updates. Each action lasts the existing 500 us.
State observations retain peak queues for compatibility; rewards use separate
event-integrated queue bytes and transmit-byte deltas from action application
to the next observation. Python rejects missing or overlapping telemetry.

## Server execution (SSH)

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
git pull git@github.com:manykun/CoPTER.git exp/acc-validation
```

Rebuild once before starting parallel simulations because ns-3 telemetry changed:

```bash
cd ns-3.33
./waf build
cd ..
python -m unittest discover -s copter -p test_acc_interval_rewards.py
python -m unittest tools.analysis.tests.test_metrics_core tools.analysis.tests.test_aba_scheduler tools.analysis.tests.test_websearch_reward_pilot
```

```bash
bash scripts/continual_validation/run_acc_websearch_rewards.sh \
  --stage smoke --run-id-prefix acc_websearch_rewards_s1 \
  --epochs 50 --seed 1 --port-base 8200 --jobs 2
```

Smoke uses two epochs per arm in separate `_smoke_` directories. Formal runs
start fresh and never load the smoke replay.

```bash
mkdir -p experiments/aba
nohup bash scripts/continual_validation/run_acc_websearch_rewards.sh \
  --stage all --run-id-prefix acc_websearch_rewards_s1 \
  --epochs 50 --seed 1 --port-base 8200 --jobs 2 \
  > experiments/aba/acc_websearch_rewards_s1_driver.log 2>&1 &
```

Resume with the identical command and `--resume`. A completed epoch must have
its normal complete marker and full matching checkpoint; failed epochs retain
raw outputs and are not silently skipped. If a failed process already saved a
checkpoint, the scheduler refuses an ambiguous restart. Inspect it rather than
overwriting it. Inputs have one shared seed-specific flow/config and a hash
manifest under `simulation/mix/aba/websearch_rewards_s1/`.

## Outputs

- Driver progress: `experiments/aba/acc_websearch_rewards_s1_driver.log`.
- Per-arm summaries: `experiments/aba/acc_websearch_rewards_s1_ARM/summaries/`.
- Raw `.rate/.queue/.fct/.pfc` files: `simulation/output/aba/acc_websearch_rewards_s1_ARM/acc/a1/epoch_XXXX/`.
- Comparison: `experiments/aba/acc_websearch_rewards_s1_comparison/REPORT.md`.
- Figures: reward/loss, mean/p95/p99 FCT, throughput/completion/queue/ECN/PFC,
  short/medium/long flow FCT, reward components, every epoch plus trailing
  ten-epoch means. Ten-epoch tables show the full trajectory.
- Size groups: short <=100 KiB, medium (100 KiB,1 MiB), long >=1 MiB.
- Watch traces contain interval timestamps, mean bytes, reward decomposition,
  queue bin and chosen action; agent metrics also retain queue-bin counts.

Per-epoch FCT uses that epoch's valid completed flows, with counts recorded
before raw-file cleanup. Completion includes all offered flows. Native reward
scales differ; compare reward trends within each arm and physical metrics
between arms. These are exploratory online-training trajectories, not frozen
policy or catastrophic-forgetting evidence. Missing PFC remains n/a.
