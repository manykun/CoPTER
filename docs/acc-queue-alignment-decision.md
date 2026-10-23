# ACC queue-penalty decision experiment

## Why this is the next experiment

The Hadoop frozen port intervention showed that forcing action `8,0` on port
349 reduced network mean FCT while lowering that port's native reward and
raising its queue.  The environment is action-sensitive; the next question is
whether the local queue penalty prevents ACC from learning a physically useful
policy.  This is an acquisition study, not a forgetting or SOR claim.

## Registered comparison

| Arm | Action space | Reward | Queue lambda | ECN lambda | Window | Replay |
|---|---|---|---:|---:|---:|---|
| `credit_control` | multiscale | tail_safe | 5 | 5 | 4 | congestion-stratified, 50% target |
| `credit_queue1` | multiscale | tail_safe | 1 | 5 | 4 | congestion-stratified, 50% target |

Both arms use the canonical Hadoop task, the same flow/config hashes, seed,
400-KB switch buffer, architecture, epsilon schedule, target-sync interval,
training budget and watched ports.  Port 349 is added to both watch lists.
Only the queue penalty changes.  Native rewards have different meanings across
arms and must not be compared as if they shared a scale.  Evaluate frozen
physical metrics on the common completed-flow intersection.

## Server sequence

Run from the repository root after pulling the experiment branch.  First run
the short pipeline smoke with a distinct ID:

```bash
git pull git@github.com:manykun/CoPTER.git exp/acc-validation
python -m unittest scripts.continual_validation.test_acc_queue_decision tools.analysis.tests.test_metrics_core

bash scripts/continual_validation/run_acc_reward_action_pilot.sh \
  --base-config configs/aba/hadoop_alistorage.yaml \
  --stage smoke --variants credit_control,credit_queue1 \
  --run-id-prefix acc_queue_decision_smoke_s1 \
  --epochs 50 --stop-after 2 --port-base 7820 --jobs 2
```

Then train the two isolated arms.  `--epochs` is the registered total budget;
keep it at 50 when resuming.  Use a fresh ID rather than altering historical
`credit` results.

```bash
RUN_ID=acc_queue_decision_s1
bash scripts/continual_validation/run_acc_reward_action_pilot.sh \
  --base-config configs/aba/hadoop_alistorage.yaml \
  --stage all --variants credit_control,credit_queue1 \
  --run-id-prefix "$RUN_ID" \
  --epochs 50 --stop-after 50 --seed 1 --port-base 7820 --jobs 2
```

If interrupted, repeat the same command with `--resume`.  Check that both
source runs completed all 50 epochs before evaluation.  Frozen evaluations
are deliberately sequential and use separate ports to avoid stale socket
collisions.  If a diagnostic directory already exists, add `--resume` to its
corresponding command.

```bash
for arm in credit_control credit_queue1; do
  if test "$arm" = credit_control; then port=7840; else port=7841; fi
  bash scripts/continual_validation/run_acc_pilot_diagnostics.sh \
    --run-dir "experiments/aba/${RUN_ID}_${arm}" \
    --output-dir "experiments/aba/${RUN_ID}_${arm}_diagnostics" \
    --stage frozen --repeats 3 --port "$port"
done

python tools/analysis/analyze_acc_queue_decision.py \
  --run-id-prefix "$RUN_ID" --repeats 3
cat "experiments/aba/${RUN_ID}_reward_decision/DECISION_REPORT.md"
```

The decision report requires both arms' protocol to match on all registered
controls, verifies completed frozen evaluations, and recomputes FCT on one
common-flow set across both checkpoints, arms, and repeats.  It also reports
port 349's `8,0` greedy frequency, Q margin, actual action frequency, and
native reward terms.  Read `frozen_comparison.csv` and
`analysis_provenance.json` for exact values and input hashes.

## Stop/advance rule

Advance to an independent-seed confirmation only if the candidate's frozen
mean FCT is at least 3% lower than both its own initial checkpoint and the
control's final checkpoint, with at most 0.1 percentage-point completion cost
and at most 5% p95 FCT cost.  Also inspect p99, throughput, queue, ECN and PFC;
the numeric screen is not a substitute for that multi-metric review.  If the
screen fails, stop this reward-adjustment line and investigate whether the
per-port objective can represent the network goal before spending on SOR.

Three repeats with identical seed and traffic check determinism; they are
not three independent samples.  This experiment alone cannot prove
catastrophic forgetting or SOR's benefit.
