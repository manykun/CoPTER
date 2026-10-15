# ACC reward × realistic-traffic 50-epoch screen

## Why this screen is needed

The available results do not yet show robust FCT learning.  On the prior
100-epoch Hadoop acquisition pilot, first-to-last mean FCT changed by
`-0.24%` for multiscale/tail-safe, `+0.81%` for multiscale/weighted, and
`-0.54%` for factorized-interpolation/tail-safe.  These are training-trajectory
changes and do not isolate reward from action-space changes.  A frozen Hadoop
fixed-action screen was much more sensitive: `high_gentle` had 113.197 µs
mean FCT versus 134.612 µs for `low_gentle`, yet the trained policy did not
reliably choose that beneficial action.  In the later matched queue-penalty
test, reducing lambda from 5 to 1 improved frozen mean FCT by 1.51% versus
the matched control and 2.22% versus its own initialization, short of the
predeclared 3% advance threshold.  Port 349 still never selected `8,0`.

This matrix asks whether the weak learning is common to all four existing
realistic traffic modes, or depends on the reward formula.  It does **not**
search for a favorable forgetting pair.  Do not compare absolute native
rewards across formulas or absolute FCT across different traffic tasks.

## Registered 4 × 3 matrix

| Canonical traffic | Source |
|---|---|
| `webserver` | `realistic_webserver`, ABA Task A |
| `cachefollower` | `realistic_cachefollower`, ABA Task B |
| `hadoop` | `realistic_hadoop`, CDC Task C |
| `alistorage` | `realistic_alistorage`, CDC Task D |

| Reward arm | Training reward |
|---|---|
| `tail5` | tail-safe, queue lambda 5, ECN lambda 5 |
| `tail1` | tail-safe, queue lambda 1, ECN lambda 5 |
| `weighted` | original weighted reward, coefficients 0.5/0.3/0.2 |

All 12 cells use the same multiscale action space, 400-KB buffer, ACC local
replay only, 4-step reward-attribution window, congestion-stratified replay
with 50% target, network/agent settings, seed 1, 50 epochs, global epsilon
decay, and target-network interval.  Thus, within a traffic task, the training
reward is the only intended treatment.  Each cell has isolated config,
checkpoint/replay, output, logs and TCP port.  The scheduler keeps at most
`--jobs` NS-3/ACC pairs active; all 12 cells are queued in one launch.  Use
`--jobs 3` only if memory and CPU allow it, otherwise use 2.  Do not run
another simulator experiment on the assigned ports at the same time.

## Server run order

Run from the CoPTER repository root.  Use an empty output prefix; never reuse
the historical pilot or queue-decision IDs.  The Python runner checks that at
least 20 GiB is free by default, since prior experiments filled the disk.

```bash
git pull git@github.com:manykun/CoPTER.git exp/acc-validation
python -m unittest scripts.continual_validation.test_acc_reward_traffic_matrix \
  tools.analysis.tests.test_metrics_core

# Small, two-cell pipeline smoke.  The smoke ID is separate from formal IDs.
bash scripts/continual_validation/run_acc_reward_traffic_matrix.sh \
  --stage smoke --tasks webserver,hadoop --rewards tail5,weighted \
  --run-id-prefix acc_reward_traffic_matrix_s1 \
  --port-base 7900 --jobs 2

# All 12 cells: at most three run concurrently.  The all stage trains,
# then evaluates each frozen initial/final policy twice, then reports.
nohup bash scripts/continual_validation/run_acc_reward_traffic_matrix.sh \
  --stage all --tasks webserver,cachefollower,hadoop,alistorage \
  --rewards tail5,tail1,weighted \
  --run-id-prefix acc_reward_traffic_matrix_s1 \
  --epochs 50 --seed 1 --port-base 7900 --jobs 3 --repeats 2 \
  > experiments/aba/acc_reward_traffic_matrix_s1_driver.log 2>&1 &
echo "PID=$!"
```

Run inside tmux if the SSH session may disconnect.  `nohup` protects only the
outer driver; preserve the same CLI arguments and add `--resume` if interrupted.
To diagnose a failed cell, inspect its
`experiments/aba/acc_reward_traffic_matrix_s1_TASK_REWARD/matrix_driver.log`
or `frozen_driver.log`.  Do not erase output or cold-start replay.

```bash
tail -f experiments/aba/acc_reward_traffic_matrix_s1_driver.log
cat experiments/aba/acc_reward_traffic_matrix_s1_matrix/REPORT.md
```

If training completed but frozen evaluation needs to be resumed, run the same
matrix arguments with `--stage frozen --resume`.  Frozen evaluations default
to ports `--port-base + 100` through `--port-base + 111`; if that range is
occupied, select a separate free range with `--eval-port-base`.  This changes
only the diagnostic runner, not the registered training plan.  An existing
diagnostic directory registers its evaluation port in `protocol.json` and
cannot be resumed on another port.  Preserve and archive an incomplete
directory before retrying with a different range; do not overwrite a
completed diagnostic run.  The runner checks every requested evaluation port
before starting any frozen evaluation.  If all frozen evaluations finished
and only reporting failed, use `--stage analyze`.  The report and
figures are under `experiments/aba/acc_reward_traffic_matrix_s1_matrix/`.

## How to interpret

Per-epoch reward/loss/FCT figures are training trajectories, not proof that a
greedy learned policy improved.  The frozen table evaluates initial and final
policies on each task's **common completed-flow intersection** across the
three arms and both checkpoints.  Completion still uses all offered flows.
Compare each final policy with its own initialization and with `tail5` final
on that same task.  Inspect mean/p95/p99 FCT, throughput, completion,
queue/ECN/PFC together.  The two identical-seed repeats check deterministic
execution; they are not independent samples or statistical significance.

Predeclared follow-up rule: only advance a reward arm to a separate-seed
confirmation if frozen mean FCT is at least 3% below both its own initial and
that task's `tail5` final, with completion cost no more than 0.1 percentage
point and p95 cost no more than 5%, and no unexplained p99/PFC regression.
If no arm meets this screen, first inspect action distributions and whether
the reward-best action agrees with the FCT-best fixed action at active
bottleneck ports.  Do not infer that SOR solves forgetting from this
single-task acquisition screen.
