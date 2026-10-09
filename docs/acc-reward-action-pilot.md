# ACC reward and threshold-action pilot

This experiment is deliberately separate from the registered ABA/CDC routes.
It uses the canonical WebServer Task A traffic, ACC local replay, global
epsilon, and the authoritative per-epoch analysis pipeline.

## Variants

| Variant | Threshold action space | Training reward | Purpose |
|---|---|---|---|
| baseline | `multiscale` | `tail_safe` | Current reference |
| weighted | `multiscale` | `weighted` | Reward-only change |
| action | `factorized_interp` | `tail_safe` | Threshold-space-only change |
| combined | `factorized_interp` | `weighted` | Optional interaction check |
| legacy | `legacy` | `weighted` | Exact original independent Kmin/Kmax/Pmax heads and reward |
| short_window | `multiscale` | `tail_safe` over 4 steps | Isolate reward credit smearing |
| credit | `multiscale` | `tail_safe` over 4 steps | Add 50% congestion-stratified local replay |

`factorized_interp` retains the original threshold coordinates and inserts
every adjacent midpoint. Kmin and Kmax each have 17 categories. Exploration,
greedy selection, and Double-DQN target selection share the same physical
validity mask, leaving 221 combinations with `Kmin < Kmax`. Pmax retains the
seven multiscale values so the action-only arm changes thresholds only.

Native `tail_safe` and `weighted` reward values have different scales. Compare
their within-run direction, not their absolute magnitudes. Cross-variant
conclusions should use FCT, completion, throughput, queue, ECN, and PFC.

The `legacy` arm is intentionally different from `factorized_interp`. It uses
the original independent grids (6 Kmin values, 4 Kmax values, and 10 Pmax
values) without pairing Kmin and Kmax into one profile and without adding
midpoints. This preserves the historical 240 categorical combinations exactly.

## Smoke

```bash
bash scripts/continual_validation/run_acc_reward_action_pilot.sh \
  --stage smoke \
  --run-id-prefix acc_pilot_webserver_s1 \
  --epochs 100 \
  --stop-after 50 \
  --port-base 7600 \
  --jobs 1
```

Smoke uses two epochs internally and writes `_smoke` run IDs.

## First 50 epochs

```bash
nohup bash scripts/continual_validation/run_acc_reward_action_pilot.sh \
  --stage all \
  --run-id-prefix acc_pilot_webserver_s1 \
  --epochs 100 \
  --stop-after 50 \
  --port-base 7600 \
  --jobs 1 \
  > experiments/aba/acc_pilot_webserver_s1_driver.log 2>&1 &
```

The registered budget is 100 epochs, but execution stops cleanly after epoch
50. Do not change `--epochs` when extending the same runs.

## Extend the same checkpoints to 100 epochs

```bash
nohup bash scripts/continual_validation/run_acc_reward_action_pilot.sh \
  --stage all \
  --run-id-prefix acc_pilot_webserver_s1 \
  --epochs 100 \
  --stop-after 100 \
  --port-base 7600 \
  --jobs 1 \
  --resume \
  > experiments/aba/acc_pilot_webserver_s1_extend_driver.log 2>&1 &
```

## Outputs

Each arm retains its normal ABA-format model, replay, logs, raw outputs, and
per-epoch CSV. The combined report is written to:

```text
experiments/aba/acc_pilot_webserver_s1_comparison/
├── REPORT.md
├── epoch_metrics.csv
└── figures/
    ├── reward_loss.png
    ├── fct_curves.png
    └── network_curves.png
```

The pilot report uses training trajectories. It must not be described as a
frozen-policy comparison or as proof of convergence.

## CDC Task C replication

Use the Hadoop/AlibabaStorage base configuration to repeat the same three-arm
acquisition pilot on Task C (`realistic_hadoop`).  `--jobs 3` runs baseline,
weighted, and action variants concurrently on three consecutive ports:

```bash
nohup bash scripts/continual_validation/run_acc_reward_action_pilot.sh \
  --base-config configs/aba/hadoop_alistorage.yaml \
  --stage all \
  --run-id-prefix acc_pilot_hadoop_s1 \
  --epochs 100 \
  --stop-after 50 \
  --port-base 7620 \
  --jobs 3 \
  > experiments/aba/acc_pilot_hadoop_s1_driver.log 2>&1 &
```

The comparison report is written to
`experiments/aba/acc_pilot_hadoop_s1_comparison/`.

## Causal diagnostics after a completed baseline pilot

Use the isolated diagnostic runner to distinguish a weak learner from an
insensitive environment. It evaluates five fixed actions and frozen epoch-0
and final greedy policies on the same canonical flow. The source checkpoint is
read-only, every point is repeated, FCT uses the common completed-flow
intersection, and queue samples are also reported relative to their physical
Kmin/Kmax thresholds.

```bash
bash scripts/continual_validation/run_acc_pilot_diagnostics.sh \
  --run-dir experiments/aba/acc_pilot_hadoop_s1_baseline \
  --output-dir experiments/aba/acc_pilot_hadoop_s1_baseline_diagnostics \
  --stage all \
  --repeats 3 \
  --port 7700
```

The source pilot must have reached its registered final epoch. Existing output
requires `--resume`; changing the source checkpoint, repeat count, or port
requires a new output directory.

After the frozen evaluations finish, condition queue measurements on physical
port activity and rank switch-to-switch port sensitivity:

```bash
python tools/analysis/analyze_acc_pilot_port_activity.py \
  --diagnostic-dir experiments/aba/acc_pilot_hadoop_s1_baseline_diagnostics
```

The resulting `port_activity/REPORT.md`, `scopes.csv`,
`per_trial_ports.csv`, and `switch_port_sensitivity.csv` distinguish all
ports from switch-to-switch ports and use normalized transmit-rate cutoffs of
1%, 5%, and 10%. Queue and rate records are matched by physical port and
time. The report includes the match coverage; low coverage means conditional
figures need a closer look at monitor sampling times. The port rankings show
throughput, queue, ECN, and watched-port reward changes, while FCT remains a
network-level measurement.

## Credit-assignment experiment after the Hadoop diagnosis

The fixed-action evaluation shows that the environment can respond: action
`8,0` (`Kmin=80 KB`, `Kmax=160 KB`, `Pmax=0.05`) materially improves Hadoop
FCT, while the trained policy does not learn to prefer it. Test credit
assignment before changing traffic, buffer, action ranges, or SOR. These
three arms keep all of those variables fixed:

| Arm | Reward history | Local replay minibatch |
|---|---:|---|
| baseline | 32 observations | uniform |
| short_window | 4 observations | uniform |
| credit | 4 observations | target 50% queue/ECN-informative transitions |

The historical 32-observation congestion/activity detector remains unchanged
in all arms. Only the reward attribution history is shortened, so the
short-window comparison does not silently redefine the monitored population.

Run a two-epoch smoke first:

```bash
python scripts/continual_validation/run_acc_reward_action_pilot.py \
  --base-config configs/aba/hadoop_alistorage.yaml \
  --stage smoke \
  --variants baseline,short_window,credit \
  --run-id-prefix acc_credit_hadoop_s1 \
  --epochs 50 \
  --stop-after 2 \
  --port-base 7800 \
  --jobs 3
```

Then start three 50-epoch arms in parallel with a fresh prefix:

```bash
nohup python scripts/continual_validation/run_acc_reward_action_pilot.py \
  --base-config configs/aba/hadoop_alistorage.yaml \
  --stage all \
  --variants baseline,short_window,credit \
  --run-id-prefix acc_credit_hadoop_s1 \
  --epochs 50 \
  --stop-after 50 \
  --port-base 7800 \
  --jobs 3 \
  > experiments/aba/acc_credit_hadoop_s1_driver.log 2>&1 &
```

The comparison report adds replay-signal availability/sample ratios and the
Q-value margin of reference action `8,0`. After training, freeze each final
policy. Run the full fixed-action suite once for the baseline and only the
frozen evaluation for the other arms:

```bash
bash scripts/continual_validation/run_acc_pilot_diagnostics.sh \
  --run-dir experiments/aba/acc_credit_hadoop_s1_baseline \
  --output-dir experiments/aba/acc_credit_hadoop_s1_baseline_diagnostics \
  --stage all --repeats 3 --port 7810

for arm in short_window credit; do
  bash scripts/continual_validation/run_acc_pilot_diagnostics.sh \
    --run-dir "experiments/aba/acc_credit_hadoop_s1_${arm}" \
    --output-dir "experiments/aba/acc_credit_hadoop_s1_${arm}_diagnostics" \
    --stage frozen --repeats 3 --port 7810
done

python tools/analysis/analyze_acc_reward_action_pilot.py \
  --run-id-prefix acc_credit_hadoop_s1 \
  --variants baseline,short_window,credit
```

Loss reduction alone is not success. Continue to ABA/SOR only if the frozen
final policy moves mean and p95 FCT toward the fixed `high_gentle` reference,
the sampled informative-transition ratio rises in the `credit` arm, and the
Q margin of action `8,0` improves without a material completion loss.
