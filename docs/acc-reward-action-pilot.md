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

`factorized_interp` retains the original threshold coordinates and inserts
every adjacent midpoint. Kmin and Kmax each have 17 categories. Exploration,
greedy selection, and Double-DQN target selection share the same physical
validity mask, leaving 221 combinations with `Kmin < Kmax`. Pmax retains the
seven multiscale values so the action-only arm changes thresholds only.

Native `tail_safe` and `weighted` reward values have different scales. Compare
their within-run direction, not their absolute magnitudes. Cross-variant
conclusions should use FCT, completion, throughput, queue, ECN, and PFC.

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
