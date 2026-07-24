# ACC catastrophic-forgetting and SOR validation

This workflow is a controlled non-stationary-objective experiment. Task A uses
a latency/queue-oriented reward (`0.25,0.55,0.20`), while task B uses a
throughput-oriented reward (`0.70,0.15,0.15`). It is designed to test the
forgetting mechanism; it must not be described as a purely natural traffic
distribution shift.

The workflow has four registered gates:

1. fixed actions must affect both tasks, and the reward-best action must differ;
2. ACC must safely acquire task A;
3. ACC must safely acquire task B and then forget task A;
4. SOR must reduce forgetting while preserving task-B plasticity.

Training uses the same number of optimizer updates per task (default 600), not
the same number of episodes. Epsilon resets to `1.0` at each task boundary and
decays with a phase-local environment-step counter. The network and replay
memory do not reset between tasks.

## Server workflow

Run every command from the repository root:

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
git pull origin exp/acc-validation
```

Use a new run id because the manifest is immutable:

```bash
RUN_ID=controlled_mixed_incast_s1
COMMON_ARGS="--run-id ${RUN_ID} --task-a mixed --task-b incast --seed 1 \
--buffer-kb 400 --updates-per-task 600 --phase-epochs 100 \
--eps-decay 2500 --acc-hidden-dims 32,64,64,32 \
--task-a-reward-weights 0.25,0.55,0.20 \
--task-b-reward-weights 0.70,0.15,0.15"
```

Prepare fixed traffic and configs:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage prepare ${COMMON_ARGS}
```

Run the six-execution fixed-action screen:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage screen ${COMMON_ARGS}
```

Inspect `experiments/continual_validation/${RUN_ID}/screen/REPORT.md`. Do not
start long training if this gate fails.

Train/evaluate ACC. The script stops immediately if task-A acquisition,
task-B acquisition, completion safety, or forgetting gates fail:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage acc ${COMMON_ARGS}
```

If ACC passes, run SOR with identical traffic, reward schedules, exploration,
network width, and optimizer-update budgets:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage sor ${COMMON_ARGS}
```

Build the final ACC/SOR comparison:

```bash
python scripts/continual_validation/analyze_forgetting.py \
  --run-dir "experiments/continual_validation/${RUN_ID}" \
  --compare acc,sor \
  --min-reward-drop 0.10 \
  --min-p95-worsening 0.10 \
  --min-forgetting-reduction 0.30 \
  --new-task-p95-tolerance 0.05 \
  --completion-tolerance 0.01 \
  --gate
```

The main outputs are `screen/REPORT.md`, `CONTINUAL_REPORT.md`,
`continual_summary.csv`, `continual_analysis.json`,
`eval/<method>/<phase>/<task>/`, and the two checkpoint directories.

## Resume and smoke

Append `--resume` to an interrupted stage with the exact same arguments. A
different argument set requires a new run id.

For pipeline validation only:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage prepare --run-id smoke_controlled --task-a mixed --task-b incast \
  --seed 1 --smoke
bash scripts/continual_validation/run_continual.sh \
  --stage screen --run-id smoke_controlled --task-a mixed --task-b incast \
  --seed 1 --smoke
```

Smoke data is capped and must not be used as experimental evidence.

## Interpretation

Acquisition requires either a 2% reward gain or a 5% common-flow p95 FCT
improvement, and completion may not fall by more than one percentage point.
ACC forgetting requires both a 10% old-task reward drop and a 10% old-task
common-flow p95 worsening. SOR must acquire both tasks, reduce the positive
forgetting score by at least 30%, and remain within the registered task-B p95
and completion tolerances.

A single seed is mechanism evidence, not a statistical generalization claim.
