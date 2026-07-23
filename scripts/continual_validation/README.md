# ACC catastrophic-forgetting and SOR validation

This workflow runs a fixed two-task curriculum with one persistent model:

1. evaluate the untrained model on task A;
2. train task A;
3. greedily evaluate tasks A and B;
4. continue the same model and replay memory on task B;
5. greedily evaluate tasks A and B again;
6. repeat with SOR under identical controller settings.

The old-task comparison is `A after A` versus `A after B`. The new-task
acquisition comparison is `B after A` versus `B after B`. Evaluation never
records experience, trains, changes epsilon, or saves over the live checkpoint.

## Preconditions

- The `m3` conda environment is active or available from the user's standard
  Miniconda/Anaconda installation.
- `ns-3.33/build/scratch/copter-sim` exists.
- Task A and B are scenarios on which the single-task ACC validation can learn.
  Prefer the action-sensitive scenario as task A and the most different passed
  scenario as task B. For the current seed-1 study the default is
  `incast -> throughput`; change it if `incast` did not pass the earlier ACC
  effectiveness analysis.
- Use a new run id for a new experiment. `--resume` is only for continuing the
  exact manifest after an interruption.

## Server commands

From the repository root:

```bash
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
git pull
```

Prepare immutable seed-1 traffic and the experiment manifest:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage prepare \
  --run-id sor_ab_seed1 \
  --task-a incast \
  --task-b throughput \
  --seed 1 \
  --buffer-kb 400 \
  --phase-epochs 30 \
  --eps-decay 2500
```

Run ACC first:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage acc \
  --run-id sor_ab_seed1 \
  --task-a incast \
  --task-b throughput \
  --seed 1 \
  --buffer-kb 400 \
  --phase-epochs 30 \
  --eps-decay 2500
```

Analyze ACC before spending time on SOR:

```bash
python scripts/continual_validation/analyze_forgetting.py \
  --run-dir experiments/continual_validation/sor_ab_seed1 \
  --method acc \
  --min-reward-drop 0.10 \
  --min-p95-worsening 0.10 \
  --gate-forgetting
```

Only if ACC both learns the tasks and crosses the registered forgetting gate,
run SOR:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage sor \
  --run-id sor_ab_seed1 \
  --task-a incast \
  --task-b throughput \
  --seed 1 \
  --buffer-kb 400 \
  --phase-epochs 30 \
  --eps-decay 2500
```

Build the final comparison:

```bash
python scripts/continual_validation/analyze_forgetting.py \
  --run-dir experiments/continual_validation/sor_ab_seed1 \
  --compare acc,sor \
  --min-reward-drop 0.10 \
  --min-p95-worsening 0.10 \
  --min-forgetting-reduction 0.30 \
  --new-task-p95-tolerance 0.05 \
  --completion-tolerance 0.01 \
  --gate
```

The main outputs are:

- `CONTINUAL_REPORT.md`
- `continual_summary.csv`
- `continual_analysis.json`
- `eval/<method>/<phase>/<task>/`
- `<method>/checkpoints/after_a/` and `after_b/`

## Registered interpretation

ACC forgetting is detected only if both conditions hold:

- old-task reward decreases by at least 10%;
- old-task common-flow p95 FCT worsens by at least 10%.

The final SOR gate additionally requires:

- both methods demonstrate task-A and task-B acquisition;
- SOR reduces the combined positive reward/FCT forgetting score by at least 30%;
- SOR task-B common-flow p95 is no more than 5% worse than ACC;
- SOR task-B completion is no more than one percentage point below ACC.

With one traffic seed, the result is evidence for the mechanism in this fixed
curriculum, not a statistical generalization claim.

## Resume and smoke check

Resume an interrupted exact run:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage acc \
  --run-id sor_ab_seed1 \
  --task-a incast \
  --task-b throughput \
  --seed 1 \
  --buffer-kb 400 \
  --phase-epochs 30 \
  --eps-decay 2500 \
  --resume
```

For pipeline validation only, use a different run id:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage all \
  --run-id smoke_sor_ab \
  --task-a incast \
  --task-b throughput \
  --seed 1 \
  --smoke
```

Smoke results must not be used as experimental evidence.
