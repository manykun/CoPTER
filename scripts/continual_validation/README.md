# ACC catastrophic-forgetting and SOR validation

For trace-derived WebServer/CacheFollower/WebSearch workload shifts, local-only
ACC training, and post-training causal action interpolation, see
`docs/acc-realistic-workload-forgetting.md` and
`scripts/continual_validation/run_realistic_acc.sh`.

This workflow changes traffic from task A to task B while keeping one common
tail-safe reward:

`R = clip(throughput - 5 * combined_queue² - 5 * combined_ecn², -1, 1)`.

Therefore a detected change cannot be attributed to switching reward weights.

The workflow can run either with four registered gates or in descriptive
`--report-only` mode:

1. fixed actions must affect both tasks in common-flow p95 or completion;
   reward-best must match the best common-flow p95 action among completion-safe
   actions and differ across tasks;
2. ACC must safely acquire task A;
3. ACC must safely acquire task B and then forget task A;
4. SOR must reduce forgetting while preserving task-B plasticity.

Training uses the same number of optimizer updates per task (default 600), not
the same number of episodes. `--epsilon-schedule phase` preserves the original
protocol that restarts exploration at each task boundary. The realistic
workload wrapper uses `--epsilon-schedule global`: epsilon is driven by the
persisted global environment-step counter and does not restart at A→B. The
network and replay memory do not reset between tasks in either mode.

ACC normally uses per-port FIFO replay plus cross-port shared/global replay.
Pass `--shared-replay false` for the local-only ablation: per-port FIFO replay
continues across A→B, while global upload/download and persistence are disabled.
This changes one replay factor without clearing task-A local memory.

Each run uses frozen flow/config inputs and a private `ns3_output/` prefix.
Two continual runs can therefore execute in parallel when they also use
different ns3-gym ports; their FCT/PFC/queue files do not overwrite one another.

## Server workflow

Run every command from the repository root:

```bash
cd /mnt/sdb1/xuduokun/projects/CoPTER
conda activate /mnt/sdb1/xuduokun/conda/envs/m3
git pull fork exp/acc-validation
```

Use a new run id because the manifest is immutable:

```bash
RUN_ID=tailsafe_mixed_incast_s1
COMMON_ARGS=(
  --run-id "$RUN_ID"
  --task-a mixed
  --task-b incast
  --seed 1
  --buffer-kb 400
  --updates-per-task 600
  --phase-epochs 100
  --eps-decay 2500
  --acc-hidden-dims "32,64,64,32"
  --reward-profile tail_safe
  --reward-queue-lambda 5
  --reward-ecn-lambda 5
)
```

Prepare fixed traffic and configs:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage prepare "${COMMON_ARGS[@]}"
```

Run the six-execution fixed-action screen:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage screen "${COMMON_ARGS[@]}"
```

Inspect `experiments/continual_validation/${RUN_ID}/screen/REPORT.md`. Do not
start long training if this gate fails.

Train/evaluate ACC. The script stops immediately if task-A acquisition,
task-B acquisition, completion safety, or forgetting gates fail:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage acc "${COMMON_ARGS[@]}"
```

To record the complete A→B trajectory without judging PASS/FAIL or stopping at
an acquisition/forgetting threshold, append `--report-only`. This mode still
writes every reward, common-flow FCT, completion, and forgetting measurement:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage acc "${COMMON_ARGS[@]}" --report-only --resume
```

If ACC passes, run SOR with identical traffic, common reward, exploration,
network width, and optimizer-update budgets:

```bash
bash scripts/continual_validation/run_continual.sh \
  --stage sor "${COMMON_ARGS[@]}" --report-only
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
  --report-only
```

The main outputs are `screen/REPORT.md`, `CONTINUAL_REPORT.md`,
`continual_summary.csv`, `continual_analysis.json`,
`eval/<method>/<phase>/<task>/`, and the two checkpoint directories.

## Resume and smoke

Append `--resume` to an interrupted stage with the exact same arguments. A
different argument set requires a new run id.

## Parallel ACC local-only and SOR experiment

The completed shared-replay ACC run can be reused. Prepare one new run for the
local-only ablation, then reuse its fixed-action screen because forced-action
evaluation never records or samples replay:

```bash
BASE_ID=tailsafe_mixed_incast_s1
LOCAL_ID=tailsafe_mixed_incast_localonly_s1

bash scripts/continual_validation/run_continual.sh \
  --stage prepare --run-id "$LOCAL_ID" --task-a mixed --task-b incast \
  --seed 1 --buffer-kb 400 --updates-per-task 600 --phase-epochs 100 \
  --eps-decay 2500 --acc-hidden-dims "32,64,64,32" \
  --reward-profile tail_safe --reward-queue-lambda 5 \
  --reward-ecn-lambda 5 --shared-replay false

mkdir -p "experiments/continual_validation/${LOCAL_ID}/screen"
cp -a "experiments/continual_validation/${BASE_ID}/screen/." \
  "experiments/continual_validation/${LOCAL_ID}/screen/"
echo "reused_from=${BASE_ID}" \
  > "experiments/continual_validation/${LOCAL_ID}/screen/REUSED_FROM"
```

Run both drivers with different ports. SOR adds 100 to its base port, so the
commands below use ports 5956 and 5856:

```bash
nohup bash scripts/continual_validation/run_continual.sh \
  --stage acc --run-id "$LOCAL_ID" --task-a mixed --task-b incast \
  --seed 1 --buffer-kb 400 --updates-per-task 600 --phase-epochs 100 \
  --eps-decay 2500 --acc-hidden-dims "32,64,64,32" \
  --reward-profile tail_safe --reward-queue-lambda 5 \
  --reward-ecn-lambda 5 --shared-replay false --port 5956 \
  --report-only --resume \
  > "experiments/continual_validation/${LOCAL_ID}/driver.log" 2>&1 &

nohup bash scripts/continual_validation/run_continual.sh \
  --stage sor --run-id "$BASE_ID" --task-a mixed --task-b incast \
  --seed 1 --buffer-kb 400 --updates-per-task 600 --phase-epochs 100 \
  --eps-decay 2500 --acc-hidden-dims "32,64,64,32" \
  --reward-profile tail_safe --reward-queue-lambda 5 \
  --reward-ecn-lambda 5 --shared-replay true --port 5756 \
  --report-only --resume \
  > "experiments/continual_validation/${BASE_ID}/sor_driver.log" 2>&1 &
```

Do not run two experiments with the same port. Monitor them using `jobs -l`
and `tail -f`; each run remains independently resumable.

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

The registered reward is `rollout_all_congested_mean`, which follows all
congested ports and the replay population more closely. The top-30% reward is
kept only as a diagnostic because it reversed the incast fixed-action ranking.

## Path-level checkpoint intervention

After a completed ACC A→B run, use frozen checkpoint intervention to test
whether forgetting is concentrated on the observed mixed→incast path. This
does not retrain either task. It copies the after-B checkpoint, restores only
selected ports' `policy` and `target` weights from after-A, and evaluates both
tasks with greedy actions and identical flow files.

```bash
bash scripts/continual_validation/run_path_intervention.sh \
  --stage all \
  --base-run-id tailsafe_mixed_incast_localonly_s1 \
  --port 5956
```

The default experiment measures single ports 323 and 371, the four-port core
set, the eight-port rack set, and same-size random controls. The watched set is
`190,191,200,201,323,371,379,380,403,404,422,423`. Each watched port records
reward and unclipped `tail_safe_raw` means over all, active, and congested
steps, action histograms, physical identifier, and a per-step JSONL trace.

Long evaluations can be split and resumed without rerunning completed cases:

```bash
bash scripts/continual_validation/run_path_intervention.sh \
  --stage prepare --base-run-id tailsafe_mixed_incast_localonly_s1

bash scripts/continual_validation/run_path_intervention.sh \
  --stage eval --base-run-id tailsafe_mixed_incast_localonly_s1 \
  --variants "after_a after_b restore_323 restore_371 restore_core random_core"

bash scripts/continual_validation/run_path_intervention.sh \
  --stage eval --base-run-id tailsafe_mixed_incast_localonly_s1 \
  --variants "restore_rack random_rack"

bash scripts/continual_validation/run_path_intervention.sh \
  --stage analyze --base-run-id tailsafe_mixed_incast_localonly_s1
```

Read `path_intervention/PATH_INTERVENTION_REPORT.md`. Compare targeted restores
with their same-size random controls. A larger old-task recovery together with
small incast cost supports path-localized functional forgetting; parameter
drift alone is not sufficient evidence.

## One-port continuous midpoint sweep

Before replacing discrete ACC/SOR with SAC or TD3, run a frozen local response
test. The sweep keeps the after-B checkpoint and all non-target ports greedy,
then overrides only port 323 with the midpoint between its dominant action and
each adjacent grid value. Continuous overrides are accepted only in greedy
evaluation, so they cannot enter discrete replay.

```bash
BASE_ID=tailsafe_mixed_incast_localonly_s1

bash scripts/continual_validation/run_continuous_sweep.sh \
  --stage baseline --base-run-id "$BASE_ID" \
  --target-port 323 --link-gbps 40 --port 6056

bash scripts/continual_validation/run_continuous_sweep.sh \
  --stage prepare --base-run-id "$BASE_ID" \
  --target-port 323 --link-gbps 40

bash scripts/continual_validation/run_continuous_sweep.sh \
  --stage sweep --base-run-id "$BASE_ID" \
  --target-port 323 --link-gbps 40 --port 6056

bash scripts/continual_validation/run_continuous_sweep.sh \
  --stage analyze --base-run-id "$BASE_ID" \
  --target-port 323 --link-gbps 40
```

The default sweep has at most seven fixed-action points: the dominant grid
point plus one lower and one upper midpoint per parameter. Add
`--include-grid-neighbors` consistently to `prepare` and `sweep` for an
endpoint+midpoint sweep of at most 13 points. Read
`continuous_sweep_port323/CONTINUOUS_SWEEP_REPORT.md`.

- A midpoint that improves mixed while preserving incast supports continuous
  control inside the current range.
- Monotonic improvement toward the outer point motivates a range ablation.
- Indistinguishable midpoints mean finer discretization is unlikely to help.
- The dominant-action fraction must be reported: a small fraction means the
  fixed local sweep characterizes one common action, not the full policy.

## Per-port physical range sweep

Use this experiment after the midpoint sweep shows an interior response. It
does **not** remap the global ACC action grid: ns-3 overrides one flattened
OpenGym port with physical `Kmin/Kmax/Pmax`, while the frozen after-B policy
continues to control every other port. Rebuild ns-3 once after pulling this
change.

The recommended sequence deliberately covers more than one port:

1. Screen 14 points on the known congested Agg-Core port 323 using `mixed`.
2. Evaluate only the completion-safe top three plus the center on `incast`.
3. Re-run the winning physical action on high-drift port 191. Treat it as a
   replication only if the report contains congested samples; otherwise move
   to another candidate such as 190, 200, or 201.
4. Run the same two-point comparison on active-but-not-congested port 371 as a
   negative control.

```bash
BASE_ID=tailsafe_mixed_incast_localonly_s1

bash build_ns3_copter.sh
python scripts/continual_validation/test_physical_range_sweep.py

# R1: 14-point old-task physical range screen.
bash scripts/continual_validation/run_physical_range_sweep.sh \
  --stage screen --base-run-id "$BASE_ID" --target-port 323 \
  --point-set wide --port 6156

# R2: center plus the automatically selected top three on the new task.
bash scripts/continual_validation/run_physical_range_sweep.sh \
  --stage safety --base-run-id "$BASE_ID" --target-port 323 \
  --point-set wide --port 6156

BEST_ACTION=$(python - "$BASE_ID" <<'PY'
import json, sys
path = (
    "experiments/continual_validation/" + sys.argv[1] +
    "/physical_range_port323_wide/top_candidates.json"
)
point = json.load(open(path, encoding="utf-8"))["candidates"][0]
print(f"{point['kmin_kb']},{point['kmax_kb']},{point['pmax']}")
PY
)
echo "Selected physical action: $BEST_ACTION"

# R3: second path candidate. Use a different socket if run in parallel.
bash scripts/continual_validation/run_physical_range_sweep.sh \
  --stage all --base-run-id "$BASE_ID" --target-port 191 \
  --point-set control --control-action "$BEST_ACTION" \
  --experiment-id physical_range_port191_control --port 6256

# R4: non-congested negative control.
bash scripts/continual_validation/run_physical_range_sweep.sh \
  --stage all --base-run-id "$BASE_ID" --target-port 371 \
  --point-set control --control-action "$BEST_ACTION" \
  --experiment-id physical_range_port371_control --port 6356
```

Each output directory contains `PHYSICAL_RANGE_REPORT.md`,
`physical_range_summary.csv`, `physical_range_analysis.json`, and the exact
`top_candidates.json` used by the safety stage. A useful wider range must
improve old-task p95 or the target port's congested reward, preserve completion,
and avoid a meaningful incast regression. Evidence for a general range effect
requires the direction to reproduce on a second **congested** port; port 371 is
only a specificity control.
