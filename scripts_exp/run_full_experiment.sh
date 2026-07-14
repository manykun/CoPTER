#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/common.sh"
parse_common_args "$@"
export SCENARIO_A_FLOW="${SCENARIO_A_FLOW:-$REPO_ROOT/simulation/mix/scenA_throughput.flow}"
export SCENARIO_B_FLOW="${SCENARIO_B_FLOW:-$REPO_ROOT/simulation/mix/scenB_incast.flow}"
export SCENARIO_A_TRAIN_FLOW="${SCENARIO_A_TRAIN_FLOW:-$SCENARIO_A_FLOW}"
export SCENARIO_B_TRAIN_FLOW="${SCENARIO_B_TRAIN_FLOW:-$SCENARIO_B_FLOW}"
export STATIC_RUNNER="${STATIC_RUNNER:-$SCRIPTS_EXP/static_runner.sh}"
export ACC_TRAIN_RUNNER="${ACC_TRAIN_RUNNER:-$SCRIPTS_EXP/acc_train_runner.sh}"
export ACC_EVAL_RUNNER="${ACC_EVAL_RUNNER:-$SCRIPTS_EXP/acc_eval_runner.sh}"
export CURRICULUM_RUNNER="${CURRICULUM_RUNNER:-$SCRIPTS_EXP/curriculum_runner.sh}"
if [[ -z "${RUN_DIR:-}" ]]; then RUN_DIR="$EXPERIMENT_ROOT/experiment_$(date +%Y%m%d_%H%M%S)_seed${SEED}"; fi
export RUN_DIR SEED RESUME SMOKE
mkdir -p "$RUN_DIR"/{configs,flows,checkpoints,metrics,fct,plots,summaries,logs}
exec 9>"$RUN_DIR/orchestrator.lock"; flock -n 9 || die "experiment already running: $RUN_DIR"
printf '%s\n' "$$" > "$RUN_DIR/orchestrator.pid"
trap 'rc=$?; write_status orchestrator failed "exit=$rc line=$LINENO"; exit $rc' ERR
write_manifest
args=(--run-dir "$RUN_DIR" --seed "$SEED"); [[ "$RESUME" == 1 ]] && args+=(--resume); [[ "$SMOKE" == 1 ]] && args+=(--smoke)
"$HERE/run_stage0_sensitivity.sh" "${args[@]}"
"$HERE/run_stage1_acc_effectiveness.sh" "${args[@]}"
"$HERE/run_stage2_forgetting.sh" "${args[@]}"
python "$HERE/generate_experiment_report.py" --run-dir "$RUN_DIR"
touch_atomic "$RUN_DIR/.done"; write_status orchestrator completed "all gates passed"
printf 'Experiment completed: %s\n' "$RUN_DIR"
