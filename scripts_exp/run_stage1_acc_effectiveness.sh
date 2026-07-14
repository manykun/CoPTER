#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_common_args "$@"; require_run_dir
[[ -f "$RUN_DIR/stage0/.done" ]] || die "stage0 gate has not passed"
STAGE="$RUN_DIR/stage1"; mkdir -p "$STAGE"/{logs,metrics,fct,checkpoints,summaries,plots}
write_status stage1 running "ACC effectiveness"
[[ "$RESUME" == 1 && -f "$STAGE/.done" ]] && exit 0
: "${ACC_TRAIN_RUNNER:?Set ACC_TRAIN_RUNNER to a command accepting run_dir seed epochs resume}"
: "${ACC_EVAL_RUNNER:?Set ACC_EVAL_RUNNER to a command accepting method flow output_fct output_metrics seed checkpoint}"
: "${SCENARIO_A_TRAIN_FLOW:?Set SCENARIO_A_TRAIN_FLOW}"; : "${SCENARIO_A_FLOW:?Set SCENARIO_A_FLOW}"
epochs="${ACC_EPOCHS:-80}"; [[ "$SMOKE" == 1 ]] && epochs=2
bash -lc "$ACC_TRAIN_RUNNER '$STAGE' '$SEED' '$epochs' '$RESUME'" >"$STAGE/logs/train.log" 2>&1
metrics="${ACC_METRICS:-$STAGE/metrics/acc_metrics.jsonl}"; checkpoint="${ACC_CHECKPOINT:-$STAGE/checkpoints/final.pt}"
[[ -s "$metrics" && -s "$checkpoint" ]] || die "ACC runner did not create metrics/checkpoint"
static_args=()
for method in ACC SECN0 SECN1 SECN2; do
  out="$STAGE/fct/${method}.fct"; eval_metrics="$STAGE/metrics/${method}_eval.jsonl"
  bash -lc "$ACC_EVAL_RUNNER '$method' '$SCENARIO_A_FLOW' '$out' '$eval_metrics' '$SEED' '$checkpoint'" >"$STAGE/logs/eval_${method}.log" 2>&1
  [[ -s "$out" ]] || die "missing eval FCT for $method"
  [[ "$method" != ACC ]] && static_args+=(--static-fct "$method=$out")
done
python "$SCRIPTS_EXP/analyze_acc_effectiveness.py" --metrics "$metrics" --greedy-fct "$STAGE/fct/ACC.fct" --output-prefix "$STAGE/summaries/effectiveness" "${static_args[@]}"
touch_atomic "$STAGE/.done"; write_status stage1 completed "ACC effectiveness gate passed"
