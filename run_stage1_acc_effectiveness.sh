#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/paddlejob/workspace/yangziwen/CoPTER"
# shellcheck source=scripts_exp/common.sh
source "$ROOT/scripts_exp/common.sh"

ONLY="both"
COMMON_ARGS=()
while (($#)); do
    case "$1" in
        --only) shift; (($#)) || die "--only requires acc|sor|both"; ONLY="$1" ;;
        --only=*) ONLY="${1#*=}" ;;
        *) COMMON_ARGS+=("$1") ;;
    esac
    shift
done
parse_common_args "${COMMON_ARGS[@]}"
[[ "$ONLY" =~ ^(acc|sor|both)$ ]] || die "--only must be acc|sor|both"

RUN_ID="${RUN_ID:-stage1_effectiveness_seed${SEED}}"
RUN_DIR="$EXPERIMENT_ROOT/$RUN_ID"
TASK_A="${TASK_A:-Hadoop_Shuffle}"
TASK_B="${TASK_B:-AliStorage_AllReduce}"
PHASE_EPOCHS="${PHASE_EPOCHS:-50}"
(( SMOKE )) && PHASE_EPOCHS=1
[[ "$PHASE_EPOCHS" =~ ^[1-9][0-9]*$ ]] || die "PHASE_EPOCHS must be positive"

acquire_lock "$RUN_DIR"
trap cleanup_children EXIT INT TERM
require_runtime
mkdir -p "$RUN_DIR/config" "$RUN_DIR/methods" "$RUN_DIR/summary"
set_status "$RUN_DIR" running "A-B-A ACC+SOR"

run_method() {
    local method="$1" source config method_dir port phase epoch task step eval_task eval_step
    source="$DEFAULT_ACC_CONFIG"; port="${ACC_PORT:-5558}"
    [[ "$method" == sor ]] && { source="$DEFAULT_SOR_CONFIG"; port="${SOR_PORT:-5558}"; }
    method_dir="$RUN_DIR/methods/$method"
    config="$RUN_DIR/config/$method.yaml"
    mkdir -p "$method_dir/models" "$method_dir/phases"
    render_config "$source" "$config" "${RUN_ID}_${method}" "$method_dir/models" "$SEED" "$port"
    set_status "$method_dir" running "A-B-A"

    local tasks=("$TASK_A" "$TASK_B" "$TASK_A")
    for phase in 1 2 3; do
        task="${tasks[$((phase - 1))]}"
        for ((epoch=1; epoch<=PHASE_EPOCHS; epoch++)); do
            step="$method_dir/phases/phase_${phase}_${task}/train/epoch_$(printf '%04d' "$epoch")"
            if step_is_done "$step"; then
                (( RESUME )) && continue
                die "$step already completed; use --resume"
            fi
            rm -rf "$step"
            run_train_or_eval "$method" "$config" "$task" train "$step"
        done

        for eval_task in "$TASK_A" "$TASK_B"; do
            eval_step="$method_dir/phases/phase_${phase}_${task}/eval/${eval_task}"
            if step_is_done "$eval_step"; then
                (( RESUME )) && continue
                die "$eval_step already completed; use --resume"
            fi
            rm -rf "$eval_step"
            run_train_or_eval "$method" "$config" "$eval_task" eval "$eval_step"
        done
        mark_done "$method_dir/phases/phase_${phase}_${task}"
    done
    mark_done "$method_dir"
}

case "$ONLY" in
    acc) run_method acc ;;
    sor) run_method sor ;;
    both) run_method acc; run_method sor ;;
esac

if [[ "$ONLY" == both ]]; then
    step_is_done "$RUN_DIR/methods/acc" || die "ACC did not complete"
    step_is_done "$RUN_DIR/methods/sor" || die "SOR did not complete"
    mark_done "$RUN_DIR"
else
    set_status "$RUN_DIR" partial "completed only $ONLY"
fi
log "stage1 finished: $RUN_DIR"
