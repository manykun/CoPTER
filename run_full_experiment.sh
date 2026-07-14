#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/paddlejob/workspace/yangziwen/CoPTER"
RESUME=0; SMOKE=0; SEED="${SEED:-1}"
ARGS=()
while (($#)); do
    case "$1" in
        --resume) RESUME=1; ARGS+=("$1") ;;
        --smoke) SMOKE=1; ARGS+=("$1") ;;
        --seed) shift; (($#)) || { echo "--seed requires a value" >&2; exit 2; }; SEED="$1"; ARGS+=(--seed "$1") ;;
        --seed=*) SEED="${1#*=}"; ARGS+=("$1") ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

RUN_ID="${RUN_ID:-full_experiment_seed${SEED}}"
BASE="${EXPERIMENT_ROOT:-$ROOT/experiments}/$RUN_ID"
mkdir -p "$BASE"
exec {LOCK_FD}>"$BASE/.lock"
flock -n "$LOCK_FD" || { echo "experiment already running: $BASE" >&2; exit 1; }
atomic_status() { local tmp="$BASE/status.tmp.$$"; printf 'state=%s\nupdated_at=%s\n' "$1" "$(date --iso-8601=seconds)" >"$tmp"; mv -f "$tmp" "$BASE/status"; }
trap 'rc=$?; ((rc == 0)) || atomic_status failed; exit $rc' EXIT
atomic_status running

stage0="$BASE/stage0"
stage1="$BASE/stage1"
if [[ ! -f "$stage0/.done" ]]; then
    RUN_ID="$RUN_ID/stage0" "$ROOT/run_stage0_sensitivity.sh" "${ARGS[@]}"
elif (( ! RESUME )); then
    echo "stage0 already complete; use --resume" >&2; exit 1
fi
[[ -f "$stage0/.done" ]] || { atomic_status gate_failed; echo "stage0 gate failed; stage1 will not run" >&2; exit 1; }

if [[ ! -f "$stage1/.done" ]]; then
    RUN_ID="$RUN_ID/stage1" "$ROOT/run_stage1_acc_effectiveness.sh" "${ARGS[@]}"
elif (( ! RESUME )); then
    echo "stage1 already complete; use --resume" >&2; exit 1
fi
[[ -f "$stage1/.done" ]] || { atomic_status failed; exit 1; }
printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" >"$BASE/.done.tmp.$$"
mv -f "$BASE/.done.tmp.$$" "$BASE/.done"
atomic_status done
