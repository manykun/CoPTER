#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/paddlejob/workspace/yangziwen/CoPTER"
# shellcheck source=scripts_exp/common.sh
source "$ROOT/scripts_exp/common.sh"
parse_common_args "$@"

RUN_ID="${RUN_ID:-stage0_sensitivity_seed${SEED}}"
RUN_DIR="$EXPERIMENT_ROOT/$RUN_ID"
CONFIG="$RUN_DIR/config/acc.yaml"
TASK="${STAGE0_TASK:-Hadoop_Shuffle}"
ACTIONS=( ${STAGE0_ACTIONS:-"0,0,0 5,3,9"} )
(( ${#ACTIONS[@]} >= 2 )) || die "stage0 requires at least two fixed actions"

acquire_lock "$RUN_DIR"
trap cleanup_children EXIT INT TERM
require_runtime
mkdir -p "$RUN_DIR/config" "$RUN_DIR/actions" "$RUN_DIR/summary"
render_config "$DEFAULT_ACC_CONFIG" "$CONFIG" "${RUN_ID}_acc" "$RUN_DIR/models" "$SEED" "${NS3_PORT:-5558}"
set_status "$RUN_DIR" running "fixed-action sensitivity gate"

for index in "${!ACTIONS[@]}"; do
    step="$RUN_DIR/actions/action_$((index + 1))"
    if step_is_done "$step"; then
        (( RESUME )) && { log "resume: skipping $(basename "$step")"; continue; }
        die "$step already completed; use --resume"
    fi
    rm -rf "$step"; mkdir -p "$step/logs" "$step/fct"
    set_status "$step" running "force_action=${ACTIONS[$index]}"
    conf="$ROOT/simulation/mix/acc_${TASK}.conf"
    FORCE_ACTION="${ACTIONS[$index]}" EVAL_GREEDY=1 EVAL_TAG="stage0_action_$((index + 1))" \
        run_ns3_agent_once acc "$CONFIG" "$conf" 1 "action_$((index + 1))" "$step/logs"
    fct="$(fct_path_from_conf "$conf")"
    [[ -s "$fct" ]] || die "missing stage0 FCT: $fct"
    cp -f "$fct" "$step/fct/$(basename "$fct")"
    printf '%s\n' "${ACTIONS[$index]}" | atomic_write "$step/action"
    mark_done "$step"
done

args=()
for index in "${!ACTIONS[@]}"; do
    file="$(find "$RUN_DIR/actions/action_$((index + 1))/fct" -maxdepth 1 -type f -name '*.fct' -print -quit)"
    [[ -n "$file" ]] || die "archived FCT missing for action $((index + 1))"
    args+=("action_$((index + 1))=$file")
done
python3 "$ROOT/scripts/analyze_fct.py" "${args[@]}" --json "$RUN_DIR/summary/fct.json" >"$RUN_DIR/summary/fct.txt"

python3 - "$RUN_DIR/summary/fct.json" "${STAGE0_MIN_REL_DIFF:-0.01}" <<'PY'
import json, sys
p, threshold = sys.argv[1], float(sys.argv[2])
data = json.load(open(p))
rows = list(data.values()) if isinstance(data, dict) else data
values = []
for row in rows:
    for key in ("avg_slowdown", "avg_fct", "avg_fct_us", "mean_fct"):
        if key in row:
            values.append(float(row[key])); break
if len(values) < 2:
    raise SystemExit("stage0 gate failed: analyzer did not provide two comparable means")
base = max(abs(values[0]), 1e-12)
relative = max(abs(v - values[0]) / base for v in values[1:])
print(f"max_relative_difference={relative:.6f} threshold={threshold:.6f}")
if relative < threshold:
    raise SystemExit("stage0 gate failed: reward/FCT response is action-insensitive")
PY

mark_done "$RUN_DIR"
log "stage0 gate passed: $RUN_DIR"
