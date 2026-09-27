#!/usr/bin/env bash
# Execute exactly one ACC or SOR agent epoch. The ABA scheduler owns iteration.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: bash copter/run_100.sh --mode ACC|SOR --phase a1|b|a2 --epoch N
  --port N --exp-name NAME --model-dir DIR --run-id ID --log FILE [options]
EOF
}

MODE="" PHASE="" EPOCH="" PORT="" EXP_NAME="" MODEL_DIR="" RUN_ID="" LOG=""
EXPERIMENT_KIND=formal
SEED=1 BUFFER_KB=400 ACTION_SPACE=multiscale HIDDEN_DIMS="32,64,64,32"
REWARD_PROFILE=tail_safe REWARD_QUEUE_LAMBDA=5 REWARD_ECN_LAMBDA=5
REWARD_WEIGHTS="0.50,0.30,0.20" EPSILON_START=1 EPSILON_END=0.05
EPSILON_DECAY=2500 EPSILON_SCHEDULE=global TARGET_INTERVAL=100 SHARED_REPLAY=false
WATCH_PORTS="" WATCH_TRACE="" FCT_SOURCE="" FCT_TRACE="" TB_ENABLE=false RESUME=false
SOR_RECENT=200 SOR_BOUNDARY=1000 SOR_CLUSTERS=32 SOR_LAMBDA_CONS=0.01
SOR_LAMBDA_REG=0.001 SOR_REF_INTERVAL=256 SOR_SYNC_INTERVAL=8 SOR_SAVE_EVERY=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;; --phase) PHASE="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;; --port) PORT="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;; --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;; --log) LOG="$2"; shift 2 ;;
    --experiment-kind) EXPERIMENT_KIND="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;; --buffer-kb) BUFFER_KB="$2"; shift 2 ;;
    --action-space) ACTION_SPACE="$2"; shift 2 ;; --hidden-dims) HIDDEN_DIMS="$2"; shift 2 ;;
    --reward-profile) REWARD_PROFILE="$2"; shift 2 ;;
    --reward-queue-lambda) REWARD_QUEUE_LAMBDA="$2"; shift 2 ;;
    --reward-ecn-lambda) REWARD_ECN_LAMBDA="$2"; shift 2 ;;
    --reward-weights) REWARD_WEIGHTS="$2"; shift 2 ;;
    --epsilon-start) EPSILON_START="$2"; shift 2 ;; --epsilon-end) EPSILON_END="$2"; shift 2 ;;
    --epsilon-decay-steps) EPSILON_DECAY="$2"; shift 2 ;;
    --epsilon-schedule) EPSILON_SCHEDULE="$2"; shift 2 ;;
    --target-update-interval) TARGET_INTERVAL="$2"; shift 2 ;;
    --shared-replay) SHARED_REPLAY="$2"; shift 2 ;;
    --watch-ports) WATCH_PORTS="$2"; shift 2 ;; --watch-trace-file) WATCH_TRACE="$2"; shift 2 ;;
    --fct-source-file) FCT_SOURCE="$2"; shift 2 ;; --fct-step-trace-file) FCT_TRACE="$2"; shift 2 ;;
    --tb-enable) TB_ENABLE="$2"; shift 2 ;; --resume) RESUME=true; shift ;;
    --sor-recent-size) SOR_RECENT="$2"; shift 2 ;; --sor-boundary-size) SOR_BOUNDARY="$2"; shift 2 ;;
    --sor-max-clusters) SOR_CLUSTERS="$2"; shift 2 ;;
    --sor-lambda-cons) SOR_LAMBDA_CONS="$2"; shift 2 ;; --sor-lambda-reg) SOR_LAMBDA_REG="$2"; shift 2 ;;
    --sor-ref-update-interval) SOR_REF_INTERVAL="$2"; shift 2 ;;
    --sor-sync-interval) SOR_SYNC_INTERVAL="$2"; shift 2 ;;
    --sor-save-buffer-every) SOR_SAVE_EVERY="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;; *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "$MODE" == ACC || "$MODE" == SOR ]] || { echo "--mode must be ACC or SOR" >&2; exit 2; }
[[ "$PHASE" == a1 || "$PHASE" == b || "$PHASE" == a2 ]] || { echo "invalid phase" >&2; exit 2; }
for value in EPOCH PORT EXP_NAME MODEL_DIR RUN_ID LOG; do
  [[ -n "${!value}" ]] || { echo "Missing required argument ($value)" >&2; usage; exit 2; }
done
[[ "$EPSILON_SCHEDULE" == global ]] || { echo "Formal ABA runs require global epsilon" >&2; exit 2; }
if [[ "$EXPERIMENT_KIND" == formal ]]; then
  [[ "$REWARD_PROFILE" == tail_safe ]] || { echo "Formal ABA runs require tail_safe reward" >&2; exit 2; }
  [[ "$ACTION_SPACE" == multiscale ]] || { echo "Formal ABA runs require multiscale actions" >&2; exit 2; }
elif [[ "$EXPERIMENT_KIND" == acc_pilot ]]; then
  [[ "$MODE" == ACC ]] || { echo "ACC pilot only supports ACC" >&2; exit 2; }
  [[ "$REWARD_PROFILE" == tail_safe || "$REWARD_PROFILE" == weighted ]] || { echo "ACC pilot reward must be tail_safe or weighted" >&2; exit 2; }
  [[ "$ACTION_SPACE" == multiscale || "$ACTION_SPACE" == factorized_interp ]] || { echo "ACC pilot action space must be multiscale or factorized_interp" >&2; exit 2; }
else
  echo "--experiment-kind must be formal or acc_pilot" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL_DIR=$(mkdir -p "$MODEL_DIR" && cd "$MODEL_DIR" && pwd)
mkdir -p "$(dirname "$LOG")"
LOG=$(cd "$(dirname "$LOG")" && pwd)/$(basename "$LOG")
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi

COMMON=(
  -p "$PORT" -e "$EXP_NAME" -d "$MODEL_DIR" --online
  -b "$BUFFER_KB" --seed "$SEED" --run_id "$RUN_ID" --phase "$PHASE"
  --action_space "$ACTION_SPACE" --acc_hidden_dims "$HIDDEN_DIMS"
  --reward_profile "$REWARD_PROFILE" --reward_queue_lambda "$REWARD_QUEUE_LAMBDA"
  --reward_ecn_lambda "$REWARD_ECN_LAMBDA" --reward_weights "$REWARD_WEIGHTS"
  --epsilon_start "$EPSILON_START" --epsilon_end "$EPSILON_END"
  --epsilon_decay_steps "$EPSILON_DECAY" --epsilon_schedule "$EPSILON_SCHEDULE"
  --target_update_interval "$TARGET_INTERVAL" --state_save_interval 1
  --watch_ports "$WATCH_PORTS" --watch_trace_file "$WATCH_TRACE"
  --fct_source_file "$FCT_SOURCE" --fct_step_trace_file "$FCT_TRACE"
  --launcher_episode "$EPOCH" --tb_enable "$TB_ENABLE"
)

if [[ "$MODE" == ACC ]]; then
  EXTRA=(--shared_replay "$SHARED_REPLAY")
  [[ "$RESUME" == true ]] && EXTRA+=(--resume)
  cd "$ROOT/copter"
  exec "$PYTHON_BIN" copter.py "${COMMON[@]}" "${EXTRA[@]}" >"$LOG" 2>&1
else
  cd "$ROOT/sor"
  exec "$PYTHON_BIN" sor_copter.py "${COMMON[@]}" \
    --sor_recent_size "$SOR_RECENT" --sor_boundary_size "$SOR_BOUNDARY" \
    --sor_max_clusters "$SOR_CLUSTERS" --sor_lambda_cons "$SOR_LAMBDA_CONS" \
    --sor_lambda_reg "$SOR_LAMBDA_REG" --sor_ref_update_interval "$SOR_REF_INTERVAL" \
    --sor_sync_interval "$SOR_SYNC_INTERVAL" --sor_save_buffer_every "$SOR_SAVE_EVERY" \
    >"$LOG" 2>&1
fi
