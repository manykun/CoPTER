#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
STAGE="all"
ABA_RUN_ID="aba_web_cache_s1"
CDC_RUN_ID="cdc_hadoop_alistorage_s1"
RESUME=0

usage() {
  cat <<'EOF'
Usage: bash scripts/continual_validation/run_aba_cdc.sh [options]

Runs two independent continual-learning routes concurrently. Each route runs
ACC and SOR concurrently, for four isolated experiment arms in total.

Options:
  --stage prepare|run|analyze|all|smoke
  --aba-run-id ID
  --cdc-run-id ID
  --resume
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --aba-run-id) ABA_RUN_ID="$2"; shift 2 ;;
    --cdc-run-id) CDC_RUN_ID="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$STAGE" in
  prepare|run|analyze|all|smoke) ;;
  *) echo "invalid stage: $STAGE" >&2; exit 2 ;;
esac
[[ "$ABA_RUN_ID" != "$CDC_RUN_ID" ]] || {
  echo "ABA and CDC run IDs must differ" >&2
  exit 2
}

mkdir -p "$ROOT/experiments/aba"

start_route() {
  local label="$1" config="$2" run_id="$3"
  ROUTE_LOG="$ROOT/experiments/aba/${run_id}_route.log"
  local command=(
    bash "$ROOT/scripts/continual_validation/run_aba.sh"
    --config "$config"
    --stage "$STAGE"
    --run-id "$run_id"
  )
  [[ "$RESUME" -eq 0 ]] || command+=(--resume)
  "${command[@]}" >"$ROUTE_LOG" 2>&1 &
  ROUTE_PID=$!
  echo "$label route PID=$ROUTE_PID log=$ROUTE_LOG"
}

start_route ABA "$ROOT/configs/aba/webserver_cachefollower.yaml" "$ABA_RUN_ID"
ABA_PID=$ROUTE_PID
ABA_LOG=$ROUTE_LOG
start_route CDC "$ROOT/configs/aba/hadoop_alistorage.yaml" "$CDC_RUN_ID"
CDC_PID=$ROUTE_PID
CDC_LOG=$ROUTE_LOG

cleanup() {
  kill "$ABA_PID" "$CDC_PID" 2>/dev/null || true
}
trap cleanup INT TERM

echo "Four arms: ABA-ACC, ABA-SOR, CDC-ACC, CDC-SOR"

status=0
wait "$ABA_PID" || { echo "ABA route failed; inspect $ABA_LOG" >&2; status=1; }
wait "$CDC_PID" || { echo "CDC route failed; inspect $CDC_LOG" >&2; status=1; }
trap - INT TERM

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
echo "ABA and CDC routes completed successfully"
