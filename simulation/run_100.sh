#!/usr/bin/env bash
# Execute exactly one ns-3 epoch. The ABA scheduler owns all iteration.
set -euo pipefail

usage() {
  echo "Usage: bash simulation/run_100.sh --config FILE --port N --epoch N --log FILE" >&2
}

CONFIG=""
PORT=""
EPOCH=""
LOG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --epoch) EPOCH="$2"; shift 2 ;;
    --log) LOG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$CONFIG" && -n "$PORT" && -n "$EPOCH" && -n "$LOG" ]] || { usage; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Configuration not found: $CONFIG" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { echo "Invalid port: $PORT" >&2; exit 2; }
[[ "$EPOCH" =~ ^[0-9]+$ ]] || { echo "Invalid epoch: $EPOCH" >&2; exit 2; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
mkdir -p "$(dirname "$LOG")"
LOG=$(cd "$(dirname "$LOG")" && pwd)/$(basename "$LOG")

echo "[ns3] epoch=$EPOCH port=$PORT config=$CONFIG"
cd "$SCRIPT_DIR"
exec ./run-copter-sim.sh "$CONFIG" --port="$PORT" >"$LOG" 2>&1
