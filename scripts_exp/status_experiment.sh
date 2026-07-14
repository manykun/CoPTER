#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find "$ROOT/experiments" -mindepth 1 -maxdepth 1 -type d -name 'experiment_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -n "$RUN_DIR" && -d "$RUN_DIR" ]] || { echo "usage: status_experiment.sh [RUN_DIR]" >&2; exit 2; }
printf 'run_dir: %s\n' "$RUN_DIR"
if [[ -f "$RUN_DIR/orchestrator.pid" ]]; then pid="$(<"$RUN_DIR/orchestrator.pid")"; if kill -0 "$pid" 2>/dev/null; then echo "process: running pid=$pid"; else echo "process: stopped pid=$pid"; fi; fi
[[ -f "$RUN_DIR/status" ]] && { echo 'status:'; sed 's/^/  /' "$RUN_DIR/status"; }
for stage in stage0 stage1 stage2; do [[ -f "$RUN_DIR/$stage/.done" ]] && state=completed || state=pending; printf '%s: %s\n' "$stage" "$state"; done
find "$RUN_DIR" -name '*.log' -type f -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- | while read -r log; do echo "latest_log: $log"; tail -n 5 "$log"; done
