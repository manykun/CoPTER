#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/root/paddlejob/workspace/yangziwen/CoPTER"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$ROOT/experiments}"
RUN="${1:-}"
[[ -n "$RUN" ]] || { echo "usage: $0 RUN_ID|RUN_DIRECTORY" >&2; exit 2; }
case "$RUN" in /*) DIR="$RUN" ;; *) DIR="$EXPERIMENT_ROOT/$RUN" ;; esac
[[ -d "$DIR" ]] || { echo "experiment not found: $DIR" >&2; exit 1; }

printf 'experiment: %s\n' "$DIR"
if [[ -f "$DIR/status" ]]; then
    cat "$DIR/status"
else
    echo 'state=not_started'
fi
printf 'completed_steps=%s\n' "$(find "$DIR" -type f -name .done | wc -l)"
printf 'running_steps=%s\n' "$(grep -rl '^state=running$' "$DIR" --include=status 2>/dev/null | wc -l)"
printf 'failed_steps=%s\n' "$(grep -rl '^state=failed$' "$DIR" --include=status 2>/dev/null | wc -l)"

find "$DIR" -type f -name status -print0 | sort -z | while IFS= read -r -d '' status; do
    state="$(awk -F= '$1 == "state" {print $2; exit}' "$status")"
    printf '%-10s %s\n' "$state" "${status%/status}"
done
