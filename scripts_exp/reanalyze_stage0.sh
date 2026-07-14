#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
RUN_DIR="${1:-}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find "$ROOT/experiments" -mindepth 1 -maxdepth 1 -type d -name 'experiment_*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
[[ -n "$RUN_DIR" && -d "$RUN_DIR/stage0/fct" ]] || {
  echo "usage: reanalyze_stage0.sh [RUN_DIR]" >&2
  exit 2
}

STAGE="$RUN_DIR/stage0"
METRICS="$STAGE/summaries/static_metrics.jsonl"
FLOW_A="$ROOT/simulation/mix/scenA_throughput.flow"
FLOW_B="$ROOT/simulation/mix/scenB_incast.flow"
OUTPUT="$STAGE/summaries/stage0_gate_completion_aware"

for path in "$METRICS" "$FLOW_A" "$FLOW_B"; do
  [[ -s "$path" ]] || { echo "missing input: $path" >&2; exit 2; }
done

args=(
  --input "$METRICS"
  --output-prefix "$OUTPUT"
  --flow "A=$FLOW_A"
  --flow "B=$FLOW_B"
)
for scenario in A B; do
  for candidate in SECN0 SECN1 SECN2; do
    path="$STAGE/fct/${scenario}_${candidate}.fct"
    [[ -s "$path" ]] || { echo "missing FCT: $path" >&2; exit 2; }
    args+=(--fct "${scenario}:${candidate}=$path")
  done
done

python "$HERE/check_stage0_gate.py" "${args[@]}"
printf '%s\n' "$(date -Is)" >"$STAGE/.done.tmp.$$"
mv -f "$STAGE/.done.tmp.$$" "$STAGE/.done"
printf 'stage=stage0\nstate=completed\ndetail=completion-aware sensitivity gate passed\nupdated_at=%s\npid=%s\n' \
  "$(date --iso-8601=seconds)" "$$" >"$RUN_DIR/status.tmp.$$"
mv -f "$RUN_DIR/status.tmp.$$" "$RUN_DIR/status"

printf '\nStage 0 completion-aware reanalysis passed.\n'
printf 'Report: %s.json\n' "$OUTPUT"
printf 'Resume full pipeline with:\n'
printf 'nohup bash scripts_exp/run_full_experiment.sh --run-dir %q --resume > full_experiment_resume.log 2>&1 &\n' "$RUN_DIR"
