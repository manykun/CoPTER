#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_common_args "$@"
require_run_dir
STAGE="$RUN_DIR/stage0"
mkdir -p "$STAGE"/{logs,fct,summaries}
STATUS_FILE="$RUN_DIR/status.json"
write_status stage0 running "static ECN sensitivity"
[[ "$RESUME" == 1 && -f "$STAGE/.done" ]] && exit 0
: "${SCENARIO_A_FLOW:?Set SCENARIO_A_FLOW to the throughput-sensitive eval trace}"
: "${SCENARIO_B_FLOW:?Set SCENARIO_B_FLOW to the incast-sensitive eval trace}"
for f in "$SCENARIO_A_FLOW" "$SCENARIO_B_FLOW"; do [[ -s "$f" ]] || die "missing flow trace: $f"; done
# The simulator command is supplied explicitly because local copter-sim builds expose different flow-file flags.
: "${STATIC_RUNNER:?Set STATIC_RUNNER to a command accepting: scenario candidate flow output_fct seed}"
METRICS="$STAGE/summaries/static_metrics.jsonl"
: > "$METRICS"
fct_args=()
for scenario in A B; do
  flow_var="SCENARIO_${scenario}_FLOW"; flow="${!flow_var}"
  for candidate in SECN0 SECN1 SECN2; do
    out="$STAGE/fct/${scenario}_${candidate}.fct"
    if [[ ! ( "$RESUME" == 1 && -s "$out" ) ]]; then
      timeout "${UNIT_TIMEOUT:-14400}" bash -lc "$STATIC_RUNNER '$scenario' '$candidate' '$flow' '$out' '$SEED'" >"$STAGE/logs/${scenario}_${candidate}.log" 2>&1
    fi
    [[ -s "$out" ]] || die "static run produced no FCT: $scenario/$candidate"
    fct_args+=(--fct "${scenario}:${candidate}=$out")
    printf '{"scenario":"%s","candidate":"%s","reward":0}\n' "$scenario" "$candidate" >> "$METRICS"
  done
done
python "$SCRIPTS_EXP/check_stage0_gate.py" \
  --input "$METRICS" \
  --output-prefix "$STAGE/summaries/stage0_gate" \
  --flow "A=$SCENARIO_A_FLOW" \
  --flow "B=$SCENARIO_B_FLOW" \
  "${fct_args[@]}"
touch_atomic "$STAGE/.done"
write_status stage0 completed "sensitivity gate passed"
