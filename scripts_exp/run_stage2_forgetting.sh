#!/usr/bin/env bash
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
parse_common_args "$@"; require_run_dir
[[ -f "$RUN_DIR/stage0/.done" ]] || die "stage0 gate has not passed"
STAGE="$RUN_DIR/stage2"; mkdir -p "$STAGE"/{logs,metrics,fct,checkpoints,summaries,plots,state}
write_status stage2 running "A-B-A forgetting and SOR"
[[ "$RESUME" == 1 && -f "$STAGE/.done" ]] && exit 0
: "${CURRICULUM_RUNNER:?Set CURRICULUM_RUNNER to a command accepting method phase flow run_dir seed epochs resume}"
: "${SCENARIO_A_TRAIN_FLOW:?Set SCENARIO_A_TRAIN_FLOW}"; : "${SCENARIO_B_TRAIN_FLOW:?Set SCENARIO_B_TRAIN_FLOW}"
epochs="${PHASE_EPOCHS:-30}"; [[ "$SMOKE" == 1 ]] && epochs=1
for method in ACC SOR; do
  for spec in p1:A p2:B p3:A; do
    phase="${spec%%:*}"; task="${spec##*:}"; flow_var="SCENARIO_${task}_TRAIN_FLOW"; flow="${!flow_var}"
    marker="$STAGE/state/${method}_${phase}.done"
    if [[ ! ( "$RESUME" == 1 && -f "$marker" ) ]]; then
      bash -lc "$CURRICULUM_RUNNER '$method' '$phase' '$flow' '$STAGE' '$SEED' '$epochs' '$RESUME'" >"$STAGE/logs/${method}_${phase}.log" 2>&1
      # The runner must archive both task evaluations before returning.
      [[ -s "$STAGE/metrics/${method}_${phase}_eval_A.jsonl" && -s "$STAGE/fct/${method}_${phase}_eval_A.fct" ]] || die "$method/$phase missing mandatory A evaluation"
      [[ -s "$STAGE/metrics/${method}_${phase}_eval_B.jsonl" && -s "$STAGE/fct/${method}_${phase}_eval_B.fct" ]] || die "$method/$phase missing mandatory B evaluation"
      touch_atomic "$marker"
    fi
  done
done
combined="$STAGE/metrics/combined_eval.jsonl"
python - "$STAGE" "$combined" <<'PY'
import glob, json, os, sys
stage, output = sys.argv[1:]
with open(output, "w") as dst:
    for method in ("ACC", "SOR"):
        for phase, when in (("p1", "before"), ("p2", "after"), ("p3", "recovery")):
            path = os.path.join(stage, "metrics", "%s_%s_eval_A.jsonl" % (method, phase))
            with open(path) as src:
                rows = [json.loads(line) for line in src if line.strip()]
            if not rows:
                raise SystemExit("empty metrics: " + path)
            row = rows[-1]
            row.update({"method": method.lower(), "phase": when, "task": "A"})
            dst.write(json.dumps(row) + "\n")
PY
fct_args=()
for method in ACC SOR; do fct_args+=(--fct "${method,,}:before=$STAGE/fct/${method}_p1_eval_A.fct" --fct "${method,,}:after=$STAGE/fct/${method}_p2_eval_A.fct"); done
python "$SCRIPTS_EXP/analyze_forgetting.py" --metrics "$combined" --output-prefix "$STAGE/summaries/forgetting" "${fct_args[@]}"
touch_atomic "$STAGE/.done"; write_status stage2 completed "ACC forgetting and SOR mitigation gates passed"
