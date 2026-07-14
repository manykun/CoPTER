#!/usr/bin/env bash
set -Eeuo pipefail
method="${1:?method}"; flow="${2:?flow}"; output_fct="${3:?output_fct}"; output_metrics="${4:?output_metrics}"; seed="${5:-1}"; checkpoint="${6:-}"
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER; PY=/root/miniconda3/envs/m3/bin/python; BIN="$ROOT/ns-3.33/build/scratch/copter-sim"; PORT="${EVAL_PORT:-5611}"
scenario=A; [[ "$flow" == *scenB* ]] && scenario=B
if [[ "$method" != ACC ]]; then exec "$ROOT/scripts_exp/static_runner.sh" "$scenario" "$method" "$flow" "$output_fct" "$seed"; fi
conf="mix/acc_scen${scenario}.conf"; model_dir="$(dirname "$checkpoint")"
model0="$(find "$model_dir" -maxdepth 1 -type f -name '*_ACC_0' -print -quit)"
[[ -n "$model0" ]] || { echo "missing ACC port-0 checkpoint in $model_dir" >&2; exit 2; }
exp="$(basename "$model0")"; exp="${exp%_ACC_0}"
mkdir -p "$(dirname "$output_fct")" "$(dirname "$output_metrics")"; export LD_LIBRARY_PATH="$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}"; unset NS_LOG
(cd "$ROOT/simulation" && "$BIN" "$conf" --port="$PORT") >"${output_fct}.ns3.log" 2>&1 & ns=$!; trap 'kill $ns 2>/dev/null || true' EXIT; sleep 5
(cd "$ROOT/copter" && "$PY" copter.py -p "$PORT" -e "$exp" -r "$exp" --eval_greedy -m ACC -d "$model_dir" -b 10000 --seed "$seed" --run_id "$exp" --phase eval_A --eval_tag eval_A --tb_enable false) >"${output_fct}.agent.log" 2>&1
wait "$ns"; trap - EXIT
source_fct="$ROOT/simulation/output/scen${scenario}/acc_scen${scenario}.fct"; [[ -s "$source_fct" ]] || { echo "missing FCT: $source_fct" >&2; exit 2; }; cp -f "$source_fct" "$output_fct"
metrics="$model_dir/${exp}_metrics.jsonl"; [[ -s "$metrics" ]] && tail -n 1 "$metrics" > "$output_metrics" || printf '{"label":"acc","reward":0}\n' > "$output_metrics"
