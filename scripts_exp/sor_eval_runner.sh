#!/usr/bin/env bash
set -Eeuo pipefail
exp="${1:?exp}"; model="${2:?model}"; scenario="${3:?scenario}"; out="${4:?fct}"; metrics="${5:?metrics}"; seed="${6:-1}"
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER; PY=/root/miniconda3/envs/m3/bin/python; BIN="$ROOT/ns-3.33/build/scratch/copter-sim"; PORT="${EVAL_PORT:-5640}"
mkdir -p "$(dirname "$out")" "$(dirname "$metrics")"; export LD_LIBRARY_PATH="$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}"; unset NS_LOG
(cd "$ROOT/simulation" && "$BIN" "mix/acc_scen${scenario}.conf" --port="$PORT") >"${out}.ns3.log" 2>&1 & ns=$!; trap 'kill $ns 2>/dev/null || true' EXIT; sleep 5
(cd "$ROOT/sor" && "$PY" sor_copter.py -p "$PORT" -e "$exp" -r "$exp" -d "$model" -b 10000 --seed "$seed" --eval_greedy --eval_tag "eval_${scenario}" --tb_enable false) >"${out}.agent.log" 2>&1
wait "$ns"; trap - EXIT
src="$ROOT/simulation/output/scen${scenario}/acc_scen${scenario}.fct"; [[ -s "$src" ]] || { echo "missing FCT: $src" >&2; exit 2; }; cp -f "$src" "$out"
file="$model/${exp}_metrics.jsonl"; [[ -s "$file" ]] && tail -n 1 "$file" > "$metrics" || exit 2
