#!/usr/bin/env bash
set -Eeuo pipefail
scenario="${1:?scenario}"; candidate="${2:?candidate}"; flow="${3:?flow}"; output="${4:?output_fct}"; seed="${5:-1}"
ROOT=/root/paddlejob/workspace/yangziwen/CoPTER
BIN="$ROOT/ns-3.33/build/scratch/copter-sim"
scenario="$(printf '%s' "$scenario" | tr '[:lower:]' '[:upper:]')"
candidate_lc="$(printf '%s' "$candidate" | tr '[:upper:]' '[:lower:]')"
conf="$ROOT/simulation/mix/static_scen${scenario}_${candidate_lc}.conf"
[[ "$flow" = /* ]] || flow="$ROOT/$flow"
[[ -x "$BIN" ]] || { echo "missing simulator: $BIN" >&2; exit 2; }
[[ -s "$conf" && -s "$flow" ]] || { echo "missing conf or flow: $conf $flow" >&2; exit 2; }
mkdir -p "$(dirname "$output")" "$ROOT/simulation/output/static"
export LD_LIBRARY_PATH="$ROOT/ns-3.33/build/lib:${LD_LIBRARY_PATH:-}"
unset NS_LOG
(cd "$ROOT/simulation" && "$BIN" "mix/$(basename "$conf")")
source_fct="$ROOT/simulation/output/static/static_scen${scenario}_${candidate_lc}.fct"
[[ -s "$source_fct" ]] || { echo "simulator produced no FCT: $source_fct" >&2; exit 2; }
cp -f "$source_fct" "$output"
