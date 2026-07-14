#!/usr/bin/env bash
# Shared experiment orchestration primitives. Source this file; do not execute it.
set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/root/paddlejob/workspace/yangziwen/CoPTER}"
SCRIPTS_EXP="$REPO_ROOT/scripts_exp"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$REPO_ROOT/experiments}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
NS3_LAUNCHER="$REPO_ROOT/simulation/run_train.sh"
ACC_LAUNCHER="$REPO_ROOT/copter/run_train.sh"
SOR_LAUNCHER="$REPO_ROOT/sor/run_sor_train.sh"
DEFAULT_ACC_CONFIG="$REPO_ROOT/sor/acc_curriculum_config.yaml"
DEFAULT_SOR_CONFIG="$REPO_ROOT/sor/sor_curriculum_config.yaml"
SEED="${SEED:-1}"
RESUME=0
SMOKE=0
ORCH_NS3_PID=""
ORCH_AGENT_PID=""

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

atomic_write() {
    local target="$1" tmp
    tmp="${target}.tmp.$$"
    mkdir -p "$(dirname "$target")"
    cat >"$tmp"
    mv -f "$tmp" "$target"
}

set_status() {
    local dir="$1" state="$2" detail="${3:-}"
    printf 'state=%s\ndetail=%s\nupdated_at=%s\npid=%s\n' \
        "$state" "$detail" "$(date --iso-8601=seconds)" "$$" | atomic_write "$dir/status"
}

mark_done() {
    local dir="$1"
    printf 'completed_at=%s\n' "$(date --iso-8601=seconds)" | atomic_write "$dir/.done"
    set_status "$dir" done
}

step_is_done() { [[ -f "$1/.done" ]]; }

acquire_lock() {
    local root="$1"
    mkdir -p "$root"
    exec {ORCH_LOCK_FD}>"$root/.lock"
    flock -n "$ORCH_LOCK_FD" || die "experiment is already running: $root"
}

cleanup_children() {
    local rc=$?
    [[ -n "$ORCH_NS3_PID" ]] && kill "$ORCH_NS3_PID" 2>/dev/null || true
    [[ -n "$ORCH_AGENT_PID" ]] && kill "$ORCH_AGENT_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    ORCH_NS3_PID=""; ORCH_AGENT_PID=""
    return "$rc"
}

require_runtime() {
    command -v flock >/dev/null || die "flock is required"
    command -v ss >/dev/null || die "ss is required"
    [[ -x "$CONDA_BIN" ]] || die "conda executable missing: $CONDA_BIN"
    [[ -x "$REPO_ROOT/ns-3.33/build/scratch/copter-sim" ]] || die "ns3 binary missing"
    [[ -x "$NS3_LAUNCHER" ]] || die "ns3 launcher missing"
}

port_in_use() { ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$1$"; }
wait_for_port() {
    local port="$1" pid="$2" timeout="${3:-60}" elapsed=0
    while (( elapsed < timeout )); do
        port_in_use "$port" && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 1; ((elapsed+=1))
    done
    return 1
}
wait_port_free() {
    local port="$1" timeout="${2:-90}" elapsed=0
    while port_in_use "$port"; do
        (( elapsed >= timeout )) && return 1
        sleep 1; ((elapsed+=1))
    done
}

parse_common_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --run-dir) RUN_DIR="$2"; shift 2 ;;
            --seed) SEED="$2"; shift 2 ;;
            --resume) RESUME=1; shift ;;
            --smoke) SMOKE=1; shift ;;
            *) die "unknown argument: $1" ;;
        esac
    done
    export RUN_DIR SEED RESUME SMOKE
}
require_run_dir() { [[ -n "${RUN_DIR:-}" ]] || die "--run-dir is required"; }
touch_atomic() { atomic_write "$1" <<<"$(date -Is)"; }
write_status() {
    local stage="$1" state="$2" detail="${3:-}"
    printf 'stage=%s\nstate=%s\ndetail=%s\nupdated_at=%s\npid=%s\n' \
        "$stage" "$state" "$detail" "$(date --iso-8601=seconds)" "$$" | atomic_write "$RUN_DIR/status"
}
write_manifest() {
    local target="$RUN_DIR/manifest.json"
    python - "$target" "$SEED" "$SMOKE" <<'PY'
import json, os, sys, datetime
path, seed, smoke = sys.argv[1:]
data = {"created_at": datetime.datetime.now().astimezone().isoformat(), "seed": int(seed),
        "smoke": bool(int(smoke)), "run_dir": os.path.dirname(path)}
with open(path + ".tmp", "w") as f: json.dump(data, f, indent=2)
os.replace(path + ".tmp", path)
PY
}

render_config() {
    local source="$1" target="$2" exp_name="$3" model_dir="$4" seed="$5" port="$6"
    awk -v exp="$exp_name" -v model="$model_dir" -v seed="$seed" -v port="$port" '
        /^exp_name:/ {print "exp_name: " exp; seen_exp=1; next}
        /^model_dir:/ {print "model_dir: " model; seen_model=1; next}
        /^seed:/ {print "seed: " seed; seen_seed=1; next}
        /^ns3_port:/ {print "ns3_port: " port; seen_port=1; next}
        {print}
        END {
          if (!seen_exp) print "exp_name: " exp
          if (!seen_model) print "model_dir: " model
          if (!seen_seed) print "seed: " seed
          if (!seen_port) print "ns3_port: " port
        }
    ' "$source" | atomic_write "$target"
}

fct_path_from_conf() {
    local conf="$1" rel
    rel="$(awk '$1 == "FCT_OUTPUT_FILE" {print $2; exit}' "$conf")"
    [[ -n "$rel" ]] || return 1
    case "$rel" in /*) printf '%s\n' "$rel" ;; *) printf '%s/simulation/%s\n' "$REPO_ROOT" "$rel" ;; esac
}

run_ns3_agent_once() {
    local method="$1" config="$2" conf="$3" eval_mode="$4" tag="$5" log_dir="$6"
    local launcher port
    launcher="$ACC_LAUNCHER"; [[ "$method" == sor ]] && launcher="$SOR_LAUNCHER"
    port="$(awk '$1 == "ns3_port:" {print $2; exit}' "$config")"
    wait_port_free "$port" 90 || die "port $port remained busy"
    mkdir -p "$log_dir"

    TRAIN_CONFIG="$config" TOTAL_EPOCHS_OVERRIDE=1 NS3_CONF_OVERRIDE="$conf" \
        EVAL_GREEDY="${EVAL_GREEDY:-0}" EVAL_TAG="${EVAL_TAG:-}" FORCE_ACTION="${FORCE_ACTION:-}" \
        "$CONDA_BIN" run --no-capture-output -n m3 bash "$launcher" \
        >"$log_dir/agent.log" 2>&1 & ORCH_AGENT_PID=$!
    wait_for_port "$port" "$ORCH_AGENT_PID" 60 || { wait "$ORCH_AGENT_PID" || true; die "agent failed before binding port $port"; }

    TRAIN_CONFIG="$config" TOTAL_EPOCHS_OVERRIDE=1 NS3_CONF_OVERRIDE="$conf" \
        "$NS3_LAUNCHER" >"$log_dir/ns3.log" 2>&1 & ORCH_NS3_PID=$!

    local ns3_rc=0 agent_rc=0
    wait "$ORCH_NS3_PID" || ns3_rc=$?; ORCH_NS3_PID=""
    wait "$ORCH_AGENT_PID" || agent_rc=$?; ORCH_AGENT_PID=""
    (( ns3_rc == 0 && agent_rc == 0 )) || die "$tag failed (ns3=$ns3_rc agent=$agent_rc)"
    wait_port_free "$port" 90 || die "port $port did not become free after $tag"
}

run_train_or_eval() {
    local method="$1" config="$2" task="$3" mode="$4" step_dir="$5"
    local conf="$REPO_ROOT/simulation/mix/${method}_${task}.conf" fct
    [[ -f "$conf" ]] || die "missing ns3 config: $conf"
    mkdir -p "$step_dir/logs" "$step_dir/fct"
    set_status "$step_dir" running "$mode:$task"
    if [[ "$mode" == eval ]]; then
        EVAL_GREEDY=1 EVAL_TAG="$(basename "$step_dir")" run_ns3_agent_once "$method" "$config" "$conf" 1 "$mode:$task" "$step_dir/logs"
    else
        run_ns3_agent_once "$method" "$config" "$conf" 0 "$mode:$task" "$step_dir/logs"
    fi
    fct="$(fct_path_from_conf "$conf")" || die "FCT_OUTPUT_FILE absent in $conf"
    [[ -s "$fct" ]] || die "missing or empty final FCT after $mode:$task: $fct"
    cp -f "$fct" "$step_dir/fct/$(basename "$fct")"
    mark_done "$step_dir"
}
