#!/bin/bash
# Run ACC baseline and SOR comparison with the same ns-3 config.
# All ns3 / agent stdout & stderr are streamed live to the terminal
# (prefixed by [<name>-ns3] / [<name>-agent]). Nothing is written to
# *.log files anymore.
#
# Usage:
#   run_acc_sor_ns3_compare.sh                # run ACC, then SOR, then compare
#   run_acc_sor_ns3_compare.sh --only acc     # only ACC
#   run_acc_sor_ns3_compare.sh --only sor     # only SOR
#   run_acc_sor_ns3_compare.sh --only both    # same as no flag
#   run_acc_sor_ns3_compare.sh --no-compare   # skip the final compare step
set -u

ONLY="both"
RUN_COMPARE=1
while [ $# -gt 0 ]; do
    case "$1" in
        --only)
            shift
            ONLY="${1:-both}"
            ;;
        --only=*)
            ONLY="${1#*=}"
            ;;
        --no-compare)
            RUN_COMPARE=0
            ;;
        -h|--help)
            sed -n '2,12p' "$0"
            exit 0
            ;;
        *)
            echo "[ERROR] unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done
case "$ONLY" in
    acc|sor|both) ;;
    *) echo "[ERROR] --only must be one of: acc, sor, both (got: $ONLY)" >&2; exit 2 ;;
esac

cleanup() {
    if [ -n "${ns3_pid:-}" ] && kill -0 "$ns3_pid" 2>/dev/null; then
        kill "$ns3_pid" 2>/dev/null || true
        wait "$ns3_pid" 2>/dev/null || true
    fi
    if [ -n "${agent_pid:-}" ] && kill -0 "$agent_pid" 2>/dev/null; then
        kill "$agent_pid" 2>/dev/null || true
        wait "$agent_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

REPO_ROOT="/root/paddlejob/workspace/yangziwen/CoPTER"
CONDA_BIN="/root/miniconda3/bin/conda"
ACC_CONFIG="$REPO_ROOT/sor/acc_eval_config.yaml"
SOR_CONFIG="$REPO_ROOT/sor/sor_eval_config.yaml"
NS3_SCRIPT="$REPO_ROOT/simulation/run_train.sh"
ACC_AGENT_SCRIPT="$REPO_ROOT/copter/run_train.sh"
SOR_AGENT_SCRIPT="$REPO_ROOT/sor/run_sor_train.sh"
COMPARE_SCRIPT="$REPO_ROOT/sor/compare_acc_sor.py"

if [ ! -x "$CONDA_BIN" ]; then
    echo "[ERROR] conda not found at $CONDA_BIN" >&2
    exit 1
fi

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx m3; then
    echo "[INFO] creating conda env m3 from $REPO_ROOT/m3_environment.yml"
    "$CONDA_BIN" env create -f "$REPO_ROOT/m3_environment.yml"
fi

# Pre-flight: refuse to start if port 5558 is already taken or stale
# copter-sim / agent processes from previous runs are still around. Stale
# processes silently corrupt the experiment (multiple ns3 servers fight
# for the same port -> agent never makes progress -> metrics file stops
# growing).
preflight_check() {
    local stale=0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq '[:.]5558$'; then
        echo "[ERROR] port 5558 is already in use:" >&2
        ss -ltnp 2>/dev/null | grep -E '[:.]5558\b' >&2 || true
        stale=1
    fi
    local leftover
    leftover=$(pgrep -af 'copter-sim|copter\.py|sor_copter\.py|simulation/run_train\.sh|copter/run_train\.sh|sor/run_sor_train\.sh' \
               | grep -v "$$" || true)
    if [ -n "$leftover" ]; then
        echo "[ERROR] stale CoPTER processes detected:" >&2
        echo "$leftover" >&2
        stale=1
    fi
    if [ "$stale" -ne 0 ]; then
        cat >&2 <<'EOF'
[HINT] Clean up before re-running, e.g.:
  pkill -f run_acc_sor_ns3_compare
  pkill -f "simulation/run_train.sh|copter/run_train.sh|sor/run_sor_train.sh"
  pkill -f "ns-3.33/build/scratch/copter-sim"
  pkill -f "copter\.py|sor_copter\.py"
EOF
        exit 1
    fi
}
preflight_check

read_yaml_int() {
    # $1 = yaml path, $2 = key name (top-level scalar), $3 = default
    "$CONDA_BIN" run --no-capture-output -n m3 python - "$1" "$2" "$3" <<'PY' 2>/dev/null
import sys, yaml
path, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    print(int(cfg.get(key, default)))
except Exception:
    print(default)
PY
}

wait_port_free() {
    # Wait until TCP port $1 is no longer in LISTEN/ESTABLISHED/TIME_WAIT.
    # $2 = max wait seconds (default 60).
    local port="$1"
    local max="${2:-60}"
    local i=0
    while [ "$i" -lt "$max" ]; do
        if ! ss -tan 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$port$"; then
            return 0
        fi
        sleep 1
        i=$((i+1))
    done
    echo "[WARN] port $port still occupied after ${max}s; proceeding anyway" >&2
    ss -tanp 2>/dev/null | grep -E "[:.]$port\b" >&2 || true
    return 1
}

run_pair() {
    local name="$1"
    local config="$2"
    local agent_script="$3"
    local ns3_tag="[${name}-ns3]"
    local agent_tag="[${name}-agent]"

    local total_epochs
    total_epochs=$(read_yaml_int "$config" "total_epochs" 20)
    local inter_sleep
    inter_sleep=$(read_yaml_int "$config" "ns3_inter_run_sleep" 5)
    local ns3_port
    ns3_port=$(read_yaml_int "$config" "ns3_port" 5558)

    echo "=============================="
    echo "[COMPARE] running $name"
    echo "[COMPARE] config       : $config"
    echo "[COMPARE] total_epochs : $total_epochs"
    echo "[COMPARE] inter_sleep  : ${inter_sleep}s between epochs"
    echo "[COMPARE] ns3_port     : $ns3_port"
    echo "[COMPARE] (each epoch = 1 fresh ns3 + 1 fresh agent, orchestrated by parent)"
    echo "=============================="

    local epoch
    for ((epoch = 1; epoch <= total_epochs; epoch++)); do
        echo "[COMPARE] $name --- epoch $epoch / $total_epochs ---"

        # 1. Start agent first so it binds the port before ns3 tries to connect.
        TRAIN_CONFIG="$config" TOTAL_EPOCHS_OVERRIDE=1 \
            "$CONDA_BIN" run --no-capture-output -n m3 \
            bash "$agent_script" > >(sed -u "s|^|$agent_tag |") 2>&1 &
        agent_pid=$!

        # 2. Wait until agent has bound ns3_port (LISTEN), give up after 60s.
        local waited=0
        while [ "$waited" -lt 60 ]; do
            if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$ns3_port$"; then
                break
            fi
            if ! kill -0 "$agent_pid" 2>/dev/null; then
                echo "[ERROR] $name agent died before binding port $ns3_port" >&2
                wait "$agent_pid" || true
                agent_pid=""
                exit 1
            fi
            sleep 1
            waited=$((waited+1))
        done
        if [ "$waited" -ge 60 ]; then
            echo "[ERROR] $name agent failed to bind port $ns3_port within 60s" >&2
            kill "$agent_pid" 2>/dev/null || true
            wait "$agent_pid" || true
            agent_pid=""
            exit 1
        fi
        echo "[COMPARE] $name agent bound port $ns3_port (pid=$agent_pid, after ${waited}s)"

        # 3. Start ns3 now that the agent is listening.
        TRAIN_CONFIG="$config" TOTAL_EPOCHS_OVERRIDE=1 \
            "$CONDA_BIN" run --no-capture-output -n m3 \
            bash "$NS3_SCRIPT" > >(sed -u "s|^|$ns3_tag |") 2>&1 &
        ns3_pid=$!
        echo "[COMPARE] $name ns3 started (pid=$ns3_pid)"

        # 4. Wait for both to finish.
        wait "$agent_pid"
        local agent_rc=$?
        agent_pid=""
        local ns3_rc=0
        if kill -0 "$ns3_pid" 2>/dev/null; then
            wait "$ns3_pid" || ns3_rc=$?
        else
            wait "$ns3_pid" || ns3_rc=$?
        fi
        ns3_pid=""
        [ "$ns3_rc" -eq 143 ] && ns3_rc=0
        echo "[COMPARE] $name epoch $epoch done agent_rc=$agent_rc ns3_rc=$ns3_rc"
        if [ "$agent_rc" -ne 0 ] || [ "$ns3_rc" -ne 0 ]; then
            echo "[ERROR] $name epoch $epoch failed" >&2
            exit 1
        fi

        # 5. Drain port (TIME_WAIT) before next epoch and apply inter-sleep.
        if [ "$epoch" -lt "$total_epochs" ]; then
            wait_port_free "$ns3_port" 90 || true
            sleep "$inter_sleep"
        fi
    done

    echo "[COMPARE] $name finished all $total_epochs epochs"
}

cd "$REPO_ROOT"
if [ ! -x "$REPO_ROOT/ns-3.33/build/scratch/copter-sim" ]; then
    echo "[INFO] ns3 binary missing; build it with build_ns3_copter.sh first or run manually if sudo is required."
fi

run_acc=0; run_sor=0
case "$ONLY" in
    acc)  run_acc=1 ;;
    sor)  run_sor=1 ;;
    both) run_acc=1; run_sor=1 ;;
esac

if [ "$run_acc" -eq 1 ]; then
    run_pair "acc_eval" "$ACC_CONFIG" "$ACC_AGENT_SCRIPT"
fi
if [ "$run_acc" -eq 1 ] && [ "$run_sor" -eq 1 ]; then
    sleep 5
fi
if [ "$run_sor" -eq 1 ]; then
    run_pair "sor_eval" "$SOR_CONFIG" "$SOR_AGENT_SCRIPT"
fi

if [ "$RUN_COMPARE" -eq 1 ]; then
    "$CONDA_BIN" run --no-capture-output -n m3 python "$COMPARE_SCRIPT"
else
    echo "[COMPARE] --no-compare set, skipping final comparison."
fi
