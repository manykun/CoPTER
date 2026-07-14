#!/bin/bash
# Run ACC baseline and SOR comparison with the same ns-3 config.
set -u

cleanup() {
    if [ -n "${ns3_pid:-}" ] && kill -0 "$ns3_pid" 2>/dev/null; then
        kill "$ns3_pid" 2>/dev/null || true
        wait "$ns3_pid" 2>/dev/null || true
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

run_pair() {
    local name="$1"
    local config="$2"
    local agent_script="$3"
    local ns3_log="$REPO_ROOT/sor/${name}_ns3.log"
    local agent_log="$REPO_ROOT/sor/${name}_agent.log"

    echo "=============================="
    echo "[COMPARE] running $name"
    echo "=============================="
    TRAIN_CONFIG="$config" "$CONDA_BIN" run -n m3 bash "$NS3_SCRIPT" > "$ns3_log" 2>&1 &
    ns3_pid=$!
    sleep 5
    TRAIN_CONFIG="$config" "$CONDA_BIN" run -n m3 bash "$agent_script" > "$agent_log" 2>&1
    local agent_rc=$?
    local ns3_rc=0
    if kill -0 "$ns3_pid" 2>/dev/null; then
        kill "$ns3_pid" 2>/dev/null || true
    fi
    wait "$ns3_pid" || ns3_rc=$?
    ns3_pid=""
    if [ "$ns3_rc" -eq 143 ]; then
        ns3_rc=0
    fi
    echo "[COMPARE] $name agent_rc=$agent_rc ns3_rc=$ns3_rc"
    if [ "$agent_rc" -ne 0 ] || [ "$ns3_rc" -ne 0 ]; then
        echo "[ERROR] $name failed. Logs: $agent_log $ns3_log" >&2
        exit 1
    fi
}

cd "$REPO_ROOT"
if [ ! -x "$REPO_ROOT/ns-3.33/build/scratch/copter-sim" ]; then
    echo "[INFO] ns3 binary missing; build it with build_ns3_copter.sh first or run manually if sudo is required."
fi

run_pair "acc_eval" "$ACC_CONFIG" "$ACC_AGENT_SCRIPT"
sleep 5
run_pair "sor_eval" "$SOR_CONFIG" "$SOR_AGENT_SCRIPT"

"$CONDA_BIN" run -n m3 python "$COMPARE_SCRIPT"
