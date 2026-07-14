#!/bin/bash
# ============================================================
# Reward-sensitivity sanity check (P0).
#
# Runs pure-greedy single-epoch evals on the SAME ns3 conf, but with the
# policy bypassed and a FIXED action applied to every port (--force_action).
# By comparing the per-step rollout reward across very different actions we
# can tell whether the reward actually responds to the ECN parameters.
#
# If two wildly-different actions yield nearly-identical reward, the reward /
# workload is action-insensitive and no continual-learning method can show a
# difference -- the experiment design must be fixed first.
#
# Usage:
#   reward_sensitivity_check.sh [conf_name]
#     conf_name defaults to acc_Hadoop_Shuffle (looked up in simulation/mix/).
#   ACTIONS="0,0,0 5,3,9 2,1,4" reward_sensitivity_check.sh acc_Hadoop_Shuffle
# ============================================================
set -u

REPO_ROOT="/root/paddlejob/workspace/yangziwen/CoPTER"
CONDA_BIN="/root/miniconda3/bin/conda"
NS3_SCRIPT="$REPO_ROOT/simulation/run_train.sh"
AGENT_SCRIPT="$REPO_ROOT/copter/run_train.sh"
CONFIG="$REPO_ROOT/sor/acc_curriculum_config.yaml"
NS3_PORT="${NS3_PORT:-5558}"

CONF_NAME="${1:-acc_Hadoop_Shuffle}"
NS3_CONF="$REPO_ROOT/simulation/mix/${CONF_NAME}.conf"
[ -f "$NS3_CONF" ] || { echo "[ERROR] missing conf: $NS3_CONF" >&2; exit 1; }

# Two (or more) very different fixed actions: kmin_idx,kmax_idx,pmax_idx
# Default: minimal-marking (low kmin/kmax/pmax) vs aggressive-marking (high).
ACTIONS="${ACTIONS:-0,0,0 5,3,9}"

METRICS="$REPO_ROOT/copter/cur_models_acc/cur_acc_metrics.jsonl"
RESULT_FILE="$REPO_ROOT/scripts/reward_sensitivity_${CONF_NAME}.txt"
: > "$RESULT_FILE"

preflight() {
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$NS3_PORT$"; then
        echo "[ERROR] port $NS3_PORT in use; clean up first." >&2; exit 1
    fi
    pgrep -af 'copter-sim|copter\.py|sor_copter\.py' | grep -v $$ >/dev/null 2>&1 \
        && { echo "[ERROR] stale CoPTER processes; clean up first." >&2; exit 1; } || true
}
preflight

wait_port_free() {
    local port="$1" max="${2:-90}" i=0
    while [ "$i" -lt "$max" ]; do
        ss -tan 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$port$" || return 0
        sleep 1; i=$((i+1))
    done
    return 1
}

ns3_pid=""; agent_pid=""
cleanup() { [ -n "$ns3_pid" ] && kill "$ns3_pid" 2>/dev/null; [ -n "$agent_pid" ] && kill "$agent_pid" 2>/dev/null; wait 2>/dev/null; }
trap cleanup EXIT INT TERM

run_forced_eval() {
    local action="$1" tag="$2"
    # Agent side: greedy eval + forced action. EVAL_GREEDY=1 prevents training/saving.
    TRAIN_CONFIG="$CONFIG" TOTAL_EPOCHS_OVERRIDE=1 EVAL_GREEDY=1 EVAL_TAG="$tag" \
        FORCE_ACTION="$action" \
        "$CONDA_BIN" run --no-capture-output -n m3 \
        bash "$AGENT_SCRIPT" > >(sed -u "s|^|[agent] |") 2>&1 &
    agent_pid=$!

    local waited=0
    while [ "$waited" -lt 60 ]; do
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$NS3_PORT$" && break
        kill -0 "$agent_pid" 2>/dev/null || { echo "[ERROR] agent died early" >&2; wait "$agent_pid"; agent_pid=""; exit 1; }
        sleep 1; waited=$((waited+1))
    done

    TRAIN_CONFIG="$CONFIG" TOTAL_EPOCHS_OVERRIDE=1 NS3_CONF_OVERRIDE="$NS3_CONF" \
        "$CONDA_BIN" run --no-capture-output -n m3 \
        bash "$NS3_SCRIPT" > >(sed -u "s|^|[ns3] |") 2>&1 &
    ns3_pid=$!

    wait "$agent_pid"; agent_pid=""
    wait "$ns3_pid" 2>/dev/null; ns3_pid=""
    wait_port_free "$NS3_PORT" 90 || true
    sleep 3

    # Extract the rollout_mean_reward of the row we just appended (matching tag).
    local reward
    reward=$("$CONDA_BIN" run --no-capture-output -n m3 python - "$METRICS" "$tag" <<'PY'
import json, sys
path, tag = sys.argv[1], sys.argv[2]
val = None
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("eval_tag") == tag and r.get("rollout_mean_reward") is not None:
            val = r["rollout_mean_reward"]
print("" if val is None else f"{val:.6f}")
PY
)
    echo "action=${action}  tag=${tag}  rollout_mean_reward=${reward}" | tee -a "$RESULT_FILE"
}

echo "############################################################"
echo "[SANITY] reward sensitivity on conf=$CONF_NAME  actions=[$ACTIONS]"
echo "############################################################"
idx=0
for act in $ACTIONS; do
    idx=$((idx+1))
    run_forced_eval "$act" "sanity_${CONF_NAME}_a${idx}_${act//,/_}"
done

echo "------------------------------------------------------------"
echo "[SANITY] summary (file: $RESULT_FILE):"
cat "$RESULT_FILE"
echo "------------------------------------------------------------"
"$CONDA_BIN" run --no-capture-output -n m3 python - "$RESULT_FILE" <<'PY'
import re, sys
vals = []
for line in open(sys.argv[1]):
    m = re.search(r"rollout_mean_reward=([0-9.]+)", line)
    if m:
        vals.append(float(m.group(1)))
if len(vals) >= 2:
    spread = max(vals) - min(vals)
    print(f"[SANITY] reward spread across actions = {spread:.6f}")
    if spread < 0.01:
        print("[SANITY][WARN] reward is action-INSENSITIVE (spread < 0.01). "
              "Fix reward/workload before running the forgetting curriculum.")
    else:
        print("[SANITY][OK] reward responds to actions; curriculum design can proceed.")
PY
