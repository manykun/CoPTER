#!/bin/bash
# ============================================================
# "Does ACC's RL actually raise the reward?" — single-scenario
# (Hadoop_Shuffle) experiment with fixed watch-ports + force_action baselines.
#
# What it does (all on simulation/mix/acc_Hadoop_Shuffle.conf):
#   1. Trains ACC from scratch for TRAIN_EPOCHS, tracking a fixed set of
#      congested ports  -> shows the per-port reward RISING across epochs.
#   2. Greedy-eval the trained policy (epsilon=0) once.
#   3. Runs 3 fixed-action baselines (force_action, no training) once each,
#      to prove (a) reward is sensitive to DCQCN actions and
#                (b) the learned policy beats the best fixed action.
#   4. Prints a summary table of the watch-port rewards.
#
# The watch ports are the ToR->receiver downlinks of the hottest incast
# receivers in Hadoop_Shuffle.flow (host 141/129/130/138/135):
#   port_idx = 162 + ((host-128)//8)*10 + ((host-128)%8)
#     host 141 -> 177   host 129 -> 163   host 130 -> 164
#     host 138 -> 174   host 135 -> 169
#
# Usage:
#   scripts/run_acc_watch_baseline.sh
#   TRAIN_EPOCHS=60 scripts/run_acc_watch_baseline.sh
#   WATCH_PORTS=177,163,164 scripts/run_acc_watch_baseline.sh
#   RESET=0 scripts/run_acc_watch_baseline.sh          # resume, don't wipe
# ============================================================
set -u

REPO_ROOT="/root/paddlejob/workspace/yangziwen/CoPTER"
CONDA_BIN="/root/miniconda3/bin/conda"
NS3_SCRIPT="$REPO_ROOT/simulation/run_train.sh"
AGENT_SCRIPT="$REPO_ROOT/copter/run_train.sh"
CONFIG="$REPO_ROOT/sor/acc_watch_hadoop_config.yaml"
NS3_CONF="$REPO_ROOT/simulation/mix/acc_Hadoop_Shuffle.conf"

COPTER_DIR="$REPO_ROOT/copter"
MODEL_DIR="$COPTER_DIR/watch_models_acc"
TB_DIR="$COPTER_DIR/tb_logs_watch_acc"
METRICS="$MODEL_DIR/watch_acc_hadoop_metrics.jsonl"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
NS3_PORT="${NS3_PORT:-5558}"
WATCH_PORTS="${WATCH_PORTS:-177,163,164,174,169}"
RESET="${RESET:-1}"

# Fixed-action baselines: "kmin_idx,kmax_idx,pmax_idx" into the discretised
# grid in copter.py (kmin[6], kmax[4], pmax[10]).
BASE_EARLY="0,1,9"   # mark very early + hard  (low Kmin, high Pmax)
BASE_MID="2,2,4"     # moderate marking
BASE_LATE="5,3,0"    # mark very late + weak   (high Kmin, low Pmax)

ns3_pid=""; agent_pid=""
cleanup() {
    [ -n "${ns3_pid:-}" ] && kill "$ns3_pid" 2>/dev/null
    [ -n "${agent_pid:-}" ] && kill "$agent_pid" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup EXIT INT TERM

preflight_check() {
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$NS3_PORT$"; then
        echo "[ERROR] port $NS3_PORT already in use; clean up stale processes first." >&2
        exit 1
    fi
    local leftover
    leftover=$(pgrep -af 'copter-sim|copter\.py|sor_copter\.py' | grep -v $$ || true)
    if [ -n "$leftover" ]; then
        echo "[ERROR] stale CoPTER processes detected:" >&2
        echo "$leftover" >&2
        exit 1
    fi
}
preflight_check

wait_port_free() {
    local port="$1" max="${2:-90}" i=0
    while [ "$i" -lt "$max" ]; do
        ss -tan 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$port$" || return 0
        sleep 1; i=$((i+1))
    done
    echo "[WARN] port $port still occupied after ${max}s" >&2
    return 1
}

# run ONE epoch: fresh agent + fresh ns3.  $1=greedy(0/1) $2=eval_tag $3=force_action(may be empty)
run_epoch() {
    local greedy="$1" tag="$2" force="$3"
    local agent_tag="[watch-agent]" ns3_tag="[watch-ns3]"

    WATCH_PORTS="$WATCH_PORTS" FORCE_ACTION="$force" \
    TRAIN_CONFIG="$CONFIG" TOTAL_EPOCHS_OVERRIDE=1 \
    EVAL_GREEDY="$greedy" EVAL_TAG="$tag" \
        "$CONDA_BIN" run --no-capture-output -n m3 \
        bash "$AGENT_SCRIPT" > >(sed -u "s|^|$agent_tag |") 2>&1 &
    agent_pid=$!

    local waited=0
    while [ "$waited" -lt 60 ]; do
        ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]$NS3_PORT$" && break
        if ! kill -0 "$agent_pid" 2>/dev/null; then
            echo "[ERROR] agent died before binding port $NS3_PORT (tag=$tag)" >&2
            wait "$agent_pid"; agent_pid=""; exit 1
        fi
        sleep 1; waited=$((waited+1))
    done
    if [ "$waited" -ge 60 ]; then
        echo "[ERROR] agent failed to bind port $NS3_PORT in 60s (tag=$tag)" >&2
        kill "$agent_pid" 2>/dev/null; wait "$agent_pid"; agent_pid=""; exit 1
    fi

    TRAIN_CONFIG="$CONFIG" TOTAL_EPOCHS_OVERRIDE=1 NS3_CONF_OVERRIDE="$NS3_CONF" \
        "$CONDA_BIN" run --no-capture-output -n m3 \
        bash "$NS3_SCRIPT" > >(sed -u "s|^|$ns3_tag |") 2>&1 &
    ns3_pid=$!

    wait "$agent_pid"; local agent_rc=$?; agent_pid=""
    local ns3_rc=0
    wait "$ns3_pid" || ns3_rc=$?; ns3_pid=""
    [ "$ns3_rc" -eq 143 ] && ns3_rc=0
    if [ "$agent_rc" -ne 0 ] || [ "$ns3_rc" -ne 0 ]; then
        echo "[ERROR] epoch failed (agent_rc=$agent_rc ns3_rc=$ns3_rc, tag=$tag)" >&2
        exit 1
    fi
    wait_port_free "$NS3_PORT" 90 || true
    sleep 3
}

cd "$REPO_ROOT"

if [ "$RESET" = "1" ]; then
    echo "[WATCH] RESET=1 -> wiping $MODEL_DIR and $TB_DIR for a clean run"
    rm -rf "$MODEL_DIR" "$TB_DIR"
fi
mkdir -p "$MODEL_DIR"

echo "############################################################"
echo "[WATCH] single-scenario ACC on Hadoop_Shuffle"
echo "[WATCH] train epochs : $TRAIN_EPOCHS"
echo "[WATCH] watch ports  : $WATCH_PORTS"
echo "[WATCH] metrics file : $METRICS"
echo "############################################################"

echo "=== [WATCH] phase 1: TRAIN for $TRAIN_EPOCHS epochs ==="
for ((ep = 1; ep <= TRAIN_EPOCHS; ep++)); do
    echo "--- [WATCH] train epoch $ep/$TRAIN_EPOCHS ---"
    run_epoch 0 "train" ""
done

echo "=== [WATCH] phase 2: greedy eval of the trained policy ==="
run_epoch 1 "eval_trained_greedy" ""

echo "=== [WATCH] phase 3: fixed-action baselines (no training) ==="
run_epoch 1 "base_early" "$BASE_EARLY"
run_epoch 1 "base_mid"   "$BASE_MID"
run_epoch 1 "base_late"  "$BASE_LATE"

echo "=== [WATCH] summary ==="
"$CONDA_BIN" run --no-capture-output -n m3 python - "$METRICS" "$WATCH_PORTS" <<'PY'
import json, sys
path, watch = sys.argv[1], [p for p in sys.argv[2].split(",") if p]
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except Exception: pass

train = [r for r in rows if r.get("eval_tag") == "train"]
def wp(r, p):
    v = (r.get("watch_ports_reward") or {}).get(str(p))
    return f"{v:.4f}" if isinstance(v, (int, float)) else "  N/A "

hdr = "epoch | rollout_mean | " + " | ".join(f"port{p}" for p in watch)
print("\n--- TRAIN trajectory (reward should RISE) ---")
print(hdr); print("-" * len(hdr))
for r in train:
    rm = r.get("rollout_mean_reward")
    rm = f"{rm:.4f}" if isinstance(rm, (int, float)) else " N/A  "
    print(f"{r.get('epoch'):>5} |   {rm}    | " + " | ".join(wp(r, p) for p in watch))

print("\n--- Trained-greedy vs fixed-action baselines (same ports) ---")
print("tag                  | rollout_mean | " + " | ".join(f"port{p}" for p in watch))
print("-" * (40 + 8 * len(watch)))
for tag in ["eval_trained_greedy", "base_early", "base_mid", "base_late"]:
    rs = [r for r in rows if r.get("eval_tag") == tag]
    if not rs:
        print(f"{tag:<20} | (missing)")
        continue
    r = rs[-1]
    rm = r.get("rollout_mean_reward")
    rm = f"{rm:.4f}" if isinstance(rm, (int, float)) else " N/A  "
    print(f"{tag:<20} |    {rm}   | " + " | ".join(wp(r, p) for p in watch))

print("\nInterpretation:")
print("  * TRAIN port rewards trending up  -> RL is learning on the single scenario.")
print("  * baselines differ from each other -> reward IS sensitive to DCQCN actions.")
print("  * eval_trained_greedy >= best baseline -> the learned policy is effective.")
print(f"\nTensorBoard: tensorboard --logdir {path.rsplit('/',2)[0]}/tb_logs_watch_acc")
PY

echo "[WATCH] done."
