#!/bin/bash
# ============================================================
# SOR-ACC continuous training - Agent side launcher
# Run ns3 side separately, for example:
# TRAIN_CONFIG=/root/paddlejob/workspace/yangziwen/CoPTER/sor/sor_training_config.yaml \
#   /root/paddlejob/workspace/yangziwen/CoPTER/simulation/run_train.sh
# ============================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COPTER_DIR="$REPO_ROOT/copter"
CONFIG_FILE="${TRAIN_CONFIG:-$SCRIPT_DIR/sor_training_config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] sor_training_config.yaml not found at: $CONFIG_FILE" >&2
    exit 1
fi

eval "$(python3 "$COPTER_DIR/load_training_config.py" "$CONFIG_FILE")"

TOTAL_EPOCHS="${TOTAL_EPOCHS_OVERRIDE:-${CFG_total_epochs:-20}}"
EXP_NAME="${CFG_exp_name:-experiment_sor_acc_continuous}"
NS3_PORT="${CFG_ns3_port:-5558}"
STATIC_STEPS="${CFG_static_steps:-4}"
TRAIN_INTERVALS="${CFG_train_intervals:-8}"
SWITCH_BUFFER="${SWITCH_BUFFER_OVERRIDE:-${CFG_switch_buffer:-10000}}"
MAX_STEPS="${CFG_max_steps:-0}"
EPS_START="${CFG_epsilon_start:-1.0}"
EPS_END="${CFG_epsilon_end:-0.05}"
EPS_DECAY="${CFG_epsilon_decay_steps:-50000}"
EPS_SCHEDULE="${CFG_epsilon_schedule:-phase}"
ACTION_SPACE="${CFG_action_space:-legacy}"
TARGET_UPDATE_INTERVAL="${CFG_target_update_interval:-100}"
SAVE_INTERVAL="${CFG_state_save_interval:-1}"
SEED="${CFG_seed:-1}"
MODEL_DIR="${CFG_model_dir:-sor_models}"
INTER_SLEEP="${CFG_ns3_inter_run_sleep:-5}"

CONV_ENABLE="${CFG_convergence_enable:-false}"
CONV_WINDOW="${CFG_convergence_reward_window:-5}"
CONV_DELTA="${CFG_convergence_reward_delta:-0.005}"
CONV_MIN_EPOCHS="${CFG_convergence_min_epochs:-10}"

TB_ENABLE="${CFG_tensorboard_enable:-false}"
TB_LOG_DIR="${CFG_tensorboard_log_dir:-tb_logs}"
TB_FLUSH_SECS="${CFG_tensorboard_flush_secs:-30}"

SOR_RECENT_SIZE="${CFG_sor_recent_size:-2000}"
SOR_BOUNDARY_SIZE="${CFG_sor_boundary_size:-20000}"
SOR_MAX_CLUSTERS="${CFG_sor_max_clusters:-32}"
SOR_PROTOTYPE_DISTANCE="${CFG_sor_prototype_distance:-1.0}"
SOR_PROTOTYPE_ETA="${CFG_sor_prototype_eta:-0.05}"
SOR_BOUNDARY_THRESHOLD="${CFG_sor_boundary_threshold:-0.5}"
SOR_ALPHA_TD="${CFG_sor_alpha_td:-1.0}"
SOR_BETA_UNDER_SAMPLE="${CFG_sor_beta_under_sample:-0.2}"
SOR_GAMMA_DRIFT="${CFG_sor_gamma_drift:-0.5}"
SOR_RHO_BOUNDARY="${CFG_sor_rho_boundary:-0.5}"
SOR_TEMPERATURE="${CFG_sor_temperature:-1.0}"
SOR_UNIFORM_MIX="${CFG_sor_uniform_mix:-0.01}"
SOR_LAMBDA_CONS="${CFG_sor_lambda_cons:-0.01}"
SOR_LAMBDA_REG="${CFG_sor_lambda_reg:-0.001}"
SOR_DRIFT_REG_THRESHOLD="${CFG_sor_drift_reg_threshold:-0.5}"
SOR_REF_UPDATE_INTERVAL="${CFG_sor_ref_update_interval:-256}"
SOR_SYNC_INTERVAL="${CFG_sor_sync_interval:-8}"
SOR_SAVE_BUFFER_EVERY="${CFG_sor_save_buffer_every:-5}"

METRICS_FILE="$SCRIPT_DIR/$MODEL_DIR/${EXP_NAME}_metrics.jsonl"

cd "$SCRIPT_DIR"

check_convergence() {
    [ "$CONV_ENABLE" != "true" ] && return 1
    [ ! -f "$METRICS_FILE" ] && return 1
    python3 - "$METRICS_FILE" "$CONV_WINDOW" "$CONV_DELTA" "$CONV_MIN_EPOCHS" <<'PY'
import json, sys
path, win, delta, min_ep = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rewards = []
with open(path) as f:
    for line in f:
        try:
            record = json.loads(line)
            if record.get("mean_reward") is not None:
                rewards.append(float(record["mean_reward"]))
        except Exception:
            pass
if len(rewards) < max(min_ep, 2 * win):
    sys.exit(1)
prev = sum(rewards[-2*win:-win]) / win
curr = sum(rewards[-win:]) / win
print(f"[converge-check] prev_window_mean={prev:.4f} curr_window_mean={curr:.4f} delta={abs(curr-prev):.4f}")
sys.exit(0 if abs(curr - prev) < delta else 1)
PY
}

for ((i = 1; i <= TOTAL_EPOCHS; i++)); do
    echo "=============================="
    echo "[SOR-Agent] Epoch $i / $TOTAL_EPOCHS"
    echo "=============================="
    start_time=$(date +%s)

    EXTRA_FLAGS=""
    if [ "${EVAL_GREEDY:-0}" = "1" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --eval_greedy"
    fi
    if [ -n "${EVAL_TAG:-}" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --eval_tag ${EVAL_TAG}"
    fi
    python sor_copter.py \
        -p "$NS3_PORT" \
        -e "$EXP_NAME" \
        --online \
        -d "$MODEL_DIR" \
        -s "$STATIC_STEPS" \
        -i "$TRAIN_INTERVALS" \
        -b "$SWITCH_BUFFER" \
        --max_steps "$MAX_STEPS" \
        --epsilon_start "$EPS_START" \
        --epsilon_end "$EPS_END" \
        --epsilon_decay_steps "$EPS_DECAY" \
        --epsilon_schedule "$EPS_SCHEDULE" \
        --action_space "$ACTION_SPACE" \
        --target_update_interval "$TARGET_UPDATE_INTERVAL" \
        --state_save_interval "$SAVE_INTERVAL" \
        --seed "$SEED" \
        --tb_enable "$TB_ENABLE" \
        --tb_log_dir "$TB_LOG_DIR" \
        --tb_flush_secs "$TB_FLUSH_SECS" \
        --sor_recent_size "$SOR_RECENT_SIZE" \
        --sor_boundary_size "$SOR_BOUNDARY_SIZE" \
        --sor_max_clusters "$SOR_MAX_CLUSTERS" \
        --sor_prototype_distance "$SOR_PROTOTYPE_DISTANCE" \
        --sor_prototype_eta "$SOR_PROTOTYPE_ETA" \
        --sor_boundary_threshold "$SOR_BOUNDARY_THRESHOLD" \
        --sor_alpha_td "$SOR_ALPHA_TD" \
        --sor_beta_under_sample "$SOR_BETA_UNDER_SAMPLE" \
        --sor_gamma_drift "$SOR_GAMMA_DRIFT" \
        --sor_rho_boundary "$SOR_RHO_BOUNDARY" \
        --sor_temperature "$SOR_TEMPERATURE" \
        --sor_uniform_mix "$SOR_UNIFORM_MIX" \
        --sor_lambda_cons "$SOR_LAMBDA_CONS" \
        --sor_lambda_reg "$SOR_LAMBDA_REG" \
        --sor_drift_reg_threshold "$SOR_DRIFT_REG_THRESHOLD" \
        --sor_ref_update_interval "$SOR_REF_UPDATE_INTERVAL" \
        --sor_sync_interval "$SOR_SYNC_INTERVAL" \
        --sor_save_buffer_every "$SOR_SAVE_BUFFER_EVERY" \
        $EXTRA_FLAGS

    rc=$?
    end_time=$(date +%s)
    echo "[SOR-Agent] epoch $i done in $((end_time - start_time))s (rc=$rc)"

    if check_convergence; then
        echo "[SOR-Agent] Convergence detected after epoch $i. Early stop."
        break
    fi

    if [ "$i" -lt "$TOTAL_EPOCHS" ]; then
        echo "[SOR-Agent] sleep $INTER_SLEEP s before next epoch..."
        sleep "$INTER_SLEEP"
    fi
done

echo "[SOR-Agent] Training finished."
