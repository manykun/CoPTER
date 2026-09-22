#!/bin/bash
# ============================================================
# CoPTER ACC 连续训练 - Agent 侧启动器
# 与 simulation/run_train.sh 对应。在两个终端分别运行。
# ============================================================
set -u

# 仓库根目录定位
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${TRAIN_CONFIG:-$SCRIPT_DIR/training_config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] training_config.yaml not found at: $CONFIG_FILE" >&2
    exit 1
fi

# 解析 yaml
eval "$(python3 "$SCRIPT_DIR/load_training_config.py" "$CONFIG_FILE")"

TOTAL_EPOCHS="${TOTAL_EPOCHS_OVERRIDE:-${CFG_total_epochs:-20}}"
EXP_NAME="${CFG_exp_name:-experiment_acc_continuous}"
NS3_PORT="${CFG_ns3_port:-5558}"
STATIC_STEPS="${CFG_static_steps:-4}"
TRAIN_INTERVALS="${CFG_train_intervals:-8}"
SWITCH_BUFFER="${SWITCH_BUFFER_OVERRIDE:-${CFG_switch_buffer:-10000}}"
MAX_STEPS="${CFG_max_steps:-0}"
EPS_START="${CFG_epsilon_start:-1.0}"
EPS_END="${CFG_epsilon_end:-0.05}"
EPS_DECAY="${CFG_epsilon_decay_steps:-50000}"
TARGET_UPDATE_INTERVAL="${CFG_target_update_interval:-100}"
SAVE_INTERVAL="${CFG_state_save_interval:-1}"
SEED="${CFG_seed:-1}"
MODEL_DIR="${CFG_model_dir:-models}"
INTER_SLEEP="${CFG_ns3_inter_run_sleep:-5}"

CONV_ENABLE="${CFG_convergence_enable:-false}"
CONV_WINDOW="${CFG_convergence_reward_window:-5}"
CONV_DELTA="${CFG_convergence_reward_delta:-0.005}"
CONV_MIN_EPOCHS="${CFG_convergence_min_epochs:-10}"

WANDB_ENABLE="${CFG_tensorboard_enable:-false}"
TB_LOG_DIR="${CFG_tensorboard_log_dir:-tb_logs}"
TB_FLUSH_SECS="${CFG_tensorboard_flush_secs:-30}"

METRICS_FILE="$SCRIPT_DIR/$MODEL_DIR/${EXP_NAME}_metrics.jsonl"

echo "============================================================"
echo " CoPTER ACC 连续训练 (Agent 侧)"
echo " exp_name      : $EXP_NAME"
echo " total_epochs  : $TOTAL_EPOCHS"
echo " ns3_port      : $NS3_PORT"
echo " epsilon       : $EPS_START -> $EPS_END over $EPS_DECAY train steps"
echo " seed          : $SEED"
echo " max_steps     : $MAX_STEPS  (0 = run until ns3 ends)"
echo " metrics file  : $METRICS_FILE"
echo "============================================================"

cd "$SCRIPT_DIR"

check_convergence() {
    # 调用 python 检查 metrics.jsonl 末尾窗口是否收敛；返回 0 表示已收敛
    [ "$CONV_ENABLE" != "true" ] && return 1
    [ ! -f "$METRICS_FILE" ] && return 1
    python3 - "$METRICS_FILE" "$CONV_WINDOW" "$CONV_DELTA" "$CONV_MIN_EPOCHS" <<'PY'
import json, sys
path, win, delta, min_ep = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rewards = []
with open(path) as f:
    for line in f:
        try:
            r = json.loads(line)
            if r.get("mean_reward") is not None:
                rewards.append(float(r["mean_reward"]))
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
    echo "[Agent] Epoch $i / $TOTAL_EPOCHS"
    echo "=============================="
    start_time=$(date +%s)

    EXTRA_FLAGS=""
    if [ "${EVAL_GREEDY:-0}" = "1" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --eval_greedy"
    fi
    if [ -n "${EVAL_TAG:-}" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --eval_tag ${EVAL_TAG}"
    fi
    if [ -n "${FORCE_ACTION:-}" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --force_action ${FORCE_ACTION}"
    fi
    if [ -n "${WATCH_PORTS:-}" ]; then
        EXTRA_FLAGS="$EXTRA_FLAGS --watch_ports ${WATCH_PORTS}"
    fi
    python copter.py \
        -p "$NS3_PORT" \
        -e "$EXP_NAME" \
        --online \
        -m ACC \
        -d "$MODEL_DIR" \
        -s "$STATIC_STEPS" \
        -i "$TRAIN_INTERVALS" \
        -b "$SWITCH_BUFFER" \
        --max_steps "$MAX_STEPS" \
        --epsilon_start "$EPS_START" \
        --epsilon_end "$EPS_END" \
        --epsilon_decay_steps "$EPS_DECAY" \
        --target_update_interval "$TARGET_UPDATE_INTERVAL" \
        --state_save_interval "$SAVE_INTERVAL" \
        --seed "$SEED" \
        --tb_enable "$WANDB_ENABLE" \
        --tb_log_dir "$TB_LOG_DIR" \
        --tb_flush_secs "$TB_FLUSH_SECS" \
        $EXTRA_FLAGS

    rc=$?
    end_time=$(date +%s)
    echo "------------------------------"
    echo "[Agent] epoch $i done in $((end_time - start_time))s (rc=$rc)"

    if check_convergence; then
        echo "[Agent] Convergence detected after epoch $i. Early stop."
        break
    fi

    if [ "$i" -lt "$TOTAL_EPOCHS" ]; then
        echo "[Agent] sleep $INTER_SLEEP s before next epoch..."
        sleep "$INTER_SLEEP"
    fi
done

echo "[Agent] Training finished."
