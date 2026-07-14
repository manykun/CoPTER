#!/bin/bash
# ============================================================
# CoPTER ACC 连续训练 - ns3 侧启动器
# 与 copter/run_train.sh 配合使用，分别在两个终端运行。
#
# 路径解析逻辑（健壮、不依赖 run-copter-sim.sh 的相对路径）：
#   - SCRIPT_DIR : 本脚本所在目录（应该是 .../simulation/）
#   - REPO_ROOT  : SCRIPT_DIR 的上一级（应该是仓库根目录）
#   - 所有 ns3 相关路径优先用 yaml 中配置的；如果是相对路径，则相对于 REPO_ROOT 解析
# ============================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COPTER_DIR="$REPO_ROOT/copter"
CONFIG_FILE="${TRAIN_CONFIG:-$COPTER_DIR/training_config.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] training_config.yaml not found at: $CONFIG_FILE" >&2
    exit 1
fi

eval "$(python3 "$COPTER_DIR/load_training_config.py" "$CONFIG_FILE")"

# 工具函数：把可能是相对的路径（相对于 REPO_ROOT）转成绝对路径
abspath_under_root() {
    local p="$1"
    case "$p" in
        /*) printf '%s\n' "$p" ;;
        *)  printf '%s\n' "$REPO_ROOT/$p" ;;
    esac
}

TOTAL_EPOCHS="${TOTAL_EPOCHS_OVERRIDE:-${CFG_total_epochs:-20}}"
NS3_PORT="${CFG_ns3_port:-5558}"
NS3_CONF=$(abspath_under_root "${NS3_CONF_OVERRIDE:-${CFG_ns3_conf:-simulation/mix/copter_Hadoop_Shuffle.conf}}")
NS3_DIR=$(abspath_under_root "${CFG_ns3_dir:-ns-3.33}")
NS3_BIN=$(abspath_under_root "${CFG_ns3_binary:-ns-3.33/build/scratch/copter-sim}")
NS3_LIB_DIR="$NS3_DIR/build/lib"
NS3_LOG_LEV="${CFG_ns3_log_level:-level_all}"
INTER_SLEEP="${CFG_ns3_inter_run_sleep:-5}"

echo "============================================================"
echo " CoPTER ACC 连续训练 (ns3 侧)"
echo " repo_root    : $REPO_ROOT"
echo " ns3_dir      : $NS3_DIR"
echo " ns3_binary   : $NS3_BIN"
echo " ns3_conf     : $NS3_CONF"
echo " ns3_port     : $NS3_PORT"
echo " total_epochs : $TOTAL_EPOCHS"
echo "============================================================"

# 路径预检
err=0
[ -f "$NS3_CONF" ] || { echo "[ERROR] ns3 conf not found: $NS3_CONF" >&2; err=1; }
[ -d "$NS3_DIR/build" ] || { echo "[ERROR] ns3 build dir not found: $NS3_DIR/build (please build ns3 first)" >&2; err=1; }
[ -x "$NS3_BIN" ] || { echo "[ERROR] ns3 binary not executable / missing: $NS3_BIN" >&2; err=1; }
[ "$err" -eq 0 ] || exit 1

export LD_LIBRARY_PATH="$NS3_LIB_DIR:${LD_LIBRARY_PATH:-}"
export NS_LOG="CongestionControlSimulator=$NS3_LOG_LEV:OpenGymInterface=$NS3_LOG_LEV"

# 切到 simulation/ 工作目录（ns3 conf 内的 TOPOLOGY_FILE / FLOW_FILE 等是相对路径，相对此 cwd 解析）
cd "$SCRIPT_DIR"

for ((i = 1; i <= TOTAL_EPOCHS; i++)); do
    echo "=============================="
    echo "[NS3] Epoch $i / $TOTAL_EPOCHS"
    echo "=============================="
    start_time=$(date +%s)

    echo "[NS3] cmd: $NS3_BIN $NS3_CONF --port=$NS3_PORT"
    "$NS3_BIN" "$NS3_CONF" --port="$NS3_PORT"
    rc=$?

    end_time=$(date +%s)
    echo "------------------------------"
    echo "[NS3] epoch $i done in $((end_time - start_time))s (rc=$rc)"

    if [ "$i" -lt "$TOTAL_EPOCHS" ]; then
        echo "[NS3] sleep $INTER_SLEEP s before next epoch..."
        sleep "$INTER_SLEEP"
    fi
done

echo "[NS3] All epochs finished."
