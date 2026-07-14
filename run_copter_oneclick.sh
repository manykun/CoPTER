#!/bin/bash
#
# One-click launcher for CoPTER multi-episode training + TensorBoard.
#
# Example:
#   ./run_copter_oneclick.sh
#   ./run_copter_oneclick.sh --episodes 300 --exp my_acc
#   ./run_copter_oneclick.sh --experiments "simulation/a.conf:expA,simulation/b.conf:expB"
#

set -euo pipefail

COPTER_ROOT="${COPTER_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
RUN_TRAINING_SH="${COPTER_ROOT}/run_training.sh"
CONDA_ENV="m3"
TB_LOGDIR="${COPTER_ROOT}/copter/runs"
TB_PORT=6006
TB_HOST="0.0.0.0"
OPEN_BROWSER=0
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tb-logdir) TB_LOGDIR="$2"; shift 2 ;;
        --tb-port) TB_PORT="$2"; shift 2 ;;
        --tb-host) TB_HOST="$2"; shift 2 ;;
        --open-browser) OPEN_BROWSER=1; shift ;;
        *) PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

if [ ! -f "${RUN_TRAINING_SH}" ]; then
    echo "ERROR: run_training.sh not found at ${RUN_TRAINING_SH}"
    exit 1
fi

# Activate conda environment before starting services/training.
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "Activated conda environment: ${CONDA_ENV}"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "Activated conda environment: ${CONDA_ENV}"
else
    echo "WARNING: Could not find conda. Please activate '${CONDA_ENV}' manually."
fi

mkdir -p "${TB_LOGDIR}"
mkdir -p "${COPTER_ROOT}/copter/training_logs"

# Start TensorBoard if not running on the target port.
if command -v lsof >/dev/null 2>&1 && lsof -iTCP:"${TB_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "TensorBoard already listening on port ${TB_PORT}, reusing existing process."
else
    echo "Starting TensorBoard at ${TB_HOST}:${TB_PORT} (logdir: ${TB_LOGDIR})"
    nohup tensorboard --logdir "${TB_LOGDIR}" --port "${TB_PORT}" --host "${TB_HOST}" \
        > "${COPTER_ROOT}/copter/training_logs/tensorboard_${TB_PORT}.log" 2>&1 &
    sleep 2
fi

TB_URL="http://127.0.0.1:${TB_PORT}"
echo "TensorBoard URL: ${TB_URL}"

if [ "${OPEN_BROWSER}" -eq 1 ]; then
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "${TB_URL}" >/dev/null 2>&1 || true
    fi
fi

echo ""
echo "Launching training with arguments: ${PASSTHROUGH_ARGS[*]:-(defaults)}"
echo "--------------------------------------------------------------"
bash "${RUN_TRAINING_SH}" "${PASSTHROUGH_ARGS[@]}"
