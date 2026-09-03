#!/bin/bash
#
# Unified CoPTER Training Script
#
# Launches ns3 simulation and RL agent for each episode, supports resumable
# training with convergence checking. Can run multiple experiments in parallel.
#
# Usage:
#   ./run_training.sh                           # Single experiment (defaults)
#   ./run_training.sh --episodes 200            # Override max episodes
#   ./run_training.sh --config c1.conf --exp E1 # Single experiment
#   ./run_training.sh --experiments "c1.conf:E1,c2.conf:E2,c3.conf:E3"  # Multi (parallel)
#
# --experiments format: "config1:exp1,config2:exp2,..."
#   Each pair gets a unique port (5556, 5557, ...) and runs in parallel.

set -euo pipefail

# ==================== Conda Environment ====================
CONDA_ENV="m3"
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "Activated conda environment: ${CONDA_ENV}"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    echo "Activated conda environment: ${CONDA_ENV}"
else
    echo "WARNING: Could not find conda. Make sure '${CONDA_ENV}' env is active."
fi

# ==================== Configuration ====================
COPTER_ROOT="${COPTER_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
NS3_DIR="${COPTER_ROOT}/ns-3.33"
NS3_BLD_DIR="${NS3_DIR}/build"
NS3_LIB_DIR="$(realpath "${NS3_BLD_DIR}")/lib"

# Defaults (override via command-line flags)
MAX_EPISODES=200
BASE_PORT=5556
NS3_CONF="${COPTER_ROOT}/simulation/mix/m3_256hosts.conf"
EXP_NAME="acc_experiment"
MODE="ACC"
FMAP_DIR=""
ONLINE=1
MODEL_DIR="${COPTER_ROOT}/copter/models"
SWITCH_BUFFER=400
TRAIN_INTERVALS=8
STATIC_STEPS=4
EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_DECAY=50000
EPSILON_SCHEDULE="phase"
TARGET_UPDATE_INTERVAL=100
WAIT_NS3_SEC=3
WAIT_BETWEEN_SEC=5
FORCE_ACTION=""
FORCE_PORT_ACTION=""
EVAL_GREEDY=0
EVAL_TAG=""
SEED=1
WATCH_PORTS=""
WATCH_TRACE_FILE=""
MAX_STEPS=0
TARGET_TRAIN_STEPS=0
TB_ENABLE="true"
ONE_SHOT=0
RUN_ID=""
PHASE=""
ACC_HIDDEN_DIMS="32,64,64,32"
ACTION_SPACE="legacy"
REWARD_WEIGHTS="0.50,0.30,0.20"
REWARD_PROFILE="weighted"
REWARD_QUEUE_LAMBDA=5.0
REWARD_ECN_LAMBDA=5.0
SHARED_REPLAY="true"
SOR_RECENT_SIZE=2000
SOR_BOUNDARY_SIZE=20000
SOR_MAX_CLUSTERS=32
SOR_PROTOTYPE_DISTANCE=1.0
SOR_PROTOTYPE_ETA=0.05
SOR_BOUNDARY_THRESHOLD=0.5
SOR_ALPHA_TD=1.0
SOR_BETA_UNDER_SAMPLE=0.2
SOR_GAMMA_DRIFT=0.5
SOR_RHO_BOUNDARY=0.5
SOR_TEMPERATURE=1.0
SOR_LAMBDA_CONS=0.01
SOR_LAMBDA_REG=0.001
SOR_DRIFT_REG_THRESHOLD=0.5
SOR_REF_UPDATE_INTERVAL=256
SOR_SYNC_INTERVAL=8
# Each episode is a separate Python process.  Formal continual-learning runs
# must persist replay every episode or the old-task memory silently disappears.
SOR_SAVE_BUFFER_EVERY=1

# Multi-experiment: array of "config:exp" pairs
EXP_LIST=()

# ==================== Parse Arguments ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --episodes)     MAX_EPISODES="$2";    shift 2 ;;
        --port)         BASE_PORT="$2";       shift 2 ;;
        --config)       NS3_CONF="$2";        shift 2 ;;
        --exp)          EXP_NAME="$2";        shift 2 ;;
        --experiments)  IFS=',' read -ra EXP_LIST <<< "$2"; shift 2 ;;
        --mode)         MODE="$2";            shift 2 ;;
        --fmap)         FMAP_DIR="$2";        shift 2 ;;
        --model-dir)    MODEL_DIR="$2";       shift 2 ;;
        --buffer)       SWITCH_BUFFER="$2";   shift 2 ;;
        --eps-start)    EPSILON_START="$2";   shift 2 ;;
        --eps-end)      EPSILON_END="$2";     shift 2 ;;
        --eps-decay)    EPSILON_DECAY="$2";   shift 2 ;;
        --epsilon-schedule) EPSILON_SCHEDULE="$2"; shift 2 ;;
        --target-update-interval) TARGET_UPDATE_INTERVAL="$2"; shift 2 ;;
        --force-action) FORCE_ACTION="$2";    shift 2 ;;
        --force-port-action) FORCE_PORT_ACTION="$2"; shift 2 ;;
        --eval-greedy)  EVAL_GREEDY=1; ONE_SHOT=1; shift ;;
        --eval-tag)     EVAL_TAG="$2";        shift 2 ;;
        --seed)         SEED="$2";            shift 2 ;;
        --watch-ports)  WATCH_PORTS="$2";     shift 2 ;;
        --watch-trace-file) WATCH_TRACE_FILE="$2"; shift 2 ;;
        --max-steps)    MAX_STEPS="$2";       shift 2 ;;
        --target-train-steps) TARGET_TRAIN_STEPS="$2"; shift 2 ;;
        --tb-enable)    TB_ENABLE="$2";       shift 2 ;;
        --one-shot)     ONE_SHOT=1;            shift ;;
        --run-id)       RUN_ID="$2";          shift 2 ;;
        --phase)        PHASE="$2";           shift 2 ;;
        --acc-hidden-dims) ACC_HIDDEN_DIMS="$2"; shift 2 ;;
        --action-space) ACTION_SPACE="$2"; shift 2 ;;
        --reward-weights) REWARD_WEIGHTS="$2"; shift 2 ;;
        --reward-profile) REWARD_PROFILE="$2"; shift 2 ;;
        --reward-queue-lambda) REWARD_QUEUE_LAMBDA="$2"; shift 2 ;;
        --reward-ecn-lambda) REWARD_ECN_LAMBDA="$2"; shift 2 ;;
        --shared-replay) SHARED_REPLAY="$2"; shift 2 ;;
        --sor-recent-size) SOR_RECENT_SIZE="$2"; shift 2 ;;
        --sor-boundary-size) SOR_BOUNDARY_SIZE="$2"; shift 2 ;;
        --sor-max-clusters) SOR_MAX_CLUSTERS="$2"; shift 2 ;;
        --sor-prototype-distance) SOR_PROTOTYPE_DISTANCE="$2"; shift 2 ;;
        --sor-prototype-eta) SOR_PROTOTYPE_ETA="$2"; shift 2 ;;
        --sor-boundary-threshold) SOR_BOUNDARY_THRESHOLD="$2"; shift 2 ;;
        --sor-alpha-td) SOR_ALPHA_TD="$2"; shift 2 ;;
        --sor-beta-under-sample) SOR_BETA_UNDER_SAMPLE="$2"; shift 2 ;;
        --sor-gamma-drift) SOR_GAMMA_DRIFT="$2"; shift 2 ;;
        --sor-rho-boundary) SOR_RHO_BOUNDARY="$2"; shift 2 ;;
        --sor-temperature) SOR_TEMPERATURE="$2"; shift 2 ;;
        --sor-lambda-cons) SOR_LAMBDA_CONS="$2"; shift 2 ;;
        --sor-lambda-reg) SOR_LAMBDA_REG="$2"; shift 2 ;;
        --sor-drift-reg-threshold) SOR_DRIFT_REG_THRESHOLD="$2"; shift 2 ;;
        --sor-ref-update-interval) SOR_REF_UPDATE_INTERVAL="$2"; shift 2 ;;
        --sor-sync-interval) SOR_SYNC_INTERVAL="$2"; shift 2 ;;
        --sor-save-buffer-every) SOR_SAVE_BUFFER_EVERY="$2"; shift 2 ;;
        --offline)      ONLINE=0;              shift ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

case "${SHARED_REPLAY}" in
    true|false) ;;
    *) echo "--shared-replay must be true or false" >&2; exit 2 ;;
esac
case "${ACTION_SPACE}" in
    legacy|multiscale) ;;
    *) echo "--action-space must be legacy or multiscale" >&2; exit 2 ;;
esac
case "${EPSILON_SCHEDULE}" in
    phase|global) ;;
    *) echo "--epsilon-schedule must be phase or global" >&2; exit 2 ;;
esac
[[ "${TARGET_UPDATE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "--target-update-interval must be a positive integer" >&2
    exit 2
}

# ==================== Resolve Experiment List ====================
# If --experiments was used, EXP_LIST has entries. Otherwise single (config, exp).
if [ ${#EXP_LIST[@]} -eq 0 ]; then
    EXP_LIST=("${NS3_CONF}:${EXP_NAME}")
fi

# Validate all configs and expand paths
RESOLVED_LIST=()
for pair in "${EXP_LIST[@]}"; do
    cfg="${pair%%:*}"
    exp="${pair#*:}"
    if [[ "$cfg" != /* ]]; then
        cfg="${COPTER_ROOT}/${cfg}"
    fi
    if [ ! -f "${cfg}" ]; then
        echo "ERROR: NS3 config not found: ${cfg}"
        exit 1
    fi
    RESOLVED_LIST+=("${cfg}:${exp}")
done

NS3_BIN="${NS3_BLD_DIR}/scratch/copter-sim"
if [ ! -f "${NS3_BIN}" ]; then
    echo "ERROR: NS3 binary not found: ${NS3_BIN}"
    echo "Run build_ns3_copter.sh first."
    exit 1
fi

export LD_LIBRARY_PATH="${NS3_LIB_DIR}:${LD_LIBRARY_PATH:-}"
# Do not force verbose ns-3 component logging. NS_LOG_UNCOND diagnostics are
# still captured, while normal runs avoid a large level_all performance cost.
export NS_LOG="${NS_LOG:-}"

SIM_DIR="${COPTER_ROOT}/simulation"
MAX_CONSECUTIVE_FAILURES=5

# ==================== FMAP Flag ====================
# ==================== Run Single Experiment (used by main loop and parallel launcher) ====================
run_single_experiment() {
    local NS3_CONF_LOCAL="$1"
    local EXP_NAME_LOCAL="$2"
    local NS3_PORT_LOCAL="$3"
    local LOG_DIR_LOCAL="$4"
    local TRAIN_LOG_LOCAL="$5"

    mkdir -p "${LOG_DIR_LOCAL}"
    local STATE_FILE="${MODEL_DIR}/${EXP_NAME_LOCAL}_train_state.json"

    get_current_episode() {
        if [ -f "${STATE_FILE}" ]; then
            python3 -c "import json; print(json.load(open('${STATE_FILE}')).get('epoch', 0))" 2>/dev/null || echo "0"
        else
            echo "0"
        fi
    }

    get_current_train_step() {
        if [ -f "${STATE_FILE}" ]; then
            python3 -c "import json; print(json.load(open('${STATE_FILE}')).get('global_train_step', 0))" 2>/dev/null || echo "0"
        else
            echo "0"
        fi
    }

    kill_ns3_processes_local() {
        local pid=$1
        if kill -0 ${pid} 2>/dev/null; then
            kill ${pid} 2>/dev/null || true
            sleep 1
            kill -9 ${pid} 2>/dev/null || true
        fi
        pkill -f "copter-sim.*--port=${NS3_PORT_LOCAL}" 2>/dev/null || true
        sleep 1
    }

    log_local() {
        local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [${EXP_NAME_LOCAL}] $*"
        echo "${msg}"
        echo "${msg}" >> "${TRAIN_LOG_LOCAL}"
    }

    local EPISODE CONSECUTIVE_FAILURES=0 RUN_COUNT=0
    EPISODE=$(get_current_episode)
    log_local "Resuming from episode ${EPISODE}"
    if [ "${TARGET_TRAIN_STEPS}" -gt 0 ] &&
       [ "$(get_current_train_step)" -ge "${TARGET_TRAIN_STEPS}" ]; then
        log_local "Target optimizer updates already reached: ${TARGET_TRAIN_STEPS}"
        return 0
    fi

    while { [ "${ONE_SHOT}" -eq 1 ] && [ "${RUN_COUNT}" -lt 1 ]; } || \
          { [ "${ONE_SHOT}" -eq 0 ] && [ "${EPISODE}" -lt "${MAX_EPISODES}" ]; }; do
        log_local ""
        log_local "========== Episode ${EPISODE}/${MAX_EPISODES} =========="
        local EPISODE_START=$(date +%s)

        > "${LOG_DIR_LOCAL}/ns3_ep${EPISODE}.log"
        > "${LOG_DIR_LOCAL}/agent_ep${EPISODE}.log"

        log_local "Starting NS3 simulation (cwd: ${SIM_DIR})..."
        cd "${SIM_DIR}"
        ${NS3_BIN} "${NS3_CONF_LOCAL}" --port=${NS3_PORT_LOCAL} > "${LOG_DIR_LOCAL}/ns3_ep${EPISODE}.log" 2>&1 &
        local NS3_PID=$!
        cd "${COPTER_ROOT}"
        log_local "NS3 started with PID ${NS3_PID}"

        sleep ${WAIT_NS3_SEC}

        if ! kill -0 ${NS3_PID} 2>/dev/null; then
            log_local "ERROR: NS3 failed to start. Check ${LOG_DIR_LOCAL}/ns3_ep${EPISODE}.log"
            CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
            if [ ${CONSECUTIVE_FAILURES} -ge ${MAX_CONSECUTIVE_FAILURES} ]; then
                log_local "FATAL: ${MAX_CONSECUTIVE_FAILURES} consecutive failures. Aborting."
                return 1
            fi
            if [ "${ONE_SHOT}" -eq 1 ]; then
                log_local "One-shot run will not retry a failed NS3 startup."
                return 1
            fi
            sleep 2
            continue
        fi

        log_local "Starting RL agent (output: ${LOG_DIR_LOCAL}/agent_ep${EPISODE}.log)..."
        local AGENT_WORKDIR="${COPTER_ROOT}/copter"
        if [ "${MODE}" = "SOR" ]; then
            AGENT_WORKDIR="${COPTER_ROOT}/sor"
        fi
        cd "${AGENT_WORKDIR}"

        local AGENT_EXIT=0
        local AGENT_ARGS
        if [ "${MODE}" = "SOR" ]; then
            AGENT_ARGS=(
                python sor_copter.py
                -p "${NS3_PORT_LOCAL}"
                -e "${EXP_NAME_LOCAL}"
                -d "${MODEL_DIR}"
                -s "${STATIC_STEPS}"
                -i "${TRAIN_INTERVALS}"
                -b "${SWITCH_BUFFER}"
                --epsilon_start "${EPSILON_START}"
                --epsilon_end "${EPSILON_END}"
                --epsilon_decay_steps "${EPSILON_DECAY}"
                --epsilon_schedule "${EPSILON_SCHEDULE}"
                --seed "${SEED}"
                --tb_enable "${TB_ENABLE}"
                --acc_hidden_dims "${ACC_HIDDEN_DIMS}"
                --reward_weights "${REWARD_WEIGHTS}"
                --reward_profile "${REWARD_PROFILE}"
                --reward_queue_lambda "${REWARD_QUEUE_LAMBDA}"
                --reward_ecn_lambda "${REWARD_ECN_LAMBDA}"
                --sor_recent_size "${SOR_RECENT_SIZE}"
                --sor_boundary_size "${SOR_BOUNDARY_SIZE}"
                --sor_max_clusters "${SOR_MAX_CLUSTERS}"
                --sor_prototype_distance "${SOR_PROTOTYPE_DISTANCE}"
                --sor_prototype_eta "${SOR_PROTOTYPE_ETA}"
                --sor_boundary_threshold "${SOR_BOUNDARY_THRESHOLD}"
                --sor_alpha_td "${SOR_ALPHA_TD}"
                --sor_beta_under_sample "${SOR_BETA_UNDER_SAMPLE}"
                --sor_gamma_drift "${SOR_GAMMA_DRIFT}"
                --sor_rho_boundary "${SOR_RHO_BOUNDARY}"
                --sor_temperature "${SOR_TEMPERATURE}"
                --sor_lambda_cons "${SOR_LAMBDA_CONS}"
                --sor_lambda_reg "${SOR_LAMBDA_REG}"
                --sor_drift_reg_threshold "${SOR_DRIFT_REG_THRESHOLD}"
                --sor_ref_update_interval "${SOR_REF_UPDATE_INTERVAL}"
                --sor_sync_interval "${SOR_SYNC_INTERVAL}"
                --sor_save_buffer_every "${SOR_SAVE_BUFFER_EVERY}"
            )
        else
            AGENT_ARGS=(
                python copter.py
                -p "${NS3_PORT_LOCAL}"
                -e "${EXP_NAME_LOCAL}"
                -m "${MODE}"
                -d "${MODEL_DIR}"
                -s "${STATIC_STEPS}"
                -i "${TRAIN_INTERVALS}"
                -b "${SWITCH_BUFFER}"
                --epsilon_start "${EPSILON_START}"
                --epsilon_end "${EPSILON_END}"
                --epsilon_decay_steps "${EPSILON_DECAY}"
                --epsilon_schedule "${EPSILON_SCHEDULE}"
                --target_update_interval "${TARGET_UPDATE_INTERVAL}"
                --seed "${SEED}"
                --tb_enable "${TB_ENABLE}"
                --acc_hidden_dims "${ACC_HIDDEN_DIMS}"
                --action_space "${ACTION_SPACE}"
                --reward_weights "${REWARD_WEIGHTS}"
                --reward_profile "${REWARD_PROFILE}"
                --reward_queue_lambda "${REWARD_QUEUE_LAMBDA}"
                --reward_ecn_lambda "${REWARD_ECN_LAMBDA}"
                --shared_replay "${SHARED_REPLAY}"
            )
        fi
        [ "${ONLINE}" -eq 1 ] && AGENT_ARGS+=(--online)
        [ -n "${FMAP_DIR}" ] && AGENT_ARGS+=(-f "${FMAP_DIR}")
        [ -n "${FORCE_ACTION}" ] && AGENT_ARGS+=(--force_action "${FORCE_ACTION}")
        [ -n "${FORCE_PORT_ACTION}" ] && AGENT_ARGS+=(--force_port_action "${FORCE_PORT_ACTION}")
        [ "${EVAL_GREEDY}" -eq 1 ] && AGENT_ARGS+=(--eval_greedy)
        [ -n "${EVAL_TAG}" ] && AGENT_ARGS+=(--eval_tag "${EVAL_TAG}")
        [ -n "${WATCH_PORTS}" ] && AGENT_ARGS+=(--watch_ports "${WATCH_PORTS}")
        [ -n "${WATCH_TRACE_FILE}" ] && AGENT_ARGS+=(--watch_trace_file "${WATCH_TRACE_FILE}")
        [ "${MAX_STEPS}" -gt 0 ] && AGENT_ARGS+=(--max_steps "${MAX_STEPS}")
        [ "${TARGET_TRAIN_STEPS}" -gt 0 ] && AGENT_ARGS+=(--max_global_train_steps "${TARGET_TRAIN_STEPS}")
        [ -n "${RUN_ID}" ] && AGENT_ARGS+=(--run_id "${RUN_ID}")
        [ -n "${PHASE}" ] && AGENT_ARGS+=(--phase "${PHASE}")
        "${AGENT_ARGS[@]}" >> "${LOG_DIR_LOCAL}/agent_ep${EPISODE}.log" 2>&1 || AGENT_EXIT=$?

        cd "${COPTER_ROOT}"

        if kill -0 ${NS3_PID} 2>/dev/null; then
            log_local "Waiting for NS3 (PID ${NS3_PID}) to finish (max 60s)..."
            local NS3_WAIT=0
            while kill -0 ${NS3_PID} 2>/dev/null && [ ${NS3_WAIT} -lt 60 ]; do
                sleep 1
                NS3_WAIT=$((NS3_WAIT + 1))
            done
        fi
        kill_ns3_processes_local ${NS3_PID}
        wait ${NS3_PID} 2>/dev/null || true

        local EPISODE_END=$(date +%s)
        local EPISODE_DURATION=$((EPISODE_END - EPISODE_START))
        log_local "Episode ${EPISODE} finished in ${EPISODE_DURATION}s (agent exit: ${AGENT_EXIT})"

        if [ ${AGENT_EXIT} -eq 0 ]; then
            CONSECUTIVE_FAILURES=0
        else
            CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
            log_local "WARNING: Agent exited with code ${AGENT_EXIT} (failure ${CONSECUTIVE_FAILURES}/${MAX_CONSECUTIVE_FAILURES})"
            log_local "Last 40 lines of agent log:"
            tail -n 40 "${LOG_DIR_LOCAL}/agent_ep${EPISODE}.log" >&2 || true
            if [ ${CONSECUTIVE_FAILURES} -ge ${MAX_CONSECUTIVE_FAILURES} ]; then
                log_local "FATAL: ${MAX_CONSECUTIVE_FAILURES} consecutive failures. Aborting."
                return 1
            fi
        fi

        RUN_COUNT=$((RUN_COUNT + 1))
        if [ "${ONE_SHOT}" -eq 1 ]; then
            [ "${AGENT_EXIT}" -eq 0 ] || return "${AGENT_EXIT}"
            break
        fi

        if [ "${TARGET_TRAIN_STEPS}" -gt 0 ] &&
           [ "$(get_current_train_step)" -ge "${TARGET_TRAIN_STEPS}" ]; then
            log_local "Reached target optimizer updates: ${TARGET_TRAIN_STEPS}"
            break
        fi

        # A registered optimizer-update budget takes precedence over the
        # heuristic convergence detector so both tasks receive equal compute.
        if [ "${TARGET_TRAIN_STEPS}" -eq 0 ] && [ -f "${STATE_FILE}" ]; then
            local CONVERGED
            CONVERGED=$(python3 -c "
import json, sys, numpy as np
with open('${STATE_FILE}') as f:
    s = json.load(f)
rh = s.get('reward_history', [])
lh = s.get('loss_history', [])
min_ep = max(40, 50)
if len(rh) >= min_ep and len(lh) >= min_ep:
    rr = rh[-20:]
    ll = lh[-20:]
    reward_std = float(np.std(rr))
    if reward_std < 1e-10:
        print('no')
        sys.exit(0)
    reward_stable = (max(rr) - min(rr)) < 0.02
    early_reward = np.mean(rh[1:21]) if len(rh) > 20 else 0
    has_improved = np.mean(rr) > early_reward + 1e-6
    loss_cv = float(np.std(ll) / (np.mean(ll) + 1e-8))
    loss_stable = loss_cv < 0.3
    loss_ok = ll[-1] < 100.0
    print('yes' if (reward_stable and has_improved and loss_stable and loss_ok) else 'no')
else:
    print('no')
" 2>/dev/null || echo "no")

            if [ "${CONVERGED}" = "yes" ]; then
                log_local "CONVERGED! Training complete."
                break
            fi
        fi

        EPISODE=$(get_current_episode)

        if [ ${EPISODE} -lt ${MAX_EPISODES} ]; then
            log_local "Waiting ${WAIT_BETWEEN_SEC}s before next episode..."
            sleep ${WAIT_BETWEEN_SEC}
        fi
    done

    if [ "${TARGET_TRAIN_STEPS}" -gt 0 ] &&
       [ "$(get_current_train_step)" -lt "${TARGET_TRAIN_STEPS}" ]; then
        log_local "ERROR: Episode safety cap reached before optimizer-update target."
        log_local "Current updates=$(get_current_train_step), target=${TARGET_TRAIN_STEPS}"
        return 1
    fi

    log_local ""
    log_local "=============================================="
    log_local "Training Complete - Total Episodes: ${EPISODE}"
    log_local "TensorBoard: tensorboard --logdir ${COPTER_ROOT}/copter/tb_logs/"
    log_local "=============================================="
}

# ==================== Main Entry ====================
LOG_BASE="${COPTER_ROOT}/copter/training_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [ ${#RESOLVED_LIST[@]} -eq 1 ]; then
    # Single experiment: run in foreground
    pair="${RESOLVED_LIST[0]}"
    NS3_CONF_S="${pair%%:*}"
    EXP_NAME_S="${pair#*:}"
    LOG_DIR_S="${LOG_BASE}/${EXP_NAME_S}"
    TRAIN_LOG_S="${LOG_BASE}/training_${EXP_NAME_S}_${TIMESTAMP}.log"

    echo "=============================================="
    echo "CoPTER Single Experiment"
    echo "Config: ${NS3_CONF_S}"
    echo "Exp:    ${EXP_NAME_S}"
    echo "Port:   ${BASE_PORT}"
    echo "=============================================="

    run_single_experiment "${NS3_CONF_S}" "${EXP_NAME_S}" "${BASE_PORT}" "${LOG_DIR_S}" "${TRAIN_LOG_S}"
else
    # Multiple experiments: run in parallel (background)
    echo "=============================================="
    echo "CoPTER Multi-Experiment (Parallel)"
    echo "Experiments: ${#RESOLVED_LIST[@]}"
    for i in "${!RESOLVED_LIST[@]}"; do
        pair="${RESOLVED_LIST[$i]}"
        cfg="${pair%%:*}"
        exp="${pair#*:}"
        port=$((BASE_PORT + i))
        echo "  [$i] ${exp} @ port ${port} (${cfg})"
    done
    echo "=============================================="

    PIDS=()
    for i in "${!RESOLVED_LIST[@]}"; do
        pair="${RESOLVED_LIST[$i]}"
        cfg="${pair%%:*}"
        exp="${pair#*:}"
        port=$((BASE_PORT + i))
        LOG_DIR_P="${LOG_BASE}/${exp}"
        TRAIN_LOG_P="${LOG_BASE}/training_${exp}_${TIMESTAMP}.log"

        run_single_experiment "${cfg}" "${exp}" "${port}" "${LOG_DIR_P}" "${TRAIN_LOG_P}" &
        PIDS+=($!)
    done

    echo "Waiting for ${#PIDS[@]} experiments to complete..."
    FAILED=0
    for i in "${!PIDS[@]}"; do
        if wait ${PIDS[$i]}; then
            echo "  Experiment ${RESOLVED_LIST[$i]#*:} (PID ${PIDS[$i]}) finished successfully."
        else
            echo "  Experiment ${RESOLVED_LIST[$i]#*:} (PID ${PIDS[$i]}) failed."
            FAILED=$((FAILED + 1))
        fi
    done

    echo ""
    echo "=============================================="
    echo "All experiments complete. Failed: ${FAILED}/${#PIDS[@]}"
    echo "TensorBoard: tensorboard --logdir ${COPTER_ROOT}/copter/tb_logs/"
    echo "=============================================="
    [ ${FAILED} -eq 0 ] || exit 1
fi
