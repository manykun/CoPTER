#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="all"
RUN_ID="accval_$(date +%Y%m%d_%H%M%S)"
SEEDS="1 2 3"
SCENARIOS="throughput incast mixed"
BUFFER_KB=400
EPISODES=50
EPS_DECAY=5000
ACC_HIDDEN_DIMS="32,64,64,32"
REWARD_WEIGHTS="0.50,0.30,0.20"
KMIN_RANGE="20000,50000"
KMAX_RANGE="50000,100000"
SMOKE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)       STAGE="$2"; shift 2 ;;
        --run-id)      RUN_ID="$2"; shift 2 ;;
        --seeds)       SEEDS="$2"; shift 2 ;;
        --scenarios)   SCENARIOS="$2"; shift 2 ;;
        --buffer-kb)   BUFFER_KB="$2"; shift 2 ;;
        --episodes)    EPISODES="$2"; shift 2 ;;
        --eps-decay)   EPS_DECAY="$2"; shift 2 ;;
        --acc-hidden-dims) ACC_HIDDEN_DIMS="$2"; shift 2 ;;
        --reward-weights) REWARD_WEIGHTS="$2"; shift 2 ;;
        --kmin-range) KMIN_RANGE="$2"; shift 2 ;;
        --kmax-range) KMAX_RANGE="$2"; shift 2 ;;
        --smoke)       SMOKE=1; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "${STAGE}" in
    prepare|sensitivity|train|eval|analyze|all) ;;
    *) echo "Invalid stage: ${STAGE}" >&2; exit 2 ;;
esac

if [[ "${RUN_ID}" =~ [^A-Za-z0-9_.-] ]]; then
    echo "run-id may contain only letters, digits, '.', '_' and '-'" >&2
    exit 2
fi

if [[ "${SMOKE}" -eq 1 ]]; then
    SEEDS="${SEEDS%% *}"
    SCENARIOS="${SCENARIOS%% *}"
    EPISODES=1
fi

RUN_DIR="${ROOT}/experiments/acc_validation/${RUN_ID}"
MODEL_DIR="${RUN_DIR}/models"
mkdir -p "${RUN_DIR}" "${MODEL_DIR}/static" "${MODEL_DIR}/train"

validate_prepared_configs() {
    local expected_kmin_min="${KMIN_RANGE%%,*}" expected_kmin_max="${KMIN_RANGE#*,}"
    local expected_kmax_min="${KMAX_RANGE%%,*}" expected_kmax_max="${KMAX_RANGE#*,}"
    for scenario in ${SCENARIOS}; do
        for seed in ${SEEDS}; do
            local config="${ROOT}/simulation/mix/acc_validation/${scenario}_seed${seed}.conf"
            [[ -f "${config}" ]] || { echo "Missing ${config}; run --stage prepare first" >&2; return 1; }
            local actual_buffer actual_kmin_min actual_kmin_max actual_kmax_min actual_kmax_max
            actual_buffer="$(awk '$1 == "BUFFER_SIZE" {print $2}' "${config}")"
            actual_kmin_min="$(awk '$1 == "OPENGYM_MIN_KMIN" {print $2}' "${config}")"
            actual_kmin_max="$(awk '$1 == "OPENGYM_MAX_KMIN" {print $2}' "${config}")"
            actual_kmax_min="$(awk '$1 == "OPENGYM_MIN_KMAX" {print $2}' "${config}")"
            actual_kmax_max="$(awk '$1 == "OPENGYM_MAX_KMAX" {print $2}' "${config}")"
            if [[ "${actual_buffer}" != "${BUFFER_KB}" ||
                  "${actual_kmin_min},${actual_kmin_max}" != "${expected_kmin_min},${expected_kmin_max}" ||
                  "${actual_kmax_min},${actual_kmax_max}" != "${expected_kmax_min},${expected_kmax_max}" ]]; then
                echo "Prepared config does not match requested buffer/K ranges: ${config}" >&2
                echo "Run this run-id's prepare stage again with the same arguments." >&2
                return 1
            fi
        done
    done
}

prepare() {
    bash "${ROOT}/scripts/acc_validation/prepare_scenarios.sh" \
        --seeds "${SEEDS}" \
        --scenarios "${SCENARIOS}" \
        --buffer-kb "${BUFFER_KB}" \
        --kmin-range "${KMIN_RANGE}" \
        --kmax-range "${KMAX_RANGE}"
}

copy_outputs() {
    local scenario="$1" seed="$2" destination="$3" metrics_file="$4"
    local base="${ROOT}/simulation/output/acc_validation/${scenario}_seed${seed}"
    mkdir -p "${destination}"
    shopt -s nullglob
    local files=("${base}".*)
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No ns-3 output files found for ${base}" >&2
        return 1
    fi
    cp "${files[@]}" "${destination}/"
    cp "${ROOT}/simulation/mix/acc_validation/${scenario}_seed${seed}.flow" \
        "${destination}/input.flow"
    cp "${ROOT}/simulation/mix/acc_validation/${scenario}_seed${seed}.conf" \
        "${destination}/input.conf"
    shopt -u nullglob
    if [[ -f "${metrics_file}" ]]; then
        tail -n 1 "${metrics_file}" > "${destination}/metrics.json"
    else
        echo "Metrics file not found: ${metrics_file}" >&2
        return 1
    fi
}

run_sensitivity() {
    local action_specs=(
        "aggressive:0,0,9"
        "balanced:2,1,4"
        "permissive:5,3,0"
    )
    for scenario in ${SCENARIOS}; do
        for seed in ${SEEDS}; do
            local config="simulation/mix/acc_validation/${scenario}_seed${seed}.conf"
            for spec in "${action_specs[@]}"; do
                local label="${spec%%:*}"
                local action="${spec#*:}"
                local exp="accval_static_${RUN_ID}_${scenario}_${label}_s${seed}"
                echo "[sensitivity] scenario=${scenario} seed=${seed} action=${label}(${action})"
                bash "${ROOT}/run_training.sh" \
                    --one-shot \
                    --eval-greedy \
                    --force-action "${action}" \
                    --seed "${seed}" \
                    --tb-enable false \
                    --config "${config}" \
                    --exp "${exp}" \
                    --buffer "${BUFFER_KB}" \
                    --reward-weights "${REWARD_WEIGHTS}" \
                    --model-dir "${MODEL_DIR}/static" \
                    --episodes 1 \
                    --run-id "${RUN_ID}" \
                    --phase sensitivity
                copy_outputs "${scenario}" "${seed}" \
                    "${RUN_DIR}/sensitivity/${scenario}/seed_${seed}/${label}" \
                    "${MODEL_DIR}/static/${exp}_metrics.jsonl"
            done
        done
    done
}

run_train() {
    for scenario in ${SCENARIOS}; do
        for seed in ${SEEDS}; do
            local exp="accval_train_${RUN_ID}_${scenario}_s${seed}"
            echo "[train] scenario=${scenario} seed=${seed} episodes=${EPISODES}"
            bash "${ROOT}/run_training.sh" \
                --config "simulation/mix/acc_validation/${scenario}_seed${seed}.conf" \
                --exp "${exp}" \
                --mode ACC \
                --seed "${seed}" \
                --buffer "${BUFFER_KB}" \
                --model-dir "${MODEL_DIR}/train" \
                --episodes "${EPISODES}" \
                --eps-start 1.0 \
                --eps-end 0.05 \
                --eps-decay "${EPS_DECAY}" \
                --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
                --reward-weights "${REWARD_WEIGHTS}" \
                --run-id "${RUN_ID}" \
                --phase train
        done
    done
}

run_eval() {
    for scenario in ${SCENARIOS}; do
        for seed in ${SEEDS}; do
            local exp="accval_train_${RUN_ID}_${scenario}_s${seed}"
            echo "[eval] scenario=${scenario} seed=${seed}"
            bash "${ROOT}/run_training.sh" \
                --one-shot \
                --eval-greedy \
                --eval-tag validation_greedy \
                --seed "${seed}" \
                --tb-enable false \
                --config "simulation/mix/acc_validation/${scenario}_seed${seed}.conf" \
                --exp "${exp}" \
                --buffer "${BUFFER_KB}" \
                --model-dir "${MODEL_DIR}/train" \
                --episodes 1 \
                --run-id "${RUN_ID}" \
                --phase eval \
                --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
                --reward-weights "${REWARD_WEIGHTS}"
            copy_outputs "${scenario}" "${seed}" \
                "${RUN_DIR}/eval/${scenario}/seed_${seed}/greedy" \
                "${MODEL_DIR}/train/${exp}_metrics.jsonl"
        done
    done
}

analyze() {
    python "${ROOT}/scripts/acc_validation/analyze_validation.py" \
        --run-dir "${RUN_DIR}" \
        --stage all
}

[[ "${STAGE}" == prepare || "${STAGE}" == all ]] && prepare
[[ "${STAGE}" == sensitivity || "${STAGE}" == all ]] && { validate_prepared_configs; run_sensitivity; }
[[ "${STAGE}" == train || "${STAGE}" == all ]] && { validate_prepared_configs; run_train; }
[[ "${STAGE}" == eval || "${STAGE}" == all ]] && { validate_prepared_configs; run_eval; }
[[ "${STAGE}" == analyze || "${STAGE}" == all ]] && analyze

echo "ACC validation stage '${STAGE}' complete: ${RUN_DIR}"
