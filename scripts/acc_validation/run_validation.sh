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
MAX_FLOWS=0

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
    prepare|sensitivity|baseline|train|eval|analyze|all) ;;
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
    MAX_FLOWS=2000
fi

RUN_DIR="${ROOT}/experiments/acc_validation/${RUN_ID}"
MODEL_DIR="${RUN_DIR}/models"
NS3_BIN="${ROOT}/ns-3.33/build/scratch/copter-sim"
NS3_LIB_DIR="${ROOT}/ns-3.33/build/lib"
mkdir -p "${RUN_DIR}" "${MODEL_DIR}/sweep" "${MODEL_DIR}/train" "${RUN_DIR}/logs"

validate_prepared_configs() {
    local expected_kmin_min="${KMIN_RANGE%%,*}" expected_kmin_max="${KMIN_RANGE#*,}"
    local expected_kmax_min="${KMAX_RANGE%%,*}" expected_kmax_max="${KMAX_RANGE#*,}"
    for scenario in ${SCENARIOS}; do
        for seed in ${SEEDS}; do
            local config="${ROOT}/simulation/mix/acc_validation/${scenario}_seed${seed}.conf"
            [[ -f "${config}" ]] || { echo "Missing ${config}; run --stage prepare first" >&2; return 1; }
            local flow_file="${ROOT}/simulation/mix/acc_validation/${scenario}_seed${seed}.flow"
            [[ -f "${flow_file}" ]] || { echo "Missing ${flow_file}; run --stage prepare first" >&2; return 1; }
            local meta_file="${ROOT}/simulation/mix/acc_validation/${scenario}_seed${seed}.meta"
            [[ -f "${meta_file}" ]] || { echo "Missing ${meta_file}; run --stage prepare again" >&2; return 1; }
            local prepared_max_flows
            prepared_max_flows="$(awk -F= '$1 == "max_flows" {print $2}' "${meta_file}")"
            if [[ "${prepared_max_flows}" != "${MAX_FLOWS}" ]]; then
                echo "Prepared flow cap (${prepared_max_flows}) does not match requested cap (${MAX_FLOWS})." >&2
                echo "Rerun this run-id's prepare stage with the same --smoke setting." >&2
                return 1
            fi
            if [[ "${SMOKE}" -eq 1 ]] && [[ "$(sed -n '1p' "${flow_file}")" -gt "${MAX_FLOWS}" ]]; then
                echo "Smoke flow has more than ${MAX_FLOWS} rows; rerun --stage prepare --smoke." >&2
                return 1
            fi
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

            local profile profile_config expected_kmin expected_kmax expected_pmax
            for profile in secn1 secn2; do
                profile_config="${ROOT}/simulation/mix/acc_validation/${scenario}_seed${seed}_${profile}.conf"
                [[ -f "${profile_config}" ]] || {
                    echo "Missing ${profile_config}; run --stage prepare again" >&2
                    return 1
                }
                if [[ "${profile}" == secn1 ]]; then
                    expected_kmin="KMIN_MAP 2 10000000000 5 40000000000 5"
                    expected_kmax="KMAX_MAP 2 10000000000 200 40000000000 200"
                    expected_pmax="PMAX_MAP 2 10000000000 0.01 40000000000 0.01"
                else
                    expected_kmin="KMIN_MAP 2 10000000000 100 40000000000 100"
                    expected_kmax="KMAX_MAP 2 10000000000 400 40000000000 400"
                    expected_pmax="PMAX_MAP 2 10000000000 0.20 40000000000 0.20"
                fi
                if ! grep -Fxq "ENABLE_COPTER 0" "${profile_config}" ||
                   ! grep -Fxq "${expected_kmin}" "${profile_config}" ||
                   ! grep -Fxq "${expected_kmax}" "${profile_config}" ||
                   ! grep -Fxq "${expected_pmax}" "${profile_config}"; then
                    echo "Prepared paper baseline has unexpected parameters: ${profile_config}" >&2
                    echo "Run --stage prepare again; do not reuse old static configs." >&2
                    return 1
                fi
            done
        done
    done
}

prepare() {
    bash "${ROOT}/scripts/acc_validation/prepare_scenarios.sh" \
        --seeds "${SEEDS}" \
        --scenarios "${SCENARIOS}" \
        --buffer-kb "${BUFFER_KB}" \
        --kmin-range "${KMIN_RANGE}" \
        --kmax-range "${KMAX_RANGE}" \
        --max-flows "${MAX_FLOWS}"
}

copy_outputs() {
    local scenario="$1" seed="$2" output_name="$3" config_name="$4" destination="$5"
    local metrics_file="${6:-}"
    local base="${ROOT}/simulation/output/acc_validation/${output_name}"
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
    cp "${ROOT}/simulation/mix/acc_validation/${config_name}.conf" \
        "${destination}/input.conf"
    cp "${ROOT}/simulation/mix/acc_validation/${scenario}_seed${seed}.meta" \
        "${destination}/input.meta"
    shopt -u nullglob
    if [[ -n "${metrics_file}" && -f "${metrics_file}" ]]; then
        tail -n 1 "${metrics_file}" > "${destination}/metrics.json"
    elif [[ -n "${metrics_file}" ]]; then
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
                local exp="accval_sweep_${RUN_ID}_${scenario}_${label}_s${seed}"
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
                    --model-dir "${MODEL_DIR}/sweep" \
                    --episodes 1 \
                    --run-id "${RUN_ID}" \
                    --phase sensitivity
                copy_outputs "${scenario}" "${seed}" "${scenario}_seed${seed}" \
                    "${scenario}_seed${seed}" \
                    "${RUN_DIR}/sensitivity/${scenario}/seed_${seed}/${label}" \
                    "${MODEL_DIR}/sweep/${exp}_metrics.jsonl"
            done
        done
    done
}

run_paper_baselines() {
    if [[ ! -x "${NS3_BIN}" ]]; then
        echo "ERROR: NS3 binary not found: ${NS3_BIN}" >&2
        echo "Run bash build_ns3_copter.sh after pulling this change." >&2
        return 1
    fi
    for scenario in ${SCENARIOS}; do
        for seed in ${SEEDS}; do
            for profile in secn1 secn2; do
                local config_name="${scenario}_seed${seed}_${profile}"
                local config="${ROOT}/simulation/mix/acc_validation/${config_name}.conf"
                local destination="${RUN_DIR}/baseline/${scenario}/seed_${seed}/${profile}"
                local log_file="${RUN_DIR}/logs/baseline_${scenario}_${profile}_s${seed}.log"
                local output_base="${ROOT}/simulation/output/acc_validation/${config_name}"
                echo "[baseline] scenario=${scenario} seed=${seed} profile=${profile}"
                rm -f "${output_base}".*
                (
                    cd "${ROOT}/simulation"
                    LD_LIBRARY_PATH="${NS3_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
                        "${NS3_BIN}" "${config}"
                ) > "${log_file}" 2>&1
                copy_outputs "${scenario}" "${seed}" "${config_name}" \
                    "${config_name}" "${destination}"
                cp "${log_file}" "${destination}/ns3.log"
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
            copy_outputs "${scenario}" "${seed}" "${scenario}_seed${seed}" \
                "${scenario}_seed${seed}" \
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
[[ "${STAGE}" == baseline || "${STAGE}" == all ]] && { validate_prepared_configs; run_paper_baselines; }
[[ "${STAGE}" == train || "${STAGE}" == all ]] && { validate_prepared_configs; run_train; }
[[ "${STAGE}" == eval || "${STAGE}" == all ]] && { validate_prepared_configs; run_eval; }
[[ "${STAGE}" == analyze || "${STAGE}" == all ]] && analyze

echo "ACC validation stage '${STAGE}' complete: ${RUN_DIR}"
