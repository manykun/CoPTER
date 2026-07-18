#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEEDS="1 2 3"
SCENARIOS="throughput incast mixed"
BUFFER_KB=400
KMIN_RANGE="20000,50000"
KMAX_RANGE="50000,100000"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)       SEEDS="$2"; shift 2 ;;
        --scenarios)   SCENARIOS="$2"; shift 2 ;;
        --buffer-kb)   BUFFER_KB="$2"; shift 2 ;;
        --kmin-range)  KMIN_RANGE="$2"; shift 2 ;;
        --kmax-range)  KMAX_RANGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

FLOW_DIR="${ROOT}/simulation/mix/acc_validation"
OUTPUT_DIR="${ROOT}/simulation/output/acc_validation"
TEMPLATE="${ROOT}/scripts/acc_validation/acc_validation.conf.in"
mkdir -p "${FLOW_DIR}" "${OUTPUT_DIR}"
IFS=',' read -r KMIN_MIN KMIN_MAX <<< "${KMIN_RANGE}"
IFS=',' read -r KMAX_MIN KMAX_MAX <<< "${KMAX_RANGE}"
if (( KMIN_MIN < 0 || KMIN_MIN >= KMIN_MAX || KMAX_MIN < 0 || KMAX_MIN >= KMAX_MAX )); then
    echo "Invalid Kmin/Kmax ranges" >&2
    exit 2
fi

for scenario in ${SCENARIOS}; do
    scenario_config="${ROOT}/tools/traffic/acc_validation/${scenario}.json"
    if [[ ! -f "${scenario_config}" ]]; then
        echo "Scenario does not exist: ${scenario_config}" >&2
        exit 1
    fi
    for seed in ${SEEDS}; do
        name="${scenario}_seed${seed}"
        python "${ROOT}/tools/traffic/TraGen.py" \
            --config "${scenario_config}" \
            --seed "${seed}" \
            --output-dir "${FLOW_DIR}" \
            --name "${name}" \
            --no-json

        flow_file="${FLOW_DIR}/${name}.flow"
        declared="$(sed -n '1p' "${flow_file}")"
        actual="$(( $(wc -l < "${flow_file}") - 1 ))"
        if [[ "${declared}" -ne "${actual}" ]]; then
            echo "Invalid flow file ${flow_file}: header=${declared}, rows=${actual}" >&2
            exit 1
        fi

        sed \
            -e "s/@SCENARIO@/${scenario}/g" \
            -e "s/@SEED@/${seed}/g" \
            -e "s/@BUFFER_KB@/${BUFFER_KB}/g" \
            -e "s/@KMIN_MIN@/${KMIN_MIN}/g" \
            -e "s/@KMIN_MAX@/${KMIN_MAX}/g" \
            -e "s/@KMAX_MIN@/${KMAX_MIN}/g" \
            -e "s/@KMAX_MAX@/${KMAX_MAX}/g" \
            "${TEMPLATE}" > "${FLOW_DIR}/${name}.conf"
        echo "Prepared ${name}: ${actual} flows, buffer=${BUFFER_KB} KB, Kmin=${KMIN_RANGE}, Kmax=${KMAX_RANGE}"
    done
done

echo "Scenario files are ready under ${FLOW_DIR}"
