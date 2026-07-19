#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEEDS="1 2 3"
SCENARIOS="throughput incast mixed"
BUFFER_KB=400
KMIN_RANGE="20000,50000"
KMAX_RANGE="50000,100000"
MAX_FLOWS=0
BASELINE_STOP_TIME="2.25"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)       SEEDS="$2"; shift 2 ;;
        --scenarios)   SCENARIOS="$2"; shift 2 ;;
        --buffer-kb)   BUFFER_KB="$2"; shift 2 ;;
        --kmin-range)  KMIN_RANGE="$2"; shift 2 ;;
        --kmax-range)  KMAX_RANGE="$2"; shift 2 ;;
        --max-flows)   MAX_FLOWS="$2"; shift 2 ;;
        --baseline-stop-time) BASELINE_STOP_TIME="$2"; shift 2 ;;
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
if ! [[ "${BASELINE_STOP_TIME}" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
   ! awk -v value="${BASELINE_STOP_TIME}" 'BEGIN {exit !(value > 2.08)}'; then
    echo "Invalid baseline stop time: ${BASELINE_STOP_TIME} (must be greater than 2.08s)" >&2
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
            --max-flows "${MAX_FLOWS}" \
            --no-json

        flow_file="${FLOW_DIR}/${name}.flow"
        declared="$(sed -n '1p' "${flow_file}")"
        actual="$(( $(wc -l < "${flow_file}") - 1 ))"
        if [[ "${declared}" -ne "${actual}" ]]; then
            echo "Invalid flow file ${flow_file}: header=${declared}, rows=${actual}" >&2
            exit 1
        fi

        {
            echo "scenario=${scenario}"
            echo "seed=${seed}"
            echo "flows=${actual}"
            echo "max_flows=${MAX_FLOWS}"
            echo "buffer_kb=${BUFFER_KB}"
            echo "kmin_range=${KMIN_RANGE}"
            echo "kmax_range=${KMAX_RANGE}"
            echo "baseline_stop_time=${BASELINE_STOP_TIME}"
        } > "${FLOW_DIR}/${name}.meta"

        render_config() {
            local output_name="$1" enable_copter="$2" stop_time="$3"
            local kmin_map="$4" kmax_map="$5" pmax_map="$6"
            local destination="$7"
            sed \
                -e "s/@SCENARIO@/${scenario}/g" \
                -e "s/@SEED@/${seed}/g" \
                -e "s/@OUTPUT_NAME@/${output_name}/g" \
                -e "s/@ENABLE_COPTER@/${enable_copter}/g" \
                -e "s/@STOP_TIME@/${stop_time}/g" \
                -e "s/@BUFFER_KB@/${BUFFER_KB}/g" \
                -e "s/@KMIN_MIN@/${KMIN_MIN}/g" \
                -e "s/@KMIN_MAX@/${KMIN_MAX}/g" \
                -e "s/@KMAX_MIN@/${KMAX_MIN}/g" \
                -e "s/@KMAX_MAX@/${KMAX_MAX}/g" \
                -e "s/@KMIN_MAP@/${kmin_map}/g" \
                -e "s/@KMAX_MAP@/${kmax_map}/g" \
                -e "s/@PMAX_MAP@/${pmax_map}/g" \
                "${TEMPLATE}" > "${destination}"
        }

        # Dynamic ACC configuration. The initial map is replaced by OpenGym
        # actions after the agent connects.
        render_config "${name}" 1 "4.00" \
            "2 10000000000 16 40000000000 64" \
            "2 10000000000 32 40000000000 128" \
            "2 10000000000 0.20 40000000000 0.20" \
            "${FLOW_DIR}/${name}.conf"

        # Paper static expert baselines. Values are applied literally on both
        # link rates; the @10/@25 Gbps labels describe the source experiments,
        # not an undocumented scaling rule.
        render_config "${name}_secn1" 0 "${BASELINE_STOP_TIME}" \
            "2 10000000000 5 40000000000 5" \
            "2 10000000000 200 40000000000 200" \
            "2 10000000000 0.01 40000000000 0.01" \
            "${FLOW_DIR}/${name}_secn1.conf"
        render_config "${name}_secn2" 0 "${BASELINE_STOP_TIME}" \
            "2 10000000000 100 40000000000 100" \
            "2 10000000000 400 40000000000 400" \
            "2 10000000000 0.20 40000000000 0.20" \
            "${FLOW_DIR}/${name}_secn2.conf"

        echo "Prepared ${name}: ${actual} flows, buffer=${BUFFER_KB} KB, Kmin=${KMIN_RANGE}, Kmax=${KMAX_RANGE}; static=SECN_1/SECN_2 stop=${BASELINE_STOP_TIME}s"
    done
done

echo "Scenario files are ready under ${FLOW_DIR}"
