#!/usr/bin/env bash
# Calibrate temporal burstiness before launching the long ACC A->B curriculum.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="all"
RUN_ID="spread_calibration_s1"
SEED=1
BUFFER_KB=400
WATCH_PORTS="all"
PORT=6256
SPREAD_PROFILE="micro"
REWARD_PROFILE="tail_safe"
REWARD_QUEUE_LAMBDA="5.0"
REWARD_ECN_LAMBDA="5.0"
REWARD_WEIGHTS="0.50,0.30,0.20"
ACC_HIDDEN_DIMS="32,64,64,32"
RESUME=0
CANDIDATES=()
CANDIDATE_RECORDS=()
SPECIFICATIONS=(
    "low_strong:1,5"
    "mid:4,3"
    "high_gentle:8,1"
)

usage() {
    cat <<'EOF'
Usage:
  bash scripts/continual_validation/run_spread_calibration.sh [options]

Stages:
  prepare   Generate and freeze the selected timing-spread candidates
  run       Evaluate three representative fixed actions per candidate
  analyze   Recommend an eligible steady->bursty task pair
  all       Run prepare, run, and analyze

Options:
  --run-id ID
  --seed N
  --buffer-kb N
  --spread-profile NAME  coarse, micro, micro32, or micro48 (default: micro)
  --watch-ports CSV      Comma-separated ports, or all (default)
  --watch-port N         Backward-compatible single-port form
  --port N
  --reward-profile NAME
  --reward-queue-lambda X
  --reward-ecn-lambda X
  --reward-weights CSV
  --acc-hidden-dims CSV
  --resume
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --buffer-kb) BUFFER_KB="$2"; shift 2 ;;
        --spread-profile) SPREAD_PROFILE="$2"; shift 2 ;;
        --watch-ports) WATCH_PORTS="$2"; shift 2 ;;
        --watch-port) WATCH_PORTS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --reward-profile) REWARD_PROFILE="$2"; shift 2 ;;
        --reward-queue-lambda) REWARD_QUEUE_LAMBDA="$2"; shift 2 ;;
        --reward-ecn-lambda) REWARD_ECN_LAMBDA="$2"; shift 2 ;;
        --reward-weights) REWARD_WEIGHTS="$2"; shift 2 ;;
        --acc-hidden-dims) ACC_HIDDEN_DIMS="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${STAGE}" in
    prepare|run|analyze|all) ;;
    *) echo "Invalid stage: ${STAGE}" >&2; exit 2 ;;
esac
case "${SPREAD_PROFILE}" in
    coarse)
        CANDIDATE_RECORDS=(
            "samepath_spread070_stress:0.70"
            "samepath_spread050_stress:0.50"
            "samepath_spread030_stress:0.30"
            "samepath_spread010_stress:0.10"
        )
        ;;
    micro)
        CANDIDATE_RECORDS=(
            "samepath_spread002_stress:0.02"
            "samepath_spread001_stress:0.01"
            "samepath_spread0005_stress:0.005"
            "samepath_spread0002_stress:0.002"
            "samepath_spread0001_stress:0.001"
        )
        ;;
    micro32)
        CANDIDATE_RECORDS=(
            "samepath32_spread005_stress:0.05"
            "samepath32_spread002_stress:0.02"
            "samepath32_spread001_stress:0.01"
            "samepath32_spread0005_stress:0.005"
        )
        ;;
    micro48)
        CANDIDATE_RECORDS=(
            "samepath48_spread005_stress:0.05"
            "samepath48_spread002_stress:0.02"
            "samepath48_spread001_stress:0.01"
        )
        ;;
    *) echo "spread-profile must be coarse, micro, micro32, or micro48" >&2; exit 2 ;;
esac
for record in "${CANDIDATE_RECORDS[@]}"; do
    CANDIDATES+=("${record%%:*}")
done
[[ "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "run-id contains unsupported characters" >&2
    exit 2
}
for value in "${SEED}" "${BUFFER_KB}" "${PORT}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || {
        echo "seed, buffer, and socket port must be integers" >&2
        exit 2
    }
done
if [[ "${WATCH_PORTS}" == "all" ]]; then
    WATCH_PORTS="$(seq -s, 0 447)"
fi
python - "${WATCH_PORTS}" <<'PY'
import sys

try:
    ports = [int(value.strip()) for value in sys.argv[1].split(",") if value.strip()]
except ValueError as exc:
    raise SystemExit(f"invalid watch ports: {exc}")
if not ports or len(ports) != len(set(ports)) or any(not 0 <= port < 448 for port in ports):
    raise SystemExit("watch ports must be unique integers in [0, 447]")
PY

RUN_DIR="${ROOT}/experiments/continual_validation/${RUN_ID}/calibration"
MANIFEST="${RUN_DIR}/calibration_manifest.json"
FLOW_DIR="${ROOT}/simulation/mix/acc_validation"
MODEL_DIR="${RUN_DIR}/models"
OUTPUT_DIR="${RUN_DIR}/ns3_output"
RUNTIME_DIR="${RUN_DIR}/runtime"
mkdir -p "${RUN_DIR}"

manifest_json() {
    local candidates_blob
    candidates_blob="$(IFS=,; echo "${CANDIDATE_RECORDS[*]}")"
    python - "${RUN_ID}" "${SEED}" "${BUFFER_KB}" "${WATCH_PORTS}" \
        "${REWARD_PROFILE}" "${REWARD_QUEUE_LAMBDA}" \
        "${REWARD_ECN_LAMBDA}" "${REWARD_WEIGHTS}" \
        "${ACC_HIDDEN_DIMS}" "${SPREAD_PROFILE}" "${candidates_blob}" <<'PY'
import json
import sys

candidates = []
for item in sys.argv[11].split(","):
    name, spread = item.split(":", 1)
    candidates.append({"name": name, "spread_fraction": float(spread)})
record = {
    "run_id": sys.argv[1],
    "seed": int(sys.argv[2]),
    "buffer_kb": int(sys.argv[3]),
    "watch_ports": [
        int(value) for value in sys.argv[4].split(",") if value.strip()
    ],
    "reward_profile": sys.argv[5],
    "reward_queue_lambda": float(sys.argv[6]),
    "reward_ecn_lambda": float(sys.argv[7]),
    "reward_weights": [float(value) for value in sys.argv[8].split(",")],
    "hidden_dims": sys.argv[9],
    "spread_profile": sys.argv[10],
    "port_scope": "switch-switch",
    "action_space": "multiscale",
    "candidates": candidates,
    "actions": ["low_strong", "mid", "high_gentle"],
}
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

check_manifest() {
    [[ -s "${MANIFEST}" ]] || {
        echo "Missing ${MANIFEST}; run --stage prepare first." >&2
        return 1
    }
    local expected
    expected="$(manifest_json)"
    EXPECTED_MANIFEST="${expected}" python - "${MANIFEST}" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    actual = json.load(handle)
expected = json.loads(os.environ["EXPECTED_MANIFEST"])
if actual != expected:
    raise SystemExit(
        "calibration manifest differs from requested arguments; use the "
        "original arguments or a new run-id"
    )
PY
}

runtime_config_path() {
    echo "${RUNTIME_DIR}/$1_seed${SEED}.conf"
}

output_base() {
    echo "${OUTPUT_DIR}/$1"
}

make_runtime_config() {
    local scenario="$1"
    local source="${RUN_DIR}/tasks/${scenario}/input.conf"
    local flow="${RUN_DIR}/tasks/${scenario}/input.flow"
    local destination
    destination="$(runtime_config_path "${scenario}")"
    local base
    base="$(output_base "${scenario}")"
    awk -v flow="${flow}" -v base="${base}" '
        $1 == "FLOW_FILE" { print "FLOW_FILE " flow; next }
        $1 == "TRACE_OUTPUT_FILE" { print "TRACE_OUTPUT_FILE " base ".tr"; next }
        $1 == "FCT_OUTPUT_FILE" { print "FCT_OUTPUT_FILE " base ".fct"; next }
        $1 == "PFC_OUTPUT_FILE" { print "PFC_OUTPUT_FILE " base ".pfc"; next }
        $1 == "QLEN_MONITOR_FILE" { print "QLEN_MONITOR_FILE " base ".queue"; next }
        $1 == "RATE_MONITOR_FILE" { print "RATE_MONITOR_FILE " base ".rate"; next }
        $1 == "THROUGHPUT_OUTPUT_FILE" { print "THROUGHPUT_OUTPUT_FILE " base ".throughput"; next }
        $1 == "QLEN_MON_FILE" { print "QLEN_MON_FILE " base ".qlen"; next }
        { print }
    ' "${source}" > "${destination}.tmp"
    mv "${destination}.tmp" "${destination}"
}

prepare() {
    if [[ -s "${MANIFEST}" ]]; then
        [[ "${RESUME}" -eq 1 ]] || {
            echo "Calibration run already exists; pass --resume or use a new run-id." >&2
            return 1
        }
        check_manifest
        echo "Calibration inputs already exist; keeping frozen files."
        return 0
    fi
    local scenario_list
    scenario_list="${CANDIDATES[*]}"
    bash "${ROOT}/scripts/acc_validation/prepare_scenarios.sh" \
        --seeds "${SEED}" \
        --scenarios "${scenario_list}" \
        --buffer-kb "${BUFFER_KB}" \
        --kmin-range "5000,50000" \
        --kmax-range "15000,100000" \
        --max-flows 0 \
        --baseline-stop-time 4.00

    mkdir -p "${RUN_DIR}/tasks" "${RUNTIME_DIR}" "${OUTPUT_DIR}"
    local scenario name destination reference
    reference="${CANDIDATES[0]}"
    for scenario in "${CANDIDATES[@]}"; do
        name="${scenario}_seed${SEED}"
        destination="${RUN_DIR}/tasks/${scenario}"
        mkdir -p "${destination}"
        cp "${FLOW_DIR}/${name}.flow" "${destination}/input.flow"
        cp "${FLOW_DIR}/${name}.conf" "${destination}/input.conf"
        cp "${FLOW_DIR}/${name}.meta" "${destination}/input.meta"
        make_runtime_config "${scenario}"
        if [[ "${scenario}" != "${reference}" ]]; then
            python "${ROOT}/scripts/continual_validation/verify_task_pair.py" \
                --task-a "${RUN_DIR}/tasks/${reference}/input.flow" \
                --task-b "${destination}/input.flow" \
                --json-output "${RUN_DIR}/identity_${scenario}.json" \
                --require-identical-flows \
                --require-timing-shift > /dev/null
        fi
    done
    manifest_json > "${MANIFEST}.tmp"
    mv "${MANIFEST}.tmp" "${MANIFEST}"
    echo "Spread calibration prepared under ${RUN_DIR}"
}

copy_outputs() {
    local scenario="$1" label="$2" exp="$3"
    local destination="${RUN_DIR}/runs/${scenario}/${label}"
    local base
    base="$(output_base "${scenario}")"
    local metrics="${MODEL_DIR}/${exp}_metrics.jsonl"
    mkdir -p "${destination}"
    shopt -s nullglob
    local files=("${base}".*)
    [[ ${#files[@]} -gt 0 ]] || {
        echo "No ns-3 output for ${scenario}/${label}" >&2
        return 1
    }
    cp "${files[@]}" "${destination}/"
    shopt -u nullglob
    cp "${RUN_DIR}/tasks/${scenario}/input.flow" "${destination}/input.flow"
    cp "$(runtime_config_path "${scenario}")" "${destination}/input.conf"
    cp "${RUN_DIR}/tasks/${scenario}/input.meta" "${destination}/input.meta"
    [[ -s "${metrics}" ]] || {
        echo "Agent metrics missing after ${scenario}/${label}" >&2
        return 1
    }
    tail -n 1 "${metrics}" > "${destination}/metrics.json"
}

run_grid() {
    check_manifest
    mkdir -p "${MODEL_DIR}" "${OUTPUT_DIR}" "${RUNTIME_DIR}"
    local exp="spread_calibration_${RUN_ID}_s${SEED}"
    local scenario specification label action base destination
    for scenario in "${CANDIDATES[@]}"; do
        make_runtime_config "${scenario}"
        for specification in "${SPECIFICATIONS[@]}"; do
            label="${specification%%:*}"
            action="${specification#*:}"
            destination="${RUN_DIR}/runs/${scenario}/${label}/metrics.json"
            if [[ -s "${destination}" ]]; then
                echo "[calibration] reuse scenario=${scenario} action=${label}"
                continue
            fi
            base="$(output_base "${scenario}")"
            rm -f "${base}".*
            echo "[calibration] scenario=${scenario} action=${label}(${action})"
            bash "${ROOT}/run_training.sh" \
                --one-shot --eval-greedy \
                --force-action "${action}" \
                --action-space multiscale \
                --eval-tag "calibration_${scenario}_${label}" \
                --config "$(runtime_config_path "${scenario}")" \
                --exp "${exp}" --mode ACC --port "${PORT}" --seed "${SEED}" \
                --buffer "${BUFFER_KB}" --model-dir "${MODEL_DIR}" --episodes 1 \
                --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
                --reward-weights "${REWARD_WEIGHTS}" \
                --reward-profile "${REWARD_PROFILE}" \
                --reward-queue-lambda "${REWARD_QUEUE_LAMBDA}" \
                --reward-ecn-lambda "${REWARD_ECN_LAMBDA}" \
                --shared-replay false \
                --watch-ports "${WATCH_PORTS}" \
                --tb-enable false --run-id "${RUN_ID}" --phase calibration
            copy_outputs "${scenario}" "${label}" "${exp}"
        done
    done
    touch "${RUN_DIR}/grid_complete"
    echo "Calibration grid complete: ${RUN_DIR}"
}

analyze() {
    check_manifest
    [[ -f "${RUN_DIR}/grid_complete" ]] || {
        echo "Calibration grid is incomplete; run --stage run --resume." >&2
        return 1
    }
    python "${ROOT}/scripts/continual_validation/analyze_spread_calibration.py" \
        --run-dir "${RUN_DIR}" \
        --min-completion 0.90 \
        --completion-tolerance 0.01 \
        --min-reward-spread 0.02 \
        --min-p95-spread 0.05 \
        --min-port-active-samples 50 \
        --min-port-congested-samples 20 \
        --port-scope switch-switch \
        --min-pair-p95-penalty 0.03
}

[[ "${STAGE}" == "prepare" || "${STAGE}" == "all" ]] && prepare
[[ "${STAGE}" == "run" || "${STAGE}" == "all" ]] && run_grid
[[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]] && analyze

echo "Spread calibration stage '${STAGE}' complete: ${RUN_DIR}"
