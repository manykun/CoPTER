#!/usr/bin/env bash
# Causal one-port interpolation with a frozen after-B ACC background policy.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="all"
BASE_RUN_ID="acc_local_b5_s1"
CALIBRATION_RUN_ID=""
ENDPOINT_SOURCE="screen"
ENDPOINT_SOURCE_EXPLICIT=0
ENDPOINT_PORT=323
TARGET_PORTS="323,321"
ALPHA_STEP="0.1"
ALPHA_MIN="0.0"
ALPHA_MAX="1.0"
LINK_GBPS="40"
SOCKET_PORT=6356
FORCE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/continual_validation/run_port_path_sweep.sh [options]

Stages: prepare, sweep, analyze, all

Options:
  --base-run-id ID          Completed ACC A->B5 continual run
  --endpoint-source NAME    screen (default) or calibration
  --calibration-run-id ID   Required only for calibration endpoint source
  --endpoint-port N         Port whose A/B optima define the path (default 323)
  --target-ports CSV        Ports receiving the same intervention path
  --alpha-step X            Initial path resolution, e.g. 0.1 or 0.02
  --alpha-min X             Optional refinement interval lower bound
  --alpha-max X             Optional refinement interval upper bound
  --link-gbps X
  --port N                  ns3-gym socket port
  --force                   Re-evaluate existing points
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --base-run-id) BASE_RUN_ID="$2"; shift 2 ;;
        --calibration-run-id)
            CALIBRATION_RUN_ID="$2"
            [[ "${ENDPOINT_SOURCE_EXPLICIT}" -eq 1 ]] || ENDPOINT_SOURCE="calibration"
            shift 2
            ;;
        --endpoint-source)
            ENDPOINT_SOURCE="$2"
            ENDPOINT_SOURCE_EXPLICIT=1
            shift 2
            ;;
        --endpoint-port) ENDPOINT_PORT="$2"; shift 2 ;;
        --target-ports) TARGET_PORTS="$2"; shift 2 ;;
        --alpha-step) ALPHA_STEP="$2"; shift 2 ;;
        --alpha-min) ALPHA_MIN="$2"; shift 2 ;;
        --alpha-max) ALPHA_MAX="$2"; shift 2 ;;
        --link-gbps) LINK_GBPS="$2"; shift 2 ;;
        --port) SOCKET_PORT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "${STAGE}" in prepare|sweep|analyze|all) ;;
    *) echo "invalid stage: ${STAGE}" >&2; exit 2 ;;
esac
case "${ENDPOINT_SOURCE}" in screen|calibration) ;;
    *) echo "endpoint-source must be screen or calibration" >&2; exit 2 ;;
esac

BASE_DIR="${ROOT}/experiments/continual_validation/${BASE_RUN_ID}"
CAL_DIR="${ROOT}/experiments/continual_validation/${CALIBRATION_RUN_ID}/calibration"
MANIFEST="${BASE_DIR}/manifest.json"
AFTER_B="${BASE_DIR}/acc/checkpoints/after_b"
OUT="${BASE_DIR}/port_path_sweep_p${ENDPOINT_PORT}"
PATH_MANIFEST="${OUT}/path_manifest.json"
MODEL_DIR="${OUT}/model_after_b"

[[ -s "${MANIFEST}" && -d "${AFTER_B}" ]] || {
    echo "Completed after-B checkpoint not found under ${BASE_DIR}" >&2
    exit 1
}
if [[ "${ENDPOINT_SOURCE}" == "calibration" ]]; then
    [[ -n "${CALIBRATION_RUN_ID}" && -s "${CAL_DIR}/port_action_summary.csv" ]] || {
        echo "Calibration endpoint source requires --calibration-run-id with port_action_summary.csv" >&2
        exit 1
    }
else
    [[ -f "${BASE_DIR}/screen/markers/complete" ]] || {
        echo "Screen endpoint source requires a completed fixed-action screen" >&2
        exit 1
    }
fi

json_value() {
    python - "${MANIFEST}" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle).get(sys.argv[2])
if isinstance(value, (list, dict)):
    print(json.dumps(value))
elif value is not None:
    print(value)
PY
}

TASK_A="$(json_value task_a)"
TASK_B="$(json_value task_b)"
SEED="$(json_value seed)"
BUFFER_KB="$(json_value buffer_kb)"
EPS_DECAY="$(json_value epsilon_decay_steps)"
HIDDEN_DIMS="$(json_value hidden_dims)"
REWARD_PROFILE="$(json_value reward_profile)"
QUEUE_LAMBDA="$(json_value reward_queue_lambda)"
ECN_LAMBDA="$(json_value reward_ecn_lambda)"
ACTION_SPACE="$(json_value action_space)"
REWARD_WEIGHTS="$(json_value reward_weights)"
[[ -n "${ACTION_SPACE}" ]] || ACTION_SPACE="legacy"
[[ -n "${REWARD_WEIGHTS}" ]] || REWARD_WEIGHTS='[0.5, 0.3, 0.2]'
REWARD_WEIGHTS="$(python - "${REWARD_WEIGHTS}" <<'PY'
import json, sys
value = json.loads(sys.argv[1])
print(",".join(str(item) for item in value))
PY
)"
EXP="continual_${BASE_RUN_ID}_acc_s${SEED}"

ensure_binary() {
    local binary="${ROOT}/ns-3.33/build/scratch/copter-sim"
    [[ -x "${binary}" ]] || {
        echo "NS3 binary missing; run: bash build_ns3_copter.sh" >&2
        exit 1
    }
    [[ ! "${ROOT}/ns-3.33/scratch/copter-sim.cc" -nt "${binary}" ]] || {
        echo "NS3 binary is stale; run: bash build_ns3_copter.sh" >&2
        exit 1
    }
}

ensure_model() {
    if [[ "${FORCE}" -eq 1 && -d "${MODEL_DIR}" ]]; then
        rm -rf "${MODEL_DIR}"
    fi
    if [[ ! -d "${MODEL_DIR}" ]]; then
        mkdir -p "${MODEL_DIR}"
        cp -a "${AFTER_B}/." "${MODEL_DIR}/"
    fi
}

prepare_path() {
    local source_args=()
    if [[ "${ENDPOINT_SOURCE}" == "calibration" ]]; then
        source_args+=(--calibration-dir "${CAL_DIR}")
    else
        source_args+=(--screen-dir "${BASE_DIR}/screen")
    fi
    python "${ROOT}/scripts/continual_validation/prepare_port_path_sweep.py" \
        "${source_args[@]}" \
        --base-run-dir "${BASE_DIR}" \
        --output "${PATH_MANIFEST}" \
        --endpoint-port "${ENDPOINT_PORT}" \
        --target-ports "${TARGET_PORTS}" \
        --alpha-step "${ALPHA_STEP}" \
        --alpha-min "${ALPHA_MIN}" \
        --alpha-max "${ALPHA_MAX}" \
        --link-gbps "${LINK_GBPS}"
}

runtime_config() {
    local target="$1" label="$2" task="$3"
    local source="${BASE_DIR}/runtime/${task}_seed${SEED}.conf"
    local destination="${OUT}/runtime/p${target}/${label}/${task}.conf"
    local base="${OUT}/runtime_output/p${target}/${label}/${task}/result"
    mkdir -p "$(dirname "${destination}")" "$(dirname "${base}")"
    awk -v base="${base}" '
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
    echo "${destination}"
}

copy_outputs() {
    local target="$1" label="$2" task="$3" config="$4" metrics="$5"
    local destination="${OUT}/eval/p${target}/${label}/${task}"
    local base="${OUT}/runtime_output/p${target}/${label}/${task}/result"
    mkdir -p "${destination}"
    tail -n 1 "${metrics}" > "${destination}/metrics.json"
    cp "${BASE_DIR}/tasks/${task}/input.flow" "${destination}/input.flow"
    cp "${config}" "${destination}/input.conf"
    shopt -s nullglob
    local files=("${base}".*)
    [[ ${#files[@]} -gt 0 ]] || { echo "ns-3 output missing: ${base}.*" >&2; exit 1; }
    cp "${files[@]}" "${destination}/"
    shopt -u nullglob
}

evaluate() {
    local target="$1" label="$2" task="$3" kmin="$4" kmax="$5" pmax="$6"
    local destination="${OUT}/eval/p${target}/${label}/${task}"
    if [[ "${FORCE}" -eq 0 && -s "${destination}/metrics.json" ]]; then
        echo "[skip] port=${target} point=${label} task=${task}"
        return
    fi
    local config metrics trace
    config="$(runtime_config "${target}" "${label}" "${task}")"
    metrics="${MODEL_DIR}/${EXP}_metrics.jsonl"
    trace="${destination}/watch_trace.jsonl"
    mkdir -p "${destination}"
    rm -f "${metrics}" "${trace}" \
        "${OUT}/runtime_output/p${target}/${label}/${task}/result".*
    echo "[eval] port=${target} point=${label} task=${task}"
    bash "${ROOT}/run_training.sh" \
        --one-shot --eval-greedy \
        --eval-tag "port_path_p${target}_${label}_${task}" \
        --force-port-action "${target},${kmin},${kmax},${pmax}" \
        --config "${config}" --exp "${EXP}" --mode ACC \
        --port "${SOCKET_PORT}" --seed "${SEED}" --buffer "${BUFFER_KB}" \
        --model-dir "${MODEL_DIR}" --episodes 1 --eps-decay "${EPS_DECAY}" \
        --acc-hidden-dims "${HIDDEN_DIMS}" --action-space "${ACTION_SPACE}" \
        --reward-weights "${REWARD_WEIGHTS}" \
        --reward-profile "${REWARD_PROFILE}" \
        --reward-queue-lambda "${QUEUE_LAMBDA}" \
        --reward-ecn-lambda "${ECN_LAMBDA}" --shared-replay false \
        --watch-ports "${TARGET_PORTS}" --watch-trace-file "${trace}" \
        --tb-enable false --run-id "${BASE_RUN_ID}" --phase port_path
    [[ -s "${metrics}" ]] || { echo "metrics missing: ${metrics}" >&2; exit 1; }
    copy_outputs "${target}" "${label}" "${task}" "${config}" "${metrics}"
}

run_sweep() {
    ensure_binary
    ensure_model
    [[ -s "${PATH_MANIFEST}" ]] || prepare_path
    while IFS=$'\t' read -r target label kmin kmax pmax; do
        evaluate "${target}" "${label}" "${TASK_A}" "${kmin}" "${kmax}" "${pmax}"
        evaluate "${target}" "${label}" "${TASK_B}" "${kmin}" "${kmax}" "${pmax}"
    done < <(python - "${PATH_MANIFEST}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
for port in manifest["target_ports"]:
    for point in manifest["points"]:
        print(port, point["label"], point["kmin_norm"], point["kmax_norm"], point["pmax"], sep="\t")
PY
    )
}

analyze() {
    python "${ROOT}/scripts/continual_validation/analyze_port_path_sweep.py" \
        --run-dir "${OUT}"
}

[[ "${STAGE}" == prepare || "${STAGE}" == all ]] && prepare_path
[[ "${STAGE}" == sweep || "${STAGE}" == all ]] && run_sweep
[[ "${STAGE}" == analyze || "${STAGE}" == all ]] && analyze
echo "Port path stage '${STAGE}' complete: ${OUT}"
