#!/usr/bin/env bash
# Frozen one-port physical Kmin/Kmax/Pmax range experiment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_RUN_ID="tailsafe_mixed_incast_localonly_s1"
STAGE="all"
TARGET_PORT=323
POINT_SET="wide"
CONTROL_ACTION=""
SOCKET_PORT=6156
TOP_K=3
COMPLETION_TOLERANCE=0.01
EXPERIMENT_ID=""
FORCE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/continual_validation/run_physical_range_sweep.sh [options]

Stages: prepare, screen, safety, analyze, all

Options:
  --base-run-id ID
  --stage NAME
  --target-port N
  --point-set wide|control
  --control-action KMIN_KB,KMAX_KB,PMAX
  --experiment-id ID          Output subdirectory name
  --port N                    ns3-gym socket port
  --top-k N                   Old-task candidates checked on new task
  --completion-tolerance X
  --force                     Re-evaluate completed outputs

The wide set screens 14 physical points on task A, then evaluates the center
and top-k completion-safe candidates on task B.  The control set compares the
center with one supplied action and is intended for a replication port or a
non-congested negative-control port.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-run-id) BASE_RUN_ID="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --target-port) TARGET_PORT="$2"; shift 2 ;;
        --point-set) POINT_SET="$2"; shift 2 ;;
        --control-action) CONTROL_ACTION="$2"; shift 2 ;;
        --experiment-id) EXPERIMENT_ID="$2"; shift 2 ;;
        --port) SOCKET_PORT="$2"; shift 2 ;;
        --top-k) TOP_K="$2"; shift 2 ;;
        --completion-tolerance) COMPLETION_TOLERANCE="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "${STAGE}" in prepare|screen|safety|analyze|all) ;;
    *) echo "invalid stage: ${STAGE}" >&2; exit 2 ;;
esac
case "${POINT_SET}" in wide|control) ;;
    *) echo "invalid point set: ${POINT_SET}" >&2; exit 2 ;;
esac
if [[ "${POINT_SET}" == control && -z "${CONTROL_ACTION}" ]]; then
    echo "--control-action is required with --point-set control" >&2
    exit 2
fi

BASE_DIR="${ROOT}/experiments/continual_validation/${BASE_RUN_ID}"
BASE_MANIFEST="${BASE_DIR}/manifest.json"
AFTER_B="${BASE_DIR}/acc/checkpoints/after_b"
[[ -s "${BASE_MANIFEST}" && -d "${AFTER_B}" ]] || {
    echo "Completed after-B ACC checkpoint not found under ${BASE_DIR}" >&2
    exit 1
}

json_value() {
    python - "${BASE_MANIFEST}" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)[sys.argv[2]])
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
EXP="continual_${BASE_RUN_ID}_acc_s${SEED}"

if [[ -z "${EXPERIMENT_ID}" ]]; then
    EXPERIMENT_ID="physical_range_port${TARGET_PORT}_${POINT_SET}"
fi
OUT="${BASE_DIR}/${EXPERIMENT_ID}"
RANGE_MANIFEST="${OUT}/range_manifest.json"
MODEL_DIR="${OUT}/model_after_b"

ensure_binary() {
    local binary="${ROOT}/ns-3.33/build/scratch/copter-sim"
    [[ -x "${binary}" ]] || {
        echo "NS3 binary missing; run: bash build_ns3_copter.sh" >&2
        exit 1
    }
    if [[ "${ROOT}/ns-3.33/scratch/copter-sim.cc" -nt "${binary}" ]]; then
        echo "NS3 binary is older than copter-sim.cc; run: bash build_ns3_copter.sh" >&2
        exit 1
    fi
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

prepare_manifest() {
    mkdir -p "${OUT}"
    local args=(
        --output "${RANGE_MANIFEST}"
        --target-port "${TARGET_PORT}"
        --point-set "${POINT_SET}"
        --base-run-id "${BASE_RUN_ID}"
        --task-a "${TASK_A}"
        --task-b "${TASK_B}"
        --buffer-kb "${BUFFER_KB}"
    )
    [[ -z "${CONTROL_ACTION}" ]] || args+=(--control-action "${CONTROL_ACTION}")
    python "${ROOT}/scripts/continual_validation/prepare_physical_range_sweep.py" "${args[@]}"
}

base_runtime_config() {
    echo "${BASE_DIR}/runtime/$1_seed${SEED}.conf"
}

make_config() {
    local label="$1" task="$2" kmin="$3" kmax="$4" pmax="$5"
    local source destination flow base tmp
    source="$(base_runtime_config "${task}")"
    destination="${OUT}/runtime/${label}_${task}.conf"
    flow="${BASE_DIR}/tasks/${task}/input.flow"
    base="${OUT}/runtime_outputs/${label}/${task}/result"
    tmp="${destination}.tmp"
    mkdir -p "$(dirname "${destination}")" "$(dirname "${base}")"
    [[ -s "${source}" && -s "${flow}" ]] || {
        echo "Missing frozen runtime input for ${task}" >&2
        exit 1
    }
    awk -v flow="${flow}" -v base="${base}" \
        -v port="${TARGET_PORT}" -v kmin="${kmin}" -v kmax="${kmax}" -v pmax="${pmax}" '
        $1 == "OPENGYM_FORCE_PORT_INDEX" ||
        $1 == "OPENGYM_FORCE_KMIN_KB" ||
        $1 == "OPENGYM_FORCE_KMAX_KB" ||
        $1 == "OPENGYM_FORCE_PMAX" { next }
        $1 == "FLOW_FILE" { print "FLOW_FILE " flow; next }
        $1 == "TRACE_OUTPUT_FILE" { print "TRACE_OUTPUT_FILE " base ".tr"; next }
        $1 == "FCT_OUTPUT_FILE" { print "FCT_OUTPUT_FILE " base ".fct"; next }
        $1 == "PFC_OUTPUT_FILE" { print "PFC_OUTPUT_FILE " base ".pfc"; next }
        $1 == "QLEN_MONITOR_FILE" { print "QLEN_MONITOR_FILE " base ".queue"; next }
        $1 == "RATE_MONITOR_FILE" { print "RATE_MONITOR_FILE " base ".rate"; next }
        $1 == "THROUGHPUT_OUTPUT_FILE" { print "THROUGHPUT_OUTPUT_FILE " base ".throughput"; next }
        $1 == "QLEN_MON_FILE" { print "QLEN_MON_FILE " base ".qlen"; next }
        { print }
        END {
            print "OPENGYM_FORCE_PORT_INDEX " port
            print "OPENGYM_FORCE_KMIN_KB " kmin
            print "OPENGYM_FORCE_KMAX_KB " kmax
            print "OPENGYM_FORCE_PMAX " pmax
        }
    ' "${source}" > "${tmp}"
    mv "${tmp}" "${destination}"
    echo "${destination}"
}

copy_outputs() {
    local label="$1" task="$2" config="$3" metrics="$4"
    local destination base
    destination="${OUT}/eval/${label}/${task}"
    base="${OUT}/runtime_outputs/${label}/${task}/result"
    mkdir -p "${destination}"
    tail -n 1 "${metrics}" > "${destination}/metrics.json"
    cp "${BASE_DIR}/tasks/${task}/input.flow" "${destination}/input.flow"
    cp "${config}" "${destination}/input.conf"
    shopt -s nullglob
    local files=("${base}".*)
    [[ ${#files[@]} -gt 0 ]] || { echo "ns-3 outputs missing: ${base}.*" >&2; exit 1; }
    cp "${files[@]}" "${destination}/"
    shopt -u nullglob
}

evaluate() {
    local label="$1" task="$2" kmin="$3" kmax="$4" pmax="$5"
    local destination config metrics trace
    destination="${OUT}/eval/${label}/${task}"
    if [[ "${FORCE}" -eq 0 && -s "${destination}/metrics.json" ]]; then
        echo "[skip] ${label}/${task} already measured"
        return 0
    fi
    config="$(make_config "${label}" "${task}" "${kmin}" "${kmax}" "${pmax}")"
    metrics="${MODEL_DIR}/${EXP}_metrics.jsonl"
    trace="${destination}/watch_trace.jsonl"
    mkdir -p "${destination}"
    rm -f "${OUT}/runtime_outputs/${label}/${task}/result".* "${metrics}" "${trace}"
    echo "[eval] port=${TARGET_PORT} point=${label} task=${task} physical=(${kmin},${kmax},${pmax})"
    bash "${ROOT}/run_training.sh" \
        --one-shot --eval-greedy --eval-tag "range_${TARGET_PORT}_${label}_${task}" \
        --config "${config}" --exp "${EXP}" --mode ACC --port "${SOCKET_PORT}" \
        --seed "${SEED}" --buffer "${BUFFER_KB}" --model-dir "${MODEL_DIR}" \
        --episodes 1 --eps-decay "${EPS_DECAY}" \
        --acc-hidden-dims "${HIDDEN_DIMS}" --reward-weights "0.50,0.30,0.20" \
        --reward-profile "${REWARD_PROFILE}" \
        --reward-queue-lambda "${QUEUE_LAMBDA}" \
        --reward-ecn-lambda "${ECN_LAMBDA}" --shared-replay false \
        --watch-ports "${TARGET_PORT}" --watch-trace-file "${trace}" \
        --tb-enable false --run-id "${BASE_RUN_ID}" --phase physical_range
    [[ -s "${metrics}" ]] || { echo "metrics missing: ${metrics}" >&2; exit 1; }
    copy_outputs "${label}" "${task}" "${config}" "${metrics}"
}

point_rows() {
    python - "${RANGE_MANIFEST}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    points = json.load(handle)["points"]
for point in points:
    print(point["label"], point["kmin_kb"], point["kmax_kb"], point["pmax"], sep="\t")
PY
}

run_screen() {
    ensure_binary
    ensure_model
    [[ -s "${RANGE_MANIFEST}" ]] || prepare_manifest
    while IFS=$'\t' read -r label kmin kmax pmax; do
        evaluate "${label}" "${TASK_A}" "${kmin}" "${kmax}" "${pmax}"
    done < <(point_rows)
    python "${ROOT}/scripts/continual_validation/analyze_physical_range_sweep.py" \
        --run-dir "${OUT}" --stage screen --top-k "${TOP_K}" \
        --completion-tolerance "${COMPLETION_TOLERANCE}"
}

run_safety() {
    ensure_binary
    ensure_model
    [[ -s "${OUT}/top_candidates.json" ]] || {
        echo "Run --stage screen first." >&2
        exit 1
    }
    while IFS=$'\t' read -r label kmin kmax pmax; do
        evaluate "${label}" "${TASK_B}" "${kmin}" "${kmax}" "${pmax}"
    done < <(python - "${RANGE_MANIFEST}" "${OUT}/top_candidates.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    points = {p["label"]: p for p in json.load(handle)["points"]}
with open(sys.argv[2], encoding="utf-8") as handle:
    labels = json.load(handle)["labels"]
for label in labels:
    point = points[label]
    print(label, point["kmin_kb"], point["kmax_kb"], point["pmax"], sep="\t")
PY
    )
    python "${ROOT}/scripts/continual_validation/analyze_physical_range_sweep.py" \
        --run-dir "${OUT}" --stage safety --top-k "${TOP_K}" \
        --completion-tolerance "${COMPLETION_TOLERANCE}"
}

run_analyze() {
    local stage="screen"
    [[ -d "${OUT}/eval/center/${TASK_B}" ]] && stage="safety"
    python "${ROOT}/scripts/continual_validation/analyze_physical_range_sweep.py" \
        --run-dir "${OUT}" --stage "${stage}" --top-k "${TOP_K}" \
        --completion-tolerance "${COMPLETION_TOLERANCE}"
}

[[ "${STAGE}" == prepare || "${STAGE}" == all ]] && prepare_manifest
[[ "${STAGE}" == screen || "${STAGE}" == all ]] && run_screen
[[ "${STAGE}" == safety || "${STAGE}" == all ]] && run_safety
[[ "${STAGE}" == analyze ]] && run_analyze
echo "Physical range stage '${STAGE}' complete: ${OUT}"
