#!/usr/bin/env bash
# Frozen one-port continuous midpoint sweep around an after-B ACC policy.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_RUN_ID="tailsafe_mixed_incast_localonly_s1"
STAGE="all"
TARGET_PORT=323
LINK_GBPS=40
SOCKET_PORT=6056
INCLUDE_GRID_NEIGHBORS=0
FORCE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/continual_validation/run_continuous_sweep.sh [options]

Stages: baseline, prepare, sweep, analyze, all

Options:
  --base-run-id ID
  --stage NAME
  --target-port N
  --link-gbps X
  --port N                    ns3-gym socket port
  --include-grid-neighbors    13-point endpoint+midpoint sweep (default: ~7)
  --force                     Rebuild/re-evaluate completed outputs
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-run-id) BASE_RUN_ID="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --target-port) TARGET_PORT="$2"; shift 2 ;;
        --link-gbps) LINK_GBPS="$2"; shift 2 ;;
        --port) SOCKET_PORT="$2"; shift 2 ;;
        --include-grid-neighbors) INCLUDE_GRID_NEIGHBORS=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "${STAGE}" in baseline|prepare|sweep|analyze|all) ;;
    *) echo "invalid stage: ${STAGE}" >&2; exit 2 ;;
esac

BASE_DIR="${ROOT}/experiments/continual_validation/${BASE_RUN_ID}"
MANIFEST="${BASE_DIR}/manifest.json"
AFTER_B="${BASE_DIR}/acc/checkpoints/after_b"
OUT="${BASE_DIR}/continuous_sweep_port${TARGET_PORT}"
MODEL_DIR="${OUT}/model_after_b"
SWEEP_MANIFEST="${OUT}/sweep_manifest.json"
[[ -s "${MANIFEST}" && -d "${AFTER_B}" ]] || {
    echo "Completed after-B ACC checkpoint not found under ${BASE_DIR}" >&2
    exit 1
}

json_value() {
    python - "${MANIFEST}" "$1" <<'PY'
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

runtime_config() {
    echo "${BASE_DIR}/runtime/$1_seed${SEED}.conf"
}

output_base() {
    awk '$1 == "FCT_OUTPUT_FILE" {sub(/\.fct$/, "", $2); print $2; exit}' \
        "$(runtime_config "$1")"
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

copy_outputs() {
    local task="$1" destination="$2" metrics="$3" base
    base="$(output_base "${task}")"
    mkdir -p "${destination}"
    tail -n 1 "${metrics}" > "${destination}/metrics.json"
    cp "${BASE_DIR}/tasks/${task}/input.flow" "${destination}/input.flow"
    cp "$(runtime_config "${task}")" "${destination}/input.conf"
    shopt -s nullglob
    local files=("${base}".*)
    [[ ${#files[@]} -gt 0 ]] || { echo "ns-3 outputs missing" >&2; exit 1; }
    cp "${files[@]}" "${destination}/"
    shopt -u nullglob
}

evaluate() {
    local label="$1" task="$2" force_value="${3:-}"
    local destination="${OUT}/eval/${label}/${task}"
    if [[ "${FORCE}" -eq 0 && -s "${destination}/metrics.json" ]]; then
        echo "[skip] ${label}/${task} already measured"
        return 0
    fi
    local base metrics trace
    base="$(output_base "${task}")"
    metrics="${MODEL_DIR}/${EXP}_metrics.jsonl"
    trace="${destination}/watch_trace.jsonl"
    mkdir -p "${destination}"
    rm -f "${base}".* "${metrics}" "${trace}"
    local force_args=()
    [[ -z "${force_value}" ]] || force_args+=(--force-port-action "${force_value}")
    echo "[eval] point=${label} task=${task} action=${force_value:-greedy}"
    bash "${ROOT}/run_training.sh" \
        --one-shot --eval-greedy --eval-tag "continuous_${label}_${task}" \
        --config "$(runtime_config "${task}")" --exp "${EXP}" --mode ACC \
        --port "${SOCKET_PORT}" --seed "${SEED}" --buffer "${BUFFER_KB}" \
        --model-dir "${MODEL_DIR}" --episodes 1 --eps-decay "${EPS_DECAY}" \
        --acc-hidden-dims "${HIDDEN_DIMS}" --reward-weights "0.50,0.30,0.20" \
        --reward-profile "${REWARD_PROFILE}" \
        --reward-queue-lambda "${QUEUE_LAMBDA}" \
        --reward-ecn-lambda "${ECN_LAMBDA}" --shared-replay false \
        --watch-ports "${TARGET_PORT}" --watch-trace-file "${trace}" \
        --tb-enable false --run-id "${BASE_RUN_ID}" --phase continuous_sweep \
        "${force_args[@]}"
    [[ -s "${metrics}" ]] || { echo "metrics missing: ${metrics}" >&2; exit 1; }
    copy_outputs "${task}" "${destination}" "${metrics}"
}

run_baseline() {
    ensure_model
    evaluate greedy "${TASK_A}"
    evaluate greedy "${TASK_B}"
}

run_prepare() {
    [[ -s "${OUT}/eval/greedy/${TASK_A}/metrics.json" ]] || {
        echo "Run --stage baseline first." >&2
        exit 1
    }
    local neighbor_args=()
    [[ "${INCLUDE_GRID_NEIGHBORS}" -eq 1 ]] && neighbor_args+=(--include-grid-neighbors)
    python "${ROOT}/scripts/continual_validation/prepare_continuous_sweep.py" \
        --baseline-metrics "${OUT}/eval/greedy/${TASK_A}/metrics.json" \
        --config "$(runtime_config "${TASK_A}")" --output "${SWEEP_MANIFEST}" \
        --port "${TARGET_PORT}" --link-gbps "${LINK_GBPS}" \
        "${neighbor_args[@]}"
}

run_sweep() {
    ensure_model
    [[ -s "${SWEEP_MANIFEST}" ]] || { echo "Run --stage prepare first." >&2; exit 1; }
    while IFS=$'\t' read -r label kmin kmax pmax; do
        local action="${TARGET_PORT},${kmin},${kmax},${pmax}"
        evaluate "${label}" "${TASK_A}" "${action}"
        evaluate "${label}" "${TASK_B}" "${action}"
    done < <(python - "${SWEEP_MANIFEST}" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    points = json.load(handle)["points"]
for point in points:
    print(point["label"], point["kmin_norm"], point["kmax_norm"], point["pmax"], sep="\t")
PY
    )
}

run_analyze() {
    python "${ROOT}/scripts/continual_validation/analyze_continuous_sweep.py" \
        --run-dir "${OUT}" --task-a "${TASK_A}" --task-b "${TASK_B}"
}

[[ "${STAGE}" == baseline || "${STAGE}" == all ]] && run_baseline
[[ "${STAGE}" == prepare || "${STAGE}" == all ]] && run_prepare
[[ "${STAGE}" == sweep || "${STAGE}" == all ]] && run_sweep
[[ "${STAGE}" == analyze || "${STAGE}" == all ]] && run_analyze
echo "Continuous sweep stage '${STAGE}' complete: ${OUT}"
