#!/usr/bin/env bash
# Frozen checkpoint intervention for localizing ACC forgetting to network paths.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE_RUN_ID="tailsafe_mixed_incast_localonly_s1"
STAGE="all"
PORT=5956
WATCH_PORTS="190,191,200,201,323,371,379,380,403,404,422,423"
CORE_PORTS="323,371,422,423"
RACK_PORTS="190,191,200,201,379,380,403,404"
VARIANTS="after_a after_b restore_323 restore_371 restore_core restore_rack random_core random_rack"
FORCE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/continual_validation/run_path_intervention.sh [options]

Options:
  --base-run-id ID      Completed ACC continual run (default: local-only run)
  --stage NAME          prepare, eval, analyze, or all
  --variants "NAMES"    Space-separated subset
  --watch-ports CSV     Ports included in detailed metrics and JSONL traces
  --core-ports CSV      Core-path restore set
  --rack-ports CSV      Rack-path restore set
  --port N              ns3-gym socket port
  --force               Rebuild model variants and overwrite evaluations

Built-in variants: after_a, after_b, restore_323, restore_371, restore_core,
restore_rack, random_core, random_rack. Random controls exclude all watched ports.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-run-id) BASE_RUN_ID="$2"; shift 2 ;;
        --stage) STAGE="$2"; shift 2 ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --watch-ports) WATCH_PORTS="$2"; shift 2 ;;
        --core-ports) CORE_PORTS="$2"; shift 2 ;;
        --rack-ports) RACK_PORTS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "${STAGE}" in prepare|eval|analyze|all) ;; *) echo "invalid stage" >&2; exit 2;; esac

BASE_DIR="${ROOT}/experiments/continual_validation/${BASE_RUN_ID}"
MANIFEST="${BASE_DIR}/manifest.json"
OUT="${BASE_DIR}/path_intervention"
AFTER_A="${BASE_DIR}/acc/checkpoints/after_a"
AFTER_B="${BASE_DIR}/acc/checkpoints/after_b"
[[ -s "${MANIFEST}" && -d "${AFTER_A}" && -d "${AFTER_B}" ]] || {
    echo "Completed ACC checkpoints not found under ${BASE_DIR}" >&2
    exit 1
}

json_value() {
    python - "$MANIFEST" "$1" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    value = json.load(handle)[sys.argv[2]]
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
EXP="continual_${BASE_RUN_ID}_acc_s${SEED}"

prepare_variant() {
    local variant="$1" destination="${OUT}/models/${variant}"
    [[ "${FORCE}" -eq 0 && -d "${destination}" ]] && return 0
    local force_args=()
    [[ "${FORCE}" -eq 1 ]] && force_args+=(--force)
    case "${variant}" in
        after_a)
            rm -rf "${destination}"
            mkdir -p "${destination}"
            cp -a "${AFTER_A}/." "${destination}/"
            ;;
        after_b)
            rm -rf "${destination}"
            mkdir -p "${destination}"
            cp -a "${AFTER_B}/." "${destination}/"
            ;;
        restore_323|restore_371)
            python "${ROOT}/scripts/continual_validation/build_port_restore.py" \
                --after-a "${AFTER_A}" --after-b "${AFTER_B}" \
                --output-dir "${destination}" --ports "${variant#restore_}" \
                "${force_args[@]}"
            ;;
        restore_core|restore_rack)
            local set="${CORE_PORTS}"
            [[ "${variant}" == "restore_rack" ]] && set="${RACK_PORTS}"
            python "${ROOT}/scripts/continual_validation/build_port_restore.py" \
                --after-a "${AFTER_A}" --after-b "${AFTER_B}" \
                --output-dir "${destination}" --ports "${set}" \
                "${force_args[@]}"
            ;;
        random_core|random_rack)
            local count=4
            [[ "${variant}" == "random_rack" ]] && count=8
            python "${ROOT}/scripts/continual_validation/build_port_restore.py" \
                --after-a "${AFTER_A}" --after-b "${AFTER_B}" \
                --output-dir "${destination}" --random-count "${count}" \
                --exclude-ports "${WATCH_PORTS}" --seed "${SEED}" \
                "${force_args[@]}"
            ;;
        *) echo "Unknown variant: ${variant}" >&2; exit 2 ;;
    esac
}

runtime_config() {
    echo "${BASE_DIR}/runtime/$1_seed${SEED}.conf"
}

output_base() {
    awk '$1 == "FCT_OUTPUT_FILE" {sub(/\.fct$/, "", $2); print $2; exit}' \
        "$(runtime_config "$1")"
}

evaluate_variant() {
    local variant="$1" task="$2"
    local destination="${OUT}/eval/${variant}/${task}"
    if [[ "${FORCE}" -eq 0 && -s "${destination}/metrics.json" ]]; then
        echo "[skip] ${variant}/${task} already measured"
        return 0
    fi
    local model_dir="${OUT}/models/${variant}"
    local config base metrics trace
    config="$(runtime_config "${task}")"
    base="$(output_base "${task}")"
    metrics="${model_dir}/${EXP}_metrics.jsonl"
    trace="${destination}/watch_trace.jsonl"
    mkdir -p "${destination}"
    rm -f "${base}".* "${metrics}" "${trace}"
    echo "[eval] variant=${variant} task=${task}"
    bash "${ROOT}/run_training.sh" \
        --one-shot --eval-greedy --eval-tag "path_${variant}_${task}" \
        --config "${config}" --exp "${EXP}" --mode ACC --port "${PORT}" \
        --seed "${SEED}" --buffer "${BUFFER_KB}" --model-dir "${model_dir}" \
        --episodes 1 --eps-decay "${EPS_DECAY}" \
        --acc-hidden-dims "${HIDDEN_DIMS}" --reward-weights "0.50,0.30,0.20" \
        --reward-profile "${REWARD_PROFILE}" \
        --reward-queue-lambda "${QUEUE_LAMBDA}" \
        --reward-ecn-lambda "${ECN_LAMBDA}" --shared-replay false \
        --watch-ports "${WATCH_PORTS}" --watch-trace-file "${trace}" \
        --tb-enable false --run-id "${BASE_RUN_ID}" --phase path_intervention
    [[ -s "${metrics}" ]] || { echo "metrics missing: ${metrics}" >&2; exit 1; }
    tail -n 1 "${metrics}" > "${destination}/metrics.json"
    cp "${BASE_DIR}/tasks/${task}/input.flow" "${destination}/input.flow"
    cp "${config}" "${destination}/input.conf"
    shopt -s nullglob
    local files=("${base}".*)
    [[ ${#files[@]} -gt 0 ]] || { echo "ns-3 outputs missing" >&2; exit 1; }
    cp "${files[@]}" "${destination}/"
    shopt -u nullglob
}

run_prepare() {
    mkdir -p "${OUT}/models" "${OUT}/eval"
    local variant
    for variant in ${VARIANTS}; do prepare_variant "${variant}"; done
    python - "${OUT}" "${BASE_RUN_ID}" "${VARIANTS}" "${WATCH_PORTS}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
record = {
    "base_run_id": sys.argv[2],
    "variants": sys.argv[3].split(),
    "watch_ports": [int(x) for x in sys.argv[4].split(",") if x],
}
(out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
PY
}

run_eval() {
    local variant task
    for variant in ${VARIANTS}; do
        [[ -d "${OUT}/models/${variant}" ]] || prepare_variant "${variant}"
        for task in "${TASK_A}" "${TASK_B}"; do
            evaluate_variant "${variant}" "${task}"
        done
    done
}

run_analyze() {
    python "${ROOT}/scripts/continual_validation/analyze_path_intervention.py" \
        --run-dir "${OUT}" --task-a "${TASK_A}" --task-b "${TASK_B}"
}

[[ "${STAGE}" == prepare || "${STAGE}" == all ]] && run_prepare
[[ "${STAGE}" == eval || "${STAGE}" == all ]] && run_eval
[[ "${STAGE}" == analyze || "${STAGE}" == all ]] && run_analyze
echo "Path intervention stage '${STAGE}' complete: ${OUT}"
