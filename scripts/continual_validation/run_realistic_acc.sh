#!/usr/bin/env bash
# ACC continual learning over trace-derived workload CDFs, with local replay
# only and a post-training causal action interpolation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="all"
PAIR="webserver-cachefollower"
RUN_ID=""
SEED=1
UPDATES=600
PHASE_EPOCHS=100
WATCH_PORTS="323,321,320,345,346,347"
ENDPOINT_PORT=323
TARGET_PORTS="323,321"
ALPHA_STEP="0.1"
PORT=6656
RESUME=0
SMOKE=0

usage() {
    cat <<'EOF'
Usage: bash scripts/continual_validation/run_realistic_acc.sh [options]

Stages: prepare, screen, acc, analyze, interpolate, all

Options:
  --pair NAME              webserver-cachefollower (default),
                           cachefollower-websearch, websearch-webserver
  --run-id ID
  --seed N
  --updates-per-task N
  --phase-epochs N
  --watch-ports CSV
  --endpoint-port N
  --target-ports CSV
  --alpha-step X
  --port N
  --resume
  --smoke

The ACC run always uses --shared-replay false. Interpolation endpoints are
derived from this run's own fixed-action screen; no task-pair search is used.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --pair) PAIR="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --updates-per-task) UPDATES="$2"; shift 2 ;;
        --phase-epochs) PHASE_EPOCHS="$2"; shift 2 ;;
        --watch-ports) WATCH_PORTS="$2"; shift 2 ;;
        --endpoint-port) ENDPOINT_PORT="$2"; shift 2 ;;
        --target-ports) TARGET_PORTS="$2"; shift 2 ;;
        --alpha-step) ALPHA_STEP="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${STAGE}" in prepare|screen|acc|analyze|interpolate|all) ;;
    *) echo "invalid stage: ${STAGE}" >&2; exit 2 ;;
esac
case "${PAIR}" in
    webserver-cachefollower)
        TASK_A="realistic_webserver"
        TASK_B="realistic_cachefollower"
        ;;
    cachefollower-websearch)
        TASK_A="realistic_cachefollower"
        TASK_B="realistic_websearch"
        ;;
    websearch-webserver)
        TASK_A="realistic_websearch"
        TASK_B="realistic_webserver"
        ;;
    *) echo "unknown realistic workload pair: ${PAIR}" >&2; exit 2 ;;
esac
[[ -n "${RUN_ID}" ]] || RUN_ID="real_${PAIR//-/_}_s${SEED}"

COMMON_ARGS=(
    --run-id "${RUN_ID}"
    --task-a "${TASK_A}"
    --task-b "${TASK_B}"
    --task-pair-mode independent
    --seed "${SEED}"
    --buffer-kb 400
    --action-space multiscale
    --shared-replay false
    --updates-per-task "${UPDATES}"
    --phase-epochs "${PHASE_EPOCHS}"
    --eps-decay 2500
    --task-b-eps-start 1.0
    --task-b-eps-decay 2500
    --acc-hidden-dims "32,64,64,32"
    --reward-profile tail_safe
    --reward-queue-lambda 5.0
    --reward-ecn-lambda 5.0
    --reward-weights "0.50,0.30,0.20"
    --screen-watch-ports "${WATCH_PORTS}"
    --port "${PORT}"
)
[[ "${RESUME}" -eq 0 ]] || COMMON_ARGS+=(--resume)
[[ "${SMOKE}" -eq 0 ]] || COMMON_ARGS+=(--smoke)

run_prepare() {
    bash "${ROOT}/scripts/continual_validation/run_continual.sh" \
        --stage prepare "${COMMON_ARGS[@]}"
}

run_screen() {
    bash "${ROOT}/scripts/continual_validation/run_continual.sh" \
        --stage screen "${COMMON_ARGS[@]}" --report-only
}

run_acc() {
    bash "${ROOT}/scripts/continual_validation/run_continual.sh" \
        --stage acc "${COMMON_ARGS[@]}" --report-only
}

run_analysis() {
    python "${ROOT}/scripts/continual_validation/analyze_port_continual.py" \
        --run-dir "${ROOT}/experiments/continual_validation/${RUN_ID}" \
        --ports "${WATCH_PORTS}"
}

run_interpolation() {
    bash "${ROOT}/scripts/continual_validation/run_port_path_sweep.sh" \
        --stage all \
        --base-run-id "${RUN_ID}" \
        --endpoint-source screen \
        --endpoint-port "${ENDPOINT_PORT}" \
        --target-ports "${TARGET_PORTS}" \
        --alpha-min 0.0 --alpha-max 1.0 --alpha-step "${ALPHA_STEP}" \
        --link-gbps 40 --port "$((PORT + 100))"
}

case "${STAGE}" in
    prepare) run_prepare ;;
    screen) run_screen ;;
    acc) run_acc ;;
    analyze) run_analysis ;;
    interpolate) run_interpolation ;;
    all)
        run_prepare
        run_screen
        run_acc
        run_analysis
        run_interpolation
        ;;
esac

echo "Realistic ACC stage '${STAGE}' complete: ${RUN_ID}"
