#!/usr/bin/env bash
# Registered ACC-local versus SOR paper protocol. Training reward is tail_safe only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE=""
BASE_RUN_ID="real_web_cache_target100_s1"
RUN_ID="sor_paper_web_cache_s1"
SEED=1
A_UPDATES=600
B_UPDATES=300
CHECKPOINTS=""
JOBS=1
PORT_BASE=7200
RESUME=0

usage() {
    cat <<'EOF'
Usage: bash scripts/continual_validation/run_sor_paper.sh [options]

Stages: smoke, acquire, continue, extend-b, analyze

Required/important options:
  --stage NAME
  --base-run-id ID       Existing realistic_webserver -> realistic_cachefollower run
  --run-id ID
  --seed N
  --a-updates N          Final A update budget (default 600)
  --b-updates N          Final B update budget (default 300)
  --checkpoints CSV      Absolute within-task update counts
  --jobs 1|2             ACC-local and SOR concurrency (default 1)
  --port-base N          Uses N for ACC-local and N+1 for SOR
  --resume

This entry point deliberately has no reward-profile or action-space option:
it locks tail_safe, multiscale (9x7), buffer=400 KB, global epsilon decay=2500,
target synchronization every 100 global optimizer updates, and per-environment-step
FCT traces for every training and frozen-evaluation episode.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --base-run-id) BASE_RUN_ID="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --a-updates) A_UPDATES="$2"; shift 2 ;;
        --b-updates) B_UPDATES="$2"; shift 2 ;;
        --checkpoints) CHECKPOINTS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --port-base) PORT_BASE="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$STAGE" in smoke|acquire|continue|extend-b|analyze) ;;
    *) echo "--stage must be smoke, acquire, continue, extend-b, or analyze" >&2; exit 2 ;;
esac
[[ "$A_UPDATES" =~ ^[1-9][0-9]*$ && "$B_UPDATES" =~ ^[1-9][0-9]*$ ]] || {
    echo "update budgets must be positive integers" >&2; exit 2;
}
case "$JOBS" in 1|2) ;; *) echo "--jobs must be 1 or 2" >&2; exit 2 ;; esac

BASE="${ROOT}/experiments/continual_validation/${BASE_RUN_ID}"
OUT="${ROOT}/experiments/continual_validation/${RUN_ID}"
[[ -f "${BASE}/manifest.json" ]] || {
    echo "Missing base manifest: ${BASE}/manifest.json" >&2; exit 1;
}

run_driver() {
    local scope="$1" a_points="$2" b_points="$3" output="$4"
    local args=(
        python "${ROOT}/scripts/continual_validation/run_sor_paper.py"
        --base-run-dir "${BASE}" --output-dir "${output}" --stage "${scope}"
        --seed "${SEED}" --a-points "${a_points}" --b-points "${b_points}"
        --jobs "${JOBS}" --port-base "${PORT_BASE}"
    )
    [[ "$RESUME" -eq 0 ]] || args+=(--resume)
    "${args[@]}"
}

case "$STAGE" in
    smoke)
        python "${ROOT}/sor/test_sor_paper_alignment.py"
        SMOKE_OUT="${OUT}_smoke"
        RESUME=1
        run_driver acquire 20 20 "$SMOKE_OUT"
        run_driver continue 20 20 "$SMOKE_OUT"
        python "${ROOT}/scripts/continual_validation/analyze_sor_paper.py" \
            --run-dir "$SMOKE_OUT"
        python "${ROOT}/scripts/continual_validation/verify_sor_paper_smoke.py" \
            --run-dir "$SMOKE_OUT"
        ;;
    acquire)
        [[ -n "$CHECKPOINTS" ]] || CHECKPOINTS="100,300,${A_UPDATES}"
        [[ ",${CHECKPOINTS}," == *",${A_UPDATES},"* ]] || {
            echo "--checkpoints must include --a-updates ${A_UPDATES}" >&2; exit 2;
        }
        run_driver acquire "$CHECKPOINTS" "100,${B_UPDATES}" "$OUT"
        python "${ROOT}/scripts/continual_validation/analyze_sor_paper.py" \
            --run-dir "$OUT" --scope acquire
        ;;
    continue)
        [[ -n "$CHECKPOINTS" ]] || CHECKPOINTS="100,${B_UPDATES}"
        [[ ",${CHECKPOINTS}," == *",${B_UPDATES},"* ]] || {
            echo "--checkpoints must include --b-updates ${B_UPDATES}" >&2; exit 2;
        }
        RESUME=1
        run_driver continue "100,300,${A_UPDATES}" "$CHECKPOINTS" "$OUT"
        ;;
    extend-b)
        [[ -f "${OUT}/protocol.json" ]] || { echo "Run acquire/continue first" >&2; exit 1; }
        [[ -n "$CHECKPOINTS" ]] || CHECKPOINTS="100,300,${B_UPDATES}"
        RESUME=1
        A_POINTS="$(python -c 'import json,sys; print(",".join(map(str,json.load(open(sys.argv[1]))["a_points"])))' "${OUT}/protocol.json")"
        run_driver continue "$A_POINTS" "$CHECKPOINTS" "$OUT"
        ;;
    analyze)
        python "${ROOT}/scripts/continual_validation/analyze_sor_paper.py" \
            --run-dir "$OUT"
        ;;
esac

echo "SOR paper stage '${STAGE}' complete: ${OUT}"
