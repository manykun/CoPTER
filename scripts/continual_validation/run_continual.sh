#!/usr/bin/env bash
#
# Reproducible A->B continual-learning experiment for ACC and SOR.
#
# The same model and replay memory continue across task phases.  Evaluation is
# greedy and frozen.  ACC and SOR receive identical traffic, action space,
# network width, phase-local epsilon schedule, switch buffer and optimizer-
# update budget and one common tail-safe reward.  Only the traffic task changes
# at the A->B boundary, so forgetting cannot be attributed to reward switching.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="all"
RUN_ID="continual_ab_seed1"
TASK_A="incast"
TASK_B="throughput"
SEED=1
BUFFER_KB=400
PHASE_EPOCHS=100
UPDATES_PER_TASK=600
UPDATES_TASK_A=""
UPDATES_TASK_B=""
EPS_DECAY=2500
EPSILON_SCHEDULE="phase"
TARGET_UPDATE_INTERVAL=100
TASK_B_EPS_START="1.0"
TASK_B_EPS_DECAY=""
ACC_HIDDEN_DIMS="32,64,64,32"
REWARD_PROFILE="tail_safe"
REWARD_QUEUE_LAMBDA="5.0"
REWARD_ECN_LAMBDA="5.0"
REWARD_WEIGHTS="0.50,0.30,0.20"
KMIN_RANGE="20000,50000"
KMAX_RANGE="50000,100000"
BASELINE_STOP_TIME="4.00"
PORT=5756
MAX_FLOWS=0
SMOKE=0
RESUME=0
REPORT_ONLY=0
SHARED_REPLAY="true"
ACTION_SPACE="legacy"
TASK_PAIR_MODE="independent"
SCREEN_WATCH_PORTS=""
SCREEN_MIN_ACTIVE_SAMPLES=0
SCREEN_MIN_CONGESTED_SAMPLES=0
SCREEN_MIN_OLD_TASK_P95_PENALTY="0.0"
KMIN_RANGE_EXPLICIT=0
KMAX_RANGE_EXPLICIT=0
UPDATES_TASK_A_EXPLICIT=0
UPDATES_TASK_B_EXPLICIT=0
TASK_B_EPS_EXPLICIT=0
REWARD_WEIGHTS_EXPLICIT=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/continual_validation/run_continual.sh [options]

Stages:
  prepare   Generate the fixed task flows/configs and write manifest.json
  screen    Fixed-action conflict screen; must pass before long training
  acc       Run frozen evaluations and A->B continual training with ACC
  sor       Repeat the identical experiment with SOR
  analyze   Build REPORT.md and apply the registered gates
  all       Run prepare, ACC, SOR and analysis

Important options:
  --run-id ID
  --task-a NAME
  --task-b NAME
  --seed N
  --updates-per-task N
  --updates-task-a N     Override optimizer-update budget for task A
  --updates-task-b N     Override optimizer-update budget for task B
  --phase-epochs N       Safety cap; update count is the actual budget
  --buffer-kb N
  --eps-decay N
  --epsilon-schedule NAME phase resets at A→B; global continues across A→B
  --target-update-interval N  Hard target sync every N global optimizer updates
  --task-b-eps-start X   Reset task-B exploration to X (default: 1.0)
  --task-b-eps-decay N   Task-B phase-local epsilon decay
  --acc-hidden-dims CSV
  --reward-profile NAME
  --reward-queue-lambda X
  --reward-ecn-lambda X
  --reward-weights CSV
  --port N
  --smoke
  --resume
  --report-only          Record all measurements without PASS/FAIL gating
  --shared-replay BOOL   ACC cross-port global replay: true (default) or false
  --action-space NAME    legacy or multiscale (ACC only)
  --task-pair-mode NAME  independent, same-flows, or workload-shift
  --screen-watch-ports CSV
  --screen-min-active-samples N
  --screen-min-congested-samples N
  --screen-min-old-task-p95-penalty X
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage) STAGE="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --task-a) TASK_A="$2"; shift 2 ;;
        --task-b) TASK_B="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --buffer-kb) BUFFER_KB="$2"; shift 2 ;;
        --phase-epochs) PHASE_EPOCHS="$2"; shift 2 ;;
        --updates-per-task) UPDATES_PER_TASK="$2"; shift 2 ;;
        --updates-task-a) UPDATES_TASK_A="$2"; UPDATES_TASK_A_EXPLICIT=1; shift 2 ;;
        --updates-task-b) UPDATES_TASK_B="$2"; UPDATES_TASK_B_EXPLICIT=1; shift 2 ;;
        --eps-decay) EPS_DECAY="$2"; shift 2 ;;
        --epsilon-schedule) EPSILON_SCHEDULE="$2"; shift 2 ;;
        --target-update-interval) TARGET_UPDATE_INTERVAL="$2"; shift 2 ;;
        --task-b-eps-start) TASK_B_EPS_START="$2"; TASK_B_EPS_EXPLICIT=1; shift 2 ;;
        --task-b-eps-decay) TASK_B_EPS_DECAY="$2"; TASK_B_EPS_EXPLICIT=1; shift 2 ;;
        --acc-hidden-dims) ACC_HIDDEN_DIMS="$2"; shift 2 ;;
        --reward-profile) REWARD_PROFILE="$2"; shift 2 ;;
        --reward-queue-lambda) REWARD_QUEUE_LAMBDA="$2"; shift 2 ;;
        --reward-ecn-lambda) REWARD_ECN_LAMBDA="$2"; shift 2 ;;
        --reward-weights) REWARD_WEIGHTS="$2"; REWARD_WEIGHTS_EXPLICIT=1; shift 2 ;;
        --kmin-range) KMIN_RANGE="$2"; KMIN_RANGE_EXPLICIT=1; shift 2 ;;
        --kmax-range) KMAX_RANGE="$2"; KMAX_RANGE_EXPLICIT=1; shift 2 ;;
        --baseline-stop-time) BASELINE_STOP_TIME="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
        --report-only) REPORT_ONLY=1; shift ;;
        --shared-replay) SHARED_REPLAY="$2"; shift 2 ;;
        --action-space) ACTION_SPACE="$2"; shift 2 ;;
        --task-pair-mode) TASK_PAIR_MODE="$2"; shift 2 ;;
        --screen-watch-ports) SCREEN_WATCH_PORTS="$2"; shift 2 ;;
        --screen-min-active-samples) SCREEN_MIN_ACTIVE_SAMPLES="$2"; shift 2 ;;
        --screen-min-congested-samples) SCREEN_MIN_CONGESTED_SAMPLES="$2"; shift 2 ;;
        --screen-min-old-task-p95-penalty) SCREEN_MIN_OLD_TASK_P95_PENALTY="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${STAGE}" in
    prepare|screen|acc|sor|analyze|all) ;;
    *) echo "Invalid stage: ${STAGE}" >&2; exit 2 ;;
esac
[[ "${RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    echo "run-id may contain only letters, digits, '.', '_' and '-'" >&2
    exit 2
}
[[ "${TASK_A}" != "${TASK_B}" ]] || {
    echo "task A and task B must be different" >&2
    exit 2
}
[[ "${PHASE_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "phase-epochs must be a positive integer" >&2
    exit 2
}
[[ "${UPDATES_PER_TASK}" =~ ^[1-9][0-9]*$ ]] || {
    echo "updates-per-task must be a positive integer" >&2
    exit 2
}
[[ -n "${UPDATES_TASK_A}" ]] || UPDATES_TASK_A="${UPDATES_PER_TASK}"
[[ -n "${UPDATES_TASK_B}" ]] || UPDATES_TASK_B="${UPDATES_PER_TASK}"
[[ -n "${TASK_B_EPS_DECAY}" ]] || TASK_B_EPS_DECAY="${EPS_DECAY}"
for value in "${UPDATES_TASK_A}" "${UPDATES_TASK_B}" "${EPS_DECAY}" "${TASK_B_EPS_DECAY}"; do
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
        echo "update budgets and epsilon decay must be positive integers" >&2
        exit 2
    }
done
[[ "${TARGET_UPDATE_INTERVAL}" =~ ^[1-9][0-9]*$ ]] || {
    echo "target-update-interval must be a positive integer" >&2
    exit 2
}
for value in "${SCREEN_MIN_ACTIVE_SAMPLES}" "${SCREEN_MIN_CONGESTED_SAMPLES}"; do
    [[ "${value}" =~ ^[0-9]+$ ]] || {
        echo "screen sample thresholds must be non-negative integers" >&2
        exit 2
    }
done
case "${SHARED_REPLAY}" in
    true|false) ;;
    *) echo "shared-replay must be true or false" >&2; exit 2 ;;
esac
case "${ACTION_SPACE}" in
    legacy|multiscale) ;;
    *) echo "action-space must be legacy or multiscale" >&2; exit 2 ;;
esac
case "${EPSILON_SCHEDULE}" in
    phase|global) ;;
    *) echo "epsilon-schedule must be phase or global" >&2; exit 2 ;;
esac
if [[ "${EPSILON_SCHEDULE}" == "global" && "${TASK_B_EPS_EXPLICIT}" -eq 1 ]]; then
    echo "task-B epsilon overrides are incompatible with --epsilon-schedule global" >&2
    exit 2
fi
case "${TASK_PAIR_MODE}" in
    independent|same-flows|workload-shift) ;;
    *) echo "task-pair-mode must be independent, same-flows, or workload-shift" >&2; exit 2 ;;
esac
if [[ "${ACTION_SPACE}" == "multiscale" ]]; then
    [[ "${KMIN_RANGE_EXPLICIT}" -eq 1 ]] || KMIN_RANGE="5000,50000"
    [[ "${KMAX_RANGE_EXPLICIT}" -eq 1 ]] || KMAX_RANGE="15000,100000"
    [[ "${KMIN_RANGE}" == "5000,50000" && "${KMAX_RANGE}" == "15000,100000" ]] || {
        echo "multiscale requires --kmin-range 5000,50000 and --kmax-range 15000,100000" >&2
        exit 2
    }
    [[ "${STAGE}" != "sor" && "${STAGE}" != "all" ]] || {
        echo "multiscale is an ACC-only experiment; run prepare, screen, and acc separately" >&2
        exit 2
    }
fi
python - "${REWARD_PROFILE}" "${REWARD_QUEUE_LAMBDA}" "${REWARD_ECN_LAMBDA}" \
    "${REWARD_WEIGHTS}" "${TASK_B_EPS_START}" \
    "${SCREEN_MIN_OLD_TASK_P95_PENALTY}" "${SCREEN_WATCH_PORTS}" <<'PY'
import sys

profile = sys.argv[1]
if profile not in ("weighted", "tail_safe"):
    raise SystemExit(f"unsupported reward profile: {profile}")
for label, raw in zip(("queue", "ECN"), sys.argv[2:]):
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{label} lambda is invalid: {exc}")
    if value < 0:
        raise SystemExit(f"{label} lambda must be non-negative")
weights = sys.argv[4].split(",")
if len(weights) != 3:
    raise SystemExit("reward weights must contain three comma-separated values")
try:
    weights = [float(value) for value in weights]
    task_b_epsilon = float(sys.argv[5])
    old_task_penalty = float(sys.argv[6])
except ValueError as exc:
    raise SystemExit(f"invalid floating-point experiment parameter: {exc}")
if any(value < 0 for value in weights) or sum(weights) <= 0:
    raise SystemExit("reward weights must be non-negative and sum to more than zero")
if not 0 <= task_b_epsilon <= 1:
    raise SystemExit("task-b-eps-start must be in [0, 1]")
if old_task_penalty < 0:
    raise SystemExit("screen-min-old-task-p95-penalty must be non-negative")
try:
    ports = [int(value.strip()) for value in sys.argv[7].split(",") if value.strip()]
except ValueError as exc:
    raise SystemExit(f"screen watch ports are invalid: {exc}")
if len(ports) != len(set(ports)) or any(port < 0 for port in ports):
    raise SystemExit("screen watch ports must be unique non-negative integers")
PY
if [[ -z "${SCREEN_WATCH_PORTS}" &&
      ( "${SCREEN_MIN_ACTIVE_SAMPLES}" -gt 0 ||
        "${SCREEN_MIN_CONGESTED_SAMPLES}" -gt 0 ) ]]; then
    echo "screen sample thresholds require --screen-watch-ports" >&2
    exit 2
fi

if [[ "${SMOKE}" -eq 1 ]]; then
    PHASE_EPOCHS=3
    UPDATES_PER_TASK=10
    UPDATES_TASK_A=10
    UPDATES_TASK_B=10
    MAX_FLOWS=2000
fi

RUN_DIR="${ROOT}/experiments/continual_validation/${RUN_ID}"
MANIFEST="${RUN_DIR}/manifest.json"
FLOW_DIR="${ROOT}/simulation/mix/acc_validation"
RUN_OUTPUT_DIR="${RUN_DIR}/ns3_output"
RUNTIME_CONFIG_DIR="${RUN_DIR}/runtime"
mkdir -p "${RUN_DIR}"

manifest_json() {
    python - "${RUN_ID}" "${TASK_A}" "${TASK_B}" "${SEED}" \
        "${BUFFER_KB}" "${PHASE_EPOCHS}" "${UPDATES_PER_TASK}" "${EPS_DECAY}" \
        "${ACC_HIDDEN_DIMS}" "${REWARD_PROFILE}" \
        "${REWARD_QUEUE_LAMBDA}" "${REWARD_ECN_LAMBDA}" "${KMIN_RANGE}" \
        "${KMAX_RANGE}" "${BASELINE_STOP_TIME}" "${MAX_FLOWS}" \
        "${ACTION_SPACE}" "${SHARED_REPLAY}" "${UPDATES_TASK_A}" \
        "${UPDATES_TASK_B}" "${TASK_B_EPS_START}" "${TASK_B_EPS_DECAY}" \
        "${EPSILON_SCHEDULE}" \
        "${TASK_PAIR_MODE}" "${SCREEN_WATCH_PORTS}" \
        "${SCREEN_MIN_ACTIVE_SAMPLES}" "${SCREEN_MIN_CONGESTED_SAMPLES}" \
        "${SCREEN_MIN_OLD_TASK_P95_PENALTY}" "${REWARD_WEIGHTS}" \
        "${UPDATES_TASK_A_EXPLICIT}" "${UPDATES_TASK_B_EXPLICIT}" \
        "${TASK_B_EPS_EXPLICIT}" "${REWARD_WEIGHTS_EXPLICIT}" \
        "${TARGET_UPDATE_INTERVAL}" <<'PY'
import json
import sys

keys = (
    "run_id", "task_a", "task_b", "seed", "buffer_kb", "phase_epochs",
    "updates_per_task", "epsilon_decay_steps", "hidden_dims",
    "reward_profile", "reward_queue_lambda", "reward_ecn_lambda", "kmin_range",
    "kmax_range", "simulator_stop_time", "max_flows", "action_space",
)
values = sys.argv[1:18]
record = dict(zip(keys, values))
for key in (
    "seed", "buffer_kb", "phase_epochs", "updates_per_task",
    "epsilon_decay_steps", "max_flows",
):
    record[key] = int(record[key])
for key in ("reward_queue_lambda", "reward_ecn_lambda"):
    record[key] = float(record[key])
if record["action_space"] == "multiscale":
    record["methods"] = ["acc"]
else:
    # Preserve legacy manifests so previously prepared runs remain resumable.
    del record["action_space"]
    record["methods"] = ["acc", "sor"]
record["curriculum"] = [record["task_a"], record["task_b"]]
record["evaluation"] = {
    "greedy": True,
    "updates": False,
    "old_task_comparison": ["after_a", "after_b"],
    "new_task_acquisition": ["after_a", "after_b"],
}
record["experiment_type"] = "traffic_shift_common_objective"
shared_replay = sys.argv[18]
updates_a, updates_b = map(int, sys.argv[19:21])
task_b_eps_start = float(sys.argv[21])
task_b_eps_decay = int(sys.argv[22])
epsilon_schedule = sys.argv[23]
task_pair_mode = sys.argv[24]
screen_watch_ports = sys.argv[25]
screen_active = int(sys.argv[26])
screen_congested = int(sys.argv[27])
screen_old_p95 = float(sys.argv[28])
reward_weights = [float(value) for value in sys.argv[29].split(",")]
updates_explicit = sys.argv[30] == "1" or sys.argv[31] == "1"
task_b_eps_explicit = sys.argv[32] == "1"
reward_weights_explicit = sys.argv[33] == "1"
target_update_interval = int(sys.argv[34])

record["epsilon_schedule"] = "reset_per_task"
if epsilon_schedule == "global":
    record["epsilon_schedule"] = {
        "scope": "global",
        "start": 1.0,
        "end": 0.05,
        "decay_steps": record["epsilon_decay_steps"],
        "task_boundary_reset": False,
    }
if updates_explicit:
    record["updates_task_a"] = updates_a
    record["updates_task_b"] = updates_b
if task_b_eps_explicit and epsilon_schedule == "phase":
    record["epsilon_schedule"] = {
        "task_a": {"start": 1.0, "end": 0.05, "decay_steps": record["epsilon_decay_steps"]},
        "task_b": {"start": task_b_eps_start, "end": 0.05, "decay_steps": task_b_eps_decay},
    }
if reward_weights_explicit:
    record["reward_weights"] = reward_weights
record["target_update_interval"] = target_update_interval
if task_pair_mode != "independent":
    record["task_pair_mode"] = task_pair_mode
if screen_watch_ports:
    record["screen_watch_ports"] = [
        int(value) for value in screen_watch_ports.split(",") if value.strip()
    ]
if screen_active or screen_congested or screen_old_p95:
    record["screen_requirements"] = {
        "min_active_samples": screen_active,
        "min_congested_samples": screen_congested,
        "min_old_task_p95_penalty": screen_old_p95,
    }
if shared_replay == "false":
    # Omit the default true value so manifests created before this ablation
    # remain byte-for-byte compatible with the expected record.
    record["shared_replay"] = False
print(json.dumps(record, indent=2, sort_keys=True))
PY
}

runtime_config_path() {
    local task="$1"
    echo "${RUNTIME_CONFIG_DIR}/${task}_seed${SEED}.conf"
}

output_base() {
    local task="$1"
    echo "${RUN_OUTPUT_DIR}/${task}_seed${SEED}"
}

ensure_runtime_configs() {
    # Every run receives immutable input flows plus a private NS-3 output
    # prefix. Separate socket ports alone are insufficient: without this,
    # parallel evaluations overwrite the same .fct/.pfc/.queue files.
    mkdir -p "${RUNTIME_CONFIG_DIR}" "${RUN_OUTPUT_DIR}"
    local task source destination flow base tmp
    for task in "${TASK_A}" "${TASK_B}"; do
        source="${RUN_DIR}/tasks/${task}/input.conf"
        flow="${RUN_DIR}/tasks/${task}/input.flow"
        destination="$(runtime_config_path "${task}")"
        base="$(output_base "${task}")"
        tmp="${destination}.tmp"
        [[ -s "${source}" && -s "${flow}" ]] || {
            echo "Missing frozen task input for ${task}" >&2
            return 1
        }
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
        ' "${source}" > "${tmp}"
        mv "${tmp}" "${destination}"
    done
}

check_manifest() {
    [[ -f "${MANIFEST}" ]] || {
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
    print("Prepared manifest does not match the requested experiment.", file=sys.stderr)
    print("Run prepare again with a new run-id or the original arguments.", file=sys.stderr)
    print("expected=" + json.dumps(expected, sort_keys=True), file=sys.stderr)
    print("actual=" + json.dumps(actual, sort_keys=True), file=sys.stderr)
    raise SystemExit(1)
PY
}

validate_configs() {
    local task name meta config
    for task in "${TASK_A}" "${TASK_B}"; do
        name="${task}_seed${SEED}"
        meta="${FLOW_DIR}/${name}.meta"
        config="${FLOW_DIR}/${name}.conf"
        [[ -s "${FLOW_DIR}/${name}.flow" && -s "${meta}" && -s "${config}" ]] || {
            echo "Missing prepared files for ${name}; run --stage prepare." >&2
            return 1
        }
        [[ "$(awk -F= '$1=="buffer_kb"{print $2}' "${meta}")" == "${BUFFER_KB}" ]] || {
            echo "Prepared buffer mismatch for ${name}" >&2
            return 1
        }
        [[ "$(awk -F= '$1=="max_flows"{print $2}' "${meta}")" == "${MAX_FLOWS}" ]] || {
            echo "Prepared flow cap mismatch for ${name}" >&2
            return 1
        }
        grep -Fxq "ENABLE_COPTER 1" "${config}" || {
            echo "Dynamic controller is disabled in ${config}" >&2
            return 1
        }
        if [[ -d "${RUN_DIR}/tasks/${task}" ]]; then
            cmp -s "${FLOW_DIR}/${name}.flow" "${RUN_DIR}/tasks/${task}/input.flow" ||
                { echo "Flow changed since prepare: ${name}.flow" >&2; return 1; }
            cmp -s "${config}" "${RUN_DIR}/tasks/${task}/input.conf" ||
                { echo "Config changed since prepare: ${name}.conf" >&2; return 1; }
            cmp -s "${meta}" "${RUN_DIR}/tasks/${task}/input.meta" ||
                { echo "Metadata changed since prepare: ${name}.meta" >&2; return 1; }
        fi
    done
}

verify_task_pair() {
    case "${TASK_PAIR_MODE}" in
        independent) return 0 ;;
        same-flows)
            python "${ROOT}/scripts/continual_validation/verify_task_pair.py" \
                --task-a "${RUN_DIR}/tasks/${TASK_A}/input.flow" \
                --task-b "${RUN_DIR}/tasks/${TASK_B}/input.flow" \
                --json-output "${RUN_DIR}/task_pair_analysis.json" \
                --report-output "${RUN_DIR}/TASK_PAIR_REPORT.md" \
                --require-identical-flows \
                --require-timing-shift
            ;;
        workload-shift)
            python "${ROOT}/scripts/continual_validation/verify_task_pair.py" \
                --task-a "${RUN_DIR}/tasks/${TASK_A}/input.flow" \
                --task-b "${RUN_DIR}/tasks/${TASK_B}/input.flow" \
                --json-output "${RUN_DIR}/task_pair_analysis.json" \
                --report-output "${RUN_DIR}/TASK_PAIR_REPORT.md" \
                --require-same-endpoint-support \
                --require-flow-distribution-shift
            ;;
    esac
}

prepare() {
    if [[ -f "${MANIFEST}" ]]; then
        if [[ "${RESUME}" -ne 1 ]]; then
            echo "Run already exists: ${RUN_DIR}" >&2
            echo "Use a new --run-id, or pass --resume only when arguments are unchanged." >&2
            return 1
        fi
        check_manifest
        validate_configs
        verify_task_pair
        ensure_runtime_configs
        echo "Prepared run already matches requested arguments; keeping fixed files."
        return 0
    fi
    bash "${ROOT}/scripts/acc_validation/prepare_scenarios.sh" \
        --seeds "${SEED}" \
        --scenarios "${TASK_A} ${TASK_B}" \
        --buffer-kb "${BUFFER_KB}" \
        --kmin-range "${KMIN_RANGE}" \
        --kmax-range "${KMAX_RANGE}" \
        --max-flows "${MAX_FLOWS}" \
        --baseline-stop-time "${BASELINE_STOP_TIME}"
    local task name destination
    for task in "${TASK_A}" "${TASK_B}"; do
        name="${task}_seed${SEED}"
        destination="${RUN_DIR}/tasks/${task}"
        mkdir -p "${destination}"
        cp "${FLOW_DIR}/${name}.flow" "${destination}/input.flow"
        cp "${FLOW_DIR}/${name}.conf" "${destination}/input.conf"
        cp "${FLOW_DIR}/${name}.meta" "${destination}/input.meta"
    done
    verify_task_pair
    manifest_json > "${MANIFEST}.tmp"
    mv "${MANIFEST}.tmp" "${MANIFEST}"
    validate_configs
    ensure_runtime_configs
    echo "Continual-learning manifest written to ${MANIFEST}"
}

copy_eval_outputs() {
    local method="$1" phase="$2" task="$3" exp="$4" model_dir="$5"
    local name="${task}_seed${SEED}"
    local destination="${RUN_DIR}/eval/${method}/${phase}/${task}"
    local base
    base="$(output_base "${task}")"
    local metrics="${model_dir}/${exp}_metrics.jsonl"
    mkdir -p "${destination}"
    shopt -s nullglob
    local files=("${base}".*)
    if [[ ${#files[@]} -eq 0 ]]; then
        echo "No ns-3 outputs produced for ${base}" >&2
        return 1
    fi
    cp "${files[@]}" "${destination}/"
    shopt -u nullglob
    cp "${RUN_DIR}/tasks/${task}/input.flow" "${destination}/input.flow"
    cp "$(runtime_config_path "${task}")" "${destination}/input.conf"
    cp "${RUN_DIR}/tasks/${task}/input.meta" "${destination}/input.meta"
    [[ -s "${metrics}" ]] || {
        echo "Metrics not produced: ${metrics}" >&2
        return 1
    }
    tail -n 1 "${metrics}" > "${destination}/metrics.json"
    local agent_log="${ROOT}/copter/training_logs/${exp}/agent_ep"
    local log_candidate
    log_candidate="$(ls -t "${agent_log}"*.log 2>/dev/null | head -n 1 || true)"
    [[ -z "${log_candidate}" ]] || cp "${log_candidate}" "${destination}/agent.log"
}

evaluate() {
    local method="$1" phase="$2" task="$3" exp="$4" model_dir="$5" port="$6"
    local name="${task}_seed${SEED}"
    local base destination trace
    base="$(output_base "${task}")"
    destination="${RUN_DIR}/eval/${method}/${phase}/${task}"
    trace="${destination}/watch_trace.jsonl"
    mkdir -p "${destination}"
    rm -f "${trace}"
    local watch_args=()
    [[ -z "${SCREEN_WATCH_PORTS}" ]] || watch_args+=(--watch-ports "${SCREEN_WATCH_PORTS}")
    rm -f "${base}".*
    echo "[${method}] frozen evaluation phase=${phase} task=${task}"
    bash "${ROOT}/run_training.sh" \
        --one-shot \
        --eval-greedy \
        --eval-tag "${phase}_${task}" \
        --config "$(runtime_config_path "${task}")" \
        --exp "${exp}" \
        --mode "${method^^}" \
        --port "${port}" \
        --seed "${SEED}" \
        --buffer "${BUFFER_KB}" \
        --model-dir "${model_dir}" \
        --episodes 1 \
        --eps-start 1.0 \
        --eps-end 0.05 \
        --eps-decay "${EPS_DECAY}" \
        --epsilon-schedule "${EPSILON_SCHEDULE}" \
        --target-update-interval "${TARGET_UPDATE_INTERVAL}" \
        --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
        --reward-weights "${REWARD_WEIGHTS}" \
        --reward-profile "${REWARD_PROFILE}" \
        --reward-queue-lambda "${REWARD_QUEUE_LAMBDA}" \
        --reward-ecn-lambda "${REWARD_ECN_LAMBDA}" \
        --shared-replay "${SHARED_REPLAY}" \
        --action-space "${ACTION_SPACE}" \
        --tb-enable false \
        --run-id "${RUN_ID}" \
        --phase "${phase}" \
        --watch-trace-file "${trace}" \
        "${watch_args[@]}"
    copy_eval_outputs "${method}" "${phase}" "${task}" "${exp}" "${model_dir}"
}

copy_screen_outputs() {
    local task="$1" label="$2" exp="$3" model_dir="$4"
    local name="${task}_seed${SEED}"
    local destination="${RUN_DIR}/screen/${task}/${label}"
    local base
    base="$(output_base "${task}")"
    local metrics="${model_dir}/${exp}_metrics.jsonl"
    mkdir -p "${destination}"
    shopt -s nullglob
    local files=("${base}".*)
    [[ ${#files[@]} -gt 0 ]] || {
        echo "No ns-3 outputs produced for conflict screen ${task}/${label}" >&2
        return 1
    }
    cp "${files[@]}" "${destination}/"
    shopt -u nullglob
    cp "${RUN_DIR}/tasks/${task}/input.flow" "${destination}/input.flow"
    cp "$(runtime_config_path "${task}")" "${destination}/input.conf"
    cp "${RUN_DIR}/tasks/${task}/input.meta" "${destination}/input.meta"
    tail -n 1 "${metrics}" > "${destination}/metrics.json"
}

screen() {
    check_manifest
    validate_configs
    ensure_runtime_configs
    local screen_dir="${RUN_DIR}/screen"
    local model_dir="${screen_dir}/models"
    local exp="continual_${RUN_ID}_screen_s${SEED}"
    local task label action name base specifications
    local watch_args=()
    [[ -z "${SCREEN_WATCH_PORTS}" ]] || watch_args+=(--watch-ports "${SCREEN_WATCH_PORTS}")
    mkdir -p "${model_dir}"
    if [[ "${ACTION_SPACE}" == "multiscale" ]]; then
        specifications=(
            "low_strong:1,5"
            "low_moderate:2,3"
            "mid:4,3"
            "high_moderate:6,2"
            "high_gentle:8,1"
        )
    else
        specifications=(
            "aggressive:0,0,9"
            "balanced:2,1,4"
            "permissive:5,3,0"
        )
    fi
    for task in "${TASK_A}" "${TASK_B}"; do
        for specification in "${specifications[@]}"; do
            label="${specification%%:*}"
            action="${specification#*:}"
            [[ -s "${screen_dir}/${task}/${label}/metrics.json" ]] && continue
            name="${task}_seed${SEED}"
            base="$(output_base "${task}")"
            rm -f "${base}".*
            echo "[screen] task=${task} action=${label}(${action}) reward=${REWARD_PROFILE}"
            bash "${ROOT}/run_training.sh" \
                --one-shot --eval-greedy \
                --force-action "${action}" \
                --action-space "${ACTION_SPACE}" \
                --eval-tag "screen_${task}_${label}" \
                --config "$(runtime_config_path "${task}")" \
                --exp "${exp}" --mode ACC --port "${PORT}" --seed "${SEED}" \
                --buffer "${BUFFER_KB}" --model-dir "${model_dir}" --episodes 1 \
                --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
                --reward-weights "${REWARD_WEIGHTS}" \
                --reward-profile "${REWARD_PROFILE}" \
                --reward-queue-lambda "${REWARD_QUEUE_LAMBDA}" \
                --reward-ecn-lambda "${REWARD_ECN_LAMBDA}" \
                --shared-replay "${SHARED_REPLAY}" \
                --target-update-interval "${TARGET_UPDATE_INTERVAL}" \
                --tb-enable false --run-id "${RUN_ID}" --phase screen \
                "${watch_args[@]}"
            copy_screen_outputs "${task}" "${label}" "${exp}" "${model_dir}"
        done
    done
    local screen_gate_args=()
    if [[ "${REPORT_ONLY}" -eq 1 ]]; then
        screen_gate_args+=(--report-only)
    else
        screen_gate_args+=(--gate)
    fi
    python "${ROOT}/scripts/continual_validation/analyze_conflict.py" \
        --run-dir "${RUN_DIR}" \
        --min-reward-spread 0.02 \
        --min-p95-spread 0.05 \
        --min-completion-spread 0.02 \
        --completion-tolerance 0.01 \
        --required-watch-ports "${SCREEN_WATCH_PORTS}" \
        --min-port-active-samples "${SCREEN_MIN_ACTIVE_SAMPLES}" \
        --min-port-congested-samples "${SCREEN_MIN_CONGESTED_SAMPLES}" \
        --min-old-task-p95-penalty "${SCREEN_MIN_OLD_TASK_P95_PENALTY}" \
        "${screen_gate_args[@]}"
    mkdir -p "${screen_dir}/markers"
    touch "${screen_dir}/markers/complete"
    [[ "${REPORT_ONLY}" -eq 1 ]] || touch "${screen_dir}/markers/pass"
}

snapshot_models() {
    local method_dir="$1" phase="$2"
    local destination="${method_dir}/checkpoints/${phase}"
    mkdir -p "${destination}"
    # Replay can be very large; phase snapshots retain networks and counters.
    # The live model directory keeps the replay needed to continue into task B.
    find "${method_dir}/models" -maxdepth 1 -type f ! -name '*.pkl' \
        -exec cp -f {} "${destination}/" \;
}

train_method() {
    local method="$1"
    local method_dir="${RUN_DIR}/${method}"
    local model_dir="${method_dir}/models"
    local exp="continual_${RUN_ID}_${method}_s${SEED}"
    local method_port="${PORT}"
    local watch_args=()
    [[ -z "${SCREEN_WATCH_PORTS}" ]] || watch_args+=(--watch-ports "${SCREEN_WATCH_PORTS}")
    [[ "${method}" == "sor" ]] && method_port=$((PORT + 100))

    check_manifest
    validate_configs
    ensure_runtime_configs
    if [[ "${REPORT_ONLY}" -eq 1 ]]; then
        [[ -f "${RUN_DIR}/screen/markers/complete" ||
           -f "${RUN_DIR}/screen/markers/pass" ]] || {
            echo "Conflict screen is incomplete. Run --stage screen first." >&2
            return 1
        }
    else
        [[ -f "${RUN_DIR}/screen/markers/pass" ]] || {
            echo "Conflict screen has not passed. Run --stage screen first." >&2
            return 1
        }
    fi
    if [[ "${method}" == "sor" ]]; then
        if [[ "${REPORT_ONLY}" -eq 1 ]]; then
            [[ -f "${RUN_DIR}/acc/markers/complete" ]] || {
                echo "ACC measurements are incomplete. Finish --stage acc first." >&2
                return 1
            }
        elif [[ ! -f "${RUN_DIR}/acc/markers/continual_gate_pass" ]]; then
            echo "ACC has not demonstrated acquisition plus forgetting; SOR is premature." >&2
            return 1
        fi
    fi
    mkdir -p "${model_dir}" "${method_dir}/checkpoints" "${method_dir}/markers"
    if [[ -f "${model_dir}/${exp}_train_state.json" &&
          "${RESUME}" -ne 1 &&
          ! -f "${method_dir}/markers/complete" ]]; then
        echo "Partial ${method} run exists. Re-run with --resume or use a new run-id." >&2
        return 1
    fi
    if [[ -f "${method_dir}/markers/complete" ]]; then
        echo "${method} curriculum already complete; skipping."
        return 0
    fi

    if [[ ! -f "${method_dir}/markers/initial_eval" ]]; then
        evaluate "${method}" initial "${TASK_A}" "${exp}" "${model_dir}" "${method_port}"
        touch "${method_dir}/markers/initial_eval"
    fi

    if [[ ! -f "${method_dir}/markers/train_a" ]]; then
        echo "[${method}] train task A=${TASK_A}, target updates=${UPDATES_TASK_A}"
        bash "${ROOT}/run_training.sh" \
            --config "$(runtime_config_path "${TASK_A}")" \
            --exp "${exp}" \
            --mode "${method^^}" \
            --port "${method_port}" \
            --seed "${SEED}" \
            --buffer "${BUFFER_KB}" \
            --model-dir "${model_dir}" \
            --episodes "${PHASE_EPOCHS}" \
            --target-train-steps "${UPDATES_TASK_A}" \
            --eps-start 1.0 \
            --eps-end 0.05 \
            --eps-decay "${EPS_DECAY}" \
            --epsilon-schedule "${EPSILON_SCHEDULE}" \
            --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
            --reward-weights "${REWARD_WEIGHTS}" \
            --reward-profile "${REWARD_PROFILE}" \
            --reward-queue-lambda "${REWARD_QUEUE_LAMBDA}" \
            --reward-ecn-lambda "${REWARD_ECN_LAMBDA}" \
            --shared-replay "${SHARED_REPLAY}" \
            --action-space "${ACTION_SPACE}" \
            --target-update-interval "${TARGET_UPDATE_INTERVAL}" \
            --tb-enable false \
            --run-id "${RUN_ID}" \
            --phase train_a \
            --sor-save-buffer-every 1 \
            "${watch_args[@]}"
        snapshot_models "${method_dir}" after_a
        touch "${method_dir}/markers/train_a"
    fi

    if [[ ! -f "${method_dir}/markers/eval_after_a" ]]; then
        evaluate "${method}" after_a "${TASK_A}" "${exp}" "${model_dir}" "${method_port}"
        evaluate "${method}" after_a "${TASK_B}" "${exp}" "${model_dir}" "${method_port}"
        touch "${method_dir}/markers/eval_after_a"
    fi
    local acquisition_mode_args=()
    if [[ "${REPORT_ONLY}" -eq 1 ]]; then
        acquisition_mode_args+=(--report-only)
    else
        acquisition_mode_args+=(--gate)
    fi
    python "${ROOT}/scripts/continual_validation/check_acquisition.py" \
        --run-dir "${RUN_DIR}" --method "${method}" --task a \
        --min-reward-gain 0.02 --min-p95-gain 0.05 \
        --completion-tolerance 0.01 "${acquisition_mode_args[@]}"

    if [[ ! -f "${method_dir}/markers/train_b" ]]; then
        local after_a_state="${method_dir}/checkpoints/after_a/${exp}_train_state.json"
        [[ -s "${after_a_state}" ]] || {
            echo "Missing after-A train state: ${after_a_state}" >&2
            return 1
        }
        local after_a_epoch after_a_updates
        read -r after_a_epoch after_a_updates < <(
            python - "${after_a_state}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    state = json.load(handle)
print(int(state["epoch"]), int(state["global_train_step"]))
PY
        )
        local target_epoch=$((after_a_epoch + PHASE_EPOCHS))
        local target_updates=$((after_a_updates + UPDATES_TASK_B))
        echo "[${method}] continue same model on task B=${TASK_B}, "\
"additional updates=${UPDATES_TASK_B}, absolute target=${target_updates}"
        bash "${ROOT}/run_training.sh" \
            --config "$(runtime_config_path "${TASK_B}")" \
            --exp "${exp}" \
            --mode "${method^^}" \
            --port "${method_port}" \
            --seed "${SEED}" \
            --buffer "${BUFFER_KB}" \
            --model-dir "${model_dir}" \
            --episodes "${target_epoch}" \
            --target-train-steps "${target_updates}" \
            --eps-start "${TASK_B_EPS_START}" \
            --eps-end 0.05 \
            --eps-decay "${TASK_B_EPS_DECAY}" \
            --epsilon-schedule "${EPSILON_SCHEDULE}" \
            --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
            --reward-weights "${REWARD_WEIGHTS}" \
            --reward-profile "${REWARD_PROFILE}" \
            --reward-queue-lambda "${REWARD_QUEUE_LAMBDA}" \
            --reward-ecn-lambda "${REWARD_ECN_LAMBDA}" \
            --shared-replay "${SHARED_REPLAY}" \
            --action-space "${ACTION_SPACE}" \
            --target-update-interval "${TARGET_UPDATE_INTERVAL}" \
            --tb-enable false \
            --run-id "${RUN_ID}" \
            --phase train_b \
            --sor-save-buffer-every 1 \
            "${watch_args[@]}"
        snapshot_models "${method_dir}" after_b
        touch "${method_dir}/markers/train_b"
    fi

    if [[ ! -f "${method_dir}/markers/eval_after_b" ]]; then
        evaluate "${method}" after_b "${TASK_A}" "${exp}" "${model_dir}" "${method_port}"
        evaluate "${method}" after_b "${TASK_B}" "${exp}" "${model_dir}" "${method_port}"
        touch "${method_dir}/markers/eval_after_b"
    fi
    python "${ROOT}/scripts/continual_validation/check_acquisition.py" \
        --run-dir "${RUN_DIR}" --method "${method}" --task b \
        --min-reward-gain 0.02 --min-p95-gain 0.05 \
        --completion-tolerance 0.01 "${acquisition_mode_args[@]}"
    if [[ "${method}" == "acc" ]]; then
        local forgetting_mode_args=()
        if [[ "${REPORT_ONLY}" -eq 1 ]]; then
            forgetting_mode_args+=(--report-only)
        else
            forgetting_mode_args+=(--gate-forgetting)
        fi
        python "${ROOT}/scripts/continual_validation/analyze_forgetting.py" \
            --run-dir "${RUN_DIR}" --method acc \
            --min-reward-drop 0.10 --min-p95-worsening 0.10 \
            --completion-tolerance 0.01 "${forgetting_mode_args[@]}"
        touch "${method_dir}/markers/analysis_complete"
        [[ "${REPORT_ONLY}" -eq 1 ]] ||
            touch "${method_dir}/markers/continual_gate_pass"
    fi
    touch "${method_dir}/markers/complete"
    echo "${method} A->B curriculum complete: ${method_dir}"
}

analyze() {
    check_manifest
    local analysis_mode_args=()
    if [[ "${REPORT_ONLY}" -eq 1 ]]; then
        analysis_mode_args+=(--report-only)
    else
        analysis_mode_args+=(--gate)
    fi
    python "${ROOT}/scripts/continual_validation/analyze_forgetting.py" \
        --run-dir "${RUN_DIR}" \
        --compare acc,sor \
        --min-reward-drop 0.10 \
        --min-p95-worsening 0.10 \
        --min-forgetting-reduction 0.30 \
        --new-task-p95-tolerance 0.05 \
        --completion-tolerance 0.01 \
        "${analysis_mode_args[@]}"
}

[[ "${STAGE}" == "prepare" || "${STAGE}" == "all" ]] && prepare
[[ "${STAGE}" == "screen" || "${STAGE}" == "all" ]] && screen
[[ "${STAGE}" == "acc" || "${STAGE}" == "all" ]] && train_method acc
[[ "${STAGE}" == "sor" || "${STAGE}" == "all" ]] && train_method sor
[[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]] && analyze

echo "Continual validation stage '${STAGE}' complete: ${RUN_DIR}"
