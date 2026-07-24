#!/usr/bin/env bash
#
# Reproducible A->B continual-learning experiment for ACC and SOR.
#
# The same model and replay memory continue across task phases.  Evaluation is
# greedy and frozen.  ACC and SOR receive identical traffic, action space,
# network width, phase-local epsilon schedule, switch buffer and optimizer-
# update budget.  Task-specific rewards deliberately create a controlled
# non-stationary objective; this is not claimed to be a natural traffic shift.

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
EPS_DECAY=2500
ACC_HIDDEN_DIMS="32,64,64,32"
TASK_A_REWARD_WEIGHTS="0.25,0.55,0.20"
TASK_B_REWARD_WEIGHTS="0.70,0.15,0.15"
KMIN_RANGE="20000,50000"
KMAX_RANGE="50000,100000"
BASELINE_STOP_TIME="4.00"
PORT=5756
MAX_FLOWS=0
SMOKE=0
RESUME=0

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
  --phase-epochs N       Safety cap; update count is the actual budget
  --buffer-kb N
  --eps-decay N
  --acc-hidden-dims CSV
  --task-a-reward-weights CSV
  --task-b-reward-weights CSV
  --port N
  --smoke
  --resume
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
        --eps-decay) EPS_DECAY="$2"; shift 2 ;;
        --acc-hidden-dims) ACC_HIDDEN_DIMS="$2"; shift 2 ;;
        --task-a-reward-weights) TASK_A_REWARD_WEIGHTS="$2"; shift 2 ;;
        --task-b-reward-weights) TASK_B_REWARD_WEIGHTS="$2"; shift 2 ;;
        --reward-weights)
            echo "--reward-weights was replaced by task-specific reward options." >&2
            exit 2
            ;;
        --kmin-range) KMIN_RANGE="$2"; shift 2 ;;
        --kmax-range) KMAX_RANGE="$2"; shift 2 ;;
        --baseline-stop-time) BASELINE_STOP_TIME="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --smoke) SMOKE=1; shift ;;
        --resume) RESUME=1; shift ;;
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
python - "${TASK_A_REWARD_WEIGHTS}" "${TASK_B_REWARD_WEIGHTS}" <<'PY'
import math
import sys

for label, raw in zip(("task A", "task B"), sys.argv[1:]):
    try:
        weights = [float(value) for value in raw.split(",")]
    except ValueError as exc:
        raise SystemExit(f"{label} reward weights are invalid: {exc}")
    if (
        len(weights) != 3
        or any(value < 0 for value in weights)
        or not math.isclose(sum(weights), 1.0, abs_tol=1e-6)
    ):
        raise SystemExit(
            f"{label} reward weights must be three non-negative values "
            f"summing to one: {raw}"
        )
PY

if [[ "${SMOKE}" -eq 1 ]]; then
    PHASE_EPOCHS=3
    UPDATES_PER_TASK=10
    MAX_FLOWS=2000
fi

RUN_DIR="${ROOT}/experiments/continual_validation/${RUN_ID}"
MANIFEST="${RUN_DIR}/manifest.json"
FLOW_DIR="${ROOT}/simulation/mix/acc_validation"
OUTPUT_DIR="${ROOT}/simulation/output/acc_validation"
mkdir -p "${RUN_DIR}"

manifest_json() {
    python - "${RUN_ID}" "${TASK_A}" "${TASK_B}" "${SEED}" \
        "${BUFFER_KB}" "${PHASE_EPOCHS}" "${UPDATES_PER_TASK}" "${EPS_DECAY}" \
        "${ACC_HIDDEN_DIMS}" "${TASK_A_REWARD_WEIGHTS}" \
        "${TASK_B_REWARD_WEIGHTS}" "${KMIN_RANGE}" \
        "${KMAX_RANGE}" "${BASELINE_STOP_TIME}" "${MAX_FLOWS}" <<'PY'
import json
import sys

keys = (
    "run_id", "task_a", "task_b", "seed", "buffer_kb", "phase_epochs",
    "updates_per_task", "epsilon_decay_steps", "hidden_dims",
    "task_a_reward_weights", "task_b_reward_weights", "kmin_range",
    "kmax_range", "simulator_stop_time", "max_flows",
)
values = sys.argv[1:]
record = dict(zip(keys, values))
for key in (
    "seed", "buffer_kb", "phase_epochs", "updates_per_task",
    "epsilon_decay_steps", "max_flows",
):
    record[key] = int(record[key])
record["methods"] = ["acc", "sor"]
record["curriculum"] = [record["task_a"], record["task_b"]]
record["evaluation"] = {
    "greedy": True,
    "updates": False,
    "old_task_comparison": ["after_a", "after_b"],
    "new_task_acquisition": ["after_a", "after_b"],
}
record["experiment_type"] = "controlled_nonstationary_objective"
record["epsilon_schedule"] = "reset_per_task"
print(json.dumps(record, indent=2, sort_keys=True))
PY
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

prepare() {
    if [[ -f "${MANIFEST}" ]]; then
        if [[ "${RESUME}" -ne 1 ]]; then
            echo "Run already exists: ${RUN_DIR}" >&2
            echo "Use a new --run-id, or pass --resume only when arguments are unchanged." >&2
            return 1
        fi
        check_manifest
        validate_configs
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
    manifest_json > "${MANIFEST}.tmp"
    mv "${MANIFEST}.tmp" "${MANIFEST}"
    validate_configs
    echo "Continual-learning manifest written to ${MANIFEST}"
}

copy_eval_outputs() {
    local method="$1" phase="$2" task="$3" exp="$4" model_dir="$5"
    local name="${task}_seed${SEED}"
    local destination="${RUN_DIR}/eval/${method}/${phase}/${task}"
    local base="${OUTPUT_DIR}/${name}"
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
    cp "${FLOW_DIR}/${name}.flow" "${destination}/input.flow"
    cp "${FLOW_DIR}/${name}.conf" "${destination}/input.conf"
    cp "${FLOW_DIR}/${name}.meta" "${destination}/input.meta"
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

reward_for_task() {
    local task="$1"
    if [[ "${task}" == "${TASK_A}" ]]; then
        echo "${TASK_A_REWARD_WEIGHTS}"
    elif [[ "${task}" == "${TASK_B}" ]]; then
        echo "${TASK_B_REWARD_WEIGHTS}"
    else
        echo "Unknown task: ${task}" >&2
        return 1
    fi
}

evaluate() {
    local method="$1" phase="$2" task="$3" exp="$4" model_dir="$5" port="$6"
    local name="${task}_seed${SEED}"
    local base="${OUTPUT_DIR}/${name}"
    local reward_weights
    reward_weights="$(reward_for_task "${task}")"
    rm -f "${base}".*
    echo "[${method}] frozen evaluation phase=${phase} task=${task}"
    bash "${ROOT}/run_training.sh" \
        --one-shot \
        --eval-greedy \
        --eval-tag "${phase}_${task}" \
        --config "simulation/mix/acc_validation/${name}.conf" \
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
        --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
        --reward-weights "${reward_weights}" \
        --tb-enable false \
        --run-id "${RUN_ID}" \
        --phase "${phase}"
    copy_eval_outputs "${method}" "${phase}" "${task}" "${exp}" "${model_dir}"
}

copy_screen_outputs() {
    local task="$1" label="$2" exp="$3" model_dir="$4"
    local name="${task}_seed${SEED}"
    local destination="${RUN_DIR}/screen/${task}/${label}"
    local base="${OUTPUT_DIR}/${name}"
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
    cp "${FLOW_DIR}/${name}.flow" "${destination}/input.flow"
    cp "${FLOW_DIR}/${name}.conf" "${destination}/input.conf"
    cp "${FLOW_DIR}/${name}.meta" "${destination}/input.meta"
    tail -n 1 "${metrics}" > "${destination}/metrics.json"
}

screen() {
    check_manifest
    validate_configs
    local screen_dir="${RUN_DIR}/screen"
    local model_dir="${screen_dir}/models"
    local exp="continual_${RUN_ID}_screen_s${SEED}"
    local task label action name base reward_weights
    mkdir -p "${model_dir}"
    for task in "${TASK_A}" "${TASK_B}"; do
        reward_weights="$(reward_for_task "${task}")"
        for specification in \
            "aggressive:0,0,9" \
            "balanced:2,1,4" \
            "permissive:5,3,0"; do
            label="${specification%%:*}"
            action="${specification#*:}"
            [[ -s "${screen_dir}/${task}/${label}/metrics.json" ]] && continue
            name="${task}_seed${SEED}"
            base="${OUTPUT_DIR}/${name}"
            rm -f "${base}".*
            echo "[screen] task=${task} action=${label}(${action}) reward=${reward_weights}"
            bash "${ROOT}/run_training.sh" \
                --one-shot --eval-greedy \
                --force-action "${action}" \
                --eval-tag "screen_${task}_${label}" \
                --config "simulation/mix/acc_validation/${name}.conf" \
                --exp "${exp}" --mode ACC --port "${PORT}" --seed "${SEED}" \
                --buffer "${BUFFER_KB}" --model-dir "${model_dir}" --episodes 1 \
                --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
                --reward-weights "${reward_weights}" \
                --tb-enable false --run-id "${RUN_ID}" --phase screen
            copy_screen_outputs "${task}" "${label}" "${exp}" "${model_dir}"
        done
    done
    python "${ROOT}/scripts/continual_validation/analyze_conflict.py" \
        --run-dir "${RUN_DIR}" \
        --min-reward-spread 0.05 \
        --min-p95-spread 0.05 \
        --completion-tolerance 0.01 \
        --gate
    mkdir -p "${screen_dir}/markers"
    touch "${screen_dir}/markers/pass"
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
    [[ "${method}" == "sor" ]] && method_port=$((PORT + 100))

    check_manifest
    validate_configs
    [[ -f "${RUN_DIR}/screen/markers/pass" ]] || {
        echo "Conflict screen has not passed. Run --stage screen first." >&2
        return 1
    }
    if [[ "${method}" == "sor" && ! -f "${RUN_DIR}/acc/markers/continual_gate_pass" ]]; then
        echo "ACC has not demonstrated acquisition plus forgetting; SOR is premature." >&2
        return 1
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
        echo "[${method}] train task A=${TASK_A}, target updates=${UPDATES_PER_TASK}"
        bash "${ROOT}/run_training.sh" \
            --config "simulation/mix/acc_validation/${TASK_A}_seed${SEED}.conf" \
            --exp "${exp}" \
            --mode "${method^^}" \
            --port "${method_port}" \
            --seed "${SEED}" \
            --buffer "${BUFFER_KB}" \
            --model-dir "${model_dir}" \
            --episodes "${PHASE_EPOCHS}" \
            --target-train-steps "${UPDATES_PER_TASK}" \
            --eps-start 1.0 \
            --eps-end 0.05 \
            --eps-decay "${EPS_DECAY}" \
            --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
            --reward-weights "${TASK_A_REWARD_WEIGHTS}" \
            --tb-enable false \
            --run-id "${RUN_ID}" \
            --phase train_a \
            --sor-save-buffer-every 1
        snapshot_models "${method_dir}" after_a
        touch "${method_dir}/markers/train_a"
    fi

    if [[ ! -f "${method_dir}/markers/eval_after_a" ]]; then
        evaluate "${method}" after_a "${TASK_A}" "${exp}" "${model_dir}" "${method_port}"
        evaluate "${method}" after_a "${TASK_B}" "${exp}" "${model_dir}" "${method_port}"
        touch "${method_dir}/markers/eval_after_a"
    fi
    python "${ROOT}/scripts/continual_validation/check_acquisition.py" \
        --run-dir "${RUN_DIR}" --method "${method}" --task a \
        --min-reward-gain 0.02 --min-p95-gain 0.05 \
        --completion-tolerance 0.01 --gate

    if [[ ! -f "${method_dir}/markers/train_b" ]]; then
        local target_epoch=$((PHASE_EPOCHS * 2))
        local target_updates=$((UPDATES_PER_TASK * 2))
        echo "[${method}] continue same model on task B=${TASK_B}, target updates=${target_updates}"
        bash "${ROOT}/run_training.sh" \
            --config "simulation/mix/acc_validation/${TASK_B}_seed${SEED}.conf" \
            --exp "${exp}" \
            --mode "${method^^}" \
            --port "${method_port}" \
            --seed "${SEED}" \
            --buffer "${BUFFER_KB}" \
            --model-dir "${model_dir}" \
            --episodes "${target_epoch}" \
            --target-train-steps "${target_updates}" \
            --eps-start 1.0 \
            --eps-end 0.05 \
            --eps-decay "${EPS_DECAY}" \
            --acc-hidden-dims "${ACC_HIDDEN_DIMS}" \
            --reward-weights "${TASK_B_REWARD_WEIGHTS}" \
            --tb-enable false \
            --run-id "${RUN_ID}" \
            --phase train_b \
            --sor-save-buffer-every 1
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
        --completion-tolerance 0.01 --gate
    if [[ "${method}" == "acc" ]]; then
        python "${ROOT}/scripts/continual_validation/analyze_forgetting.py" \
            --run-dir "${RUN_DIR}" --method acc \
            --min-reward-drop 0.10 --min-p95-worsening 0.10 \
            --completion-tolerance 0.01 --gate-forgetting
        touch "${method_dir}/markers/continual_gate_pass"
    fi
    touch "${method_dir}/markers/complete"
    echo "${method} A->B curriculum complete: ${method_dir}"
}

analyze() {
    check_manifest
    python "${ROOT}/scripts/continual_validation/analyze_forgetting.py" \
        --run-dir "${RUN_DIR}" \
        --compare acc,sor \
        --min-reward-drop 0.10 \
        --min-p95-worsening 0.10 \
        --min-forgetting-reduction 0.30 \
        --new-task-p95-tolerance 0.05 \
        --completion-tolerance 0.01
}

[[ "${STAGE}" == "prepare" || "${STAGE}" == "all" ]] && prepare
[[ "${STAGE}" == "screen" || "${STAGE}" == "all" ]] && screen
[[ "${STAGE}" == "acc" || "${STAGE}" == "all" ]] && train_method acc
[[ "${STAGE}" == "sor" || "${STAGE}" == "all" ]] && train_method sor
[[ "${STAGE}" == "analyze" || "${STAGE}" == "all" ]] && analyze

echo "Continual validation stage '${STAGE}' complete: ${RUN_DIR}"
