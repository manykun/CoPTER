#!/usr/bin/env python3
"""Resumable A1 -> B -> A2 scheduler for matched ACC/SOR experiments."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.analyze_aba_epoch import NETWORK_FIELDS, PORT_FIELDS
from tools.analysis.metrics_core import sha256_file


def parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip("'\"")


def load_config(path: Path) -> dict:
    config = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.split("#", 1)[0].strip()
        if not stripped:
            continue
        if ":" not in stripped or line[:1].isspace():
            raise ValueError(f"{path}:{number}: only top-level key: value YAML is supported")
        key, value = stripped.split(":", 1)
        config[key.strip()] = parse_scalar(value)
    return config


def absolute(path) -> Path:
    result = Path(str(path))
    return result.resolve() if result.is_absolute() else (ROOT / result).resolve()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_config(config: dict) -> None:
    required = (
        "run_id", "task_a", "task_b", "task_a_conf", "task_a_flow",
        "task_b_conf", "task_b_flow", "a1_epochs", "b_epochs", "a2_epochs",
        "methods", "acc_port", "sor_port",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing configuration keys: {', '.join(missing)}")
    invariants = {
        "epsilon_schedule": "global",
        "action_space": "multiscale",
        "reward_profile": "tail_safe",
    }
    for key, required_value in invariants.items():
        if str(config.get(key)) != required_value:
            raise ValueError(f"formal ABA requires {key}: {required_value}")
    methods = [item.strip().lower() for item in str(config["methods"]).split(",")]
    if not methods or any(item not in ("acc", "sor") for item in methods):
        raise ValueError("methods must be ACC,SOR or a subset")
    if len(set(methods)) != len(methods):
        raise ValueError("methods contains duplicates")
    if "acc" in methods and "sor" in methods and int(config["acc_port"]) == int(config["sor_port"]):
        raise ValueError("ACC and SOR ports must differ")
    for key in ("task_a_conf", "task_a_flow", "task_b_conf", "task_b_flow"):
        if not absolute(config[key]).is_file():
            raise ValueError(f"configured input does not exist: {config[key]}")


def protocol(config: dict, config_path: Path, smoke: bool) -> dict:
    epochs = {
        "a1": 2 if smoke else int(config["a1_epochs"]),
        "b": 2 if smoke else int(config["b_epochs"]),
        "a2": 2 if smoke else int(config["a2_epochs"]),
    }
    inputs = {}
    for task in ("a", "b"):
        conf = absolute(config[f"task_{task}_conf"])
        flow = absolute(config[f"task_{task}_flow"])
        inputs[task] = {
            "conf": str(conf.relative_to(ROOT)),
            "conf_sha256": sha256_file(conf),
            "flow": str(flow.relative_to(ROOT)),
            "flow_sha256": sha256_file(flow),
        }
    return {
        "schema_version": 1,
        "run_id": str(config["run_id"]),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "curriculum": ["a1", "b", "a2"],
        "epochs": epochs,
        "tasks": {"a": str(config["task_a"]), "b": str(config["task_b"])},
        "inputs": inputs,
        "methods": [item.strip().lower() for item in str(config["methods"]).split(",")],
        "seed": int(config.get("seed", 1)),
        "action_space": config["action_space"],
        "reward_profile": config["reward_profile"],
        "epsilon_schedule": config["epsilon_schedule"],
        "epsilon_decay_steps": int(config.get("epsilon_decay_steps", 2500)),
        "target_update_interval": int(config.get("target_update_interval", 100)),
        "switch_buffer_kb": int(config.get("switch_buffer_kb", 400)),
        "hidden_dims": str(config.get("acc_hidden_dims", "32,64,64,32")),
        "reward_queue_lambda": float(config.get("reward_queue_lambda", 5.0)),
        "reward_ecn_lambda": float(config.get("reward_ecn_lambda", 5.0)),
        "reward_weights": str(config.get("reward_weights", "0.50,0.30,0.20")),
        "rate_warmup_buckets": int(config.get("warmup_rate_buckets", 2)),
        "acc_shared_replay": bool(config.get("shared_replay", False)),
        "sor_local_replay_capacity": 1000,
        "sor_global_replay_capacity": 100000,
        "smoke": bool(smoke),
    }


def prepare(config: dict, config_path: Path, smoke: bool, allow_existing=True) -> tuple[Path, dict]:
    validate_config(config)
    run_dir = ROOT / "experiments" / "aba" / str(config["run_id"])
    current = protocol(config, config_path, smoke)
    protocol_path = run_dir / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        # Time and the absolute config source are descriptive; all scientific
        # inputs and budgets must remain identical on resume.
        comparable_previous = dict(previous)
        comparable_current = dict(current)
        for item in (comparable_previous, comparable_current):
            item.pop("created_at", None)
            item.pop("config", None)
        if comparable_previous != comparable_current:
            raise RuntimeError("existing protocol differs from requested inputs; refusing mutation")
    else:
        atomic_json(protocol_path, current)
    for method in current["methods"]:
        for directory in ("models", "checkpoints", "logs"):
            (run_dir / method / directory).mkdir(parents=True, exist_ok=True)
    (run_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    output_root = ROOT / "simulation" / "output" / "aba" / str(config["run_id"])
    output_root.mkdir(parents=True, exist_ok=True)
    return run_dir, current


def phase_plan(protocol_data: dict):
    global_epoch = 0
    for phase in ("a1", "b", "a2"):
        task_key = "b" if phase == "b" else "a"
        for phase_epoch in range(1, int(protocol_data["epochs"][phase]) + 1):
            global_epoch += 1
            yield phase, task_key, phase_epoch, global_epoch


OUTPUT_DIRECTIVES = {
    "TRACE_OUTPUT_FILE": ".tr",
    "FCT_OUTPUT_FILE": ".fct",
    "PFC_OUTPUT_FILE": ".pfc",
    "QLEN_MONITOR_FILE": ".queue",
    "RATE_MONITOR_FILE": ".rate",
    "THROUGHPUT_OUTPUT_FILE": ".throughput",
    "QLEN_MON_FILE": ".qlen",
}


def render_runtime_conf(source: Path, flow: Path, output_dir: Path, destination: Path, buffer_kb: int):
    replacements = {
        "FLOW_FILE": str(flow),
        "BUFFER_SIZE": str(buffer_kb),
        **{key: str(output_dir / f"result{suffix}") for key, suffix in OUTPUT_DIRECTIVES.items()},
    }
    seen = set()
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if fields and fields[0] in replacements:
            lines.append(f"{fields[0]} {replacements[fields[0]]}")
            seen.add(fields[0])
        else:
            lines.append(line)
    missing = set(replacements) - seen
    if missing:
        raise ValueError(f"source config lacks directives: {sorted(missing)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def append_rows(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))
    def row_key(row):
        base = (row["method"], int(row["global_epoch"]))
        return (
            base + (row.get("record_source"), row.get("identifier"), row.get("agent_port"))
            if "identifier" in fields else base
        )

    by_key = {row_key(row): row for row in existing}
    pending = []

    def equivalent(left, right):
        for field in fields:
            a, b = left.get(field), right.get(field)
            if a in (None, "") and b in (None, ""):
                continue
            try:
                if abs(float(a) - float(b)) <= 1e-12:
                    continue
            except (TypeError, ValueError):
                pass
            if str(a) != str(b):
                return False
        return True

    for row in rows:
        key = row_key(row)
        if key in by_key:
            if equivalent(by_key[key], row):
                continue
            raise RuntimeError(f"conflicting duplicate summary row {key} in {path}")
        by_key[key] = row
        pending.append(row)
    if not pending:
        return
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(pending)
    os.replace(temporary, path)


def ensure_exact_resume(method: str, model_dir: Path, exp_name: str, global_epoch: int):
    if global_epoch == 1:
        return
    state_path = model_dir / f"{exp_name}_train_state.json"
    if not state_path.exists():
        raise RuntimeError(f"missing train state before epoch {global_epoch}: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    node_count = int(state.get("node_number", len(state.get("replay_size_per_port", [])) or 448))
    if method == "acc":
        replay = [model_dir / f"{exp_name}_rb_port{port}.pkl" for port in range(node_count)]
        models = [model_dir / f"{exp_name}_ACC_{port}" for port in range(node_count)]
        required = models + replay + [model_dir / f"{exp_name}_rng.pkl"]
    else:
        replay = [model_dir / f"{exp_name}_sor_rb_port{port}.pkl" for port in range(node_count)]
        models = [model_dir / f"{exp_name}_SORACC_{port}" for port in range(node_count)]
        required = models + replay + [
            model_dir / f"{exp_name}_sor_global_rb.pkl",
            model_dir / f"{exp_name}_sor_rng.pkl",
        ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            f"epoch {global_epoch} would cold-start {method}: {len(missing)} snapshots missing"
        )


def ensure_epoch_ready_for_training(model_dir: Path, exp_name: str, global_epoch: int):
    """Require an unambiguous N-1 checkpoint before training epoch N.

    Completed and recoverable epochs are handled before this check.  If the
    checkpoint has already advanced to N but the raw/analysis completion proof is
    absent, rerunning would silently train the same epoch twice, so stop instead.
    """
    state_path = model_dir / f"{exp_name}_train_state.json"
    if global_epoch == 1:
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            raise RuntimeError(
                "refusing to start epoch 1 with an existing model state "
                f"(checkpoint epoch={state.get('epoch')}); use a new run-id or "
                "repair the interrupted epoch explicitly"
            )
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    checkpoint_epoch = int(state.get("epoch", -1))
    expected_epoch = global_epoch - 1
    if checkpoint_epoch != expected_epoch:
        raise RuntimeError(
            f"refusing to train epoch {global_epoch}: checkpoint epoch={checkpoint_epoch}, "
            f"expected {expected_epoch}. The epoch is incomplete or ambiguous; raw data "
            "was preserved and no cold/double training was attempted"
        )


def update_root_state(run_dir: Path, method: str, payload: dict):
    lock_path = run_dir / ".state.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = run_dir / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"methods": {}}
        state.setdefault("methods", {})[method] = payload
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, state)
        fcntl.flock(lock, fcntl.LOCK_UN)


def cleanup_raw(raw_dir: Path):
    for suffix in (".fct", ".queue", ".rate", ".pfc", ".throughput", ".qlen", ".tr"):
        for path in raw_dir.glob(f"*{suffix}"):
            path.unlink()
    trace = raw_dir / "fct_steps.csv"
    if trace.exists():
        trace.unlink()


def recoverable_agent_epoch(metrics_path: Path, raw_dir: Path, run_id: str, phase: str, epoch: int):
    if not metrics_path.exists() or not (raw_dir / "fct_steps.csv").exists():
        return None
    matches = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if (
            record.get("run_id") == run_id
            and record.get("phase") == phase
            and int(record.get("epoch", -1)) == epoch
            and not record.get("eval_greedy")
        ):
            matches.append(record)
    if len(matches) != 1:
        return None
    with (raw_dir / "fct_steps.csv").open(encoding="utf-8", newline="") as handle:
        trace = list(csv.DictReader(handle))
    if not trace or str(trace[-1].get("terminal", "")).lower() != "true":
        return None
    if any(not (raw_dir / f"result{suffix}").exists() for suffix in (".fct", ".queue", ".rate", ".throughput", ".pfc")):
        return None
    return matches[0]


def checkpoint_boundary(run_dir: Path, method: str, phase: str, model_dir: Path):
    destination = run_dir / method / "checkpoints" / phase
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(model_dir, temporary)
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)


def finalize_epoch(
    config, protocol_data, run_dir, method, phase, phase_epoch, global_epoch,
    flow, runtime_conf, raw_dir, model_dir, exp_name, summary_metrics, summary_ports,
):
    metric_row = json.loads((raw_dir / "epoch_metrics.json").read_text(encoding="utf-8"))
    if int(metric_row["global_epoch"]) != global_epoch:
        raise RuntimeError(f"analyzed epoch mismatch in {raw_dir}")
    with (raw_dir / "epoch_ports.csv").open(encoding="utf-8", newline="") as handle:
        port_rows = list(csv.DictReader(handle))
    append_rows(summary_metrics, NETWORK_FIELDS, [metric_row])
    append_rows(summary_ports, PORT_FIELDS, port_rows)
    summaries = run_dir / "summaries"
    summaries.mkdir(parents=True, exist_ok=True)
    shutil.copy2(summary_metrics, summaries / f"{method}_epoch_metrics.csv")
    shutil.copy2(summary_ports, summaries / f"{method}_epoch_ports.csv")
    state_file = model_dir / f"{exp_name}_train_state.json"
    model_state = json.loads(state_file.read_text(encoding="utf-8"))
    if int(model_state.get("epoch", -1)) != global_epoch:
        raise RuntimeError(
            f"model epoch={model_state.get('epoch')} does not match analyzed epoch={global_epoch}"
        )
    state_payload = {
        "last_complete_epoch": global_epoch,
        "phase": phase,
        "phase_epoch": phase_epoch,
        "global_train_step": metric_row.get("global_train_step"),
        "global_env_step": metric_row.get("global_env_step"),
        "epsilon": metric_row.get("epsilon"),
        "model_state_sha256": sha256_file(state_file),
        "flow_sha256": sha256_file(flow),
        "runtime_conf_sha256": sha256_file(runtime_conf),
    }
    if phase_epoch == int(protocol_data["epochs"][phase]):
        checkpoint_boundary(run_dir, method, phase, model_dir)
    elif str(config.get("retain_raw", "phase_boundaries")) == "phase_boundaries":
        cleanup_raw(raw_dir)
    atomic_json(run_dir / method / "state.json", state_payload)
    update_root_state(run_dir, method, state_payload)
    atomic_json(
        raw_dir / "complete.json",
        {**state_payload, "analysis_version": metric_row["analysis_version"]},
    )


def _method_worker_locked(config: dict, protocol_data: dict, run_dir: Path, method: str, resume: bool):
    method_upper = method.upper()
    port = int(config[f"{method}_port"])
    exp_name = f"aba_{config['run_id']}_{method}"
    model_dir = run_dir / method / "models"
    metrics_path = model_dir / f"{exp_name}_metrics.jsonl"
    output_root = ROOT / "simulation" / "output" / "aba" / str(config["run_id"]) / method
    summary_metrics = output_root / "epoch_metrics.csv"
    summary_ports = output_root / "epoch_ports.csv"
    runtime_root = ROOT / "simulation" / "mix" / "aba" / "webserver_cachefollower" / "runtime" / str(config["run_id"]) / method
    total_epochs = sum(int(protocol_data["epochs"][phase]) for phase in ("a1", "b", "a2"))

    for phase, task_key, phase_epoch, global_epoch in phase_plan(protocol_data):
        raw_dir = output_root / phase / f"epoch_{global_epoch:04d}"
        complete_path = raw_dir / "complete.json"
        if complete_path.exists():
            if not resume:
                raise RuntimeError(f"completed epoch exists without --resume: {complete_path}")
            continue
        raw_dir.mkdir(parents=True, exist_ok=True)
        source_conf = absolute(config[f"task_{task_key}_conf"])
        flow = absolute(config[f"task_{task_key}_flow"])
        runtime_conf = runtime_root / phase / f"epoch_{global_epoch:04d}.conf"
        render_runtime_conf(source_conf, flow, raw_dir, runtime_conf, int(config.get("switch_buffer_kb", 400)))
        ensure_exact_resume(method, model_dir, exp_name, global_epoch)
        if (raw_dir / "epoch_metrics.json").exists():
            if not resume:
                raise RuntimeError(f"unfinished analyzed epoch exists without --resume: {raw_dir}")
            finalize_epoch(
                config, protocol_data, run_dir, method, phase, phase_epoch, global_epoch,
                flow, runtime_conf, raw_dir, model_dir, exp_name, summary_metrics, summary_ports,
            )
            print(f"[{method}] finalized interrupted epoch {global_epoch}", flush=True)
            continue
        recovered_record = recoverable_agent_epoch(
            metrics_path, raw_dir, str(config["run_id"]), phase, global_epoch
        )
        if recovered_record is not None:
            if not resume:
                raise RuntimeError(f"completed-but-unanalyzed epoch exists without --resume: {raw_dir}")
            recovery_analyze = [
                sys.executable, str(ROOT / "tools" / "analysis" / "analyze_aba_epoch.py"),
                "--run-dir", str(run_dir), "--run-id", str(config["run_id"]),
                "--method", method, "--phase", phase, "--task", str(protocol_data["tasks"][task_key]),
                "--epoch", str(global_epoch), "--phase-epoch", str(phase_epoch),
                "--raw-dir", str(raw_dir), "--flow", str(flow), "--runtime-conf", str(runtime_conf),
                "--agent-metrics", str(metrics_path), "--metrics-offset", "0",
                "--duration-seconds", str(recovered_record.get("wall_time_seconds", 0.0)),
                "--ns3-exit-code", "0", "--agent-exit-code", "0",
                "--warmup-rate-buckets", str(config.get("warmup_rate_buckets", 2)),
            ]
            subprocess.run(recovery_analyze, cwd=ROOT, check=True)
            finalize_epoch(
                config, protocol_data, run_dir, method, phase, phase_epoch, global_epoch,
                flow, runtime_conf, raw_dir, model_dir, exp_name, summary_metrics, summary_ports,
            )
            print(f"[{method}] analyzed and finalized interrupted epoch {global_epoch}", flush=True)
            continue
        ensure_epoch_ready_for_training(model_dir, exp_name, global_epoch)
        metrics_offset = metrics_path.stat().st_size if metrics_path.exists() else 0
        log_dir = run_dir / method / "logs" / phase
        ns3_log = log_dir / f"epoch_{global_epoch:04d}_ns3.log"
        agent_log = log_dir / f"epoch_{global_epoch:04d}_agent.log"
        ns3_cmd = [
            "bash", str(ROOT / "simulation" / "run_100.sh"),
            "--config", str(runtime_conf), "--port", str(port),
            "--epoch", str(global_epoch), "--log", str(ns3_log),
        ]
        agent_cmd = [
            "bash", str(ROOT / "copter" / "run_100.sh"), "--mode", method_upper,
            "--phase", phase, "--epoch", str(global_epoch), "--port", str(port),
            "--exp-name", exp_name, "--model-dir", str(model_dir),
            "--run-id", str(config["run_id"]), "--log", str(agent_log),
            "--seed", str(config.get("seed", 1)), "--buffer-kb", str(config.get("switch_buffer_kb", 400)),
            "--action-space", str(config["action_space"]), "--hidden-dims", str(config.get("acc_hidden_dims", "32,64,64,32")),
            "--reward-profile", str(config["reward_profile"]), "--reward-queue-lambda", str(config.get("reward_queue_lambda", 5)),
            "--reward-ecn-lambda", str(config.get("reward_ecn_lambda", 5)), "--reward-weights", str(config.get("reward_weights", "0.50,0.30,0.20")),
            "--epsilon-start", str(config.get("epsilon_start", 1)), "--epsilon-end", str(config.get("epsilon_end", 0.05)),
            "--epsilon-decay-steps", str(config.get("epsilon_decay_steps", 2500)), "--epsilon-schedule", str(config["epsilon_schedule"]),
            "--target-update-interval", str(config.get("target_update_interval", 100)), "--shared-replay", str(config.get("shared_replay", False)).lower(),
            "--watch-ports", str(config.get("watch_ports", "")), "--watch-trace-file", str(raw_dir / "watch_trace.jsonl"),
            "--fct-source-file", str(raw_dir / "result.fct"), "--fct-step-trace-file", str(raw_dir / "fct_steps.csv"),
            "--tb-enable", "false", "--sor-recent-size", str(config.get("sor_recent_size", 200)),
            "--sor-boundary-size", str(config.get("sor_boundary_size", 1000)), "--sor-max-clusters", str(config.get("sor_max_clusters", 32)),
            "--sor-lambda-cons", str(config.get("sor_lambda_cons", 0.01)), "--sor-lambda-reg", str(config.get("sor_lambda_reg", 0.001)),
            "--sor-ref-update-interval", str(config.get("sor_ref_update_interval", 256)), "--sor-sync-interval", str(config.get("sor_sync_interval", 8)),
            "--sor-save-buffer-every", str(config.get("sor_save_buffer_every", 1)),
        ]
        if global_epoch > 1:
            agent_cmd.append("--resume")
        env = os.environ.copy()
        env["PYTHON_BIN"] = sys.executable
        started = time.monotonic()
        ns3 = subprocess.Popen(ns3_cmd, cwd=ROOT, env=env)
        time.sleep(float(config.get("ns3_start_wait_seconds", 3)))
        if ns3.poll() is not None:
            raise RuntimeError(f"ns-3 exited before agent start (code {ns3.returncode}); see {ns3_log}")
        agent = subprocess.Popen(agent_cmd, cwd=ROOT, env=env)
        agent_code = agent.wait()
        try:
            ns3_code = ns3.wait(timeout=float(config.get("ns3_exit_timeout_seconds", 90)))
        except subprocess.TimeoutExpired:
            ns3.terminate()
            try:
                ns3.wait(timeout=10)
            except subprocess.TimeoutExpired:
                ns3.kill()
            raise RuntimeError(f"ns-3 did not exit after agent epoch {global_epoch}")
        duration = time.monotonic() - started
        if agent_code != 0 or ns3_code != 0:
            raise RuntimeError(
                f"{method} epoch {global_epoch} failed: agent={agent_code}, ns3={ns3_code}; "
                f"see {agent_log} and {ns3_log}"
            )
        analyze_cmd = [
            sys.executable, str(ROOT / "tools" / "analysis" / "analyze_aba_epoch.py"),
            "--run-dir", str(run_dir), "--run-id", str(config["run_id"]),
            "--method", method, "--phase", phase, "--task", str(protocol_data["tasks"][task_key]),
            "--epoch", str(global_epoch), "--phase-epoch", str(phase_epoch),
            "--raw-dir", str(raw_dir), "--flow", str(flow), "--runtime-conf", str(runtime_conf),
            "--agent-metrics", str(metrics_path), "--metrics-offset", str(metrics_offset),
            "--duration-seconds", str(duration), "--ns3-exit-code", str(ns3_code),
            "--agent-exit-code", str(agent_code), "--warmup-rate-buckets", str(config.get("warmup_rate_buckets", 2)),
        ]
        subprocess.run(analyze_cmd, cwd=ROOT, check=True)
        finalize_epoch(
            config, protocol_data, run_dir, method, phase, phase_epoch, global_epoch,
            flow, runtime_conf, raw_dir, model_dir, exp_name, summary_metrics, summary_ports,
        )
        print(f"[{method}] epoch {global_epoch}/{total_epochs} complete ({phase} {phase_epoch})", flush=True)
    return method


def method_worker(config: dict, protocol_data: dict, run_dir: Path, method: str, resume: bool):
    """Run one method while preventing duplicate schedulers for the same run."""
    lock_path = run_dir / method / ".run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another ABA scheduler already owns {method} for run "
                f"{protocol_data['run_id']}"
            ) from exc
        try:
            return _method_worker_locked(
                config, protocol_data, run_dir, method, resume
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run_methods(config: dict, protocol_data: dict, run_dir: Path, resume: bool):
    methods = protocol_data["methods"]
    jobs = min(int(config.get("parallel_methods", 2)), len(methods))
    if jobs <= 1:
        for method in methods:
            method_worker(config, protocol_data, run_dir, method, resume)
        return
    errors = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(method_worker, config, protocol_data, run_dir, method, resume): method
            for method in methods
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append((futures[future], exc))
    if errors:
        details = "; ".join(f"{method}: {exc}" for method, exc in errors)
        raise RuntimeError(f"ABA method failures: {details}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "run", "analyze", "all", "smoke"), default="all")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = absolute(args.config)
    config = load_config(config_path)
    if args.run_id:
        config["run_id"] = args.run_id
    smoke = args.stage == "smoke"
    if smoke and not args.run_id:
        config["run_id"] = f"{config['run_id']}_smoke"
    run_dir, protocol_data = prepare(config, config_path, smoke)
    if args.stage == "prepare":
        print(f"ABA protocol prepared: {run_dir}")
        return
    if args.stage in ("run", "all", "smoke"):
        run_methods(config, protocol_data, run_dir, args.resume)
    if args.stage in ("analyze", "all", "smoke"):
        subprocess.run(
            [sys.executable, str(ROOT / "tools" / "analysis" / "build_aba_report.py"), "--run-dir", str(run_dir)],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
