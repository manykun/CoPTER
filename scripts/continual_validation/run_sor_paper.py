#!/usr/bin/env python3
"""Matched ACC-local/SOR paper protocol over realistic WebServer -> CacheFollower."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
from pathlib import Path
import shutil
import socket
import tempfile

from run_acc_attribution import execute, fingerprint
from run_return_a import ROOT, digest, read, write
from analyze_forgetting import load_eval


METHODS = ("acc_local", "sor")
TASK_A = "realistic_webserver"
TASK_B = "realistic_cachefollower"
REPRESENTATIVE_PORTS = (323, 321, 320, 345, 346, 347)


def points(value):
    result = [int(item) for item in value.split(",") if item]
    if not result or result != sorted(set(result)) or result[0] <= 0:
        raise argparse.ArgumentTypeError("use increasing positive update counts")
    return result


def state_path(model, exp):
    return model / f"{exp}_train_state.json"


def state(model, exp):
    return read(state_path(model, exp))


def command(manifest, method, model, exp, port, watch_ports):
    schedule = manifest["epsilon_schedule"]
    mode = "ACC" if method.startswith("acc") else "SOR"
    cmd = [
        "env", f"COPTER_LOG_BASE={model.parent / 'logs' / model.name}",
        "COPTER_COMPLETE_CHECKPOINT_ONLY=1",
        "bash", str(ROOT / "run_training.sh"),
        "--exp", exp, "--mode", mode, "--model-dir", str(model),
        "--port", str(port), "--seed", str(manifest["seed"]),
        "--buffer", "400", "--epsilon-schedule", "global",
        "--eps-start", str(schedule["start"]),
        "--eps-end", str(schedule["end"]),
        "--eps-decay", str(schedule["decay_steps"]),
        "--target-update-interval", "100", "--action-space", "multiscale",
        "--acc-hidden-dims", "32,64,64,32", "--reward-profile", "tail_safe",
        "--reward-queue-lambda", "5.0", "--reward-ecn-lambda", "5.0",
        "--reward-weights", "0.50,0.30,0.20", "--tb-enable", "false",
        "--watch-ports", ",".join(map(str, watch_ports)),
        "--run-id", manifest["paper_run_id"], "--sor-save-buffer-every", "1",
    ]
    if method == "acc_local":
        cmd += ["--shared-replay", "false"]
    elif method == "acc_global":
        cmd += ["--shared-replay", "true"]
    elif method == "sor_replay_only":
        cmd += ["--sor-lambda-cons", "0", "--sor-lambda-reg", "0"]
    return cmd


def check_snapshot(model, exp, method, target):
    saved = state(model, exp)
    if int(saved["global_train_step"]) != target:
        raise ValueError(f"{method} checkpoint is at {saved['global_train_step']}, expected {target}")
    count = int(saved["node_number"])
    if method.startswith("acc"):
        required = [f"{exp}_ACC_{port}" for port in range(count)]
        required += [f"{exp}_rb_port{port}.pkl" for port in range(count)]
    else:
        required = [f"{exp}_SORACC_{port}" for port in range(count)]
        required += [f"{exp}_sor_rb_port{port}.pkl" for port in range(count)]
        required += [f"{exp}_sor_global_rb.pkl", f"{exp}_sor_rng.pkl"]
    missing = [name for name in required if not (model / name).is_file()]
    if missing:
        raise ValueError(f"incomplete {method} checkpoint: {missing[:3]}")
    if not method.startswith("acc"):
        if saved.get("action_space") != "multiscale":
            raise ValueError("SOR checkpoint does not use multiscale")
        if int(saved.get("local_replay_capacity", -1)) != 1000:
            raise ValueError("SOR local replay capacity is not 1000")
        if int(saved.get("global_replay_capacity", -1)) != 100000:
            raise ValueError("SOR global replay capacity is not 100000")


def copy_checkpoint(source, destination):
    before = fingerprint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".paper-copy-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "data"
        shutil.copytree(source, staging, ignore=shutil.ignore_patterns("*_metrics.jsonl"))
        if fingerprint(staging) != before or fingerprint(source) != before:
            raise ValueError("checkpoint changed during atomic copy")
        staging.rename(destination)


def evaluate_task(base, manifest, method, model, exp, eval_root, label, task, port, watch_ports):
    destination = eval_root / method / label / task
    marker = destination / "complete.json"
    if marker.exists():
        if read(marker)["model_hashes"] != fingerprint(model):
            raise ValueError(f"immutable evaluation {destination} has different model bytes")
        load_eval(destination)
        return
    before = fingerprint(model)
    print(f"[frozen] method={method} label={label} task={task}", flush=True)
    cmd = command(manifest, method, model, exp, port, watch_ports)
    execute(base, task, destination, cmd, [
        "--one-shot", "--eval-greedy", "--episodes", "1", "--phase", "paper_eval",
        "--eval-tag", f"{method}_{label}_{task}",
        "--watch-trace-file", str(destination / "watch_trace.jsonl"),
    ])
    if fingerprint(model) != before:
        raise ValueError("frozen evaluation modified model/replay")
    metrics_path = model / f"{exp}_metrics.jsonl"
    last = None
    with metrics_path.open() as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    if not last or not last.get("eval_greedy"):
        raise ValueError("missing frozen evaluation metrics")
    write(destination / "metrics.json", last)
    load_eval(destination)
    write(marker, {"model_hashes": before, "watch_ports": list(watch_ports)})


def train_to(base, manifest, method, model, exp, task, target, destination, port, phase):
    current = state(model, exp)
    now = int(current["global_train_step"])
    check_snapshot(model, exp, method, now)
    if now > target:
        raise ValueError("missing earlier checkpoint; refusing to train backwards")
    if now < target:
        execute(base, task, destination, command(
            manifest, method, model, exp, port, REPRESENTATIVE_PORTS
        ), ["--phase", phase, "--episodes", str(int(current["epoch"]) + 2000),
            "--target-train-steps", str(target)])
    check_snapshot(model, exp, method, target)


def bootstrap(base, manifest, method, model, exp, port):
    model.mkdir(parents=True, exist_ok=True)
    if state_path(model, exp).exists():
        check_snapshot(model, exp, method, int(state(model, exp)["global_train_step"]))
        return
    if list(model.iterdir()):
        raise ValueError(f"interrupted bootstrap in {model}")
    execute(base, TASK_A, model.parent / "train/bootstrap", command(
        manifest, method, model, exp, port, REPRESENTATIVE_PORTS
    ), ["--phase", "train_a", "--episodes", "200", "--target-train-steps", "1"])


def acquire_worker(root_string, method):
    root = Path(root_string)
    protocol = read(root / "protocol.json")
    base = Path(protocol["base"])
    manifest = protocol["manifest"]
    port = protocol["ports"][method]
    exp = f"paper_{root.name}_{method}"
    model = root / method / "a/models"
    bootstrap(base, manifest, method, model, exp, port)
    for update in protocol["a_points"]:
        snapshot = root / method / "a/checkpoints" / str(update)
        if not snapshot.exists():
            train_to(base, manifest, method, model, exp, TASK_A, update,
                     root / method / "a/train" / str(update), port, "train_a")
            copy_checkpoint(model, snapshot)
        check_snapshot(snapshot, exp, method, update)
        watched = range(448) if update == protocol["a_points"][-1] else REPRESENTATIVE_PORTS
        evaluate_task(base, manifest, method, snapshot, exp, root / "eval",
                      f"a_{update}", TASK_A, port, watched)
        if update == protocol["a_points"][-1]:
            evaluate_task(base, manifest, method, snapshot, exp, root / "eval",
                          f"a_{update}", TASK_B, port, watched)


def continue_worker(root_string, method):
    root = Path(root_string)
    protocol = read(root / "protocol.json")
    base = Path(protocol["base"])
    manifest = protocol["manifest"]
    port = protocol["ports"][method]
    exp = f"paper_{root.name}_{method}"
    a_end = protocol["a_points"][-1]
    source = root / method / "a/checkpoints" / str(a_end)
    check_snapshot(source, exp, method, a_end)
    for arm in ("aa", "ab"):
        model = root / method / arm / "models"
        if not model.exists():
            copy_checkpoint(source, model)
        provenance = root / method / arm / "fork.json"
        if not provenance.exists():
            write(provenance, {"source_hashes": fingerprint(source)})
        if read(provenance)["source_hashes"] != fingerprint(source):
            raise ValueError("after-A fork provenance changed")
    arm_models = [root / method / arm / "models" for arm in ("aa", "ab")]
    if all(int(state(model, exp)["global_train_step"]) == a_end for model in arm_models):
        if fingerprint(arm_models[0]) != fingerprint(arm_models[1]):
            raise ValueError(f"{method} AA/AB arms are not byte-identical at fork")
    for arm, task in (("aa", TASK_A), ("ab", TASK_B)):
        model = root / method / arm / "models"
        for update in protocol["b_points"]:
            target = a_end + update
            snapshot = root / method / arm / "checkpoints" / str(update)
            if not snapshot.exists():
                train_to(base, manifest, method, model, exp, task, target,
                         root / method / arm / "train" / str(update), port,
                         f"train_{arm}")
                copy_checkpoint(model, snapshot)
            check_snapshot(snapshot, exp, method, target)
            watched = range(448) if update == protocol["b_points"][-1] else REPRESENTATIVE_PORTS
            evaluate_task(base, manifest, method, snapshot, exp, root / "eval",
                          f"{arm}_{update}", TASK_A, port, watched)
            if arm == "ab":
                evaluate_task(base, manifest, method, snapshot, exp, root / "eval",
                              f"{arm}_{update}", TASK_B, port, watched)


def run_workers(root, scope, jobs):
    worker = acquire_worker if scope == "acquire" else continue_worker
    if jobs == 1:
        for method in METHODS:
            worker(str(root), method)
    else:
        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, str(root), method) for method in METHODS]
            for future in futures:
                future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("acquire", "continue"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--a-points", type=points, default=points("100,300,600"))
    parser.add_argument("--b-points", type=points, default=points("100,300"))
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--port-base", type=int, default=7200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    base, root = args.base_run_dir.resolve(), args.output_dir.resolve()
    if root == base or root in base.parents or base in root.parents:
        parser.error("base and output must be separate sibling directories")
    if not 1 <= args.port_base <= 65533:
        parser.error("port-base must leave two valid ports")
    manifest = read(base / "manifest.json")
    if manifest.get("task_a") != TASK_A or manifest.get("task_b") != TASK_B:
        parser.error(f"base must be {TASK_A} -> {TASK_B}")
    manifest = dict(manifest)
    manifest.update({
        "seed": args.seed, "task_a": TASK_A, "task_b": TASK_B,
        "buffer_kb": 400, "action_space": "multiscale", "shared_replay": False,
        "hidden_dims": [32, 64, 64, 32], "reward_profile": "tail_safe",
        "reward_queue_lambda": 5.0, "reward_ecn_lambda": 5.0,
        "reward_weights": [0.5, 0.3, 0.2], "target_update_interval": 100,
        "epsilon_schedule": {"scope": "global", "start": 1.0, "end": 0.05,
                             "decay_steps": 2500}, "paper_run_id": root.name,
    })
    protocol = {
        "format_version": 1, "base": str(base), "manifest": manifest,
        "methods": list(METHODS), "a_points": args.a_points,
        "b_points": args.b_points, "ports": {"acc_local": args.port_base,
                                                "sor": args.port_base + 1},
        "representative_ports": list(REPRESENTATIVE_PORTS),
        "task_hashes": {task: {name: digest(base / "tasks" / task / name)
                         for name in ("input.conf", "input.flow")}
                        for task in (TASK_A, TASK_B)},
        "replay": {"acc_local": 1000, "sor_local": 1000, "sor_global": 100000},
        "reward_training": "tail_safe_only",
    }
    if root.exists():
        previous = read(root / "protocol.json")
        extending_b = (args.stage == "continue" and args.resume and
                       previous["b_points"] == args.b_points[:len(previous["b_points"])] and
                       dict(previous, b_points=args.b_points) == protocol)
        continuation_exists = any(
            (root / method / arm).exists()
            for method in previous["methods"] for arm in ("aa", "ab")
        )
        extending_a = (args.stage == "acquire" and args.resume and
                       not continuation_exists and
                       previous["a_points"] == args.a_points[:len(previous["a_points"])] and
                       dict(previous, a_points=args.a_points) == protocol)
        extending = extending_a or extending_b
        if previous != protocol and not extending:
            parser.error("existing output requires identical protocol and --resume")
        if extending:
            write(root / "protocol.json", protocol)
        elif not args.resume:
            parser.error("existing output requires --resume")
    else:
        if args.stage != "acquire":
            parser.error("run acquire before continue")
        root.mkdir(parents=True)
        write(root / "protocol.json", protocol)
    with (root / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in protocol["ports"].values():
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        if args.stage == "acquire":
            initial_hashes = {}
            for method in METHODS:
                model = root / method / "initial_models"
                model.mkdir(parents=True, exist_ok=True)
                if any(not path.name.endswith("_metrics.jsonl") for path in model.iterdir()):
                    raise ValueError(f"initial model directory is not pristine: {model}")
                exp = f"paper_{root.name}_{method}"
                evaluate_task(
                    base, manifest, method, model, exp, root / "eval", "a_0",
                    TASK_A, protocol["ports"][method], REPRESENTATIVE_PORTS,
                )
                metrics = read(root / "eval" / method / "a_0" / TASK_A / "metrics.json")
                initial_hashes[method] = metrics["startup_parameter_hashes"]["policy_net"]
            matched = len(set(initial_hashes.values())) == 1
            write(root / "initialization_check.json", {
                "policy_parameter_hashes": initial_hashes,
                "matched": matched,
            })
            if not matched:
                raise ValueError("ACC and SOR initial policy tensors differ")
        run_workers(root, args.stage, args.jobs)
        write(root / f"{args.stage}_complete.json", {"complete": True})
    print(f"SOR paper {args.stage} complete: {root}")


if __name__ == "__main__":
    main()
