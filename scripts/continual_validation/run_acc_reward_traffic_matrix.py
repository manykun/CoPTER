#!/usr/bin/env python3
"""Bounded-parallel, isolated 50-epoch ACC reward x traffic screen."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_aba import load_config  # noqa: E402
from run_acc_reward_action_pilot import atomic_yaml, build_config  # noqa: E402

TASKS = {
    "webserver": ("configs/aba/webserver_cachefollower.yaml", "a", "A"),
    "cachefollower": ("configs/aba/webserver_cachefollower.yaml", "b", "B"),
    "hadoop": ("configs/aba/hadoop_alistorage.yaml", "a", "C"),
    "alistorage": ("configs/aba/hadoop_alistorage.yaml", "b", "D"),
}
REWARDS = ("tail5", "tail1", "weighted")


def selection(value: str, allowed, label: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names or len(names) != len(set(names)) or any(name not in allowed for name in names):
        raise ValueError(f"{label} must be a unique comma-separated subset of {','.join(allowed)}")
    return names


def task_base(task: str) -> dict:
    filename, slot, symbol = TASKS[task]
    base = load_config(ROOT / filename)
    if slot == "b":
        for key in ("task", "conf", "flow"):
            base[f"task_a{'' if key == 'task' else '_' + key}"] = base[
                f"task_b{'' if key == 'task' else '_' + key}"
            ]
    # This is a single-task acquisition run: both recorded task inputs point
    # to the same canonical source.  No B-phase training is scheduled.
    for key in ("task", "conf", "flow"):
        suffix = "" if key == "task" else f"_{key}"
        base[f"task_b{suffix}"] = base[f"task_a{suffix}"]
    base["route"] = symbol * 3
    return base


def cell_config(task: str, reward: str, run_id: str, epochs: int,
                port: int, seed: int) -> dict:
    base = task_base(task)
    # Reuse the completed credit study's 4-step attribution and stratified
    # local replay in every cell.  Reward formula is the only within-task
    # treatment; action space and all learner settings are shared.
    config = build_config(base, run_id, "credit_control", epochs, port, seed)
    config["watch_ports"] = ",".join(dict.fromkeys(
        [part.strip() for part in str(base.get("watch_ports", "")).split(",") if part.strip()]
        + ["349"]
    ))
    config["route"] = base["route"]
    if reward == "tail1":
        config["reward_queue_lambda"] = 1.0
    elif reward == "weighted":
        config["reward_profile"] = "weighted"
    return config


def plan_for(args, tasks: list[str], rewards: list[str], prefix: str) -> dict:
    cells = []
    for task in tasks:
        for reward in rewards:
            # Stable full-matrix coordinates keep a resumed subset on the
            # original port, even if CLI ordering changes.
            index = list(TASKS).index(task) * len(REWARDS) + REWARDS.index(reward)
            run_id = f"{prefix}_{task}_{reward}"
            config = cell_config(task, reward, run_id, args.epochs,
                                 args.port_base + index, args.seed)
            cells.append({"task": task, "reward": reward, "run_id": run_id,
                          "port": config["acc_port"], "config": config})
    return {
        "schema_version": 1,
        "prefix": prefix,
        "tasks": tasks,
        "rewards": rewards,
        "epochs": 2 if args.stage == "smoke" else args.epochs,
        "registered_epochs": args.epochs,
        "seed": args.seed,
        "port_base": args.port_base,
        "cells": cells,
    }


def prepare_plan(args, plan: dict) -> Path:
    plan_dir = ROOT / "experiments" / "aba" / f"{plan['prefix']}_matrix"
    plan_path = plan_dir / "plan.json"
    if args.stage in ("frozen", "analyze") and not plan_path.exists():
        raise RuntimeError(f"missing registered matrix plan: {plan_path}")
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding="utf-8"))
        if previous != plan:
            raise RuntimeError(f"existing matrix plan differs: {plan_path}; choose a new prefix")
    else:
        plan_dir.mkdir(parents=True, exist_ok=True)
        temporary = plan_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(plan_path)
    for cell in plan["cells"]:
        run_dir = ROOT / "experiments" / "aba" / cell["run_id"]
        config_path = run_dir / "pilot_config.yaml"
        if args.stage in ("frozen", "analyze") and not config_path.exists():
            raise RuntimeError(f"missing trained-cell configuration: {config_path}")
        if config_path.exists():
            if load_config(config_path) != cell["config"]:
                raise RuntimeError(f"existing cell configuration differs: {config_path}")
        else:
            atomic_yaml(config_path, cell["config"])
    return plan_dir


def run_cell(cell: dict, stage: str, resume: bool) -> None:
    run_dir = ROOT / "experiments" / "aba" / cell["run_id"]
    command = [sys.executable, str(SCRIPT_DIR / "run_aba.py"),
               "--config", str(run_dir / "pilot_config.yaml"),
               "--stage", "smoke" if stage == "smoke" else "prepare" if stage == "prepare" else "all",
               "--run-id", cell["run_id"]]
    if stage == "smoke":
        command.extend(("--max-global-epoch", "2"))
    if resume:
        command.append("--resume")
    log = run_dir / "matrix_driver.log"
    print(f"[{cell['task']}/{cell['reward']}] start port={cell['port']} log={log}", flush=True)
    with log.open("a" if resume else "w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{cell['run_id']} failed (exit {result.returncode}); inspect {log}")
    print(f"[{cell['task']}/{cell['reward']}] complete", flush=True)


def frozen_port(cell: dict, eval_port_base: int) -> int:
    index = list(TASKS).index(cell["task"]) * len(REWARDS) + REWARDS.index(cell["reward"])
    return eval_port_base + index


def check_frozen_ports(plan: dict, eval_port_base: int) -> None:
    for cell in plan["cells"]:
        port = frozen_port(cell, eval_port_base)
        output = ROOT / "experiments" / "aba" / f"{cell['run_id']}_diagnostics"
        manifest = output / "protocol.json"
        if output.exists():
            if not manifest.is_file():
                raise RuntimeError(f"existing diagnostic directory lacks protocol: {output}")
            registered_port = json.loads(manifest.read_text(encoding="utf-8")).get("port")
            if registered_port != port:
                raise RuntimeError(
                    f"{output} is registered for evaluation port {registered_port}, not {port}; "
                    "preserve and archive the incomplete directory before changing ports"
                )
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(
                    f"evaluation port {port} is unavailable for {cell['run_id']}; "
                    "choose a free --eval-port-base without changing --port-base"
                ) from exc


def frozen_cell(cell: dict, eval_port_base: int, repeats: int) -> None:
    run_dir = ROOT / "experiments" / "aba" / cell["run_id"]
    output = run_dir.with_name(run_dir.name + "_diagnostics")
    command = [sys.executable, str(SCRIPT_DIR / "run_acc_pilot_diagnostics.py"),
               "--run-dir", str(run_dir), "--output-dir", str(output),
               "--stage", "frozen", "--repeats", str(repeats),
               "--port", str(frozen_port(cell, eval_port_base))]
    if output.exists():
        command.append("--resume")
    log = run_dir / "frozen_driver.log"
    print(f"[{cell['task']}/{cell['reward']}] frozen evaluation log={log}", flush=True)
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"frozen evaluation failed for {cell['run_id']}; inspect {log}")


def analyze(plan_dir: Path, require_frozen: bool) -> None:
    command = [sys.executable,
               str(ROOT / "tools" / "analysis" / "analyze_acc_reward_traffic_matrix.py"),
               "--plan", str(plan_dir / "plan.json")]
    if require_frozen:
        command.append("--require-frozen")
    subprocess.run(command, cwd=ROOT, check=True)


def check_training_ports(plan: dict) -> None:
    for cell in plan["cells"]:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", int(cell["port"])))
            except OSError as exc:
                raise RuntimeError(
                    f"training port {cell['port']} is unavailable for {cell['run_id']}"
                ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "train", "frozen", "analyze", "all"), default="all")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--rewards", default=",".join(REWARDS))
    parser.add_argument("--run-id-prefix", default="acc_reward_traffic_matrix_s1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=7900)
    parser.add_argument("--eval-port-base", type=int,
                        help="frozen evaluation ports; defaults to --port-base + 100 and is not part of the training plan")
    parser.add_argument("--jobs", type=int, default=3,
                        help="maximum concurrent NS-3/ACC pairs (not total cells)")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    tasks = selection(args.tasks, TASKS, "--tasks")
    rewards = selection(args.rewards, REWARDS, "--rewards")
    if args.epochs != 50:
        parser.error("this registered screen requires --epochs 50")
    if not 1 <= args.jobs <= 4 or args.repeats < 2:
        parser.error("--jobs must be in [1,4] and --repeats must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id_prefix):
        parser.error("--run-id-prefix must be a simple alphanumeric/hyphen/underscore name")
    if args.port_base < 1024 or args.port_base + 11 > 65535:
        parser.error("--port-base is outside the available TCP port range")
    eval_port_base = args.eval_port_base if args.eval_port_base is not None else args.port_base + 100
    if eval_port_base < 1024 or eval_port_base + 11 > 65535:
        parser.error("--eval-port-base is outside the available TCP port range")
    if args.min_free_gb < 0:
        parser.error("--min-free-gb cannot be negative")
    prefix = args.run_id_prefix + ("_smoke" if args.stage == "smoke" else "")
    plan = plan_for(args, tasks, rewards, prefix)
    if args.stage in ("prepare", "smoke", "train", "all"):
        free_gb = shutil.disk_usage(ROOT).free / 2**30
        if free_gb < args.min_free_gb:
            parser.error(f"only {free_gb:.1f} GiB free; minimum is {args.min_free_gb:g} GiB")
    if args.stage in ("smoke", "train", "all"):
        check_training_ports(plan)
    if args.stage in ("frozen", "all"):
        check_frozen_ports(plan, eval_port_base)
    plan_dir = prepare_plan(args, plan)
    if args.stage in ("prepare", "smoke", "train", "all"):
        failures = []
        with ThreadPoolExecutor(max_workers=min(args.jobs, len(plan["cells"]))) as pool:
            futures = {pool.submit(run_cell, cell, args.stage, args.resume): cell
                       for cell in plan["cells"]}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(str(exc))
        if failures:
            raise RuntimeError("matrix cells failed: " + "; ".join(failures))
    if args.stage in ("frozen", "all"):
        # These evaluations are deliberately serialized: each writes a large
        # raw trace and multiple NS-3 processes can overload a shared host.
        for cell in plan["cells"]:
            frozen_cell(cell, eval_port_base, args.repeats)
    if args.stage in ("train", "frozen", "analyze", "all"):
        analyze(plan_dir, args.stage in ("frozen", "all"))
    print(f"matrix stage {args.stage} complete: {plan_dir}", flush=True)


if __name__ == "__main__":
    main()
