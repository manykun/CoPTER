#!/usr/bin/env python3
"""Run isolated ACC-only reward/action-space acquisition pilots."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_aba import load_config  # noqa: E402


VARIANTS = {
    "baseline": ("multiscale", "tail_safe"),
    "weighted": ("multiscale", "weighted"),
    "action": ("factorized_interp", "tail_safe"),
    "combined": ("factorized_interp", "weighted"),
}


def atomic_yaml(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for key, value in config.items():
                if isinstance(value, bool):
                    value = str(value).lower()
                handle.write(f"{key}: {value}\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_config(base: dict, run_id: str, variant: str, epochs: int,
                 port: int, seed: int) -> dict:
    action_space, reward_profile = VARIANTS[variant]
    config = dict(base)
    config.update({
        "run_id": run_id,
        "experiment_kind": "acc_pilot",
        "route": "AAA",
        "a1_epochs": int(epochs),
        "b_epochs": 0,
        "a2_epochs": 0,
        "methods": "ACC",
        "parallel_methods": 1,
        "acc_port": int(port),
        "sor_port": int(port) + 100,
        "seed": int(seed),
        "action_space": action_space,
        "reward_profile": reward_profile,
        "shared_replay": False,
    })
    return config


def run_variant(args, base, variant, index):
    suffix = "_smoke" if args.stage == "smoke" else ""
    run_id = f"{args.run_id_prefix}{suffix}_{variant}"
    run_dir = ROOT / "experiments" / "aba" / run_id
    config_path = run_dir / "pilot_config.yaml"
    config = build_config(
        base, run_id, variant, args.epochs, args.port_base + index,
        args.seed,
    )
    if config_path.exists():
        existing = load_config(config_path)
        if existing != config:
            raise RuntimeError(
                f"existing pilot config differs for {run_id}; use a new prefix"
            )
    else:
        atomic_yaml(config_path, config)

    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_aba.py"),
        "--config", str(config_path),
        "--stage", args.stage,
        "--run-id", run_id,
        "--max-global-epoch", str(args.stop_after),
    ]
    if args.resume:
        command.append("--resume")
    print(
        f"[{variant}] action={config['action_space']} "
        f"reward={config['reward_profile']} port={config['acc_port']} "
        f"run={run_id}",
        flush=True,
    )
    subprocess.run(command, cwd=ROOT, check=True)
    return run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config", type=Path,
        default=ROOT / "configs" / "aba" / "webserver_cachefollower.yaml",
    )
    parser.add_argument(
        "--stage", choices=("prepare", "run", "analyze", "all", "smoke"),
        default="all",
    )
    parser.add_argument(
        "--variants", default="baseline,weighted,action",
        help="comma-separated subset of baseline,weighted,action,combined",
    )
    parser.add_argument("--run-id-prefix", default="acc_pilot_webserver_s1")
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Registered total budget; keep this unchanged when resuming.",
    )
    parser.add_argument(
        "--stop-after", type=int, default=50,
        help="Run only through this epoch; resume later with a larger value.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=7600)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.epochs <= 100:
        raise ValueError("--epochs must be in [1, 100]")
    if not 1 <= args.stop_after <= args.epochs:
        raise ValueError("--stop-after must be in [1, --epochs]")
    requested = [item.strip() for item in args.variants.split(",") if item.strip()]
    if not requested or any(item not in VARIANTS for item in requested):
        raise ValueError(
            "--variants must contain baseline, weighted, action, or combined"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("--variants contains duplicates")
    if args.jobs < 1:
        raise ValueError("--jobs must be positive")

    base_path = args.base_config.resolve()
    base = load_config(base_path)
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.jobs, len(requested))) as pool:
        futures = {
            pool.submit(run_variant, args, base, variant, index): variant
            for index, variant in enumerate(requested)
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append((futures[future], exc))
    if failures:
        detail = "; ".join(f"{name}: {error}" for name, error in failures)
        raise RuntimeError(f"ACC pilot failures: {detail}")
    if args.stage in ("analyze", "all", "smoke"):
        prefix = args.run_id_prefix + ("_smoke" if args.stage == "smoke" else "")
        subprocess.run([
            sys.executable,
            str(ROOT / "tools" / "analysis" / "analyze_acc_reward_action_pilot.py"),
            "--run-id-prefix", prefix,
            "--variants", ",".join(requested),
        ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
