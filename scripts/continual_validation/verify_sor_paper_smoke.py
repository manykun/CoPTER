#!/usr/bin/env python3
"""Fail-fast checks for the registered 20+20 update SOR paper smoke run."""

import argparse
from pathlib import Path

from run_return_a import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve(); protocol = read(root / "protocol.json")
    checks = []
    for method in protocol["methods"]:
        exp = f"paper_{root.name}_{method}"
        a = read(root / method / "a/checkpoints/20" / f"{exp}_train_state.json")
        ab = read(root / method / "ab/checkpoints/20" / f"{exp}_train_state.json")
        checks += [
            (f"{method}: A update count", a["global_train_step"] == 20),
            (f"{method}: AB cumulative update count", ab["global_train_step"] == 40),
            (f"{method}: global environment clock continues", ab["global_env_step"] > a["global_env_step"]),
            (f"{method}: epsilon does not restart", ab["epsilon"] <= a["epsilon"]),
            (f"{method}: no premature target sync", ab.get("target_update_count", 0) == 0),
        ]
        if method == "sor":
            checks += [
                ("sor: multiscale action space", ab.get("action_space") == "multiscale"),
                ("sor: local replay cap", ab.get("local_replay_capacity") == 1000 and
                 max(ab.get("replay_size_per_port", [0])) <= 1000),
                ("sor: global replay cap", ab.get("global_replay_capacity") == 100000 and
                 ab.get("global_replay_size", 0) <= 100000),
            ]
    initialization = read(root / "initialization_check.json")
    checks.append(("ACC/SOR initial policy tensors match", initialization["matched"]))
    analysis = read(root / "analysis.json")
    for row in analysis["resources"]:
        checks.append((f"{row['method']}: no Q inflation warning",
                       int(row["q_inflation_events"]) == 0))
    failed = [name for name, passed in checks if not passed]
    lines = ["# SOR paper smoke report", "", "| Check | Result |", "|---|---:|"]
    lines += [f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in checks]
    (root / "SMOKE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise SystemExit("Smoke checks failed: " + "; ".join(failed))
    print(f"Smoke checks passed: {root / 'SMOKE_REPORT.md'}")


if __name__ == "__main__":
    main()
