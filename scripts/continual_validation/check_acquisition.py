#!/usr/bin/env python3
"""Measure task acquisition, optionally applying a fail-fast gate."""

import argparse
import json
from pathlib import Path

from analyze_forgetting import acquisition_passed, comparison, load_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=("acc", "sor"))
    parser.add_argument("--task", required=True, choices=("a", "b"))
    parser.add_argument("--min-reward-gain", type=float, default=0.02)
    parser.add_argument("--min-p95-gain", type=float, default=0.05)
    parser.add_argument("--completion-tolerance", type=float, default=0.01)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write measurements without assigning or enforcing PASS/FAIL",
    )
    args = parser.parse_args()
    if args.gate and args.report_only:
        parser.error("--gate and --report-only are mutually exclusive")

    manifest = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    base = args.run_dir / "eval" / args.method
    if args.task == "a":
        task_name = manifest["task_a"]
        before = load_eval(base / "initial" / task_name)
        after = load_eval(base / "after_a" / task_name)
    else:
        task_name = manifest["task_b"]
        before = load_eval(base / "after_a" / task_name)
        after = load_eval(base / "after_b" / task_name)
    result = comparison(before, after)
    result["passed"] = (
        None
        if args.report_only
        else acquisition_passed(
            result,
            args.min_reward_gain,
            args.min_p95_gain,
            args.completion_tolerance,
        )
    )
    output = {
        "evaluation_mode": "descriptive" if args.report_only else "gated",
        "method": args.method,
        "task_phase": args.task,
        "task_name": task_name,
        "thresholds": {
            "min_reward_gain": args.min_reward_gain,
            "min_p95_gain": args.min_p95_gain,
            "completion_tolerance": args.completion_tolerance,
        },
        "result": result,
    }
    path = args.run_dir / args.method / f"acquisition_{args.task}.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    status = (
        "RECORDED (descriptive, no gate)"
        if args.report_only
        else ("PASS" if result["passed"] else "FAIL")
    )
    print(f"{args.method.upper()} task {args.task.upper()} acquisition: {status}")
    print(
        f"reward_change={result['reward_change']:.4f}, "
        f"p95_worsening={result['p95_fct_worsening']:.4f}, "
        f"completion_drop={result['completion_drop']:.4f}"
    )
    if args.gate and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
