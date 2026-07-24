#!/usr/bin/env python3
"""Gate a cheap fixed-action screen before continual-learning training."""

import argparse
import csv
import json
from pathlib import Path

from analyze_forgetting import load_eval, summarize


ACTIONS = ("aggressive", "balanced", "permissive")


def relative_spread(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return 0.0
    return (max(values) - min(values)) / max(abs(max(values)), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--min-reward-spread", type=float, default=0.05)
    parser.add_argument("--min-p95-spread", type=float, default=0.05)
    parser.add_argument("--completion-tolerance", type=float, default=0.01)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    tasks = (manifest["task_a"], manifest["task_b"])
    decision = {"tasks": {}}
    rows = []
    for task in tasks:
        measurements = {}
        for action in ACTIONS:
            result = summarize(
                load_eval(args.run_dir / "screen" / task / action)
            )
            measurements[action] = result
            rows.append({"task": task, "action": action, **result})
        rewards = [item["reward"] for item in measurements.values()]
        p95s = [item["p95_fct_us"] for item in measurements.values()]
        completions = [
            item["completion_ratio"] for item in measurements.values()
        ]
        best_action = max(
            measurements,
            key=lambda name: (
                measurements[name]["reward"]
                if measurements[name]["reward"] is not None
                else float("-inf")
            ),
        )
        best_completion = max(completions)
        completion_safe = (
            measurements[best_action]["completion_ratio"]
            >= best_completion - args.completion_tolerance
        )
        reward_spread = relative_spread(rewards)
        p95_spread = relative_spread(p95s)
        passed = (
            reward_spread >= args.min_reward_spread
            and p95_spread >= args.min_p95_spread
            and completion_safe
        )
        decision["tasks"][task] = {
            "passed": passed,
            "best_reward_action": best_action,
            "reward_spread": reward_spread,
            "p95_spread": p95_spread,
            "best_action_completion_safe": completion_safe,
            "measurements": measurements,
        }

    best_actions = [
        decision["tasks"][task]["best_reward_action"] for task in tasks
    ]
    decision["different_best_actions"] = len(set(best_actions)) == len(tasks)
    decision["passed"] = (
        all(decision["tasks"][task]["passed"] for task in tasks)
        and decision["different_best_actions"]
    )

    screen_dir = args.run_dir / "screen"
    with (screen_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (screen_dir / "analysis.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    lines = [
        "# Controlled conflict screen",
        "",
        f"- Overall: **{'PASS' if decision['passed'] else 'FAIL'}**",
        f"- Different reward-best actions: **{decision['different_best_actions']}**",
        "",
        "| Task | Best action | Reward spread | p95 spread | Completion safe | Gate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        item = decision["tasks"][task]
        lines.append(
            f"| {task} | {item['best_reward_action']} | "
            f"{item['reward_spread']:.2%} | {item['p95_spread']:.2%} | "
            f"{item['best_action_completion_safe']} | "
            f"{'PASS' if item['passed'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "This screen validates action sensitivity and objective conflict. "
            "It does not by itself demonstrate learning or forgetting.",
            "",
        ]
    )
    (screen_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Conflict-screen analysis written to {screen_dir / 'REPORT.md'}")
    print(f"Selected gate: {'PASS' if decision['passed'] else 'FAIL'}")
    if args.gate and not decision["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
