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
        runs = {
            action: load_eval(args.run_dir / "screen" / task / action)
            for action in ACTIONS
        }
        common_flows = set.intersection(
            *(set(run["flows"]) for run in runs.values())
        )
        measurements = {}
        for action in ACTIONS:
            # Completion uses all offered flows; latency statistics use the
            # identical flow intersection across every fixed action so a
            # policy cannot look better merely because hard flows did not
            # finish.
            result = summarize(runs[action], common_flows)
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
        safe_actions = [
            name
            for name, item in measurements.items()
            if item["completion_ratio"]
            >= best_completion - args.completion_tolerance
        ]
        best_safe_p95_action = min(
            safe_actions,
            key=lambda name: measurements[name]["p95_fct_us"],
        )
        completion_safe = (
            measurements[best_action]["completion_ratio"]
            >= best_completion - args.completion_tolerance
        )
        reward_aligned = best_action == best_safe_p95_action
        reward_spread = relative_spread(rewards)
        p95_spread = relative_spread(p95s)
        passed = (
            reward_spread >= args.min_reward_spread
            and p95_spread >= args.min_p95_spread
            and completion_safe
            and reward_aligned
        )
        decision["tasks"][task] = {
            "passed": passed,
            "best_reward_action": best_action,
            "reward_spread": reward_spread,
            "p95_spread": p95_spread,
            "best_action_completion_safe": completion_safe,
            "best_safe_p95_action": best_safe_p95_action,
            "reward_aligned_with_safe_p95": reward_aligned,
            "measurements": measurements,
            "common_flows": len(common_flows),
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
        "| Task | Common flows | Reward best | Safe-p95 best | Reward spread | p95 spread | Completion safe | Aligned | Gate |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in tasks:
        item = decision["tasks"][task]
        lines.append(
            f"| {task} | {item['common_flows']} | {item['best_reward_action']} | "
            f"{item['best_safe_p95_action']} | "
            f"{item['reward_spread']:.2%} | {item['p95_spread']:.2%} | "
            f"{item['best_action_completion_safe']} | "
            f"{item['reward_aligned_with_safe_p95']} | "
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
