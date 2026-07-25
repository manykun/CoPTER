#!/usr/bin/env python3
"""Analyze a cheap fixed-action screen, optionally applying a gate."""

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
    parser.add_argument("--min-completion-spread", type=float, default=0.02)
    parser.add_argument("--completion-tolerance", type=float, default=0.01)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Report measurements without assigning or enforcing PASS/FAIL",
    )
    args = parser.parse_args()
    if args.gate and args.report_only:
        parser.error("--gate and --report-only are mutually exclusive")

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
        completion_spread = max(completions) - min(completions)
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
        performance_sensitive = (
            p95_spread >= args.min_p95_spread
            or completion_spread >= args.min_completion_spread
        )
        passed = (
            reward_spread >= args.min_reward_spread
            and performance_sensitive
            and completion_safe
            and reward_aligned
        )
        decision["tasks"][task] = {
            "passed": None if args.report_only else passed,
            "best_reward_action": best_action,
            "reward_spread": reward_spread,
            "p95_spread": p95_spread,
            "completion_spread": completion_spread,
            "performance_sensitive": performance_sensitive,
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
        None
        if args.report_only
        else (
            all(decision["tasks"][task]["passed"] for task in tasks)
            and decision["different_best_actions"]
        )
    )
    decision["evaluation_mode"] = (
        "descriptive" if args.report_only else "gated"
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
        (
            "- Evaluation mode: **descriptive measurements only; no PASS/FAIL "
            "gate is applied**"
            if args.report_only
            else f"- Overall: **{'PASS' if decision['passed'] else 'FAIL'}**"
        ),
        f"- Different reward-best actions: **{decision['different_best_actions']}**",
        "- Primary reward: **all-congested-port mean**",
        "",
        (
            "| Task | Common flows | Reward best | Safe-p95 best | Reward spread | "
            "p95 spread | Completion spread |"
            if args.report_only
            else "| Task | Common flows | Reward best | Safe-p95 best | Reward "
            "spread | p95 spread | Completion spread | Sensitive | Completion "
            "safe | Aligned | Gate |"
        ),
        (
            "|---|---:|---|---|---:|---:|---:|"
            if args.report_only
            else "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for task in tasks:
        item = decision["tasks"][task]
        prefix = (
            f"| {task} | {item['common_flows']} | {item['best_reward_action']} | "
            f"{item['best_safe_p95_action']} | "
            f"{item['reward_spread']:.2%} | {item['p95_spread']:.2%} | "
            f"{item['completion_spread']:.2%}"
        )
        if args.report_only:
            lines.append(prefix + " |")
        else:
            lines.append(
                prefix
                + f" | {item['performance_sensitive']} | "
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
    if args.report_only:
        print("Evaluation mode: descriptive results (no PASS/FAIL gate)")
    else:
        print(f"Selected gate: {'PASS' if decision['passed'] else 'FAIL'}")
    if args.gate and not decision["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
