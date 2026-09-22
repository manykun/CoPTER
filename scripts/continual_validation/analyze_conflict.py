#!/usr/bin/env python3
"""Analyze a cheap fixed-action screen, optionally applying a gate."""

import argparse
import csv
import json
from pathlib import Path

from analyze_forgetting import load_eval, summarize


def relative_spread(values):
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return 0.0
    return (max(values) - min(values)) / max(abs(max(values)), 1e-12)


def parse_ports(value):
    if not value:
        return []
    try:
        ports = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("required watch ports must be comma-separated integers") from exc
    if len(ports) != len(set(ports)) or any(port < 0 for port in ports):
        raise ValueError("required watch ports must be unique non-negative integers")
    return ports


def watch_port_summary(runs, port):
    """Summarize whether one physical port carries both fixed-action runs."""
    records = {}
    for action, run in runs.items():
        metrics = run["metrics"].get("watch_ports_metrics", {})
        record = metrics.get(str(port))
        if record is not None:
            active_means = record.get("means_active", {})
            congested_means = record.get("means_congested", {})
            records[action] = {
                "identifier": record.get("identifier"),
                "active_steps": int(record.get("active_steps", 0)),
                "congested_steps": int(record.get("congested_steps", 0)),
                "reward_active": active_means.get("reward"),
                "reward_congested": congested_means.get("reward"),
                "tail_safe_raw_active": active_means.get("tail_safe_raw"),
                "tail_safe_raw_congested": congested_means.get(
                    "tail_safe_raw"
                ),
            }
    if len(records) != len(runs):
        return {
            "present_in_all_actions": False,
            "active_steps_min": 0,
            "congested_steps_max": 0,
            "actions": records,
        }
    return {
        "present_in_all_actions": True,
        "active_steps_min": min(item["active_steps"] for item in records.values()),
        "congested_steps_max": max(
            item["congested_steps"] for item in records.values()
        ),
        "actions": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--min-reward-spread", type=float, default=0.05)
    parser.add_argument("--min-p95-spread", type=float, default=0.05)
    parser.add_argument("--min-completion-spread", type=float, default=0.02)
    parser.add_argument("--completion-tolerance", type=float, default=0.01)
    parser.add_argument("--required-watch-ports", default="")
    parser.add_argument("--min-port-active-samples", type=int, default=0)
    parser.add_argument("--min-port-congested-samples", type=int, default=0)
    parser.add_argument("--min-old-task-p95-penalty", type=float, default=0.0)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Report measurements without assigning or enforcing PASS/FAIL",
    )
    args = parser.parse_args()
    if args.gate and args.report_only:
        parser.error("--gate and --report-only are mutually exclusive")
    if args.min_port_active_samples < 0 or args.min_port_congested_samples < 0:
        parser.error("port sample thresholds must be non-negative")
    if args.min_old_task_p95_penalty < 0:
        parser.error("old-task p95 penalty must be non-negative")
    try:
        required_ports = parse_ports(args.required_watch_ports)
    except ValueError as exc:
        parser.error(str(exc))

    manifest = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    tasks = (manifest["task_a"], manifest["task_b"])
    action_sets = []
    for task in tasks:
        task_dir = args.run_dir / "screen" / task
        actions = {
            path.name
            for path in task_dir.iterdir()
            if path.is_dir() and (path / "metrics.json").is_file()
        }
        action_sets.append(actions)
    actions = sorted(set.intersection(*action_sets))
    if len(actions) < 2:
        raise SystemExit(
            "conflict screen requires at least two common fixed actions; "
            f"found {actions}"
        )
    decision = {"tasks": {}}
    port_details = {str(port): {"tasks": {}} for port in required_ports}
    rows = []
    for task in tasks:
        runs = {
            action: load_eval(args.run_dir / "screen" / task / action)
            for action in actions
        }
        for port in required_ports:
            port_details[str(port)]["tasks"][task] = watch_port_summary(
                runs, port
            )
        common_flows = set.intersection(
            *(set(run["flows"]) for run in runs.values())
        )
        measurements = {}
        for action in actions:
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
    old_task, new_task = tasks
    old_best = decision["tasks"][old_task]["best_reward_action"]
    new_best = decision["tasks"][new_task]["best_reward_action"]
    old_measurements = decision["tasks"][old_task]["measurements"]
    old_best_p95 = old_measurements[old_best]["p95_fct_us"]
    new_action_old_p95 = old_measurements[new_best]["p95_fct_us"]
    old_task_p95_penalty = (
        (new_action_old_p95 - old_best_p95) / old_best_p95
        if old_best_p95 not in (None, 0) and new_action_old_p95 is not None
        else None
    )
    old_task_conflict_passed = (
        old_task_p95_penalty is not None
        and old_task_p95_penalty >= args.min_old_task_p95_penalty
    )
    decision["directional_conflict"] = {
        "old_task": old_task,
        "old_task_preferred_action": old_best,
        "new_task_preferred_action": new_best,
        "old_task_p95_preferred_us": old_best_p95,
        "old_task_p95_with_new_action_us": new_action_old_p95,
        "old_task_p95_penalty": old_task_p95_penalty,
        "minimum_required": args.min_old_task_p95_penalty,
        "passed": None if args.report_only else old_task_conflict_passed,
    }
    for port, detail in port_details.items():
        detail["passed"] = all(
            item["present_in_all_actions"]
            and item["active_steps_min"] >= args.min_port_active_samples
            and item["congested_steps_max"] >= args.min_port_congested_samples
            for item in detail["tasks"].values()
        )
    port_screen_passed = all(item["passed"] for item in port_details.values())
    decision["shared_port_screen"] = {
        "required_ports": required_ports,
        "minimum_active_samples": args.min_port_active_samples,
        "minimum_congested_samples": args.min_port_congested_samples,
        "ports": port_details,
        "passed": None if args.report_only else port_screen_passed,
    }
    decision["passed"] = (
        None
        if args.report_only
        else (
            all(decision["tasks"][task]["passed"] for task in tasks)
            and decision["different_best_actions"]
            and old_task_conflict_passed
            and port_screen_passed
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
        (
            "- Old-task p95 penalty from using the new-task action: "
            f"**{old_task_p95_penalty:.2%}**"
            if old_task_p95_penalty is not None
            else "- Old-task p95 penalty from using the new-task action: **n/a**"
        ),
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
            "## Shared-path port screen",
            "",
        ]
    )
    if not required_ports:
        lines.append("No required watch ports were configured.")
    else:
        lines.extend(
            [
                "| Port | Task | Present | Min active samples | Max congested samples |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for port in required_ports:
            for task in tasks:
                item = port_details[str(port)]["tasks"][task]
                lines.append(
                    f"| {port} | {task} | {item['present_in_all_actions']} | "
                    f"{item['active_steps_min']} | {item['congested_steps_max']} |"
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
