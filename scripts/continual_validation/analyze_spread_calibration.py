#!/usr/bin/env python3
"""Analyze a cheap spread/action grid and recommend a controlled A/B pair."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from analyze_conflict import relative_spread, watch_port_summary
from analyze_forgetting import load_eval, summarize
from verify_task_pair import load_flows


def relative_worsening(baseline, candidate):
    if baseline in (None, 0) or candidate is None:
        return None
    return (candidate - baseline) / baseline


def format_percent(value):
    return "n/a" if value is None else f"{value:.2%}"


def is_switch_switch_port(detail):
    """Return whether every parseable physical identifier joins two switches."""
    identifiers = {
        value.get("identifier")
        for value in detail.get("actions", {}).values()
        if value.get("identifier")
    }
    endpoints = []
    for identifier in identifiers:
        pieces = str(identifier).split("-")
        if len(pieces) != 2:
            continue
        try:
            endpoints.append(tuple(int(piece) for piece in pieces))
        except ValueError:
            continue
    return bool(endpoints) and all(
        left >= 256 and right >= 256 for left, right in endpoints
    )


def select_pair(results, ordered_names, minimum_penalty):
    """Return the strongest eligible steady->bursty directional conflict."""
    pairs = []
    for left_index, task_a in enumerate(ordered_names):
        for task_b in ordered_names[left_index + 1:]:
            old = results[task_a]
            new = results[task_b]
            shared_ready_ports = sorted(
                set(old.get("ready_ports", []))
                & set(new.get("ready_ports", []))
            )
            old_action = old["best_reward_action"]
            new_action = new["best_reward_action"]
            old_p95 = old["measurements"][old_action]["p95_fct_us"]
            old_with_new = old["measurements"][new_action]["p95_fct_us"]
            new_p95 = new["measurements"][new_action]["p95_fct_us"]
            new_with_old = new["measurements"][old_action]["p95_fct_us"]
            old_penalty = relative_worsening(old_p95, old_with_new)
            new_penalty = relative_worsening(new_p95, new_with_old)
            eligible = (
                old["eligible"]
                and new["eligible"]
                and old_action != new_action
                and bool(shared_ready_ports)
                and old_penalty is not None
                and new_penalty is not None
                and old_penalty >= minimum_penalty
                and new_penalty >= minimum_penalty
            )
            pair = {
                "task_a": task_a,
                "task_b": task_b,
                "task_a_action": old_action,
                "task_b_action": new_action,
                "old_task_p95_penalty": old_penalty,
                "new_task_p95_penalty_with_old_action": new_penalty,
                "shared_ready_ports": shared_ready_ports,
                "eligible": eligible,
            }
            pair["score"] = (
                old_penalty + new_penalty
                if old_penalty is not None and new_penalty is not None
                else None
            )
            pairs.append(pair)
    recommended = max(
        (pair for pair in pairs if pair["eligible"]),
        key=lambda pair: pair["score"],
        default=None,
    )
    return recommended, pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--min-completion", type=float, default=0.90)
    parser.add_argument("--completion-tolerance", type=float, default=0.01)
    parser.add_argument("--min-reward-spread", type=float, default=0.02)
    parser.add_argument("--min-p95-spread", type=float, default=0.05)
    parser.add_argument("--min-port-active-samples", type=int, default=50)
    parser.add_argument("--min-port-congested-samples", type=int, default=20)
    parser.add_argument(
        "--port-scope",
        choices=("all", "switch-switch"),
        default="switch-switch",
    )
    parser.add_argument("--min-pair-p95-penalty", type=float, default=0.03)
    args = parser.parse_args()

    manifest = json.loads(
        (args.run_dir / "calibration_manifest.json").read_text(encoding="utf-8")
    )
    candidates = manifest["candidates"]
    ordered_names = [item["name"] for item in candidates]
    watch_ports = manifest.get("watch_ports")
    if watch_ports is None:
        watch_ports = [int(manifest["watch_port"])]
    watch_ports = [int(port) for port in watch_ports]
    actions = manifest["actions"]

    identity_sets = {}
    for name in ordered_names:
        flows = load_flows(args.run_dir / "tasks" / name / "input.flow")
        identity_sets[name] = Counter(identity for identity, _ in flows)
    reference = identity_sets[ordered_names[0]]
    same_identity = all(value == reference for value in identity_sets.values())

    results = {}
    rows = []
    for candidate in candidates:
        name = candidate["name"]
        runs = {
            action: load_eval(args.run_dir / "runs" / name / action)
            for action in actions
        }
        common_flows = set.intersection(
            *(set(run["flows"]) for run in runs.values())
        )
        measurements = {
            action: summarize(run, common_flows)
            for action, run in runs.items()
        }
        ports = {
            str(port): watch_port_summary(runs, port)
            for port in watch_ports
        }
        ready_ports = [
            int(port) for port, detail in ports.items()
            if detail["present_in_all_actions"]
            and detail["active_steps_min"] >= args.min_port_active_samples
            and detail["congested_steps_max"] >= args.min_port_congested_samples
            and (
                args.port_scope == "all"
                or is_switch_switch_port(detail)
            )
        ]
        for action, measurement in measurements.items():
            rows.append({
                "scenario": name,
                "spread_fraction": candidate["spread_fraction"],
                "action": action,
                **measurement,
                "ready_port_count": len(ready_ports),
            })

        rewards = [value["reward"] for value in measurements.values()]
        p95s = [value["p95_fct_us"] for value in measurements.values()]
        completions = [
            value["completion_ratio"] for value in measurements.values()
        ]
        best_reward_action = max(
            measurements,
            key=lambda action: (
                measurements[action]["reward"]
                if measurements[action]["reward"] is not None
                else float("-inf")
            ),
        )
        best_completion = max(completions)
        safe_actions = [
            action for action, value in measurements.items()
            if value["completion_ratio"]
            >= best_completion - args.completion_tolerance
        ]
        best_safe_p95_action = min(
            safe_actions,
            key=lambda action: measurements[action]["p95_fct_us"],
        )
        reward_spread = relative_spread(rewards)
        p95_spread = relative_spread(p95s)
        completion_floor = min(completions)
        completion_safe = (
            measurements[best_reward_action]["completion_ratio"]
            >= best_completion - args.completion_tolerance
        )
        aligned = best_reward_action == best_safe_p95_action
        port_ready = bool(ready_ports)
        eligible = (
            same_identity
            and completion_floor >= args.min_completion
            and reward_spread >= args.min_reward_spread
            and p95_spread >= args.min_p95_spread
            and completion_safe
            and aligned
            and port_ready
        )
        results[name] = {
            "spread_fraction": candidate["spread_fraction"],
            "common_flows": len(common_flows),
            "best_reward_action": best_reward_action,
            "best_safe_p95_action": best_safe_p95_action,
            "reward_spread": reward_spread,
            "p95_spread": p95_spread,
            "completion_floor": completion_floor,
            "completion_safe": completion_safe,
            "reward_aligned_with_safe_p95": aligned,
            "ports": ports,
            "ready_ports": ready_ports,
            "port_ready": port_ready,
            "eligible": eligible,
            "measurements": measurements,
        }

    recommended, pairs = select_pair(
        results, ordered_names, args.min_pair_p95_penalty
    )
    decision = {
        "same_flow_identity_multiset": same_identity,
        "watch_ports": watch_ports,
        "port_scope": args.port_scope,
        "thresholds": {
            "min_completion": args.min_completion,
            "completion_tolerance": args.completion_tolerance,
            "min_reward_spread": args.min_reward_spread,
            "min_p95_spread": args.min_p95_spread,
            "min_port_active_samples": args.min_port_active_samples,
            "min_port_congested_samples": args.min_port_congested_samples,
            "min_pair_p95_penalty": args.min_pair_p95_penalty,
        },
        "scenarios": results,
        "pairs": pairs,
        "recommended_pair": recommended,
    }
    (args.run_dir / "calibration_analysis.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    with (args.run_dir / "calibration_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    env_path = args.run_dir / "recommended_pair.env"
    if recommended is None:
        if env_path.exists():
            env_path.unlink()
    else:
        recommended_ports = ",".join(
            map(str, recommended["shared_ready_ports"][:12])
        )
        env_path.write_text(
            f"RECOMMENDED_TASK_A={recommended['task_a']}\n"
            f"RECOMMENDED_TASK_B={recommended['task_b']}\n"
            f"RECOMMENDED_WATCH_PORTS={recommended_ports}\n",
            encoding="utf-8",
        )

    lines = [
        "# Spread calibration report",
        "",
        f"- Same flow-identity multiset: **{same_identity}**",
        f"- Watched ports: **{len(watch_ports)}**",
        f"- Ready-port scope: **{args.port_scope}**",
        (
            f"- Recommended pair: **{recommended['task_a']} → "
            f"{recommended['task_b']}**"
            if recommended
            else "- Recommended pair: **NONE**"
        ),
        "",
        "| Scenario | Spread | Completion floor | Reward best | p95 best | "
        "Reward spread | p95 spread | Ready ports | Eligible |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for candidate in candidates:
        item = results[candidate["name"]]
        lines.append(
            f"| {candidate['name']} | {candidate['spread_fraction']:.4g} | "
            f"{item['completion_floor']:.2%} | {item['best_reward_action']} | "
            f"{item['best_safe_p95_action']} | {item['reward_spread']:.2%} | "
            f"{item['p95_spread']:.2%} | {len(item['ready_ports'])} | "
            f"{item['eligible']} |"
        )
    lines.extend([
        "",
        "## Directional pair conflicts",
        "",
        "| Task A | Task B | A action | B action | B-action cost on A | "
        "A-action cost on B | Shared ready ports | Eligible |",
        "|---|---|---|---|---:|---:|---|---:|",
    ])
    for pair in pairs:
        old_penalty = pair["old_task_p95_penalty"]
        new_penalty = pair["new_task_p95_penalty_with_old_action"]
        lines.append(
            f"| {pair['task_a']} | {pair['task_b']} | "
            f"{pair['task_a_action']} | {pair['task_b_action']} | "
            f"{format_percent(old_penalty)} | {format_percent(new_penalty)} | "
            f"{','.join(map(str, pair['shared_ready_ports'][:12])) or 'none'} | "
            f"{pair['eligible']} |"
        )
    lines.extend([
        "",
        "A recommended pair must keep at least 90% completion, activate and "
        "congest at least one shared switch-to-switch port, remain reward/p95 sensitive, select different "
        "actions, and show at least 3% p95 cost in both directions.",
        "",
    ])
    (args.run_dir / "CALIBRATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Calibration analysis written to {args.run_dir / 'CALIBRATION_REPORT.md'}")
    if recommended:
        print(f"Recommended pair: {recommended['task_a']} -> {recommended['task_b']}")
    else:
        print("No task pair satisfies the registered calibration criteria.")


if __name__ == "__main__":
    main()
