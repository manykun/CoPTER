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


def relative_reward_penalty(best, candidate):
    """Return the fractional reward lost by replacing a task's best action."""
    if best is None or candidate is None:
        return None
    return (best - candidate) / max(abs(best), 1e-12)


def port_action_profile(detail):
    """Summarize fixed-action reward preferences for one watched port."""
    rewards = {
        action: record.get("reward_congested")
        for action, record in detail.get("actions", {}).items()
        if record.get("reward_congested") is not None
    }
    if not rewards:
        return {
            "best_action": None,
            "reward_spread": 0.0,
            "rewards": {},
        }
    return {
        "best_action": max(rewards, key=rewards.get),
        "reward_spread": relative_spread(rewards.values()),
        "rewards": rewards,
    }


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


def controlled_pair(pair_mode, same_identity, same_endpoint_support):
    if pair_mode == "timing-only":
        return same_identity
    if pair_mode == "workload-shift":
        return same_endpoint_support and not same_identity
    raise ValueError(f"unsupported pair mode: {pair_mode}")


def select_pair(
    results,
    ordered_names,
    minimum_penalty,
    minimum_port_reward_penalty,
    anchor_first=False,
):
    """Return the strongest eligible steady->bursty directional conflict."""
    pairs = []
    for left_index, task_a in enumerate(ordered_names):
        if anchor_first and left_index > 0:
            break
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
            port_conflicts = []
            for port in shared_ready_ports:
                old_profile = old["port_action_profiles"][str(port)]
                new_profile = new["port_action_profiles"][str(port)]
                old_port_action = old_profile["best_action"]
                new_port_action = new_profile["best_action"]
                if (
                    old_port_action is None
                    or new_port_action is None
                    or old_port_action == new_port_action
                ):
                    continue
                old_best_reward = old_profile["rewards"][old_port_action]
                old_cross_reward = old_profile["rewards"].get(new_port_action)
                new_best_reward = new_profile["rewards"][new_port_action]
                new_cross_reward = new_profile["rewards"].get(old_port_action)
                old_reward_penalty = relative_reward_penalty(
                    old_best_reward, old_cross_reward
                )
                new_reward_penalty = relative_reward_penalty(
                    new_best_reward, new_cross_reward
                )
                conflict_eligible = (
                    old_profile["reward_spread"]
                    >= minimum_port_reward_penalty
                    and new_profile["reward_spread"]
                    >= minimum_port_reward_penalty
                    and old_reward_penalty is not None
                    and new_reward_penalty is not None
                    and old_reward_penalty >= minimum_port_reward_penalty
                    and new_reward_penalty >= minimum_port_reward_penalty
                )
                port_conflicts.append({
                    "port": port,
                    "task_a_action": old_port_action,
                    "task_b_action": new_port_action,
                    "task_a_reward_spread": old_profile["reward_spread"],
                    "task_b_reward_spread": new_profile["reward_spread"],
                    "task_a_reward_penalty": old_reward_penalty,
                    "task_b_reward_penalty": new_reward_penalty,
                    "eligible": conflict_eligible,
                })
            eligible_port_conflicts = [
                conflict for conflict in port_conflicts
                if conflict["eligible"]
            ]
            eligible = (
                old["eligible"]
                and new["eligible"]
                and old_action != new_action
                and bool(shared_ready_ports)
                and bool(eligible_port_conflicts)
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
                "port_conflicts": port_conflicts,
                "eligible_port_conflicts": eligible_port_conflicts,
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
    parser.add_argument(
        "--min-port-reward-penalty", type=float, default=0.03
    )
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
    pair_mode = manifest.get("pair_mode", "timing-only")
    pair_anchor_first = bool(manifest.get("pair_anchor_first", False))

    identity_sets = {}
    for name in ordered_names:
        flows = load_flows(args.run_dir / "tasks" / name / "input.flow")
        identity_sets[name] = Counter(identity for identity, _ in flows)
    reference = identity_sets[ordered_names[0]]
    same_identity = all(value == reference for value in identity_sets.values())
    endpoint_supports = {
        name: {identity[:4] for identity in identities}
        for name, identities in identity_sets.items()
    }
    reference_endpoints = endpoint_supports[ordered_names[0]]
    same_endpoint_support = all(
        value == reference_endpoints for value in endpoint_supports.values()
    )
    pair_controlled = controlled_pair(
        pair_mode, same_identity, same_endpoint_support
    )

    results = {}
    rows = []
    port_rows = []
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
        port_action_profiles = {
            str(port): port_action_profile(ports[str(port)])
            for port in ready_ports
        }
        for port in ready_ports:
            for action, record in ports[str(port)]["actions"].items():
                port_rows.append({
                    "scenario": name,
                    "port": port,
                    "identifier": record.get("identifier"),
                    "action": action,
                    "active_steps": record.get("active_steps"),
                    "congested_steps": record.get("congested_steps"),
                    "reward_active": record.get("reward_active"),
                    "reward_congested": record.get("reward_congested"),
                    "tail_safe_raw_active": record.get(
                        "tail_safe_raw_active"
                    ),
                    "tail_safe_raw_congested": record.get(
                        "tail_safe_raw_congested"
                    ),
                })
        eligible = (
            pair_controlled
            and completion_floor >= args.min_completion
            and reward_spread >= args.min_reward_spread
            and p95_spread >= args.min_p95_spread
            and completion_safe
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
            "port_action_profiles": port_action_profiles,
            "eligible": eligible,
            "measurements": measurements,
        }

    recommended, pairs = select_pair(
        results,
        ordered_names,
        args.min_pair_p95_penalty,
        args.min_port_reward_penalty,
        pair_anchor_first,
    )
    decision = {
        "same_flow_identity_multiset": same_identity,
        "same_endpoint_support": same_endpoint_support,
        "pair_mode": pair_mode,
        "pair_anchor_first": pair_anchor_first,
        "controlled_pair": pair_controlled,
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
            "min_port_reward_penalty": args.min_port_reward_penalty,
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
    port_summary_path = args.run_dir / "port_action_summary.csv"
    if port_rows:
        with port_summary_path.open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(port_rows[0]))
            writer.writeheader()
            writer.writerows(port_rows)
    elif port_summary_path.exists():
        port_summary_path.unlink()

    env_path = args.run_dir / "recommended_pair.env"
    if recommended is None:
        if env_path.exists():
            env_path.unlink()
    else:
        recommended_ports = ",".join(
            str(item["port"])
            for item in recommended["eligible_port_conflicts"][:12]
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
        f"- Same endpoint support: **{same_endpoint_support}**",
        f"- Pair mode: **{pair_mode}**",
        (
            "- Pair anchor: **first candidate only**"
            if pair_anchor_first
            else "- Pair anchor: **all ordered pairs**"
        ),
        f"- Controlled pair: **{pair_controlled}**",
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
        "## Action measurements",
        "",
        "| Scenario | Action | Completion | p95 FCT (us) | Reward |",
        "|---|---|---:|---:|---:|",
    ])
    for candidate in candidates:
        item = results[candidate["name"]]
        for action in actions:
            measurement = item["measurements"][action]
            reward = measurement["reward"]
            reward_text = "n/a" if reward is None else f"{reward:.6f}"
            lines.append(
                f"| {candidate['name']} | {action} | "
                f"{measurement['completion_ratio']:.2%} | "
                f"{measurement['p95_fct_us']:.2f} | {reward_text} |"
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
        "## Shared-port reward conflicts",
        "",
        "| Task A | Task B | Port | A action | B action | "
        "B-action reward cost on A | A-action reward cost on B | Eligible |",
        "|---|---|---:|---|---|---:|---:|---:|",
    ])
    for pair in pairs:
        for conflict in pair["port_conflicts"]:
            lines.append(
                f"| {pair['task_a']} | {pair['task_b']} | "
                f"{conflict['port']} | {conflict['task_a_action']} | "
                f"{conflict['task_b_action']} | "
                f"{format_percent(conflict['task_a_reward_penalty'])} | "
                f"{format_percent(conflict['task_b_reward_penalty'])} | "
                f"{conflict['eligible']} |"
            )
    lines.extend([
        "",
        "A recommended pair must keep at least 90% completion, activate and "
        "congest at least one shared switch-to-switch port, remain reward/p95 sensitive, select different "
        "actions, show at least 3% p95 cost in both directions, and contain at "
        "least one shared port whose reward-optimal action changes with at "
        "least 3% reward cost in both directions.",
        "Reward-best and p95-best actions are reported separately; exact "
        "equality is not required because the pair gate directly measures "
        "the p95 cost of exchanging the two learned reward-optimal actions.",
        "All ready-port/action measurements are written to "
        "`port_action_summary.csv` for diagnostics.",
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
