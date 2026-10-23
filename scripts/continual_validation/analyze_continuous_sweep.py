#!/usr/bin/env python3
"""Descriptive analysis of a frozen one-port continuous midpoint sweep."""

import argparse
import csv
import json
from pathlib import Path

from analyze_forgetting import load_eval, summarize


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def change(value, reference):
    if value is None or reference is None or abs(reference) < 1e-12:
        return None
    return (value - reference) / abs(reference)


def port_value(metrics, port, population, key):
    return (
        metrics.get("watch_ports_metrics", {})
        .get(str(port), {})
        .get(f"means_{population}", {})
        .get(key)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task-a", required=True)
    parser.add_argument("--task-b", required=True)
    args = parser.parse_args()

    manifest = json.loads(
        (args.run_dir / "sweep_manifest.json").read_text(encoding="utf-8")
    )
    port = manifest["target_port"]
    point_map = {point["label"]: point for point in manifest["points"]}
    labels = ["greedy"] + [point["label"] for point in manifest["points"]]
    runs = {}
    for label in labels:
        runs[label] = {}
        for task in (args.task_a, args.task_b):
            directory = args.run_dir / "eval" / label / task
            if not (directory / "metrics.json").exists():
                raise SystemExit(f"evaluation missing: {label}/{task}")
            runs[label][task] = load_eval(directory)

    common = {}
    for task in (args.task_a, args.task_b):
        keys = set(runs[labels[0]][task]["flows"])
        for label in labels[1:]:
            keys &= set(runs[label][task]["flows"])
        common[task] = keys

    summaries = {
        label: {
            task: summarize(runs[label][task], common[task])
            for task in (args.task_a, args.task_b)
        }
        for label in labels
    }
    center = summaries["center"]
    rows = []
    for label in labels:
        point = point_map.get(label, {})
        physical = point.get("physical", {})
        old = summaries[label][args.task_a]
        new = summaries[label][args.task_b]
        old_metrics = runs[label][args.task_a]["metrics"]
        new_metrics = runs[label][args.task_b]["metrics"]
        row = {
            "label": label,
            "kind": point.get("kind", "policy"),
            "dimension": point.get("dimension"),
            "kmin_norm": point.get("kmin_norm"),
            "kmax_norm": point.get("kmax_norm"),
            "pmax": point.get("pmax"),
            "kmin_kb": physical.get("kmin_kb"),
            "kmax_kb": physical.get("kmax_kb"),
            "old_common_flows": len(common[args.task_a]),
            "old_p95_fct_us": old["p95_fct_us"],
            "old_p95_change_vs_center": change(
                old["p95_fct_us"], center[args.task_a]["p95_fct_us"]
            ),
            "old_completion": old["completion_ratio"],
            "old_network_reward": old["reward"],
            "old_port_reward_active": port_value(
                old_metrics, port, "active", "reward"
            ),
            "old_port_reward_congested": port_value(
                old_metrics, port, "congested", "reward"
            ),
            "old_port_raw_congested": port_value(
                old_metrics, port, "congested", "tail_safe_raw"
            ),
            "new_common_flows": len(common[args.task_b]),
            "new_p95_fct_us": new["p95_fct_us"],
            "new_p95_change_vs_center": change(
                new["p95_fct_us"], center[args.task_b]["p95_fct_us"]
            ),
            "new_completion": new["completion_ratio"],
            "new_network_reward": new["reward"],
            "new_port_reward_active": port_value(
                new_metrics, port, "active", "reward"
            ),
            "new_port_reward_congested": port_value(
                new_metrics, port, "congested", "reward"
            ),
        }
        rows.append(row)

    candidates = [row for row in rows if row["kind"] == "midpoint"]
    best_p95 = min(candidates, key=lambda row: row["old_p95_fct_us"])
    reward_candidates = [
        row for row in candidates if row["old_port_reward_congested"] is not None
    ]
    best_reward = (
        max(reward_candidates, key=lambda row: row["old_port_reward_congested"])
        if reward_candidates else None
    )
    result = {
        "task_a": args.task_a,
        "task_b": args.task_b,
        "target_port": port,
        "common_flows": {task: len(keys) for task, keys in common.items()},
        "dominant_action_indices": manifest["dominant_action_indices"],
        "dominant_action_fraction": manifest["dominant_action_fraction"],
        "best_midpoint_by_old_p95": best_p95["label"],
        "best_midpoint_by_old_port_reward": (
            best_reward["label"] if best_reward else None
        ),
        "rows": rows,
    }
    (args.run_dir / "continuous_sweep_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.run_dir / "continuous_sweep_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# One-port continuous action sweep",
        "",
        f"- Target port: **{port}**",
        f"- Old task: **{args.task_a}**",
        f"- New task: **{args.task_b}**",
        f"- Dominant discrete action: **{manifest['dominant_action_indices']}** "
        f"({manifest['dominant_action_fraction']:.2%} of measured actions)",
        "- Frozen evaluation; no replay or network weights are updated.",
        "- Descriptive output only; no PASS/FAIL gate is applied.",
        "",
        "## Results",
        "",
        "| Point | Kind | Kmin (KB) | Kmax (KB) | Pmax | Old p95 (us) | vs center | Old port reward | New p95 (us) | vs center |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['kind']} | {fmt(row['kmin_kb'], 2)} | "
            f"{fmt(row['kmax_kb'], 2)} | {fmt(row['pmax'], 3)} | "
            f"{fmt(row['old_p95_fct_us'], 2)} | "
            f"{fmt(None if row['old_p95_change_vs_center'] is None else 100 * row['old_p95_change_vs_center'], 2)}% | "
            f"{fmt(row['old_port_reward_congested'])} | "
            f"{fmt(row['new_p95_fct_us'], 2)} | "
            f"{fmt(None if row['new_p95_change_vs_center'] is None else 100 * row['new_p95_change_vs_center'], 2)}% |"
        )
    lines += [
        "",
        "## Reading the result",
        "",
        f"- Best midpoint by old-task p95: **{best_p95['label']}**.",
        "- A midpoint that beats the fixed center while preserving the new task supports continuous control within the current range.",
        "- A monotonic improvement toward an outer grid point suggests testing the range boundary next.",
        "- Nearly identical midpoint results indicate that finer discretization is unlikely to help.",
        "",
    ]
    if best_reward is not None:
        lines.insert(-1, f"- Best midpoint by old-port reward: **{best_reward['label']}**.")
    (args.run_dir / "CONTINUOUS_SWEEP_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Analysis written to {args.run_dir / 'CONTINUOUS_SWEEP_REPORT.md'}")


if __name__ == "__main__":
    main()
