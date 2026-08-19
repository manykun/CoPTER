#!/usr/bin/env python3
"""Analyze frozen per-port physical ECN range interventions."""

import argparse
import csv
import json
from pathlib import Path

from analyze_forgetting import load_eval, summarize


def relative_change(value, reference):
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


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_available(root, labels, task):
    loaded = {}
    for label in labels:
        directory = root / "eval" / label / task
        if (directory / "metrics.json").exists():
            loaded[label] = load_eval(directory)
    return loaded


def shared_summaries(runs):
    if not runs:
        return {}, set()
    common = None
    for run in runs.values():
        keys = set(run["flows"])
        common = keys if common is None else common & keys
    return {
        label: summarize(run, common) for label, run in runs.items()
    }, common


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--stage", choices=("screen", "safety"), required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--completion-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    manifest_path = args.run_dir / "range_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    port = manifest["target_port"]
    task_a, task_b = manifest["task_a"], manifest["task_b"]
    points = manifest["points"]
    labels = [entry["label"] for entry in points]
    point_map = {entry["label"]: entry for entry in points}

    old_runs = load_available(args.run_dir, labels, task_a)
    if set(old_runs) != set(labels):
        missing = sorted(set(labels) - set(old_runs))
        raise SystemExit(f"old-task evaluation missing: {', '.join(missing)}")
    old_summaries, old_common = shared_summaries(old_runs)
    center = old_summaries["center"]
    completion_floor = center["completion_ratio"] - args.completion_tolerance
    eligible = [
        label for label in labels if label != "center"
        and old_summaries[label]["completion_ratio"] >= completion_floor
    ]
    ranked = sorted(eligible, key=lambda label: old_summaries[label]["p95_fct_us"])
    top_labels = (
        [label for label in labels if label != "center"]
        if manifest["point_set"] == "control"
        else ranked[:args.top_k]
    )
    safety_labels = ["center"] + top_labels
    selection = {
        "target_port": port,
        "task": task_a,
        "top_k": args.top_k,
        "completion_tolerance": args.completion_tolerance,
        "center_completion": center["completion_ratio"],
        "labels": safety_labels,
        "candidates": [point_map[label] for label in top_labels],
    }
    (args.run_dir / "top_candidates.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    new_runs = load_available(args.run_dir, safety_labels, task_b)
    if args.stage == "safety" and set(new_runs) != set(safety_labels):
        missing = sorted(set(safety_labels) - set(new_runs))
        raise SystemExit(f"new-task safety evaluation missing: {', '.join(missing)}")
    new_summaries, new_common = shared_summaries(new_runs)

    rows = []
    for label in labels:
        point = point_map[label]
        old = old_summaries[label]
        old_metrics = old_runs[label]["metrics"]
        new = new_summaries.get(label)
        new_metrics = new_runs[label]["metrics"] if label in new_runs else {}
        rows.append({
            **point,
            "old_common_flows": len(old_common),
            "old_completion": old["completion_ratio"],
            "old_p95_fct_us": old["p95_fct_us"],
            "old_p95_change_vs_center": relative_change(
                old["p95_fct_us"], center["p95_fct_us"]
            ),
            "old_network_reward": old["reward"],
            "old_port_reward_active": port_value(
                old_metrics, port, "active", "reward"
            ),
            "old_port_reward_congested": port_value(
                old_metrics, port, "congested", "reward"
            ),
            "selected_for_safety": label in safety_labels,
            "new_common_flows": len(new_common) if new is not None else None,
            "new_completion": new["completion_ratio"] if new else None,
            "new_p95_fct_us": new["p95_fct_us"] if new else None,
            "new_p95_change_vs_center": relative_change(
                new["p95_fct_us"], new_summaries["center"]["p95_fct_us"]
            ) if new and "center" in new_summaries else None,
            "new_network_reward": new["reward"] if new else None,
            "new_port_reward_active": port_value(
                new_metrics, port, "active", "reward"
            ),
            "new_port_reward_congested": port_value(
                new_metrics, port, "congested", "reward"
            ),
        })

    result = {
        "stage": args.stage,
        "target_port": port,
        "task_a": task_a,
        "task_b": task_b,
        "old_common_flows": len(old_common),
        "new_common_flows": len(new_common),
        "safety_labels": safety_labels,
        "rows": rows,
    }
    (args.run_dir / "physical_range_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.run_dir / "physical_range_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Per-port physical ECN range sweep",
        "",
        f"- Target OpenGym port: **{port}**",
        f"- Frozen policy: **{manifest['base_run_id']} after-B**",
        f"- Old/new tasks: **{task_a} / {task_b}**",
        f"- Point set: **{manifest['point_set']}**",
        "- Descriptive experiment; network weights and replay are not updated.",
        "",
        "| Point | Kmin KB | Kmax KB | Pmax | Old completion | Old p95 us | vs center | Old port reward | Safety | New p95 us | vs center |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        old_change = row["old_p95_change_vs_center"]
        new_change = row["new_p95_change_vs_center"]
        lines.append(
            f"| {row['label']} | {row['kmin_kb']} | {row['kmax_kb']} | "
            f"{row['pmax']:.2f} | {fmt(row['old_completion'], 4)} | "
            f"{fmt(row['old_p95_fct_us'], 2)} | "
            f"{fmt(None if old_change is None else 100 * old_change, 2)}% | "
            f"{fmt(row['old_port_reward_congested'], 4)} | "
            f"{'yes' if row['selected_for_safety'] else 'no'} | "
            f"{fmt(row['new_p95_fct_us'], 2)} | "
            f"{fmt(None if new_change is None else 100 * new_change, 2)}% |"
        )
    lines += [
        "",
        "## Selection",
        "",
        "Safety evaluation uses the center plus the old-task top candidates "
        f"whose completion is within {args.completion_tolerance:.2%} of center.",
        f"Selected: **{', '.join(safety_labels)}**.",
        "",
    ]
    (args.run_dir / "PHYSICAL_RANGE_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Analysis written to {args.run_dir / 'PHYSICAL_RANGE_REPORT.md'}")


if __name__ == "__main__":
    main()
