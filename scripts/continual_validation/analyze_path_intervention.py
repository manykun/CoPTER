#!/usr/bin/env python3
"""Analyze selected-port after-A policy restoration without PASS/FAIL gates."""

import argparse
import csv
import json
from pathlib import Path

from analyze_forgetting import load_eval, summarize


def relative(numerator, denominator):
    if numerator is None or denominator is None or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def common_summaries(*runs):
    keys = set(runs[0]["flows"])
    for run in runs[1:]:
        keys &= set(run["flows"])
    return len(keys), [summarize(run, keys) for run in runs]


def port_metric(metrics, port, population, key):
    detail = metrics.get("watch_ports_metrics", {}).get(str(port), {})
    return detail.get(f"means_{population}", {}).get(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--task-a", required=True)
    parser.add_argument("--task-b", required=True)
    args = parser.parse_args()

    eval_root = args.run_dir / "eval"
    variants = sorted(path.name for path in eval_root.iterdir() if path.is_dir())
    for required in ("after_a", "after_b"):
        if required not in variants:
            raise SystemExit(f"required reference missing: {required}")
    runs = {
        variant: {
            args.task_a: load_eval(eval_root / variant / args.task_a),
            args.task_b: load_eval(eval_root / variant / args.task_b),
        }
        for variant in variants
    }

    global_rows = []
    after_a_old = runs["after_a"][args.task_a]
    after_b_old = runs["after_b"][args.task_a]
    after_b_new = runs["after_b"][args.task_b]
    for variant in variants:
        common_old, old = common_summaries(
            after_a_old, after_b_old, runs[variant][args.task_a]
        )
        old_a, old_b, old_variant = old
        common_new, new = common_summaries(
            after_b_new, runs[variant][args.task_b]
        )
        new_b, new_variant = new
        row = {
            "variant": variant,
            "old_task_common_flows": common_old,
            "old_task_p95_us": old_variant["p95_fct_us"],
            "old_task_reward": old_variant["reward"],
            "old_task_completion": old_variant["completion_ratio"],
            "p95_recovery": relative(
                old_b["p95_fct_us"] - old_variant["p95_fct_us"],
                old_b["p95_fct_us"] - old_a["p95_fct_us"],
            ),
            "reward_recovery": relative(
                old_variant["reward"] - old_b["reward"],
                old_a["reward"] - old_b["reward"],
            ),
            "new_task_common_flows": common_new,
            "new_task_p95_us": new_variant["p95_fct_us"],
            "new_task_reward": new_variant["reward"],
            "new_task_completion": new_variant["completion_ratio"],
            "new_task_p95_cost": relative(
                new_variant["p95_fct_us"] - new_b["p95_fct_us"],
                new_b["p95_fct_us"],
            ),
            "new_task_completion_cost": (
                new_b["completion_ratio"] - new_variant["completion_ratio"]
            ),
        }
        global_rows.append(row)

    watch_ports = sorted({
        int(port)
        for variant in variants
        for port in runs[variant][args.task_a]["metrics"]
        .get("watch_ports_metrics", {})
    })
    port_rows = []
    for port in watch_ports:
        for variant in variants:
            metrics_a = runs["after_a"][args.task_a]["metrics"]
            metrics_b = runs["after_b"][args.task_a]["metrics"]
            metrics_v = runs[variant][args.task_a]["metrics"]
            detail = metrics_v.get("watch_ports_metrics", {}).get(str(port), {})
            for population in ("active", "congested"):
                reward_a = port_metric(metrics_a, port, population, "reward")
                reward_b = port_metric(metrics_b, port, population, "reward")
                reward_v = port_metric(metrics_v, port, population, "reward")
                raw_a = port_metric(metrics_a, port, population, "tail_safe_raw")
                raw_b = port_metric(metrics_b, port, population, "tail_safe_raw")
                raw_v = port_metric(metrics_v, port, population, "tail_safe_raw")
                port_rows.append({
                    "variant": variant,
                    "port": port,
                    "identifier": detail.get("identifier"),
                    "population": population,
                    "samples": detail.get(f"{population}_steps"),
                    "reward_after_a": reward_a,
                    "reward_after_b": reward_b,
                    "reward_variant": reward_v,
                    "reward_recovery": relative(
                        None if reward_v is None or reward_b is None else reward_v - reward_b,
                        None if reward_a is None or reward_b is None else reward_a - reward_b,
                    ),
                    "raw_after_a": raw_a,
                    "raw_after_b": raw_b,
                    "raw_variant": raw_v,
                    "raw_recovery": relative(
                        None if raw_v is None or raw_b is None else raw_v - raw_b,
                        None if raw_a is None or raw_b is None else raw_a - raw_b,
                    ),
                })

    result = {
        "task_a": args.task_a,
        "task_b": args.task_b,
        "definition": (
            "Recovery=1 means the restored checkpoint recovers the entire "
            "after-A to after-B change; 0 means no change from after-B."
        ),
        "global": global_rows,
        "ports": port_rows,
    }
    (args.run_dir / "path_intervention_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for filename, rows in (
        ("path_intervention_summary.csv", global_rows),
        ("path_intervention_ports.csv", port_rows),
    ):
        with (args.run_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)

    lines = [
        "# Path-level policy restoration report",
        "",
        f"- Old task: **{args.task_a}**",
        f"- New task: **{args.task_b}**",
        "- This report is descriptive; it does not apply a PASS/FAIL gate.",
        "- Recovery 100% means full reversal of the after-A → after-B change; "
        "0% means identical to after-B.",
        "",
        "## Network-level results",
        "",
        "| Variant | Old p95 (us) | p95 recovery | Old reward | reward recovery | New p95 cost | New completion cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows:
        lines.append(
            f"| {row['variant']} | {fmt(row['old_task_p95_us'], 2)} | "
            f"{fmt(None if row['p95_recovery'] is None else 100 * row['p95_recovery'], 2)}% | "
            f"{fmt(row['old_task_reward'])} | "
            f"{fmt(None if row['reward_recovery'] is None else 100 * row['reward_recovery'], 2)}% | "
            f"{fmt(None if row['new_task_p95_cost'] is None else 100 * row['new_task_p95_cost'], 2)}% | "
            f"{fmt(None if row['new_task_completion_cost'] is None else 100 * row['new_task_completion_cost'], 3)} pp |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Compare each targeted restore with its same-size random control. A larger old-task recovery with a small new-task cost supports path-localized forgetting. The per-port CSV reports both clipped reward and unclipped tail-safe raw reward for active and congested samples.",
        "",
        "Detailed files: `path_intervention_ports.csv` and `path_intervention_analysis.json`.",
        "",
    ]
    (args.run_dir / "PATH_INTERVENTION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Analysis written to {args.run_dir / 'PATH_INTERVENTION_REPORT.md'}")


if __name__ == "__main__":
    main()
