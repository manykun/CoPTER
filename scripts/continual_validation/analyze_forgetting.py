#!/usr/bin/env python3
"""Analyze A->B catastrophic forgetting and the retention/plasticity of SOR."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def parse_fct(path):
    flows = {}
    occurrences = defaultdict(int)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_number}: expected at least 8 columns")
            base_key = tuple(fields[:6])
            occurrence = occurrences[base_key]
            occurrences[base_key] += 1
            key = base_key + (occurrence,)
            fct_ns = float(fields[6])
            ideal_ns = float(fields[7])
            if fct_ns > 0 and ideal_ns > 0:
                flows[key] = (fct_ns, ideal_ns)
    return flows


def load_eval(directory):
    fct_files = sorted(directory.glob("*.fct"))
    if not fct_files:
        raise ValueError(f"no FCT file under {directory}")
    input_flow = directory / "input.flow"
    metrics_file = directory / "metrics.json"
    if not input_flow.exists() or not metrics_file.exists():
        raise ValueError(f"incomplete evaluation directory: {directory}")
    expected = int(input_flow.read_text(encoding="utf-8").splitlines()[0])
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    # The registered reward follows all congested ports, which is closer to
    # the replay population optimized by ACC/SOR.  The former top-30% metric
    # is retained in metrics.json as a diagnostic, but it can reverse action
    # rankings by discarding persistently congested ports.
    primary_reward = metrics.get("rollout_all_congested_mean")
    if primary_reward is None:
        primary_reward = metrics.get("rollout_mean_reward")
    return {
        "directory": directory,
        "flows": parse_fct(fct_files[0]),
        "expected": expected,
        "reward": number_or_none(primary_reward),
        "reward_top30": number_or_none(metrics.get("rollout_mean_reward")),
        "metrics": metrics,
    }


def number_or_none(value):
    return None if value is None else float(value)


def summarize(run, common_keys=None):
    selected = (
        run["flows"]
        if common_keys is None
        else {key: run["flows"][key] for key in common_keys}
    )
    fct_us = [value[0] / 1000.0 for value in selected.values()]
    slowdowns = [max(1.0, value[0] / value[1]) for value in selected.values()]
    return {
        "expected_flows": run["expected"],
        "completed_flows": len(run["flows"]),
        "matched_flows": len(selected),
        "completion_ratio": (
            len(run["flows"]) / run["expected"] if run["expected"] else None
        ),
        "avg_fct_us": mean(fct_us) if fct_us else None,
        "p95_fct_us": percentile(fct_us, 0.95),
        "p99_slowdown": percentile(slowdowns, 0.99),
        "reward": run["reward"],
    }


def relative_drop(before, after):
    if before is None or after is None:
        return None
    return (before - after) / max(abs(before), 1e-12)


def relative_worsening(before, after):
    if before is None or after is None or before == 0:
        return None
    return (after - before) / before


def comparison(before, after):
    common = set(before["flows"]) & set(after["flows"])
    before_summary = summarize(before, common)
    after_summary = summarize(after, common)
    return {
        "common_flows": len(common),
        "before": before_summary,
        "after": after_summary,
        "reward_change": (
            None
            if before_summary["reward"] is None or after_summary["reward"] is None
            else (
                after_summary["reward"] - before_summary["reward"]
            ) / max(abs(before_summary["reward"]), 1e-12)
        ),
        "reward_drop": relative_drop(
            before_summary["reward"], after_summary["reward"]
        ),
        "avg_fct_worsening": relative_worsening(
            before_summary["avg_fct_us"], after_summary["avg_fct_us"]
        ),
        "p95_fct_worsening": relative_worsening(
            before_summary["p95_fct_us"], after_summary["p95_fct_us"]
        ),
        "completion_drop": (
            before_summary["completion_ratio"] - after_summary["completion_ratio"]
        ),
    }


def cross_method(left, right):
    common = set(left["flows"]) & set(right["flows"])
    return {
        "common_flows": len(common),
        "acc": summarize(left, common),
        "sor": summarize(right, common),
    }


def acquisition_passed(result, reward_gain, p95_gain, completion_tolerance):
    reward_change = result["reward_change"]
    p95_worsening = result["p95_fct_worsening"]
    reward_improved = reward_change is not None and reward_change >= reward_gain
    p95_improved = p95_worsening is not None and p95_worsening <= -p95_gain
    completion_safe = (
        result["completion_drop"] is not None
        and result["completion_drop"] <= completion_tolerance
    )
    return (reward_improved or p95_improved) and completion_safe


def method_analysis(root, method, task_a, task_b, args):
    base = root / "eval" / method
    initial_a = load_eval(base / "initial" / task_a)
    after_a_a = load_eval(base / "after_a" / task_a)
    after_a_b = load_eval(base / "after_a" / task_b)
    after_b_a = load_eval(base / "after_b" / task_a)
    after_b_b = load_eval(base / "after_b" / task_b)

    acquire_a = comparison(initial_a, after_a_a)
    acquire_b = comparison(after_a_b, after_b_b)
    forgetting = comparison(after_a_a, after_b_a)
    forgetting_detected = (
        forgetting["reward_drop"] is not None
        and forgetting["reward_drop"] >= args.min_reward_drop
        and forgetting["p95_fct_worsening"] is not None
        and forgetting["p95_fct_worsening"] >= args.min_p95_worsening
    )
    score = max(forgetting["reward_drop"] or 0.0, 0.0) + max(
        forgetting["p95_fct_worsening"] or 0.0, 0.0
    )
    return {
        "method": method,
        "task_a_acquisition": {
            **acquire_a,
            "passed": acquisition_passed(
                acquire_a,
                args.min_acquisition_reward_gain,
                args.min_acquisition_p95_gain,
                args.completion_tolerance,
            ),
        },
        "task_b_acquisition": {
            **acquire_b,
            "passed": acquisition_passed(
                acquire_b,
                args.min_acquisition_reward_gain,
                args.min_acquisition_p95_gain,
                args.completion_tolerance,
            ),
        },
        "forgetting": {
            **forgetting,
            "detected": forgetting_detected,
            "score": score,
        },
        "_after_b_b": after_b_b,
    }


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def flatten_rows(analyses):
    rows = []
    for method, result in analyses.items():
        for comparison_name in (
            "task_a_acquisition", "task_b_acquisition", "forgetting"
        ):
            item = result[comparison_name]
            rows.append(
                {
                    "method": method,
                    "comparison": comparison_name,
                    "common_flows": item["common_flows"],
                    "reward_before": item["before"]["reward"],
                    "reward_after": item["after"]["reward"],
                    "reward_change": item["reward_change"],
                    "reward_drop": item["reward_drop"],
                    "p95_fct_before_us": item["before"]["p95_fct_us"],
                    "p95_fct_after_us": item["after"]["p95_fct_us"],
                    "p95_fct_worsening": item["p95_fct_worsening"],
                    "completion_before": item["before"]["completion_ratio"],
                    "completion_after": item["after"]["completion_ratio"],
                    "completion_drop": item["completion_drop"],
                    "passed": item.get("passed", item.get("detected")),
                    "forgetting_score": item.get("score"),
                }
            )
    return rows


def write_report(path, manifest, analyses, comparison_result, decision, args):
    task_a = manifest["task_a"]
    task_b = manifest["task_b"]
    lines = [
        "# ACC/SOR continual-learning report",
        "",
        "## Experiment",
        "",
        f"- Curriculum: **{task_a} → {task_b}**",
        f"- Seed: **{manifest['seed']}**",
        f"- Training budget: **{manifest.get('updates_per_task', 'legacy')} "
        "optimizer updates per task**",
        "- Evaluation: **greedy, frozen, identical task flow files**",
        "",
        "## Decision",
        "",
    ]
    acc = analyses.get("acc")
    sor = analyses.get("sor")
    if acc:
        lines.append(
            "- ACC catastrophic forgetting: "
            f"**{'PASS' if acc['forgetting']['detected'] else 'FAIL'}**"
        )
        lines.append(
            "- ACC learned both tasks: "
            f"**{'PASS' if acc['task_a_acquisition']['passed'] and acc['task_b_acquisition']['passed'] else 'FAIL'}**"
        )
    if sor:
        lines.append(
            "- SOR learned both tasks: "
            f"**{'PASS' if sor['task_a_acquisition']['passed'] and sor['task_b_acquisition']['passed'] else 'FAIL'}**"
        )
    if comparison_result:
        lines.append(
            "- SOR retention/plasticity gate: "
            f"**{'PASS' if comparison_result['passed'] else 'FAIL'}**"
        )
    lines.extend(
        [
            "",
            "Catastrophic forgetting is registered only when old-task reward drops "
            f"by at least {args.min_reward_drop:.0%} and common-flow p95 FCT worsens "
            f"by at least {args.min_p95_worsening:.0%}. A method must also demonstrate "
            "task acquisition without exceeding the completion-loss tolerance; "
            "retaining an untrained or unsafe policy is not counted as success.",
            "",
            "## Measurements",
            "",
            "| Method | Comparison | Common flows | Reward before | Reward after | "
            "Reward drop | p95 before (us) | p95 after (us) | p95 worsening | "
            "Completion before | Completion after |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in flatten_rows(analyses):
        lines.append(
            f"| {row['method']} | {row['comparison']} | {row['common_flows']} | "
            f"{fmt(row['reward_before'])} | {fmt(row['reward_after'])} | "
            f"{fmt(row['reward_drop'])} | {fmt(row['p95_fct_before_us'], 2)} | "
            f"{fmt(row['p95_fct_after_us'], 2)} | "
            f"{fmt(row['p95_fct_worsening'])} | "
            f"{fmt(row['completion_before'])} | {fmt(row['completion_after'])} |"
        )
    if comparison_result:
        lines.extend(
            [
                "",
                "## ACC vs SOR",
                "",
                f"- Forgetting reduction: **{fmt(comparison_result['forgetting_reduction'])}** "
                f"(required ≥ {args.min_forgetting_reduction:.0%})",
                f"- SOR new-task p95 / ACC new-task p95: "
                f"**{fmt(comparison_result['new_task_p95_ratio'])}** "
                f"(required ≤ {1 + args.new_task_p95_tolerance:.2f})",
                f"- SOR new-task completion: **{fmt(comparison_result['new_task']['sor']['completion_ratio'])}**",
                f"- ACC new-task completion: **{fmt(comparison_result['new_task']['acc']['completion_ratio'])}**",
            ]
        )
    lines.extend(
        [
            "",
            "## Registered thresholds",
            "",
            "```json",
            json.dumps(
                {
                    "min_reward_drop": args.min_reward_drop,
                    "min_p95_worsening": args.min_p95_worsening,
                    "min_acquisition_reward_gain": args.min_acquisition_reward_gain,
                    "min_acquisition_p95_gain": args.min_acquisition_p95_gain,
                    "min_forgetting_reduction": args.min_forgetting_reduction,
                    "new_task_p95_tolerance": args.new_task_p95_tolerance,
                    "completion_tolerance": args.completion_tolerance,
                },
                indent=2,
            ),
            "```",
            "",
            "## Gate details",
            "",
            "```json",
            json.dumps(decision, indent=2),
            "```",
            "",
            "This single-seed experiment is mechanism evidence, not a statistical "
            "generalization claim across random traffic realizations.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--method", choices=("acc", "sor"))
    group.add_argument("--compare", help="Must be acc,sor")
    parser.add_argument("--min-reward-drop", type=float, default=0.10)
    parser.add_argument("--min-p95-worsening", type=float, default=0.10)
    parser.add_argument("--min-acquisition-reward-gain", type=float, default=0.02)
    parser.add_argument("--min-acquisition-p95-gain", type=float, default=0.05)
    parser.add_argument("--min-forgetting-reduction", type=float, default=0.30)
    parser.add_argument("--new-task-p95-tolerance", type=float, default=0.05)
    parser.add_argument("--completion-tolerance", type=float, default=0.01)
    parser.add_argument("--gate-forgetting", action="store_true")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    manifest_path = args.run_dir / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_a, task_b = manifest["task_a"], manifest["task_b"]

    methods = [args.method] if args.method else args.compare.split(",")
    if methods not in (["acc"], ["sor"], ["acc", "sor"]):
        raise SystemExit("--compare must be acc,sor")
    analyses = {
        method: method_analysis(args.run_dir, method, task_a, task_b, args)
        for method in methods
    }

    comparison_result = None
    if methods == ["acc", "sor"]:
        acc = analyses["acc"]
        sor = analyses["sor"]
        new_task = cross_method(acc.pop("_after_b_b"), sor.pop("_after_b_b"))
        acc_score = acc["forgetting"]["score"]
        sor_score = sor["forgetting"]["score"]
        reduction = (
            (acc_score - sor_score) / acc_score if acc_score > 0 else 0.0
        )
        acc_p95 = new_task["acc"]["p95_fct_us"]
        sor_p95 = new_task["sor"]["p95_fct_us"]
        p95_ratio = (
            sor_p95 / acc_p95
            if acc_p95 is not None and sor_p95 is not None and acc_p95 > 0
            else math.inf
        )
        completion_safe = (
            new_task["sor"]["completion_ratio"]
            >= new_task["acc"]["completion_ratio"] - args.completion_tolerance
        )
        comparison_result = {
            "forgetting_reduction": reduction,
            "retention_passed": reduction >= args.min_forgetting_reduction,
            "new_task": new_task,
            "new_task_p95_ratio": p95_ratio,
            "new_task_p95_safe": p95_ratio <= 1.0 + args.new_task_p95_tolerance,
            "new_task_completion_safe": completion_safe,
        }
        comparison_result["passed"] = (
            acc["forgetting"]["detected"]
            and acc["task_a_acquisition"]["passed"]
            and acc["task_b_acquisition"]["passed"]
            and sor["task_a_acquisition"]["passed"]
            and sor["task_b_acquisition"]["passed"]
            and comparison_result["retention_passed"]
            and comparison_result["new_task_p95_safe"]
            and completion_safe
        )
    else:
        analyses[methods[0]].pop("_after_b_b")

    acc_forgetting_pass = (
        analyses.get("acc", {}).get("forgetting", {}).get("detected", False)
    )
    selected_pass = (
        comparison_result["passed"]
        if comparison_result is not None
        else (
            acc_forgetting_pass
            and analyses["acc"]["task_a_acquisition"]["passed"]
            and analyses["acc"]["task_b_acquisition"]["passed"]
        )
        if methods == ["acc"]
        else analyses["sor"]["task_a_acquisition"]["passed"]
        and analyses["sor"]["task_b_acquisition"]["passed"]
    )
    decision = {
        "selected_pass": selected_pass,
        "acc_forgetting_pass": acc_forgetting_pass,
        "comparison": comparison_result,
        "methods": analyses,
    }

    rows = flatten_rows(analyses)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "continual_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.run_dir / "continual_analysis.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_report(
        args.run_dir / "CONTINUAL_REPORT.md",
        manifest,
        analyses,
        comparison_result,
        decision,
        args,
    )
    print(f"Analysis written to {args.run_dir / 'CONTINUAL_REPORT.md'}")
    print(f"Selected gate: {'PASS' if selected_pass else 'FAIL'}")
    if (args.gate or args.gate_forgetting) and not selected_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
