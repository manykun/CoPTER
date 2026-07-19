#!/usr/bin/env python3
"""Analyze the reproducible ACC validation experiment without third-party packages."""

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
    with path.open() as handle:
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
            if fct_ns <= 0 or ideal_ns <= 0:
                continue
            flows[key] = (fct_ns, ideal_ns)
    return flows


def expected_flows(directory):
    flow_file = directory / "input.flow"
    if not flow_file.exists():
        return None
    with flow_file.open() as handle:
        return int(handle.readline().strip())


def read_metrics(directory):
    path = directory / "metrics.json"
    if not path.exists():
        return {}
    text = path.read_text().strip()
    return json.loads(text) if text else {}


def read_config_value(directory, key):
    path = directory / "input.conf"
    if not path.exists():
        return None
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2 and fields[0] == key:
                try:
                    return float(fields[1])
                except ValueError:
                    return fields[1]
    return None


def summarize(flows, expected, common_keys=None):
    selected = flows if common_keys is None else {key: flows[key] for key in common_keys}
    fcts = [value[0] / 1000.0 for value in selected.values()]
    slowdowns = [max(1.0, value[0] / value[1]) for value in selected.values()]
    return {
        "expected_flows": expected,
        "completed_flows": len(flows),
        "matched_flows": len(selected),
        "completion_ratio": (len(flows) / expected) if expected else None,
        "avg_fct_us": mean(fcts) if fcts else None,
        "p95_fct_us": percentile(fcts, 0.95),
        "p99_fct_us": percentile(fcts, 0.99),
        "avg_slowdown": mean(slowdowns) if slowdowns else None,
        "p95_slowdown": percentile(slowdowns, 0.95),
        "p99_slowdown": percentile(slowdowns, 0.99),
    }


def load_runs(root, kind):
    runs = {}
    if not root.exists():
        return runs
    for method_dir in root.glob("*/seed_*/*"):
        if not method_dir.is_dir():
            continue
        relative = method_dir.relative_to(root)
        scenario, seed_dir, method = relative.parts
        fct_files = sorted(method_dir.glob("*.fct"))
        if not fct_files:
            continue
        seed = int(seed_dir.removeprefix("seed_"))
        runs[(kind, scenario, seed, method)] = {
            "kind": kind,
            "directory": method_dir,
            "flows": parse_fct(fct_files[0]),
            "expected": expected_flows(method_dir),
            "metrics": read_metrics(method_dir),
        }
    return runs


def build_rows(runs):
    rows = []
    grouped = defaultdict(list)
    for key in runs:
        grouped[key[:3]].append(key)
    for _, keys in grouped.items():
        common = None
        for key in keys:
            flow_keys = set(runs[key]["flows"])
            common = flow_keys if common is None else common & flow_keys
        common = common or set()
        for key in sorted(keys):
            kind, scenario, seed, method = key
            run = runs[key]
            row = {
                "kind": run["kind"],
                "scenario": scenario,
                "seed": seed,
                "method": method,
                "simulator_stop_time": read_config_value(run["directory"], "SIMULATOR_STOP_TIME"),
                **summarize(run["flows"], run["expected"], common),
            }
            metrics = run["metrics"]
            for metric_name in (
                "rollout_mean_reward", "rollout_all_congested_mean",
                "reward_throughput_mean", "reward_queue_mean", "reward_ecn_mean",
                "reward_avg_queue_mean", "reward_peak_queue_mean", "reward_avg_ecn_mean",
            ):
                row[metric_name] = metrics.get(metric_name)
            rows.append(row)
    return rows


def relative_spread(values):
    values = [value for value in values if value is not None]
    return (max(values) - min(values)) / max(values) if values and max(values) else 0.0


def rank_values(values):
    """Return average ranks, including ties, without requiring SciPy."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def correlation(left, right):
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) *
        sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else None


def sensitivity_gate(rows, threshold):
    by_scenario_seed = defaultdict(list)
    for row in rows:
        if row["kind"] == "sensitivity":
            by_scenario_seed[(row["scenario"], row["seed"])].append(row)
    scenario_results = {}
    for scenario in sorted({key[0] for key in by_scenario_seed}):
        seed_results = []
        for (candidate, seed), group in sorted(by_scenario_seed.items()):
            if candidate != scenario:
                continue
            methods = {row["method"] for row in group}
            complete = all((row["completion_ratio"] or 0.0) >= 0.99 for row in group)
            spread = relative_spread([row["p95_fct_us"] for row in group])
            reward_spread = relative_spread([row["rollout_mean_reward"] for row in group])
            paired = [row for row in group if row["p95_fct_us"] is not None and row["rollout_mean_reward"] is not None]
            reward_fct_rank_correlation = None
            if len(paired) >= 2:
                reward_fct_rank_correlation = correlation(
                    rank_values([row["rollout_mean_reward"] for row in paired]),
                    rank_values([-row["p95_fct_us"] for row in paired]),
                )
            seed_results.append({
                "seed": seed,
                "methods": sorted(methods),
                "complete": complete,
                "p95_fct_spread": spread,
                "reward_spread": reward_spread,
                "reward_vs_lower_fct_spearman": reward_fct_rank_correlation,
                "passed": complete and len(methods) >= 3 and spread >= threshold,
            })
        passes = sum(item["passed"] for item in seed_results)
        scenario_results[scenario] = {
            "seeds": seed_results,
            "passed": bool(seed_results) and passes >= math.ceil(len(seed_results) / 2),
        }
    passed_scenarios = sum(result["passed"] for result in scenario_results.values())
    required = math.ceil(2 * len(scenario_results) / 3) if scenario_results else 1
    return {"passed": passed_scenarios >= required, "required_scenarios": required,
            "scenarios": scenario_results}


def baseline_gate(rows):
    by_scenario_seed = defaultdict(list)
    for row in rows:
        if row["kind"] == "baseline":
            by_scenario_seed[(row["scenario"], row["seed"])].append(row)
    scenario_results = {}
    for scenario in sorted({key[0] for key in by_scenario_seed}):
        seed_results = []
        for (candidate, seed), group in sorted(by_scenario_seed.items()):
            if candidate != scenario:
                continue
            methods = {row["method"] for row in group}
            complete = all((row["completion_ratio"] or 0.0) >= 0.99 for row in group)
            seed_results.append({
                "seed": seed,
                "methods": sorted(methods),
                "complete": complete,
                "passed": complete and methods == {"secn1", "secn2"},
            })
        passes = sum(item["passed"] for item in seed_results)
        scenario_results[scenario] = {
            "seeds": seed_results,
            "passed": bool(seed_results) and passes >= math.ceil(len(seed_results) / 2),
        }
    passed_scenarios = sum(result["passed"] for result in scenario_results.values())
    required = math.ceil(2 * len(scenario_results) / 3) if scenario_results else 1
    return {"passed": passed_scenarios >= required, "required_scenarios": required,
            "scenarios": scenario_results}


def effectiveness_gate(rows, improvement):
    lookup = {(row["kind"], row["scenario"], row["seed"], row["method"]): row for row in rows}
    scenario_results = {}
    scenarios = sorted({row["scenario"] for row in rows if row["kind"] == "eval"})
    for scenario in scenarios:
        seeds = sorted({row["seed"] for row in rows if row["kind"] == "eval" and row["scenario"] == scenario})
        seed_results = []
        for seed in seeds:
            greedy = lookup.get(("eval", scenario, seed, "greedy"))
            baselines = [
                row for row in rows
                if row["kind"] == "baseline" and row["scenario"] == scenario and row["seed"] == seed
            ]
            candidates = [row for row in baselines if row["p95_fct_us"] is not None]
            methods = {row["method"] for row in candidates}
            if not greedy or methods != {"secn1", "secn2"}:
                seed_results.append({"seed": seed, "passed": False,
                                     "reason": "missing SECN_1/SECN_2 baseline or evaluation"})
                continue
            best = min(candidates, key=lambda row: row["p95_fct_us"])
            complete = (greedy["completion_ratio"] or 0.0) >= 0.99 and all(
                (row["completion_ratio"] or 0.0) >= 0.99 for row in candidates
            )
            improves_any = any(
                greedy["p95_fct_us"] <= row["p95_fct_us"] * (1.0 - improvement)
                for row in candidates
            )
            near_best = greedy["p95_fct_us"] <= best["p95_fct_us"] * 1.05
            seed_results.append({
                "seed": seed,
                "greedy_p95_fct_us": greedy["p95_fct_us"],
                "secn1_p95_fct_us": next(row["p95_fct_us"] for row in candidates if row["method"] == "secn1"),
                "secn2_p95_fct_us": next(row["p95_fct_us"] for row in candidates if row["method"] == "secn2"),
                "best_paper_baseline": best["method"],
                "best_paper_baseline_p95_fct_us": best["p95_fct_us"],
                "complete": complete,
                "improves_at_least_one_baseline": improves_any,
                "near_best_paper_baseline": near_best,
                "passed": complete and improves_any and near_best,
            })
        passes = sum(item["passed"] for item in seed_results)
        scenario_results[scenario] = {
            "seeds": seed_results,
            "passed": bool(seed_results) and passes >= math.ceil(len(seed_results) / 2),
        }
    passed_scenarios = sum(result["passed"] for result in scenario_results.values())
    required = math.ceil(2 * len(scenario_results) / 3) if scenario_results else 1
    return {"passed": passed_scenarios >= required, "required_scenarios": required,
            "scenarios": scenario_results}


def write_report(path, rows, sensitivity, baselines, effectiveness):
    lines = ["# ACC validation report", "", "## Decision", ""]
    kinds = {row["kind"] for row in rows}
    def status(result, available):
        return "NOT RUN" if not available else ("PASS" if result["passed"] else "FAIL")
    lines.append(f"- Action sensitivity: **{status(sensitivity, 'sensitivity' in kinds)}**")
    lines.append(f"- Paper static baselines complete: **{status(baselines, 'baseline' in kinds)}**")
    lines.append(f"- Learned ACC effectiveness: **{status(effectiveness, 'eval' in kinds)}**")
    lines.extend(["", "The gate requires p95 FCT action sensitivity in at least two thirds of scenarios, "
                  "complete SECN_1/SECN_2 paper baselines, and greedy ACC to improve at least one paper "
                  "baseline by 5% while staying within 5% of the better paper baseline.", "", "## Measurements", "",
                  "| Phase | Scenario | Seed | Method | Stop (s) | Completion | p95 FCT (us) | p99 slowdown | Reward |",
                  "|---|---|---:|---|---:|---:|---:|---:|---:|"])
    for row in sorted(rows, key=lambda item: (item["kind"], item["scenario"], item["seed"], item["method"])):
        def fmt(value, digits=4):
            return "n/a" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {row['kind']} | {row['scenario']} | {row['seed']} | {row['method']} | "
            f"{fmt(row['simulator_stop_time'], 2)} | {fmt(row['completion_ratio'])} | {fmt(row['p95_fct_us'], 2)} | "
            f"{fmt(row['p99_slowdown'], 3)} | {fmt(row['rollout_mean_reward'])} |"
        )
    lines.extend(["", "## Gate details", "", "```json",
                  json.dumps({"sensitivity": sensitivity, "baseline": baselines,
                              "effectiveness": effectiveness}, indent=2),
                  "```", ""])
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--baseline-run-dir", type=Path, default=None,
                        help="Optional run directory supplying action sweeps and paper baselines.")
    parser.add_argument("--stage", choices=("sensitivity", "baseline", "effectiveness", "all"), default="all")
    parser.add_argument("--sensitivity-threshold", type=float, default=0.05)
    parser.add_argument("--improvement-threshold", type=float, default=0.05)
    parser.add_argument("--gate", action="store_true", help="exit non-zero when the selected gate fails")
    args = parser.parse_args()

    runs = {}
    baseline_run_dir = args.baseline_run_dir or args.run_dir
    if args.stage in ("sensitivity", "all"):
        runs.update(load_runs(baseline_run_dir / "sensitivity", "sensitivity"))
    if args.stage in ("baseline", "effectiveness", "all"):
        runs.update(load_runs(baseline_run_dir / "baseline", "baseline"))
    if args.stage in ("effectiveness", "all"):
        runs.update(load_runs(args.run_dir / "eval", "eval"))
    rows = build_rows(runs)
    sensitivity = sensitivity_gate(rows, args.sensitivity_threshold)
    baselines = baseline_gate(rows)
    effectiveness = effectiveness_gate(rows, args.improvement_threshold)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (args.run_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    result = {"sensitivity": sensitivity, "baseline": baselines, "effectiveness": effectiveness}
    (args.run_dir / "analysis.json").write_text(json.dumps(result, indent=2))
    write_report(args.run_dir / "REPORT.md", rows, sensitivity, baselines, effectiveness)

    selected_passed = {
        "sensitivity": sensitivity["passed"],
        "baseline": baselines["passed"],
        "effectiveness": effectiveness["passed"],
    }.get(args.stage, False)
    if args.stage == "all":
        selected_passed = sensitivity["passed"] and baselines["passed"] and effectiveness["passed"]
    print(f"Analysis written to {args.run_dir / 'REPORT.md'}")
    print(f"Selected gate: {'PASS' if selected_passed else 'FAIL'}")
    if args.gate and not selected_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
