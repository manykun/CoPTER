#!/usr/bin/env python3
"""Check whether stage-0 experiments show scenario-dependent reward separation."""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


def load_records(path):
    if not os.path.isfile(path):
        raise ValueError("input does not exist: %s" % path)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if not text.strip():
            raise ValueError("input is empty: %s" % path)
        if path.lower().endswith(".jsonl"):
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            value = json.loads(text)
            if isinstance(value, list):
                values = value
            elif isinstance(value, dict):
                values = next((value[key] for key in ("records", "results", "data")
                               if isinstance(value.get(key), list)), [value])
            else:
                raise ValueError("JSON root must be an object or array")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot parse %s: %s" % (path, exc))
    if not values or not all(isinstance(row, dict) for row in values):
        raise ValueError("input contains no object records")
    return values


def field(row, names, required=True):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    if required:
        raise ValueError("record is missing one of: %s" % ", ".join(names))
    return None


def number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s is not numeric: %r" % (label, value))
    if not math.isfinite(result):
        raise ValueError("%s is not finite: %r" % (label, value))
    return result


def percentile(values, fraction):
    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def load_flow_count(path):
    if not os.path.isfile(path):
        raise ValueError("flow input does not exist: %s" % path)
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        rows = sum(1 for line in handle if line.strip())
    try:
        declared = int(first)
    except ValueError:
        raise ValueError("flow input has invalid declared count: %s" % path)
    if declared <= 0 or rows != declared:
        raise ValueError("flow input count mismatch in %s: declared=%d rows=%d" %
                         (path, declared, rows))
    return declared


def load_fct_files(named_paths, expected_count=None):
    flows_by_name = {}
    for name, path in named_paths.items():
        if not os.path.isfile(path):
            raise ValueError("FCT input does not exist: %s" % path)
        flows = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) < 8:
                    raise ValueError("%s:%d has fewer than 8 columns" % (path, line_number))
                try:
                    measured_fct = float(parts[6])
                    standalone_fct = float(parts[7])
                except ValueError:
                    raise ValueError("%s:%d has invalid FCT columns" % (path, line_number))
                if not math.isfinite(measured_fct) or not math.isfinite(standalone_fct):
                    raise ValueError("%s:%d has non-finite FCT" % (path, line_number))
                if standalone_fct <= 0:
                    raise ValueError("%s:%d has non-positive standalone FCT" % (path, line_number))
                key = tuple(parts[:6])
                if key in flows:
                    raise ValueError("%s:%d has duplicate flow identity" % (path, line_number))
                flows[key] = (measured_fct, standalone_fct)
        if not flows:
            raise ValueError("FCT input has no flows: %s" % path)
        if expected_count is not None and len(flows) > expected_count:
            raise ValueError("FCT input has more completed flows than scheduled: %s" % path)
        flows_by_name[name] = flows
    common = set.intersection(*(set(flows) for flows in flows_by_name.values()))
    if not common:
        raise ValueError("FCT inputs have no common completed flows")
    result = {}
    for name, flows in flows_by_name.items():
        measured = [flows[key][0] for key in common]
        slowdowns = [flows[key][0] / flows[key][1] for key in common]
        result[name] = {
            "avg_fct": sum(measured) / len(measured),
            "p95_fct": percentile(measured, 0.95),
            "avg_slowdown": sum(slowdowns) / len(slowdowns),
            "p95_slowdown": percentile(slowdowns, 0.95),
            "common_flows": len(common),
            "completed_flows": len(flows),
            "scheduled_flows": expected_count,
            "completion_rate": (len(flows) / expected_count
                                if expected_count is not None else None),
            "unfinished_flows": (expected_count - len(flows)
                                 if expected_count is not None else None),
        }
    return result


def parse_assignments(items, separator="="):
    result = {}
    for item in items or []:
        if separator not in item:
            raise ValueError("expected NAME%sPATH, got %r" % (separator, item))
        name, path = item.split(separator, 1)
        if not name or not path or name in result:
            raise ValueError("invalid or duplicate assignment: %r" % item)
        result[name] = path
    return result


def write_outputs(prefix, report, rows, fieldnames, plot=None):
    parent = os.path.dirname(os.path.abspath(prefix))
    if not os.path.isdir(parent):
        raise ValueError("output directory does not exist: %s" % parent)
    with open(prefix + ".json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(prefix + ".csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return False
        plot(plt)
        plt.tight_layout()
        plt.savefig(prefix + ".png", dpi=160)
        plt.close()
        return True
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="stage-0 JSON or JSONL metrics")
    parser.add_argument("--output-prefix", required=True, help="path prefix for JSON/CSV/PNG")
    parser.add_argument("--scenario-a", default="A")
    parser.add_argument("--scenario-b", default="B")
    parser.add_argument("--min-difference", type=float, default=0.10,
                        help="minimum relative common-flow slowdown spread")
    parser.add_argument("--min-completion-difference", type=float, default=0.03,
                        help="minimum completion-rate spread in a sensitive scenario")
    parser.add_argument("--completion-tie", type=float, default=0.01,
                        help="completion-rate margin within which slowdown breaks a tie")
    parser.add_argument("--flow", action="append", default=[], metavar="SCENARIO=PATH",
                        help="scheduled flow trace; repeat for each scenario")
    parser.add_argument("--fct", action="append", default=[], metavar="SCENARIO:CANDIDATE=PATH",
                        help="completed-flow FCT output; repeat for each candidate")
    args = parser.parse_args(argv)
    try:
        records = load_records(args.input)
        grouped = defaultdict(list)
        for row in records:
            scenario = str(field(row, ("scenario", "scen", "environment", "env")))
            candidate = str(field(row, ("candidate", "action", "policy", "method", "config")))
            reward = number(field(row, ("reward", "mean_reward", "avg_reward", "return")), "reward")
            grouped[(scenario, candidate)].append(reward)
        summaries = defaultdict(dict)
        rows = []
        for (scenario, candidate), rewards in sorted(grouped.items()):
            mean = sum(rewards) / len(rewards)
            summaries[scenario][candidate] = mean
            rows.append({"scenario": scenario, "candidate": candidate,
                         "mean_reward": mean, "samples": len(rewards)})
        required = (args.scenario_a, args.scenario_b)
        for scenario in required:
            if scenario not in summaries or len(summaries[scenario]) < 2:
                raise ValueError("scenario %r needs at least two candidates" % scenario)
        flow_paths = parse_assignments(args.flow)
        scenario_results = {}
        for scenario in required:
            values = summaries[scenario]
            if scenario not in flow_paths:
                raise ValueError("scenario %r needs --flow %s=PATH" % (scenario, scenario))
            scheduled = load_flow_count(flow_paths[scenario])
            named_fct = {}
            for item in args.fct:
                try:
                    name, path = item.split("=", 1)
                    item_scenario, _ = name.split(":", 1)
                except ValueError:
                    raise ValueError("invalid --fct %r; expected SCENARIO:CANDIDATE=PATH" % item)
                if item_scenario == scenario:
                    named_fct[name] = path
            fct_summary = load_fct_files(named_fct, expected_count=scheduled)
            candidate_stats = {name.split(":", 1)[1]: stats
                               for name, stats in fct_summary.items()
                               if name.startswith(scenario + ":")}
            if len(candidate_stats) < 2:
                raise ValueError("scenario %r needs at least two --fct candidates" % scenario)
            rates = {name: stats["completion_rate"] for name, stats in candidate_stats.items()}
            slowdowns = {name: stats["avg_slowdown"] for name, stats in candidate_stats.items()}
            max_rate = max(rates.values())
            contenders = [name for name, rate in rates.items()
                          if max_rate - rate <= args.completion_tie]
            best = min(contenders, key=lambda name: slowdowns[name])
            completion_spread = max(rates.values()) - min(rates.values())
            slowdown_low, slowdown_high = min(slowdowns.values()), max(slowdowns.values())
            slowdown_spread = ((slowdown_high - slowdown_low) /
                               max(abs(slowdown_low), 1e-12))
            completion_pass = completion_spread >= args.min_completion_difference
            slowdown_pass = slowdown_spread >= args.min_difference
            scenario_results[scenario] = {
                "best_candidate": best,
                "selection_rule": "maximize completion rate; use slowdown within completion tie margin",
                "completion_rate_spread": completion_spread,
                "slowdown_relative_spread": slowdown_spread,
                "completion_difference_pass": completion_pass,
                "slowdown_difference_pass": slowdown_pass,
                "difference_pass": completion_pass or slowdown_pass,
                "fct": candidate_stats,
                "mean_rewards": values,
            }
        best_different = scenario_results[required[0]]["best_candidate"] != scenario_results[required[1]]["best_candidate"]
        passed = best_different and all(item["difference_pass"] for item in scenario_results.values())
        report = {"analysis": "stage0_completion_aware_gate", "pass": passed,
                  "thresholds": {"min_completion_rate_difference": args.min_completion_difference,
                                 "min_relative_slowdown_difference": args.min_difference,
                                 "completion_tie_margin": args.completion_tie,
                                 "scenario_passes_on_either_metric": True,
                                 "best_candidate_must_differ": True},
                  "best_candidates_differ": best_different, "scenarios": scenario_results}
        def plot(plt):
            candidates = sorted(set().union(*(scenario_results[s]["fct"] for s in required)))
            figure = plt.gcf()
            figure.set_size_inches(10, 4)
            completion_axis, slowdown_axis = figure.subplots(1, 2)
            x = list(range(len(candidates)))
            width = 0.38
            for index, scenario in enumerate(required):
                offset = (index - 0.5) * width
                stats = scenario_results[scenario]["fct"]
                completion_axis.bar([value + offset for value in x],
                                    [stats[candidate]["completion_rate"] for candidate in candidates],
                                    width=width, label=scenario)
                slowdown_axis.bar([value + offset for value in x],
                                  [stats[candidate]["avg_slowdown"] for candidate in candidates],
                                  width=width, label=scenario)
            for axis, label in ((completion_axis, "Completion rate"),
                                (slowdown_axis, "Common-flow mean slowdown")):
                axis.set_xticks(x, candidates)
                axis.set_ylabel(label)
                axis.legend()
        report["png_written"] = write_outputs(args.output_prefix, report, rows,
                                               ["scenario", "candidate", "mean_reward", "samples"], plot)
        # Rewrite JSON so it records whether plotting was available.
        with open(args.output_prefix + ".json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps(report, sort_keys=True))
        return 0 if passed else 1
    except (ValueError, OSError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
