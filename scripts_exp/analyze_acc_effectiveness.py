#!/usr/bin/env python3
"""Evaluate ACC reward improvement and greedy FCT against static policies."""
import argparse
import json
import os
import sys
from collections import defaultdict

from check_stage0_gate import (field, load_fct_files, load_records, number,
                               parse_assignments, write_outputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="JSON or JSONL reward metrics")
    parser.add_argument("--greedy-fct", required=True, help="ACC greedy ns-3 .fct file")
    parser.add_argument("--static-fct", action="append", required=True, metavar="NAME=PATH",
                        help="static-policy .fct; repeat for at least two policies")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--acc-label", default="acc")
    parser.add_argument("--min-reward-improvement", type=float, default=0.15)
    parser.add_argument("--best-static-tolerance", type=float, default=0.03)
    parser.add_argument("--nonbest-improvement", type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        records = load_records(args.metrics)
        rewards = defaultdict(list)
        for row in records:
            try:
                label = str(field(row, ("label", "method", "policy", "phase", "stage", "eval_tag"))).lower()
            except ValueError:
                label = "train"
            rewards[label].append(number(field(row, ("reward", "rollout_mean_reward", "mean_reward", "avg_reward", "return")), "reward"))
        baseline_key, acc_key = args.baseline_label.lower(), args.acc_label.lower()
        if baseline_key not in rewards or acc_key not in rewards:
            train_values = rewards.get("train", [])
            if len(train_values) >= 2:
                rewards[baseline_key] = [train_values[0]]
                rewards[acc_key] = [train_values[-1]]
            else:
                raise ValueError("metrics need labels %r and %r or at least two training epochs" % (args.baseline_label, args.acc_label))
        reward_means = {name: sum(values) / len(values) for name, values in rewards.items()}
        baseline_reward, acc_reward = reward_means[baseline_key], reward_means[acc_key]
        reward_improvement = (acc_reward - baseline_reward) / max(abs(baseline_reward), 1e-12)

        static_paths = parse_assignments(args.static_fct)
        if len(static_paths) < 2:
            raise ValueError("at least two --static-fct policies are required")
        if "greedy" in static_paths:
            raise ValueError("static policy name 'greedy' is reserved")
        fct = load_fct_files(dict(static_paths, greedy=args.greedy_fct))
        best_static = min(static_paths, key=lambda name: fct[name]["avg_fct"])
        greedy_avg = fct["greedy"]["avg_fct"]
        within_best = greedy_avg <= fct[best_static]["avg_fct"] * (1 + args.best_static_tolerance)
        nonbest = [name for name in static_paths if name != best_static]
        improvements = {name: (fct[name]["avg_fct"] - greedy_avg) / fct[name]["avg_fct"] for name in nonbest}
        improves_nonbest = any(value >= args.nonbest_improvement for value in improvements.values())
        reward_pass = reward_improvement >= args.min_reward_improvement
        passed = reward_pass and within_best and improves_nonbest

        rows = []
        for name in sorted(fct):
            rows.append({"policy": name, "avg_fct": fct[name]["avg_fct"],
                         "p95_fct": fct[name]["p95_fct"],
                         "common_flows": fct[name]["common_flows"],
                         "relative_to_greedy": (fct[name]["avg_fct"] - greedy_avg) / fct[name]["avg_fct"]})
        report = {"analysis": "acc_effectiveness", "pass": passed,
                  "thresholds": {"min_reward_improvement": args.min_reward_improvement,
                                 "best_static_fct_tolerance": args.best_static_tolerance,
                                 "min_nonbest_fct_improvement": args.nonbest_improvement},
                  "reward": {"baseline_label": args.baseline_label, "acc_label": args.acc_label,
                             "baseline_mean": baseline_reward, "acc_mean": acc_reward,
                             "relative_improvement": reward_improvement, "pass": reward_pass},
                  "fct": {"best_static": best_static, "greedy_within_best_tolerance": within_best,
                          "nonbest_improvements": improvements,
                          "improves_at_least_one_nonbest": improves_nonbest, "policies": fct}}
        def plot(plt):
            names = sorted(fct)
            plt.bar(names, [fct[name]["avg_fct"] for name in names])
            plt.ylabel("Average FCT")
            plt.xticks(rotation=25, ha="right")
        report["png_written"] = write_outputs(args.output_prefix, report, rows,
                                               ["policy", "avg_fct", "p95_fct", "common_flows", "relative_to_greedy"], plot)
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
