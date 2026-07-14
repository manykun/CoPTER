#!/usr/bin/env python3
"""Detect forgetting from before/after rewards and FCT, comparing ACC with SOR."""
import argparse
import json
import sys

from check_stage0_gate import (field, load_fct_files, load_records, number,
                               parse_assignments, write_outputs)


def reward_pair(records, method):
    before, after = [], []
    for row in records:
        row_method = str(field(row, ("method", "algorithm", "model"))).lower()
        if row_method != method.lower():
            continue
        phase = str(field(row, ("phase", "checkpoint", "stage", "when"))).lower()
        reward = number(field(row, ("reward", "rollout_mean_reward", "mean_reward", "avg_reward", "return")), "reward")
        if phase in ("before", "pre", "initial", "old"):
            before.append(reward)
        elif phase in ("after", "post", "final", "new"):
            after.append(reward)
    if not before or not after:
        raise ValueError("method %r needs before and after reward records" % method)
    return sum(before) / len(before), sum(after) / len(after)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, help="JSON or JSONL reward metrics")
    parser.add_argument("--fct", action="append", required=True, metavar="METHOD:PHASE=PATH",
                        help="repeat for acc:before, acc:after, sor:before, sor:after")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--acc-label", default="acc")
    parser.add_argument("--sor-label", default="sor")
    parser.add_argument("--min-reward-drop", type=float, default=0.10)
    parser.add_argument("--min-fct-worsening", type=float, default=0.10)
    args = parser.parse_args(argv)
    try:
        records = load_records(args.metrics)
        methods = (args.acc_label.lower(), args.sor_label.lower())
        reward = {}
        for method in methods:
            before, after = reward_pair(records, method)
            reward[method] = {"before": before, "after": after,
                              "drop": (before - after) / max(abs(before), 1e-12)}

        assignments = parse_assignments(args.fct)
        expected = {method + ":" + phase for method in methods for phase in ("before", "after")}
        missing = expected - set(assignments)
        if missing:
            raise ValueError("missing --fct assignments: %s" % ", ".join(sorted(missing)))
        selected = {name: assignments[name] for name in expected}
        fct = load_fct_files(selected)
        results = {}
        rows = []
        for method in methods:
            before, after = fct[method + ":before"], fct[method + ":after"]
            avg_worsening = (after["avg_fct"] - before["avg_fct"]) / before["avg_fct"]
            p95_worsening = (after["p95_fct"] - before["p95_fct"]) / before["p95_fct"]
            reward_bad = reward[method]["drop"] >= args.min_reward_drop
            fct_bad = max(avg_worsening, p95_worsening) >= args.min_fct_worsening
            score = max(reward[method]["drop"], 0.0) + max(avg_worsening, p95_worsening, 0.0)
            results[method] = {"reward": reward[method], "avg_fct_worsening": avg_worsening,
                               "p95_fct_worsening": p95_worsening,
                               "reward_forgetting": reward_bad, "fct_forgetting": fct_bad,
                               "forgetting_detected": reward_bad and fct_bad,
                               "forgetting_score": score}
            rows.append({"method": method, "reward_before": reward[method]["before"],
                         "reward_after": reward[method]["after"], "reward_drop": reward[method]["drop"],
                         "avg_fct_before": before["avg_fct"], "avg_fct_after": after["avg_fct"],
                         "avg_fct_worsening": avg_worsening, "p95_fct_before": before["p95_fct"],
                         "p95_fct_after": after["p95_fct"], "p95_fct_worsening": p95_worsening,
                         "forgetting_detected": reward_bad and fct_bad, "forgetting_score": score})
        acc, sor = results[methods[0]], results[methods[1]]
        sor_lower = sor["forgetting_score"] < acc["forgetting_score"]
        passed = acc["forgetting_detected"] and sor_lower
        report = {"analysis": "forgetting", "pass": passed,
                  "thresholds": {"min_reward_drop": args.min_reward_drop,
                                 "min_avg_or_p95_fct_worsening": args.min_fct_worsening,
                                 "sor_forgetting_must_be_lower": True},
                  "sor_forgetting_lower": sor_lower, "methods": results,
                  "common_flows": next(iter(fct.values()))["common_flows"]}
        def plot(plt):
            names = list(methods)
            positions = range(len(names))
            width = 0.35
            plt.bar([x - width / 2 for x in positions], [results[name]["reward"]["drop"] for name in names],
                    width, label="Reward drop")
            plt.bar([x + width / 2 for x in positions],
                    [max(results[name]["avg_fct_worsening"], results[name]["p95_fct_worsening"]) for name in names],
                    width, label="FCT worsening")
            plt.xticks(list(positions), names)
            plt.ylabel("Relative change")
            plt.legend()
        report["png_written"] = write_outputs(args.output_prefix, report, rows,
                                               ["method", "reward_before", "reward_after", "reward_drop",
                                                "avg_fct_before", "avg_fct_after", "avg_fct_worsening",
                                                "p95_fct_before", "p95_fct_after", "p95_fct_worsening",
                                                "forgetting_detected", "forgetting_score"], plot)
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
