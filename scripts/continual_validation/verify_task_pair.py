#!/usr/bin/env python3
"""Verify that a controlled A/B task pair changes timing, not flow identity."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean


def load_flows(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty flow file: {path}")
    declared = int(lines[0])
    flows = []
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(f"{path}:{line_number}: expected 6 columns")
        src, dst, pg, dport, size = map(int, fields[:5])
        flows.append(((src, dst, pg, dport, size), float(fields[5])))
    if declared != len(flows):
        raise ValueError(
            f"{path}: declared {declared} flows but found {len(flows)}"
        )
    return flows


def coefficient_of_variation(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    if average == 0:
        return 0.0
    variance = sum((value - average) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / average


def timing_summary(flows):
    starts = sorted(start for _, start in flows)
    gaps = [right - left for left, right in zip(starts, starts[1:])]
    simultaneous = Counter(starts)
    window_start = 0
    peak_flows_1us = 0
    for window_end, value in enumerate(starts):
        while value - starts[window_start] > 1e-6:
            window_start += 1
        peak_flows_1us = max(peak_flows_1us, window_end - window_start + 1)
    return {
        "first_start_s": starts[0] if starts else None,
        "last_start_s": starts[-1] if starts else None,
        "unique_start_times": len(simultaneous),
        "max_simultaneous_flows": max(simultaneous.values(), default=0),
        "peak_flows_in_1us": peak_flows_1us,
        "mean_interarrival_s": mean(gaps) if gaps else None,
        "interarrival_cv": coefficient_of_variation(gaps),
    }


def identity_digest(identities):
    payload = "\n".join(
        " ".join(map(str, identity))
        for identity in sorted(identities.elements())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def analyze(task_a, task_b):
    flows_a = load_flows(task_a)
    flows_b = load_flows(task_b)
    identities_a = Counter(identity for identity, _ in flows_a)
    identities_b = Counter(identity for identity, _ in flows_b)
    starts_a = [start for _, start in flows_a]
    starts_b = [start for _, start in flows_b]
    same_identity = identities_a == identities_b
    same_timing = starts_a == starts_b
    return {
        "task_a": str(task_a),
        "task_b": str(task_b),
        "flows_a": len(flows_a),
        "flows_b": len(flows_b),
        "same_flow_identity_multiset": same_identity,
        "same_timing_sequence": same_timing,
        "timing_only_shift": same_identity and not same_timing,
        "identity_sha256_a": identity_digest(identities_a),
        "identity_sha256_b": identity_digest(identities_b),
        "timing_a": timing_summary(flows_a),
        "timing_b": timing_summary(flows_b),
    }


def write_report(result, path):
    a = result["timing_a"]
    b = result["timing_b"]
    lines = [
        "# Controlled task-pair verification",
        "",
        f"- Same flow-identity multiset: **{result['same_flow_identity_multiset']}**",
        f"- Timing-only shift: **{result['timing_only_shift']}**",
        f"- Flow counts: **{result['flows_a']} / {result['flows_b']}**",
        "",
        "| Task | Unique start times | Peak flows in 1 us | Interarrival CV |",
        "|---|---:|---:|---:|",
        (
            f"| A | {a['unique_start_times']} | {a['peak_flows_in_1us']} | "
            f"{a['interarrival_cv']:.4f} |"
        ),
        (
            f"| B | {b['unique_start_times']} | {b['peak_flows_in_1us']} | "
            f"{b['interarrival_cv']:.4f} |"
        ),
        "",
        "The identity key is `(src,dst,pg,dport,size)`; start time is excluded.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-a", required=True, type=Path)
    parser.add_argument("--task-b", required=True, type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--require-identical-flows", action="store_true")
    parser.add_argument("--require-timing-shift", action="store_true")
    args = parser.parse_args()

    result = analyze(args.task_a, args.task_b)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        write_report(result, args.report_output)
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.require_identical_flows and not result["same_flow_identity_multiset"]:
        raise SystemExit("task pair changed flow identity; expected a timing-only shift")
    if args.require_timing_shift and not result["timing_only_shift"]:
        raise SystemExit("task pair does not contain a timing-only distribution shift")


if __name__ == "__main__":
    main()
