#!/usr/bin/env python3
"""Parse ns3 .fct files and compare FCT statistics across runs.

FCT file format (copter-sim.cc qp_finish):
    src_ip(hex) dst_ip(hex) sport dport size(bytes) start_time(ns) fct(ns) standalone_fct(ns)

Usage:
    python3 analyze_fct.py label1=path1.fct label2=path2.fct ... [--json out.json]

Reports per-run: n_flows, avg FCT, P50/P95/P99 FCT, avg slowdown,
small-flow (<100KB) avg FCT, large-flow (>=1MB) avg FCT.
Slowdown = fct / standalone_fct (per flow, standalone is ideal no-congestion FCT).
"""
import sys
import json


def parse_fct(path):
    flows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                size = int(parts[4])
                fct = int(parts[6])
                sfct = int(parts[7])
            except ValueError:
                continue
            if fct <= 0 or sfct <= 0:
                continue
            flows.append((size, fct, sfct))
    return flows


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def stats(flows):
    if not flows:
        return None
    fcts = sorted(f for _, f, _ in flows)
    slowdowns = sorted(f / s for _, f, s in flows)
    small = [f for sz, f, _ in flows if sz < 100_000]
    large = [f for sz, f, _ in flows if sz >= 1_000_000]
    small_sd = sorted(f / s for sz, f, s in flows if sz < 100_000)
    return {
        "n_flows": len(flows),
        "avg_fct_us": sum(fcts) / len(fcts) / 1e3,
        "p50_fct_us": pct(fcts, 50) / 1e3,
        "p95_fct_us": pct(fcts, 95) / 1e3,
        "p99_fct_us": pct(fcts, 99) / 1e3,
        "avg_slowdown": sum(slowdowns) / len(slowdowns),
        "p99_slowdown": pct(slowdowns, 99),
        "small_avg_fct_us": (sum(small) / len(small) / 1e3) if small else None,
        "small_p99_slowdown": pct(small_sd, 99) if small_sd else None,
        "large_avg_fct_us": (sum(large) / len(large) / 1e3) if large else None,
    }


def main():
    args = [a for a in sys.argv[1:]]
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = args[i + 1]
        del args[i:i + 2]

    results = {}
    for spec in args:
        label, path = spec.split("=", 1)
        s = stats(parse_fct(path))
        results[label] = s

    cols = ["n_flows", "avg_fct_us", "p50_fct_us", "p95_fct_us", "p99_fct_us",
            "avg_slowdown", "p99_slowdown", "small_avg_fct_us", "small_p99_slowdown", "large_avg_fct_us"]
    header = f"{'label':<28}" + "".join(f"{c:>18}" for c in cols)
    print(header)
    for label, s in results.items():
        if s is None:
            print(f"{label:<28}  (no flows)")
            continue
        row = f"{label:<28}"
        for c in cols:
            v = s[c]
            row += f"{v:>18.2f}" if isinstance(v, float) else f"{v!s:>18}"
        print(row)

    if json_out:
        with open(json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved to {json_out}")


if __name__ == "__main__":
    main()
