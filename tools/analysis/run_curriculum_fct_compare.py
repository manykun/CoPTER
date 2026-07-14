#!/usr/bin/env python3
"""Aggregate per-phase FCT/throughput files produced during the
catastrophic-forgetting curriculum (run_forgetting_curriculum.sh).

Expected layout under `simulation/output/<TASK>/`:
  <method>_<task>.fct          single-shot from the most recent eval (overwritten each run)
  <method>_<task>.throughput   ditto

Because the orchestrator currently overwrites the same files every time
ns3 reruns the same conf, this script must be paired with a small
post-eval copy step (handled by run_forgetting_curriculum.sh patch
optional) OR pointed at a directory tree the user prepares manually,
e.g.:
  output_archive/<method>/phase<N>/<task>/{<method>_<task>.fct,...}

This script implements both modes:
  --layout flat       expects current overwrite layout, only reports the
                      LAST eval (sanity check)
  --layout archived   expects output_archive/<method>/phase<N>/<task>/...
                      and produces an HxW heatmap of FCT-Avg per (phase, task)
                      for ACC vs SOR side-by-side.
"""
import argparse
import os
import re
import sys
from collections import defaultdict


def parse_overall_fct(path):
    """Return dict with keys avg/mid/p95/p99 (FCT in ns).

    Supports two formats:
      (a) ns3 raw per-flow log produced by copter-sim: each line has 8 columns
          (sip dip sport dport size start_ns fct_ns prio). The 7th column is
          the per-flow FCT in nanoseconds.
      (b) summary file (tools/analysis/analysis_fct.py output): one line with
          exactly 4 floats = avg / mid / p95 / p99 inside the first 5 lines.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            lines = f.readlines()
        # try summary format first (cheap)
        for line in lines[:5]:
            parts = line.strip().split()
            try:
                floats = [float(x) for x in parts]
            except ValueError:
                continue
            if len(floats) == 4:
                return {"avg": floats[0], "mid": floats[1], "p95": floats[2], "p99": floats[3]}
        # fall back: raw per-flow log -> compute summary on the fly
        fct_vals = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            try:
                fct_vals.append(float(parts[6]))
            except ValueError:
                continue
        if not fct_vals:
            return None
        fct_vals.sort()
        n = len(fct_vals)
        def pct(p):
            i = int(n * p)
            return fct_vals[i if i < n else n - 1]
        return {
            "avg": sum(fct_vals) / n,
            "mid": pct(0.5),
            "p95": pct(0.95),
            "p99": pct(0.99),
        }
    except Exception as exc:
        print(f"[WARN] cannot parse {path}: {exc}", file=sys.stderr)
    return None


def collect_archived(root, methods, tasks, max_phase):
    """root/<method>/phase<N>/<task>/<method>_<task>.fct"""
    table = defaultdict(dict)  # (method, task) -> {phase: avg}
    for m in methods:
        for p in range(1, max_phase + 1):
            for t in tasks:
                fct = os.path.join(root, m, f"phase{p}", t, f"{m}_{t}.fct")
                stats = parse_overall_fct(fct)
                if stats is None:
                    continue
                table[(m, t)][p] = stats["avg"]
    return table


def collect_flat(repo_root, methods, tasks):
    """Just read the current single-shot output/<task>/<method>_<task>.fct."""
    out = {}
    for m in methods:
        for t in tasks:
            fct = os.path.join(repo_root, "simulation", "output", t, f"{m}_{t}.fct")
            stats = parse_overall_fct(fct)
            if stats is not None:
                out[(m, t)] = stats
    return out


def print_archived_table(table, methods, tasks, max_phase):
    print("\n=== Per-phase FCT-Avg (lower better) -- archived layout ===")
    header = "  task / method                | " + " | ".join(f"phase{p}" for p in range(1, max_phase + 1))
    print(header)
    print("-" * len(header))
    for t in tasks:
        for m in methods:
            row = [f"  {t:<24} {m:<5}"]
            for p in range(1, max_phase + 1):
                v = table.get((m, t), {}).get(p)
                row.append(f"{v:7.3f}" if v is not None else "   N/A ")
            print(" | ".join(row))


def maybe_heatmap(out_dir, table, methods, tasks, max_phase):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"[INFO] matplotlib/numpy unavailable, skipping heatmap: {exc}")
        return
    os.makedirs(out_dir, exist_ok=True)
    for m in methods:
        grid = np.full((len(tasks), max_phase), np.nan)
        for i, t in enumerate(tasks):
            for p in range(1, max_phase + 1):
                v = table.get((m, t), {}).get(p)
                if v is not None:
                    grid[i, p - 1] = v
        fig, ax = plt.subplots(figsize=(1.5 + 1.0 * max_phase, 1.0 + 0.5 * len(tasks)))
        im = ax.imshow(grid, aspect="auto")
        ax.set_xticks(range(max_phase))
        ax.set_xticklabels([f"phase{p}" for p in range(1, max_phase + 1)])
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks)
        ax.set_title(f"{m.upper()} FCT-Avg across phases")
        for i in range(len(tasks)):
            for j in range(max_phase):
                v = grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="FCT Avg")
        path = os.path.join(out_dir, f"fct_heatmap_{m}.pdf")
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        print(f"[INFO] wrote {path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("flat", "archived"), default="flat")
    ap.add_argument("--archive_root", default=os.path.join(repo_root, "simulation", "output_archive"),
                    help="Used when --layout=archived; expected: <root>/<method>/phase<N>/<task>/<method>_<task>.fct")
    ap.add_argument("--methods", default="acc,sor")
    ap.add_argument("--tasks", default="Hadoop_Shuffle,AliStorage_AllReduce,CacheFollower_burst_incast")
    ap.add_argument("--phases", type=int, default=4)
    ap.add_argument("--plot_dir", default=os.path.join(repo_root, "tools", "analysis", "forgetting_fct"))
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    tasks = [t for t in args.tasks.split(",") if t]

    if args.layout == "flat":
        flat = collect_flat(repo_root, methods, tasks)
        print("\n=== Latest-run FCT (flat layout) ===")
        for m in methods:
            print(f"\n  [{m}]")
            for t in tasks:
                stats = flat.get((m, t))
                if stats is None:
                    print(f"    {t:<28} N/A")
                else:
                    print(f"    {t:<28} avg={stats['avg']:.3f} mid={stats['mid']:.3f} p95={stats['p95']:.3f} p99={stats['p99']:.3f}")
        return

    table = collect_archived(args.archive_root, methods, tasks, args.phases)
    print_archived_table(table, methods, tasks, args.phases)
    # Forgetting summary (FCT: lower is better, so forgetting = last - min).
    print("\n=== Forgetting on FCT-Avg (last_phase - min_phase, lower = less forgetting) ===")
    print(f"  {'task':<28} | " + " | ".join(f"{m:^14}" for m in methods))
    for t in tasks:
        row = [f"  {t:<28}"]
        for m in methods:
            phase_map = table.get((m, t), {})
            vals = [v for v in phase_map.values() if v is not None]
            if not vals:
                row.append(f"{'N/A':^14}")
                continue
            last_phase = max(phase_map.keys())
            forgetting = phase_map[last_phase] - min(vals)
            row.append(f"{forgetting:>13.2f} ")
        print(" | ".join(row))
    # Recovery on Hadoop revisit (phase4 vs phase1).
    print("\n=== Recovery on Hadoop_Shuffle revisit (phase1 / phase4, FCT lower better; >1 = recovered) ===")
    for m in methods:
        pm = table.get((m, "Hadoop_Shuffle"), {})
        v1, v4 = pm.get(1), pm.get(args.phases)
        if v1 is None or v4 is None or v4 == 0:
            print(f"  {m:<5} N/A")
        else:
            print(f"  {m:<5} {v1/v4:.4f}  (phase1={v1:.1f}, phase{args.phases}={v4:.1f})")
    maybe_heatmap(args.plot_dir, table, methods, tasks, args.phases)


if __name__ == "__main__":
    main()
