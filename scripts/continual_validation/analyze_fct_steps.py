#!/usr/bin/env python3
"""Summarize and plot per-environment-step FCT traces from a paper run."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def trace_identity(root, path, first):
    relative = path.relative_to(root)
    parts = relative.parts
    method = next((value for value in ("acc_local", "sor") if value in parts), "unknown")
    arm = next((value for value in ("aa", "ab") if value in parts), "a")
    return {
        "path": str(relative),
        "method": method,
        "arm": arm,
        "phase": first.get("phase") or "unknown",
        "episode": optional_number(first.get("episode"), int),
        "eval_tag": first.get("eval_tag") or "",
    }


def load_trace(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def optional_number(value, kind=float):
    return None if value in (None, "", "None") else kind(value)


def build_artifacts(root):
    root = Path(root).resolve()
    paths = sorted(root.rglob("fct_steps_ep*.csv"))
    summaries = []
    training_points = defaultdict(lambda: defaultdict(float))
    for path in paths:
        records = load_trace(path)
        if not records:
            continue
        identity = trace_identity(root, path, records[0])
        completed = sum(int(row.get("step_completed_flows") or 0) for row in records)
        fct_sum = sum(float(row.get("step_fct_sum_us") or 0) for row in records)
        summaries.append({
            **identity,
            "environment_steps": len(records),
            "completed_flows": completed,
            "mean_fct_us": fct_sum / completed if completed else None,
            "final_cumulative_mean_fct_us": optional_number(records[-1].get("cumulative_mean_fct_us")),
            "final_rolling_p95_fct_us": optional_number(records[-1].get("rolling_p95_fct_us")),
            "global_train_step_start": optional_number(records[0].get("global_train_step"), int),
            "global_train_step_end": optional_number(records[-1].get("global_train_step"), int),
        })
        if identity["phase"].startswith("train"):
            key = (identity["method"], identity["arm"], identity["phase"])
            for row in records:
                step = optional_number(row.get("global_train_step"), int)
                count = int(row.get("step_completed_flows") or 0)
                if step is None or not count:
                    continue
                bucket = training_points[key][step]
                # Keep exact FCT sum and flow count for each optimizer update.
                training_points[key][step] = bucket + float(row.get("step_fct_sum_us") or 0)
                training_points[key][("count", step)] += count

    summary_path = root / "fct_step_summary.csv"
    if summaries:
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

    plot_path = root / "fct_training_curves.png"
    if training_points:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return summaries, None
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
        phase_axes = {"a": axes[0], "aa": axes[1], "ab": axes[1]}
        for (method, arm, phase), buckets in sorted(training_points.items()):
            x = sorted(key for key in buckets if isinstance(key, int))
            y = [buckets[step] / buckets[("count", step)] for step in x]
            axis = phase_axes[arm]
            label = method if arm == "a" else f"{method}-{arm.upper()}"
            axis.plot(x, y, linewidth=1.1, alpha=0.85, label=label)
        axes[0].set_title("Task A training")
        axes[1].set_title("Matched AA / AB continuation")
        for axis in axes:
            axis.set_xlabel("Global optimizer update")
            axis.set_ylabel("Mean FCT of flows completed at update (us)")
            axis.grid(alpha=0.25)
            if axis.lines:
                axis.legend()
        figure.suptitle("Per-step FCT during ACC-local and SOR training")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=180)
        plt.close(figure)
    else:
        plot_path = None
    return summaries, plot_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summaries, plot = build_artifacts(args.run_dir)
    print(f"FCT step traces summarized: {len(summaries)} episodes")
    if plot is not None:
        print(f"FCT curve written to {plot}")


if __name__ == "__main__":
    main()
