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
        identity["complete_epoch"] = (
            str(records[-1].get("terminal") or "").lower() == "true"
        )
        completed = sum(int(row.get("step_completed_flows") or 0) for row in records)
        fct_sum = sum(float(row.get("step_fct_sum_us") or 0) for row in records)
        summaries.append({
            **identity,
            "environment_steps": len(records),
            "completed_flows": completed,
            "mean_fct_us": fct_sum / completed if completed else None,
            "final_cumulative_mean_fct_us": optional_number(records[-1].get("cumulative_mean_fct_us")),
            "final_cumulative_mean_slowdown": optional_number(
                records[-1].get("cumulative_mean_slowdown")
            ),
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

        # A trace file is one simulator episode, which is the experiment's
        # training-epoch unit.  Keep a training-only table with an explicit
        # name so it is not confused with the per-environment-step records.
        epoch_rows = [row for row in summaries
                      if str(row["phase"]).startswith("train")]
        epoch_summary_path = root / "fct_epoch_summary.csv"
        with epoch_summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(epoch_rows)

    plot_path = root / "fct_training_curves.png"
    if training_points:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return summaries, None
        methods = [value for value in ("acc_local", "sor")
                   if any(key[0] == value for key in training_points)]
        figure, axes = plt.subplots(len(methods), 2,
                                    figsize=(12, 4.2 * len(methods)),
                                    squeeze=False, sharey=False)
        for (method, arm, phase), buckets in sorted(training_points.items()):
            if method not in methods or arm not in ("aa", "ab"):
                continue
            x = sorted(key for key in buckets if isinstance(key, int))
            y = [buckets[step] / buckets[("count", step)] for step in x]
            row = methods.index(method)
            column = 0 if arm == "aa" else 1
            axes[row, column].plot(x, y, linewidth=1.1, alpha=0.85,
                                   label=f"{method}-{arm.upper()}")
        for row, method in enumerate(methods):
            axes[row, 0].set_title(f"{method}: AA / Task-A continuation")
            axes[row, 1].set_title(f"{method}: AB / Task-B training")
            for axis in axes[row]:
                axis.set_xlabel("Global optimizer update")
                axis.set_ylabel("Mean FCT of flows completed at update (us)")
                axis.grid(alpha=0.25)
                if axis.lines:
                    axis.legend()
        figure.suptitle("Per-step FCT by method and workload")
        figure.tight_layout()
        figure.savefig(plot_path, dpi=180)
        plt.close(figure)

        # Per-epoch view.  Mean FCT is exact because sums and completed-flow
        # counts are accumulated across every environment step in the epoch.
        # The p95 series is the rolling p95 at the end of the epoch; exact
        # full-epoch p95 requires retaining per-flow samples or a mergeable
        # quantile sketch and must not be inferred from step-level p95 values.
        epoch_rows = [row for row in summaries
                      if str(row["phase"]).startswith("train")]
        epoch_plot = root / "fct_epoch_curves.png"
        figure, axes = plt.subplots(3, 2, figsize=(13, 10), squeeze=False)
        arms = (("aa", "AA / Task-A continuation"),
                ("ab", "AB / Task-B training"))
        metrics = (
            ("mean_fct_us", "Exact epoch mean FCT (us)"),
            ("final_rolling_p95_fct_us", "End-of-epoch rolling p95 FCT (us)"),
            ("completed_flows", "Completed flows in epoch"),
        )
        for column, (arm, title) in enumerate(arms):
            for method in methods:
                selected = sorted(
                    (row for row in epoch_rows
                     if row["method"] == method and row["arm"] == arm),
                    key=lambda row: row["episode"],
                )
                x = list(range(1, len(selected) + 1))
                for row, (metric, ylabel) in enumerate(metrics):
                    y = [item[metric] if item["complete_epoch"] else float("nan")
                         for item in selected]
                    axes[row, column].plot(
                        x, y,
                        linewidth=1.2, label=method,
                    )
                    partial = [(index, item[metric])
                               for index, item in zip(x, selected)
                               if not item["complete_epoch"]]
                    if partial:
                        axes[row, column].scatter(
                            [item[0] for item in partial],
                            [item[1] for item in partial],
                            marker="x", s=32,
                            label=(f"{method} partial" if row == 0 else None),
                        )
                    axes[row, column].set_ylabel(ylabel)
                    axes[row, column].grid(alpha=.25)
            axes[0, column].set_title(title)
            axes[2, column].set_xlabel("Branch training epoch")
            axes[0, column].legend()
        figure.suptitle("FCT statistics by training epoch")
        figure.tight_layout()
        figure.savefig(epoch_plot, dpi=180)
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
