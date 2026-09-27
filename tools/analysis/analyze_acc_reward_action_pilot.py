#!/usr/bin/env python3
"""Build a cross-variant report for the ACC reward/action pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_variant(prefix, variant):
    run_id = f"{prefix}_{variant}"
    run_dir = ROOT / "experiments" / "aba" / run_id
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    metrics = run_dir / "summaries" / "acc_epoch_metrics.csv"
    if not metrics.exists():
        metrics = (
            ROOT / "simulation" / "output" / "aba" / run_id /
            "acc" / "epoch_metrics.csv"
        )
    with metrics.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["global_epoch"]))
    if not rows:
        raise RuntimeError(f"{run_id} has no completed epoch rows")
    return run_id, protocol, rows


def window_mean(rows, field, first):
    selected = rows[:10] if first else rows[-10:]
    values = [number(row.get(field)) for row in selected]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def percent_change(before, after):
    if before in (None, 0) or after is None:
        return None
    return 100.0 * (after / before - 1.0)


def fmt(value, digits=3, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def make_plots(output, data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Plot generation unavailable: {exc}"]
    output.mkdir(parents=True, exist_ok=True)

    def plot(filename, specs, log_fields=()):
        fig, axes = plt.subplots(len(specs), 1, figsize=(12, 3.4 * len(specs)), sharex=True)
        if len(specs) == 1:
            axes = [axes]
        for axis, (field, label) in zip(axes, specs):
            for variant, (_, protocol, rows) in data.items():
                points = [
                    (int(row["global_epoch"]), number(row.get(field)))
                    for row in rows
                ]
                points = [point for point in points if point[1] is not None]
                if points:
                    legend = (
                        f"{variant}: {protocol['reward_profile']} / "
                        f"{protocol['action_space']}"
                    )
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        label=legend, linewidth=1.5,
                    )
            axis.set_ylabel(label)
            axis.grid(alpha=.25)
            if field in log_fields:
                axis.set_yscale("log")
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("Training epoch")
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    plot("reward_loss.png", [
        ("reward", "Native training reward"),
        ("loss", "TD loss"),
    ], {"loss"})
    plot("fct_curves.png", [
        ("mean_fct_us", "Mean FCT (us)"),
        ("p95_fct_us", "p95 FCT (us)"),
        ("p99_fct_us", "p99 FCT (us)"),
    ])
    plot("network_curves.png", [
        ("completion_ratio", "Completion ratio"),
        ("mean_throughput_mbps", "Mean throughput (Mbps)"),
        ("mean_queue_kb", "Mean queue (KB)"),
        ("mean_ecn_rate", "Mean ECN rate"),
        ("pfc_duty", "PFC duty"),
    ])
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--variants", required=True)
    args = parser.parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    data = {variant: read_variant(args.run_id_prefix, variant) for variant in variants}
    output = ROOT / "experiments" / "aba" / f"{args.run_id_prefix}_comparison"
    figures = output / "figures"
    notes = make_plots(figures, data)

    output.mkdir(parents=True, exist_ok=True)
    combined = output / "epoch_metrics.csv"
    sample_rows = next(iter(data.values()))[2]
    fields = ["variant", "action_space", "reward_profile"] + list(sample_rows[0])
    with combined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, (_, protocol, rows) in data.items():
            for row in rows:
                writer.writerow({
                    "variant": variant,
                    "action_space": protocol["action_space"],
                    "reward_profile": protocol["reward_profile"],
                    **row,
                })

    metrics = (
        ("reward", "Reward"),
        ("loss", "Loss"),
        ("mean_fct_us", "Mean FCT"),
        ("p95_fct_us", "p95 FCT"),
        ("p99_fct_us", "p99 FCT"),
        ("completion_ratio", "Completion"),
        ("mean_throughput_mbps", "Throughput"),
        ("mean_queue_kb", "Queue"),
        ("mean_ecn_rate", "ECN"),
    )
    lines = [
        "# ACC reward/action-space pilot", "",
        "- Scope: single-task ACC acquisition on the canonical Task A traffic.",
        "- Comparison: first 10 epochs versus last 10 epochs.",
        "- Native rewards from tail_safe and weighted have different scales and are not compared across profiles.",
        "- FCT and physical metrics are per-epoch training trajectories, not frozen-policy evaluations.",
        "", "## Configuration", "",
        "| Variant | Action space | Reward | Epochs |", "|---|---|---|---:|",
    ]
    for variant, (_, protocol, rows) in data.items():
        lines.append(
            f"| {variant} | {protocol['action_space']} | "
            f"{protocol['reward_profile']} | {len(rows)} |"
        )
    lines += [
        "", "## First-10 to last-10 changes", "",
        "| Variant | Reward | Loss | Mean FCT | p95 FCT | p99 FCT | Completion pp | Throughput | Queue | ECN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, (_, _, rows) in data.items():
        values = {}
        for field, _ in metrics:
            before = window_mean(rows, field, True)
            after = window_mean(rows, field, False)
            values[field] = (
                (after - before) * 100.0
                if field == "completion_ratio" and before is not None and after is not None
                else percent_change(before, after)
            )
        lines.append(
            f"| {variant} | {fmt(values['reward'],2,'%')} | "
            f"{fmt(values['loss'],2,'%')} | {fmt(values['mean_fct_us'],2,'%')} | "
            f"{fmt(values['p95_fct_us'],2,'%')} | {fmt(values['p99_fct_us'],2,'%')} | "
            f"{fmt(values['completion_ratio'],4)} | {fmt(values['mean_throughput_mbps'],2,'%')} | "
            f"{fmt(values['mean_queue_kb'],2,'%')} | {fmt(values['mean_ecn_rate'],2,'%')} |"
        )
    lines += [
        "", "## Figures", "",
        "- `figures/reward_loss.png`",
        "- `figures/fct_curves.png`",
        "- `figures/network_curves.png`",
    ]
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ACC pilot comparison written to {output}")


if __name__ == "__main__":
    main()
