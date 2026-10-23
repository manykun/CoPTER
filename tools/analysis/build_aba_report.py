#!/usr/bin/env python3
"""Build descriptive ABA ACC/SOR tables and curves from authoritative CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.metrics_core import ANALYSIS_VERSION, PERCENTILE_METHOD, sha256_file


def number(value):
    if value in (None, "", "None", "n/a"):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["global_epoch"] = int(row["global_epoch"])
        row["phase_epoch"] = int(row["phase_epoch"])
    return rows


def fmt(value, digits=3):
    value = number(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def phase_rows(rows, phase):
    return sorted((row for row in rows if row["phase"] == phase), key=lambda row: row["global_epoch"])


def delta(first, last, key):
    left, right = number(first.get(key)), number(last.get(key))
    if left is None or right is None:
        return None
    return right - left


def route_name(protocol):
    return str(protocol.get("route", "ABA")).upper()


def phase_name(protocol, phase):
    route = route_name(protocol)
    if len(route) == 3:
        return {
            "a1": f"{route[0]}1",
            "b": route[1],
            "a2": f"{route[2]}2",
        }.get(phase, phase.upper())
    return phase.upper()


def make_plots(figures: Path, by_method: dict, port_rows: dict, boundaries: tuple, route: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Plot generation unavailable: {exc}"]
    figures.mkdir(parents=True, exist_ok=True)
    colors = {"acc": "#1f77b4", "sor": "#d95f02"}

    def plot_group(filename, specs, log_fields=()):
        fig, axes = plt.subplots(len(specs), 1, figsize=(12, 3.5 * len(specs)), sharex=True)
        if len(specs) == 1:
            axes = [axes]
        for axis, (field, label) in zip(axes, specs):
            for method, rows in by_method.items():
                points = [(row["global_epoch"], number(row.get(field))) for row in rows]
                points = [(x, y) for x, y in points if y is not None]
                if points:
                    axis.plot([p[0] for p in points], [p[1] for p in points], label=method.upper(), color=colors.get(method))
            for boundary in boundaries[:-1]:
                axis.axvline(boundary + 0.5, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            if field in log_fields:
                axis.set_yscale("log")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels)
        axes[-1].set_xlabel(f"Global training epoch ({route[0]}1 -> {route[1]} -> {route[2]}2)")
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=180)
        plt.close(fig)

    plot_group("reward_loss.png", [("reward", "Training reward"), ("loss", "Training loss")], {"loss"})
    plot_group("fct_curves.png", [
        ("mean_fct_us", "Mean FCT (us)"), ("p95_fct_us", "p95 FCT (us)"),
        ("p99_fct_us", "p99 FCT (us)"), ("p99_slowdown", "p99 slowdown"),
    ])
    plot_group("network_curves.png", [
        ("completion_ratio", "Completion ratio"),
        ("mean_throughput_mbps", "Mean throughput (Mbps)"),
        ("mean_queue_kb", "Mean queue (KB)"),
        ("mean_ecn_rate", "Mean ECN rate"),
        ("pfc_duty", "PFC duty"),
    ])
    plot_group("q_replay_curves.png", [
        ("q_prediction_abs_max", "Q prediction abs max"),
        ("q_target_abs_max", "Q target abs max"),
        ("td_error_abs_p95_max", "TD error abs p95 max"),
        ("local_replay_size", "Local replay entries"),
        ("global_replay_size", "Global replay entries"),
    ])

    # Structured replay composition is recorded by SOR as JSON maps. The two
    # panels distinguish what remains in memory from how often retained
    # transitions have been sampled.
    for method, rows in by_method.items():
        decoded = []
        labels = set()
        for row in rows:
            item = {"epoch": row["global_epoch"]}
            for field in ("global_replay_task_sizes", "global_replay_task_sample_counts"):
                try:
                    value = json.loads(row.get(field) or "{}")
                except (TypeError, json.JSONDecodeError):
                    value = {}
                item[field] = {str(key): float(count) for key, count in value.items()}
                labels.update(item[field])
            decoded.append(item)
        if not labels:
            continue
        labels = sorted(labels, key=lambda value: (value not in ("a1", "b", "a2"), value))
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for axis, field, title in (
            (axes[0], "global_replay_task_sizes", "Retained replay composition"),
            (axes[1], "global_replay_task_sample_counts", "Cumulative sampling composition of retained entries"),
        ):
            x = [item["epoch"] for item in decoded]
            values = [[item[field].get(label, 0.0) for item in decoded] for label in labels]
            totals = [sum(series[index] for series in values) for index in range(len(x))]
            fractions = [
                [series[index] / totals[index] if totals[index] else 0.0 for index in range(len(x))]
                for series in values
            ]
            axis.stackplot(x, *fractions, labels=labels, alpha=.85)
            for boundary in boundaries[:-1]:
                axis.axvline(boundary + .5, color="black", linestyle="--", linewidth=1)
            axis.set_ylabel("Fraction")
            axis.set_ylim(0, 1)
            axis.set_title(title)
            axis.grid(alpha=.2)
        axes[0].legend(loc="upper left", ncol=max(1, len(labels)))
        axes[-1].set_xlabel("Global training epoch (A1 -> B -> A2)")
        fig.tight_layout()
        fig.savefig(figures / f"replay_composition_{method}.png", dpi=180)
        plt.close(fig)

    watched = (320, 321, 323, 345, 346, 347)
    fig, axes = plt.subplots(len(watched), 3, figsize=(15, 3.0 * len(watched)), sharex=True)
    for row_index, port in enumerate(watched):
        for method, rows in port_rows.items():
            selected = sorted(
                (row for row in rows if row.get("record_source") == "watch"
                 and str(row.get("agent_port")) == str(port)),
                key=lambda row: row["global_epoch"],
            )
            x = [row["global_epoch"] for row in selected]
            for column, field in enumerate(("port_reward", "port_loss", "port_q_prediction_abs_max")):
                points = [(epoch, number(item.get(field))) for epoch, item in zip(x, selected)]
                points = [(epoch, value) for epoch, value in points if value is not None]
                if points:
                    axes[row_index, column].plot(
                        [item[0] for item in points], [item[1] for item in points],
                        label=method.upper(), color=colors.get(method), linewidth=1,
                    )
                for boundary in boundaries[:-1]:
                    axes[row_index, column].axvline(boundary + .5, color="black", linestyle="--", linewidth=.6)
                axes[row_index, column].grid(alpha=.2)
        axes[row_index, 0].set_ylabel(f"Port {port}\nreward")
        axes[row_index, 1].set_ylabel("loss")
        axes[row_index, 2].set_ylabel("Q abs max")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels)
    for axis in axes[-1]:
        axis.set_xlabel("Global epoch")
    fig.suptitle("Representative watched-port training signals")
    fig.tight_layout()
    fig.savefig(figures / "representative_ports.png", dpi=180)
    plt.close(fig)

    phase_ends = {"a1": boundaries[0], "b": boundaries[1], "a2": boundaries[2]}
    metrics = (
        ("mean_queue_kb", "Mean queue (KB)", "physical"),
        ("mean_ecn_rate", "Mean ECN rate", "physical"),
        ("pfc_duty", "PFC duty", "physical"),
    )
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    for axis, (field, label, source) in zip(axes, metrics):
        samples, labels = [], []
        for method in by_method:
            for phase, epoch in phase_ends.items():
                values = [
                    number(row.get(field)) for row in port_rows[method]
                    if row.get("record_source") == source and row["global_epoch"] == epoch
                ]
                values = [value for value in values if value is not None]
                if values:
                    samples.append(values)
                    labels.append(f"{method.upper()}-{phase.upper()}")
        if samples:
            axis.boxplot(samples, labels=labels, showfliers=False)
            axis.tick_params(axis="x", rotation=45)
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Per-port distributions at phase boundaries")
    fig.tight_layout()
    fig.savefig(figures / "port_distributions.png", dpi=180)
    plt.close(fig)
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    output_root = root / "simulation" / "output" / "aba" / protocol["run_id"]
    by_method = {}
    port_rows = {}
    for method in protocol["methods"]:
        metrics_path = run_dir / "summaries" / f"{method}_epoch_metrics.csv"
        ports_path = run_dir / "summaries" / f"{method}_epoch_ports.csv"
        if not metrics_path.exists():
            metrics_path = output_root / method / "epoch_metrics.csv"
        if not ports_path.exists():
            ports_path = output_root / method / "epoch_ports.csv"
        rows = read_csv(metrics_path)
        rows.sort(key=lambda row: row["global_epoch"])
        by_method[method] = rows
        port_rows[method] = read_csv(ports_path)
    a1_end = int(protocol["epochs"]["a1"])
    b_end = a1_end + int(protocol["epochs"]["b"])
    total = b_end + int(protocol["epochs"]["a2"])
    route = route_name(protocol)
    notes = make_plots(
        run_dir / "figures", by_method, port_rows,
        (a1_end, b_end, total), route,
    )

    lines = [
        f"# {route} ACC/SOR training report", "",
        f"- Curriculum: **{protocol['tasks']['a']} -> {protocol['tasks']['b']} -> {protocol['tasks']['a']}**",
        f"- Epoch budget: **{protocol['epochs']['a1']} / {protocol['epochs']['b']} / {protocol['epochs']['a2']}**",
        "- Evaluation scope: per-epoch training trajectory; no frozen-policy evaluation.",
        "- Interpretation: descriptive adaptation and return-A recovery only; cross-task reward levels are not a standalone proof of catastrophic forgetting.",
        "", "## Data completeness", "",
        "| Method | Network rows | Expected | Unique epochs | Port rows |", "|---|---:|---:|---:|---:|",
    ]
    for method, rows in by_method.items():
        unique = len({row["global_epoch"] for row in rows})
        lines.append(f"| {method.upper()} | {len(rows)} | {total} | {unique} | {len(port_rows[method])} |")

    lines += ["", "## Phase-end metrics", "",
              "| Method | Phase | Epoch | Reward | Loss | Mean FCT us | p95 FCT us | p99 FCT us | Completion | Throughput Mbps | Queue KB | ECN | PFC duty |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, rows in by_method.items():
        for phase in ("a1", "b", "a2"):
            selected = phase_rows(rows, phase)
            if not selected:
                continue
            row = selected[-1]
            lines.append(
                f"| {method.upper()} | {phase_name(protocol, phase)} | {row['global_epoch']} | {fmt(row.get('reward'),4)} | "
                f"{fmt(row.get('loss'),4)} | {fmt(row.get('mean_fct_us'))} | {fmt(row.get('p95_fct_us'))} | "
                f"{fmt(row.get('p99_fct_us'))} | {fmt(row.get('completion_ratio'),4)} | "
                f"{fmt(row.get('mean_throughput_mbps'))} | {fmt(row.get('mean_queue_kb'))} | "
                f"{fmt(row.get('mean_ecn_rate'),6)} | {fmt(row.get('pfc_duty'),6)} |"
            )

    lines += ["", "## Within-phase changes", "",
              "| Method | Phase | Reward delta | Loss delta | p95 FCT delta us | Completion delta pp | Queue delta KB | ECN delta |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for method, rows in by_method.items():
        for phase in ("a1", "b", "a2"):
            selected = phase_rows(rows, phase)
            if not selected:
                continue
            first, last = selected[0], selected[-1]
            completion = delta(first, last, "completion_ratio")
            lines.append(
                f"| {method.upper()} | {phase_name(protocol, phase)} | {fmt(delta(first,last,'reward'),4)} | "
                f"{fmt(delta(first,last,'loss'),4)} | {fmt(delta(first,last,'p95_fct_us'))} | "
                f"{fmt(None if completion is None else completion*100,4)} | "
                f"{fmt(delta(first,last,'mean_queue_kb'))} | {fmt(delta(first,last,'mean_ecn_rate'),6)} |"
            )

    lines += ["", "## Runtime and resource use", "",
              "| Method | Total runtime h | Mean epoch s | Peak RSS MB | Final train step | Final epsilon |",
              "|---|---:|---:|---:|---:|---:|"]
    for method, rows in by_method.items():
        durations = [number(row.get("duration_seconds")) for row in rows]
        durations = [value for value in durations if value is not None]
        rss = [number(row.get("process_max_rss_mb")) for row in rows]
        rss = [value for value in rss if value is not None]
        last = rows[-1] if rows else {}
        lines.append(
            f"| {method.upper()} | {fmt(sum(durations)/3600 if durations else None,2)} | "
            f"{fmt(mean(durations) if durations else None,1)} | {fmt(max(rss) if rss else None,1)} | "
            f"{fmt(last.get('global_train_step'),0)} | {fmt(last.get('epsilon'),4)} |"
        )

    lines += ["", "## Q-value diagnostics", "",
              "| Method | Max inflated ports/epoch | Q prediction abs max | Q target abs max | Warning limit |",
              "|---|---:|---:|---:|---:|"]
    for method, rows in by_method.items():
        def maximum(field):
            values = [number(row.get(field)) for row in rows]
            values = [value for value in values if value is not None]
            return max(values) if values else None
        lines.append(
            f"| {method.upper()} | {fmt(maximum('q_inflated_ports'),0)} | "
            f"{fmt(maximum('q_prediction_abs_max'),4)} | {fmt(maximum('q_target_abs_max'),4)} | "
            f"{fmt(maximum('q_inflation_limit'),2)} |"
        )

    lines += ["", "## Structured replay diagnostics at phase ends", "",
              "| Method | Phase | Entries | Clusters | Boundary entries | Evictions | Max drift | Retained task mix | Sample-count mix |",
              "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for method, rows in by_method.items():
        for phase in ("a1", "b", "a2"):
            selected = phase_rows(rows, phase)
            if not selected:
                continue
            row = selected[-1]
            retained = row.get("global_replay_task_sizes") or "n/a"
            sampled = row.get("global_replay_task_sample_counts") or "n/a"
            lines.append(
                f"| {method.upper()} | {phase_name(protocol, phase)} | {fmt(row.get('global_replay_size'),0)} | "
                f"{fmt(row.get('global_replay_clusters'),0)} | "
                f"{fmt(row.get('global_replay_boundary_entries'),0)} | "
                f"{fmt(row.get('global_replay_evictions'),0)} | "
                f"{fmt(row.get('global_replay_drift_score_max'),4)} | `{retained}` | `{sampled}` |"
            )

    lines += ["", "## Figures", "",
              "- `figures/reward_loss.png`: per-epoch training reward and loss.",
              "- `figures/fct_curves.png`: exact per-epoch mean/p95/p99 FCT and p99 slowdown.",
              "- `figures/network_curves.png`: completion, throughput, queue, ECN, and PFC."]
    lines += ["- `figures/q_replay_curves.png`: Q/TD diagnostics and replay occupancy.",
              "- `figures/replay_composition_METHOD.png`: retained and sampled A1/B/A2 replay proportions when available.",
              "- `figures/representative_ports.png`: reward, loss, and Q for ports 320/321/323/345/346/347.",
              "- `figures/port_distributions.png`: physical-port queue, ECN, and PFC distributions at phase boundaries."]
    lines.extend(f"- {note}" for note in notes)
    lines += ["", "## Analysis provenance", "",
              f"- Analysis version: `{ANALYSIS_VERSION}`",
              f"- Percentile method: `{PERCENTILE_METHOD}`",
              "- Every epoch p95 is computed from that epoch's complete flow population; p95 values are never averaged across steps.",
              "- Detailed per-port values are in each method's `epoch_ports.csv`."]
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "analysis_version": ANALYSIS_VERSION,
        "metric_formula_version": ANALYSIS_VERSION,
        "percentile_method": PERCENTILE_METHOD,
        "rate_warmup_buckets": protocol.get("rate_warmup_buckets", 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(run_dir / "protocol.json"),
        "canonical_inputs": protocol.get("inputs", {}),
        "analyzers": {
            str(path.relative_to(root)): sha256_file(path)
            for path in (
                root / "tools" / "analysis" / "metrics_core.py",
                root / "tools" / "analysis" / "analyze_aba_epoch.py",
                root / "tools" / "analysis" / "build_aba_report.py",
            )
        },
        "inputs": {},
    }
    for method in protocol["methods"]:
        for name in ("epoch_metrics.csv", "epoch_ports.csv"):
            path = run_dir / "summaries" / f"{method}_{name}"
            if not path.exists():
                path = output_root / method / name
            if path.exists():
                provenance["inputs"][f"{method}/{name}"] = {
                    "path": str(path), "sha256": sha256_file(path)
                }
    (run_dir / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"ABA report written to {run_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
