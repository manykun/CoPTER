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


def read_port_rows(run_id):
    path = (
        ROOT / "experiments" / "aba" / run_id / "summaries" /
        "acc_epoch_ports.csv"
    )
    if not path.exists():
        path = (
            ROOT / "simulation" / "output" / "aba" / run_id /
            "acc" / "epoch_ports.csv"
        )
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row.get("record_source") == "watch"
        ]


def read_frozen_diagnostics(run_id):
    path = (
        ROOT / "experiments" / "aba" / f"{run_id}_diagnostics" /
        "summary.json"
    )
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
                    # Keep all observations visible; trailing 10-epoch means
                    # reveal sustained trends without selecting endpoints.
                    if len(points) >= 10:
                        axis.plot([p[0] for p in points[9:]],
                                  [mean(p[1] for p in points[i-9:i+1]) for i in range(9, len(points))],
                                  linewidth=2.3, linestyle="--", label=f"{variant}: trailing 10")
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
    plot("replay_signal.png", [
        ("replay_congested_available_ratio", "Congested fraction in replay"),
        ("replay_congested_sample_ratio", "Congested fraction in minibatch"),
    ])
    if any("short_mean_fct_us" in rows[0] for _, _, rows in data.values()):
        for group in ("short", "medium", "long"):
            plot(f"fct_{group}.png", [(f"{group}_{metric}_fct_us", f"{group} {metric} FCT (us)") for metric in ("mean", "p95", "p99")])
        plot("reward_components.png", [("reward_throughput_mean", "Utilization term input"), ("reward_queue_mean", "Queue mapping score")])
        plot("q_epsilon.png", [("q_prediction_abs_max", "Q prediction abs max"), ("q_target_abs_max", "Q target abs max"), ("epsilon", "Epsilon")])
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
    fields = [
        "variant", "action_space", "reward_profile",
        "reward_queue_lambda", "reward_ecn_lambda",
    ] + list(sample_rows[0])
    with combined.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for variant, (_, protocol, rows) in data.items():
            for row in rows:
                writer.writerow({
                    "variant": variant,
                    "action_space": protocol["action_space"],
                    "reward_profile": protocol["reward_profile"],
                    "reward_queue_lambda": protocol["reward_queue_lambda"],
                    "reward_ecn_lambda": protocol["reward_ecn_lambda"],
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
        "- Native rewards with different formulas or penalty coefficients are not compared across arms; compare physical metrics.",
        "- FCT and physical metrics are per-epoch training trajectories, not frozen-policy evaluations.",
        "", "## Configuration", "",
        "| Variant | Action space | Reward | Queue / ECN lambda | Reward window | Replay sampling | Congested target | Epochs |",
        "|---|---|---|---|---:|---|---:|---:|",
    ]
    for variant, (_, protocol, rows) in data.items():
        lines.append(
            f"| {variant} | {protocol['action_space']} | "
            f"{protocol['reward_profile']} | "
            f"{protocol['reward_queue_lambda']:g} / {protocol['reward_ecn_lambda']:g} | "
            f"{protocol.get('reward_window_size', 32)} | "
            f"{protocol.get('replay_sampling', 'uniform')} | "
            f"{protocol.get('replay_congested_fraction', 0.5):.2f} | "
            f"{len(rows)} |"
        )
    if any(protocol["reward_profile"] in ("acc_paper", "acc_3816bc6_aligned") for _, protocol, _ in data.values()):
        lines += ["", "Both arms use action-interval time-mean queue bytes and byte-counter utilization; no frozen evaluation is scheduled.",
                  "acc_paper: 0.7 utilization + 0.3 queue score, thresholds 20*2^n KiB (n=0..9; above final threshold capped at score 0.1).",
                  "acc_3816bc6_aligned: historical 0.5/0.5 formula and 400-KiB normalization; interval average replaces the historic instantaneous queue input.",
                  "Size groups: short <=100 KiB; medium (100 KiB, 1 MiB); long >=1 MiB. Counts are completed flows in each epoch; no cross-epoch intersection.",
                  "Figures show every epoch plus trailing 10-epoch means. Reward queue-bin counts and exact interval timestamps are in watch traces and agent metrics."]
        lines += ["", "## Ten-epoch physical trend", "", "| Variant | Epochs | Mean FCT us | p95 FCT us | Completion | Throughput Mbps |", "|---|---|---:|---:|---:|---:|"]
        for variant, (_, _, rows) in data.items():
            for start in range(0, len(rows), 10):
                window = rows[start:start+10]
                values = [window_mean(window, key, True) for key in ("mean_fct_us", "p95_fct_us", "completion_ratio", "mean_throughput_mbps")]
                lines.append(f"| {variant} | {window[0]['global_epoch']}–{window[-1]['global_epoch']} | " + " | ".join(fmt(value) for value in values) + " |")
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
        "", "## Learning-signal diagnostics", "",
        "| Variant | Replay signal available first/last 10 | Signal sampled first/last 10 | Reference action greedy on watched ports (last 10) | Reference Q margin (last 10) |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, (run_id, _, rows) in data.items():
        available_first = window_mean(
            rows, "replay_congested_available_ratio", True
        )
        available_last = window_mean(
            rows, "replay_congested_available_ratio", False
        )
        sampled_first = window_mean(
            rows, "replay_congested_sample_ratio", True
        )
        sampled_last = window_mean(
            rows, "replay_congested_sample_ratio", False
        )
        port_rows = read_port_rows(run_id)
        final_epochs = {
            int(row["global_epoch"]) for row in rows[-10:]
        }
        final_ports = [
            row for row in port_rows
            if int(row["global_epoch"]) in final_epochs
        ]
        greedy_values = [
            number(row.get("reference_greedy_fraction"))
            for row in final_ports
        ]
        margin_values = [
            number(row.get("reference_margin_mean"))
            for row in final_ports
        ]
        greedy_values = [value for value in greedy_values if value is not None]
        margin_values = [value for value in margin_values if value is not None]
        lines.append(
            f"| {variant} | {fmt(available_first,3)} / {fmt(available_last,3)} | "
            f"{fmt(sampled_first,3)} / {fmt(sampled_last,3)} | "
            f"{fmt(mean(greedy_values) if greedy_values else None,3)} | "
            f"{fmt(mean(margin_values) if margin_values else None,4)} |"
        )

    frozen = {
        variant: read_frozen_diagnostics(run_id)
        for variant, (run_id, _, _) in data.items()
    }
    if any(value is not None for value in frozen.values()):
        lines += [
            "", "## Frozen-policy validation", "",
            "| Variant | Reward change | Mean FCT change | p95 FCT change | p99 FCT change | Completion change pp | Frozen repeats available |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for variant in data:
            summary = frozen.get(variant) or {}
            initial = summary.get("frozen_initial") or {}
            final = summary.get("frozen_final") or {}
            repeats_available = bool(initial and final)
            completion_before = number(initial.get("completion_ratio"))
            completion_after = number(final.get("completion_ratio"))
            completion_change = (
                100.0 * (completion_after - completion_before)
                if completion_before is not None and completion_after is not None
                else None
            )
            lines.append(
                f"| {variant} | "
                f"{fmt(percent_change(number(initial.get('reward')), number(final.get('reward'))),2,'%')} | "
                f"{fmt(percent_change(number(initial.get('mean_fct_us')), number(final.get('mean_fct_us'))),2,'%')} | "
                f"{fmt(percent_change(number(initial.get('p95_fct_us')), number(final.get('p95_fct_us'))),2,'%')} | "
                f"{fmt(percent_change(number(initial.get('p99_fct_us')), number(final.get('p99_fct_us'))),2,'%')} | "
                f"{fmt(completion_change,4)} | "
                f"{'yes' if repeats_available else 'no'} |"
            )
        baseline_summary = frozen.get("baseline") or {}
        reference = baseline_summary.get("fixed_high_gentle") or {}
        if reference:
            lines += [
                "",
                "Baseline fixed `high_gentle` reference: "
                f"mean FCT **{fmt(number(reference.get('mean_fct_us')))} us**, "
                f"p95 FCT **{fmt(number(reference.get('p95_fct_us')))} us**. "
                "The learned frozen policies should move toward this physical reference.",
            ]
    lines += [
        "", "## Figures", "",
        "- `figures/reward_loss.png`",
        "- `figures/fct_curves.png`",
        "- `figures/network_curves.png`",
        "- `figures/replay_signal.png`",
    ]
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ACC pilot comparison written to {output}")


if __name__ == "__main__":
    main()
