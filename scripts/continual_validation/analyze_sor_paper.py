#!/usr/bin/env python3
"""Descriptive matched-AA analysis for the ACC-local versus SOR paper run."""

import argparse
import csv
import json
import math
from pathlib import Path

from analyze_forgetting import load_eval, percentile, summarize
from analyze_fct_steps import build_artifacts as build_fct_step_artifacts
from analyze_port_continual import frozen_port_summary
from run_return_a import read, write


LOWER_BETTER = {"avg_fct_us", "p95_fct_us", "p99_slowdown"}
NETWORK_METRICS = (
    "reward", "throughput", "avg_fct_us", "p95_fct_us", "p99_slowdown",
    "completion_ratio", "queue", "ecn",
)


def common_summary(runs):
    common = set.intersection(*(set(run["flows"]) for run in runs))
    return [summarize(run, common) for run in runs], len(common)


def enrich(summary, run):
    metrics = run["metrics"]
    summary = dict(summary)
    summary["throughput"] = metrics.get("reward_avg_tx_rate_mean")
    summary["queue"] = metrics.get("reward_avg_queue_mean")
    summary["ecn"] = metrics.get("reward_avg_ecn_mean")
    return summary


def adjusted(base, aa, ab, metric):
    before, control, shifted = base.get(metric), aa.get(metric), ab.get(metric)
    if before is None or control is None or shifted is None:
        return None
    scale = max(abs(before), 1e-12)
    return ((shifted - control) / scale if metric in LOWER_BETTER
            else (control - shifted) / scale)


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def pct(value):
    return "n/a" if value is None else f"{100 * value:+.2f}%"


def network_analysis(root, protocol):
    a_end, b_end = protocol["a_points"][-1], protocol["b_points"][-1]
    task_a, task_b = protocol["manifest"]["task_a"], protocol["manifest"]["task_b"]
    rows = []
    for method in protocol["methods"]:
        locations = {
            "after_a": root / "eval" / method / f"a_{a_end}" / task_a,
            "aa": root / "eval" / method / f"aa_{b_end}" / task_a,
            "ab": root / "eval" / method / f"ab_{b_end}" / task_a,
        }
        runs = {name: load_eval(path) for name, path in locations.items()}
        summaries, common = common_summary(list(runs.values()))
        values = {name: enrich(summary, runs[name])
                  for name, summary in zip(runs, summaries)}
        for metric in NETWORK_METRICS:
            rows.append({
                "method": method, "task": task_a, "metric": metric,
                "common_flows": common, "after_a": values["after_a"].get(metric),
                "aa": values["aa"].get(metric), "ab": values["ab"].get(metric),
                "aa_corrected_degradation": adjusted(
                    values["after_a"], values["aa"], values["ab"], metric),
            })
        b_before = load_eval(root / "eval" / method / f"a_{a_end}" / task_b)
        b_after = load_eval(root / "eval" / method / f"ab_{b_end}" / task_b)
        b_summaries, b_common = common_summary([b_before, b_after])
        b_values = [enrich(value, run) for value, run in zip(b_summaries, [b_before, b_after])]
        for metric in NETWORK_METRICS:
            before, after = b_values[0].get(metric), b_values[1].get(metric)
            change = None if before is None or after is None else (after-before)/max(abs(before), 1e-12)
            rows.append({
                "method": method, "task": task_b, "metric": metric,
                "common_flows": b_common, "after_a": before, "aa": None,
                "ab": after, "aa_corrected_degradation": change,
            })
    return rows


def acquisition_analysis(root, protocol):
    final = protocol["a_points"][-1]
    task = protocol["manifest"]["task_a"]
    rows = []
    for method in protocol["methods"]:
        before = load_eval(root / "eval" / method / "a_0" / task)
        after = load_eval(root / "eval" / method / f"a_{final}" / task)
        summaries, common = common_summary([before, after])
        values = [enrich(item, run) for item, run in zip(summaries, [before, after])]
        for metric in NETWORK_METRICS:
            first, last = values[0].get(metric), values[1].get(metric)
            rows.append({"method": method, "metric": metric, "common_flows": common,
                         "before": first, "after": last,
                         "change": (None if first is None or last is None else
                                    (last-first)/max(abs(first), 1e-12))})
    return rows


def write_acquisition_report(root, protocol):
    rows = acquisition_analysis(root, protocol)
    write_csv(root / "acquisition_metrics.csv", rows)
    lines = ["# SOR paper Task-A acquisition report", "",
             f"- Task: **{protocol['manifest']['task_a']}**",
             f"- Budget: **{protocol['a_points'][-1]} global optimizer updates**",
             "- Training reward: **tail_safe only**", "",
             "| Method | Reward | Throughput | Avg FCT | p95 FCT | p99 slowdown | Completion | Evidence ready |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in protocol["methods"]:
        selected = {row["metric"]: row["change"] for row in rows if row["method"] == method}
        physical_improvement = any((selected.get(name) is not None and
                                    (selected[name] < 0 if name in LOWER_BETTER or name in ("queue", "ecn")
                                     else selected[name] > 0))
                                   for name in ("throughput", "avg_fct_us", "p99_slowdown", "queue", "ecn"))
        ready = (selected.get("reward") is not None and selected["reward"] > 0 and
                 selected.get("completion_ratio") is not None and
                 selected["completion_ratio"] >= -0.001 and physical_improvement)
        lines.append("| " + method + " | " + " | ".join(pct(selected.get(metric)) for metric in
                     ("reward", "throughput", "avg_fct_us", "p95_fct_us", "p99_slowdown", "completion_ratio")) +
                     f" | {'YES' if ready else 'NO'} |")
    lines += ["", "Negative FCT/slowdown change is improvement; positive reward/throughput/completion change is improvement."]
    path = root / "ACQUISITION_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_fct_step_artifacts(root)
    print(f"Acquisition analysis written to {path}")


def port_analysis(root, protocol):
    a_end, b_end = protocol["a_points"][-1], protocol["b_points"][-1]
    task = protocol["manifest"]["task_a"]
    rows = []
    fields = ("reward_congested", "queue_p95_kb", "ecn_mean", "pfc_pause_duty")
    for method in protocol["methods"]:
        paths = {
            "after_a": root / "eval" / method / f"a_{a_end}" / task,
            "aa": root / "eval" / method / f"aa_{b_end}" / task,
            "ab": root / "eval" / method / f"ab_{b_end}" / task,
        }
        for port in range(448):
            values = {}
            try:
                for label, path in paths.items():
                    values[label] = frozen_port_summary(path, port, 400)
            except (KeyError, ValueError):
                continue
            if not any(values[label].get("active_steps") for label in values):
                continue
            row = {"method": method, "port": port,
                   "identifier": values["after_a"].get("identifier")}
            for field in fields:
                before = values["after_a"].get(field)
                control = values["aa"].get(field)
                shifted = values["ab"].get(field)
                row[f"{field}_after_a"] = before
                row[f"{field}_aa"] = control
                row[f"{field}_ab"] = shifted
                if before is None or control is None or shifted is None:
                    row[f"{field}_adjusted"] = None
                elif field == "reward_congested":
                    row[f"{field}_adjusted"] = control - shifted
                else:
                    row[f"{field}_adjusted"] = shifted - control
            rows.append(row)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def training_records(root, protocol, method):
    exp = f"paper_{root.name}_{method}"
    paths = [root / method / "a/models" / f"{exp}_metrics.jsonl",
             root / method / "ab/models" / f"{exp}_metrics.jsonl"]
    records = []
    for path in paths:
        if path.exists():
            with path.open() as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
    return [record for record in records if not record.get("eval_greedy")]


def resource_diagnostics(root, protocol):
    rows = []
    for method in protocol["methods"]:
        records = training_records(root, protocol, method)
        port_details = [detail for record in records
                        for detail in record.get("train_port_metrics", {}).values()]
        replay = records[-1].get("replay", {}) if records else {}
        global_replay = replay.get("global", {}) if isinstance(replay.get("global"), dict) else {}
        rows.append({
            "method": method,
            "epochs": len(records),
            "wall_time_seconds": sum(float(record.get("wall_time_seconds", 0)) for record in records),
            "max_rss_mb": max([float(record.get("process_max_rss_mb", 0)) for record in records] or [0]),
            "q_prediction_abs_max": max([float(detail.get("q_prediction_abs_max", 0)) for detail in port_details] or [0]),
            "q_target_abs_max": max([float(detail.get("q_target_abs_max", 0)) for detail in port_details] or [0]),
            "td_error_abs_max": max([float(detail.get("td_error_abs_max", 0)) for detail in port_details] or [0]),
            "q_inflation_events": sum(int(detail.get("q_inflation_events", 0)) for detail in port_details),
            "local_replay_size": replay.get("local_size_total"),
            "global_replay_size": global_replay.get("size", replay.get("global_size")),
            "replay_task_sizes": json.dumps(global_replay.get("task_sizes", {}), sort_keys=True),
            "replay_task_sample_counts": json.dumps(global_replay.get("task_sample_counts", {}), sort_keys=True),
            "loss_td_mean": (sum(float(detail.get("loss_td_mean", 0)) for detail in port_details) / len(port_details)
                             if port_details and method == "sor" else None),
            "loss_cons_mean": (sum(float(detail.get("loss_cons_mean", 0)) for detail in port_details) / len(port_details)
                               if port_details and method == "sor" else None),
            "loss_reg_mean": (sum(float(detail.get("loss_reg_mean", 0)) for detail in port_details) / len(port_details)
                              if port_details and method == "sor" else None),
        })
    return rows


def write_plots(root, protocol, network, ports):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    outputs = []
    old = [row for row in network if row["task"] == "realistic_webserver"]
    metrics = ["reward", "p95_fct_us", "completion_ratio", "p99_slowdown"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, metric in zip(axes.flat, metrics):
        selected = [row for row in old if row["metric"] == metric]
        axis.bar([row["method"] for row in selected],
                 [100 * (row["aa_corrected_degradation"] or 0) for row in selected])
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(metric); axis.set_ylabel("AA-corrected degradation (%)")
        axis.grid(axis="y", alpha=.3)
    figure.suptitle("ACC-local vs SOR: matched-AA old-task degradation")
    figure.tight_layout()
    output = root / "network_forgetting.png"; figure.savefig(output, dpi=180); plt.close(figure)
    outputs.append(output)

    port_metrics = ["reward_congested_adjusted", "queue_p95_kb_adjusted",
                    "ecn_mean_adjusted", "pfc_pause_duty_adjusted"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    for axis, metric in zip(axes.flat, port_metrics):
        groups = []
        labels = []
        for method in protocol_methods(ports):
            values = [row[metric] for row in ports
                      if row["method"] == method and row.get(metric) is not None]
            if values:
                groups.append(values); labels.append(method)
        if groups:
            axis.boxplot(groups, labels=labels, showfliers=True)
        axis.axhline(0, color="black", linewidth=.8)
        axis.set_title(metric); axis.grid(axis="y", alpha=.3)
    figure.suptitle("Per-port AA-corrected old-task changes")
    figure.tight_layout()
    output = root / "port_forgetting_distribution.png"; figure.savefig(output, dpi=180); plt.close(figure)
    outputs.append(output)

    for method in protocol["methods"]:
        records = training_records(root, protocol, method)
        boundary = sum(record.get("phase") == "train_a" for record in records)
        for port in protocol["representative_ports"]:
            samples = [record["train_port_metrics"][str(port)] for record in records
                       if str(port) in record.get("train_port_metrics", {})]
            if not samples:
                continue
            x = range(1, len(samples) + 1)
            figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axes[0].plot(x, [row.get("batch_reward_mean") for row in samples])
            axes[0].set_ylabel("Batch reward")
            axes[1].plot(x, [row.get("loss_mean") for row in samples], color="tab:red")
            axes[1].set_yscale("symlog", linthresh=1e-4); axes[1].set_ylabel("Loss")
            axes[2].plot(x, [row.get("q_prediction_abs_max") for row in samples], label="Q prediction")
            axes[2].plot(x, [row.get("q_target_abs_max") for row in samples], label="Q target")
            axes[2].set_ylabel("Absolute max"); axes[2].set_xlabel("Training epoch")
            axes[2].legend()
            for axis in axes:
                axis.grid(alpha=.25)
                if 0 < boundary < len(samples):
                    axis.axvline(boundary + .5, color="black", linestyle="--")
            figure.suptitle(f"{method} port {port}: reward, loss and Q")
            figure.tight_layout()
            output = root / f"training_{method}_p{port}.png"
            figure.savefig(output, dpi=180); plt.close(figure); outputs.append(output)
    return outputs


def protocol_methods(rows):
    return list(dict.fromkeys(row["method"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("acquire", "full"), default="full")
    args = parser.parse_args()
    root = args.run_dir.resolve(); protocol = read(root / "protocol.json")
    if args.scope == "acquire":
        if not (root / "acquire_complete.json").exists():
            raise ValueError("acquire stage is incomplete")
        write_acquisition_report(root, protocol)
        return
    if not (root / "continue_complete.json").exists():
        raise ValueError("continue stage is incomplete")
    network = network_analysis(root, protocol)
    ports = port_analysis(root, protocol)
    resources = resource_diagnostics(root, protocol)
    write_csv(root / "network_metrics.csv", network)
    write_csv(root / "port_metrics.csv", ports)
    write_csv(root / "resource_diagnostics.csv", resources)
    outputs = write_plots(root, protocol, network, ports)
    _, fct_plot = build_fct_step_artifacts(root)
    if fct_plot is not None:
        outputs.append(fct_plot)
    a_end, b_end = protocol["a_points"][-1], protocol["b_points"][-1]
    lines = [
        "# SOR paper matched-AA report", "",
        f"- Curriculum: **{protocol['manifest']['task_a']} → {protocol['manifest']['task_b']}**",
        f"- Budgets: A={a_end}, B={b_end} global optimizer updates",
        "- Reward used for training: **tail_safe only**",
        "- Positive AA-corrected degradation means AB is worse than matched AA.", "",
        "## Network old-task retention", "",
        "| Method | Reward | Throughput | Avg FCT | p95 FCT | p99 slowdown | Completion |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in protocol["methods"]:
        selected = {row["metric"]: row["aa_corrected_degradation"] for row in network
                    if row["method"] == method and row["task"] == protocol["manifest"]["task_a"]}
        lines.append("| " + method + " | " + " | ".join(pct(selected.get(metric)) for metric in
                     ("reward", "throughput", "avg_fct_us", "p95_fct_us", "p99_slowdown", "completion_ratio")) + " |")
    lines += ["", "## Task-B final performance", "",
              "| Method | Reward | Throughput | Avg FCT | p95 FCT | p99 slowdown | Completion |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for method in protocol["methods"]:
        selected = {row["metric"]: row["ab"] for row in network
                    if row["method"] == method and row["task"] == protocol["manifest"]["task_b"]}
        lines.append(f"| {method} | {fmt(selected.get('reward'))} | {fmt(selected.get('throughput'))} | "
                     f"{fmt(selected.get('avg_fct_us'))} | {fmt(selected.get('p95_fct_us'))} | "
                     f"{fmt(selected.get('p99_slowdown'))} | {pct(selected.get('completion_ratio'))} |")
    lines += ["", "## Artifacts", "",
              "- `network_metrics.csv`: raw after-A/AA/AB values and corrected changes.",
              "- `port_metrics.csv`: all measured active ports, including the six representative ports."]
    lines += ["- `fct_step_summary.csv`: per-episode step-trace coverage and mean FCT.",
              "- `fct_training_curves.png`: FCT observed at each global optimizer update."]
    lines += ["", "## Resource and stability diagnostics", "",
              "| Method | Time (s) | Peak RSS (MB) | Q pred max | Q target max | TD max | Q inflation events | Local replay | Global replay |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in resources:
        lines.append(f"| {row['method']} | {fmt(row['wall_time_seconds'], 1)} | {fmt(row['max_rss_mb'], 1)} | "
                     f"{fmt(row['q_prediction_abs_max'])} | {fmt(row['q_target_abs_max'])} | "
                     f"{fmt(row['td_error_abs_max'])} | {row['q_inflation_events']} | "
                     f"{row['local_replay_size'] or 0} | {row['global_replay_size'] or 0} |")
    lines += ["", "SOR A/B retained and sampled composition is recorded in `resource_diagnostics.csv`."]
    for output in outputs:
        lines.append(f"- `{output.name}`")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(root / "analysis.json", {"network": network, "ports": ports,
                                    "resources": resources})
    print(f"Analysis written to {root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
