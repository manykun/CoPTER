#!/usr/bin/env python3
"""Summarize per-port ACC loss and frozen physical forgetting metrics."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from analyze_forgetting import load_eval, summarize
from analyze_port_path_sweep import (
    config_value,
    find_single,
    metric_detail,
    nested,
    percentile,
    pfc_summary,
    read_trace,
)


def parse_ports(value):
    ports = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not ports or len(ports) != len(set(ports)) or any(port < 0 for port in ports):
        raise ValueError("ports must be unique non-negative integers")
    return ports


def load_jsonl(path):
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def training_summary(records, phase, port):
    selected = []
    for record in records:
        detail = record.get("train_port_metrics", {}).get(str(port))
        if record.get("phase") == phase and detail:
            selected.append((record, detail))
    losses = [detail["loss_mean"] for _, detail in selected]
    rewards = [
        detail.get("batch_reward_mean") for _, detail in selected
        if detail.get("batch_reward_mean") is not None
    ]
    return {
        "epochs": len(selected),
        "updates": sum(detail.get("updates", 0) for _, detail in selected),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_median": statistics.median(losses) if losses else None,
        "loss_max": max(losses) if losses else None,
        "batch_reward_first": rewards[0] if rewards else None,
        "batch_reward_last": rewards[-1] if rewards else None,
    }


def frozen_port_summary(directory, port, buffer_kb):
    metrics = json.loads(
        (directory / "metrics.json").read_text(encoding="utf-8")
    )
    detail = metric_detail(metrics, port)
    trace = read_trace(directory / "watch_trace.jsonl", port)
    congested = [item for item in trace if item.get("congested")]
    population = congested or [item for item in trace if item.get("active")]
    queue = [
        item["peak_queue"] * buffer_kb
        for item in population if item.get("peak_queue") is not None
    ]
    ecn = [
        item["avg_ecn"] for item in population
        if item.get("avg_ecn") is not None
    ]
    stop = config_value(directory / "input.conf", "SIMULATOR_STOP_TIME", 0.0)
    pfc = pfc_summary(
        find_single(directory, ".pfc"), detail.get("identifier"), stop
    )
    return {
        "identifier": detail.get("identifier"),
        "active_steps": detail.get("active_steps"),
        "congested_steps": detail.get("congested_steps"),
        "reward_active": nested(detail, "active", "reward"),
        "reward_congested": nested(detail, "congested", "reward"),
        "tail_safe_raw_congested": nested(
            detail, "congested", "tail_safe_raw"
        ),
        "queue_mean_kb": sum(queue) / len(queue) if queue else None,
        "queue_p95_kb": percentile(queue, 0.95),
        "queue_max_kb": max(queue, default=None),
        "ecn_mean": sum(ecn) / len(ecn) if ecn else None,
        "ecn_p95": percentile(ecn, 0.95),
        "ecn_max": max(ecn, default=None),
        "ecn_positive_ratio": (
            sum(value > 0 for value in ecn) / len(ecn) if ecn else None
        ),
        **pfc,
    }


def relative_change(after, before):
    if after is None or before is None or abs(before) < 1e-12:
        return None
    return (after - before) / abs(before)


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def write_training_plots(run_dir, records, ports):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    outputs = []
    for port in ports:
        samples = []
        for record in records:
            detail = record.get("train_port_metrics", {}).get(str(port))
            if record.get("phase") in ("train_a", "train_b") and detail:
                samples.append((record, detail))
        if not samples:
            continue
        boundary = sum(record.get("phase") == "train_a" for record, _ in samples)
        x = list(range(1, len(samples) + 1))
        losses = [detail.get("loss_mean") for _, detail in samples]
        rewards = [detail.get("batch_reward_mean") for _, detail in samples]
        figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(x, rewards, marker="o", markersize=3)
        axes[0].set_ylabel("Batch reward")
        axes[0].grid(alpha=0.3)
        axes[1].plot(x, losses, marker="o", markersize=3, color="tab:red")
        if losses and all(value is not None and value > 0 for value in losses):
            axes[1].set_yscale("log")
        axes[1].set_ylabel("Mean TD loss")
        axes[1].set_xlabel("Port training epoch")
        axes[1].grid(alpha=0.3)
        if 0 < boundary < len(samples):
            for axis in axes:
                axis.axvline(boundary + 0.5, color="black", linestyle="--")
            axes[0].text(
                boundary + 0.7, axes[0].get_ylim()[1], "A → B",
                va="top", ha="left",
            )
        figure.suptitle(f"ACC port {port}: training reward and DDQN loss")
        figure.tight_layout()
        output = run_dir / f"port_training_p{port}.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(str(output))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ports", required=True)
    args = parser.parse_args()
    try:
        ports = parse_ports(args.ports)
    except ValueError as exc:
        parser.error(str(exc))
    manifest = json.loads(
        (args.run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    task_a, task_b = manifest["task_a"], manifest["task_b"]
    seed = manifest["seed"]
    exp = f"continual_{manifest['run_id']}_acc_s{seed}"
    metrics_path = args.run_dir / "acc" / "models" / f"{exp}_metrics.jsonl"
    records = load_jsonl(metrics_path)

    training_rows = []
    for port in ports:
        for phase in ("train_a", "train_b"):
            training_rows.append({
                "port": port,
                "phase": phase,
                **training_summary(records, phase, port),
            })

    evaluation_rows = []
    loaded = {}
    for task in (task_a, task_b):
        phases = (
            ("initial", "after_a", "after_b")
            if task == task_a else ("after_a", "after_b")
        )
        loaded[task] = {
            phase: load_eval(args.run_dir / "eval" / "acc" / phase / task)
            for phase in phases
        }
        common = set.intersection(
            *(set(run["flows"]) for run in loaded[task].values())
        )
        for phase, run in loaded[task].items():
            directory = args.run_dir / "eval" / "acc" / phase / task
            network = summarize(run, common)
            for port in ports:
                evaluation_rows.append({
                    "port": port,
                    "task": task,
                    "phase": phase,
                    "common_flows": len(common),
                    "completion": network["completion_ratio"],
                    "p95_fct_us": network["p95_fct_us"],
                    **frozen_port_summary(
                        directory, port, manifest["buffer_kb"]
                    ),
                })
                row = evaluation_rows[-1]
                row["reward_population"] = (
                    "congested"
                    if row["reward_congested"] is not None else "active"
                )
                row["reward_selected"] = row[
                    f"reward_{row['reward_population']}"
                ]

    with (args.run_dir / "PORT_TRAINING_SUMMARY.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_rows[0]))
        writer.writeheader()
        writer.writerows(training_rows)
    with (args.run_dir / "PORT_EVAL_SUMMARY.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluation_rows[0]))
        writer.writeheader()
        writer.writerows(evaluation_rows)

    comparisons = []
    for port in ports:
        for name, task, before_phase, after_phase in (
            ("task_a_acquisition", task_a, "initial", "after_a"),
            ("task_b_acquisition", task_b, "after_a", "after_b"),
            ("forgetting", task_a, "after_a", "after_b"),
        ):
            before = next(
                row for row in evaluation_rows
                if row["port"] == port and row["task"] == task
                and row["phase"] == before_phase
            )
            after = next(
                row for row in evaluation_rows
                if row["port"] == port and row["task"] == task
                and row["phase"] == after_phase
            )
            comparisons.append({
                "port": port,
                "comparison": name,
                "task": task,
                "reward_change": relative_change(
                    after["reward_selected"], before["reward_selected"]
                ),
                "reward_population": (
                    before["reward_population"]
                    if before["reward_population"] == after["reward_population"]
                    else "mixed"
                ),
                "queue_p95_change": relative_change(
                    after["queue_p95_kb"], before["queue_p95_kb"]
                ),
                "ecn_mean_change": relative_change(
                    after["ecn_mean"], before["ecn_mean"]
                ),
                "pfc_duty_change": relative_change(
                    after["pfc_pause_duty"], before["pfc_pause_duty"]
                ),
                "before": before,
                "after": after,
            })
    result = {
        "run_id": manifest["run_id"],
        "task_a": task_a,
        "task_b": task_b,
        "ports": ports,
        "training": training_rows,
        "evaluation": evaluation_rows,
        "comparisons": comparisons,
        "plots": write_training_plots(args.run_dir, records, ports),
    }
    (args.run_dir / "port_continual_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# ACC per-port continual-learning report", "",
        f"- Curriculum: **{task_a} → {task_b}**",
        f"- Watched ports: **{','.join(map(str, ports))}**",
        "- Frozen after-A/after-B measurements; no PASS/FAIL gate.",
        "", "## Training loss", "",
        "| Port | Phase | Epochs | Updates | Loss first | Loss last | "
        "Loss median | Loss max | Batch reward first | Batch reward last |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in training_rows:
        lines.append(
            f"| {row['port']} | {row['phase']} | {row['epochs']} | "
            f"{row['updates']} | {fmt(row['loss_first'])} | "
            f"{fmt(row['loss_last'])} | {fmt(row['loss_median'])} | "
            f"{fmt(row['loss_max'])} | {fmt(row['batch_reward_first'])} | "
            f"{fmt(row['batch_reward_last'])} |"
        )
    lines += [
        "", "## Acquisition and forgetting", "",
        "| Port | Comparison | Reward change | Queue p95 change | ECN mean change | "
        "PFC duty change | Reward population | Reward before/after | Queue p95 before/after KB | "
        "ECN before/after | PFC pauses before/after |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        before, after = item["before"], item["after"]
        percent = lambda value: "n/a" if value is None else f"{value:.2%}"
        lines.append(
            f"| {item['port']} | {item['comparison']} | "
            f"{percent(item['reward_change'])} | "
            f"{percent(item['queue_p95_change'])} | "
            f"{percent(item['ecn_mean_change'])} | "
            f"{percent(item['pfc_duty_change'])} | "
            f"{item['reward_population']} | "
            f"{fmt(before['reward_selected'])} / {fmt(after['reward_selected'])} | "
            f"{fmt(before['queue_p95_kb'], 2)} / {fmt(after['queue_p95_kb'], 2)} | "
            f"{fmt(before['ecn_mean'])} / {fmt(after['ecn_mean'])} | "
            f"{before['pfc_pause_events'] if before['pfc_pause_events'] is not None else 'n/a'} / "
            f"{after['pfc_pause_events'] if after['pfc_pause_events'] is not None else 'n/a'} |"
        )
    lines += [
        "",
        "A TD-loss spike at the A→B boundary demonstrates adaptation pressure, "
        "not forgetting by itself. Forgetting requires a frozen-A reward drop "
        "with queue, ECN, or PFC degradation on the target port and a stable "
        "negative-control port.", "",
    ]
    report = args.run_dir / "PORT_CONTINUAL_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Analysis written to {report}")


if __name__ == "__main__":
    main()
