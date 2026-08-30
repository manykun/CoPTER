#!/usr/bin/env python3
"""Analyze causal one-port interpolation, including queue/ECN/PFC metrics."""

import argparse
import csv
import json
from pathlib import Path

from analyze_forgetting import load_eval, summarize


def percentile(values, fraction):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def metric_detail(metrics, port):
    return metrics.get("watch_ports_metrics", {}).get(str(port), {})


def nested(detail, population, key):
    return detail.get(f"means_{population}", {}).get(key)


def read_trace(path, port):
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if int(item.get("port", -1)) == port:
            records.append(item)
    return records


def config_value(path, key, default=None):
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == key:
            return float(fields[1])
    return default


def pfc_summary(path, identifier, stop_time):
    result = {
        "pfc_mapping_available": False,
        "pfc_pause_events": None,
        "pfc_resume_events": None,
        "pfc_pause_total_s": None,
        "pfc_pause_duty": None,
        "pfc_max_pause_s": None,
        "pfc_final_paused": None,
    }
    if not path.exists() or not identifier or "-" not in identifier:
        return result
    try:
        switch, peer = (int(value) for value in identifier.split("-", 1))
    except ValueError:
        return result
    events = []
    has_peer_column = False
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        if len(fields) >= 6:
            has_peer_column = True
            if int(fields[1]) != switch or int(fields[5]) != peer:
                continue
        else:
            continue
        events.append((int(fields[0]) / 1e9, int(fields[4])))
    if not has_peer_column:
        return result
    events.sort()
    paused_at = None
    durations = []
    pause_events = 0
    resume_events = 0
    for timestamp, event_type in events:
        if event_type == 1:
            pause_events += 1
            if paused_at is None:
                paused_at = timestamp
        elif event_type == 0:
            resume_events += 1
            if paused_at is not None:
                durations.append(max(0.0, timestamp - paused_at))
                paused_at = None
    final_paused = paused_at is not None
    if final_paused:
        durations.append(max(0.0, stop_time - paused_at))
    total = sum(durations)
    result.update({
        "pfc_mapping_available": True,
        "pfc_pause_events": pause_events,
        "pfc_resume_events": resume_events,
        "pfc_pause_total_s": total,
        "pfc_pause_duty": total / stop_time if stop_time > 0 else None,
        "pfc_max_pause_s": max(durations, default=0.0),
        "pfc_final_paused": final_paused,
    })
    return result


def find_single(directory, suffix):
    matches = list(directory.glob(f"*{suffix}"))
    return matches[0] if matches else directory / f"missing{suffix}"


def summarize_port(directory, port, buffer_kb, point):
    metrics = json.loads(
        (directory / "metrics.json").read_text(encoding="utf-8")
    )
    detail = metric_detail(metrics, port)
    trace = read_trace(directory / "watch_trace.jsonl", port)
    congested = [item for item in trace if item.get("congested")]
    population = congested or [item for item in trace if item.get("active")]
    queue_values = [item.get("peak_queue") for item in population]
    ecn_values = [
        item["avg_ecn"] for item in population
        if item.get("avg_ecn") is not None
    ]
    queue_kb = [value * buffer_kb for value in queue_values if value is not None]
    kmin = point["physical"]["kmin_kb"]
    kmax = point["physical"]["kmax_kb"]
    config = directory / "input.conf"
    stop_time = config_value(config, "SIMULATOR_STOP_TIME", 0.0)
    pfc = pfc_summary(find_single(directory, ".pfc"), detail.get("identifier"), stop_time)
    return {
        "identifier": detail.get("identifier"),
        "samples": detail.get("samples"),
        "active_steps": detail.get("active_steps"),
        "congested_steps": detail.get("congested_steps"),
        "reward_active": nested(detail, "active", "reward"),
        "reward_congested": nested(detail, "congested", "reward"),
        "tail_safe_raw_congested": nested(
            detail, "congested", "tail_safe_raw"
        ),
        "queue_mean_kb": (
            sum(queue_kb) / len(queue_kb) if queue_kb else None
        ),
        "queue_p95_kb": percentile(queue_kb, 0.95),
        "queue_max_kb": max(queue_kb, default=None),
        "queue_ge_kmin_ratio": (
            sum(value >= kmin for value in queue_kb) / len(queue_kb)
            if queue_kb else None
        ),
        "queue_ge_kmax_ratio": (
            sum(value >= kmax for value in queue_kb) / len(queue_kb)
            if queue_kb else None
        ),
        "ecn_mean": (
            sum(ecn_values) / len(ecn_values) if ecn_values else None
        ),
        "ecn_p95": percentile(ecn_values, 0.95),
        "ecn_max": max(ecn_values, default=None),
        "ecn_positive_ratio": (
            sum(value > 0 for value in ecn_values) / len(ecn_values)
            if ecn_values else None
        ),
        "ecn_deviation_from_003": (
            abs(sum(ecn_values) / len(ecn_values) - 0.03)
            if ecn_values else None
        ),
        **pfc,
    }


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def reward_preference(rows, tolerance):
    candidates = [row for row in rows if row.get("reward_selected") is not None]
    if not candidates:
        return {
            "sensitive": False,
            "reward_spread": 0.0,
            "best_reward": None,
            "optimal_alpha_min": None,
            "optimal_alpha_max": None,
        }
    rewards = [row["reward_selected"] for row in candidates]
    best = max(rewards)
    scale = max(abs(best), 1e-12)
    spread = (max(rewards) - min(rewards)) / scale
    optimal = [
        row["alpha"] for row in candidates
        if best - row["reward_selected"] <= tolerance * scale
    ]
    return {
        "sensitive": spread > tolerance,
        "reward_spread": spread,
        "best_reward": best,
        "optimal_alpha_min": min(optimal),
        "optimal_alpha_max": max(optimal),
    }


def disjoint_preferences(left, right):
    if not left["sensitive"] or not right["sensitive"]:
        return False
    return (
        left["optimal_alpha_max"] < right["optimal_alpha_min"]
        or right["optimal_alpha_max"] < left["optimal_alpha_min"]
    )


def write_plots(run_dir, rows, ports, tasks):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    outputs = []
    metrics = (
        ("reward_selected", "Port reward"),
        ("queue_p95_kb", "Queue p95 (KB)"),
        ("ecn_mean", "ECN marking rate"),
        ("pfc_pause_duty", "PFC pause duty"),
    )
    for port in ports:
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for axis, (key, title) in zip(axes.flat, metrics):
            for task in tasks:
                selected = sorted(
                    (row for row in rows
                     if row["port"] == port and row["task"] == task),
                    key=lambda row: row["alpha"],
                )
                x = [row["alpha"] for row in selected if row[key] is not None]
                y = [row[key] for row in selected if row[key] is not None]
                axis.plot(x, y, marker="o", label=task)
            axis.set_title(title)
            axis.grid(alpha=0.3)
        axes[1, 0].set_xlabel("Interpolation alpha (A → B)")
        axes[1, 1].set_xlabel("Interpolation alpha (A → B)")
        axes[0, 0].legend()
        figure.suptitle(f"ACC causal action path: port {port}")
        figure.tight_layout()
        output = run_dir / f"port_path_p{port}.png"
        figure.savefig(output, dpi=180)
        plt.close(figure)
        outputs.append(str(output))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--preference-tolerance", type=float, default=0.01,
        help="Relative reward tolerance for flat/tied alpha preferences",
    )
    args = parser.parse_args()
    if not 0 <= args.preference_tolerance < 1:
        parser.error("--preference-tolerance must be in [0, 1)")
    manifest = json.loads(
        (args.run_dir / "path_manifest.json").read_text(encoding="utf-8")
    )
    tasks = (manifest["task_a"], manifest["task_b"])
    points = manifest["points"]
    rows = []
    for port in manifest["target_ports"]:
        loaded = {task: {} for task in tasks}
        for point in points:
            for task in tasks:
                directory = args.run_dir / "eval" / f"p{port}" / point["label"] / task
                if not (directory / "metrics.json").exists():
                    raise SystemExit(f"evaluation missing: p{port}/{point['label']}/{task}")
                loaded[task][point["label"]] = load_eval(directory)
        common = {
            task: set.intersection(
                *(set(run["flows"]) for run in loaded[task].values())
            )
            for task in tasks
        }
        for point in points:
            for task in tasks:
                label = point["label"]
                directory = args.run_dir / "eval" / f"p{port}" / label / task
                network = summarize(loaded[task][label], common[task])
                port_summary = summarize_port(
                    directory, port, manifest["buffer_kb"], point
                )
                reward_population = (
                    "congested"
                    if port_summary["reward_congested"] is not None
                    else "active"
                )
                reward_selected = port_summary[
                    f"reward_{reward_population}"
                ]
                rows.append({
                    "port": port,
                    "task": task,
                    "label": label,
                    "alpha": point["alpha"],
                    "kmin_norm": point["kmin_norm"],
                    "kmax_norm": point["kmax_norm"],
                    "pmax": point["pmax"],
                    **point["physical"],
                    "common_flows": len(common[task]),
                    "completion": network["completion_ratio"],
                    "p95_fct_us": network["p95_fct_us"],
                    **port_summary,
                    "reward_population": reward_population,
                    "reward_selected": reward_selected,
                })
    with (args.run_dir / "port_path_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    per_port = {}
    for port in manifest["target_ports"]:
        port_rows = [row for row in rows if row["port"] == port]
        preferences = {}
        for task in tasks:
            preferences[task] = reward_preference(
                [row for row in port_rows if row["task"] == task],
                args.preference_tolerance,
            )
        per_port[str(port)] = {
            "preferences": preferences,
            "different_best_alpha": disjoint_preferences(
                preferences[tasks[0]], preferences[tasks[1]]
            ),
        }
    result = {
        "manifest": manifest,
        "preference_tolerance": args.preference_tolerance,
        "ports": per_port,
        "rows": rows,
    }
    result["plots"] = write_plots(
        args.run_dir, rows, manifest["target_ports"], tasks
    )
    (args.run_dir / "port_path_analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Single-port causal interpolation report", "",
        f"- Tasks: **{tasks[0]} → {tasks[1]}**",
        f"- Endpoint source port: **{manifest['endpoint_port']}**",
        f"- Endpoint actions: **{manifest['endpoint_a']['action']} → "
        f"{manifest['endpoint_b']['action']}**",
        f"- Endpoint source: **{manifest.get('endpoint_source', {'kind': 'calibration'})['kind']}**",
        f"- Alpha step: **{manifest['alpha_step']}**",
        f"- Degenerate endpoint path: **{manifest.get('degenerate_path', False)}**",
        f"- Preference tolerance: **{args.preference_tolerance:.1%}**",
        "- Frozen after-B policy; only the selected port is overridden.",
        "- DDQN loss is intentionally absent because evaluation performs no updates.",
        "", "## Port preference summary", "",
        "| Port | A sensitivity / optimal alpha | B sensitivity / optimal alpha | Conflict |",
        "|---:|---|---|---:|",
    ]
    for port, item in per_port.items():
        pref_a = item["preferences"][tasks[0]]
        pref_b = item["preferences"][tasks[1]]
        describe = lambda pref: (
            f"{'sensitive' if pref['sensitive'] else 'flat'} / "
            f"{fmt(pref['optimal_alpha_min'], 2)}–"
            f"{fmt(pref['optimal_alpha_max'], 2)}"
        )
        lines.append(
            f"| {port} | {describe(pref_a)} | {describe(pref_b)} | "
            f"{item['different_best_alpha']} |"
        )
    lines += [
        "", "## Measurements", "",
        "| Port | Task | alpha | Reward (population) | Queue p95 KB | Queue max KB | "
        "ECN mean | ECN>0 | PFC pauses | PFC duty | Completion | p95 FCT us |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['port']} | {row['task']} | {row['alpha']:.2f} | "
            f"{fmt(row['reward_selected'])} ({row['reward_population']}) | "
            f"{fmt(row['queue_p95_kb'], 2)} | "
            f"{fmt(row['queue_max_kb'], 2)} | {fmt(row['ecn_mean'])} | "
            f"{fmt(row['ecn_positive_ratio'], 3)} | "
            f"{row['pfc_pause_events'] if row['pfc_pause_events'] is not None else 'n/a'} | "
            f"{fmt(row['pfc_pause_duty'], 6)} | {fmt(row['completion'], 4)} | "
            f"{fmt(row['p95_fct_us'], 2)} |"
        )
    lines += [
        "",
        "A differing reward-best alpha on the target port, accompanied by "
        "queue/ECN/PFC changes and absent on the control port, supports a "
        "causal local task conflict. Flat curves mean finer discretization "
        "is unlikely to expose forgetting.", "",
    ]
    (args.run_dir / "PORT_PATH_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"Analysis written to {args.run_dir / 'PORT_PATH_REPORT.md'}")


if __name__ == "__main__":
    main()
