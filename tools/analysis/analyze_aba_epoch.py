#!/usr/bin/env python3
"""Analyze one completed ABA training epoch with the authoritative formulas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.metrics_core import (
    ANALYSIS_VERSION,
    aggregate_agent_losses,
    conf_value,
    find_single,
    parse_pfc,
    parse_queue,
    parse_rate,
    parse_throughput,
    provenance,
    select_agent_epoch_record,
    summarize_fct,
)


NETWORK_FIELDS = [
    "run_id", "method", "phase", "task", "global_epoch", "phase_epoch",
    "duration_seconds", "global_env_step", "global_train_step", "epsilon",
    "reward", "loss", "td_loss", "consistency_loss", "reference_loss",
    "q_prediction_abs_max", "q_target_abs_max", "td_error_abs_p95_max",
    "expected_q_bound", "q_inflation_limit", "q_inflated_ports",
    "local_replay_size", "global_replay_size", "global_replay_clusters",
    "global_replay_boundary_entries", "global_replay_evictions",
    "global_replay_drift_score_max", "global_replay_task_sizes",
    "global_replay_task_sample_counts",
    "expected_flows", "completed_flows", "completion_ratio",
    "mean_fct_us", "p50_fct_us", "p95_fct_us", "p99_fct_us",
    "mean_slowdown", "p95_slowdown", "p99_slowdown",
    "mean_throughput_mbps", "p95_throughput_mbps", "p99_throughput_mbps",
    "mean_queue_kb", "p95_queue_kb", "p99_queue_kb", "max_queue_kb",
    "mean_ecn_rate", "p95_ecn_rate", "p99_ecn_rate", "ecn_positive_ratio",
    "pfc_pause_count", "pfc_pause_total_s", "pfc_max_pause_s", "pfc_duty",
    "ns3_exit_code", "agent_exit_code", "process_max_rss_mb",
    "analysis_version",
]

PORT_FIELDS = [
    "run_id", "method", "phase", "task", "global_epoch", "phase_epoch",
    "record_source", "agent_port", "switch_id", "port_id", "identifier",
    "reward_population", "port_reward", "tail_safe_raw", "active_steps",
    "congested_steps", "port_loss", "port_q_prediction_abs_max",
    "port_q_target_abs_max", "port_td_error_abs_p95_max",
    "dominant_action", "dominant_action_fraction",
    "queue_samples", "mean_queue_kb", "p95_queue_kb", "p99_queue_kb", "max_queue_kb",
    "rate_samples", "mean_throughput_mbps", "p95_throughput_mbps", "p99_throughput_mbps",
    "mean_ecn_rate", "p95_ecn_rate", "p99_ecn_rate", "ecn_positive_ratio",
    "pfc_mapping_available", "pfc_pause_count", "pfc_pause_total_s",
    "pfc_max_pause_s", "pfc_duty", "analysis_version",
]


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_csv(path: Path, rows, fields) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--method", choices=("acc", "sor"), required=True)
    result.add_argument("--phase", choices=("a1", "b", "a2"), required=True)
    result.add_argument("--task", required=True)
    result.add_argument("--epoch", type=int, required=True)
    result.add_argument("--phase-epoch", type=int, required=True)
    result.add_argument("--raw-dir", type=Path, required=True)
    result.add_argument("--flow", type=Path, required=True)
    result.add_argument("--runtime-conf", type=Path, required=True)
    result.add_argument("--agent-metrics", type=Path, required=True)
    result.add_argument("--metrics-offset", type=int, default=0)
    result.add_argument("--duration-seconds", type=float, required=True)
    result.add_argument("--ns3-exit-code", type=int, required=True)
    result.add_argument("--agent-exit-code", type=int, required=True)
    result.add_argument("--warmup-rate-buckets", type=int, default=2)
    return result


def main():
    args = parser().parse_args()
    raw = args.raw_dir.resolve()
    inputs = {
        "fct": find_single(raw, ".fct"),
        "queue": find_single(raw, ".queue"),
        "rate": find_single(raw, ".rate"),
        "throughput": find_single(raw, ".throughput"),
        "pfc": find_single(raw, ".pfc"),
        "agent_metrics": args.agent_metrics.resolve(),
    }
    fct = summarize_fct(inputs["fct"], args.flow)
    queue, queue_ports = parse_queue(inputs["queue"])
    rate, rate_ports = parse_rate(inputs["rate"], args.warmup_rate_buckets)
    throughput, throughput_ports = parse_throughput(inputs["throughput"])
    stop_time = float(conf_value(args.runtime_conf, "SIMULATOR_STOP_TIME", 0.0))
    monitored_ports = set(queue_ports) | set(rate_ports) | set(throughput_ports)
    pfc, pfc_ports = parse_pfc(
        inputs["pfc"], stop_time, port_universe=monitored_ports
    )
    agent_record = select_agent_epoch_record(
        args.agent_metrics,
        args.metrics_offset,
        args.run_id,
        args.phase,
        args.epoch,
    )
    training = aggregate_agent_losses(agent_record)
    # The dedicated throughput stream is authoritative. Rate-derived throughput
    # remains available in provenance/debug data but is not mixed into the row.
    row = {
        "run_id": args.run_id,
        "method": args.method,
        "phase": args.phase,
        "task": args.task,
        "global_epoch": args.epoch,
        "phase_epoch": args.phase_epoch,
        "duration_seconds": args.duration_seconds,
        "global_env_step": agent_record.get("global_env_step"),
        "global_train_step": agent_record.get("global_train_step", agent_record.get("global_step")),
        "epsilon": agent_record.get("epsilon"),
        **training,
        **fct,
        "mean_throughput_mbps": throughput.get("mean"),
        "p95_throughput_mbps": throughput.get("p95"),
        "p99_throughput_mbps": throughput.get("p99"),
        "mean_queue_kb": queue.get("mean"),
        "p95_queue_kb": queue.get("p95"),
        "p99_queue_kb": queue.get("p99"),
        "max_queue_kb": queue.get("max"),
        "mean_ecn_rate": rate.get("ecn_rate_mean"),
        "p95_ecn_rate": rate.get("ecn_rate_p95"),
        "p99_ecn_rate": rate.get("ecn_rate_p99"),
        "ecn_positive_ratio": rate.get("ecn_positive_ratio"),
        **pfc,
        "ns3_exit_code": args.ns3_exit_code,
        "agent_exit_code": args.agent_exit_code,
        "process_max_rss_mb": agent_record.get("process_max_rss_mb"),
        "analysis_version": ANALYSIS_VERSION,
    }
    port_rows = []
    # The simulator emits the physical switch-peer key in all four monitor
    # streams. PFC is included only when its sixth (peer-id) column is present;
    # legacy five-column PFC therefore stays n/a instead of being guessed.
    physical_keys = set(queue_ports) | set(rate_ports) | set(throughput_ports) | set(pfc_ports)
    for key in sorted(physical_keys):
        q = queue_ports.get(key, {})
        r = rate_ports.get(key, {})
        t = throughput_ports.get(key, {})
        p = pfc_ports.get(key, {})
        pfc_mapped = bool(pfc.get("pfc_mapping_available"))
        port_rows.append({
            "run_id": args.run_id, "method": args.method, "phase": args.phase,
            "task": args.task, "global_epoch": args.epoch, "phase_epoch": args.phase_epoch,
            "record_source": "physical",
            "switch_id": key[0], "port_id": key[1], "identifier": f"{key[0]}-{key[1]}",
            "queue_samples": q.get("samples"), "mean_queue_kb": q.get("mean"),
            "p95_queue_kb": q.get("p95"), "p99_queue_kb": q.get("p99"),
            "max_queue_kb": q.get("max"), "rate_samples": r.get("ecn_rate_samples"),
            "mean_throughput_mbps": t.get("mean"), "p95_throughput_mbps": t.get("p95"),
            "p99_throughput_mbps": t.get("p99"), "mean_ecn_rate": r.get("ecn_rate_mean"),
            "p95_ecn_rate": r.get("ecn_rate_p95"), "p99_ecn_rate": r.get("ecn_rate_p99"),
            "ecn_positive_ratio": r.get("ecn_positive_ratio"),
            "pfc_mapping_available": pfc_mapped,
            "pfc_pause_count": p.get("pfc_pause_count", 0) if pfc_mapped else None,
            "pfc_pause_total_s": p.get("pfc_pause_total_s", 0.0) if pfc_mapped else None,
            "pfc_max_pause_s": p.get("pfc_max_pause_s", 0.0) if pfc_mapped else None,
            "pfc_duty": p.get("pfc_duty", 0.0) if pfc_mapped else None,
            "analysis_version": ANALYSIS_VERSION,
        })
    for agent_port, detail in sorted(
        (agent_record.get("watch_ports_metrics") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        congested_steps = int(detail.get("congested_steps") or 0)
        active_steps = int(detail.get("active_steps") or 0)
        population = "congested" if congested_steps else "active"
        means = detail.get(f"means_{population}") or {}
        actions = detail.get(f"actions_{population}") or {}
        dominant_action = None
        dominant_count = 0
        if actions:
            dominant_action, dominant_count = max(
                actions.items(), key=lambda item: (item[1], item[0])
            )
        denominator = congested_steps if population == "congested" else active_steps
        identifier = detail.get("identifier")
        training_detail = (agent_record.get("train_port_metrics") or {}).get(str(agent_port), {})
        switch_id = port_id = None
        if identifier and "-" in str(identifier):
            try:
                switch_id, port_id = (int(value) for value in str(identifier).split("-", 1))
            except ValueError:
                pass
        port_rows.append({
            "run_id": args.run_id, "method": args.method, "phase": args.phase,
            "task": args.task, "global_epoch": args.epoch, "phase_epoch": args.phase_epoch,
            "record_source": "watch", "agent_port": int(agent_port),
            "switch_id": switch_id, "port_id": port_id, "identifier": identifier,
            "reward_population": population, "port_reward": means.get("reward"),
            "tail_safe_raw": means.get("tail_safe_raw"), "active_steps": active_steps,
            "congested_steps": congested_steps, "dominant_action": dominant_action,
            "port_loss": training_detail.get("loss_mean"),
            "port_q_prediction_abs_max": training_detail.get("q_prediction_abs_max"),
            "port_q_target_abs_max": training_detail.get("q_target_abs_max"),
            "port_td_error_abs_p95_max": training_detail.get("td_error_abs_p95_max"),
            "dominant_action_fraction": (
                dominant_count / denominator if denominator else None
            ),
            "analysis_version": ANALYSIS_VERSION,
        })
    prov = provenance(inputs, args.flow, args.runtime_conf, args.warmup_rate_buckets)
    prov["rate_stream_summary"] = rate
    prov["agent_record_epoch"] = agent_record.get("epoch")
    atomic_json(raw / "epoch_metrics.json", row)
    write_csv(raw / "epoch_ports.csv", port_rows, PORT_FIELDS)
    atomic_json(raw / "analysis_provenance.json", prov)
    print(json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
