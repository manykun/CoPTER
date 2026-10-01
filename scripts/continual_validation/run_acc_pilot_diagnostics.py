#!/usr/bin/env python3
"""Causal action sensitivity and frozen-policy diagnostics for an ACC pilot.

The source pilot is never mutated.  Fixed-action evaluations use a fresh model
directory, while the final frozen evaluation uses a replay-free copy of the
source checkpoint.  All comparisons use the canonical source flow and the
common completed-flow intersection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "copter") not in sys.path:
    sys.path.insert(0, str(ROOT / "copter"))

from scripts.continual_validation.run_aba import (  # noqa: E402
    load_config,
    render_runtime_conf,
)
from structures import acc_action_from_indices  # noqa: E402
from tools.analysis.metrics_core import (  # noqa: E402
    conf_value,
    parse_fct,
    parse_pfc,
    parse_queue,
    parse_rate,
    parse_throughput,
    read_queue_records,
    read_rate_records,
    summarize_fct,
)


FIXED_ACTIONS = {
    "low_gentle": (0, 0),
    "low_strong": (0, 6),
    "mid": (4, 3),
    "high_gentle": (8, 0),
    "high_strong": (8, 6),
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(directory: Path) -> dict:
    return {
        path.name: sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
        and not path.name.endswith("_metrics.jsonl")
        and "_rb_port" not in path.name
        and not path.name.endswith("_shared_rb.pkl")
    }


def source_checkpoint(run_dir: Path):
    model = run_dir / "acc" / "models"
    states = list(model.glob("*_train_state.json"))
    if len(states) != 1:
        raise ValueError(f"expected one ACC train state under {model}")
    state = read_json(states[0])
    exp = states[0].name[: -len("_train_state.json")]
    node_count = int(state.get("node_number", 0))
    if node_count <= 0:
        raise ValueError("invalid source node count")
    missing = [
        model / f"{exp}_ACC_{port}"
        for port in range(node_count)
        if not (model / f"{exp}_ACC_{port}").is_file()
    ]
    if missing:
        raise ValueError(f"incomplete source checkpoint; first missing: {missing[0]}")
    return model, exp, state


def prepare_final_copy(source: Path, destination: Path, exp: str) -> None:
    source_hashes = model_fingerprint(source)
    marker = destination / "source_fingerprint.json"
    if destination.exists():
        if not marker.exists() or read_json(marker) != source_hashes:
            raise ValueError("existing final-model copy does not match the source")
        return
    destination.mkdir(parents=True)
    for path in source.iterdir():
        if not path.is_file() or path.name.endswith("_metrics.jsonl"):
            continue
        target = destination / path.name
        if "_rb_port" in path.name or path.name.endswith("_shared_rb.pkl"):
            # AgentHelper verifies replay on every non-cold checkpoint load,
            # including frozen evaluation.  A read-only symlink avoids a
            # second multi-gigabyte copy while retaining exact load semantics.
            target.symlink_to(path.resolve())
        else:
            shutil.copy2(path, target)
    write_json(marker, source_hashes)
    if not (destination / f"{exp}_train_state.json").is_file():
        raise ValueError("final-model copy lacks train state")


def last_eval_record(metrics: Path, tag: str) -> dict:
    matches = []
    with metrics.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("eval_greedy") and record.get("eval_tag") == tag:
                matches.append(record)
    if len(matches) != 1:
        raise ValueError(f"expected one metrics record for {tag}, found {len(matches)}")
    return matches[0]


def action_histogram(record: dict) -> Counter:
    return Counter({
        key: int(value)
        for key, value in (record.get("action_histogram") or {}).items()
    })


def tv_distance(left: Counter, right: Counter):
    left_total, right_total = sum(left.values()), sum(right.values())
    if not left_total or not right_total:
        return None
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in set(left) | set(right)
    )


def dominant_action(record: dict):
    histogram = action_histogram(record)
    if not histogram:
        return None, None
    action, count = histogram.most_common(1)[0]
    return action, count / sum(histogram.values())


def metric_record(destination: Path, flow: Path, runtime_conf: Path,
                  agent_record: dict, common_keys=None) -> dict:
    fct = summarize_fct(destination / "result.fct", flow, common_keys)
    queue, _ = parse_queue(destination / "result.queue")
    rate, rate_ports = parse_rate(destination / "result.rate", 2)
    throughput, _ = parse_throughput(destination / "result.throughput")
    stop_time = float(conf_value(runtime_conf, "SIMULATOR_STOP_TIME", 0.0))
    pfc, _ = parse_pfc(
        destination / "result.pfc", stop_time, port_universe=rate_ports
    )
    dominant, fraction = dominant_action(agent_record)
    return {
        **fct,
        "reward": agent_record.get("rollout_mean_reward"),
        "all_congested_reward": agent_record.get("rollout_all_congested_mean"),
        "mean_throughput_mbps": throughput.get("mean"),
        "mean_queue_kb": queue.get("mean"),
        "p95_queue_kb": queue.get("p95"),
        "max_queue_kb": queue.get("max"),
        "mean_ecn_rate": rate.get("ecn_rate_mean"),
        "ecn_positive_ratio": rate.get("ecn_positive_ratio"),
        "pfc_duty": pfc.get("pfc_duty"),
        "dominant_action": dominant,
        "dominant_action_fraction": fraction,
    }


def threshold_activation(destination: Path, action, action_space: str) -> dict:
    parameter = acc_action_from_indices(action, action_space)
    conf = destination / "input.conf"
    min_kmin = float(conf_value(conf, "OPENGYM_MIN_KMIN", 5000.0))
    max_kmin = float(conf_value(conf, "OPENGYM_MAX_KMIN", 50000.0))
    min_kmax = float(conf_value(conf, "OPENGYM_MIN_KMAX", 15000.0))
    max_kmax = float(conf_value(conf, "OPENGYM_MAX_KMAX", 100000.0))
    capacities = {}
    for record in read_rate_records(destination / "result.rate"):
        capacities[(record["switch_id"], record["port_id"])] = record[
            "max_rate_bps"
        ]
    above_min = above_max = matched = 0
    buffer_ratios = []
    for record in read_queue_records(destination / "result.queue"):
        key = (record["switch_id"], record["port_id"])
        if key not in capacities or capacities[key] <= 0:
            continue
        scale = capacities[key] / 25e9
        kmin = (
            min_kmin + (max_kmin - min_kmin) * parameter.k_min_norm
        ) * scale
        kmax = (
            min_kmax + (max_kmax - min_kmax) * parameter.k_max_norm
        ) * scale
        queue = record["queue_bytes"]
        matched += 1
        above_min += queue >= kmin
        above_max += queue >= kmax
        if record["switch_buffer_bytes"] > 0:
            buffer_ratios.append(queue / record["switch_buffer_bytes"])
    return {
        "queue_threshold_samples": matched,
        "queue_ge_kmin_ratio": above_min / matched if matched else None,
        "queue_ge_kmax_ratio": above_max / matched if matched else None,
        "buffer_utilization_mean": mean(buffer_ratios) if buffer_ratios else None,
        "buffer_utilization_max": max(buffer_ratios) if buffer_ratios else None,
    }


def run_evaluation(config: dict, protocol: dict, source_conf: Path, flow: Path,
                   output: Path, model: Path, exp: str, label: str,
                   repetition: int, port: int, forced_action=None) -> Path:
    destination = output / "eval" / label / f"repeat_{repetition}"
    marker = destination / "complete.json"
    if marker.exists():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    local_flow = destination / "input.flow"
    shutil.copy2(flow, local_flow)
    runtime_conf = destination / "input.conf"
    render_runtime_conf(
        source_conf, local_flow, destination, runtime_conf,
        int(protocol["switch_buffer_kb"]),
    )
    tag = f"{label}_repeat_{repetition}"
    watch_trace = destination / "watch_trace.jsonl"
    command = [
        "bash", str(ROOT / "run_training.sh"),
        "--config", str(runtime_conf), "--exp", exp, "--mode", "ACC",
        "--model-dir", str(model), "--buffer", str(protocol["switch_buffer_kb"]),
        "--eps-start", str(config.get("epsilon_start", 1.0)),
        "--eps-end", str(config.get("epsilon_end", 0.05)),
        "--eps-decay", str(protocol["epsilon_decay_steps"]),
        "--epsilon-schedule", "global",
        "--target-update-interval", str(protocol["target_update_interval"]),
        "--seed", str(protocol["seed"]), "--tb-enable", "false",
        "--acc-hidden-dims", str(protocol["hidden_dims"]),
        "--action-space", str(protocol["action_space"]),
        "--reward-profile", str(protocol["reward_profile"]),
        "--reward-queue-lambda", str(protocol["reward_queue_lambda"]),
        "--reward-ecn-lambda", str(protocol["reward_ecn_lambda"]),
        "--reward-weights", str(protocol["reward_weights"]),
        "--shared-replay", "false", "--eval-greedy", "--one-shot",
        "--eval-tag", tag, "--port", str(port),
        "--run-id", f"{protocol['run_id']}_diagnostics", "--phase", "a1",
        "--watch-ports", str(config.get("watch_ports", "")),
        "--watch-trace-file", str(watch_trace),
    ]
    if forced_action is not None:
        command += ["--force-action", ",".join(map(str, forced_action))]
    before = model_fingerprint(model)
    environment = os.environ.copy()
    environment["COPTER_LOG_BASE"] = str(destination / "launcher_logs")
    with (destination / "driver.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command, cwd=ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=True
        )
    after = model_fingerprint(model)
    if before != after:
        raise RuntimeError(f"frozen evaluation modified model files for {label}")
    agent_record = last_eval_record(model / f"{exp}_metrics.jsonl", tag)
    write_json(destination / "agent_metrics.json", agent_record)
    write_json(marker, {
        "label": label, "repeat": repetition,
        "model_fingerprint": before, "forced_action": forced_action,
    })
    return destination


def common_keys(destinations):
    key_sets = [set(parse_fct(path / "result.fct")) for path in destinations]
    return set.intersection(*key_sets) if key_sets else set()


def average_rows(rows, fields):
    result = {}
    for field in fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        result[field] = mean(values) if values else None
        result[f"{field}_std"] = pstdev(values) if len(values) > 1 else 0.0
    return result


def fmt(value, digits=3, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def fmt_percent(value, digits=2):
    return "n/a" if value is None else f"{100.0 * value:.{digits}f}%"


def analyze(output: Path, protocol: dict, flow: Path) -> None:
    groups = {}
    for label_dir in sorted((output / "eval").iterdir()):
        if not label_dir.is_dir():
            continue
        groups[label_dir.name] = sorted(label_dir.glob("repeat_*"))
    all_destinations = [path for paths in groups.values() for path in paths]
    common = common_keys(all_destinations)
    rows = []
    for label, destinations in groups.items():
        for destination in destinations:
            agent = read_json(destination / "agent_metrics.json")
            row = {
                "label": label,
                "repeat": int(destination.name.split("_")[-1]),
                **metric_record(
                    destination, destination / "input.flow",
                    destination / "input.conf", agent, common,
                ),
            }
            if label.startswith("fixed_"):
                action_name = label[len("fixed_"):]
                action = FIXED_ACTIONS[action_name]
                row.update(threshold_activation(
                    destination, action, protocol["action_space"]
                ))
            rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with (output / "diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    numeric = [
        "reward", "mean_fct_us", "p95_fct_us", "p99_fct_us",
        "completion_ratio", "mean_throughput_mbps", "mean_queue_kb",
        "mean_ecn_rate", "ecn_positive_ratio", "queue_ge_kmin_ratio",
        "queue_ge_kmax_ratio", "buffer_utilization_mean",
    ]
    summary = {
        label: average_rows(
            [row for row in rows if row["label"] == label], numeric
        )
        for label in groups
    }
    write_json(output / "summary.json", summary)

    initial_records = [
        read_json(path / "agent_metrics.json")
        for path in groups.get("frozen_initial", [])
    ]
    final_records = [
        read_json(path / "agent_metrics.json")
        for path in groups.get("frozen_final", [])
    ]
    initial_hist = Counter()
    final_hist = Counter()
    for record in initial_records:
        initial_hist.update(action_histogram(record))
    for record in final_records:
        final_hist.update(action_histogram(record))
    policy_tv = tv_distance(initial_hist, final_hist)
    initial_hashes = {json.dumps(record.get("startup_parameter_hashes"), sort_keys=True) for record in initial_records}
    final_hashes = {json.dumps(record.get("startup_parameter_hashes"), sort_keys=True) for record in final_records}

    lines = [
        "# ACC pilot causal diagnostics", "",
        f"- Source run: **{protocol['run_id']}**",
        f"- Common completed flows across every evaluation: **{len(common)}**",
        f"- Frozen initial/final action TV distance: **{fmt(policy_tv, 4)}**",
        f"- Initial parameter hashes repeat-consistent: **{len(initial_hashes) == 1}**",
        f"- Final parameter hashes repeat-consistent: **{len(final_hashes) == 1}**",
        f"- Initial and final parameter hashes differ: **{initial_hashes != final_hashes}**",
        "", "## Fixed-action sensitivity", "",
        "| Action | Reward | Mean FCT us | p95 FCT us | p99 FCT us | Completion | Throughput Mbps | Queue KB | ECN | q>=Kmin | q>=Kmax |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in FIXED_ACTIONS:
        item = summary.get(f"fixed_{name}", {})
        lines.append(
            f"| {name} | {fmt(item.get('reward'),4)} | "
            f"{fmt(item.get('mean_fct_us'))} | {fmt(item.get('p95_fct_us'))} | "
            f"{fmt(item.get('p99_fct_us'))} | {fmt(item.get('completion_ratio'),4)} | "
            f"{fmt(item.get('mean_throughput_mbps'))} | {fmt(item.get('mean_queue_kb'))} | "
            f"{fmt(item.get('mean_ecn_rate'),6)} | "
            f"{fmt_percent(item.get('queue_ge_kmin_ratio'))} | "
            f"{fmt_percent(item.get('queue_ge_kmax_ratio'))} |"
        )
    lines += [
        "", "## Frozen policy", "",
        "| Checkpoint | Reward | Mean FCT us | p95 FCT us | p99 FCT us | Completion | Throughput Mbps | Queue KB | ECN | Dominant action |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, title in (("frozen_initial", "epoch 0"), ("frozen_final", "epoch 100")):
        item = summary.get(label, {})
        records = initial_records if label == "frozen_initial" else final_records
        histogram = Counter()
        for record in records:
            histogram.update(action_histogram(record))
        dominant = histogram.most_common(1)[0][0] if histogram else "n/a"
        lines.append(
            f"| {title} | {fmt(item.get('reward'),4)} | "
            f"{fmt(item.get('mean_fct_us'))} | {fmt(item.get('p95_fct_us'))} | "
            f"{fmt(item.get('p99_fct_us'))} | {fmt(item.get('completion_ratio'),4)} | "
            f"{fmt(item.get('mean_throughput_mbps'))} | {fmt(item.get('mean_queue_kb'))} | "
            f"{fmt(item.get('mean_ecn_rate'),6)} | {dominant} |"
        )
    lines += [
        "", "## Interpretation rules", "",
        "- Small fixed-action spreads mean the environment is insensitive to the control knobs.",
        "- Large fixed-action spreads with a small frozen-policy change mean the learner failed to exploit available control.",
        "- A large frozen action-TV with unchanged physical metrics means policy changes are not causally useful.",
        "- q>=Kmin near 0% is a dormant marking regime; q>=Kmax near 100% is a saturated regime.",
        "", "Detailed repetitions and standard deviations are in `diagnostics.csv` and `summary.json`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("fixed", "frozen", "all", "analyze"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--port", type=int, default=7700)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output_dir.resolve()
    if output == run_dir or output in run_dir.parents or run_dir in output.parents:
        parser.error("output must be a separate sibling directory")
    if args.repeats < 2:
        parser.error("use at least two repeats")
    protocol = read_json(run_dir / "protocol.json")
    if protocol["action_space"] != "multiscale":
        parser.error("this diagnostic currently registers the multiscale baseline only")
    config_path = run_dir / "pilot_config.yaml"
    if not config_path.exists():
        config_path = Path(protocol["config"])
    config = load_config(config_path)
    source_conf = ROOT / protocol["inputs"]["a"]["conf"]
    flow = ROOT / protocol["inputs"]["a"]["flow"]
    source_model, exp, state = source_checkpoint(run_dir)
    expected_epoch = sum(int(value) for value in protocol["epochs"].values())
    if int(state.get("epoch", -1)) != expected_epoch:
        parser.error(
            "source checkpoint is incomplete: "
            f"state epoch={state.get('epoch')}, expected={expected_epoch}"
        )
    experiment = {
        "schema_version": 1,
        "source_run": str(run_dir),
        "source_run_id": protocol["run_id"],
        "source_protocol_sha256": sha256(run_dir / "protocol.json"),
        "source_model_fingerprint": model_fingerprint(source_model),
        "source_global_train_step": state.get("global_train_step"),
        "repeats": args.repeats,
        "port": args.port,
        "fixed_actions": {key: list(value) for key, value in FIXED_ACTIONS.items()},
    }
    manifest = output / "protocol.json"
    if output.exists():
        if not args.resume and args.stage != "analyze":
            parser.error("existing output requires --resume")
        if not manifest.exists() or read_json(manifest) != experiment:
            parser.error("existing diagnostic protocol differs from the request")
    elif args.stage == "analyze":
        parser.error("missing diagnostic output")
    else:
        output.mkdir(parents=True)
        write_json(manifest, experiment)

    if args.stage != "analyze":
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        fixed_model = output / "models" / "fixed"
        initial_model = output / "models" / "initial"
        final_model = output / "models" / "final"
        fixed_model.mkdir(parents=True, exist_ok=True)
        initial_model.mkdir(parents=True, exist_ok=True)
        prepare_final_copy(source_model, final_model, exp)
        if args.stage in ("fixed", "all"):
            for name, action in FIXED_ACTIONS.items():
                for repetition in range(1, args.repeats + 1):
                    run_evaluation(
                        config, protocol, source_conf, flow, output,
                        fixed_model, exp, f"fixed_{name}", repetition,
                        args.port, action,
                    )
        if args.stage in ("frozen", "all"):
            for label, model in (
                ("frozen_initial", initial_model),
                ("frozen_final", final_model),
            ):
                for repetition in range(1, args.repeats + 1):
                    run_evaluation(
                        config, protocol, source_conf, flow, output,
                        model, exp, label, repetition, args.port,
                    )
    analyze(output, protocol, flow)
    print(f"ACC pilot diagnostics written to {output}")


if __name__ == "__main__":
    main()
