#!/usr/bin/env python3
"""Frozen ACC port-set intervention with activity-matched random controls.

The selection uses only existing fixed-action link measurements.  It is an
exploratory mechanism test, not an optimized policy or a significance test.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import socket
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "copter") not in sys.path:
    sys.path.insert(0, str(ROOT / "copter"))

from scripts.continual_validation.run_aba import load_config  # noqa: E402
from scripts.continual_validation.run_acc_pilot_diagnostics import (  # noqa: E402
    common_keys,
    metric_record,
    model_fingerprint,
    prepare_final_copy,
    read_json,
    run_evaluation,
    sha256,
    source_checkpoint,
    write_json,
)
from tools.analysis.analyze_acc_pilot_port_activity import (  # noqa: E402
    topology_switches,
)
from tools.analysis.metrics_core import (  # noqa: E402
    ANALYSIS_VERSION,
    conf_value,
    parse_pfc,
    parse_queue,
    parse_rate,
    parse_throughput,
)
from structures import validate_acc_action_indices  # noqa: E402


def parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("--sizes must be increasing positive integers") from exc
    if not sizes or any(size <= 0 for size in sizes) or tuple(sorted(set(sizes))) != sizes:
        raise ValueError("--sizes must be increasing positive integers")
    return sizes


def wait_port_free(port: int, timeout_s: int = 180) -> None:
    for _ in range(timeout_s):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(1)
    raise TimeoutError(f"port {port} remained unavailable for {timeout_s}s")


def completed_trials(diagnostic: Path, label: str, flow_hash: str) -> list[Path]:
    directory = diagnostic / "eval" / label
    paths = sorted(directory.glob("repeat_*"))
    expected = int(read_json(diagnostic / "protocol.json")["repeats"])
    if len(paths) != expected:
        raise ValueError(f"{directory}: expected {expected} repeats, found {len(paths)}")
    for path in paths:
        for name in ("complete.json", "input.flow", "result.throughput"):
            if not (path / name).is_file():
                raise ValueError(f"{path}: missing {name}")
        if sha256(path / "input.flow") != flow_hash:
            raise ValueError(f"{path}: fixed-action flow differs from source")
    return paths


def average_port_throughput(paths: list[Path]) -> dict[tuple[int, int], float]:
    values = defaultdict(list)
    for path in paths:
        _, ports = parse_throughput(path / "result.throughput")
        for key, detail in ports.items():
            if detail.get("mean") is not None:
                values[key].append(float(detail["mean"]))
    return {key: mean(samples) for key, samples in values.items()}


def select_port_sets(identifiers: list[str], switch_ids: set[int],
                     active_rates: dict, low: dict, high: dict,
                     sizes: tuple[int, ...], seed: int) -> dict:
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("frozen policy did not record a complete port mapping")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("frozen policy port mapping contains duplicates")
    eligible = []
    for agent_port, identifier in enumerate(identifiers):
        try:
            switch, peer = (int(item) for item in identifier.split("-", 1))
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"invalid port identifier {identifier!r}") from exc
        key = switch, peer
        rate = active_rates.get(key, {}).get("tx_rate_mean")
        if switch not in switch_ids or peer not in switch_ids:
            continue
        if rate is None or rate <= 0.01 or key not in low or key not in high:
            continue
        eligible.append({
            "agent_port": agent_port,
            "identifier": identifier,
            "tx_rate_mean": float(rate),
            "high_minus_low_throughput_mbps": high[key] - low[key],
        })
    responsive = sorted(
        (item for item in eligible if item["high_minus_low_throughput_mbps"] > 0),
        key=lambda item: (-item["high_minus_low_throughput_mbps"], item["agent_port"]),
    )
    count = sizes[-1]
    if len(responsive) < count:
        raise ValueError(
            f"only {len(responsive)} active switch-switch ports have positive "
            f"high-minus-low throughput; requested top {count}"
        )
    targets = [item["agent_port"] for item in responsive[:count]]
    # Random controls come from the more active half of the same eligible
    # switch-switch population, excluding every selected target.
    controls = sorted(
        (item for item in eligible if item["agent_port"] not in targets),
        key=lambda item: (-item["tx_rate_mean"], item["agent_port"]),
    )
    pool = controls[:max(count, (len(controls) + 1) // 2)]
    if len(pool) < count:
        raise ValueError(f"only {len(pool)} disjoint active control ports; need {count}")
    random_order = random.Random(seed).sample(
        [item["agent_port"] for item in pool], count
    )
    return {
        "target_order": targets,
        "control_order": random_order,
        "eligible_count": len(eligible),
        "random_pool": [item["agent_port"] for item in pool],
        "selection_basis": "positive fixed high-minus-low throughput among active switch-switch ports",
        "active_tx_rate_threshold": 0.01,
        "ranking": responsive,
    }


def labels(plan: dict, sizes: tuple[int, ...]) -> list[tuple[str, tuple[int, ...] | None, bool]]:
    result = [("policy", None, False)]
    for size in sizes:
        result.append((f"target_{size}", tuple(plan["target_order"][:size]), False))
        result.append((f"control_{size}", tuple(plan["control_order"][:size]), False))
    result.append(("all_high_gentle", None, True))
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(output: Path, flow: Path, plan: dict, sizes: tuple[int, ...],
            repeats: int, request: dict) -> None:
    trials = [
        (label, ports, all_high, output / "eval" / label / f"repeat_{repeat}")
        for label, ports, all_high in labels(plan, sizes)
        for repeat in range(1, repeats + 1)
    ]
    for label, ports, all_high, path in trials:
        marker = path / "complete.json"
        if not marker.is_file():
            raise ValueError(f"missing completed evaluation: {path}")
        recorded = read_json(marker)
        expected_ports = [list(ports), list(request["action"])] if ports else None
        if recorded.get("forced_ports_action") != expected_ports:
            raise ValueError(f"forced port set differs from plan: {path}")
        if recorded.get("forced_action") != (list(request["action"]) if all_high else None):
            raise ValueError(f"all-port action differs from plan: {path}")
    common = common_keys([path for _, _, _, path in trials])
    if not common:
        raise ValueError("no common completed flows across interventions")
    rows = []
    port_rows = []
    startup_hashes = set()
    for label, ports, _, path in trials:
        record = read_json(path / "agent_metrics.json")
        startup_hashes.add(json.dumps(record.get("startup_parameter_hashes"), sort_keys=True))
        result = metric_record(path, flow, path / "input.conf", record, common)
        _, physical_queue = parse_queue(path / "result.queue")
        _, physical_rate = parse_rate(path / "result.rate", 2)
        _, physical_throughput = parse_throughput(path / "result.throughput")
        stop_time = float(conf_value(path / "input.conf", "SIMULATOR_STOP_TIME", 0.0))
        _, physical_pfc = parse_pfc(
            path / "result.pfc", stop_time, port_universe=physical_rate
        )
        rows.append({
            "label": label,
            "repeat": int(path.name.split("_")[-1]),
            "forced_ports": ",".join(map(str, ports or ())),
            "common_completed_flows": len(common),
            **result,
        })
        watched = record.get("watch_ports_metrics") or {}
        ranking = record.get("watch_ports_q_ranking") or {}
        for port, detail in watched.items():
            identifier = detail.get("identifier")
            physical_key = None
            if identifier:
                physical_key = tuple(int(part) for part in identifier.split("-", 1))
            queue = physical_queue.get(physical_key, {})
            rate = physical_rate.get(physical_key, {})
            throughput = physical_throughput.get(physical_key, {})
            pfc = physical_pfc.get(physical_key, {})
            population = (
                "congested" if detail.get("congested_steps") else "active"
            )
            reward = (detail.get(f"means_{population}") or {}).get("reward")
            port_rows.append({
                "label": label,
                "repeat": int(path.name.split("_")[-1]),
                "agent_port": port,
                "identifier": identifier,
                "forced": int(int(port) in (ports or ())),
                "population": population,
                "active_steps": detail.get("active_steps"),
                "congested_steps": detail.get("congested_steps"),
                "reward": reward,
                "queue_mean_kb": queue.get("mean"),
                "queue_p95_kb": queue.get("p95"),
                "queue_max_kb": queue.get("max"),
                "ecn_rate_mean": rate.get("ecn_rate_mean"),
                "ecn_positive_ratio": rate.get("ecn_positive_ratio"),
                "throughput_mean_mbps": throughput.get("mean"),
                "pfc_pause_count": pfc.get("pfc_pause_count"),
                "pfc_duty": pfc.get("pfc_duty"),
                "reference_greedy_fraction": (ranking.get(port) or {}).get("reference_greedy_fraction"),
                "reference_margin_mean": (ranking.get(port) or {}).get("reference_margin_mean"),
            })
    if len(startup_hashes) != 1:
        raise ValueError("frozen evaluations did not start from identical model parameters")
    write_csv(output / "intervention_metrics.csv", rows)
    write_csv(output / "intervention_ports.csv", port_rows)
    fields = (
        "reward", "mean_fct_us", "p95_fct_us", "p99_fct_us",
        "completion_ratio", "mean_throughput_mbps", "mean_queue_kb",
        "mean_ecn_rate", "pfc_duty",
    )
    summary = {}
    for label, _, _ in labels(plan, sizes):
        selected = [row for row in rows if row["label"] == label]
        summary[label] = {}
        for key in fields:
            values = [float(row[key]) for row in selected if row.get(key) is not None]
            summary[label][key] = mean(values) if values else None
        summary[label]["repeat_count"] = len(selected)
        summary[label]["mean_fct_std_us"] = pstdev(
            float(row["mean_fct_us"]) for row in selected
        ) if len(selected) > 1 else 0.0
    write_json(output / "summary.json", summary)
    base = summary["policy"]
    lines = [
        "# Frozen ACC port-set intervention", "",
        f"- Source: **{request['source_run_id']}**; action: **{','.join(map(str, request['action']))}**.",
        f"- Common completed flows: **{len(common)}**; repeats per arm: **{repeats}**.",
        f"- Active switch-switch candidates: **{plan['eligible_count']}**.",
        f"- Target order: **{','.join(map(str, plan['target_order']))}**.",
        f"- Random control order (seed {request['seed']}): **{','.join(map(str, plan['control_order']))}**.",
        "- Port selection uses fixed-action link throughput and baseline activity, not intervention FCT.",
        "- Repeats with identical traffic/seed are repeatability checks, not independent seeds.",
        "", "| Arm | Mean FCT us | vs policy | p95 us | p99 us | Completion | Throughput Mbps | Reward |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in labels(plan, sizes):
        item = summary[label]
        change = 100 * (item["mean_fct_us"] / base["mean_fct_us"] - 1)
        lines.append(
            f"| {label} | {item['mean_fct_us']:.3f} | {change:+.2f}% | "
            f"{item['p95_fct_us']:.3f} | {item['p99_fct_us']:.3f} | "
            f"{item['completion_ratio']:.4f} | "
            f"{item['mean_throughput_mbps']:.3f} | "
            f"{item['reward']:.4f} |"
        )
    lines += [
        "", "| Set size | Target minus equal-size control, mean FCT us |",
        "|---:|---:|",
    ]
    for size in sizes:
        difference = (
            summary[f"target_{size}"]["mean_fct_us"]
            - summary[f"control_{size}"]["mean_fct_us"]
        )
        lines.append(f"| {size} | {difference:+.3f} |")
    lines += [
        "", "Selection cannot attribute the all-port FCT gain to any one link. "
        "Compare each target set with the equal-size random control; "
        "port-level queue, ECN, reward, and Q rankings are in the CSV files.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    input_hashes = {}
    for _, _, _, path in trials:
        for name in (
            "input.conf", "input.flow", "result.fct", "result.queue",
            "result.rate", "result.throughput", "result.pfc",
            "agent_metrics.json",
        ):
            input_hashes[str(path.relative_to(output) / name)] = sha256(path / name)
    write_json(output / "analysis_provenance.json", {
        "analysis_version": ANALYSIS_VERSION,
        "source_protocol_sha256": request["source_protocol_sha256"],
        "diagnostic_protocol_sha256": request["diagnostic_protocol_sha256"],
        "flow_sha256": request["flow_sha256"],
        "source_conf_sha256": request["source_conf_sha256"],
        "percentile_method": "linear interpolation at (n-1)*q",
        "rate_warmup_buckets": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hashes,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=("all", "analyze"), default="all")
    parser.add_argument("--sizes", default="1,2,4")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port", type=int, default=7820)
    parser.add_argument("--action", default="8,0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sizes = parse_sizes(args.sizes)
    if args.repeats < 1 or not 1 <= args.port <= 65535:
        parser.error("--repeats must be positive and --port must be valid")
    run_dir = args.run_dir.resolve()
    diagnostic = (args.diagnostic_dir or Path(f"{run_dir}_diagnostics")).resolve()
    output = (args.output_dir or Path(f"{run_dir}_port_intervention")).resolve()
    if (
        output in (run_dir, diagnostic)
        or output in run_dir.parents or output in diagnostic.parents
        or run_dir in output.parents or diagnostic in output.parents
    ):
        parser.error("output must be a separate sibling directory")
    source_protocol = read_json(run_dir / "protocol.json")
    if source_protocol["action_space"] != "multiscale":
        parser.error("port-set intervention currently requires multiscale ACC")
    action = validate_acc_action_indices(args.action.split(","), "multiscale")
    source_model, exp, state = source_checkpoint(run_dir)
    expected_epoch = sum(int(value) for value in source_protocol["epochs"].values())
    if int(state.get("epoch", -1)) != expected_epoch:
        parser.error("source checkpoint is incomplete")
    config_path = run_dir / "pilot_config.yaml"
    config = load_config(config_path if config_path.exists() else Path(source_protocol["config"]))
    flow = ROOT / source_protocol["inputs"]["a"]["flow"]
    source_conf = ROOT / source_protocol["inputs"]["a"]["conf"]
    diagnostic_protocol = read_json(diagnostic / "protocol.json")
    if (
        Path(diagnostic_protocol["source_run"]).resolve() != run_dir
        or diagnostic_protocol["source_protocol_sha256"] != sha256(run_dir / "protocol.json")
        or diagnostic_protocol["source_model_fingerprint"] != model_fingerprint(source_model)
    ):
        parser.error("fixed-action diagnostic does not match this source checkpoint")
    low_paths = completed_trials(diagnostic, "fixed_low_gentle", sha256(flow))
    high_paths = completed_trials(diagnostic, "fixed_high_gentle", sha256(flow))
    request = {
        "schema_version": 1,
        "source_run_id": source_protocol["run_id"],
        "source_protocol_sha256": sha256(run_dir / "protocol.json"),
        "diagnostic_protocol_sha256": sha256(diagnostic / "protocol.json"),
        "source_model_fingerprint": model_fingerprint(source_model),
        "flow_sha256": sha256(flow),
        "source_conf_sha256": sha256(source_conf),
        "fixed_throughput_sha256": {
            str(path.relative_to(diagnostic)): sha256(path / "result.throughput")
            for path in low_paths + high_paths
        },
        "sizes": list(sizes), "repeats": args.repeats,
        "seed": args.seed, "port": args.port, "action": list(action),
    }
    manifest = output / "protocol.json"
    if output.exists():
        if not manifest.is_file() or read_json(manifest) != request:
            parser.error("existing output has a different protocol")
        if args.stage != "analyze" and not args.resume:
            parser.error("existing output requires --resume")
    elif args.stage == "analyze":
        parser.error("missing intervention output")
    else:
        output.mkdir(parents=True)
        write_json(manifest, request)
    model = output / "models" / "final"
    if args.stage != "analyze":
        prepare_final_copy(source_model, model, exp)
        for repeat in range(1, args.repeats + 1):
            wait_port_free(args.port)
            run_evaluation(
                config, source_protocol, source_conf, flow, output, model,
                exp, "policy", repeat, args.port,
            )
    plan_path = output / "plan.json"
    if plan_path.is_file():
        plan = read_json(plan_path)
    elif args.stage == "analyze":
        parser.error("missing port selection plan")
    else:
        baseline = output / "eval" / "policy" / "repeat_1"
        identifiers = read_json(baseline / "agent_metrics.json").get("port_identifiers")
        if identifiers is None or len(identifiers) != int(state["node_number"]):
            raise ValueError("baseline did not record all agent-to-physical port identifiers")
        _, active_rates = parse_rate(baseline / "result.rate", 2)
        plan = select_port_sets(
            identifiers,
            topology_switches(baseline / "input.conf"),
            active_rates,
            average_port_throughput(low_paths),
            average_port_throughput(high_paths),
            sizes, args.seed,
        )
        write_json(plan_path, plan)
    watch = sorted(set(
        int(item) for item in str(config.get("watch_ports", "")).split(",") if item.strip()
    ) | set(plan["target_order"]) | set(plan["control_order"]))
    if args.stage != "analyze":
        for label, ports, all_high in labels(plan, sizes)[1:]:
            for repeat in range(1, args.repeats + 1):
                wait_port_free(args.port)
                run_evaluation(
                    config, source_protocol, source_conf, flow, output, model,
                    exp, label, repeat, args.port,
                    forced_action=action if all_high else None,
                    forced_ports_action=(ports, action) if ports else None,
                    watch_ports=",".join(map(str, watch)),
                )
                print(f"completed {label} repeat {repeat}", flush=True)
    analyze(output, flow, plan, sizes, args.repeats, request)
    print(f"ACC port intervention written to {output}")


if __name__ == "__main__":
    main()
