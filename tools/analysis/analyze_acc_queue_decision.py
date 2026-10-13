#!/usr/bin/env python3
"""Compare the registered ACC queue-penalty pilot on identical frozen flows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.analysis.metrics_core import (
    ANALYSIS_VERSION,
    conf_value,
    parse_fct,
    parse_pfc,
    parse_queue,
    parse_rate,
    parse_throughput,
    summarize_fct,
)

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("credit_control", "credit_queue1")
PHASES = ("frozen_initial", "frozen_final")
FIELDS = (
    "mean_fct_us", "p95_fct_us", "p99_fct_us", "completion_ratio",
    "mean_throughput_mbps", "mean_queue_kb", "mean_ecn_rate", "pfc_duty",
    "port349_reference_greedy_fraction", "port349_reference_margin_mean",
    "port349_action_8_0_fraction", "port349_congested_steps",
    "port349_throughput_component", "port349_queue_cost_sq",
    "port349_ecn_cost_sq", "port349_native_reward",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pair(protocols: dict) -> None:
    control, candidate = (protocols[arm] for arm in ARMS)
    # Compare every recorded protocol field except run-specific metadata and
    # the one registered treatment.  Future protocol additions are therefore
    # held fixed by default instead of silently becoming confounders.
    allowed = {"run_id", "config", "config_sha256", "created_at", "reward_queue_lambda"}
    differences = [
        key for key in sorted(set(control) | set(candidate))
        if key not in allowed and control.get(key) != candidate.get(key)
    ]
    if differences:
        raise ValueError(f"pilot arms differ outside queue lambda: {differences}")
    if control.get("reward_profile") != "tail_safe" or control.get("action_space") != "multiscale":
        raise ValueError("registered comparison requires tail_safe and multiscale")
    if (control.get("tasks") or {}).get("a") != "realistic_hadoop":
        raise ValueError("registered comparison requires the canonical Hadoop task")
    if (control.get("reward_queue_lambda"), candidate.get("reward_queue_lambda")) != (5.0, 1.0):
        raise ValueError("registered queue lambdas must be 5.0 and 1.0")
    if control.get("reward_ecn_lambda") != 5.0:
        raise ValueError("registered ECN lambda must remain 5.0")
    if control.get("reward_window_size") != 4 or control.get("replay_sampling") != "congestion_stratified":
        raise ValueError("registered comparison requires the same credit sampler/window")
    if control.get("diagnostic_reference_action") != "8,0":
        raise ValueError("registered Q reference must be action 8,0")


def evaluation_rows(run_dir: Path, diagnostic_dir: Path, expected_repeats: int):
    result = {}
    diagnostic = read_json(diagnostic_dir / "protocol.json")
    if diagnostic.get("source_protocol_sha256") != sha256(run_dir / "protocol.json"):
        raise ValueError(f"diagnostic protocol does not match source: {diagnostic_dir}")
    if int(diagnostic.get("repeats", 0)) != expected_repeats:
        raise ValueError(f"repeat count differs: {diagnostic_dir}")
    for phase in PHASES:
        paths = sorted((diagnostic_dir / "eval" / phase).glob("repeat_*"))
        if len(paths) != expected_repeats:
            raise ValueError(f"expected {expected_repeats} {phase} repeats: {diagnostic_dir}")
        for path in paths:
            required = (
                "complete.json", "input.flow", "input.conf", "result.fct",
                "result.queue", "result.rate", "result.throughput",
                "result.pfc", "agent_metrics.json",
            )
            missing = [name for name in required if not (path / name).is_file()]
            if missing:
                raise ValueError(f"incomplete frozen evaluation {path}: {missing}")
        result[phase] = paths
    return result


def port349_fields(agent: dict) -> dict:
    ranking = (agent.get("watch_ports_q_ranking") or {}).get("349")
    detail = (agent.get("watch_ports_metrics") or {}).get("349")
    if not ranking or not detail or not ranking.get("samples"):
        raise ValueError("port 349 Q/action observations are missing")
    population = "congested" if detail.get("congested_steps") else "active"
    actions = detail.get(f"actions_{population}") or {}
    total = sum(int(value) for value in actions.values())
    if not total:
        raise ValueError("port 349 action histogram is empty")
    components = detail.get(f"means_{population}") or {}
    required = ("throughput", "queue_cost_sq", "ecn_cost_sq", "reward")
    if any(components.get(key) is None for key in required):
        raise ValueError("port 349 reward decomposition is missing")
    return {
        "port349_reference_greedy_fraction": ranking.get("reference_greedy_fraction"),
        "port349_reference_margin_mean": ranking.get("reference_margin_mean"),
        "port349_action_8_0_fraction": int(actions.get("8,0", 0)) / total,
        "port349_congested_steps": int(detail.get("congested_steps") or 0),
        "port349_throughput_component": components["throughput"],
        "port349_queue_cost_sq": components["queue_cost_sq"],
        "port349_ecn_cost_sq": components["ecn_cost_sq"],
        "port349_native_reward": components["reward"],
    }


def measure(path: Path, common: set) -> dict:
    fct = summarize_fct(path / "result.fct", path / "input.flow", common)
    queue, _ = parse_queue(path / "result.queue")
    rate, rate_ports = parse_rate(path / "result.rate", warmup_buckets=2)
    throughput, _ = parse_throughput(path / "result.throughput")
    stop = float(conf_value(path / "input.conf", "SIMULATOR_STOP_TIME", 0.0))
    pfc, _ = parse_pfc(path / "result.pfc", stop, port_universe=rate_ports)
    return {
        **{key: fct[key] for key in ("mean_fct_us", "p95_fct_us", "p99_fct_us", "completion_ratio")},
        "mean_throughput_mbps": throughput.get("mean"),
        "mean_queue_kb": queue.get("mean"),
        "mean_ecn_rate": rate.get("ecn_rate_mean"),
        "pfc_duty": pfc.get("pfc_duty"),
        **port349_fields(read_json(path / "agent_metrics.json")),
    }


def decision(rows: dict) -> dict:
    initial = rows[("credit_queue1", "frozen_initial")]
    candidate = rows[("credit_queue1", "frozen_final")]
    control = rows[("credit_control", "frozen_final")]
    return {
        "candidate_vs_control_mean_fct_percent": 100 * (candidate["mean_fct_us"] / control["mean_fct_us"] - 1),
        "candidate_acquisition_mean_fct_percent": 100 * (candidate["mean_fct_us"] / initial["mean_fct_us"] - 1),
        "completion_cost_pp": 100 * (control["completion_ratio"] - candidate["completion_ratio"]),
        "p95_cost_percent": 100 * (candidate["p95_fct_us"] / control["p95_fct_us"] - 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two repeats are required")
    run_dirs = {arm: ROOT / "experiments" / "aba" / f"{args.run_id_prefix}_{arm}" for arm in ARMS}
    protocols = {arm: read_json(path / "protocol.json") for arm, path in run_dirs.items()}
    validate_pair(protocols)
    evaluations = {
        arm: evaluation_rows(path, path.with_name(path.name + "_diagnostics"), args.repeats)
        for arm, path in run_dirs.items()
    }
    paths = [path for arm in ARMS for phase in PHASES for path in evaluations[arm][phase]]
    common = set.intersection(*(set(parse_fct(path / "result.fct")) for path in paths))
    if not common:
        raise ValueError("no common completed flows across the two arms")
    rows = {}
    for arm in ARMS:
        for phase in PHASES:
            measured = [measure(path, common) for path in evaluations[arm][phase]]
            rows[(arm, phase)] = {
                key: mean(row[key] for row in measured) if all(row[key] is not None for row in measured) else None
                for key in FIELDS
            }
    result = decision(rows)
    advance = (
        result["candidate_vs_control_mean_fct_percent"] <= -3.0
        and result["candidate_acquisition_mean_fct_percent"] <= -3.0
        and result["completion_cost_pp"] <= 0.1
        and result["p95_cost_percent"] <= 5.0
    )
    output = ROOT / "experiments" / "aba" / f"{args.run_id_prefix}_reward_decision"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "frozen_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("arm", "checkpoint", *FIELDS))
        writer.writeheader()
        for (arm, phase), values in rows.items():
            writer.writerow({"arm": arm, "checkpoint": phase, **values})
    provenance = {
        "analysis_version": ANALYSIS_VERSION,
        "percentile_method": "linear interpolation at (n-1)*q",
        "rate_warmup_buckets": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "common_completed_flows": len(common),
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for directory in paths
            for path in (directory / "input.flow", directory / "input.conf", directory / "result.fct", directory / "agent_metrics.json")
        },
    }
    (output / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# ACC queue-penalty decision experiment", "",
        "- Task: canonical Hadoop acquisition; fixed traffic, seed, network, multiscale action, credit sampler/window and training budget.",
        "- Only training queue lambda differs: control 5.0, candidate 1.0; ECN lambda remains 5.0.",
        f"- Same completed-flow intersection across both arms/checkpoints/repeats: **{len(common)}**.",
        "- Native rewards have different definitions and are not compared across arms.",
        "- Repeated evaluations with identical traffic and seed check repeatability, not independent-seed significance.",
        "", "| Arm | Frozen checkpoint | Mean FCT us | p95 us | p99 us | Completion | Throughput Mbps | Queue KB | ECN | PFC duty | 349: 8,0 greedy | 349: Q margin | 349: selected 8,0 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (arm, phase), item in rows.items():
        def fmt(key, digits=3):
            value = item[key]
            return "n/a" if value is None else f"{value:.{digits}f}"
        lines.append(
            f"| {arm} | {phase} | {fmt('mean_fct_us')} | {fmt('p95_fct_us')} | "
            f"{fmt('p99_fct_us')} | {fmt('completion_ratio', 4)} | "
            f"{fmt('mean_throughput_mbps')} | {fmt('mean_queue_kb')} | "
            f"{fmt('mean_ecn_rate', 6)} | {fmt('pfc_duty', 6)} | "
            f"{fmt('port349_reference_greedy_fraction', 3)} | "
            f"{fmt('port349_reference_margin_mean', 4)} | "
            f"{fmt('port349_action_8_0_fraction', 3)} |"
        )
    lines += [
        "", "## Port 349 native reward decomposition", "",
        "Each row averages that arm's own congested samples (active samples only if none congested); these are not paired observations.",
        "The unclipped formula is throughput - queue lambda × queue cost² - 5 × ECN cost².",
        "", "| Arm | Frozen checkpoint | Throughput term | Queue cost² | ECN cost² | Native reward | Congested steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for (arm, phase), item in rows.items():
        lines.append(
            f"| {arm} | {phase} | {item['port349_throughput_component']:.4f} | "
            f"{item['port349_queue_cost_sq']:.5f} | {item['port349_ecn_cost_sq']:.5f} | "
            f"{item['port349_native_reward']:.4f} | {item['port349_congested_steps']:.1f} |"
        )
    lines += [
        "", "## Predeclared advance check", "",
        f"- Candidate final vs control final mean FCT: **{result['candidate_vs_control_mean_fct_percent']:+.2f}%** (require <= -3%).",
        f"- Candidate own frozen acquisition: **{result['candidate_acquisition_mean_fct_percent']:+.2f}%** (require <= -3%).",
        f"- Completion cost vs control: **{result['completion_cost_pp']:+.3f} pp** (require <= 0.1 pp).",
        f"- p95 cost vs control: **{result['p95_cost_percent']:+.2f}%** (require <= 5%).",
        f"- Advance to a separate independent-seed confirmation: **{'YES' if advance else 'NO'}**.",
        "- This is a screening rule, not a claim of statistical significance or forgetting.",
    ]
    (output / "DECISION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Decision report written to {output}")


if __name__ == "__main__":
    main()
