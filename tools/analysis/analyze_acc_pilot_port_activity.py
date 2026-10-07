#!/usr/bin/env python3
"""Condition ACC fixed-action queue measurements on observed port activity.

This is a descriptive analysis of completed frozen evaluations.  A queue
sample is paired with the most recent rate sample on the same physical port,
within one rate-monitor interval.  Unpaired samples are reported, never
silently classified as idle.  The script does not infer flow-to-port FCT.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "copter") not in sys.path:
    sys.path.insert(0, str(ROOT / "copter"))

from structures import acc_action_from_indices  # noqa: E402
from tools.analysis.metrics_core import (  # noqa: E402
    ANALYSIS_VERSION,
    conf_value,
    parse_queue,
    parse_rate,
    parse_throughput,
    percentile,
    read_queue_records,
    read_rate_records,
    sha256_file,
)

FIXED_ACTIONS = {
    "fixed_low_gentle": (0, 0),
    "fixed_low_strong": (0, 6),
    "fixed_mid": (4, 3),
    "fixed_high_gentle": (8, 0),
    "fixed_high_strong": (8, 6),
}
ACTIVE_CUTOFFS = (0.01, 0.05, 0.10)


def topology_switches(conf: Path) -> set[int]:
    value = conf_value(conf, "TOPOLOGY_FILE")
    if not value:
        raise ValueError(f"{conf}: missing TOPOLOGY_FILE")
    path = Path(str(value))
    if not path.is_absolute():
        path = ROOT / "simulation" / path
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().split()
        identifiers = handle.readline().split()
    count = int(header[1])
    if len(identifiers) != count:
        raise ValueError(f"{path}: expected {count} switch IDs")
    return {int(identifier) for identifier in identifiers}


def rate_index(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for item in records:
        grouped[(item["switch_id"], item["port_id"])].append(item)
    result = {}
    for key, items in grouped.items():
        items.sort(key=lambda item: item["time_s"])
        times = [item["time_s"] for item in items]
        intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
        # A single rate sample cannot define a valid interval for joining.
        if not intervals:
            continue
        result[key] = (times, items, median(intervals))
    return result


def preceding_rate(index: dict, key: tuple[int, int], time_s: float):
    series = index.get(key)
    if series is None:
        return None
    times, items, interval = series
    position = bisect_right(times, time_s) - 1
    if position < 0 or time_s - times[position] > interval + 1e-9:
        return None
    return items[position]


def threshold_bytes(conf: Path, action: tuple[int, int], capacity_bps: float):
    physical = acc_action_from_indices(action, "multiscale")
    scale = capacity_bps / 25e9
    kmin = (
        float(conf_value(conf, "OPENGYM_MIN_KMIN", 5000))
        + (
            float(conf_value(conf, "OPENGYM_MAX_KMIN", 50000))
            - float(conf_value(conf, "OPENGYM_MIN_KMIN", 5000))
        ) * physical.k_min_norm
    ) * scale
    kmax = (
        float(conf_value(conf, "OPENGYM_MIN_KMAX", 15000))
        + (
            float(conf_value(conf, "OPENGYM_MAX_KMAX", 100000))
            - float(conf_value(conf, "OPENGYM_MIN_KMAX", 15000))
        ) * physical.k_max_norm
    ) * scale
    return kmin, kmax


def summarize(samples: list[dict]) -> dict:
    queues = [item["queue_bytes"] / 1024 for item in samples]
    tx = [item["tx_rate"] for item in samples]
    ecn = [item["ecn_rate"] for item in samples]
    return {
        "samples": len(samples),
        "queue_mean_kb": mean(queues) if queues else None,
        "queue_p95_kb": percentile(queues, 0.95),
        "queue_max_kb": max(queues) if queues else None,
        "tx_rate_mean": mean(tx) if tx else None,
        "ecn_rate_mean": mean(ecn) if ecn else None,
        "ecn_positive_ratio": sum(value > 0 for value in ecn) / len(ecn) if ecn else None,
        "queue_positive_ratio": sum(value > 0 for value in queues) / len(queues) if queues else None,
        "queue_ge_8kb_ratio": sum(value >= 8 for value in queues) / len(queues) if queues else None,
        "queue_ge_24kb_ratio": sum(value >= 24 for value in queues) / len(queues) if queues else None,
        "queue_ge_kmin_ratio": (
            sum(item["ge_kmin"] for item in samples) / len(samples)
            if samples else None
        ),
        "queue_ge_kmax_ratio": (
            sum(item["ge_kmax"] for item in samples) / len(samples)
            if samples else None
        ),
    }


def analyze_trial(destination: Path, label: str, repeat: int, switches: set[int]):
    conf = destination / "input.conf"
    required = [
        destination / name for name in (
            "complete.json", "result.queue", "result.rate",
            "result.throughput", "agent_metrics.json", "input.conf",
        )
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"incomplete evaluation {destination}: {missing}")
    if topology_switches(conf) != switches:
        raise ValueError(f"topology changed within evaluations: {destination}")
    all_rate_records = read_rate_records(destination / "result.rate")
    warmup = sorted({item["time_s"] for item in all_rate_records})[:2]
    skipped = set(warmup)
    rate_records = [
        item for item in all_rate_records
        if item["time_s"] not in skipped
    ]
    index = rate_index(rate_records)
    action = FIXED_ACTIONS.get(label)
    paired = defaultdict(list)
    unpaired = defaultdict(int)
    total = defaultdict(int)
    for item in read_queue_records(destination / "result.queue"):
        key = (item["switch_id"], item["port_id"])
        total[key] += 1
        rate = preceding_rate(index, key, item["time_s"])
        if rate is None or rate["max_rate_bps"] <= 0:
            unpaired[key] += 1
            continue
        kmin, kmax = threshold_bytes(conf, action, rate["max_rate_bps"]) if action else (None, None)
        paired[key].append({
            "queue_bytes": item["queue_bytes"],
            "tx_rate": rate["tx_rate"],
            "ecn_rate": rate["ecn_rate"],
            "ge_kmin": kmin is not None and item["queue_bytes"] >= kmin,
            "ge_kmax": kmax is not None and item["queue_bytes"] >= kmax,
        })
    _, throughput_ports = parse_throughput(destination / "result.throughput")
    _, rate_ports = parse_rate(destination / "result.rate", 2)
    _, queue_ports = parse_queue(destination / "result.queue")
    watch = json.loads((destination / "agent_metrics.json").read_text(encoding="utf-8"))
    agent_ids = {}
    watch_reward = {}
    for agent_id, detail in (watch.get("watch_ports_metrics") or {}).items():
        identifier = detail.get("identifier")
        if identifier and "-" in identifier:
            try:
                key = tuple(map(int, identifier.split("-", 1)))
                agent_ids[key] = int(agent_id)
                watch_reward[key] = (
                    (detail.get("means_congested") or {}).get("reward")
                    if detail.get("congested_steps")
                    else (detail.get("means_active") or {}).get("reward")
                )
            except ValueError:
                pass
    rows = []
    for key in sorted(set(total) | set(rate_ports) | set(throughput_ports)):
        samples = paired[key]
        is_switch = key[0] in switches and key[1] in switches
        fields = {
            "label": label, "repeat": repeat, "switch_id": key[0],
            "peer_id": key[1], "identifier": f"{key[0]}-{key[1]}",
            "agent_port": agent_ids.get(key), "switch_switch": is_switch,
            "watched_port_reward": watch_reward.get(key),
            "queue_samples_total": total[key], "queue_samples_unpaired": unpaired[key],
            "pairing_ratio": len(samples) / total[key] if total[key] else None,
            "throughput_mean_mbps": throughput_ports.get(key, {}).get("mean"),
            "rate_ecn_mean_authoritative": rate_ports.get(key, {}).get("ecn_rate_mean"),
            "rate_tx_mean_authoritative": rate_ports.get(key, {}).get("tx_rate_mean"),
            "queue_p95_kb_authoritative": queue_ports.get(key, {}).get("p95"),
        }
        for cutoff in (0.0, *ACTIVE_CUTOFFS):
            selected = samples if cutoff == 0 else [s for s in samples if s["tx_rate"] > cutoff]
            suffix = "all" if cutoff == 0 else f"tx_gt_{int(cutoff * 100)}pct"
            for name, value in summarize(selected).items():
                if action is None and name.startswith("queue_ge_"):
                    value = None
                fields[f"{suffix}_{name}"] = value
        rows.append(fields)
    return rows


def write_csv(path: Path, rows: list[dict], fields=None):
    if not rows and fields is None:
        raise ValueError(f"no rows for {path}")
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def average(value_rows: list[dict], field: str):
    values = [float(row[field]) for row in value_rows if row.get(field) is not None]
    return mean(values) if values else None


def fmt(value, digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.diagnostic_dir.resolve()
    output = (args.output_dir or source / "port_activity").resolve()
    evaluation = source / "eval"
    destinations = sorted(evaluation.glob("*/repeat_*"))
    if not destinations:
        parser.error(f"no completed evaluations under {evaluation}")
    switches = topology_switches(destinations[0] / "input.conf")
    rows = []
    input_hashes = {}
    for destination in destinations:
        label = destination.parent.name
        repeat = int(destination.name.split("_")[-1])
        rows.extend(analyze_trial(destination, label, repeat, switches))
        input_hashes[str(destination.relative_to(source))] = {
            name: sha256_file(destination / name)
            for name in (
                "result.queue", "result.rate", "result.throughput",
                "input.conf", "input.flow", "agent_metrics.json",
            )
        }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "per_trial_ports.csv", rows)
    scopes = []
    for label in sorted({row["label"] for row in rows}):
        label_rows = [row for row in rows if row["label"] == label]
        for scope, eligible in (
            ("all_ports", label_rows),
            ("switch_switch", [row for row in label_rows if row["switch_switch"]]),
        ):
            for cutoff in (0, *ACTIVE_CUTOFFS):
                suffix = "all" if cutoff == 0 else f"tx_gt_{int(cutoff * 100)}pct"
                pooled = []
                for row in eligible:
                    count = row.get(f"{suffix}_samples") or 0
                    pooled.append((row, count))
                sample_total = sum(count for _, count in pooled)
                def weighted(field):
                    return (
                        sum(float(row[f"{suffix}_{field}"]) * count for row, count in pooled if count)
                        / sample_total if sample_total else None
                    )
                scopes.append({
                    "label": label, "scope": scope, "activity": suffix,
                    "samples": sample_total,
                    "queue_ge_kmin_ratio": weighted("queue_ge_kmin_ratio") if label in FIXED_ACTIONS else None,
                    "queue_ge_kmax_ratio": weighted("queue_ge_kmax_ratio") if label in FIXED_ACTIONS else None,
                    "queue_mean_kb": weighted("queue_mean_kb"),
                    "queue_ge_8kb_ratio": weighted("queue_ge_8kb_ratio"),
                    "queue_ge_24kb_ratio": weighted("queue_ge_24kb_ratio"),
                    "tx_rate_mean": weighted("tx_rate_mean"),
                    "ecn_rate_mean": weighted("ecn_rate_mean"),
                    "ports_with_samples": sum(count > 0 for _, count in pooled),
                    "pairing_ratio": (
                        sum(row["all_samples"] for row in eligible)
                        / sum(row["queue_samples_total"] for row in eligible)
                        if sum(row["queue_samples_total"] for row in eligible) else None
                    ),
                })
    write_csv(output / "scopes.csv", scopes)
    fixed = [row for row in rows if row["label"] in FIXED_ACTIONS and row["switch_switch"]]
    by_key = defaultdict(list)
    for row in fixed:
        by_key[(row["label"], row["identifier"])].append(row)
    rankings = []
    for identifier in sorted({identifier for _, identifier in by_key}):
        low = by_key.get(("fixed_low_gentle", identifier), [])
        high = by_key.get(("fixed_high_gentle", identifier), [])
        if not low or not high:
            continue
        def delta(field):
            h, l = average(high, field), average(low, field)
            return h - l if h is not None and l is not None else None
        rankings.append({
            "identifier": identifier,
            "agent_port": next((row["agent_port"] for row in low + high if row["agent_port"] is not None), None),
            "low_active_5pct_samples": sum(row["tx_gt_5pct_samples"] for row in low),
            "high_active_5pct_samples": sum(row["tx_gt_5pct_samples"] for row in high),
            "low_queue_ge_kmin_ratio": average(low, "tx_gt_5pct_queue_ge_kmin_ratio"),
            "high_queue_ge_kmin_ratio": average(high, "tx_gt_5pct_queue_ge_kmin_ratio"),
            "delta_throughput_mbps": delta("throughput_mean_mbps"),
            "delta_tx_rate": delta("rate_tx_mean_authoritative"),
            "delta_queue_p95_kb": delta("queue_p95_kb_authoritative"),
            "delta_ecn_rate": delta("rate_ecn_mean_authoritative"),
            "delta_watched_reward": delta("watched_port_reward"),
        })
    rankings.sort(
        key=lambda row: abs(row["delta_throughput_mbps"] or 0), reverse=True
    )
    write_csv(output / "switch_port_sensitivity.csv", rankings, [
        "identifier", "agent_port", "low_active_5pct_samples",
        "high_active_5pct_samples", "low_queue_ge_kmin_ratio",
        "high_queue_ge_kmin_ratio", "delta_throughput_mbps",
        "delta_tx_rate", "delta_queue_p95_kb", "delta_ecn_rate",
        "delta_watched_reward",
    ])
    top = rankings[:10]
    base = next((row for row in scopes if row["label"] == "fixed_mid" and row["scope"] == "switch_switch" and row["activity"] == "all"), None)
    lines = [
        "# ACC pilot port activity", "",
        f"- Evaluations: **{len(destinations)}**; monitored switches: **{len(switches)}**.",
        "- Queue samples use the latest same-port rate sample within one rate-monitor interval.",
        "- Activity means normalized tx rate above 1%, 5%, or 10%; the first two rate buckets are skipped.",
        "- The port table ranks absolute throughput change between high_gentle and low_gentle; it does not attribute FCT to a port.",
        "- Kmin/Kmax differ by action. The shared 8/24 KiB queue columns allow the same physical comparison across actions.",
    ]
    if base:
        lines.append(
            f"- Mid action, switch-switch pairing coverage: **{fmt(base['pairing_ratio'] * 100 if base['pairing_ratio'] is not None else None, 1)}%**."
        )
        if base["pairing_ratio"] is not None and base["pairing_ratio"] < 0.9:
            lines.append(
                "- Pairing coverage is below 90%; inspect queue and rate sampling times before interpreting conditional percentages."
            )
    lines += [
        "", "## Fixed-action threshold activation", "",
        "| Action | Scope | Activity | Samples | q≥Kmin | q≥Kmax | q≥8 KiB | q≥24 KiB | Mean queue KiB | Mean ECN |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scopes:
        if row["label"] not in FIXED_ACTIONS:
            continue
        lines.append(
            f"| {row['label'].removeprefix('fixed_')} | {row['scope']} | {row['activity']} | "
            f"{row['samples']} | {fmt(row['queue_ge_kmin_ratio'] * 100 if row['queue_ge_kmin_ratio'] is not None else None, 2)}% | "
            f"{fmt(row['queue_ge_kmax_ratio'] * 100 if row['queue_ge_kmax_ratio'] is not None else None, 2)}% | "
            f"{fmt(row['queue_ge_8kb_ratio'] * 100 if row['queue_ge_8kb_ratio'] is not None else None, 2)}% | "
            f"{fmt(row['queue_ge_24kb_ratio'] * 100 if row['queue_ge_24kb_ratio'] is not None else None, 2)}% | "
            f"{fmt(row['queue_mean_kb'])} | {fmt(row['ecn_rate_mean'], 6)} |"
        )
    lines += [
        "", "## Most sensitive switch-switch ports", "",
        "| Physical port | Agent port | Δ throughput Mbps | Δ queue p95 KB | Δ ECN | Δ watched reward | Active samples low/high |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top:
        lines.append(
            f"| {row['identifier']} | {row['agent_port'] if row['agent_port'] is not None else 'n/a'} | "
            f"{fmt(row['delta_throughput_mbps'])} | {fmt(row['delta_queue_p95_kb'])} | "
            f"{fmt(row['delta_ecn_rate'], 6)} | {fmt(row['delta_watched_reward'], 4)} | "
            f"{row['low_active_5pct_samples']}/{row['high_active_5pct_samples']} |"
        )
    lines += [
        "", "Detailed physical ports are in `per_trial_ports.csv`; all scopes are in `scopes.csv`.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps({
        "analysis_version": ANALYSIS_VERSION,
        "join": "latest preceding same-port rate within median rate interval",
        "rate_warmup_buckets": 2,
        "input_sha256": input_hashes,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Port activity report written to {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
