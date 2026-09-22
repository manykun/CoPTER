#!/usr/bin/env python3
"""Authoritative metric formulas for CoPTER experiments.

All formal ABA and continual-learning reports import this module.  Keep parsing
and aggregation here deterministic, dependency-light, and covered by fixtures.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ANALYSIS_VERSION = "aba-metrics-v1"
PERCENTILE_METHOD = "linear: position=(n-1)*q"
DEFAULT_RATE_WARMUP_BUCKETS = 2


def percentile(values: Iterable[float], fraction: float) -> Optional[float]:
    """Linear-interpolated quantile using position=(n-1)*q."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def numeric_summary(values: Iterable[float], prefix: str = "") -> Dict[str, Optional[float]]:
    selected = [float(value) for value in values if value is not None]
    key = (lambda name: f"{prefix}_{name}" if prefix else name)
    return {
        key("mean"): mean(selected) if selected else None,
        key("p50"): percentile(selected, 0.50),
        key("p95"): percentile(selected, 0.95),
        key("p99"): percentile(selected, 0.99),
        key("max"): max(selected) if selected else None,
        key("samples"): len(selected),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_flow_count(path: Path) -> int:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            return int(line.split()[0])
    raise ValueError(f"empty flow file: {path}")


def parse_fct(path: Path) -> Dict[Tuple[object, ...], Tuple[float, float]]:
    """Return flow identity -> (actual_ns, ideal_ns) for valid completions."""
    flows: Dict[Tuple[object, ...], Tuple[float, float]] = {}
    occurrences: Dict[Tuple[str, ...], int] = defaultdict(int)
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_number}: expected at least 8 columns")
            base_key = tuple(fields[:6])
            occurrence = occurrences[base_key]
            occurrences[base_key] += 1
            try:
                actual_ns = float(fields[6])
                ideal_ns = float(fields[7])
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid FCT value") from exc
            if actual_ns > 0 and ideal_ns > 0:
                flows[base_key + (occurrence,)] = (actual_ns, ideal_ns)
    return flows


def summarize_fct(
    fct_path: Path,
    flow_path: Path,
    common_keys: Optional[Iterable[Tuple[object, ...]]] = None,
) -> Dict[str, Optional[float]]:
    """Summarize one epoch; common_keys is used only for checkpoint comparisons."""
    flows = parse_fct(fct_path)
    expected = expected_flow_count(flow_path)
    if common_keys is None:
        selected = flows
    else:
        keys = set(common_keys)
        selected = {key: value for key, value in flows.items() if key in keys}
    fct_us = [actual / 1000.0 for actual, _ in selected.values()]
    slowdowns = [max(1.0, actual / ideal) for actual, ideal in selected.values()]
    return {
        "expected_flows": expected,
        "completed_flows": len(flows),
        "matched_flows": len(selected),
        "completion_ratio": len(flows) / expected if expected else None,
        "mean_fct_us": mean(fct_us) if fct_us else None,
        "p50_fct_us": percentile(fct_us, 0.50),
        "p95_fct_us": percentile(fct_us, 0.95),
        "p99_fct_us": percentile(fct_us, 0.99),
        "mean_slowdown": mean(slowdowns) if slowdowns else None,
        "p95_slowdown": percentile(slowdowns, 0.95),
        "p99_slowdown": percentile(slowdowns, 0.99),
    }


def _summarize_grouped(
    grouped: Mapping[Tuple[int, int], Sequence[float]],
    scale: float,
) -> Tuple[Dict[str, Optional[float]], Dict[Tuple[int, int], Dict[str, Optional[float]]]]:
    all_values = [value * scale for values in grouped.values() for value in values]
    per_port = {
        key: numeric_summary((value * scale for value in values))
        for key, values in grouped.items()
    }
    return numeric_summary(all_values), per_port


def read_queue_records(path: Path) -> List[dict]:
    """Read simulator queue records without applying aggregation."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(f"{path}:{line_number}: expected 5 queue columns")
            records.append({
                "switch_id": int(fields[0]),
                "switch_buffer_bytes": int(fields[1]),
                "port_id": int(fields[2]),
                "queue_bytes": float(fields[3]),
                "time_s": float(fields[4]),
            })
    return records


def parse_queue(path: Path) -> Tuple[dict, dict]:
    """Parse queue bytes and report KB (1024 bytes per KB)."""
    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for record in read_queue_records(path):
        grouped[(record["switch_id"], record["port_id"])].append(
            record["queue_bytes"]
        )
    return _summarize_grouped(grouped, 1.0 / 1024.0)


def _skip_time_buckets(records: List[Tuple[float, Tuple[int, int], float, float]], count: int):
    if count <= 0:
        return records
    buckets = sorted({record[0] for record in records})
    skipped = set(buckets[:count])
    return [record for record in records if record[0] not in skipped]


def read_rate_records(path: Path) -> List[dict]:
    """Read simulator normalized tx/ECN rate records without aggregation."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 6:
                raise ValueError(f"{path}:{line_number}: expected 6 rate columns")
            records.append({
                "switch_id": int(fields[0]),
                "port_id": int(fields[1]),
                "max_rate_bps": int(fields[2]),
                "tx_rate": float(fields[3]),
                "ecn_rate": float(fields[4]),
                "time_s": float(fields[5]),
            })
    return records


def parse_rate(path: Path, warmup_buckets: int = DEFAULT_RATE_WARMUP_BUCKETS) -> Tuple[dict, dict]:
    """Parse normalized tx-rate and ECN-rate monitor values.

    The simulator writes link capacity in column 3 and normalized tx/ECN
    fractions in columns 4/5. Absolute bps throughput is read from the
    dedicated throughput file by :func:`parse_throughput`.
    """
    records = [
        (
            record["time_s"],
            (record["switch_id"], record["port_id"]),
            record["tx_rate"],
            record["ecn_rate"],
        )
        for record in read_rate_records(path)
    ]
    records = _skip_time_buckets(records, warmup_buckets)
    tx: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    ecn: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for _, key, txrate, ecnrate in records:
        tx[key].append(txrate)
        ecn[key].append(ecnrate)
    all_tx = [value for values in tx.values() for value in values]
    all_ecn = [value for values in ecn.values() for value in values]
    overall = {
        **numeric_summary(all_tx, "tx_rate"),
        **numeric_summary(all_ecn, "ecn_rate"),
        "ecn_positive_ratio": (
            sum(value > 0 for value in all_ecn) / len(all_ecn) if all_ecn else None
        ),
        "rate_warmup_buckets": warmup_buckets,
    }
    per_port = {}
    for key in set(tx) | set(ecn):
        tx_values = tx.get(key, [])
        ecn_values = ecn.get(key, [])
        per_port[key] = {
            **numeric_summary(tx_values, "tx_rate"),
            **numeric_summary(ecn_values, "ecn_rate"),
            "ecn_positive_ratio": (
                sum(value > 0 for value in ecn_values) / len(ecn_values)
                if ecn_values else None
            ),
        }
    return overall, per_port


def read_throughput_records(path: Path) -> List[dict]:
    """Read absolute throughput records without applying aggregation."""
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected throughput columns")
            records.append({
                "switch_id": int(fields[0]),
                "port_id": int(fields[1]),
                "throughput_bps": float(fields[2]),
                "time_s": float(fields[3]) if len(fields) > 3 else None,
                "max_rate_bps": float(fields[4]) if len(fields) > 4 else None,
            })
    return records


def parse_throughput(path: Path) -> Tuple[dict, dict]:
    """Parse throughput monitor bps -> Mbps without rate warm-up filtering."""
    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for record in read_throughput_records(path):
        grouped[(record["switch_id"], record["port_id"])].append(
            record["throughput_bps"]
        )
    return _summarize_grouped(grouped, 1.0 / 1_000_000.0)


def summarize_watch_port_trace(records: Sequence[Mapping[str, object]], buffer_kb: float) -> dict:
    """Authoritative summary for Agent watch-port traces.

    Congested samples are preferred; active samples are the explicit fallback.
    Queue values in the trace are normalized by switch-buffer size.
    """
    congested = [record for record in records if record.get("congested")]
    active = [record for record in records if record.get("active")]
    population = congested or active
    queue = [
        float(record["peak_queue"]) * float(buffer_kb)
        for record in population if record.get("peak_queue") is not None
    ]
    ecn = [
        float(record["avg_ecn"])
        for record in population if record.get("avg_ecn") is not None
    ]
    return {
        "population": "congested" if congested else "active",
        "samples": len(population),
        "queue_mean_kb": mean(queue) if queue else None,
        "queue_p95_kb": percentile(queue, 0.95),
        "queue_p99_kb": percentile(queue, 0.99),
        "queue_max_kb": max(queue) if queue else None,
        "ecn_mean": mean(ecn) if ecn else None,
        "ecn_p95": percentile(ecn, 0.95),
        "ecn_p99": percentile(ecn, 0.99),
        "ecn_max": max(ecn) if ecn else None,
        "ecn_positive_ratio": sum(value > 0 for value in ecn) / len(ecn) if ecn else None,
    }


def parse_pfc(
    path: Path,
    stop_time_s: float,
    port_universe: Optional[Iterable[Tuple[int, int]]] = None,
) -> Tuple[dict, dict]:
    """Parse pause/resume events keyed by physical switch-peer link.

    Files without the peer column have no safe port mapping and return n/a
    rather than fabricated zeros.
    """
    events: Dict[Tuple[int, int], List[Tuple[float, int]]] = defaultdict(list)
    mapping_available = True
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 6:
                mapping_available = False
                continue
            try:
                switch_id = int(fields[1])
                peer_id = int(fields[5])
                if peer_id < 0:
                    mapping_available = False
                    continue
                events[(switch_id, peer_id)].append(
                    (float(fields[0]) / 1e9, int(fields[4]))
                )
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid PFC record") from exc
    empty = {
        "pfc_mapping_available": False,
        "pfc_pause_count": None,
        "pfc_resume_count": None,
        "pfc_pause_total_s": None,
        "pfc_max_pause_s": None,
        "pfc_duty": None,
        "pfc_final_paused": None,
    }
    if not mapping_available and events:
        raise ValueError(
            f"{path}: mixed mapped and unmapped PFC records; refusing partial statistics"
        )
    if not mapping_available:
        return empty, {}
    universe = set(port_universe or ())
    if not events and not universe:
        return {
            "pfc_mapping_available": True,
            "pfc_pause_count": 0,
            "pfc_resume_count": 0,
            "pfc_pause_total_s": 0.0,
            "pfc_max_pause_s": 0.0,
            "pfc_duty": 0.0,
            "pfc_final_paused": False,
        }, {}
    per_port = {}
    all_durations: List[float] = []
    total_pause_count = 0
    total_resume_count = 0
    any_final_paused = False
    for key in sorted(set(events) | universe):
        port_events = events.get(key, [])
        paused_at = None
        durations: List[float] = []
        pause_count = 0
        resume_count = 0
        for timestamp, event_type in sorted(port_events):
            if event_type == 1:
                pause_count += 1
                if paused_at is None:
                    paused_at = timestamp
            elif event_type == 0 and paused_at is not None:
                resume_count += 1
                durations.append(max(0.0, timestamp - paused_at))
                paused_at = None
            elif event_type == 0:
                resume_count += 1
        final_paused = paused_at is not None
        if final_paused:
            durations.append(max(0.0, stop_time_s - paused_at))
        total = sum(durations)
        per_port[key] = {
            "pfc_mapping_available": True,
            "pfc_pause_count": pause_count,
            "pfc_resume_count": resume_count,
            "pfc_pause_total_s": total,
            "pfc_max_pause_s": max(durations, default=0.0),
            "pfc_duty": total / stop_time_s if stop_time_s > 0 else None,
            "pfc_final_paused": final_paused,
        }
        total_pause_count += pause_count
        total_resume_count += resume_count
        any_final_paused = any_final_paused or final_paused
        all_durations.extend(durations)
    total_duration = sum(all_durations)
    overall = {
        "pfc_mapping_available": True,
        "pfc_pause_count": total_pause_count,
        "pfc_resume_count": total_resume_count,
        "pfc_pause_total_s": total_duration,
        "pfc_max_pause_s": max(all_durations, default=0.0),
        # The network-level duty is a mean link-duty ratio. Summed port-seconds
        # divided only by wall time can exceed one and is not a duty cycle.
        "pfc_duty": (
            mean(value["pfc_duty"] for value in per_port.values())
            if per_port and stop_time_s > 0 else None
        ),
        "pfc_final_paused": any_final_paused,
    }
    return overall, per_port


def conf_value(path: Path, key: str, default=None):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2 and fields[0] == key:
                return fields[1]
    return default


def find_single(directory: Path, suffix: str) -> Path:
    matches = sorted(Path(directory).glob(f"*{suffix}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {suffix} file under {directory}, found {len(matches)}"
        )
    return matches[0]


def read_jsonl_since(path: Path, offset: int = 0) -> List[dict]:
    records = []
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        for raw in handle:
            if raw.strip():
                records.append(json.loads(raw.decode("utf-8")))
    return records


def select_agent_epoch_record(
    path: Path,
    offset: int,
    run_id: str,
    phase: str,
    global_epoch: int,
) -> dict:
    records = [record for record in read_jsonl_since(path, offset) if not record.get("eval_greedy")]
    matching = [
        record for record in records
        if record.get("run_id") == run_id
        and record.get("phase") == phase
        and int(record.get("epoch", -1)) == int(global_epoch)
    ]
    if len(matching) != 1:
        raise ValueError(
            f"expected one agent metric record for run={run_id} phase={phase} "
            f"epoch={global_epoch}; found {len(matching)} after byte offset {offset}"
        )
    return matching[0]


def aggregate_agent_losses(record: Mapping[str, object]) -> Dict[str, Optional[float]]:
    details = [
        detail for detail in (record.get("train_port_metrics") or {}).values()
        if isinstance(detail, Mapping)
    ]

    def average_keys(*keys):
        values = []
        for detail in details:
            for key in keys:
                value = detail.get(key)
                if value is not None:
                    values.append(float(value))
                    break
        return mean(values) if values else None

    def maximum(key):
        values = [float(detail[key]) for detail in details if detail.get(key) is not None]
        return max(values) if values else None

    def weighted_epoch_mean(key):
        pairs = [
            (float(detail[key]), int(detail.get("updates", 1)))
            for detail in details if detail.get(key) is not None
        ]
        total = sum(max(weight, 0) for _, weight in pairs)
        return (
            sum(value * max(weight, 0) for value, weight in pairs) / total
            if total else None
        )

    replay = record.get("replay") or {}
    global_detail = replay.get("global") if isinstance(replay.get("global"), Mapping) else {}

    epoch_reward = weighted_epoch_mean("batch_reward_mean")
    epoch_loss = weighted_epoch_mean("loss_mean")
    return {
        # Prefer the complete per-port epoch summaries. The helper-level
        # recent deques are bounded windows and are only a compatibility
        # fallback for historical records without train_port_metrics.
        "reward": epoch_reward if epoch_reward is not None else _number(record.get("mean_reward")),
        "loss": epoch_loss if epoch_loss is not None else _number(record.get("mean_loss")),
        "td_loss": average_keys("loss_td_mean", "td_loss_mean", "loss_td"),
        "consistency_loss": average_keys(
            "loss_cons_mean", "loss_consistency_mean", "consistency_loss_mean", "loss_cons"
        ),
        "reference_loss": average_keys(
            "loss_reg_mean", "loss_reference_mean", "reference_loss_mean", "loss_reg"
        ),
        "q_prediction_abs_max": maximum("q_prediction_abs_max"),
        "q_target_abs_max": maximum("q_target_abs_max"),
        "td_error_abs_p95_max": maximum("td_error_abs_p95_max"),
        "expected_q_bound": maximum("expected_q_bound"),
        "q_inflation_limit": maximum("q_inflation_limit"),
        "q_inflated_ports": sum(bool(detail.get("q_inflated")) for detail in details),
        "local_replay_size": _number(replay.get("local_size_total")),
        "global_replay_size": _number(
            replay.get("global_size", global_detail.get("size"))
        ),
        "global_replay_clusters": _number(global_detail.get("clusters")),
        "global_replay_boundary_entries": _number(
            global_detail.get("boundary_entries")
        ),
        "global_replay_evictions": _number(global_detail.get("evictions")),
        "global_replay_drift_score_max": _number(
            global_detail.get("drift_score_max")
        ),
        "global_replay_task_sizes": (
            json.dumps(global_detail.get("task_sizes"), sort_keys=True)
            if isinstance(global_detail.get("task_sizes"), Mapping) else None
        ),
        "global_replay_task_sample_counts": (
            json.dumps(global_detail.get("task_sample_counts"), sort_keys=True)
            if isinstance(global_detail.get("task_sample_counts"), Mapping) else None
        ),
    }


def _number(value):
    return None if value is None else float(value)


def provenance(
    inputs: Mapping[str, Path],
    flow_path: Path,
    conf_path: Path,
    warmup_buckets: int,
) -> dict:
    core_path = Path(__file__).resolve()
    epoch_analyzer = core_path.with_name("analyze_aba_epoch.py")
    return {
        "analysis_version": ANALYSIS_VERSION,
        "metric_formula_version": ANALYSIS_VERSION,
        "analyzers": {
            str(core_path): sha256_file(core_path),
            str(epoch_analyzer): sha256_file(epoch_analyzer),
        },
        "percentile_method": PERCENTILE_METHOD,
        "rate_warmup_buckets": int(warmup_buckets),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flow_sha256": sha256_file(flow_path),
        "conf_sha256": sha256_file(conf_path),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
