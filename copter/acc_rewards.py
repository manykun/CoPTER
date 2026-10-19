"""ACC reward formulas with explicit byte units and action-interval telemetry."""

from bisect import bisect_left
import json
import math

INTERVAL_PROFILES = ("acc_paper", "acc_3816bc6_aligned")


def calculate_acc_reward(profile, queue_bytes, utilization):
    if profile not in INTERVAL_PROFILES:
        raise ValueError(f"unsupported ACC reward: {profile}")
    if not all(math.isfinite(x) and x >= 0 for x in (queue_bytes, utilization)):
        raise ValueError("non-finite or negative interval telemetry")
    if profile == "acc_paper":
        thresholds = [20 * 1024 * 2**i for i in range(10)]
        index = min(bisect_left(thresholds, queue_bytes), 9)
        queue_score = 1.0 - index / 10.0
        throughput_weight = .7
    else:
        # 3816bc6 normalizes actual bytes by 400 KB, then bisects 2**i/1024.
        normalized = min(1.0, queue_bytes / (400 * 1024))
        index = bisect_left([2**i / 1024 for i in range(11)], normalized)
        queue_score = 1.0 - index / 10.0
        throughput_weight = .5
    # Keep the historic utilization unmodified; upstream >1 is a diagnostic,
    # not permission to silently change the historic reward formula.
    reward = throughput_weight * utilization + (1 - throughput_weight) * queue_score
    return {
        "reward": reward, "queue": queue_score, "throughput": utilization,
        "reward_queue_bin": index,
        "reward_throughput_term": throughput_weight * utilization,
        "reward_queue_term": (1 - throughput_weight) * queue_score,
    }


def parse_reward_interval(info, n_ports, previous_end=None):
    if isinstance(info, str):
        info = json.loads(info)
    interval = info.get("reward_interval") if isinstance(info, dict) else None
    if not isinstance(interval, dict) or interval.get("version") != 1:
        raise RuntimeError("missing reward_interval v1; rebuild the ns-3 simulator")
    start, end = int(interval["start_ns"]), int(interval["end_ns"])
    if start < 0 or end <= start or (previous_end is not None and start != previous_end):
        raise RuntimeError("reward intervals must be positive, contiguous and non-overlapping")
    for field in ("mean_queue_bytes", "tx_utilization"):
        values = interval[field]
        if len(values) != n_ports or not all(math.isfinite(v) and v >= 0 for v in values):
            raise RuntimeError(f"invalid reward interval {field}")
    return interval
