"""Per-port rollout measurements shared by the ACC and SOR drivers."""

import json
from collections import Counter
from pathlib import Path


COMPONENT_KEYS = (
    "reward",
    "throughput",
    "queue",
    "ecn",
    "avg_tx_rate",
    "avg_queue",
    "peak_queue",
    "avg_ecn",
    "peak_ecn",
    "combined_queue",
    "combined_ecn",
    "queue_cost_sq",
    "ecn_cost_sq",
    "tail_safe_raw",
    "tail_safe_clipped",
)


def parse_watch_ports(value):
    """Parse a comma-separated port list while rejecting duplicates."""
    if not value:
        return []
    try:
        ports = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("watch ports must be comma-separated integers") from exc
    if len(ports) != len(set(ports)):
        raise ValueError("watch ports must not contain duplicates")
    if any(port < 0 for port in ports):
        raise ValueError("watch ports must be non-negative")
    return ports


def parse_forced_port_action(value):
    """Parse PORT,KMIN_NORM,KMAX_NORM,PMAX for frozen local sweeps."""
    if not value:
        return None
    fields = [item.strip() for item in value.split(",")]
    if len(fields) != 4:
        raise ValueError(
            "forced port action must be PORT,KMIN_NORM,KMAX_NORM,PMAX"
        )
    try:
        port = int(fields[0])
        values = tuple(float(item) for item in fields[1:])
    except ValueError as exc:
        raise ValueError("forced port action contains a non-numeric value") from exc
    if port < 0:
        raise ValueError("forced port must be non-negative")
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("normalized Kmin, Kmax, and Pmax must be in [0, 1]")
    return (port, *values)


class PortMetricTracker:
    """Collect all-step, active-step, and congested-step metrics for fixed ports."""

    def __init__(self, watch_ports, trace_path=""):
        self.watch_ports = list(watch_ports)
        self.trace_path = Path(trace_path).expanduser() if trace_path else None
        self._stats = {port: self._empty_stats() for port in self.watch_ports}
        self._trace = []

    @staticmethod
    def _empty_stats():
        return {
            "samples": 0,
            "active_steps": 0,
            "congested_steps": 0,
            "sums_all": Counter(),
            "sums_active": Counter(),
            "sums_congested": Counter(),
            "actions_all": Counter(),
            "actions_active": Counter(),
            "actions_congested": Counter(),
            "reward_ema_congested": None,
            "identifier": None,
        }

    def validate(self, n_ports):
        invalid = [port for port in self.watch_ports if port >= n_ports]
        if invalid:
            raise ValueError(
                f"watch ports outside [0, {n_ports - 1}]: "
                + ",".join(map(str, invalid))
            )

    def observe(self, step, actions, network_helper):
        if not self.watch_ports:
            return
        identifiers = getattr(network_helper, "port_identifier_map", {}) or {}
        for port in self.watch_ports:
            components = {
                key: float(value)
                for key, value in network_helper
                .get_port_current_reward_components(port).items()
                if key in COMPONENT_KEYS
            }
            missing = [key for key in COMPONENT_KEYS if key not in components]
            if missing:
                raise ValueError(
                    f"port {port} reward components missing: {','.join(missing)}"
                )
            active = bool(network_helper.is_port_active(port))
            congested = bool(network_helper.is_port_congested(port))
            action = actions[port] if actions is not None else None
            action_key = (
                ",".join(str(index) for index in action)
                if action is not None else "uncontrolled"
            )

            stats = self._stats[port]
            stats["samples"] += 1
            stats["actions_all"][action_key] += 1
            for key, value in components.items():
                stats["sums_all"][key] += value
            if active:
                stats["active_steps"] += 1
                stats["actions_active"][action_key] += 1
                for key, value in components.items():
                    stats["sums_active"][key] += value
            if congested:
                stats["congested_steps"] += 1
                stats["actions_congested"][action_key] += 1
                for key, value in components.items():
                    stats["sums_congested"][key] += value
                previous = stats["reward_ema_congested"]
                stats["reward_ema_congested"] = (
                    components["reward"]
                    if previous is None
                    else 0.9 * previous + 0.1 * components["reward"]
                )

            identifier = identifiers.get(port, identifiers.get(str(port)))
            if identifier is not None:
                stats["identifier"] = str(identifier)
            if self.trace_path is not None:
                self._trace.append({
                    "step": int(step),
                    "port": port,
                    "identifier": stats["identifier"],
                    "active": active,
                    "congested": congested,
                    "action": action_key,
                    **components,
                })

    @staticmethod
    def _means(sums, count):
        return {
            key: (sums[key] / count if count else None)
            for key in COMPONENT_KEYS
        }

    def summary(self):
        output = {}
        for port, stats in self._stats.items():
            output[str(port)] = {
                "identifier": stats["identifier"],
                "samples": stats["samples"],
                "active_steps": stats["active_steps"],
                "congested_steps": stats["congested_steps"],
                "active_ratio": (
                    stats["active_steps"] / stats["samples"]
                    if stats["samples"] else None
                ),
                "congested_ratio": (
                    stats["congested_steps"] / stats["samples"]
                    if stats["samples"] else None
                ),
                "means_all": self._means(
                    stats["sums_all"], stats["samples"]
                ),
                "means_active": self._means(
                    stats["sums_active"], stats["active_steps"]
                ),
                "means_congested": self._means(
                    stats["sums_congested"], stats["congested_steps"]
                ),
                "reward_ema_congested": stats["reward_ema_congested"],
                "actions_all": dict(stats["actions_all"]),
                "actions_active": dict(stats["actions_active"]),
                "actions_congested": dict(stats["actions_congested"]),
            }
        return output

    def legacy_rewards(self):
        return {
            str(port): self._stats[port]["reward_ema_congested"]
            for port in self.watch_ports
        }

    def write_trace(self):
        if self.trace_path is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.trace_path.with_suffix(self.trace_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in self._trace:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        temporary.replace(self.trace_path)
