#!/usr/bin/env python3
"""Build a local midpoint sweep around one port's dominant discrete action."""

import argparse
import json
from collections import Counter
from pathlib import Path


KMIN = (0.0, 0.0949, 0.2259, 0.4066, 0.6560, 1.0)
KMAX = (0.0, 0.25, 0.5, 1.0)
PMAX = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
GRIDS = (KMIN, KMAX, PMAX)
NAMES = ("kmin", "kmax", "pmax")


def config_value(path, key):
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == key:
            return float(fields[1])
    raise ValueError(f"{key} not found in {path}")


def physical(values, config, link_gbps):
    scale = link_gbps / 25.0
    kmin_low = config_value(config, "OPENGYM_MIN_KMIN") * scale
    kmin_high = config_value(config, "OPENGYM_MAX_KMIN") * scale
    kmax_low = config_value(config, "OPENGYM_MIN_KMAX") * scale
    kmax_high = config_value(config, "OPENGYM_MAX_KMAX") * scale
    return {
        "kmin_kb": (kmin_low + values[0] * (kmin_high - kmin_low)) / 1000.0,
        "kmax_kb": (kmax_low + values[1] * (kmax_high - kmax_low)) / 1000.0,
        "pmax": values[2],
    }


def dominant_action(metrics, port):
    details = metrics.get("watch_ports_metrics", {}).get(str(port))
    if not details:
        raise ValueError(f"port {port} is absent from baseline watch metrics")
    histogram = details.get("actions_congested") or details.get("actions_active")
    if not histogram:
        raise ValueError(f"port {port} has no active or congested action samples")
    action, count = Counter(histogram).most_common(1)[0]
    try:
        indices = tuple(int(value) for value in action.split(","))
    except ValueError as exc:
        raise ValueError(f"baseline action is not a discrete triple: {action}") from exc
    if len(indices) != 3:
        raise ValueError(f"baseline action is not a triple: {action}")
    for index, grid in zip(indices, GRIDS):
        if not 0 <= index < len(grid):
            raise ValueError(f"baseline action index is out of range: {action}")
    return indices, count, sum(histogram.values())


def build_points(indices, include_grid_neighbors):
    center = tuple(grid[index] for grid, index in zip(GRIDS, indices))
    points = [{"label": "center", "kind": "grid", "values": center}]
    seen = {center}
    for head, (name, grid, index) in enumerate(zip(NAMES, GRIDS, indices)):
        for direction, neighbor in (("low", index - 1), ("high", index + 1)):
            if not 0 <= neighbor < len(grid):
                continue
            midpoint = list(center)
            midpoint[head] = (grid[index] + grid[neighbor]) / 2.0
            midpoint = tuple(midpoint)
            if midpoint not in seen:
                points.append({
                    "label": f"{name}_{direction}_mid",
                    "kind": "midpoint",
                    "dimension": name,
                    "values": midpoint,
                })
                seen.add(midpoint)
            if include_grid_neighbors:
                endpoint = list(center)
                endpoint[head] = grid[neighbor]
                endpoint = tuple(endpoint)
                if endpoint not in seen:
                    points.append({
                        "label": f"{name}_{direction}_grid",
                        "kind": "grid_neighbor",
                        "dimension": name,
                        "values": endpoint,
                    })
                    seen.add(endpoint)
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=323)
    parser.add_argument("--link-gbps", type=float, default=40.0)
    parser.add_argument("--include-grid-neighbors", action="store_true")
    args = parser.parse_args()

    metrics = json.loads(args.baseline_metrics.read_text(encoding="utf-8"))
    indices, dominant_count, total_count = dominant_action(metrics, args.port)
    points = build_points(indices, args.include_grid_neighbors)
    for point in points:
        values = point.pop("values")
        point["kmin_norm"], point["kmax_norm"], point["pmax"] = values
        point["physical"] = physical(values, args.config, args.link_gbps)

    result = {
        "target_port": args.port,
        "link_gbps": args.link_gbps,
        "baseline_metrics": str(args.baseline_metrics.resolve()),
        "config": str(args.config.resolve()),
        "dominant_action_indices": list(indices),
        "dominant_action_count": dominant_count,
        "action_sample_count": total_count,
        "dominant_action_fraction": dominant_count / total_count,
        "include_grid_neighbors": args.include_grid_neighbors,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {len(points)} local sweep points around action {indices}: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
