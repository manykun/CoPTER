#!/usr/bin/env python3
"""Prepare a causal one-port interpolation path between task optima."""

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "copter"))
from structures import acc_action_from_indices  # noqa: E402


ACTION_INDICES = {
    "low_gentle": (1, 1),
    "low_mid": (1, 3),
    "low_strong": (1, 5),
    "mid_gentle": (4, 1),
    "mid": (4, 3),
    "mid_strong": (4, 5),
    "high_gentle": (8, 1),
    "high_mid": (8, 3),
    "high_strong": (8, 5),
}


def parse_ports(value):
    try:
        ports = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not ports or len(ports) != len(set(ports)) or any(port < 0 for port in ports):
        raise argparse.ArgumentTypeError(
            "ports must be unique comma-separated non-negative integers"
        )
    return ports


def best_action(summary_path, scenario, port):
    candidates = []
    with summary_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scenario"] != scenario or int(row["port"]) != port:
                continue
            raw = row.get("reward_congested", "")
            if raw not in (None, ""):
                candidates.append((float(raw), row["action"]))
    if not candidates:
        raise ValueError(
            f"no congested-port reward for port {port}, scenario {scenario}"
        )
    return max(candidates)[1]


def normalized_action(label):
    if label not in ACTION_INDICES:
        raise ValueError(f"unsupported calibration action label: {label}")
    value = acc_action_from_indices(ACTION_INDICES[label], "multiscale")
    return (value.k_min_norm, value.k_max_norm, value.p_max)


def alpha_values(step, minimum=0.0, maximum=1.0):
    if not 0 < step <= 1:
        raise ValueError("alpha step must be in (0, 1]")
    if not 0 <= minimum < maximum <= 1:
        raise ValueError("require 0 <= alpha min < alpha max <= 1")
    values = []
    current = minimum
    while current < maximum - 1e-12:
        values.append(round(current, 10))
        current += step
    values.append(round(maximum, 10))
    return values


def interpolate(left, right, alpha):
    return tuple(
        (1.0 - alpha) * start + alpha * end
        for start, end in zip(left, right)
    )


def physical(values, config, link_gbps):
    raw = {}
    for line in config.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in {
            "OPENGYM_MIN_KMIN", "OPENGYM_MAX_KMIN",
            "OPENGYM_MIN_KMAX", "OPENGYM_MAX_KMAX",
        }:
            raw[fields[0]] = float(fields[1])
    missing = {
        "OPENGYM_MIN_KMIN", "OPENGYM_MAX_KMIN",
        "OPENGYM_MIN_KMAX", "OPENGYM_MAX_KMAX",
    } - set(raw)
    if missing:
        raise ValueError(f"config is missing ranges: {sorted(missing)}")
    scale = link_gbps / 25.0
    kmin = raw["OPENGYM_MIN_KMIN"] + values[0] * (
        raw["OPENGYM_MAX_KMIN"] - raw["OPENGYM_MIN_KMIN"]
    )
    kmax = raw["OPENGYM_MIN_KMAX"] + values[1] * (
        raw["OPENGYM_MAX_KMAX"] - raw["OPENGYM_MIN_KMAX"]
    )
    return {
        "kmin_kb": kmin * scale / 1000.0,
        "kmax_kb": kmax * scale / 1000.0,
        "pmax": values[2],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--base-run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint-port", type=int, default=323)
    parser.add_argument("--target-ports", type=parse_ports, default=[323, 321])
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=1.0)
    parser.add_argument("--link-gbps", type=float, default=40.0)
    args = parser.parse_args()

    base_manifest = json.loads(
        (args.base_run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    task_a, task_b = base_manifest["task_a"], base_manifest["task_b"]
    summary_path = args.calibration_dir / "port_action_summary.csv"
    action_a = best_action(summary_path, task_a, args.endpoint_port)
    action_b = best_action(summary_path, task_b, args.endpoint_port)
    endpoint_a = normalized_action(action_a)
    endpoint_b = normalized_action(action_b)
    config = args.base_run_dir / "runtime" / f"{task_a}_seed{base_manifest['seed']}.conf"
    points = []
    for alpha in alpha_values(args.alpha_step, args.alpha_min, args.alpha_max):
        values = interpolate(endpoint_a, endpoint_b, alpha)
        points.append({
            "label": f"alpha_{int(round(alpha * 100)):03d}",
            "alpha": alpha,
            "kmin_norm": values[0],
            "kmax_norm": values[1],
            "pmax": values[2],
            "physical": physical(values, config, args.link_gbps),
        })
    result = {
        "base_run_id": base_manifest["run_id"],
        "calibration_dir": str(args.calibration_dir.resolve()),
        "task_a": task_a,
        "task_b": task_b,
        "seed": base_manifest["seed"],
        "buffer_kb": base_manifest["buffer_kb"],
        "endpoint_port": args.endpoint_port,
        "target_ports": args.target_ports,
        "link_gbps": args.link_gbps,
        "alpha_step": args.alpha_step,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "endpoint_a": {"action": action_a, "normalized": endpoint_a},
        "endpoint_b": {"action": action_b, "normalized": endpoint_b},
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Prepared {len(points)} points from {action_a} to {action_b} "
        f"for ports {args.target_ports}: {args.output}"
    )


if __name__ == "__main__":
    main()
