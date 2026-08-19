#!/usr/bin/env python3
"""Prepare physical ECN range points for a frozen one-port intervention."""

import argparse
import json
from pathlib import Path


WIDE_POINTS = (
    ("center", "reference", 63, 120, 0.50),
    ("ref_kmin_57", "prior_best", 57, 120, 0.50),
    ("ref_kmax_110", "prior_best", 63, 110, 0.50),
    ("ref_pmax_055", "prior_best", 63, 120, 0.55),
    ("kmin_24", "range", 24, 120, 0.50),
    ("kmin_32", "range", 32, 120, 0.50),
    ("kmin_96", "range", 96, 120, 0.50),
    ("kmax_80", "range", 63, 80, 0.50),
    ("kmax_240", "range", 63, 240, 0.50),
    ("kmax_320", "range", 63, 320, 0.50),
    ("pmax_001", "range", 63, 120, 0.01),
    ("pmax_005", "range", 63, 120, 0.05),
    ("pmax_010", "range", 63, 120, 0.10),
    ("pmax_100", "range", 63, 120, 1.00),
)


def parse_action(value):
    fields = value.split(",")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("expected KMIN_KB,KMAX_KB,PMAX")
    try:
        kmin, kmax = int(fields[0]), int(fields[1])
        pmax = float(fields[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if kmin < 0 or kmax <= kmin or not 0.0 <= pmax <= 1.0:
        raise argparse.ArgumentTypeError(
            "require 0 <= Kmin < Kmax and 0 <= Pmax <= 1"
        )
    return kmin, kmax, pmax


def point(label, kind, values):
    kmin, kmax, pmax = values
    return {
        "label": label,
        "kind": kind,
        "kmin_kb": kmin,
        "kmax_kb": kmax,
        "pmax": pmax,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-port", required=True, type=int)
    parser.add_argument("--point-set", choices=("wide", "control"), default="wide")
    parser.add_argument(
        "--control-action",
        type=parse_action,
        help="For point-set=control: KMIN_KB,KMAX_KB,PMAX",
    )
    parser.add_argument("--base-run-id", required=True)
    parser.add_argument("--task-a", required=True)
    parser.add_argument("--task-b", required=True)
    parser.add_argument("--buffer-kb", type=int, required=True)
    args = parser.parse_args()

    if args.target_port < 0:
        parser.error("--target-port must be non-negative")
    if args.point_set == "control" and args.control_action is None:
        parser.error("--control-action is required for point-set=control")

    if args.point_set == "wide":
        points = [point(label, kind, (kmin, kmax, pmax))
                  for label, kind, kmin, kmax, pmax in WIDE_POINTS]
    else:
        points = [
            point("center", "reference", (63, 120, 0.50)),
            point("candidate", "control_candidate", args.control_action),
        ]

    invalid = [entry for entry in points if entry["kmax_kb"] > args.buffer_kb]
    if invalid:
        labels = ", ".join(entry["label"] for entry in invalid)
        parser.error(
            f"Kmax exceeds {args.buffer_kb} KB buffer for point(s): {labels}"
        )

    result = {
        "base_run_id": args.base_run_id,
        "target_port": args.target_port,
        "point_set": args.point_set,
        "task_a": args.task_a,
        "task_b": args.task_b,
        "buffer_kb": args.buffer_kb,
        "points": points,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(points)} physical points: {args.output}")


if __name__ == "__main__":
    main()
