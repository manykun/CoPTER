#!/usr/bin/env python3
"""Create an after-B checkpoint directory with selected ports restored to after-A."""

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import torch


PORT_PATTERN = re.compile(r"_ACC_(\d+)$")


def parse_ports(value):
    if not value:
        return []
    ports = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(ports) != len(set(ports)):
        raise ValueError("duplicate port in list")
    return ports


def checkpoint_map(directory):
    mapping = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = PORT_PATTERN.search(path.name)
        if match:
            port = int(match.group(1))
            if port in mapping:
                raise ValueError(f"multiple checkpoints for port {port} in {directory}")
            mapping[port] = path
    if not mapping:
        raise ValueError(f"no *_ACC_<port> checkpoints under {directory}")
    return mapping


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Restore selected after-A port policies into an after-B model copy."
    )
    parser.add_argument("--after-a", required=True, type=Path)
    parser.add_argument("--after-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ports", help="Comma-separated explicit port indices.")
    group.add_argument("--random-count", type=int, help="Number of random control ports.")
    parser.add_argument("--exclude-ports", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for directory in (args.after_a, args.after_b):
        if not directory.is_dir():
            raise SystemExit(f"checkpoint directory not found: {directory}")
    if args.output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"output already exists: {args.output_dir}; pass --force to replace"
            )
        shutil.rmtree(args.output_dir)

    after_a = checkpoint_map(args.after_a)
    after_b = checkpoint_map(args.after_b)
    common = sorted(set(after_a) & set(after_b))
    if set(after_a) != set(after_b):
        raise SystemExit("after-A and after-B checkpoint port sets differ")

    try:
        excluded = set(parse_ports(args.exclude_ports))
        if args.ports:
            selected = parse_ports(args.ports)
        else:
            if args.random_count is None or args.random_count <= 0:
                raise ValueError("--random-count must be positive")
            candidates = [port for port in common if port not in excluded]
            if args.random_count > len(candidates):
                raise ValueError("random count exceeds eligible port count")
            selected = sorted(random.Random(args.seed).sample(
                candidates, args.random_count
            ))
    except ValueError as exc:
        raise SystemExit(str(exc))

    missing = sorted(set(selected) - set(common))
    if missing:
        raise SystemExit(f"selected ports have no checkpoint: {missing}")

    shutil.copytree(args.after_b, args.output_dir)
    output_map = checkpoint_map(args.output_dir)
    restored = []
    for port in selected:
        source = torch_load(after_a[port])
        target = torch_load(output_map[port])
        for key in ("policy", "target"):
            if key not in source or key not in target:
                raise SystemExit(
                    f"checkpoint for port {port} lacks required key {key!r}"
                )
            target[key] = source[key]
        temporary = output_map[port].with_name(output_map[port].name + ".tmp")
        torch.save(target, temporary)
        temporary.replace(output_map[port])
        restored.append({
            "port": port,
            "source": str(after_a[port]),
            "target": str(output_map[port]),
            "replaced_keys": ["policy", "target"],
        })

    manifest = {
        "after_a": str(args.after_a.resolve()),
        "after_b": str(args.after_b.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "seed": args.seed,
        "excluded_ports": sorted(excluded),
        "selected_ports": selected,
        "restored": restored,
        "preserved": "all non-policy checkpoint keys and all unselected ports",
    }
    (args.output_dir / "restore_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Restored {len(selected)} port policies into {args.output_dir}: "
        + ",".join(map(str, selected))
    )


if __name__ == "__main__":
    main()
