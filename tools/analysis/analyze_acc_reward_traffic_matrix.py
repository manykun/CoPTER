#!/usr/bin/env python3
"""Analyze matched ACC reward arms within each canonical traffic task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.metrics_core import (  # noqa: E402
    ANALYSIS_VERSION,
    conf_value,
    parse_fct,
    parse_pfc,
    parse_queue,
    parse_rate,
    parse_throughput,
    summarize_fct,
)

TRAIN_FIELDS = (
    "reward", "loss", "mean_fct_us", "p95_fct_us", "p99_fct_us",
    "completion_ratio", "mean_throughput_mbps", "mean_queue_kb",
    "mean_ecn_rate", "pfc_duty", "replay_congested_sample_ratio",
)
PHYSICAL_FIELDS = (
    "mean_fct_us", "p95_fct_us", "p99_fct_us", "completion_ratio",
    "mean_throughput_mbps", "mean_queue_kb", "mean_ecn_rate", "pfc_duty",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def fmt(value, digits=2, suffix=""):
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def relative(before, after):
    if before in (None, 0) or after is None:
        return None
    return 100.0 * (after / before - 1.0)


def window(rows: list[dict], field: str, first: bool):
    selected = rows[:10] if first else rows[-10:]
    values = [number(row.get(field)) for row in selected]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def training_rows(cell: dict, expected: int) -> tuple[dict, list[dict]]:
    run_dir = ROOT / "experiments" / "aba" / cell["run_id"]
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    if protocol["run_id"] != cell["run_id"] or protocol["seed"] != cell["config"]["seed"]:
        raise ValueError(f"protocol identity mismatch: {run_dir}")
    if protocol["action_space"] != "multiscale" or protocol.get("acc_shared_replay"):
        raise ValueError(f"unexpected action/replay setting: {run_dir}")
    expected_config = cell["config"]
    for actual_key, config_key in (
        ("reward_profile", "reward_profile"),
        ("reward_queue_lambda", "reward_queue_lambda"),
        ("reward_ecn_lambda", "reward_ecn_lambda"),
        ("reward_window_size", "reward_window_size"),
        ("replay_sampling", "replay_sampling"),
        ("replay_congested_fraction", "replay_congested_fraction"),
    ):
        if protocol.get(actual_key) != expected_config.get(config_key):
            raise ValueError(f"protocol treatment/control mismatch ({actual_key}): {run_dir}")
    config_path = run_dir / "pilot_config.yaml"
    if protocol.get("config_sha256") != sha256(config_path):
        raise ValueError(f"pilot configuration changed after protocol registration: {config_path}")
    if protocol["epochs"] != {"a1": expected, "b": 0, "a2": 0}:
        raise ValueError(f"incomplete or unexpected epoch budget: {run_dir}")
    path = run_dir / "summaries" / "acc_epoch_metrics.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["global_epoch"]))
    if [int(row["global_epoch"]) for row in rows] != list(range(1, expected + 1)):
        raise ValueError(f"missing or duplicate completed epochs: {path}")
    if any(row.get("task") != protocol["tasks"]["a"] for row in rows):
        raise ValueError(f"training task mismatch: {path}")
    return protocol, rows


def validate_task_arms(arms: dict) -> None:
    """Within a traffic task, reward definition is the only treatment."""
    protocols = list(arms.values())
    base = protocols[0]
    allowed = {"run_id", "config", "config_sha256", "created_at",
               "reward_profile", "reward_queue_lambda"}
    for current in protocols[1:]:
        different = [key for key in sorted(set(base) | set(current))
                     if key not in allowed and base.get(key) != current.get(key)]
        if different:
            raise ValueError(f"reward arms differ outside treatment: {different}")


def evaluation_paths(cell: dict, protocol: dict) -> dict[str, list[Path]] | None:
    run_dir = ROOT / "experiments" / "aba" / cell["run_id"]
    diagnostic = run_dir.with_name(run_dir.name + "_diagnostics")
    manifest = diagnostic / "protocol.json"
    if not manifest.exists():
        return None
    details = json.loads(manifest.read_text(encoding="utf-8"))
    if details.get("source_protocol_sha256") != sha256(run_dir / "protocol.json"):
        raise ValueError(f"frozen diagnostic source mismatch: {diagnostic}")
    repeats = int(details["repeats"])
    if repeats < 2:
        raise ValueError(f"frozen diagnostic needs >=2 repeatability checks: {diagnostic}")
    result = {}
    for checkpoint in ("frozen_initial", "frozen_final"):
        paths = [diagnostic / "eval" / checkpoint / f"repeat_{index}"
                 for index in range(1, repeats + 1)]
        required = ("complete.json", "input.flow", "input.conf", "result.fct",
                    "result.queue", "result.rate", "result.throughput", "result.pfc")
        for path in paths:
            missing = [name for name in required if not (path / name).is_file()]
            if missing:
                raise ValueError(f"incomplete frozen evaluation {path}: {missing}")
        result[checkpoint] = paths
    return result


def frozen_metrics(path: Path, common_keys: set) -> dict:
    fct = summarize_fct(path / "result.fct", path / "input.flow", common_keys)
    queue, _ = parse_queue(path / "result.queue")
    rate, rate_ports = parse_rate(path / "result.rate", warmup_buckets=2)
    throughput, _ = parse_throughput(path / "result.throughput")
    stop = float(conf_value(path / "input.conf", "SIMULATOR_STOP_TIME", 0.0))
    pfc, _ = parse_pfc(path / "result.pfc", stop, port_universe=rate_ports)
    return {
        **{name: fct[name] for name in ("mean_fct_us", "p95_fct_us", "p99_fct_us", "completion_ratio")},
        "matched_flows": fct["matched_flows"],
        "mean_throughput_mbps": throughput.get("mean"),
        "mean_queue_kb": queue.get("mean"),
        "mean_ecn_rate": rate.get("ecn_rate_mean"),
        "pfc_duty": pfc.get("pfc_duty"),
    }


def frozen_table(plan: dict, sources: dict, require_frozen: bool):
    records = []
    intersections = {}
    hashes = {}
    for task in plan["tasks"]:
        task_cells = [cell for cell in plan["cells"] if cell["task"] == task]
        paths_by_cell = {}
        for cell in task_cells:
            protocol, _ = sources[cell["run_id"]]
            paths = evaluation_paths(cell, protocol)
            if paths is None:
                if require_frozen:
                    raise ValueError(f"missing frozen diagnostic: {cell['run_id']}")
                paths_by_cell = {}
                break
            paths_by_cell[cell["run_id"]] = paths
        if not paths_by_cell:
            continue
        all_paths = [path for paths in paths_by_cell.values()
                     for group in paths.values() for path in group]
        common = set.intersection(*(set(parse_fct(path / "result.fct")) for path in all_paths))
        if not common:
            raise ValueError(f"no common completed flows for {task}")
        intersections[task] = len(common)
        for cell in task_cells:
            for checkpoint, paths in paths_by_cell[cell["run_id"]].items():
                for repeat, path in enumerate(paths, 1):
                    records.append({"task": task, "reward": cell["reward"],
                                    "checkpoint": checkpoint, "repeat": repeat,
                                    **frozen_metrics(path, common)})
                    for name in ("input.flow", "input.conf", "result.fct", "result.queue",
                                 "result.rate", "result.throughput", "result.pfc"):
                        source = path / name
                        hashes[str(source.relative_to(ROOT))] = sha256(source)
    return records, intersections, hashes


def write_csv(path: Path, records: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def make_plots(directory: Path, plan: dict, sources: dict) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Plot generation unavailable: {exc}"]
    directory.mkdir(exist_ok=True)
    for task in plan["tasks"]:
        cells = [cell for cell in plan["cells"] if cell["task"] == task]
        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for cell in cells:
            rows = sources[cell["run_id"]][1]
            x = [int(row["global_epoch"]) for row in rows]
            for axis, field in zip(axes, ("mean_fct_us", "p95_fct_us", "p99_fct_us")):
                axis.plot(x, [number(row.get(field)) for row in rows],
                          label=cell["reward"], linewidth=1.4)
                axis.set_ylabel(field.replace("_", " "))
                axis.grid(alpha=.25)
        axes[0].legend()
        axes[-1].set_xlabel("Training epoch (not frozen evaluation)")
        fig.tight_layout()
        fig.savefig(directory / f"{task}_training_fct.png", dpi=170)
        plt.close(fig)

        fig, axes = plt.subplots(len(cells), 2, figsize=(12, 3 * len(cells)), sharex=True)
        if len(cells) == 1:
            axes = [axes]
        for pair, cell in zip(axes, cells):
            rows = sources[cell["run_id"]][1]
            x = [int(row["global_epoch"]) for row in rows]
            pair[0].plot(x, [number(row.get("reward")) for row in rows])
            pair[0].set_ylabel(f"{cell['reward']} native reward")
            pair[1].plot(x, [number(row.get("loss")) for row in rows])
            pair[1].set_ylabel("TD loss")
            pair[1].set_yscale("log")
            for axis in pair:
                axis.grid(alpha=.25)
        axes[-1][-1].set_xlabel("Training epoch")
        fig.tight_layout()
        fig.savefig(directory / f"{task}_reward_loss.png", dpi=170)
        plt.close(fig)
    return []


def report(plan: dict, sources: dict, frozen: list[dict], intersections: dict,
           notes: list[str]) -> str:
    lines = ["# ACC reward × traffic matrix", "",
             "- All cells: 50-epoch single-task ACC, multiscale actions, 400-KB buffer, local-only replay, 4-step reward attribution, 50% congestion-stratified target, common seed and epsilon/target schedule.",
             "- Reward arms: `tail5` = tail-safe queue/ECN 5/5; `tail1` = tail-safe 1/5; `weighted` = original 0.5/0.3/0.2 reward.",
             "- Native reward scales differ: compare each arm with itself, never compare their absolute reward values.",
             "- Training FCT uses each epoch's valid completed flows; frozen FCT uses a task-local common completed-flow intersection across all reward arms, checkpoints and repeats.",
             "- Same-seed frozen repeats test determinism, not independent-seed significance; this is acquisition, not catastrophic-forgetting evidence.",
             "", "## Training: first 10 → last 10 epochs", "",
             "| Task | Reward | Reward Δ | TD loss Δ | Mean FCT Δ | p95 FCT Δ | p99 FCT Δ | Completion Δ pp | Throughput Δ | Queue Δ | ECN Δ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in plan["cells"]:
        rows = sources[cell["run_id"]][1]
        changes = {field: relative(window(rows, field, True), window(rows, field, False))
                   for field in TRAIN_FIELDS}
        before = window(rows, "completion_ratio", True)
        after = window(rows, "completion_ratio", False)
        completion_pp = None if before is None or after is None else 100 * (after - before)
        lines.append(
            f"| {cell['task']} | {cell['reward']} | {fmt(changes['reward'],2,'%')} | "
            f"{fmt(changes['loss'],2,'%')} | {fmt(changes['mean_fct_us'],2,'%')} | "
            f"{fmt(changes['p95_fct_us'],2,'%')} | {fmt(changes['p99_fct_us'],2,'%')} | "
            f"{fmt(completion_pp,4)} | {fmt(changes['mean_throughput_mbps'],2,'%')} | "
            f"{fmt(changes['mean_queue_kb'],2,'%')} | {fmt(changes['mean_ecn_rate'],2,'%')} |"
        )
    if frozen:
        grouped = defaultdict(list)
        for row in frozen:
            grouped[(row["task"], row["reward"], row["checkpoint"])].append(row)
        average = {}
        for key, rows in grouped.items():
            average[key] = {field: mean(values) if (values := [number(row.get(field)) for row in rows if number(row.get(field)) is not None]) else None
                            for field in PHYSICAL_FIELDS}
        lines += ["", "## Frozen policy: final versus initialization", "",
                  "| Task | Reward | Common flows | Mean FCT Δ | p95 FCT Δ | p99 FCT Δ | Completion Δ pp | Throughput Δ | Queue Δ | ECN Δ | PFC duty Δ |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for task in plan["tasks"]:
            for reward in plan["rewards"]:
                before = average[(task, reward, "frozen_initial")]
                after = average[(task, reward, "frozen_final")]
                completion_pp = (None if before["completion_ratio"] is None or after["completion_ratio"] is None
                                 else 100 * (after["completion_ratio"] - before["completion_ratio"]))
                lines.append(
                    f"| {task} | {reward} | {intersections[task]} | "
                    f"{fmt(relative(before['mean_fct_us'],after['mean_fct_us']),2,'%')} | "
                    f"{fmt(relative(before['p95_fct_us'],after['p95_fct_us']),2,'%')} | "
                    f"{fmt(relative(before['p99_fct_us'],after['p99_fct_us']),2,'%')} | "
                    f"{fmt(completion_pp,4)} | "
                    f"{fmt(relative(before['mean_throughput_mbps'],after['mean_throughput_mbps']),2,'%')} | "
                    f"{fmt(relative(before['mean_queue_kb'],after['mean_queue_kb']),2,'%')} | "
                    f"{fmt(relative(before['mean_ecn_rate'],after['mean_ecn_rate']),2,'%')} | "
                    f"{fmt(relative(before['pfc_duty'],after['pfc_duty']),2,'%')} |"
                )
        lines += ["", "Compare final FCT and completion against `tail5` within the same task using `frozen_metrics.csv`; do not select a winner from TD loss alone."]
    else:
        lines += ["", "Frozen evaluations are not complete; training trajectories alone cannot establish a causal FCT improvement."]
    lines += ["", "## Figures", "", "For each task, `figures/TASK_training_fct.png` and `figures/TASK_reward_loss.png` show all reward arms."]
    lines += ["", *notes]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = plan_path.parent
    sources = {cell["run_id"]: training_rows(cell, plan["epochs"])
               for cell in plan["cells"]}
    for task in plan["tasks"]:
        validate_task_arms({cell["reward"]: sources[cell["run_id"]][0]
                            for cell in plan["cells"] if cell["task"] == task})
    frozen, intersections, eval_hashes = frozen_table(plan, sources, args.require_frozen)
    if frozen and len(intersections) != len(plan["tasks"]):
        raise ValueError("partial frozen matrix; refusing a mixed report")
    notes = make_plots(output / "figures", plan, sources)
    training = [{"task": cell["task"], "reward_arm": cell["reward"], **row}
                for cell in plan["cells"] for row in sources[cell["run_id"]][1]]
    write_csv(output / "training_metrics.csv", training,
              ["task", "reward_arm", *sources[plan["cells"][0]["run_id"]][1][0]])
    if frozen:
        write_csv(output / "frozen_metrics.csv", frozen,
                  ["task", "reward", "checkpoint", "repeat", "matched_flows", *PHYSICAL_FIELDS])
    (output / "REPORT.md").write_text(report(plan, sources, frozen, intersections, notes), encoding="utf-8")
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_version": ANALYSIS_VERSION,
        "plan_sha256": sha256(plan_path),
        "protocol_sha256": {cell["run_id"]: sha256(ROOT / "experiments" / "aba" / cell["run_id"] / "protocol.json")
                            for cell in plan["cells"]},
        "training_metrics_sha256": {
            cell["run_id"]: sha256(ROOT / "experiments" / "aba" / cell["run_id"] /
                                   "summaries" / "acc_epoch_metrics.csv")
            for cell in plan["cells"]
        },
        "task_common_completed_flows": intersections,
        "frozen_input_sha256": eval_hashes,
        "percentile_method": "linear position (n - 1) × q",
        "rate_warmup_buckets": 2,
    }
    (output / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"ACC reward/traffic matrix report written to {output}")


if __name__ == "__main__":
    main()
