#!/usr/bin/env python3
"""Analyse ACC curriculum results and diagnose catastrophic forgetting.

Usage:
  python3 scripts/analyze_forgetting.py [metrics_jsonl]

If no path is given, defaults to copter/cur_models_acc-0624/cur_acc_metrics.jsonl.
"""
import json
import sys
from collections import defaultdict, OrderedDict

DEFAULT_PATH = "copter/cur_models_acc-0624/cur_acc_metrics.jsonl"


def load(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def summarize(data):
    groups = OrderedDict()
    for d in data:
        tag = d.get("eval_tag", "unknown")
        groups.setdefault(tag, []).append(d)

    print("\n" + "=" * 80)
    print("TRAINING PROGRESS PER PHASE (in-scenario learning curve)")
    print("=" * 80)
    print("Metric: rollout_mean_reward = mean of top-30% congested ports")
    print("        (robust to structurally-stuck ports; reflects agent's true ceiling)")
    print()
    print(f"{'Phase':40s} {'Ep':>3s} {'r_start':>8s} {'r_end':>8s} {'Δ':>7s} {'median':>7s} {'loss_end':>9s} {'cong%':>6s}")
    for tag, recs in groups.items():
        if not tag.startswith("train_"):
            continue
        rewards = [r.get("rollout_mean_reward") for r in recs if r.get("rollout_mean_reward") is not None]
        medians = [r.get("rollout_median_reward") for r in recs if r.get("rollout_median_reward") is not None]
        losses = [r.get("mean_loss") for r in recs if r.get("mean_loss") is not None]
        cong_ratios = [r.get("congested_step_ratio", 0.0) for r in recs]
        if not rewards:
            continue
        avg_cong = sum(cong_ratios) / len(cong_ratios) if cong_ratios else 0.0
        loss_end = losses[-1] if losses else float("nan")
        median_end = medians[-1] if medians else float("nan")
        print(f"{tag:40s} {len(recs):3d} {rewards[0]:8.4f} {rewards[-1]:8.4f} "
              f"{rewards[-1] - rewards[0]:+7.4f} {median_end:7.4f} {loss_end:9.4f} {avg_cong*100:5.1f}%")

    # ---- Catastrophic forgetting detection ----
    print("\n" + "=" * 80)
    print("CATASTROPHIC FORGETTING (greedy eval reward on prior tasks over time)")
    print("=" * 80)
    # Build a matrix: scenario -> [eval_reward after phase 1, phase 2, ...]
    eval_records = defaultdict(dict)
    for tag, recs in groups.items():
        if not tag.startswith("eval_"):
            continue
        # tag format: eval_p{N}_{ScenarioName}
        parts = tag.split("_", 2)
        if len(parts) < 3 or not parts[1].startswith("p"):
            continue
        phase_num = int(parts[1][1:])
        scenario = parts[2]
        if recs and recs[-1].get("rollout_mean_reward") is not None:
            eval_records[scenario][phase_num] = recs[-1]["rollout_mean_reward"]

    if not eval_records:
        print("(no eval records found)")
        return

    scenarios = list(eval_records.keys())
    phases = sorted({p for m in eval_records.values() for p in m.keys()})

    header = "Scenario".ljust(35) + "".join(f"  after P{p:>2}" for p in phases) + "  Δ(P{}-P{})".format(phases[-1], phases[0])
    print(header)
    for sc in scenarios:
        row = sc.ljust(35)
        first = last = None
        for p in phases:
            v = eval_records[sc].get(p)
            if v is None:
                row += "  " + " " * 8
            else:
                row += f"  {v:8.4f}"
                if first is None:
                    first = v
                last = v
        if first is not None and last is not None:
            drop = last - first
            marker = "  ⬇FORGOT" if drop < -0.005 else ("  ⬆GAINED" if drop > 0.005 else "")
            row += f"  {drop:+8.4f}{marker}"
        print(row)

    # Overall forgetting metric
    forgets = []
    for sc, m in eval_records.items():
        # Only consider scenarios trained before the last phase
        # Skip scenarios trained on the LAST phase (they have no "after" record)
        train_phase = None
        for tag in groups:
            if tag == f"train_p{phases[-1]}_{sc}":
                train_phase = phases[-1]
                break
        if train_phase == phases[-1]:
            continue
        vals = [m[p] for p in phases if p in m]
        if len(vals) >= 2:
            forget = max(vals) - vals[-1]
            forgets.append((sc, forget))
    if forgets:
        print("\n" + "-" * 80)
        print("Per-scenario forgetting = max_reward_over_phases - final_reward")
        for sc, f in forgets:
            print(f"  {sc:35s}  forgetting = {f:+.4f}")
        avg = sum(f for _, f in forgets) / len(forgets)
        print(f"  {'AVERAGE':35s}  forgetting = {avg:+.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    data = load(path)
    print(f"Loaded {len(data)} records from {path}")
    summarize(data)
