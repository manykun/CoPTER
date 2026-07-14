#!/usr/bin/env python3
"""Compare ACC and SOR catastrophic-forgetting curricula.

Reads metrics jsonl files written by `scripts/run_forgetting_curriculum.sh`
(default paths: cur_models_acc/cur_acc_metrics.jsonl and
cur_models_sor/cur_sor_metrics.jsonl), groups epochs by `eval_tag` (set by
the orchestrator to `train_p{phase}_{task}` or `eval_p{phase}_{task}`), and
prints per-task / per-phase reward tables plus three summary metrics:

  * Forgetting:   max(eval_reward[t] over phases) - eval_reward[t] at last phase
  * Backward Transfer (BWT[t]): average of (eval_reward[t] after later phases) -
                                eval_reward[t] right after first training that t
  * Recovery (Hadoop revisit):  reward at phase4-eval-Hadoop / reward at phase1-eval-Hadoop

Optionally writes matplotlib line plots (one per task) showing reward
trajectory across phase-end evaluations for ACC vs SOR.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from statistics import mean


PHASE_TAG_RE = re.compile(r"^(train|eval)_p(\d+)_(.+)$")


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        print(f"[WARN] missing metrics file: {path}", file=sys.stderr)
        return rows
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def group_by_eval_tag(rows):
    """Return {eval_tag: [reward, ...]} only for entries with mean_reward set
    and an eval_tag matching p<idx>_<task>."""
    bucket = defaultdict(list)
    for row in rows:
        tag = row.get("eval_tag", "")
        if not tag:
            continue
        m = PHASE_TAG_RE.match(tag)
        if not m:
            continue
        reward = row.get("rollout_mean_reward")
        if reward is None:
            reward = row.get("mean_reward")
        if reward is None:
            continue
        bucket[tag].append(float(reward))
    return bucket


def per_phase_eval_table(bucket):
    """Build {task: {phase: mean_reward_during_eval}} from eval_p{phase}_{task} entries."""
    table = defaultdict(dict)
    for tag, rewards in bucket.items():
        m = PHASE_TAG_RE.match(tag)
        if not m or m.group(1) != "eval":
            continue
        phase = int(m.group(2))
        task = m.group(3)
        table[task][phase] = mean(rewards) if rewards else None
    return table


def forgetting(table):
    out = {}
    for task, phase_map in table.items():
        if not phase_map:
            continue
        rewards = [v for v in phase_map.values() if v is not None]
        if not rewards:
            continue
        last_phase = max(phase_map.keys())
        last_v = phase_map[last_phase]
        if last_v is None:
            continue
        out[task] = max(rewards) - last_v
    return out


def backward_transfer(table, phase_to_train_task):
    """For each task t first trained at phase p_t, BWT is mean of
    (eval_reward[t] at phase>p_t) - eval_reward[t] at phase p_t.
    """
    out = {}
    for task, phase_map in table.items():
        if task not in phase_to_train_task.values():
            continue
        train_phases = [p for p, t in phase_to_train_task.items() if t == task]
        if not train_phases:
            continue
        first_train = min(train_phases)
        baseline = phase_map.get(first_train)
        if baseline is None:
            continue
        later = [v for p, v in phase_map.items() if p > first_train and v is not None]
        if not later:
            continue
        out[task] = mean(v - baseline for v in later)
    return out


def recovery(table, phase_to_train_task, revisit_phase=4, revisit_task="Hadoop_Shuffle"):
    if revisit_task not in table or revisit_phase not in table[revisit_task]:
        return None
    earlier_phases = [p for p, t in phase_to_train_task.items()
                      if t == revisit_task and p < revisit_phase]
    if not earlier_phases:
        return None
    first = min(earlier_phases)
    base = table[revisit_task].get(first)
    revisit = table[revisit_task].get(revisit_phase)
    if base in (None, 0) or revisit is None:
        return None
    return revisit / base


def fmt(v):
    if v is None:
        return "  N/A "
    return f"{v:+.4f}" if isinstance(v, float) else str(v)


def print_table(name, table, phases, tasks):
    print(f"\n=== {name} : per-phase greedy-eval mean reward ===")
    header = "  task                            | " + " | ".join(f"phase{p}" for p in phases)
    print(header)
    print("-" * len(header))
    for task in tasks:
        row = [f"  {task:<30}"]
        for p in phases:
            v = table.get(task, {}).get(p)
            row.append(f"{v:+.4f}" if v is not None else "  N/A ")
        print(" | ".join(row))


def maybe_plot(out_dir, tables, phases, tasks):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[INFO] matplotlib unavailable, skipping plots: {exc}")
        return
    os.makedirs(out_dir, exist_ok=True)
    for task in tasks:
        plt.figure(figsize=(6, 4))
        for label, table in tables.items():
            phase_map = table.get(task, {})
            xs = [p for p in phases if p in phase_map]
            ys = [phase_map[p] for p in xs]
            if xs:
                plt.plot(xs, ys, marker="o", label=label)
        plt.title(f"Greedy-eval reward on {task} across phases")
        plt.xlabel("Phase end")
        plt.ylabel("Mean reward")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = os.path.join(out_dir, f"forgetting_{task}.pdf")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"[INFO] wrote {path}")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(description="Compare ACC vs SOR catastrophic-forgetting curricula.")
    ap.add_argument("--acc", default=os.path.join(repo_root, "copter", "cur_models_acc", "cur_acc_metrics.jsonl"))
    ap.add_argument("--sor", default=os.path.join(repo_root, "sor", "cur_models_sor", "cur_sor_metrics.jsonl"))
    ap.add_argument("--phases", default="Hadoop_Shuffle,AliStorage_AllReduce,CacheFollower_burst_incast,Hadoop_Shuffle",
                    help="Comma-separated curriculum task order (matches run_forgetting_curriculum.sh PHASES).")
    ap.add_argument("--tasks", default="Hadoop_Shuffle,AliStorage_AllReduce,CacheFollower_burst_incast",
                    help="Comma-separated list of evaluation tasks.")
    ap.add_argument("--plot_dir", default=os.path.join(repo_root, "sor", "forgetting_plots"))
    args = ap.parse_args()

    phases_order = [t for t in args.phases.split(",") if t]
    eval_tasks = [t for t in args.tasks.split(",") if t]
    phase_to_train_task = {i + 1: t for i, t in enumerate(phases_order)}
    phase_indices = list(range(1, len(phases_order) + 1))

    acc_table = per_phase_eval_table(group_by_eval_tag(read_jsonl(args.acc)))
    sor_table = per_phase_eval_table(group_by_eval_tag(read_jsonl(args.sor)))

    print_table("ACC (FIFO replay baseline)", acc_table, phase_indices, eval_tasks)
    print_table("SOR (structured replay)", sor_table, phase_indices, eval_tasks)

    print("\n=== Forgetting (max_eval - last_eval, lower is better) ===")
    f_acc = forgetting(acc_table)
    f_sor = forgetting(sor_table)
    for task in eval_tasks:
        print(f"  {task:<30} ACC={fmt(f_acc.get(task))}   SOR={fmt(f_sor.get(task))}")

    print("\n=== Backward Transfer (BWT, higher is better) ===")
    bwt_acc = backward_transfer(acc_table, phase_to_train_task)
    bwt_sor = backward_transfer(sor_table, phase_to_train_task)
    for task in eval_tasks:
        print(f"  {task:<30} ACC={fmt(bwt_acc.get(task))}   SOR={fmt(bwt_sor.get(task))}")

    print("\n=== Recovery on revisit (phase4_Hadoop / phase1_Hadoop, closer to >=1.0 is better) ===")
    print(f"  ACC : {fmt(recovery(acc_table, phase_to_train_task))}")
    print(f"  SOR : {fmt(recovery(sor_table, phase_to_train_task))}")

    maybe_plot(args.plot_dir, {"ACC": acc_table, "SOR": sor_table}, phase_indices, eval_tasks)


if __name__ == "__main__":
    main()
