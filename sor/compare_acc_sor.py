#!/usr/bin/env python3
import argparse
import json
import os
from statistics import mean


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def summarize(rows):
    rewards = [float(row["mean_reward"]) for row in rows if row.get("mean_reward") is not None]
    losses = [float(row["mean_loss"]) for row in rows if row.get("mean_loss") is not None]
    steps = [int(row["steps_this_epoch"]) for row in rows if row.get("steps_this_epoch") is not None]
    return {
        "epochs": len(rows),
        "reward_mean": mean(rewards) if rewards else None,
        "reward_last": rewards[-1] if rewards else None,
        "reward_best": max(rewards) if rewards else None,
        "loss_mean": mean(losses) if losses else None,
        "loss_last": losses[-1] if losses else None,
        "steps_total": sum(steps) if steps else None,
    }


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_acc = os.path.join(repo_root, "copter", "eval_models_acc_baseline", "eval_acc_baseline_metrics.jsonl")
    default_sor = os.path.join(repo_root, "sor", "eval_models_sor", "eval_sor_metrics.jsonl")

    parser = argparse.ArgumentParser(description="Compare ACC and SOR metrics jsonl files.")
    parser.add_argument("--acc", default=default_acc)
    parser.add_argument("--sor", default=default_sor)
    args = parser.parse_args()

    if not os.path.exists(args.acc):
        raise FileNotFoundError(f"ACC metrics file not found: {args.acc}")
    if not os.path.exists(args.sor):
        raise FileNotFoundError(f"SOR metrics file not found: {args.sor}")

    acc_rows = [row for row in read_jsonl(args.acc) if row.get("mean_reward") is not None]
    sor_rows = [row for row in read_jsonl(args.sor) if row.get("mean_reward") is not None]
    acc = summarize(acc_rows)
    sor = summarize(sor_rows)
    print("metric,acc,sor,delta_sor_minus_acc")
    for key in ["epochs", "reward_mean", "reward_last", "reward_best", "loss_mean", "loss_last", "steps_total"]:
        acc_value = acc.get(key)
        sor_value = sor.get(key)
        delta = sor_value - acc_value if isinstance(acc_value, (int, float)) and isinstance(sor_value, (int, float)) else None
        print(f"{key},{fmt(acc_value)},{fmt(sor_value)},{fmt(delta)}")


if __name__ == "__main__":
    main()
