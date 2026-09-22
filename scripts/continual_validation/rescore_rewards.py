#!/usr/bin/env python3
"""Score the SAME frozen observations with weighted and tail-safe rewards.

This is descriptive rescoring, not an off-policy performance estimate for a
retrained policy. Never clip a mean raw reward to infer mean clipped reward.
"""
import argparse
import csv
import math
from pathlib import Path
from analyze_acc_attribution import export


def number(value):
    return None if value in (None, '') else float(value)


def score(row, weights, recorded_profile):
    parts = [number(row.get(k)) for k in ['throughput', 'queue', 'ecn']]
    weighted = sum(w * v for w, v in zip(weights, parts)) if all(v is not None for v in parts) else None
    tail = number(row.get('tail_safe_clipped'))
    if tail is None and recorded_profile == 'tail_safe':
        tail = number(row.get('reward'))
    return dict(weighted_score=weighted, tail_safe_score=tail,
                tail_safe_raw=number(row.get('tail_safe_raw')),
                tail_safe_available=tail is not None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report-dir', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--weights', default='0.50,0.30,0.20')
    p.add_argument('--recorded-profile', choices=['weighted', 'tail_safe'], default='tail_safe')
    args = p.parse_args()
    weights = [float(x) for x in args.weights.split(',')]
    if len(weights) != 3 or any(not math.isfinite(x) or x < 0 for x in weights) or abs(sum(weights)-1) > 1e-8:
        p.error('Use three finite nonnegative weights summing to one')
    root, out = args.report_dir.resolve(), args.output_dir.resolve()
    if out == root or out in root.parents:
        p.error('Use a separate output directory')
    with (root / 'reward_components.csv').open() as f:
        components = list(csv.DictReader(f))
    with (root / 'network.csv').open() as f:
        net = {(r['task'], r['label'], r['repeat']): r for r in csv.DictReader(f)}
    rows = []
    for r in components:
        recorded = args.recorded_profile
        # Reward-ablation labels carry their actual generating profile.
        if r['label'].startswith(('weighted_', 'tail_safe_')):
            recorded = 'weighted' if r['label'].startswith('weighted_') else 'tail_safe'
        item = dict(r, **score(r, weights, recorded), recorded_profile=recorded)
        n = net[r['task'], r['label'], r['repeat']]
        for key in ['p95_fct_us', 'completion_ratio', 'matched_flows']:
            item['network_' + key] = number(n[key])
        rows.append(item)
    out.mkdir(parents=True, exist_ok=True)
    export(out / 'dual_scores.csv', rows)
    lines = ['# Same-trajectory reward comparison', '',
        f'Weighted coefficients: {weights}. No policy is retrained or altered.',
        'Both scores use exactly the same population within each row. Populations may change across checkpoints.',
        'Network FCT uses the source report common-flow intersection; completion includes all offered flows.',
        'Old weighted recordings without mean clipped tail-safe retain n/a; clip(mean(raw)) is never substituted.', '',
        '| Task | Checkpoint | Repeat | Weighted score | Tail-safe score | Network p95 us | Completion |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['port'] != 'network':
            continue
        def fmt(v):
            return 'n/a' if v is None else f'{v:.6f}'
        lines.append(f"| {r['task']} | {r['label']} | {r['repeat']} | {fmt(r['weighted_score'])} | "
                     f"{fmt(r['tail_safe_score'])} | {r['network_p95_fct_us']:.3f} | {r['network_completion_ratio']:.4%} |")
    lines += ['', 'Compare within-task directions, not the absolute magnitudes of different reward formulas.',
              'A more aligned score on these trajectories does not prove training with it will improve FCT.',
              'Per-port all/active/congested dual scores and network context are in dual_scores.csv.',
              'No PASS/FAIL or automatic reward selection. Native rewards from different formulas are not directly comparable.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(out / 'REPORT.md')


if __name__ == '__main__':
    main()
