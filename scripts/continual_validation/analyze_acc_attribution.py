#!/usr/bin/env python3
"""Separate replay training signals, frozen rewards, components and FCT.

Works offline on an original continual run or a run_acc_attribution.py output.
All summaries are descriptive; no selection of favorable seeds/checkpoints.
"""
import argparse
import csv
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

from analyze_forgetting import load_eval, summarize
from analyze_port_continual import frozen_port_summary


def read(path):
    return json.loads(path.read_text())


def export(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def contributions(values, manifest):
    """Mean(raw) is decomposable; clip(mean(raw)) is NOT mean(clipped)."""
    result = dict(values)
    if manifest.get('reward_profile') == 'tail_safe':
        for component, factor in [('queue', manifest['reward_queue_lambda']),
                                  ('ecn', manifest['reward_ecn_lambda'])]:
            cost = values.get(component + '_cost_sq')
            result[component + '_contribution'] = -factor * cost if cost is not None else None
        tx, q, e = (values.get('throughput'), result.get('queue_contribution'), result.get('ecn_contribution'))
        result['raw_reconstructed'] = tx + q + e if all(v is not None for v in (tx, q, e)) else None
        raw, reward = values.get('tail_safe_raw'), values.get('reward')
        result['clipping_mean_difference'] = reward - raw if raw is not None and reward is not None else None
    return result


def summarize_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['task'], row['label'])].append(row)
    output = []
    for (task, label), samples in sorted(groups.items()):
        row = dict(task=task, label=label, repeats=len(samples))
        for key in ['reward', 'p95_fct_us', 'completion_ratio', 'p99_slowdown']:
            values = [s[key] for s in samples if s.get(key) is not None]
            for name, fn in [('mean', stats.mean), ('min', min), ('max', max)]:
                row[key + '_' + name] = fn(values) if values else None
            row[key + '_std'] = stats.stdev(values) if len(values) > 1 else None
        output.append(row)
    return output


def control_effects(rows, task_a, points):
    indexed = {(r['label'], r['task'], r['repeat']): r for r in rows}
    repetitions = sorted({r['repeat'] for r in rows})
    result = []
    for n in points:
        for repeat in repetitions:
            labels = ['aa_0', f'aa_{n}', 'ab_0', f'ab_{n}']
            selected = [indexed.get((label, task_a, repeat)) for label in labels]
            if any(r is None for r in selected):
                raise ValueError(f'Missing paired control evaluation at {n}, repeat {repeat}')
            a0, aa, b0, ab = selected
            item = dict(updates=n, repeat=repeat)
            for key in ['reward', 'p95_fct_us', 'completion_ratio']:
                item[key + '_aa_delta'] = aa[key] - a0[key]
                item[key + '_ab_delta'] = ab[key] - b0[key]
                item[key + '_ab_minus_aa_delta'] = item[key + '_ab_delta'] - item[key + '_aa_delta']
            result.append(item)
    return result


def training_rows(root, ports):
    rows = []
    for path in sorted(root.rglob('*_metrics.jsonl')):
        if 'models' not in path.relative_to(root).parts or 'checkpoints' in path.relative_to(root).parts:
            continue
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get('eval_greedy'):
                    continue
                for port in ports:
                    d = r.get('train_port_metrics', {}).get(str(port), {})
                    if d.get('updates', 0) <= 0:
                        continue
                    rows.append(dict(source=str(path.relative_to(root)), port=port,
                        phase=r.get('phase'), global_train_step=r.get('global_train_step'),
                        epoch=r.get('epoch'), epsilon=r.get('epsilon'), updates=d['updates'],
                        batch_reward=d.get('batch_reward_mean'), loss=d.get('loss_mean'),
                        online_rollout_reward=r.get('rollout_all_congested_mean')))
    return rows


def fmt(value):
    return 'n/a' if value is None else f'{value:.6g}'


def analyze(root, out, ports):
    protocol_path = root / 'protocol.json'
    protocol = read(protocol_path) if protocol_path.exists() else {}
    manifest = protocol.get('manifest') or read(root / 'manifest.json')
    tasks = [manifest['task_a'], manifest['task_b']]
    runs = defaultdict(list)
    for path in sorted((root / 'eval').rglob('metrics.json')):
        parts = path.parent.relative_to(root / 'eval').parts
        if path.parent.name.startswith('repeat_'):
            label, task, rep = '/'.join(parts[:-2]), parts[-2], int(parts[-1].split('_')[-1])
        else:
            label, task, rep = '/'.join(parts[:-1]), parts[-1], 1
        if task not in tasks:
            continue
        if protocol.get('stage') in ['repeat', 'control'] and not (path.parent / 'complete.json').exists():
            raise ValueError(f'Incomplete frozen evaluation: {path.parent}')
        runs[task].append((label, rep, load_eval(path.parent)))
    if not all(runs[t] for t in tasks):
        raise ValueError('Missing evaluation of one or both tasks')
    if protocol.get('stage') in ['repeat', 'control']:
        labels = (['after_a', 'after_b'] if protocol['stage'] == 'repeat' else
                  [f'{arm}_{n}' for arm in ['aa', 'ab'] for n in [0] + protocol['points']])
        expected = {(label, i) for label in labels for i in range(1, protocol['repeats'] + 1)}
        for task in tasks:
            if {(l, i) for l, i, _ in runs[task]} != expected:
                raise ValueError(f'Incomplete repeat matrix for {task}; run --resume first')
    network, components, physical = [], [], []
    for task in tasks:
        selected = runs[task]
        common = set.intersection(*(set(run['flows']) for _, _, run in selected))
        if not common:
            raise ValueError(f'No common completed flows for {task}')
        for label, rep, run in selected:
            identity = dict(task=task, label=label, repeat=rep)
            network.append(dict(identity, **summarize(run, common)))
            metrics = run['metrics']
            global_values = {k[len('reward_'):-len('_mean')]: v for k, v in metrics.items()
                             if k.startswith('reward_') and k.endswith('_mean') and isinstance(v, (float, int))}
            global_values['reward'] = metrics.get('rollout_all_congested_mean')
            components.append(dict(identity, port='network', population='congested_with_topk_fallback',
                samples=metrics.get('rollout_reward_samples'),
                fallback_samples=metrics.get('rollout_reward_fallback_samples'),
                **contributions(global_values, manifest)))
            for port in ports:
                detail = metrics.get('watch_ports_metrics', {}).get(str(port), {})
                for population, count in [('all', 'samples'), ('active', 'active_steps'), ('congested', 'congested_steps')]:
                    components.append(dict(identity, port=port, population=population,
                        samples=detail.get(count), **contributions(detail.get('means_' + population, {}), manifest)))
                physical.append(dict(identity, port=port,
                    **frozen_port_summary(run['directory'], port, manifest['buffer_kb'])))
    out.mkdir(parents=True, exist_ok=True)
    training = training_rows(root, ports)
    grouped = summarize_groups(network)
    effects = control_effects(network, tasks[0], protocol['points']) if protocol.get('stage') == 'control' else []
    for name, rows in [('network', network), ('repeat_summary', grouped), ('reward_components', components),
                       ('physical_ports', physical), ('training', training), ('control_effects', effects)]:
        export(out / (name + '.csv'), rows)
    (out / 'analysis.json').write_text(json.dumps(dict(protocol=protocol, network=network,
        repeat_summary=grouped, control_effects=effects), indent=2) + '\n')
    lines = ['# ACC reward / drift attribution', '',
        f"Replay treatment: {protocol.get('replay_policy', 'original experiment')}",
        'Frozen common-flow FCT: one intersection per task across ALL reported nodes and repeats.',
        'Completion uses all offered flows. Mean/std/range are descriptive, not confidence intervals.',
        'Identical-seed reruns measure reproducibility noise, NOT independent traffic generalization.', '',
        '| Task | Checkpoint | Repeats | Reward mean | Reward range | p95 mean us | p95 range us | Completion mean |',
        '|---|---|---:|---:|---|---:|---|---:|']
    for r in grouped:
        lines.append(f"| {r['task']} | {r['label']} | {r['repeats']} | {fmt(r['reward_mean'])} | "
            f"{fmt(r['reward_min'])}–{fmt(r['reward_max'])} | {fmt(r['p95_fct_us_mean'])} | "
            f"{fmt(r['p95_fct_us_min'])}–{fmt(r['p95_fct_us_max'])} | {r['completion_ratio_mean']:.4%} |")
    lines += ['', '## Frozen network reward decomposition (repeat means)', '',
        '| Task | Checkpoint | Throughput term | Queue contribution | ECN contribution | Raw reward | Clipped reward |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in grouped:
        selected = [c for c in components if c['port'] == 'network'
                    and c['task'] == r['task'] and c['label'] == r['label']]
        values = []
        for key in ['throughput', 'queue_contribution', 'ecn_contribution', 'tail_safe_raw', 'reward']:
            available = [c[key] for c in selected if c.get(key) is not None]
            values.append(fmt(stats.mean(available)) if available else 'n/a')
        lines.append(f"| {r['task']} | {r['label']} | " + ' | '.join(values) + ' |')
    if effects:
        lines += ['', '## Matched training control: frozen A',
            'Difference = (A->B end - its start) - (A->A end - its start).',
            'Negative reward / positive FCT differences indicate extra deterioration after B.',
            'These are paired descriptive deltas, not a statistical significance test.', '',
            '| Updates | Mean extra reward change | Mean extra p95 us | Mean extra completion pp |',
            '|---:|---:|---:|---:|']
        for n in protocol['points']:
            rs = [r for r in effects if r['updates'] == n]
            lines.append(f"| {n} | {stats.mean(r['reward_ab_minus_aa_delta'] for r in rs):.6f} | "
                f"{stats.mean(r['p95_fct_us_ab_minus_aa_delta'] for r in rs):.3f} | "
                f"{100 * stats.mean(r['completion_ratio_ab_minus_aa_delta'] for r in rs):.4f} |")
    lines += ['', '## Reward interpretation',
        '- training.csv separates replay batch reward, online rollout reward and loss; neither is frozen evaluation.',
        '- reward_components.csv separates fixed watch-port ALL / active / congested populations. No fallback between them.',
        '- Network primary reward includes top-K fallback when no ports are congested; historical fallback counts may be unavailable.',
        '- For tail_safe: raw = throughput - queue_lambda * queue_cost_sq - ecn_lambda * ecn_cost_sq.',
        '- Compare raw_reconstructed with tail_safe_raw; clipping_mean_difference is mean(clipped)-mean(raw), not clip(mean).',
        '- Physical signals and PFC (missing remains n/a): physical_ports.csv. ECN increases alone do not imply harm.',
        '- Compare completion alongside common-flow latency: incomplete flows are excluded from FCT.',
        '- Equal sample counts do not establish identical states; six watched ports do not represent the full network.',
        '- Reset replay controls are matched cold-replay interventions, not seamless continuation of the old experiment.',
        '- No automatic PASS/FAIL, no reward retuning, no selection of the most favorable checkpoint.']
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        lines += ['', 'Matplotlib unavailable; CSVs are complete.']
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        for col, task in enumerate(tasks):
            rs = [r for r in grouped if r['task'] == task]
            for ax, key in [(axes[0, col], 'reward'), (axes[1, col], 'p95_fct_us')]:
                means = [r[key + '_mean'] for r in rs]
                if any(v is None for v in means):
                    continue
                ax.errorbar(range(len(rs)), means, fmt='o', capsize=4,
                            yerr=[[r[key + '_mean'] - r[key + '_min'] for r in rs],
                                  [r[key + '_max'] - r[key + '_mean'] for r in rs]])
                ax.set(title=task, ylabel='Frozen ' + key)
                ax.set_xticks(range(len(rs)))
                ax.set_xticklabels([r['label'] for r in rs], rotation=40, ha='right')
                ax.grid(alpha=.25)
        fig.suptitle('Frozen evaluation: mean and observed min-max (not confidence intervals)')
        fig.tight_layout()
        fig.savefig(out / 'frozen.png', dpi=160)
        plt.close(fig)
        lines += ['', '![Frozen evaluations](frozen.png)']
        for port in ports:
            fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
            groups = defaultdict(list)
            for r in training:
                if r['port'] == port:
                    groups[(r['source'], r['phase'])].append(r)
            for (source, phase), rs in groups.items():
                rs.sort(key=lambda r: r['global_train_step'])
                for ax, key in zip(axes, ['batch_reward', 'online_rollout_reward', 'loss']):
                    valid = [r for r in rs if r[key] is not None]
                    ax.plot([r['global_train_step'] for r in valid], [r[key] for r in valid], label=phase)
                    ax.set_ylabel(key)
                    ax.grid(alpha=.25)
            axes[1].set_ylabel('Online NETWORK reward\n(not port reward)')
            axes[2].set_yscale('symlog', linthresh=1e-4)
            axes[2].set_xlabel('Cumulative optimizer updates')
            if groups:
                axes[0].legend()
            fig.suptitle(f'Port {port}: replay / online network rollout / TD loss (not frozen)')
            fig.tight_layout()
            fig.savefig(out / f'training_p{port}.png', dpi=160)
            plt.close(fig)
            if groups:
                lines += ['', f'![Training port {port}](training_p{port}.png)']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(out / 'REPORT.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ports', default='323,321,320,345,346,347')
    args = parser.parse_args()
    ports = [int(p) for p in args.ports.split(',')]
    root = args.run_dir.resolve()
    out = args.output_dir.resolve() if args.output_dir else root / 'attribution_report'
    if out == root or out in root.parents:
        parser.error('Report output must not be the run root or its ancestor')
    analyze(root, out, ports)


if __name__ == '__main__':
    main()
