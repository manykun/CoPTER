#!/usr/bin/env python3
"""Resume a completed ACC after-B snapshot in an isolated A-relearning run."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from analyze_forgetting import load_eval, summarize
from analyze_port_continual import frozen_port_summary

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_KEYS = {
    'TRACE_OUTPUT_FILE': '.tr', 'FCT_OUTPUT_FILE': '.fct',
    'PFC_OUTPUT_FILE': '.pfc', 'QLEN_MONITOR_FILE': '.queue',
    'RATE_MONITOR_FILE': '.rate', 'THROUGHPUT_OUTPUT_FILE': '.throughput',
    'QLEN_MON_FILE': '.qlen',
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def replay_source(base, source, state, exp):
    """Phase snapshots intentionally omit .pkl; validate the live replay donor."""
    names = [f'{exp}_rb_port{i}.pkl' for i in range(state['node_number'])]
    weights = [f'{exp}_ACC_{i}' for i in range(state['node_number'])]
    for name in weights:
        if not (source / name).is_file():
            raise ValueError(f'Incomplete after-B snapshot: missing {name}')
    if all((source / name).is_file() for name in names):
        return source
    live = base / 'acc/models'
    live_state_path = live / f'{exp}_train_state.json'
    if not live_state_path.is_file():
        raise ValueError(f'No live replay donor state: {live_state_path}')
    live_state = read(live_state_path)
    for key in ('global_train_step', 'global_env_step', 'epoch', 'epsilon',
                'phase', 'node_number', 'replay_size_per_port'):
        if key not in state or live_state.get(key) != state[key]:
            raise ValueError(f'Live models do not match after-B state ({key}); refusing replay reuse')
    for name in weights:
        if not (live / name).is_file() or digest(live / name) != digest(source / name):
            raise ValueError(f'Live checkpoint differs from after-B: {name}; refusing replay reuse')
    for name in names:
        if not (live / name).is_file():
            raise ValueError(f'Missing local replay file: {live / name}')
    return live


def config(source, flow, destination, output):
    lines = []
    for line in source.read_text().splitlines():
        words = line.split()
        key = words[0] if words else ''
        if key == 'FLOW_FILE':
            line = f'{key} {flow}'
        elif key in OUTPUT_KEYS:
            line = f'{key} {output / ("result" + OUTPUT_KEYS[key])}'
        lines.append(line)
    destination.write_text('\n'.join(lines) + '\n')


def analyze(out, base, manifest, points, ports):
    rows, port_rows = [], []
    for task in (manifest['task_a'], manifest['task_b']):
        locations = [('after_a', base / 'eval/acc/after_a' / task),
                     ('return_0', out / 'eval/return_0' / task)]
        locations += [(f'return_{n}', out / 'eval' / f'return_{n}' / task) for n in points]
        runs = [(label, load_eval(path)) for label, path in locations]
        common = set.intersection(*(set(run['flows']) for _, run in runs))
        if not common:
            raise ValueError(f'No common completed flows for {task}')
        for (label, run), (_, directory) in zip(runs, locations):
            rows.append(dict(task=task, checkpoint=label, **summarize(run, common)))
            for port in ports:
                port_rows.append(dict(task=task, checkpoint=label, port=port,
                    **frozen_port_summary(directory, port, manifest['buffer_kb'])))
    for name, values in [('network', rows), ('ports', port_rows)]:
        with (out / f'{name}.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write(out / 'analysis.json', dict(network=rows, ports=port_rows))
    report = ['# ACC A → B → A recovery', '',
        'Frozen evaluations. Within each task all checkpoints use one common-flow intersection.',
        'Completion uses all offered flows; port populations can change between checkpoints.',
        'return_0 is a fresh evaluation of the copied after-B snapshot.', '',
        '| Task | Checkpoint | Common flows | Reward | p95 FCT (us) | Completion |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        reward = 'n/a' if row['reward'] is None else f"{row['reward']:.6f}"
        report.append(f"| {row['task']} | {row['checkpoint']} | {row['matched_flows']} | "
                      f"{reward} | {row['p95_fct_us']:.3f} | {row['completion_ratio']:.4%} |")
    report += ['', 'Compare recovery of A with any cost on B; no PASS/FAIL gate is applied.',
               'Per-port reward, queue, ECN, PFC and sample counts: ports.csv.',
               'Small changes require repeatability checks before causal claims.']
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        report += ['', 'Plotting unavailable: install matplotlib or plot the CSV on another machine.']
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for col, task in enumerate((manifest['task_a'], manifest['task_b'])):
            selected = [r for r in rows if r['task'] == task]
            for axis, key, title in ((axes[0, col], 'reward', 'Frozen reward'),
                                     (axes[1, col], 'p95_fct_us', 'Common-flow p95 FCT (us)')):
                axis.plot([0] + points, [r[key] for r in selected[1:]], 'o-')
                axis.axhline(selected[0][key], linestyle='--', color='gray', label='after-A')
                axis.set(title=task, ylabel=title, xlabel='Additional A optimizer updates')
                axis.grid(alpha=.25)
                axis.legend()
        fig.tight_layout()
        fig.savefig(out / 'recovery.png', dpi=180)
        plt.close(fig)
        model = out / 'models'
        records = [json.loads(line) for path in model.glob('*_metrics.jsonl')
                   for line in path.read_text().splitlines() if line.strip()]
        for port in ports:
            samples = [(r['global_train_step'], r['train_port_metrics'][str(port)])
                       for r in records if r.get('phase') == 'return_a'
                       and str(port) in r.get('train_port_metrics', {})]
            if not samples:
                continue
            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
            origin = read(out / 'protocol.json')['start_step']
            for axis, key, label in ((axes[0], 'batch_reward_mean', 'Batch reward'),
                                     (axes[1], 'loss_mean', 'Mean TD loss')):
                axis.plot([s-origin for s, _ in samples], [d.get(key) for _, d in samples], 'o-')
                axis.set_ylabel(label)
                axis.grid(alpha=.25)
            axes[1].set_yscale('symlog', linthresh=1e-4)
            axes[1].set_xlabel('Additional A optimizer updates')
            fig.suptitle(f'ACC return to A: port {port}')
            fig.tight_layout()
            fig.savefig(out / f'training_p{port}.png', dpi=180)
            plt.close(fig)
        report += ['', '![Frozen recovery](recovery.png)']
    (out / 'REPORT.md').write_text('\n'.join(report) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--updates', default='20,50,100', help='Cumulative additional update counts')
    parser.add_argument('--ports', default='323,321,320,345,346,347')
    parser.add_argument('--port', type=int, default=6856)
    parser.add_argument('--stage', choices=['run', 'analyze'], default='run')
    args = parser.parse_args()
    base, out = args.base_run_dir.resolve(), args.output_dir.resolve()
    points = [int(x) for x in args.updates.split(',')]
    ports = [int(x) for x in args.ports.split(',')]
    if not points or points != sorted(set(points)) or points[0] <= 0:
        parser.error('--updates must be strictly increasing positive counts')
    if out == base or base in out.parents:
        parser.error('Use a separate sibling output directory to preserve the base experiment')
    manifest = read(base / 'manifest.json')
    source = base / 'acc/checkpoints/after_b'
    states = list(source.glob('*_train_state.json'))
    if len(states) != 1:
        raise ValueError('Expected exactly one after-B training state')
    state = read(states[0])
    if not ports or len(set(ports)) != len(ports) or min(ports) < 0:
        parser.error('--ports must be unique non-negative integers')
    if manifest.get('shared_replay', True):
        raise ValueError('This recovery experiment requires the existing local-only ACC run')
    schedule = manifest.get('epsilon_schedule', {})
    if not isinstance(schedule, dict) or schedule.get('scope') != 'global':
        raise ValueError('Base experiment must already use global epsilon; do not change protocol')
    old_exp = states[0].name.removesuffix('_train_state.json')
    exp = 'return_' + out.name
    protocol = dict(base=str(base), manifest=manifest, start_step=state['global_train_step'],
                    points=points, ports=ports, socket_port=args.port, experiment=exp)
    if args.stage == 'run':
        if out.exists():
            raise ValueError(f'Output already exists: {out}. Use --stage analyze or a fresh output-dir')
        donor = replay_source(base, source, state, old_exp)
        print(f'Validated local replay donor: {donor}', flush=True)
        out.mkdir(parents=True)
        write(out / 'protocol.json', protocol)
        model = out / 'models'
        model.mkdir()
        for path in source.iterdir():
            if path.is_file() and not path.name.endswith('_metrics.jsonl'):
                shutil.copy2(path, model / path.name.replace(old_exp, exp))
        for i in range(state['node_number']):
            name = f'{old_exp}_rb_port{i}.pkl'
            shutil.copy2(donor / name, model / name.replace(old_exp, exp))
        write(out / 'replay_provenance.json', dict(source=str(donor),
              global_train_step=state['global_train_step'],
              files={f'{exp}_rb_port{i}.pkl': digest(model / f'{exp}_rb_port{i}.pkl')
                     for i in range(state['node_number'])}))
        state_path = model / (exp + '_train_state.json')
        state['exp_name'] = exp
        write(state_path, state)
        common = ['bash', str(ROOT / 'run_training.sh'), '--exp', exp, '--mode', 'ACC',
                  '--model-dir', str(model), '--port', str(args.port),
                  '--seed', str(manifest['seed']), '--buffer', str(manifest['buffer_kb']),
                  '--shared-replay', 'false', '--epsilon-schedule', 'global',
                  '--eps-start', str(schedule['start']), '--eps-end', str(schedule['end']),
                  '--eps-decay', str(schedule['decay_steps']), '--tb-enable', 'false',
                  '--watch-ports', args.ports, '--run-id', out.name]
        for flag, key in [('action-space', 'action_space'), ('acc-hidden-dims', 'hidden_dims'),
                          ('reward-profile', 'reward_profile'), ('reward-queue-lambda', 'reward_queue_lambda'),
                          ('reward-ecn-lambda', 'reward_ecn_lambda'), ('target-update-interval', 'target_update_interval')]:
            value = manifest[key]
            if isinstance(value, list):
                value = ','.join(map(str, value))
            common += ['--' + flag, str(value)]
        weights = manifest.get('reward_weights', [0.5, 0.3, 0.2])
        common += ['--reward-weights', ','.join(map(str, weights))]

        def execute(task, destination, extra):
            destination.mkdir(parents=True)
            task_dir = base / 'tasks' / task
            shutil.copy2(task_dir / 'input.flow', destination / 'input.flow')
            config(task_dir / 'input.conf', destination / 'input.flow',
                   destination / 'input.conf', destination)
            subprocess.run(common + ['--config', str(destination / 'input.conf')] + extra,
                           cwd=ROOT, check=True)

        def evaluate(label):
            saved_state = state_path.read_bytes()
            for task in (manifest['task_a'], manifest['task_b']):
                dest = out / 'eval' / label / task
                execute(task, dest, ['--one-shot', '--eval-greedy', '--episodes', '1',
                    '--phase', label, '--eval-tag', label + '_' + task,
                    '--watch-trace-file', str(dest / 'watch_trace.jsonl')])
                metrics = model / (exp + '_metrics.jsonl')
                write(dest / 'metrics.json', json.loads(metrics.read_text().splitlines()[-1]))
                load_eval(dest)
                if state_path.read_bytes() != saved_state:
                    raise ValueError('Frozen evaluation unexpectedly modified training state')

        evaluate('return_0')
        for n in points:
            target = protocol['start_step'] + n
            current = read(state_path)
            execute(manifest['task_a'], out / 'train' / str(n),
                    ['--phase', 'return_a', '--episodes', str(current['epoch'] + 100),
                     '--target-train-steps', str(target)])
            if read(state_path)['global_train_step'] != target:
                raise ValueError(f'Training did not reach exact target {target}; analysis stopped')
            shutil.copytree(model, out / 'checkpoints' / f'return_{n}')
            evaluate(f'return_{n}')
    else:
        if read(out / 'protocol.json') != protocol:
            raise ValueError('Arguments differ from saved protocol')
    analyze(out, base, manifest, points, ports)
    print(f'Recovery report: {out / "REPORT.md"}')


if __name__ == '__main__':
    main()
