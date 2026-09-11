#!/usr/bin/env python3
"""Isolated fixed-input repeats and matched after-A A->A/A->B controls.

No reward/optimizer changes. Repeats measure rerun variability, not traffic-seed
generalization. Missing after-A replay is never silently borrowed from after-B.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from run_return_a import ROOT, config, digest, read, write, replay_source
from analyze_forgetting import load_eval


def fingerprint(directory):
    return {p.name: digest(p) for p in sorted(directory.iterdir())
            if p.is_file() and not p.name.endswith('_metrics.jsonl')}


def snapshot_info(base, phase):
    source = base / 'acc/checkpoints' / phase
    states = list(source.glob('*_train_state.json'))
    if len(states) != 1:
        raise ValueError(f'Expected one training state in {source}')
    exp = states[0].name[:-len('_train_state.json')]
    state = read(states[0])
    for port in range(state['node_number']):
        if not (source / f'{exp}_ACC_{port}').is_file():
            raise ValueError(f'Incomplete checkpoint {source}: port {port}')
    return source, exp, state


def initialize(base, phase, destination, exp, replay):
    source, old, state = snapshot_info(base, phase)
    donor = replay_source(base, source, state, old) if replay == 'require' else None
    before = fingerprint(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A failed copy must not look like an initialized model on --resume.
    temporary = tempfile.TemporaryDirectory(prefix='.copy-', dir=destination.parent)
    staging = Path(temporary.name)
    for p in source.iterdir():
        if p.is_file() and not p.name.endswith(('.pkl', '_metrics.jsonl')):
            shutil.copy2(p, staging / p.name.replace(old, exp))
    if donor:
        for port in range(state['node_number']):
            name = f'{old}_rb_port{port}.pkl'
            expected = digest(donor / name)
            copied = staging / name.replace(old, exp)
            shutil.copy2(donor / name, copied)
            if digest(copied) != expected or digest(donor / name) != expected:
                raise ValueError('Replay donor changed during copy')
    if fingerprint(source) != before:
        raise ValueError('Source changed during copy; stop concurrent source training')
    state['exp_name'] = exp
    if replay == 'reset':
        # The loader initializes empty per-port buffers when .pkl is absent.
        # Preserve optimizer, target, RNG and all global scheduling counters.
        state['replay_size_per_port'] = [0] * state['node_number']
    write(staging / f'{exp}_train_state.json', state)
    provenance = dict(phase=phase, checkpoint_hashes=before,
                replay_policy=replay, donor=str(donor) if donor else None,
                model_hashes=fingerprint(staging))
    write(staging / 'initialization.json', provenance)
    staging.rename(destination)
    temporary.cleanup()
    return provenance


def command(manifest, model, exp, port, ports):
    schedule = manifest['epsilon_schedule']
    cmd = ['bash', str(ROOT / 'run_training.sh'), '--exp', exp, '--mode', 'ACC',
           '--model-dir', str(model), '--port', str(port), '--seed', str(manifest['seed']),
           '--buffer', str(manifest['buffer_kb']), '--shared-replay', 'false',
           '--epsilon-schedule', 'global', '--eps-start', str(schedule['start']),
           '--eps-end', str(schedule['end']), '--eps-decay', str(schedule['decay_steps']),
           '--tb-enable', 'false', '--watch-ports', ','.join(map(str, ports))]
    for flag, key in [('action-space', 'action_space'), ('acc-hidden-dims', 'hidden_dims'),
                      ('reward-profile', 'reward_profile'), ('reward-queue-lambda', 'reward_queue_lambda'),
                      ('reward-ecn-lambda', 'reward_ecn_lambda'), ('target-update-interval', 'target_update_interval')]:
        value = manifest[key]
        cmd += ['--' + flag, ','.join(map(str, value)) if isinstance(value, list) else str(value)]
    cmd += ['--reward-weights', ','.join(map(str, manifest.get('reward_weights', [.5, .3, .2])))]
    return cmd


def execute(base, task, dest, cmd, extra):
    dest.mkdir(parents=True, exist_ok=True)
    origin = base / 'tasks' / task
    shutil.copy2(origin / 'input.flow', dest / 'input.flow')
    config(origin / 'input.conf', dest / 'input.flow', dest / 'input.conf', dest)
    subprocess.run(cmd + ['--config', str(dest / 'input.conf')] + extra, cwd=ROOT, check=True)


def evaluate(base, manifest, model, exp, root, label, cmd, repeats):
    for task in (manifest['task_a'], manifest['task_b']):
        for repetition in range(1, repeats + 1):
            dest = root / label / task / f'repeat_{repetition}'
            if (dest / 'complete.json').exists():
                marker = read(dest / 'complete.json')
                if marker['model_hashes'] != fingerprint(model):
                    raise ValueError('Completed evaluation model no longer matches')
                load_eval(dest)
                continue
            before = fingerprint(model)
            print(f'[frozen] {label} {task} repeat={repetition}', flush=True)
            execute(base, task, dest, cmd, ['--one-shot', '--eval-greedy', '--episodes', '1',
                    '--phase', 'attribution_eval', '--eval-tag', f'{label}_{task}_{repetition}',
                    '--watch-trace-file', str(dest / 'watch_trace.jsonl')])
            if before != fingerprint(model):
                raise ValueError('Frozen evaluation modified weights/state/replay')
            path = model / f'{exp}_metrics.jsonl'
            with path.open() as handle:
                last = None
                for line in handle:
                    if line.strip():
                        last = json.loads(line)
            if not last or not last.get('eval_greedy'):
                raise ValueError('Missing frozen metrics')
            write(dest / 'metrics.json', last)
            load_eval(dest)
            write(dest / 'complete.json', dict(model_hashes=before))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stage', choices=['repeat', 'control'], required=True)
    parser.add_argument('--updates', default='100,300,600')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--ports', default='323,321,320,345,346,347')
    parser.add_argument('--port', type=int, default=6956)
    parser.add_argument('--replay-policy', choices=['require', 'reset'], default='require',
                        help='Control only: reset explicitly clears BOTH arms, not an exact continuation')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    base, out = args.base_run_dir.resolve(), args.output_dir.resolve()
    points = [int(x) for x in args.updates.split(',')]
    ports = [int(x) for x in args.ports.split(',')]
    if not points or points != sorted(set(points)) or points[0] <= 0:
        parser.error('--updates must be strictly increasing positive integers')
    if args.repeats < 2 or not 1 <= args.port <= 65535:
        parser.error('Use at least 2 repeats and a valid socket port')
    if out == base or base in out.parents or out in base.parents:
        parser.error('Output must be a separate sibling directory')
    manifest = read(base / 'manifest.json')
    if manifest.get('shared_replay', True) or manifest.get('epsilon_schedule', {}).get('scope') != 'global':
        raise ValueError('Requires local-only ACC and global epsilon; do not alter the base protocol')
    _, _, state = snapshot_info(base, 'after_a')
    if not ports or len(ports) != len(set(ports)) or min(ports) < 0 or max(ports) >= state['node_number']:
        parser.error('Invalid watched ports')
    # Validate replay before creating output or running expensive evaluations.
    if args.stage == 'control' and args.replay_policy == 'require':
        src, old, st = snapshot_info(base, 'after_a')
        replay_source(base, src, st, old)
    phases = ['after_a', 'after_b'] if args.stage == 'repeat' else ['after_a']
    protocol = dict(base=str(base), manifest=manifest, stage=args.stage, points=points,
                    repeats=args.repeats, ports=ports, socket_port=args.port,
                    replay_policy=args.replay_policy if args.stage == 'control' else 'unused_frozen',
                    checkpoint_hashes={p: fingerprint(snapshot_info(base, p)[0]) for p in phases},
                    task_hashes={t: {n: digest(base / 'tasks' / t / n) for n in ['input.flow', 'input.conf']}
                                 for t in [manifest['task_a'], manifest['task_b']]},
                    repeat_scope='identical flow files, checkpoint RNG and configured seed; rerun variability only')
    if out.exists():
        if not args.resume or read(out / 'protocol.json') != protocol:
            raise ValueError('Existing output: use --resume with identical arguments and unchanged sources')
    else:
        out.mkdir(parents=True)
        write(out / 'protocol.json', protocol)
    if args.stage == 'repeat':
        for phase in phases:
            model, exp = out / 'models' / phase, f'attr_{out.name}_{phase}'
            if not model.exists():
                provenance = initialize(base, phase, model, exp, 'unused_frozen')
                write(model.parent / f'{phase}_provenance.json', provenance)
            cmd = command(manifest, model, exp, args.port, ports)
            evaluate(base, manifest, model, exp, out / 'eval', phase, cmd, args.repeats)
    else:
        # Initialize BOTH arms before either trains. Exact same checkpoint and
        # replay treatment; no reuse of the final replay from the A->B source.
        for arm in ['aa', 'ab']:
            model, exp = out / arm / 'models', f'attr_{out.name}_{arm}'
            if not model.exists():
                provenance = initialize(base, 'after_a', model, exp, args.replay_policy)
                write(out / arm / 'provenance.json', provenance)
        # All initial model, target, optimizer and replay bytes must match,
        # excluding only experiment-named metadata and the state exp_name.
        normalized = []
        for arm in ['aa', 'ab']:
            exp = f'attr_{out.name}_{arm}'
            hashes = read(out / arm / 'models/initialization.json')['model_hashes']
            normalized.append({name.replace(exp, 'same'): value for name, value in hashes.items()
                               if not name.endswith('_train_state.json')})
        if normalized[0] != normalized[1]:
            raise ValueError('Control arms did not start with identical checkpoint/replay bytes')
        for arm, task in [('aa', manifest['task_a']), ('ab', manifest['task_b'])]:
            model, exp = out / arm / 'models', f'attr_{out.name}_{arm}'
            cmd = command(manifest, model, exp, args.port, ports)
            state_path = model / f'{exp}_train_state.json'
            for n in [0] + points:
                snap = out / arm / 'checkpoints' / f'update_{n}'
                if not (snap / 'saved.json').exists():
                    current = read(state_path)
                    target = state['global_train_step'] + n
                    if current['global_train_step'] > target:
                        raise ValueError('Missing earlier checkpoint; cannot safely resume')
                    if current['global_train_step'] < target:
                        execute(base, task, out / arm / 'train' / str(n), cmd,
                                ['--phase', f'control_{arm}', '--episodes', str(current['epoch'] + 1000),
                                 '--target-train-steps', str(target)])
                    if read(state_path)['global_train_step'] != target:
                        raise ValueError(f'Exact optimizer budget {target} not reached')
                    snap.mkdir(parents=True, exist_ok=True)
                    for p in model.iterdir():
                        if p.is_file() and not p.name.endswith(('.pkl', '_metrics.jsonl')):
                            shutil.copy2(p, snap / p.name)
                    write(snap / 'saved.json', dict(global_train_step=target))
                # Evaluate saved frozen checkpoint, not the mutable live model;
                # this also permits resuming old evaluation nodes safely.
                eval_cmd = command(manifest, snap, exp, args.port, ports)
                evaluate(base, manifest, snap, exp, out / 'eval', f'{arm}_{n}', eval_cmd, args.repeats)
    print(f'Done. Analyze with analyze_acc_attribution.py --run-dir {out}', flush=True)


if __name__ == '__main__':
    main()
