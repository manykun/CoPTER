#!/usr/bin/env python3
"""Sparse frozen learning curves, matched AA/AB, and bounded return-A tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
from pathlib import Path
import shutil
import socket
import tempfile

from run_acc_attribution import command as base_command, execute, evaluate, fingerprint
from run_return_a import read, write, digest
from run_reward_ablation import first_training_record, verify_start


def command(manifest, model, exp, port, ports):
    # Forks intentionally keep serialized exp_name, so isolate driver logs
    # by model directory rather than allowing equal epoch names to overwrite.
    return ['env', 'COPTER_LOG_BASE=' + str(model.parent/'logs'/model.name)] + \
        base_command(manifest, model, exp, port, ports)


def points(text):
    result = [int(x) for x in text.split(',')]
    if not result or result != sorted(set(result)) or result[0] <= 0:
        raise argparse.ArgumentTypeError('Use increasing positive update counts')
    return result


def state(model, exp):
    return read(model / (exp + '_train_state.json'))


def check_snapshot(model, exp, target):
    s = state(model, exp)
    if s['global_train_step'] != target:
        raise ValueError('Checkpoint update count mismatch')
    for i in range(s['node_number']):
        for suffix in [f'_ACC_{i}', f'_rb_port{i}.pkl']:
            if not (model / (exp + suffix)).is_file():
                raise ValueError(f'Missing weights/replay: {model / (exp + suffix)}')


def copy_checkpoint(source, dest):
    """Atomic full copy, no shared writable model/replay files across arms."""
    before = fingerprint(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.copy-', dir=dest.parent) as tmp:
        staging = Path(tmp) / 'data'
        shutil.copytree(source, staging, ignore=shutil.ignore_patterns('*_metrics.jsonl'))
        if fingerprint(staging) != before or fingerprint(source) != before:
            raise ValueError('Checkpoint changed during copy')
        staging.rename(dest)


def frozen(base, m, model, exp, root, label, port, ports, tasks):
    # The shared evaluator evaluates both manifest tasks. Equal names evaluate
    # task A once; its second visit is safely skipped by the immutable marker.
    selected = dict(m, task_a=tasks[0], task_b=tasks[-1])
    evaluate(base, selected, model, exp, root / 'eval', label,
             command(m, model, exp, port, ports), 1)


def train_to(base, m, root, profile, arm, model, exp, task, target, port, ports):
    current = state(model, exp)
    if current['global_train_step'] > target:
        raise ValueError('Missing earlier checkpoint; refusing to run backwards')
    if current['global_train_step'] < target:
        execute(base, task, root / profile / arm / 'train' / str(target),
                command(m, model, exp, port, ports),
                ['--phase', 'train_' + arm, '--episodes', str(current['epoch'] + 1000),
                 '--target-train-steps', str(target)])
    check_snapshot(model, exp, target)


def worker(root_string, profile, scope):
    root = Path(root_string)
    p = read(root / 'protocol.json')
    base = Path(p['base'])
    m = dict(p['manifest'], reward_profile=profile, reward_weights=[.5, .3, .2])
    port = p['socket_ports'][profile]
    ports = p['ports']
    # Distinct model paths and socket ports; keep exp identical across each
    # profile's branches to preserve every serialized state field at fork.
    exp = 'study_' + root.name + '_' + profile
    expected = read(root / 'initialization_check.json')['network_hashes']
    a, b = m['task_a'], m['task_b']
    if scope in ('acquire', 'all'):
        model = root / profile / 'a/models'
        model.mkdir(parents=True, exist_ok=True)
        if not (model / (exp + '_train_state.json')).exists():
            if list(model.iterdir()):
                raise ValueError('Interrupted bootstrap without valid state; inspect before restarting')
            execute(base, a, root / profile / 'a/train/bootstrap',
                    command(m, model, exp, port, ports),
                    # One-shot is too short for a replay warm-up: a whole
                    # episode can finish with zero optimizer updates when
                    # each local buffer has not crossed train_set_size yet.
                    # Let the launcher run bounded episodes until update 1.
                    ['--phase', 'train_a', '--episodes', '200',
                     '--target-train-steps', '1'])
        verify_start(first_training_record(model, exp), expected)
        for n in p['a_points']:
            snap = root / profile / 'a/checkpoints' / str(n)
            if not snap.exists():
                train_to(base, m, root, profile, 'a', model, exp, a, n, port, ports)
                copy_checkpoint(model, snap)
            check_snapshot(snap, exp, n)
            frozen(base, m, snap, exp, root, f'{profile}_a_{n}', port, ports,
                   [a, b] if n == p['a_points'][-1] else [a])
    if scope == 'acquire':
        return
    a_end = p['a_points'][-1]
    source = root / profile / 'a/checkpoints' / str(a_end)
    check_snapshot(source, exp, a_end)
    # Both forks retain policy, target, optimizer, RNG, global epsilon and
    # their OWN full local replay. No cross-port global replay is enabled.
    for arm in ('aa', 'ab'):
        model = root / profile / arm / 'models'
        provenance = root / profile / arm / 'fork.json'
        if not model.exists():
            copy_checkpoint(source, model)
        if not provenance.exists():
            if fingerprint(model) != fingerprint(source):
                raise ValueError('Unverified existing fork does not match after-A')
            write(provenance, dict(source_hashes=fingerprint(source)))
        if read(provenance)['source_hashes'] != fingerprint(source):
            raise ValueError('Fork source changed')
    for arm, task in [('aa', a), ('ab', b)]:
        model = root / profile / arm / 'models'
        for n in p['b_points']:
            target = a_end + n
            snap = root / profile / arm / 'checkpoints' / str(n)
            if not snap.exists():
                train_to(base, m, root, profile, arm, model, exp, task, target, port, ports)
                copy_checkpoint(model, snap)
            check_snapshot(snap, exp, target)
            frozen(base, m, snap, exp, root, f'{profile}_{arm}_{n}', port, ports,
                   [a] if arm == 'aa' else [a, b])
    b_end = a_end + p['b_points'][-1]
    source = root / profile / 'ab/checkpoints' / str(p['b_points'][-1])
    model = root / profile / 'return/models'
    if not model.exists():
        copy_checkpoint(source, model)
    for n in p['return_points']:
        snap = root / profile / 'return/checkpoints' / str(n)
        if not snap.exists():
            train_to(base, m, root, profile, 'return', model, exp, a, b_end+n, port, ports)
            copy_checkpoint(model, snap)
        check_snapshot(snap, exp, b_end+n)
        frozen(base, m, snap, exp, root, f'{profile}_return_{n}', port, ports, [a, b])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['acquire', 'continue', 'all', 'analyze'], default='acquire')
    parser.add_argument('--a-points', type=points, default=points('100,300,600'))
    parser.add_argument('--b-points', type=points, default=points('100,300'))
    parser.add_argument('--return-points', type=points, default=points('50,100'))
    parser.add_argument('--port', type=int, default=7156)
    parser.add_argument('--jobs', type=int, choices=[1, 2], default=1)
    parser.add_argument('--ports', default='323,321,320,345,346,347')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--extend-a', action='store_true', help='Append A checkpoints before either continuation arm starts; preserves completed training')
    args = parser.parse_args()
    base, root = args.base_run_dir.resolve(), args.output_dir.resolve()
    if args.extend_a and (not root.exists() or args.stage!='acquire' or not args.resume):
        parser.error('--extend-a requires an existing acquire run and --resume')
    if base == root or base in root.parents or root in base.parents:
        parser.error('Use a separate sibling output directory')
    if not 1 <= args.port <= 65534:
        parser.error('Two consecutive valid socket ports required')
    ports = [int(x) for x in args.ports.split(',')]
    if not ports or min(ports) < 0 or len(ports) != len(set(ports)):
        parser.error('Invalid watched ports')
    m = read(base / 'manifest.json')
    if m.get('shared_replay', True) or m.get('epsilon_schedule', {}).get('scope') != 'global':
        parser.error('Requires local-only replay and continuous global epsilon')
    if m.get('action_space') != 'multiscale':
        parser.error('This study registers the multiscale midpoint reference action 4,3')
    p = dict(stage='learning_study', manifest=m, base=str(base), profiles=['tail_safe', 'weighted'],
             a_points=args.a_points, b_points=args.b_points, return_points=args.return_points,
             ports=ports, socket_ports=dict(tail_safe=args.port, weighted=args.port+1),
             replay_policy='fresh_A_then_full_local_replay_preserved_in_matched_forks',
             fixed_reference_action='4,3',
             task_hashes={t:{n:digest(base/'tasks'/t/n) for n in ['input.conf','input.flow']}
                          for t in [m['task_a'],m['task_b']]})
    if root.exists():
        previous = read(root/'protocol.json')
        extend = (args.extend_a and args.resume and args.stage == 'acquire' and
                  p['a_points'][:len(previous['a_points'])] == previous['a_points'] and
                  len(p['a_points']) > len(previous['a_points']) and
                  dict(previous, a_points=p['a_points']) == p)
        if (previous != p and not extend) or (args.stage in ('acquire','all') and not args.resume):
            parser.error('Existing output: identical protocol and --resume required')
    elif args.stage in ('continue','analyze'):
        parser.error('Run acquire first')
    else:
        root.mkdir(parents=True)
        write(root/'protocol.json', p)
    # OS lock releases on crash; never infer liveness from a stale PID file.
    with (root/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.extend_a:
            if not root.exists() or args.stage != 'acquire' or not args.resume:
                parser.error('--extend-a requires an existing acquire run and --resume')
            if any((root/profile/arm).exists() for profile in p['profiles'] for arm in ['aa','ab','return']):
                parser.error('Cannot extend A after continuation branches were created')
            previous=read(root/'protocol.json')
            if previous != p:
                write(root/f'protocol_before_A_{previous["a_points"][-1]}.json',previous)
                write(root/'protocol.json',p)
                (root/'acquire_complete.json').unlink(missing_ok=True)
        if args.stage != 'analyze':
            for port in p['socket_ports'].values():
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', port))
            if args.stage in ('acquire','all'):
                expected = None
                for profile in p['profiles']:
                    exp = 'study_' + root.name + '_' + profile
                    model = root/profile/'initial_models'
                    model.mkdir(parents=True, exist_ok=True)
                    if fingerprint(model):
                        raise ValueError('Initial directory contains learned state')
                    manifest = dict(m, reward_profile=profile, reward_weights=[.5,.3,.2])
                    frozen(base, manifest, model, exp, root, profile+'_a_0',
                           p['socket_ports'][profile], ports, [m['task_a']])
                    record = read(root/'eval'/f'{profile}_a_0'/m['task_a']/'repeat_1/metrics.json')
                    expected = expected or record.get('startup_network_hashes')
                    verify_start(record, expected)
                write(root/'initialization_check.json', dict(network_hashes=expected, matched=True))
                # One predeclared fixed reference, not a search for the best
                # action. Its trajectories are rescored under both rewards.
                model = root/'tail_safe/initial_models'
                exp = 'study_' + root.name + '_tail_safe'
                manifest = dict(m, reward_profile='tail_safe', reward_weights=[.5,.3,.2])
                evaluate(base, manifest, model, exp, root/'eval', 'reference_mid',
                         command(manifest, model, exp, args.port, ports) +
                         ['--force-action', p['fixed_reference_action']], 1)
            else:
                from analyze_learning_study import validate_matrix
                validate_matrix(root, 'acquire')
            scope = 'all' if args.stage in ('continue','all') else 'acquire'
            if args.jobs == 1:
                for profile in p['profiles']:
                    worker(str(root), profile, scope)
            else:
                # Two independent profile workers. No parallel runs share an
                # experiment name, log path, model directory or socket port.
                with ProcessPoolExecutor(max_workers=2) as pool:
                    futures = [pool.submit(worker, str(root), profile, scope) for profile in p['profiles']]
                    for future in futures:
                        future.result()
            write(root/(scope+'_complete.json'), dict(complete=True))
        else:
            scope = 'all' if (root/'all_complete.json').exists() else 'acquire'
        from analyze_learning_study import analyze_study
        analyze_study(root, scope)


if __name__ == '__main__':
    main()
