#!/usr/bin/env python3
"""Fresh paired ACC A->B training; only the reward profile differs."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path

from run_acc_attribution import command, execute, evaluate
from run_return_a import read, write, digest
from analyze_acc_attribution import analyze


def first_training_record(model, exp):
    path = model / f'{exp}_metrics.jsonl'
    with path.open() as handle:
        for line in handle:
            r = json.loads(line)
            if not r.get('eval_greedy'):
                return r
    raise ValueError('No training records')


def verify_start(record, expected):
    if record.get('startup_train_step') != 0 or record.get('startup_replay_entries') != 0:
        raise ValueError('Reward ablation must start with fresh counters and EMPTY replay')
    hashes = record.get('startup_network_hashes')
    if not hashes or hashes != expected:
        raise ValueError('Initial networks differ: refusing unmatched reward comparison')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run-dir', type=Path, required=True, help='Only manifest/task inputs are reused, never trained models')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--updates-per-task', type=int, default=600)
    p.add_argument('--port', type=int, default=7056)
    p.add_argument('--ports', default='323,321,320,345,346,347')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stage', choices=['run', 'analyze'], default='run')
    args = p.parse_args()
    base, out = args.base_run_dir.resolve(), args.output_dir.resolve()
    m = read(base / 'manifest.json')
    ports = [int(x) for x in args.ports.split(',')]
    if args.updates_per_task <= 0 or not 1 <= args.port <= 65535:
        p.error('Invalid budget or port')
    if not ports or min(ports) < 0 or len(ports) != len(set(ports)):
        p.error('Watched ports must be unique nonnegative integers')
    if out == base or out in base.parents or base in out.parents:
        p.error('Use a separate sibling directory')
    if m.get('shared_replay', True) or m.get('epsilon_schedule', {}).get('scope') != 'global':
        raise ValueError('Source must use local-only replay and global epsilon')
    if m.get('reward_weights', [.5,.3,.2]) != [.5,.3,.2]:
        raise ValueError('This registered comparison uses original weighted coefficients .50,.30,.20')
    tasks = [m['task_a'], m['task_b']]
    protocol = dict(stage='reward_ablation', manifest=m, base=str(base), ports=ports,
                    updates_per_task=args.updates_per_task, socket_port=args.port,
                    profiles=['tail_safe', 'weighted'], repeats=1,
                    replay_policy='fresh_empty_then_preserved_within_each_profile',
                    task_hashes={t:{n:digest(base/'tasks'/t/n) for n in ['input.conf','input.flow']} for t in tasks})
    if out.exists():
        if (not args.resume and args.stage == 'run') or read(out/'protocol.json') != protocol:
            raise ValueError('Existing output: use identical arguments with --resume, or a new directory')
    elif args.stage == 'analyze':
        raise ValueError('Missing experiment')
    else:
        out.mkdir(parents=True)
        write(out/'protocol.json', protocol)
    if args.stage == 'run':
        initial_hashes = None
        # Run both fresh initial evaluations BEFORE either expensive training.
        for profile in protocol['profiles']:
            manifest = dict(m, reward_profile=profile, reward_weights=[.5,.3,.2])
            exp = f'reward_{out.name}_{profile}'
            model = out / profile / 'initial_models'
            model.mkdir(parents=True, exist_ok=True)
            if list(model.glob('*_ACC_*')) or list(model.glob('*.pkl')) or list(model.glob('*_train_state.json')):
                raise ValueError('Initial evaluation directory must not contain learned state')
            cmd = command(manifest, model, exp, args.port, ports)
            evaluate(base, manifest, model, exp, out/'eval', profile+'_initial', cmd, 1)
            for task in tasks:
                r = read(out/'eval'/f'{profile}_initial'/task/'repeat_1/metrics.json')
                if initial_hashes is None:
                    initial_hashes = r.get('startup_network_hashes')
                verify_start(r, initial_hashes)
        write(out/'initialization_check.json', dict(matched=True, network_hashes=initial_hashes,
              seed=m['seed'], replay_entries=0, global_train_step=0))
        for profile in protocol['profiles']:
            manifest = dict(m, reward_profile=profile, reward_weights=[.5,.3,.2])
            exp = f'reward_{out.name}_{profile}'
            model = out/profile/'models'
            model.mkdir(parents=True, exist_ok=True)
            state_path = model / f'{exp}_train_state.json'
            cmd = command(manifest, model, exp, args.port, ports)
            if not state_path.exists():
                # Validate actual training initialization after ONE epoch,
                # rather than discovering a mismatch after 600 updates.
                if list(model.iterdir()):
                    raise ValueError('Model directory contains files but no valid state; inspect interrupted bootstrap')
                execute(base, tasks[0], out/profile/'train/bootstrap', cmd,
                        ['--phase', 'train_a', '--one-shot', '--episodes', '1',
                         '--target-train-steps', str(args.updates_per_task)])
            verify_start(first_training_record(model, exp), initial_hashes)
            for index, (phase, task) in enumerate(zip(['after_a', 'after_b'], tasks), 1):
                target = args.updates_per_task * index
                snap = out/profile/'checkpoints'/phase
                if not snap.exists():
                    current = read(state_path) if state_path.exists() else dict(global_train_step=0, epoch=0)
                    if current['global_train_step'] > target:
                        raise ValueError('Missing earlier checkpoint; cannot safely resume')
                    if current['global_train_step'] < target:
                        execute(base, task, out/profile/'train'/phase, cmd,
                            ['--phase', 'train_a' if index == 1 else 'train_b',
                             '--episodes', str(current['epoch']+1000), '--target-train-steps', str(target)])
                    if read(state_path)['global_train_step'] != target:
                        raise ValueError('Exact training budget not reached')
                    verify_start(first_training_record(model, exp), initial_hashes)
                    # Keep complete replay snapshots for future matched controls.
                    snap.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(prefix='.snapshot-', dir=snap.parent) as tmp:
                        staging = Path(tmp)/'data'
                        shutil.copytree(model, staging, ignore=shutil.ignore_patterns('*_metrics.jsonl'))
                        staging.rename(snap)
                frozen_cmd = command(manifest, snap, exp, args.port, ports)
                evaluate(base, manifest, snap, exp, out/'eval', profile+'_'+phase, frozen_cmd, 1)
    for profile in protocol['profiles']:
        for phase in ['initial','after_a','after_b']:
            for task in tasks:
                if not (out/'eval'/f'{profile}_{phase}'/task/'repeat_1/complete.json').exists():
                    raise ValueError('Incomplete reward comparison; resume before analysis')
    analyze(out, out/'attribution_report', ports)
    print('Do not compare native rewards across profiles. Rescore both trajectories with rescore_rewards.py.')


if __name__ == '__main__':
    main()
