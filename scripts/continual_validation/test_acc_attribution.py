"""Fast simulator-free tests for attribution isolation and data semantics."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_acc_attribution as run
import analyze_acc_attribution as analysis


class AttributionTests(unittest.TestCase):
    def fixture(self, root):
        base = root / 'base'
        base.mkdir()
        manifest = dict(task_a='A', task_b='B', seed=1, buffer_kb=400, shared_replay=False,
            epsilon_schedule=dict(scope='global', start=1, end=.05, decay_steps=2500),
            action_space='multiscale', hidden_dims='32,64,64,32', reward_profile='tail_safe',
            reward_queue_lambda=5, reward_ecn_lambda=5, target_update_interval=100)
        run.write(base / 'manifest.json', manifest)
        for phase, step in [('after_a', 600), ('after_b', 1200)]:
            snap = base / 'acc/checkpoints' / phase
            snap.mkdir(parents=True)
            run.write(snap / 'old_train_state.json', dict(global_train_step=step,
                      global_env_step=4800, epoch=86, epsilon=.05, node_number=1))
            (snap / 'old_ACC_0').write_bytes(b'same-weights')
        for task in ['A', 'B']:
            path = base / 'tasks' / task
            path.mkdir(parents=True)
            (path / 'input.flow').write_text('2\n')
            (path / 'input.conf').write_text('FLOW_FILE old\nFCT_OUTPUT_FILE old\nSIMULATOR_STOP_TIME 4\n')
        return base

    def fake_run(self, cmd, **kwargs):
        def arg(flag):
            return cmd[cmd.index(flag) + 1]
        model, exp = Path(arg('--model-dir')), arg('--exp')
        state_path = model / (exp + '_train_state.json')
        state = run.read(state_path)
        if '--eval-greedy' not in cmd:
            self.targets.append((arg('--phase'), int(arg('--target-train-steps'))))
            state['global_train_step'] = int(arg('--target-train-steps'))
            state['epoch'] += 1
            run.write(state_path, state)
        else:
            self.evals += 1
            path = Path(arg('--config')).parent
            (path / 'result.fct').write_text('1 2 3 4 5 6 1000 100\n')
        row = dict(eval_greedy='--eval-greedy' in cmd, rollout_all_congested_mean=.5,
                   global_train_step=state['global_train_step'], watch_ports_metrics={})
        with (model / (exp + '_metrics.jsonl')).open('a') as handle:
            handle.write(json.dumps(row) + '\n')

    def invoke(self, argv):
        with patch.object(sys, 'argv', ['run'] + argv), patch.object(run.subprocess, 'run', self.fake_run):
            run.main()

    def test_repeats_and_resume_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.fixture(Path(tmp))
            out = Path(tmp) / 'repeat'
            self.targets, self.evals = [], 0
            before = run.fingerprint(base / 'acc/checkpoints/after_a')
            args = ['--base-run-dir', str(base), '--output-dir', str(out), '--stage', 'repeat', '--ports', '0', '--repeats', '2']
            self.invoke(args)
            self.assertEqual(self.evals, 8)
            self.assertEqual(self.targets, [])
            self.assertEqual(before, run.fingerprint(base / 'acc/checkpoints/after_a'))
            self.invoke(args + ['--resume'])
            self.assertEqual(self.evals, 8)
            # CSVs also work with no matplotlib installation.
            with patch.dict(sys.modules, {'matplotlib': None}):
                analysis.analyze(out, out / 'report', [0])
            data = run.read(out / 'report/analysis.json')
            self.assertTrue(all(r['completion_ratio'] == .5 for r in data['network']))
            self.assertTrue(all(r['matched_flows'] == 1 for r in data['network']))

    def test_control_same_budget_and_explicit_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.fixture(Path(tmp))
            out = Path(tmp) / 'control'
            self.targets, self.evals = [], 0
            args = ['--base-run-dir', str(base), '--output-dir', str(out), '--stage', 'control',
                    '--ports', '0', '--repeats', '2', '--updates', '20,100']
            with self.assertRaises(ValueError):
                self.invoke(args)
            self.assertFalse(out.exists())
            args += ['--replay-policy', 'reset']
            self.invoke(args)
            self.assertEqual(self.targets, [('control_aa', 620), ('control_aa', 700),
                                            ('control_ab', 620), ('control_ab', 700)])
            self.assertEqual(self.evals, 24)
            self.invoke(args + ['--resume'])
            self.assertEqual(self.evals, 24)
            with patch.dict(sys.modules, {'matplotlib': None}):
                analysis.analyze(out, out / 'report', [0])
            result = run.read(out / 'report/analysis.json')
            self.assertEqual(len(result['control_effects']), 4)
            self.assertTrue(all(r['p95_fct_us_ab_minus_aa_delta'] == 0 for r in result['control_effects']))
            # A/B inputs and after-A source remain untouched.
            self.assertEqual(run.read(base / 'acc/checkpoints/after_a/old_train_state.json')['global_train_step'], 600)

    def test_components_use_mean_cost_not_squared_mean(self):
        result = analysis.contributions(dict(throughput=.8, queue_cost_sq=.1,
            ecn_cost_sq=.02, tail_safe_raw=.2, reward=.25),
            dict(reward_profile='tail_safe', reward_queue_lambda=5, reward_ecn_lambda=5))
        self.assertAlmostEqual(result['raw_reconstructed'], .2)
        self.assertAlmostEqual(result['clipping_mean_difference'], .05)

    def test_difference_in_differences_sign(self):
        rows = [dict(label=label, task='A', repeat=1, reward=reward,
                     p95_fct_us=fct, completion_ratio=.9)
                for label, reward, fct in [('aa_0', .5, 100), ('aa_100', .51, 99),
                                           ('ab_0', .5, 100), ('ab_100', .48, 105)]]
        effect = analysis.control_effects(rows, 'A', [100])[0]
        self.assertAlmostEqual(effect['reward_ab_minus_aa_delta'], -.03)
        self.assertEqual(effect['p95_fct_us_ab_minus_aa_delta'], 6)

    def test_frozen_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.fixture(Path(tmp))
            out = Path(tmp) / 'repeat'
            self.targets, self.evals = [], 0
            def mutating(cmd, **kwargs):
                self.fake_run(cmd, **kwargs)
                model = Path(cmd[cmd.index('--model-dir') + 1])
                (next(model.glob('*_ACC_0'))).write_bytes(b'bad update')
            argv = ['run', '--base-run-dir', str(base), '--output-dir', str(out),
                    '--stage', 'repeat', '--ports', '0', '--repeats', '2']
            with patch.object(sys, 'argv', argv), patch.object(run.subprocess, 'run', mutating):
                with self.assertRaisesRegex(ValueError, 'Frozen evaluation modified'):
                    run.main()

    def test_incomplete_repeat_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = self.fixture(Path(tmp))
            out = Path(tmp) / 'repeat'
            self.targets, self.evals = [], 0
            self.invoke(['--base-run-dir', str(base), '--output-dir', str(out),
                         '--stage', 'repeat', '--ports', '0', '--repeats', '2'])
            (out / 'eval/after_b/B/repeat_2/complete.json').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete frozen evaluation'):
                analysis.analyze(out, out / 'report', [0])


if __name__ == '__main__':
    unittest.main()
