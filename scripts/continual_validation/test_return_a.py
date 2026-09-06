"""Mock simulator integration: exact budgets, isolated outputs, frozen evaluations."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_return_a as app


class RecoveryTest(unittest.TestCase):
    def test_curriculum(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, out = Path(tmp) / 'base', Path(tmp) / 'recovery'
            snapshot = base / 'acc/checkpoints/after_b'
            snapshot.mkdir(parents=True)
            manifest = dict(task_a='A', task_b='B', seed=1, buffer_kb=400,
                shared_replay=False, epsilon_schedule=dict(scope='global', start=1, end=.05, decay_steps=2500),
                action_space='multiscale', hidden_dims='32,64,64,32', reward_profile='tail_safe',
                reward_queue_lambda=5, reward_ecn_lambda=5, target_update_interval=100)
            app.write(base / 'manifest.json', manifest)
            app.write(snapshot / 'old_train_state.json', dict(global_train_step=1200, epoch=172, epsilon=.05, node_number=1))
            (snapshot / 'old_rb_port0.pkl').write_bytes(b'replay')
            (snapshot / 'old_ACC_0').write_bytes(b'weights')
            for task in ['A', 'B']:
                directory = base / 'tasks' / task
                directory.mkdir(parents=True)
                (directory / 'input.flow').write_text('2\n')
                (directory / 'input.conf').write_text('FLOW_FILE old\nFCT_OUTPUT_FILE old\nSIMULATOR_STOP_TIME 4\n')
            calls = []

            def fake_run(cmd, **kwargs):
                def arg(flag):
                    return cmd[cmd.index(flag)+1]
                model = Path(arg('--model-dir'))
                state = model / (arg('--exp') + '_train_state.json')
                calls.append(cmd)
                if '--eval-greedy' not in cmd:
                    value = app.read(state)
                    value['global_train_step'] = int(arg('--target-train-steps'))
                    value['epoch'] += 3
                    app.write(state, value)
                else:
                    directory = Path(arg('--config')).parent
                    (directory / 'result.fct').write_text('1 2 3 4 5 6 1000 100\n')
                (model / (arg('--exp') + '_metrics.jsonl')).write_text('{"rollout_all_congested_mean":0.5}\n')

            original = (snapshot / 'old_train_state.json').read_bytes()
            argv = ['run', '--base-run-dir', str(base), '--output-dir', str(out)]
            with patch.object(sys, 'argv', argv), patch.object(app.subprocess, 'run', fake_run), patch.object(app, 'analyze') as analysis:
                app.main()
                analysis.assert_called_once()
            self.assertEqual(original, (snapshot / 'old_train_state.json').read_bytes())
            targets = [int(c[c.index('--target-train-steps')+1]) for c in calls if '--target-train-steps' in c]
            self.assertEqual(targets, [1220, 1250, 1300])
            self.assertEqual(sum('--eval-greedy' in c for c in calls), 8)
            self.assertEqual((out / 'models/return_recovery_rb_port0.pkl').read_bytes(), b'replay')
            for task in ['A', 'B']:
                import shutil
                shutil.copytree(out / 'eval/return_0' / task, base / 'eval/acc/after_a' / task)
            app.analyze(out, base, manifest, [20, 50, 100], [0])
            result = app.read(out / 'analysis.json')
            self.assertEqual(len(result['network']), 10)
            self.assertTrue(all(r['matched_flows'] == 1 for r in result['network']))
            self.assertTrue(all(r['completion_ratio'] == .5 for r in result['network']))
            with patch.object(sys, 'argv', argv):
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    app.main()


if __name__ == '__main__':
    unittest.main()
