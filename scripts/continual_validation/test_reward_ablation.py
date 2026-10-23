import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import run_reward_ablation as app
import run_acc_attribution as runtime
from test_acc_attribution import AttributionTests
from rescore_rewards import score


class RewardTests(unittest.TestCase):
    def test_rescore_same_population(self):
        row = dict(throughput='.4', queue='.8', ecn='.6', reward='-.2', tail_safe_raw='-.3')
        result = score(row, [.5,.3,.2], 'tail_safe')
        self.assertAlmostEqual(result['weighted_score'], .56)
        self.assertEqual(result['tail_safe_score'], -.2)

    def test_no_clip_of_mean(self):
        row = dict(throughput=.4, queue=.8, ecn=.6, reward=.56, tail_safe_raw=-2)
        self.assertIsNone(score(row, [.5,.3,.2], 'weighted')['tail_safe_score'])
        row['tail_safe_clipped'] = -.4
        self.assertEqual(score(row, [.5,.3,.2], 'weighted')['tail_safe_score'], -.4)

    def test_initialization_checks(self):
        expected = dict(policy_net='same', target_net='same')
        record = dict(startup_train_step=0, startup_replay_entries=0, startup_network_hashes=expected)
        app.verify_start(record, expected)
        for changed in [dict(record, startup_train_step=600), dict(record, startup_replay_entries=1),
                        dict(record, startup_network_hashes={'policy_net':'different'})]:
            with self.assertRaises(ValueError):
                app.verify_start(changed, expected)

    def test_fresh_paired_curriculum_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = AttributionTests().fixture(root)
            out = root/'reward_pair'
            calls = []
            expected = dict(policy_net='same', target_net='same')
            def fake(cmd, **kwargs):
                def arg(k):
                    return cmd[cmd.index(k)+1]
                calls.append(cmd)
                model, exp = Path(arg('--model-dir')), arg('--exp')
                state_path = model/f'{exp}_train_state.json'
                state = app.read(state_path) if state_path.exists() else dict(global_train_step=0, epoch=0, node_number=1)
                start = state['global_train_step']
                frozen = '--eval-greedy' in cmd
                if not frozen:
                    target = int(arg('--target-train-steps'))
                    state['global_train_step'] = min(start+7, target) if '--one-shot' in cmd else target
                    state['epoch'] += 1
                    app.write(state_path, state)
                    (model/f'{exp}_ACC_0').write_bytes(b'trained')
                    (model/f'{exp}_rb_port0.pkl').write_text(arg('--reward-profile'))
                dest = Path(arg('--config')).parent
                (dest/'result.fct').write_text('1 2 3 4 5 6 1000 100\n')
                row = dict(eval_greedy=frozen, reward_profile=arg('--reward-profile'),
                    rollout_all_congested_mean=.5, startup_train_step=start,
                    startup_replay_entries=0 if start == 0 else 10,
                    startup_network_hashes=expected if start == 0 else {'policy_net':'trained'},
                    global_train_step=state['global_train_step'])
                with (model/f'{exp}_metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
            argv = ['run', '--base-run-dir', str(base), '--output-dir', str(out),
                    '--updates-per-task', '20', '--ports', '0']
            with patch.object(sys, 'argv', argv), patch.object(runtime.subprocess, 'run', fake), patch.object(app, 'analyze'):
                app.main()
            self.assertEqual(sum('--eval-greedy' in c for c in calls), 12)
            for profile in ['weighted','tail_safe']:
                exp=f'reward_{out.name}_{profile}'
                for phase, step in [('after_a',20),('after_b',40)]:
                    snap=out/profile/'checkpoints'/phase
                    self.assertEqual(app.read(snap/f'{exp}_train_state.json')['global_train_step'],step)
                    self.assertEqual((snap/f'{exp}_rb_port0.pkl').read_text(),profile)
            count=len(calls)
            with patch.object(sys,'argv',argv+['--resume']), patch.object(runtime.subprocess,'run',fake), patch.object(app,'analyze'):
                app.main()
            self.assertEqual(len(calls),count)


if __name__ == '__main__':
    unittest.main()
