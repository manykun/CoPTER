"""Simulator-free integration tests; these do not validate NS-3 performance."""
import json
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import run_learning_study as app
import run_acc_attribution as runtime
import analyze_learning_study as report
import test_acc_attribution as fixtures


class StudyTests(unittest.TestCase):
    def test_math(self):
        self.assertEqual(report.tvd({'a':10},{'a':20}),0)
        self.assertEqual(report.tvd({'a':10},{'b':10}),1)
        self.assertIsNone(report.tvd({}, {'b':10}))
        self.assertIsNone(report.recovery(1,2,1.5))
        self.assertIsNone(report.recovery(1,1,1,True))
        self.assertAlmostEqual(report.recovery(100,110,105,True),.5)
        self.assertAlmostEqual(report.recovery(.8,.6,.9),1.5)
        self.assertIsNone(report.relative(0,1))

    def test_points(self):
        for value in ['0','2,1','1,1']:
            with self.assertRaises(Exception):
                app.points(value)
        self.assertEqual(app.points('1,2'),[1,2])

    def test_full_study_resume_and_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); base=fixtures.AttributionTests().fixture(root); out=root/'study'
            calls=[]; hashes={'policy_net':'initial','target_net':'initial'}
            def fake(cmd,**kwargs):
                def arg(flag): return cmd[cmd.index(flag)+1]
                calls.append(cmd)
                model,exp=Path(arg('--model-dir')),arg('--exp')
                state_path=model/(exp+'_train_state.json')
                s=app.read(state_path) if state_path.exists() else dict(global_train_step=0,epoch=0,node_number=1)
                start=s['global_train_step']; frozen='--eval-greedy' in cmd
                if not frozen:
                    target=int(arg('--target-train-steps'))
                    s.update(global_train_step=target,epoch=s['epoch']+1)
                    app.write(state_path,s)
                    (model/(exp+'_ACC_0')).write_bytes(b'weights')
                    (model/(exp+'_rb_port0.pkl')).write_bytes(b'replay')
                dest=Path(arg('--config')).parent
                (dest/'result.fct').write_text('1 2 3 4 5 6 1000 100\n')
                detail=dict(samples=10,active_steps=10,congested_steps=10,actions_all={'4,3':10})
                means=dict(reward=.5,throughput=.5,queue=.5,ecn=.5,tail_safe_clipped=.5)
                for pop in ['all','active','congested']: detail['means_'+pop]=means
                row=dict(eval_greedy=frozen,reward_profile=arg('--reward-profile'),
                         rollout_all_congested_mean=.5,global_train_step=s['global_train_step'],
                         startup_train_step=start,startup_replay_entries=0 if start==0 else 10,
                         startup_network_hashes=hashes if start==0 else {'policy_net':'trained'},
                         watch_ports_metrics={'0':detail},phase=arg('--phase'),epoch=s['epoch'],
                         train_port_metrics={'0':dict(updates=3,batch_reward_mean=.5,loss_mean=.01)},
                         reward_throughput_mean=.5,reward_queue_mean=.5,reward_ecn_mean=.5,
                         reward_tail_safe_clipped_mean=.5)
                with (model/(exp+'_metrics.jsonl')).open('a') as f: f.write(json.dumps(row)+'\n')
            argv=['run','--base-run-dir',str(base),'--output-dir',str(out),'--ports','0',
                  '--a-points','7,14','--b-points','7,14','--return-points','7,14','--jobs','2']
            def invoke(extra):
                with patch.object(sys,'argv',argv+extra),patch.object(runtime.subprocess,'run',fake),patch.object(app.socket,'socket'),patch.object(app,'ProcessPoolExecutor',ThreadPoolExecutor),patch.dict(sys.modules,{'matplotlib':None}):
                    app.main()
            invoke(['--stage','acquire'])
            self.assertEqual(sum('--eval-greedy' in c for c in calls),10)
            self.assertFalse((out/'all_complete.json').exists())
            with self.assertRaises(ValueError): report.validate_matrix(out,'all')
            argv[argv.index('--a-points')+1]='7,14,21'
            invoke(['--stage','acquire','--resume','--extend-a'])
            self.assertEqual(sum('--eval-greedy' in c for c in calls),14)
            self.assertTrue((out/'protocol_before_A_14.json').exists())
            invoke(['--stage','continue'])
            self.assertEqual(sum('--eval-greedy' in c for c in calls),34)
            for profile in ['tail_safe','weighted']:
                exp='study_'+out.name+'_'+profile
                aa=app.read(out/profile/'aa/fork.json')
                ab=app.read(out/profile/'ab/fork.json')
                self.assertEqual(aa,ab)
                self.assertEqual(app.state(out/profile/'return/models',exp)['global_train_step'],49)
                self.assertTrue((out/profile/'return/checkpoints/14'/(exp+'_rb_port0.pkl')).exists())
            count=len(calls); invoke(['--stage','all','--resume'])
            self.assertEqual(len(calls),count)
            self.assertTrue((out/'study_report/action_changes.csv').exists())
            self.assertIn('return_A_residual', (out/'study_report/REPORT.md').read_text())
            rows=(out/'study_report/comparisons.csv').read_text()
            self.assertIn('weighted_score_up_with_p95_worse',rows)
            self.assertEqual(len({c[1] for c in calls if c[0]=='env'}),
                             len({c[c.index('--model-dir')+1] for c in calls}))
            # Changed traffic or deleted replay must not silently resume.
            snap=out/'tail_safe/a/checkpoints/21'
            next(snap.glob('*rb_port0.pkl')).unlink()
            with self.assertRaises(ValueError): invoke(['--stage','continue'])


if __name__=='__main__': unittest.main()
