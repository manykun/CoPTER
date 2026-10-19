"""Reward formulas, telemetry isolation and native queue integral fixtures."""
import bisect
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from acc_rewards import calculate_acc_reward, parse_reward_interval

ROOT = Path(__file__).resolve().parents[1]


class IntervalRewardTests(unittest.TestCase):
    def test_history_matches_git_formula_with_original_normalization(self):
        rng = random.Random(1)
        for queue in [0, 400, 800, 1600, 409600, 800000] + [rng.random()*409600 for _ in range(100)]:
            normalized = min(1., queue / 409600)
            score = [1, .9, .8, .7, .6, .5, .4, .3, .2, .1, 0][bisect.bisect_left([2**i/1024 for i in range(11)], normalized)]
            self.assertAlmostEqual(calculate_acc_reward("acc_3816bc6_aligned", queue, .8)["reward"], .4 + .5*score)

    def test_paper_boundaries_and_high_queue_policy(self):
        self.assertEqual(calculate_acc_reward("acc_paper", 20*1024, .5)["reward_queue_bin"], 0)
        self.assertEqual(calculate_acc_reward("acc_paper", 20*1024+1, .5)["reward_queue_bin"], 1)
        self.assertAlmostEqual(calculate_acc_reward("acc_paper", 10**9, .5)["queue"], .1)

    def test_missing_duplicate_overlap_and_gap_telemetry_fail(self):
        with self.assertRaises(RuntimeError):
            parse_reward_interval({}, 1)
        def record(start, end):
            return {"reward_interval": {"version": 1, "start_ns": start, "end_ns": end, "mean_queue_bytes": [2048], "tx_utilization": [.7]}}
        interval = parse_reward_interval(json.dumps(record(100, 600)), 1)
        self.assertEqual(parse_reward_interval(record(600, 1100), 1, interval["end_ns"])["start_ns"], 600)
        for start, end in ((100, 600), (599, 1100), (601, 1100), (600, 600)):
            with self.assertRaises(RuntimeError):
                parse_reward_interval(record(start, end), 1, 600)

    def test_reward_uses_interval_mean_not_old_peak_or_history(self):
        # The terminal tests install ns3gym/loguru stubs without requiring a simulator.
        import test_network_helper_terminal
        from network_helper import NetworkHelper
        from structures import NetworkHelperParameters
        from collections import deque
        for profile in ("acc_paper", "acc_3816bc6_aligned"):
            helper = NetworkHelper.__new__(NetworkHelper)
            helper.nhp = NetworkHelperParameters(reward_profile=profile, reward_window_size=1)
            for field in ("reward_qlen_window", "reward_ecn_window", "reward_txrate_window"):
                setattr(helper, field, [deque([.9, .8])])
            helper.reward_interval = {"start_ns": 0, "end_ns": 500000, "mean_queue_bytes": [2048], "tx_utilization": [.7]}
            result = helper.get_port_current_reward_components(0)
            self.assertAlmostEqual(result["reward"], calculate_acc_reward(profile, 2048, .7)["reward"])
            self.assertAlmostEqual(result["avg_queue"], 2048/409600)

    def test_native_integral_counts_idle_time_and_resets_backlog(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if not compiler:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "queue_fixture"
            subprocess.run([compiler, "-std=c++11", "-I", str(ROOT / "ns-3.33/src/point-to-point/model"), str(ROOT / "copter/test_action_queue_integral.cc"), "-o", str(executable)], check=True)
            subprocess.run([str(executable)], check=True)


if __name__ == "__main__":
    unittest.main()
