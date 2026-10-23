#!/usr/bin/env python3
"""Fast fixtures for the registered queue-penalty acquisition screen."""

import unittest

from scripts.continual_validation.run_acc_reward_action_pilot import build_config
from tools.analysis.analyze_acc_queue_decision import (
    decision,
    port349_fields,
    validate_pair,
)


class QueueDecisionTest(unittest.TestCase):
    def test_only_queue_penalty_changes(self):
        base = {
            "route": "CDC", "watch_ports": "320,321,323,345,346,347",
            "reward_queue_lambda": 5.0, "reward_ecn_lambda": 5.0,
        }
        control = build_config(base, "control", "credit_control", 50, 7800, 1)
        candidate = build_config(base, "candidate", "credit_queue1", 50, 7801, 1)
        self.assertEqual(control["watch_ports"], "320,321,323,345,346,347,349")
        self.assertEqual(candidate["watch_ports"], control["watch_ports"])
        self.assertEqual(control["reward_queue_lambda"], 5.0)
        self.assertEqual(candidate["reward_queue_lambda"], 1.0)
        excluded = {"run_id", "acc_port", "sor_port", "reward_queue_lambda"}
        self.assertEqual(
            {key: value for key, value in control.items() if key not in excluded},
            {key: value for key, value in candidate.items() if key not in excluded},
        )

    def test_pair_rejects_confounds(self):
        common = {
            "tasks": {"a": "realistic_hadoop"}, "action_space": "multiscale",
            "reward_profile": "tail_safe", "reward_ecn_lambda": 5.0,
            "reward_window_size": 4, "replay_sampling": "congestion_stratified",
            "diagnostic_reference_action": "8,0",
        }
        arms = {
            "credit_control": dict(common, reward_queue_lambda=5.0),
            "credit_queue1": dict(common, reward_queue_lambda=1.0),
        }
        validate_pair(arms)
        arms["credit_queue1"]["reward_window_size"] = 32
        with self.assertRaisesRegex(ValueError, "outside queue lambda"):
            validate_pair(arms)

    def test_port349_q_and_actions(self):
        agent = {
            "watch_ports_q_ranking": {"349": {
                "samples": 4, "reference_greedy_fraction": 0.25,
                "reference_margin_mean": -0.1,
            }},
            "watch_ports_metrics": {"349": {
                "congested_steps": 4, "actions_congested": {"8,0": 1, "2,3": 3},
                "means_congested": {
                    "throughput": 0.7, "queue_cost_sq": 0.01,
                    "ecn_cost_sq": 0.001, "reward": 0.645,
                },
            }},
        }
        result = port349_fields(agent)
        self.assertEqual(result["port349_action_8_0_fraction"], 0.25)
        self.assertEqual(result["port349_congested_steps"], 4)
        self.assertEqual(result["port349_queue_cost_sq"], 0.01)

    def test_decision_is_relative_to_matched_control_and_own_initial(self):
        rows = {
            ("credit_queue1", "frozen_initial"): {"mean_fct_us": 130},
            ("credit_queue1", "frozen_final"): {
                "mean_fct_us": 120, "completion_ratio": 0.985,
                "p95_fct_us": 450,
            },
            ("credit_control", "frozen_final"): {
                "mean_fct_us": 129, "completion_ratio": 0.9855,
                "p95_fct_us": 460,
            },
        }
        result = decision(rows)
        self.assertLess(result["candidate_vs_control_mean_fct_percent"], -3)
        self.assertAlmostEqual(result["completion_cost_pp"], 0.05)


if __name__ == "__main__":
    unittest.main()
