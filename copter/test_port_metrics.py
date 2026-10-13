#!/usr/bin/env python3
"""Unit tests for fixed-port rollout measurement."""

import json
import tempfile
import unittest
from pathlib import Path

from port_metrics import (
    PortMetricTracker,
    parse_forced_port_action,
    parse_watch_ports,
)


class FakeNetwork:
    port_identifier_map = {1: "288-295"}

    def get_port_current_reward_components(self, port):
        return {
            "reward": 0.5,
            "throughput": 0.2,
            "queue": 0.8,
            "ecn": 0.6,
            "avg_tx_rate": 0.2,
            "avg_queue": 0.1,
            "peak_queue": 0.2,
            "avg_ecn": 0.1,
            "peak_ecn": 0.2,
            "combined_queue": 0.15,
            "combined_ecn": 0.15,
            "queue_cost_sq": 0.03,
            "ecn_cost_sq": 0.04,
            "tail_safe_raw": -0.25,
            "tail_safe_clipped": -0.25,
        }

    def is_port_active(self, port):
        return True

    def is_port_congested(self, port):
        return True


class PortMetricTrackerTests(unittest.TestCase):
    def test_parse_and_validation(self):
        self.assertEqual(parse_watch_ports("1, 3"), [1, 3])
        with self.assertRaises(ValueError):
            parse_watch_ports("1,1")
        with self.assertRaises(ValueError):
            PortMetricTracker([4]).validate(4)
        self.assertEqual(
            parse_forced_port_action("323,0.5,0.75,0.2"),
            (323, 0.5, 0.75, 0.2),
        )
        with self.assertRaises(ValueError):
            parse_forced_port_action("323,0.5,1.2,0.2")

    def test_summary_and_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            trace = Path(temporary) / "trace.jsonl"
            tracker = PortMetricTracker([1], trace)
            tracker.validate(2)
            tracker.observe(5, [(0, 0, 0), (2, 1, 4)], FakeNetwork())
            summary = tracker.summary()["1"]
            self.assertEqual(summary["identifier"], "288-295")
            self.assertEqual(summary["active_steps"], 1)
            self.assertEqual(summary["congested_steps"], 1)
            self.assertAlmostEqual(summary["means_congested"]["reward"], 0.5)
            self.assertAlmostEqual(
                summary["means_congested"]["tail_safe_raw"], -0.25
            )
            self.assertEqual(summary["actions_congested"], {"2,1,4": 1})
            self.assertEqual(tracker.legacy_rewards(), {"1": 0.5})
            tracker.write_trace()
            record = json.loads(trace.read_text(encoding="utf-8"))
            self.assertEqual(record["port"], 1)
            self.assertEqual(record["action"], "2,1,4")


if __name__ == "__main__":
    unittest.main()
