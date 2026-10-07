import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.analysis.analyze_acc_pilot_port_activity import (
    analyze_trial,
    preceding_rate,
    rate_index,
    summarize,
    topology_switches,
)


class AccPilotPortActivityTest(unittest.TestCase):
    def test_temporal_join_uses_same_port_and_recent_rate(self):
        records = [
            {"switch_id": 288, "port_id": 295, "time_s": 2.0, "tx_rate": 0.1},
            {"switch_id": 288, "port_id": 295, "time_s": 3.0, "tx_rate": 0.2},
        ]
        index = rate_index(records)
        self.assertEqual(preceding_rate(index, (288, 295), 2.5)["tx_rate"], 0.1)
        self.assertIsNone(preceding_rate(index, (288, 294), 2.5))
        self.assertIsNone(preceding_rate(index, (288, 295), 4.5))

    def test_fixed_action_active_switch_port_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            (path / "complete.json").write_text("{}\n", encoding="utf-8")
            (path / "input.flow").write_text("0\n", encoding="utf-8")
            (path / "input.conf").write_text(
                "TOPOLOGY_FILE mix/256hosts.topo\n"
                "OPENGYM_MIN_KMIN 5000\nOPENGYM_MAX_KMIN 50000\n"
                "OPENGYM_MIN_KMAX 15000\nOPENGYM_MAX_KMAX 100000\n",
                encoding="utf-8",
            )
            (path / "result.rate").write_text(
                "288 295 40000000000 0 0 0\n"
                "288 295 40000000000 0 0 1\n"
                "288 295 40000000000 0.06 0.1 2\n"
                "288 295 40000000000 0.02 0.0 3\n",
                encoding="utf-8",
            )
            (path / "result.queue").write_text(
                "288 409600 295 4000 2.1\n"
                "288 409600 295 16000 2.2\n"
                "288 409600 295 32000 2.3\n"
                "288 409600 295 4000 4.5\n",
                encoding="utf-8",
            )
            (path / "result.throughput").write_text(
                "288 295 2000000000 2.0 40000000000\n",
                encoding="utf-8",
            )
            (path / "agent_metrics.json").write_text(json.dumps({
                "watch_ports_metrics": {
                    "323": {"identifier": "288-295", "congested_steps": 3,
                            "means_congested": {"reward": 0.5}}
                }
            }), encoding="utf-8")
            row, = analyze_trial(
                path, "fixed_low_gentle", 1,
                topology_switches(path / "input.conf"),
            )
            self.assertTrue(row["switch_switch"])
            self.assertEqual(row["agent_port"], 323)
            self.assertEqual(row["all_samples"], 3)
            self.assertEqual(row["queue_samples_unpaired"], 1)
            self.assertEqual(row["tx_gt_5pct_samples"], 3)
            self.assertAlmostEqual(row["all_queue_ge_kmin_ratio"], 2 / 3)
            self.assertAlmostEqual(row["all_queue_ge_kmax_ratio"], 1 / 3)
            self.assertEqual(row["watched_port_reward"], 0.5)

            diagnostic = path / "diagnostic"
            trial = diagnostic / "eval" / "fixed_low_gentle" / "repeat_1"
            trial.mkdir(parents=True)
            for name in (
                "complete.json", "input.conf", "input.flow", "result.queue",
                "result.rate", "result.throughput", "agent_metrics.json",
            ):
                (trial / name).write_bytes((path / name).read_bytes())
            subprocess.run([
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "analyze_acc_pilot_port_activity.py"),
                "--diagnostic-dir", str(diagnostic),
            ], check=True, capture_output=True, text=True)
            self.assertTrue((diagnostic / "port_activity" / "REPORT.md").is_file())

    def test_empty_sample_summary(self):
        self.assertIsNone(summarize([])["queue_ge_kmin_ratio"])


if __name__ == "__main__":
    unittest.main()
