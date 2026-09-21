#!/usr/bin/env python3

import math
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.analysis.metrics_core import (
    aggregate_agent_losses,
    parse_pfc,
    parse_queue,
    parse_rate,
    parse_throughput,
    percentile,
    summarize_fct,
)


FIXTURES = Path(__file__).parent / "fixtures"


class MetricsCoreTest(unittest.TestCase):
    def test_structured_replay_composition_is_preserved(self):
        result = aggregate_agent_losses({
            "replay": {"global": {
                "size": 12,
                "clusters": 3,
                "task_sizes": {"a1": 8, "b": 4},
                "task_sample_counts": {"a1": 16, "b": 12},
            }},
            "train_port_metrics": {},
        })
        self.assertEqual(result["global_replay_size"], 12.0)
        self.assertEqual(json.loads(result["global_replay_task_sizes"]), {"a1": 8, "b": 4})
        self.assertEqual(
            json.loads(result["global_replay_task_sample_counts"]),
            {"a1": 16, "b": 12},
        )

    def test_linear_percentile(self):
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.95), 3.85)

    def test_fct_and_completion(self):
        result = summarize_fct(FIXTURES / "sample.fct", FIXTURES / "input.flow")
        self.assertEqual(result["completed_flows"], 3)
        self.assertEqual(result["completion_ratio"], 0.75)
        self.assertEqual(result["mean_fct_us"], 2.0)
        self.assertAlmostEqual(result["p95_fct_us"], 2.9)
        self.assertAlmostEqual(result["p99_slowdown"], 2.98)

    def test_queue_units(self):
        overall, ports = parse_queue(FIXTURES / "sample.queue")
        self.assertAlmostEqual(overall["mean"], 7 / 3)
        self.assertEqual(ports[(1, 2)]["max"], 2.0)

    def test_rate_warmup_and_throughput(self):
        overall, ports = parse_rate(FIXTURES / "sample.rate", warmup_buckets=2)
        self.assertEqual(overall["tx_rate_mean"], 0.4)
        self.assertEqual(overall["ecn_positive_ratio"], 0.5)
        throughput, _ = parse_throughput(FIXTURES / "sample.throughput")
        self.assertEqual(throughput["mean"], 2.0)

    def test_pfc_durations(self):
        overall, ports = parse_pfc(FIXTURES / "sample.pfc", stop_time_s=4.0)
        self.assertTrue(overall["pfc_mapping_available"])
        self.assertEqual(overall["pfc_pause_count"], 2)
        self.assertEqual(overall["pfc_pause_total_s"], 2.0)
        self.assertEqual(overall["pfc_max_pause_s"], 1.0)
        self.assertEqual(overall["pfc_duty"], 0.5)
        self.assertEqual(ports[(1, 2)]["pfc_pause_count"], 2)

    def test_pfc_network_duty_includes_zero_event_ports(self):
        overall, ports = parse_pfc(
            FIXTURES / "sample.pfc",
            stop_time_s=4.0,
            port_universe={(1, 2), (1, 3)},
        )
        self.assertEqual(overall["pfc_duty"], 0.25)
        self.assertEqual(ports[(1, 3)]["pfc_pause_count"], 0)
        self.assertEqual(ports[(1, 3)]["pfc_duty"], 0.0)

    def test_empty_pfc_stream_is_zero_not_missing_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pfc"
            path.write_text("", encoding="utf-8")
            overall, ports = parse_pfc(path, stop_time_s=4.0)
            self.assertTrue(overall["pfc_mapping_available"])
            self.assertEqual(overall["pfc_pause_count"], 0)
            self.assertEqual(overall["pfc_duty"], 0.0)
            self.assertEqual(ports, {})

    def test_unmapped_pfc_stream_is_na(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unmapped.pfc"
            path.write_text("1000000000 1 1 2 1 -1\n", encoding="utf-8")
            overall, ports = parse_pfc(path, stop_time_s=4.0)
            self.assertFalse(overall["pfc_mapping_available"])
            self.assertIsNone(overall["pfc_pause_count"])
            self.assertIsNone(overall["pfc_duty"])
            self.assertEqual(ports, {})

    def test_epoch_cli_writes_complete_metric_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory)
            for source, destination in (
                ("sample.fct", "result.fct"), ("sample.queue", "result.queue"),
                ("sample.rate", "result.rate"), ("sample.throughput", "result.throughput"),
                ("sample.pfc", "result.pfc"),
            ):
                shutil.copy2(FIXTURES / source, raw / destination)
            subprocess.run([
                sys.executable, str(FIXTURES.parents[1] / "analyze_aba_epoch.py"),
                "--run-dir", str(raw), "--run-id", "fixture", "--method", "acc",
                "--phase", "a1", "--task", "task_a", "--epoch", "1",
                "--phase-epoch", "1", "--raw-dir", str(raw),
                "--flow", str(FIXTURES / "input.flow"),
                "--runtime-conf", str(FIXTURES / "sample.conf"),
                "--agent-metrics", str(FIXTURES / "agent_metrics.jsonl"),
                "--duration-seconds", "1", "--ns3-exit-code", "0", "--agent-exit-code", "0",
            ], check=True, stdout=subprocess.DEVNULL)
            metrics = json.loads((raw / "epoch_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["p95_fct_us"], 2.9)
            self.assertEqual(metrics["q_inflated_ports"], 0)
            self.assertTrue((raw / "epoch_ports.csv").exists())
            with (raw / "epoch_ports.csv").open(encoding="utf-8", newline="") as handle:
                ports = list(csv.DictReader(handle))
            physical = [row for row in ports if row["record_source"] == "physical"]
            self.assertTrue(physical)
            self.assertTrue(
                any(row["mean_queue_kb"] and row["mean_ecn_rate"] and row["pfc_duty"]
                    for row in physical)
            )
            self.assertTrue((raw / "analysis_provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
