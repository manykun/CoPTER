#!/usr/bin/env python3
"""Configuration and analysis fixtures for the ACC reward/traffic matrix."""

import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.continual_validation.run_acc_reward_traffic_matrix import (
    REWARDS, TASKS, cell_config, plan_for, prepare_plan, selection, task_base,
)
from scripts.continual_validation import run_acc_reward_traffic_matrix as runner
from tools.analysis.analyze_acc_reward_traffic_matrix import (
    relative, validate_task_arms,
)
from tools.analysis import analyze_acc_reward_traffic_matrix as analysis

FIXTURES = Path(__file__).resolve().parents[2] / "tools" / "analysis" / "tests" / "fixtures"


class RewardTrafficMatrixTest(unittest.TestCase):
    def test_four_canonical_single_task_inputs(self):
        expected = {
            "webserver": "realistic_webserver",
            "cachefollower": "realistic_cachefollower",
            "hadoop": "realistic_hadoop",
            "alistorage": "realistic_alistorage",
        }
        for task in TASKS:
            with self.subTest(task=task):
                base = task_base(task)
                self.assertEqual(base["task_a"], expected[task])
                self.assertEqual(base["task_b"], expected[task])
                self.assertEqual(base["task_a_conf"], base["task_b_conf"])
                self.assertEqual(base["task_a_flow"], base["task_b_flow"])

    def test_reward_is_only_within_task_treatment(self):
        configs = {
            reward: cell_config("hadoop", reward, f"trial_{reward}", 50, 7900, 1)
            for reward in REWARDS
        }
        excluded = {"run_id", "reward_profile", "reward_queue_lambda"}
        common = lambda config: {key: value for key, value in config.items() if key not in excluded}
        self.assertEqual(common(configs["tail5"]), common(configs["tail1"]))
        self.assertEqual(common(configs["tail5"]), common(configs["weighted"]))
        self.assertEqual(configs["tail5"]["reward_queue_lambda"], 5.0)
        self.assertEqual(configs["tail1"]["reward_queue_lambda"], 1.0)
        self.assertEqual(configs["weighted"]["reward_profile"], "weighted")
        self.assertEqual(configs["tail5"]["action_space"], "multiscale")
        self.assertEqual(configs["tail5"]["replay_sampling"], "congestion_stratified")

    def test_stable_ports_and_selection_validation(self):
        args = SimpleNamespace(stage="all", epochs=50, port_base=7900, seed=1)
        complete = plan_for(args, list(TASKS), list(REWARDS), "matrix")
        reduced = plan_for(args, ["hadoop"], ["weighted"], "matrix")
        full_cell = next(cell for cell in complete["cells"] if cell["task"] == "hadoop" and cell["reward"] == "weighted")
        self.assertEqual(reduced["cells"][0]["port"], full_cell["port"])
        with self.assertRaises(ValueError):
            selection("tail5,tail5", REWARDS, "--rewards")

    def test_existing_plan_and_cell_config_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(stage="train")
            plan = {"prefix": "fixture", "cells": [{
                "run_id": "fixture_hadoop_tail5", "config": {
                    "run_id": "fixture_hadoop_tail5", "reward_queue_lambda": 5.0,
                },
            }]}
            with patch.object(runner, "ROOT", root):
                prepare_plan(args, plan)
                prepare_plan(args, plan)
                changed = json.loads(json.dumps(plan))
                changed["cells"][0]["config"]["reward_queue_lambda"] = 1.0
                with self.assertRaisesRegex(RuntimeError, "plan differs"):
                    prepare_plan(args, changed)

    def test_analysis_rejects_hidden_confound(self):
        protocol = {"action_space": "multiscale", "seed": 1,
                    "inputs": {"a": {"flow_sha256": "abc"}},
                    "reward_profile": "tail_safe", "reward_queue_lambda": 5.0}
        candidate = dict(protocol, reward_queue_lambda=1.0)
        validate_task_arms({"tail5": protocol, "tail1": candidate})
        candidate["seed"] = 2
        with self.assertRaisesRegex(ValueError, "outside treatment"):
            validate_task_arms({"tail5": protocol, "tail1": candidate})

    def test_percent_change_direction(self):
        self.assertAlmostEqual(relative(130.0, 126.1), -3.0)
        self.assertIsNone(relative(0.0, 1.0))

    def test_training_report_requires_every_epoch_and_keeps_reward_scales_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prefix = "fixture"
            cells = []
            for reward in ("tail5", "weighted"):
                run_id = f"{prefix}_hadoop_{reward}"
                run_dir = root / "experiments" / "aba" / run_id
                (run_dir / "summaries").mkdir(parents=True)
                config_path = run_dir / "pilot_config.yaml"
                config_path.write_text("fixture: true\n", encoding="utf-8")
                expected_config = {
                    "seed": 1, "reward_profile": "weighted" if reward == "weighted" else "tail_safe",
                    "reward_queue_lambda": 5.0, "reward_ecn_lambda": 5.0,
                    "reward_window_size": 4, "replay_sampling": "congestion_stratified",
                    "replay_congested_fraction": 0.5,
                }
                protocol = {
                    "run_id": run_id, "seed": 1,
                    "config_sha256": analysis.sha256(config_path),
                    "action_space": "multiscale", "acc_shared_replay": False,
                    "epochs": {"a1": 50, "b": 0, "a2": 0},
                    "tasks": {"a": "realistic_hadoop", "b": "realistic_hadoop"},
                    "inputs": {"a": {"flow_sha256": "same"}},
                    **{key: value for key, value in expected_config.items() if key != "seed"},
                }
                (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
                path = run_dir / "summaries" / "acc_epoch_metrics.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("global_epoch", "task", "reward", "loss", "mean_fct_us", "completion_ratio"))
                    writer.writeheader()
                    writer.writerows({"global_epoch": epoch, "task": "realistic_hadoop",
                                      "reward": 0.3 + epoch / 1000, "loss": 1 / epoch,
                                      "mean_fct_us": 130 - epoch / 10,
                                      "completion_ratio": 0.98}
                                     for epoch in range(1, 51))
                cells.append({"task": "hadoop", "reward": reward,
                              "run_id": run_id, "config": expected_config})
            output = root / "experiments" / "aba" / f"{prefix}_matrix"
            output.mkdir()
            plan = {"tasks": ["hadoop"], "rewards": ["tail5", "weighted"],
                    "epochs": 50, "cells": cells}
            plan_path = output / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with patch.object(analysis, "ROOT", root), patch.object(sys, "argv", ["analyze", "--plan", str(plan_path)]):
                analysis.main()
            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("Frozen evaluations are not complete", report)
            self.assertIn("| hadoop | weighted |", report)
            self.assertTrue((output / "training_metrics.csv").is_file())
            with (root / "experiments" / "aba" / cells[0]["run_id"] / "summaries" / "acc_epoch_metrics.csv").open("a", encoding="utf-8") as handle:
                handle.write("1,realistic_hadoop,0.3,1,130,0.98\n")
            with patch.object(analysis, "ROOT", root), patch.object(sys, "argv", ["analyze", "--plan", str(plan_path)]):
                with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                    analysis.main()

    def test_frozen_fct_uses_common_completed_flows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = []
            sources = {}
            for reward in ("tail5", "weighted"):
                run_id = f"fixture_hadoop_{reward}"
                run_dir = root / "experiments" / "aba" / run_id
                run_dir.mkdir(parents=True)
                protocol_path = run_dir / "protocol.json"
                protocol_path.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
                diagnostic = run_dir.with_name(run_dir.name + "_diagnostics")
                diagnostic.mkdir()
                (diagnostic / "protocol.json").write_text(json.dumps({
                    "source_protocol_sha256": analysis.sha256(protocol_path),
                    "repeats": 2,
                }), encoding="utf-8")
                for checkpoint in ("frozen_initial", "frozen_final"):
                    for repeat in (1, 2):
                        eval_dir = diagnostic / "eval" / checkpoint / f"repeat_{repeat}"
                        eval_dir.mkdir(parents=True)
                        (eval_dir / "complete.json").write_text("{}", encoding="utf-8")
                        for source, destination in (
                            ("input.flow", "input.flow"), ("sample.conf", "input.conf"),
                            ("sample.fct", "result.fct"), ("sample.queue", "result.queue"),
                            ("sample.rate", "result.rate"),
                            ("sample.throughput", "result.throughput"),
                            ("sample.pfc", "result.pfc"),
                        ):
                            shutil.copy2(FIXTURES / source, eval_dir / destination)
                cells.append({"task": "hadoop", "reward": reward, "run_id": run_id})
                sources[run_id] = ({"run_id": run_id}, [])
            plan = {"tasks": ["hadoop"], "cells": cells}
            with patch.object(analysis, "ROOT", root):
                records, intersections, hashes = analysis.frozen_table(plan, sources, True)
            self.assertEqual(intersections, {"hadoop": 3})
            self.assertEqual(len(records), 8)
            self.assertTrue(all(row["mean_fct_us"] == 2.0 for row in records))
            self.assertEqual(len(hashes), 8 * 7)


if __name__ == "__main__":
    unittest.main()
