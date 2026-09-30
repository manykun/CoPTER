#!/usr/bin/env python3

import csv
import fcntl
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "run_aba", ROOT / "scripts" / "continual_validation" / "run_aba.py"
)
RUN_ABA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_ABA)


class ABASchedulerTest(unittest.TestCase):
    def test_registered_routes_are_isolated_and_hashed(self):
        configs = [
            RUN_ABA.load_config(ROOT / "configs" / "aba" / "webserver_cachefollower.yaml"),
            RUN_ABA.load_config(ROOT / "configs" / "aba" / "hadoop_alistorage.yaml"),
        ]
        for config in configs:
            RUN_ABA.validate_config(config)
        ports = [
            int(config[f"{method}_port"])
            for config in configs
            for method in ("acc", "sor")
        ]
        self.assertEqual(len(ports), len(set(ports)))
        self.assertNotEqual(RUN_ABA.runtime_root(configs[0]), RUN_ABA.runtime_root(configs[1]))

        manifest_path = ROOT / "simulation" / "mix" / "aba" / "hadoop_alistorage" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for task in manifest["tasks"].values():
            for kind in ("conf", "flow"):
                path = manifest_path.parent / task[kind]
                self.assertEqual(RUN_ABA.sha256_file(path), task[f"{kind}_sha256"])
            generator = ROOT / task["generator_config"]
            self.assertEqual(
                RUN_ABA.sha256_file(generator), task["generator_config_sha256"]
            )

    def test_phase_plan_is_contiguous(self):
        protocol = {"epochs": {"a1": 2, "b": 2, "a2": 2}}
        rows = list(RUN_ABA.phase_plan(protocol))
        self.assertEqual([row[3] for row in rows], [1, 2, 3, 4, 5, 6])
        self.assertEqual([row[0] for row in rows], ["a1", "a1", "b", "b", "a2", "a2"])

    def test_acc_pilot_accepts_exact_legacy_action_and_reward(self):
        config = RUN_ABA.load_config(
            ROOT / "configs" / "aba" / "hadoop_alistorage.yaml"
        )
        config.update({
            "run_id": "legacy_weighted_fixture",
            "experiment_kind": "acc_pilot",
            "route": "CCC",
            "a1_epochs": 100,
            "b_epochs": 0,
            "a2_epochs": 0,
            "methods": "ACC",
            "action_space": "legacy",
            "reward_profile": "weighted",
        })
        RUN_ABA.validate_config(config)

    def test_runtime_config_rewrites_every_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = ROOT / "simulation" / "mix" / "aba" / "webserver_cachefollower" / "task_a_webserver.conf"
            destination = root / "epoch.conf"
            flow = root / "input.flow"
            flow.write_text("0\n", encoding="utf-8")
            output = root / "output"
            RUN_ABA.render_runtime_conf(source, flow, output, destination, 400)
            text = destination.read_text(encoding="utf-8")
            self.assertIn(f"FLOW_FILE {flow}", text)
            for directive, suffix in RUN_ABA.OUTPUT_DIRECTIVES.items():
                self.assertIn(f"{directive} {output / ('result' + suffix)}", text)

    def test_runtime_root_is_route_local(self):
        aba = {
            "task_a_conf": "simulation/mix/aba/webserver_cachefollower/task_a_webserver.conf",
            "task_b_conf": "simulation/mix/aba/webserver_cachefollower/task_b_cachefollower.conf",
        }
        cdc = {
            "task_a_conf": "simulation/mix/aba/hadoop_alistorage/task_c_hadoop.conf",
            "task_b_conf": "simulation/mix/aba/hadoop_alistorage/task_d_alistorage.conf",
            "runtime_root": "simulation/mix/aba/hadoop_alistorage/runtime",
        }
        self.assertNotEqual(RUN_ABA.runtime_root(aba), RUN_ABA.runtime_root(cdc))
        self.assertEqual(
            RUN_ABA.runtime_root(cdc),
            ROOT / "simulation" / "mix" / "aba" / "hadoop_alistorage" / "runtime",
        )

    def test_resume_refuses_missing_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "experiment_train_state.json").write_text(
                '{"node_number": 2, "global_train_step": 1}', encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                RUN_ABA.ensure_exact_resume("acc", model, "experiment", 2)

    def test_resume_refuses_missing_complete_model_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "experiment_train_state.json").write_text(
                '{"node_number": 1, "global_train_step": 1}', encoding="utf-8"
            )
            (model / "experiment_rb_port0.pkl").write_bytes(b"replay")
            (model / "experiment_rng.pkl").write_bytes(b"rng")
            with self.assertRaises(RuntimeError):
                RUN_ABA.ensure_exact_resume("acc", model, "experiment", 2)

    def test_training_requires_previous_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            state = model / "experiment_train_state.json"
            state.write_text(json.dumps({"epoch": 1}), encoding="utf-8")
            RUN_ABA.ensure_epoch_ready_for_training(model, "experiment", 2)
            state.write_text(json.dumps({"epoch": 2}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                RUN_ABA.ensure_epoch_ready_for_training(model, "experiment", 2)

    def test_epoch_one_refuses_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "experiment_train_state.json").write_text(
                json.dumps({"epoch": 1}), encoding="utf-8"
            )
            with self.assertRaises(RuntimeError):
                RUN_ABA.ensure_epoch_ready_for_training(model, "experiment", 1)

    def test_summary_rejects_duplicate_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.csv"
            fields = ["method", "global_epoch", "value"]
            RUN_ABA.append_rows(path, fields, [{"method": "acc", "global_epoch": 1, "value": 2}])
            RUN_ABA.append_rows(path, fields, [{"method": "acc", "global_epoch": 1, "value": 2}])
            with self.assertRaises(RuntimeError):
                RUN_ABA.append_rows(path, fields, [{"method": "acc", "global_epoch": 1, "value": 3}])

    def test_method_lock_rejects_duplicate_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            lock_path = run_dir / "acc" / ".run.lock"
            lock_path.parent.mkdir(parents=True)
            with lock_path.open("a+") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(RuntimeError):
                    RUN_ABA.method_worker(
                        {}, {"run_id": "fixture"}, run_dir, "acc", False
                    )


if __name__ == "__main__":
    unittest.main()
