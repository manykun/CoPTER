import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "copter"))
sys.path.insert(0, str(ROOT / "tools" / "traffic"))

from structures import (
    ACC_MULTISCALE_PMAX_VALUES,
    ACC_MULTISCALE_THRESHOLD_PROFILES,
    DCQCNParameters,
    acc_action_dimensions,
    acc_action_from_indices,
    validate_acc_action_indices,
)
from TraGen import downsample_flows, expand_hosts


def load_analysis_module():
    path = ROOT / "scripts" / "acc_validation" / "analyze_validation.py"
    spec = importlib.util.spec_from_file_location("acc_validation_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ACCValidationTests(unittest.TestCase):
    def test_config_template_has_explicit_stop_time(self):
        template = (ROOT / "scripts" / "acc_validation" / "acc_validation.conf.in").read_text()
        self.assertIn("SIMULATOR_STOP_TIME @STOP_TIME@", template)
        prepare = (ROOT / "scripts" / "acc_validation" / "prepare_scenarios.sh").read_text()
        self.assertIn('BASELINE_STOP_TIME="4.00"', prepare)
        self.assertIn('render_config "${name}" 1 "4.00"', prepare)
        self.assertIn('render_config "${name}_secn1" 0 "${BASELINE_STOP_TIME}"', prepare)
        self.assertIn('render_config "${name}_secn2" 0 "${BASELINE_STOP_TIME}"', prepare)

    def test_only_baseline_stage_requires_static_config_validation(self):
        runner = (ROOT / "scripts" / "acc_validation" / "run_validation.sh").read_text()
        self.assertIn(
            '[[ "${STAGE}" == baseline || "${STAGE}" == all ]] && { validate_prepared_configs 1;',
            runner,
        )
        self.assertIn(
            '[[ "${STAGE}" == train || "${STAGE}" == all ]] && { validate_prepared_configs 0;',
            runner,
        )
        self.assertIn(
            '[[ "${STAGE}" == eval || "${STAGE}" == all ]] && { validate_prepared_configs 0;',
            runner,
        )

    def test_action_mapping(self):
        self.assertEqual(
            acc_action_from_indices((0, 0, 9)),
            DCQCNParameters(0.0, 0.0, 1.0),
        )
        self.assertEqual(
            acc_action_from_indices((5, 3, 0)),
            DCQCNParameters(1.0, 1.0, 0.1),
        )

    def test_action_bounds(self):
        with self.assertRaises(ValueError):
            validate_acc_action_indices((6, 0, 0))
        with self.assertRaises(ValueError):
            validate_acc_action_indices((0, 0))

    def test_multiscale_action_mapping_and_bounds(self):
        self.assertEqual(acc_action_dimensions("multiscale"), (9, 7))
        self.assertEqual(
            acc_action_from_indices((0, 0), "multiscale"),
            DCQCNParameters(0.0, 0.0, 0.05),
        )
        self.assertEqual(
            acc_action_from_indices((8, 6), "multiscale"),
            DCQCNParameters(1.0, 1.0, 1.0),
        )
        with self.assertRaises(ValueError):
            validate_acc_action_indices((9, 0), "multiscale")
        with self.assertRaises(ValueError):
            validate_acc_action_indices((0, 7), "multiscale")
        with self.assertRaises(ValueError):
            validate_acc_action_indices((0, 0, 0), "multiscale")

    def test_multiscale_profiles_are_valid_physical_threshold_pairs(self):
        kmin_min, kmin_max = 5.0, 50.0
        kmax_min, kmax_max = 15.0, 100.0
        for kmin_norm, kmax_norm in ACC_MULTISCALE_THRESHOLD_PROFILES:
            kmin = kmin_min + kmin_norm * (kmin_max - kmin_min)
            kmax = kmax_min + kmax_norm * (kmax_max - kmax_min)
            self.assertLess(kmin, kmax)
        self.assertEqual(ACC_MULTISCALE_PMAX_VALUES, (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0))

    def test_compact_host_range(self):
        self.assertEqual(expand_hosts({"start": 4, "end": 7}), [4, 5, 6, 7])
        self.assertEqual(expand_hosts([2, "3"]), [2, 3])
        with self.assertRaises(ValueError):
            expand_hosts({"start": 7, "end": 4})

    def test_smoke_flow_downsampling_is_even_and_bounded(self):
        flows = list(range(10))
        self.assertEqual(downsample_flows(flows, 4), [0, 2, 5, 7])
        self.assertIs(downsample_flows(flows, 0), flows)
        with self.assertRaises(ValueError):
            downsample_flows(flows, -1)

    def test_stress_pair_changes_only_arrival_spread(self):
        traffic = ROOT / "tools" / "traffic" / "acc_validation"
        steady = json.loads(
            (traffic / "samepath_steady_stress.json").read_text()
        )[0]
        burst = json.loads(
            (traffic / "samepath_burst_stress.json").read_text()
        )[0]
        steady_spread = steady.pop("spread_fraction")
        burst_spread = burst.pop("spread_fraction")
        self.assertEqual(steady, burst)
        self.assertEqual(steady_spread, 0.95)
        self.assertEqual(burst_spread, 0.0)
        sources = steady["src_hosts"]["end"] - steady["src_hosts"]["start"] + 1
        cycles = round(steady["duration_s"] / steady["period_s"])
        self.assertEqual(sources * cycles, 1920)

    def test_percentile_and_rank_correlation(self):
        analysis = load_analysis_module()
        self.assertEqual(analysis.percentile([1, 2, 3], 0.5), 2)
        correlation = analysis.correlation(
            analysis.rank_values([0.1, 0.2, 0.3]),
            analysis.rank_values([10, 20, 30]),
        )
        self.assertAlmostEqual(correlation, 1.0)

    def test_validation_gates(self):
        analysis = load_analysis_module()
        rows = []
        for scenario in ("throughput", "incast", "mixed"):
            for seed in (1, 2, 3):
                for method, p95, reward in (
                    ("aggressive", 120.0, 0.60),
                    ("balanced", 100.0, 0.70),
                    ("permissive", 140.0, 0.50),
                ):
                    rows.append({
                        "kind": "sensitivity", "scenario": scenario,
                        "seed": seed, "method": method, "p95_fct_us": p95,
                        "completion_ratio": 1.0, "rollout_mean_reward": reward,
                    })
                for method, p95 in (("secn1", 110.0), ("secn2", 100.0)):
                    rows.append({
                        "kind": "baseline", "scenario": scenario,
                        "seed": seed, "method": method, "p95_fct_us": p95,
                        "completion_ratio": 1.0, "rollout_mean_reward": None,
                    })
                rows.append({
                    "kind": "eval", "scenario": scenario, "seed": seed,
                    "method": "greedy", "p95_fct_us": 94.0,
                    "completion_ratio": 1.0, "rollout_mean_reward": 0.75,
                })
        self.assertTrue(analysis.sensitivity_gate(rows, 0.05)["passed"])
        self.assertTrue(analysis.baseline_gate(rows)["passed"])
        self.assertTrue(analysis.effectiveness_gate(rows, 0.05)["passed"])

    def test_finite_horizon_baselines_and_completion_guard(self):
        analysis = load_analysis_module()
        rows = [
            {"kind": "baseline", "scenario": "incast", "seed": 1,
             "method": "secn1", "p95_fct_us": 110.0, "completion_ratio": 0.88},
            {"kind": "baseline", "scenario": "incast", "seed": 1,
             "method": "secn2", "p95_fct_us": 125.0, "completion_ratio": 0.82},
            {"kind": "eval", "scenario": "incast", "seed": 1,
             "method": "greedy", "p95_fct_us": 100.0, "completion_ratio": 0.87},
        ]
        self.assertTrue(analysis.baseline_gate(rows)["passed"])
        self.assertTrue(analysis.effectiveness_gate(rows, 0.05, 0.01)["passed"])
        rows[-1]["completion_ratio"] = 0.869
        self.assertFalse(analysis.effectiveness_gate(rows, 0.05, 0.01)["passed"])

    def test_effectiveness_uses_cross_method_common_flows(self):
        analysis = load_analysis_module()
        flow_a = ("a",)
        flow_b = ("b",)
        flow_c = ("c",)
        runs = {
            ("baseline", "incast", 1, "secn1"): {
                "kind": "baseline", "directory": Path("missing"), "expected": 3,
                "flows": {flow_a: (1000, 500), flow_b: (5000, 500), flow_c: (9000, 500)},
                "metrics": {},
            },
            ("baseline", "incast", 1, "secn2"): {
                "kind": "baseline", "directory": Path("missing"), "expected": 3,
                "flows": {flow_a: (1200, 500), flow_b: (6000, 500)}, "metrics": {},
            },
            ("eval", "incast", 1, "greedy"): {
                "kind": "eval", "directory": Path("missing"), "expected": 3,
                "flows": {flow_a: (900, 500), flow_c: (8000, 500)}, "metrics": {},
            },
        }
        rows = analysis.build_rows(runs)
        self.assertEqual({row["matched_flows"] for row in rows}, {1})
        self.assertEqual(
            {row["method"]: row["p95_fct_us"] for row in rows},
            {"secn1": 1.0, "secn2": 1.2, "greedy": 0.9},
        )

    def test_static_baseline_loader_does_not_require_agent_metrics(self):
        analysis = load_analysis_module()
        with tempfile.TemporaryDirectory() as directory:
            method = Path(directory) / "throughput" / "seed_1" / "secn1"
            method.mkdir(parents=True)
            (method / "input.flow").write_text("1\n")
            (method / "sample.fct").write_text(
                "0 1 3 100 4000 2.000000000 1000 500\n"
            )
            runs = analysis.load_runs(Path(directory), "baseline")
            rows = analysis.build_rows(runs)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["method"], "secn1")
        self.assertIsNone(rows[0]["rollout_mean_reward"])

    def test_fct_parser_preserves_duplicate_flow_rows(self):
        analysis = load_analysis_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.fct"
            row = "0 1 3 100 4000 2.000000000 1000 500\n"
            path.write_text(row + row)
            flows = analysis.parse_fct(path)
        self.assertEqual(len(flows), 2)


if __name__ == "__main__":
    unittest.main()
