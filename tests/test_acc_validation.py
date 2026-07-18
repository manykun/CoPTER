import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "copter"))
sys.path.insert(0, str(ROOT / "tools" / "traffic"))

from structures import DCQCNParameters, acc_action_from_indices, validate_acc_action_indices
from TraGen import expand_hosts


def load_analysis_module():
    path = ROOT / "scripts" / "acc_validation" / "analyze_validation.py"
    spec = importlib.util.spec_from_file_location("acc_validation_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ACCValidationTests(unittest.TestCase):
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

    def test_compact_host_range(self):
        self.assertEqual(expand_hosts({"start": 4, "end": 7}), [4, 5, 6, 7])
        self.assertEqual(expand_hosts([2, "3"]), [2, 3])
        with self.assertRaises(ValueError):
            expand_hosts({"start": 7, "end": 4})

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
                rows.append({
                    "kind": "eval", "scenario": scenario, "seed": seed,
                    "method": "greedy", "p95_fct_us": 94.0,
                    "completion_ratio": 1.0, "rollout_mean_reward": 0.75,
                })
        self.assertTrue(analysis.sensitivity_gate(rows, 0.05)["passed"])
        self.assertTrue(analysis.effectiveness_gate(rows, 0.05)["passed"])

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
