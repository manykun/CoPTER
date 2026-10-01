import importlib.util
import tempfile
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "continual_validation" / "run_acc_pilot_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("acc_pilot_diagnostics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AccPilotDiagnosticTests(unittest.TestCase):
    def test_tv_distance(self):
        self.assertEqual(MODULE.tv_distance(Counter(a=2), Counter(a=3)), 0.0)
        self.assertEqual(MODULE.tv_distance(Counter(a=2), Counter(b=3)), 1.0)
        self.assertAlmostEqual(
            MODULE.tv_distance(Counter(a=1, b=1), Counter(a=1, c=1)),
            0.5,
        )

    def test_multiscale_threshold_activation_uses_link_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / "input.conf").write_text(
                "OPENGYM_MIN_KMIN 5000\n"
                "OPENGYM_MAX_KMIN 50000\n"
                "OPENGYM_MIN_KMAX 15000\n"
                "OPENGYM_MAX_KMAX 100000\n",
                encoding="utf-8",
            )
            # Profile 0 at 40 Gbps maps to Kmin=8,000 and Kmax=24,000 bytes.
            (destination / "result.rate").write_text(
                "1 2 40000000000 0.5 0.0 0.1\n",
                encoding="utf-8",
            )
            (destination / "result.queue").write_text(
                "1 409600 2 4000 0.1\n"
                "1 409600 2 8000 0.2\n"
                "1 409600 2 24000 0.3\n",
                encoding="utf-8",
            )
            result = MODULE.threshold_activation(
                destination, (0, 0), "multiscale"
            )
            self.assertEqual(result["queue_threshold_samples"], 3)
            self.assertAlmostEqual(result["queue_ge_kmin_ratio"], 2 / 3)
            self.assertAlmostEqual(result["queue_ge_kmax_ratio"], 1 / 3)
            self.assertAlmostEqual(
                result["buffer_utilization_max"], 24000 / 409600
            )


if __name__ == "__main__":
    unittest.main()
