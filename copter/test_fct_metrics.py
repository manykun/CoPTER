import csv
import tempfile
import unittest
from pathlib import Path

from fct_metrics import FCTStepTracker


class FCTStepTrackerTest(unittest.TestCase):
    def test_incremental_step_and_cumulative_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "result.fct"
            trace = root / "steps.csv"
            source.write_text("", encoding="utf-8")
            tracker = FCTStepTracker(source, trace, episode=7, phase="train_a", rolling_flows=2)
            tracker.observe(0, global_env_step=10, global_train_step=3)
            with source.open("a", encoding="utf-8") as handle:
                handle.write("a b 1 2 100 0 1000 500\n")
                handle.write("a b 1 3 200 0 3000 1000\n")
            tracker.observe(1, global_env_step=11, global_train_step=3)
            with source.open("a", encoding="utf-8") as handle:
                handle.write("a b 1 4 300 0 5000 1000\n")
            tracker.observe(2, global_env_step=12, global_train_step=4, terminal=True)
            summary = tracker.summary()
            tracker.close()

            with trace.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(int(rows[0]["step_completed_flows"]), 0)
            self.assertEqual(int(rows[1]["step_completed_flows"]), 2)
            self.assertEqual(float(rows[1]["step_mean_fct_us"]), 2.0)
            self.assertEqual(int(rows[2]["cumulative_completed_flows"]), 3)
            self.assertAlmostEqual(float(rows[2]["cumulative_mean_fct_us"]), 3.0)
            self.assertAlmostEqual(float(rows[2]["rolling_mean_fct_us"]), 4.0)
            self.assertEqual(rows[2]["terminal"], "True")
            self.assertEqual(summary["completed_flows"], 3)


if __name__ == "__main__":
    unittest.main()
