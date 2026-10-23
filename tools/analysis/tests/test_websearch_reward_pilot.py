from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/continual_validation"))
from run_acc_reward_action_pilot import build_config
from run_aba import load_config, protocol
from tools.analysis.metrics_core import summarize_fct_by_size


class WebSearchPilotTests(unittest.TestCase):
    def test_paired_configs_differ_only_in_reward_and_isolation(self):
        path = ROOT / "configs/aba/webserver_cachefollower.yaml"
        base = load_config(path)
        configs = [build_config(base, f"fixture_{arm}", arm, 50, 8200+i, 1) for i, arm in enumerate(("acc_paper", "acc_3816bc6_aligned"))]
        for config in configs:
            self.assertEqual(config["reward_window_size"], 1)
            self.assertFalse(config["shared_replay"])
            plan = protocol(config, path, True)
            self.assertEqual(plan["epochs"], {"a1": 2, "b": 0, "a2": 0})
        for field in ("run_id", "reward_profile", "acc_port", "sor_port"):
            for config in configs:
                config.pop(field)
        self.assertEqual(configs[0], configs[1])

    def test_size_group_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.fct"
            path.write_text("1 2 3 4 102400 0 1000 100\n1 2 3 4 102401 0 2000 100\n1 2 3 4 1048576 0 3000 100\n")
            stats = summarize_fct_by_size(path)
            for group in ("short", "medium", "long"):
                self.assertEqual(stats[f"{group}_completed_flows"], 1)
            self.assertEqual(stats["long_p99_fct_us"], 3)


if __name__ == "__main__":
    unittest.main()
