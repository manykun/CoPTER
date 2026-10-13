import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "copter"))
from structures import parse_forced_ports_action
from scripts.continual_validation.run_acc_port_intervention import (
    labels,
    parse_sizes,
    select_port_sets,
)


class AccPortInterventionTests(unittest.TestCase):
    def test_force_port_set_parser(self):
        self.assertEqual(
            parse_forced_ports_action("323,321:8,0", "multiscale"),
            ((323, 321), (8, 0)),
        )
        for value in ("323,323:8,0", "-1:8,0", "323:9,0", "323:8", "323, :8,0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_forced_ports_action(value, "multiscale")

    def test_selection_is_switch_only_active_deterministic_and_disjoint(self):
        identifiers = ["10-11", "10-12", "11-12", "12-10", "10-99", "12-11"]
        active = {
            (10, 11): {"tx_rate_mean": .4},
            (10, 12): {"tx_rate_mean": .3},
            (11, 12): {"tx_rate_mean": .5},
            (12, 10): {"tx_rate_mean": .2},
            (10, 99): {"tx_rate_mean": .8},
            (12, 11): {"tx_rate_mean": .009},
        }
        low = {key: 10.0 for key in active}
        high = {**low, (10, 11): 15.0, (10, 12): 12.0}
        one = select_port_sets(identifiers, {10, 11, 12}, active, low, high, (1, 2), 7)
        two = select_port_sets(identifiers, {10, 11, 12}, active, low, high, (1, 2), 7)
        self.assertEqual(one, two)
        self.assertEqual(one["target_order"], [0, 1])
        self.assertEqual(set(one["control_order"]), {2, 3})
        self.assertEqual(one["eligible_count"], 4)
        self.assertEqual(labels(one, (1, 2))[-1][0], "all_high_gentle")

    def test_sizes_must_increase(self):
        self.assertEqual(parse_sizes("1,2,4"), (1, 2, 4))
        for value in ("", "2,1", "1,1", "0,1", "a"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_sizes(value)


if __name__ == "__main__":
    unittest.main()
