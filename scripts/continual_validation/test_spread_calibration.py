import importlib.util
from pathlib import Path


HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location(
    "spread_calibration", HERE / "analyze_spread_calibration.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def scenario(best_action, own_p95, other_action, other_p95, eligible=True):
    return {
        "best_reward_action": best_action,
        "eligible": eligible,
        "measurements": {
            best_action: {"p95_fct_us": own_p95},
            other_action: {"p95_fct_us": other_p95},
        },
    }


def test_select_pair_requires_bidirectional_action_cost():
    results = {
        "steady": scenario("high", 100.0, "mid", 130.0),
        "burst": scenario("mid", 100.0, "high", 125.0),
    }
    recommended, pairs = MODULE.select_pair(
        results, ["steady", "burst"], 0.03
    )
    assert recommended is not None
    assert recommended["task_a"] == "steady"
    assert recommended["task_b"] == "burst"
    assert pairs[0]["old_task_p95_penalty"] == 0.3
    assert pairs[0]["new_task_p95_penalty_with_old_action"] == 0.25


def test_select_pair_rejects_ineligible_scenario():
    results = {
        "steady": scenario("high", 100.0, "mid", 130.0),
        "burst": scenario("mid", 100.0, "high", 125.0, eligible=False),
    }
    recommended, _ = MODULE.select_pair(
        results, ["steady", "burst"], 0.03
    )
    assert recommended is None


if __name__ == "__main__":
    test_select_pair_requires_bidirectional_action_cost()
    test_select_pair_rejects_ineligible_scenario()
    print("spread calibration tests passed")
