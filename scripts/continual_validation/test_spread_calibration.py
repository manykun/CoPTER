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
        "ready_ports": [323],
        "port_action_profiles": {
            "323": {
                "best_action": best_action,
                "reward_spread": 0.30,
                "rewards": {best_action: 1.0, other_action: 0.70},
            }
        },
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
        results, ["steady", "burst"], 0.03, 0.03
    )
    assert recommended is not None
    assert recommended["task_a"] == "steady"
    assert recommended["task_b"] == "burst"
    assert recommended["shared_ready_ports"] == [323]
    assert pairs[0]["old_task_p95_penalty"] == 0.3
    assert pairs[0]["new_task_p95_penalty_with_old_action"] == 0.25
    assert pairs[0]["eligible_port_conflicts"][0]["port"] == 323


def test_select_pair_rejects_ineligible_scenario():
    results = {
        "steady": scenario("high", 100.0, "mid", 130.0),
        "burst": scenario("mid", 100.0, "high", 125.0, eligible=False),
    }
    recommended, _ = MODULE.select_pair(
        results, ["steady", "burst"], 0.03, 0.03
    )
    assert recommended is None


def test_select_pair_rejects_same_per_port_reward_action():
    results = {
        "steady": scenario("high", 100.0, "mid", 130.0),
        "burst": scenario("mid", 100.0, "high", 125.0),
    }
    results["burst"]["port_action_profiles"]["323"] = {
        "best_action": "high",
        "reward_spread": 0.30,
        "rewards": {"high": 1.0, "mid": 0.70},
    }
    recommended, pairs = MODULE.select_pair(
        results, ["steady", "burst"], 0.03, 0.03
    )
    assert recommended is None
    assert pairs[0]["eligible_port_conflicts"] == []


def test_switch_switch_port_filter_excludes_host_links():
    switch_link = {
        "actions": {
            "low": {"identifier": "288-295"},
            "high": {"identifier": "295-288"},
        }
    }
    host_link = {
        "actions": {
            "low": {"identifier": "274-147"},
            "high": {"identifier": "274-147"},
        }
    }
    assert MODULE.is_switch_switch_port(switch_link)
    assert not MODULE.is_switch_switch_port(host_link)
    assert not MODULE.is_switch_switch_port({"actions": {}})


def test_controlled_pair_modes_separate_timing_and_workload_shifts():
    assert MODULE.controlled_pair("timing-only", True, True)
    assert not MODULE.controlled_pair("timing-only", False, True)
    assert MODULE.controlled_pair("workload-shift", False, True)
    assert not MODULE.controlled_pair("workload-shift", True, True)
    assert not MODULE.controlled_pair("workload-shift", False, False)


def test_port_action_profile_uses_congested_reward():
    detail = {
        "actions": {
            "gentle": {
                "reward_active": 0.9,
                "reward_congested": 0.4,
            },
            "strong": {
                "reward_active": 0.5,
                "reward_congested": 0.6,
            },
        }
    }
    profile = MODULE.port_action_profile(detail)
    assert profile["best_action"] == "strong"
    assert profile["rewards"] == {"gentle": 0.4, "strong": 0.6}
    assert abs(profile["reward_spread"] - 1 / 3) < 1e-12


if __name__ == "__main__":
    test_select_pair_requires_bidirectional_action_cost()
    test_select_pair_rejects_ineligible_scenario()
    test_select_pair_rejects_same_per_port_reward_action()
    test_switch_switch_port_filter_excludes_host_links()
    test_controlled_pair_modes_separate_timing_and_workload_shifts()
    test_port_action_profile_uses_congested_reward()
    print("spread calibration tests passed")
