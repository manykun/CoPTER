#!/usr/bin/env python3
"""Regression tests for the local interpolation and exact-port PFC metrics."""

import csv
import importlib.util
import json
import tempfile
from pathlib import Path


HERE = Path(__file__).parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load("prepare_port_path_sweep", "prepare_port_path_sweep.py")
ANALYZE = load("analyze_port_path_sweep", "analyze_port_path_sweep.py")


def test_alpha_interval_and_interpolation():
    assert PREPARE.alpha_values(0.1) == [
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
    ]
    assert PREPARE.alpha_values(0.02, 0.4, 0.44) == [0.4, 0.42, 0.44]
    interpolated = PREPARE.interpolate(
        (0.0, 0.2, 0.4), (1.0, 0.6, 0.8), 0.5
    )
    assert all(
        abs(actual - expected) < 1e-12
        for actual, expected in zip(interpolated, (0.5, 0.4, 0.6))
    )


def test_best_action_uses_congested_reward():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "ports.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("scenario", "port", "action", "reward_congested"),
            )
            writer.writeheader()
            writer.writerows([
                {"scenario": "a", "port": 323, "action": "low_mid", "reward_congested": 0.4},
                {"scenario": "a", "port": 323, "action": "mid", "reward_congested": 0.3},
            ])
        assert PREPARE.best_action(path, "a", 323) == "low_mid"


def test_best_screen_action_prefers_congested_then_active_reward():
    with tempfile.TemporaryDirectory() as temporary:
        screen = Path(temporary)
        for action, congested, active in (
            ("low_moderate", None, 0.8),
            ("high_moderate", 0.7, 0.9),
        ):
            directory = screen / "task" / action
            directory.mkdir(parents=True)
            (directory / "metrics.json").write_text(
                json.dumps({
                    "watch_ports_metrics": {
                        "323": {
                            "means_congested": {"reward": congested},
                            "means_active": {"reward": active},
                        }
                    }
                }),
                encoding="utf-8",
            )
        assert PREPARE.best_screen_action(screen, "task", 323) == "high_moderate"
        assert PREPARE.normalized_action("high_moderate")


def test_reward_preference_rejects_flat_and_accepts_disjoint_ranges():
    flat = ANALYZE.reward_preference([
        {"alpha": 0.0, "reward_selected": 1.0},
        {"alpha": 1.0, "reward_selected": 1.005},
    ], 0.01)
    high = ANALYZE.reward_preference([
        {"alpha": 0.0, "reward_selected": 0.8},
        {"alpha": 0.8, "reward_selected": 1.0},
        {"alpha": 1.0, "reward_selected": 0.999},
    ], 0.01)
    low = ANALYZE.reward_preference([
        {"alpha": 0.0, "reward_selected": 1.0},
        {"alpha": 0.2, "reward_selected": 0.999},
        {"alpha": 1.0, "reward_selected": 0.8},
    ], 0.01)
    assert not flat["sensitive"]
    assert high["optimal_alpha_min"] == 0.8
    assert high["optimal_alpha_max"] == 1.0
    assert ANALYZE.disjoint_preferences(low, high)


def test_pfc_summary_filters_exact_switch_peer_and_pairs_durations():
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "result.pfc"
        path.write_text(
            "1000000000 288 1 2 1 295\n"
            "1100000000 288 1 3 1 296\n"
            "1250000000 288 1 2 0 295\n",
            encoding="utf-8",
        )
        result = ANALYZE.pfc_summary(path, "288-295", 4.0)
        assert result["pfc_mapping_available"]
        assert result["pfc_pause_events"] == 1
        assert result["pfc_resume_events"] == 1
        assert abs(result["pfc_pause_total_s"] - 0.25) < 1e-12
        assert abs(result["pfc_pause_duty"] - 0.0625) < 1e-12
        assert not result["pfc_final_paused"]


def test_epoch_metrics_persist_per_port_loss():
    source = (HERE.parents[1] / "copter" / "agent_helper.py").read_text()
    for key in (
        '"train_port_metrics"', '"loss_mean"', '"loss_median"',
        '"loss_max"', '"batch_reward_mean"',
    ):
        assert key in source


def test_pfc_trace_writes_peer_node_column():
    source = (
        HERE.parents[1] / "ns-3.33" / "scratch" / "copter-sim.cc"
    ).read_text()
    assert "peer_node_id" in source
    assert 'fprintf(fout, "%lu %u %u %u %u %d\\n"' in source


def test_realistic_driver_disables_global_replay_and_uses_screen_path():
    source = (HERE / "run_realistic_acc.sh").read_text(encoding="utf-8")
    assert "--shared-replay false" in source
    assert "--endpoint-source screen" in source
    root = HERE.parents[1] / "tools" / "traffic" / "acc_validation"
    for name in (
        "realistic_webserver",
        "realistic_cachefollower",
        "realistic_websearch",
    ):
        config = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
        assert config[0]["load"] == 0.60
        assert config[0]["pattern"] == "poisson_random"


if __name__ == "__main__":
    test_alpha_interval_and_interpolation()
    test_best_action_uses_congested_reward()
    test_best_screen_action_prefers_congested_then_active_reward()
    test_reward_preference_rejects_flat_and_accepts_disjoint_ranges()
    test_pfc_summary_filters_exact_switch_peer_and_pairs_durations()
    test_epoch_metrics_persist_per_port_loss()
    test_pfc_trace_writes_peer_node_column()
    test_realistic_driver_disables_global_replay_and_uses_screen_path()
    print("port-local metric tests passed")
