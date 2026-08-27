#!/usr/bin/env python3
"""Regression tests for the local interpolation and exact-port PFC metrics."""

import csv
import importlib.util
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


if __name__ == "__main__":
    test_alpha_interval_and_interpolation()
    test_best_action_uses_congested_reward()
    test_pfc_summary_filters_exact_switch_peer_and_pairs_durations()
    test_epoch_metrics_persist_per_port_loss()
    test_pfc_trace_writes_peer_node_column()
    print("port-local metric tests passed")
