#!/usr/bin/env python3
"""Regression test for descriptive path-intervention analysis."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_path_intervention.py")


def write_eval(root, variant, task, fct_ns, reward, port_reward):
    directory = root / "eval" / variant / task
    directory.mkdir(parents=True)
    lines = [
        f"0 {index} 1 {index + 1} 1000 {index} {fct_ns + index} 100"
        for index in range(10)
    ]
    (directory / "result.fct").write_text("\n".join(lines) + "\n")
    (directory / "input.flow").write_text("10\n")
    port_detail = {
        "identifier": "288-295",
        "active_steps": 10,
        "congested_steps": 8,
        "means_active": {"reward": port_reward, "tail_safe_raw": port_reward},
        "means_congested": {
            "reward": port_reward,
            "tail_safe_raw": port_reward,
        },
    }
    (directory / "metrics.json").write_text(json.dumps({
        "rollout_all_congested_mean": reward,
        "watch_ports_metrics": {"323": port_detail},
    }))


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for variant, old_fct, old_reward, port_reward in (
            ("after_a", 100, 0.8, 0.8),
            ("after_b", 200, 0.6, 0.6),
            ("restore_323", 120, 0.75, 0.76),
        ):
            write_eval(root, variant, "mixed", old_fct, old_reward, port_reward)
            write_eval(root, variant, "incast", 100, 0.7, 0.7)
        subprocess.run([
            sys.executable, str(SCRIPT), "--run-dir", str(root),
            "--task-a", "mixed", "--task-b", "incast",
        ], check=True)
        result = json.loads(
            (root / "path_intervention_analysis.json").read_text()
        )
        restored = next(
            row for row in result["global"] if row["variant"] == "restore_323"
        )
        assert 0.79 < restored["p95_recovery"] < 0.81
        port = next(
            row for row in result["ports"]
            if row["variant"] == "restore_323"
            and row["port"] == 323
            and row["population"] == "congested"
        )
        assert 0.79 < port["reward_recovery"] < 0.81
        assert (root / "PATH_INTERVENTION_REPORT.md").exists()
    print("path intervention analysis tests passed")


if __name__ == "__main__":
    main()
