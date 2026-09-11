#!/usr/bin/env python3
"""Regression tests for local continuous-sweep point generation."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("prepare_continuous_sweep.py")
ANALYZER = Path(__file__).with_name("analyze_continuous_sweep.py")


def write_eval(root, label, task, fct_ns, reward):
    directory = root / "eval" / label / task
    directory.mkdir(parents=True)
    lines = [
        f"0 {index} 1 {index + 1} 1000 {index} {fct_ns + index} 100"
        for index in range(10)
    ]
    (directory / "result.fct").write_text("\n".join(lines) + "\n")
    (directory / "input.flow").write_text("10\n")
    detail = {
        "means_active": {"reward": reward, "tail_safe_raw": reward},
        "means_congested": {"reward": reward, "tail_safe_raw": reward},
    }
    (directory / "metrics.json").write_text(json.dumps({
        "rollout_all_congested_mean": reward,
        "watch_ports_metrics": {"323": detail},
    }))


def main():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metrics = root / "metrics.json"
        config = root / "input.conf"
        output = root / "sweep_manifest.json"
        metrics.write_text(json.dumps({
            "watch_ports_metrics": {
                "323": {
                    "actions_congested": {"2,1,4": 10, "3,1,4": 2}
                }
            }
        }))
        config.write_text(
            "OPENGYM_MIN_KMIN 20000\n"
            "OPENGYM_MAX_KMIN 50000\n"
            "OPENGYM_MIN_KMAX 50000\n"
            "OPENGYM_MAX_KMAX 100000\n"
        )
        subprocess.run([
            sys.executable, str(SCRIPT),
            "--baseline-metrics", str(metrics),
            "--config", str(config),
            "--output", str(output),
            "--port", "323",
            "--link-gbps", "40",
        ], check=True)
        result = json.loads(output.read_text())
        assert result["dominant_action_indices"] == [2, 1, 4]
        assert len(result["points"]) == 7
        center = result["points"][0]
        assert center["label"] == "center"
        assert abs(center["physical"]["kmin_kb"] - 42.8432) < 1e-6
        assert abs(center["physical"]["kmax_kb"] - 100.0) < 1e-6
        assert abs(center["physical"]["pmax"] - 0.5) < 1e-12

        labels = ["greedy"] + [point["label"] for point in result["points"]]
        for label in labels:
            old_fct = 900 if label == "kmin_low_mid" else 1000
            write_eval(root, label, "mixed", old_fct, 0.7)
            write_eval(root, label, "incast", 1000, 0.6)
        subprocess.run([
            sys.executable, str(ANALYZER),
            "--run-dir", str(root),
            "--task-a", "mixed",
            "--task-b", "incast",
        ], check=True)
        analysis = json.loads(
            (root / "continuous_sweep_analysis.json").read_text()
        )
        assert analysis["best_midpoint_by_old_p95"] == "kmin_low_mid"
        assert (root / "CONTINUOUS_SWEEP_REPORT.md").exists()
    print("continuous sweep preparation tests passed")


if __name__ == "__main__":
    main()
