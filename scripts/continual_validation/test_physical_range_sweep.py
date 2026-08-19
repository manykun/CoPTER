#!/usr/bin/env python3
"""Regression tests for physical per-port range preparation and analysis."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
PREPARE = HERE / "prepare_physical_range_sweep.py"
ANALYZE = HERE / "analyze_physical_range_sweep.py"
SIMULATOR = HERE.parents[1] / "ns-3.33" / "scratch" / "copter-sim.cc"


def write_eval(root, label, task, fct_ns, completion=1.0):
    directory = root / "eval" / label / task
    directory.mkdir(parents=True)
    completed = int(10 * completion)
    lines = [
        f"0 {index} 1 {index + 1} 1000 {index} {fct_ns + index} 100"
        for index in range(completed)
    ]
    (directory / "result.fct").write_text("\n".join(lines) + "\n")
    (directory / "input.flow").write_text("10\n")
    reward = 0.6 + 0.001 * completed
    (directory / "metrics.json").write_text(json.dumps({
        "rollout_all_congested_mean": reward,
        "watch_ports_metrics": {
            "323": {
                "means_active": {"reward": reward},
                "means_congested": {"reward": reward, "tail_safe_raw": reward},
            }
        },
    }))


def main():
    source = SIMULATOR.read_text(encoding="utf-8")
    for key in (
        "OPENGYM_FORCE_PORT_INDEX",
        "OPENGYM_FORCE_KMIN_KB",
        "OPENGYM_FORCE_KMAX_KB",
        "OPENGYM_FORCE_PMAX",
    ):
        assert key in source

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / "range_manifest.json"
        subprocess.run([
            sys.executable, str(PREPARE),
            "--output", str(manifest),
            "--target-port", "323",
            "--point-set", "wide",
            "--base-run-id", "fixture",
            "--task-a", "mixed",
            "--task-b", "incast",
            "--buffer-kb", "400",
        ], check=True)
        record = json.loads(manifest.read_text())
        assert len(record["points"]) == 14
        assert record["points"][0]["label"] == "center"
        assert any(point["kmax_kb"] == 320 for point in record["points"])
        assert any(point["pmax"] == 0.01 for point in record["points"])

        for index, point in enumerate(record["points"]):
            # kmin_24, kmin_32, and kmax_80 become the three best candidates.
            ranking = {"kmin_24": 700, "kmin_32": 800, "kmax_80": 900}
            write_eval(root, point["label"], "mixed", ranking.get(point["label"], 1100 + index))
        subprocess.run([
            sys.executable, str(ANALYZE),
            "--run-dir", str(root),
            "--stage", "screen",
            "--top-k", "3",
        ], check=True)
        selected = json.loads((root / "top_candidates.json").read_text())
        assert selected["labels"] == ["center", "kmin_24", "kmin_32", "kmax_80"]

        for label in selected["labels"]:
            write_eval(root, label, "incast", 1000)
        subprocess.run([
            sys.executable, str(ANALYZE),
            "--run-dir", str(root),
            "--stage", "safety",
            "--top-k", "3",
        ], check=True)
        analysis = json.loads((root / "physical_range_analysis.json").read_text())
        assert analysis["new_common_flows"] == 10
        assert (root / "PHYSICAL_RANGE_REPORT.md").exists()
        assert (root / "physical_range_summary.csv").exists()

    print("physical range sweep tests passed")


if __name__ == "__main__":
    main()
