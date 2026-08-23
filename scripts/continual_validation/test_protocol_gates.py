import importlib.util
import json
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FORGETTING = load_module("analyze_forgetting")
CONFLICT = load_module("analyze_conflict")
TASK_PAIR = load_module("verify_task_pair")


def write_eval(
    directory, fcts, expected, reward, all_reward=None, watch_metrics=None
):
    directory.mkdir(parents=True)
    lines = [
        f"0 {index} 1 {index + 1} 1000 {index} {fct} 100"
        for index, fct in enumerate(fcts)
    ]
    (directory / "result.fct").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (directory / "input.flow").write_text(
        f"{expected}\n", encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "rollout_mean_reward": reward,
                "rollout_all_congested_mean": all_reward,
                "watch_ports_metrics": watch_metrics or {},
            }
        ),
        encoding="utf-8",
    )


def test_acquisition_rejects_unsafe_completion_drop():
    result = {
        "reward_change": 0.20,
        "p95_fct_worsening": -0.20,
        "completion_drop": 0.02,
    }
    assert not FORGETTING.acquisition_passed(result, 0.02, 0.05, 0.01)


def test_conflict_screen_requires_different_best_actions():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "manifest.json").write_text(
            '{"task_a":"mixed","task_b":"incast"}', encoding="utf-8"
        )
        for task in ("mixed", "incast"):
            for action, reward, fct in (
                ("aggressive", 0.8, 100),
                ("balanced", 0.7, 120),
                ("permissive", 0.6, 140),
            ):
                write_eval(root / "screen" / task / action, [fct] * 10, 10, reward)
        original = sys.argv
        try:
            sys.argv = [
                str(HERE / "analyze_conflict.py"),
                "--run-dir", str(root),
            ]
            CONFLICT.main()
        finally:
            sys.argv = original
        decision = json.loads(
            (root / "screen" / "analysis.json").read_text(encoding="utf-8")
        )
        assert not decision["passed"]
        assert not decision["different_best_actions"]


def test_all_congested_reward_and_completion_sensitivity_can_pass():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "manifest.json").write_text(
            '{"task_a":"mixed","task_b":"incast"}', encoding="utf-8"
        )
        # mixed: permissive is reward/p95 best and p95-sensitive.
        for action, reward, fct in (
            ("aggressive", 0.30, 130),
            ("balanced", 0.31, 120),
            ("permissive", 0.33, 100),
        ):
            write_eval(
                root / "screen" / "mixed" / action,
                [fct] * 100,
                100,
                reward,
                reward,
            )
        # incast: top-30 says permissive, but the registered all-congested
        # reward and the only completion-safe action both select balanced.
        for action, top30, all_reward, completed, fct in (
            ("aggressive", 0.20, -0.10, 97, 102),
            ("balanced", 0.19, -0.08, 100, 100),
            ("permissive", 0.21, -0.11, 97, 101),
        ):
            write_eval(
                root / "screen" / "incast" / action,
                [fct] * completed,
                100,
                top30,
                all_reward,
            )
        original = sys.argv
        try:
            sys.argv = [
                str(HERE / "analyze_conflict.py"),
                "--run-dir", str(root),
            ]
            CONFLICT.main()
        finally:
            sys.argv = original
        decision = json.loads(
            (root / "screen" / "analysis.json").read_text(encoding="utf-8")
        )
        assert decision["passed"]
        assert decision["tasks"]["incast"]["performance_sensitive"]
        assert (
            decision["tasks"]["incast"]["best_reward_action"] == "balanced"
        )


def test_conflict_screen_checks_directional_penalty_and_shared_port():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "manifest.json").write_text(
            '{"task_a":"steady","task_b":"burst"}', encoding="utf-8"
        )
        port = {
            "323": {
                "identifier": "288-295",
                "active_steps": 10,
                "congested_steps": 6,
            }
        }
        for task, values in {
            "steady": (("steady_best", 0.9, 100), ("burst_best", 0.7, 130)),
            "burst": (("steady_best", 0.7, 130), ("burst_best", 0.9, 100)),
        }.items():
            for action, reward, fct in values:
                write_eval(
                    root / "screen" / task / action,
                    [fct] * 10,
                    10,
                    reward,
                    reward,
                    port,
                )
        original = sys.argv
        try:
            sys.argv = [
                str(HERE / "analyze_conflict.py"),
                "--run-dir", str(root),
                "--required-watch-ports", "323",
                "--min-port-active-samples", "10",
                "--min-port-congested-samples", "5",
                "--min-old-task-p95-penalty", "0.20",
            ]
            CONFLICT.main()
        finally:
            sys.argv = original
        decision = json.loads(
            (root / "screen" / "analysis.json").read_text(encoding="utf-8")
        )
        assert decision["passed"]
        assert decision["directional_conflict"]["old_task_p95_penalty"] == 0.3
        assert decision["shared_port_screen"]["ports"]["323"]["passed"]


def test_same_flow_pair_verifier_ignores_only_start_time():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        task_a = root / "a.flow"
        task_b = root / "b.flow"
        task_a.write_text(
            "2\n0 128 3 100 4096 2.000000000\n"
            "1 128 3 100 8192 2.001000000\n",
            encoding="utf-8",
        )
        task_b.write_text(
            "2\n0 128 3 100 4096 2.000000000\n"
            "1 128 3 100 8192 2.000000000\n",
            encoding="utf-8",
        )
        result = TASK_PAIR.analyze(task_a, task_b)
        assert result["same_flow_identity_multiset"]
        assert result["timing_only_shift"]
        assert result["timing_b"]["max_simultaneous_flows"] == 2
        assert result["timing_b"]["peak_flows_in_1us"] == 2


if __name__ == "__main__":
    test_acquisition_rejects_unsafe_completion_drop()
    test_conflict_screen_requires_different_best_actions()
    test_all_congested_reward_and_completion_sensitivity_can_pass()
    test_conflict_screen_checks_directional_penalty_and_shared_port()
    test_same_flow_pair_verifier_ignores_only_start_time()
    print("continual protocol gate tests passed")
