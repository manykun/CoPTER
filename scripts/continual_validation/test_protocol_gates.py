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


def write_eval(directory, fcts, expected, reward):
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
        json.dumps({"rollout_mean_reward": reward}), encoding="utf-8"
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


if __name__ == "__main__":
    test_acquisition_rejects_unsafe_completion_drop()
    test_conflict_screen_requires_different_best_actions()
    print("continual protocol gate tests passed")
