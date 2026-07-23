import importlib.util
import sys
import tempfile
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_forgetting.py")
SPEC = importlib.util.spec_from_file_location("analyze_forgetting", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_fct(path, fct_values):
    lines = []
    for index, value in enumerate(fct_values):
        # Six identity columns followed by measured and ideal FCT in ns.
        lines.append(f"0 {index} 1 {index + 1} 1000 {index} {value} 100")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_run(root, method, phase, task, fct_values, reward, expected=None):
    directory = root / "eval" / method / phase / task
    directory.mkdir(parents=True)
    write_fct(directory / "result.fct", fct_values)
    expected = expected or len(fct_values)
    (directory / "input.flow").write_text(
        str(expected) + "\n", encoding="utf-8"
    )
    (directory / "metrics.json").write_text(
        '{"rollout_mean_reward": %s}\n' % reward, encoding="utf-8"
    )


def test_detects_acc_forgetting_and_sor_retention():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "manifest.json").write_text(
            '{"task_a":"incast","task_b":"throughput","seed":1,'
            '"phase_epochs":30}',
            encoding="utf-8",
        )
        # ACC learns A/B, then loses reward and doubles old-task p95.
        make_run(root, "acc", "initial", "incast", [200, 220, 240], 0.50)
        make_run(root, "acc", "after_a", "incast", [100, 110, 120], 0.80)
        make_run(root, "acc", "after_a", "throughput", [300, 320, 340], 0.55)
        make_run(root, "acc", "after_b", "incast", [200, 220, 240], 0.60)
        make_run(root, "acc", "after_b", "throughput", [150, 160, 170], 0.75)

        # SOR learns both, retains A, and remains competitive on B.
        make_run(root, "sor", "initial", "incast", [200, 220, 240], 0.50)
        make_run(root, "sor", "after_a", "incast", [100, 110, 120], 0.80)
        make_run(root, "sor", "after_a", "throughput", [300, 320, 340], 0.55)
        make_run(root, "sor", "after_b", "incast", [105, 115, 125], 0.78)
        make_run(root, "sor", "after_b", "throughput", [152, 162, 172], 0.76)

        class Args:
            min_reward_drop = 0.10
            min_p95_worsening = 0.10
            min_acquisition_reward_gain = 0.02
            min_acquisition_p95_gain = 0.05

        acc = MODULE.method_analysis(root, "acc", "incast", "throughput", Args)
        sor = MODULE.method_analysis(root, "sor", "incast", "throughput", Args)
        assert acc["forgetting"]["detected"]
        assert acc["task_a_acquisition"]["passed"]
        assert acc["task_b_acquisition"]["passed"]
        assert not sor["forgetting"]["detected"]
        assert sor["task_a_acquisition"]["passed"]
        assert sor["task_b_acquisition"]["passed"]

        original_argv = sys.argv
        try:
            sys.argv = [
                str(MODULE_PATH),
                "--run-dir", str(root),
                "--compare", "acc,sor",
                "--gate",
            ]
            MODULE.main()
        finally:
            sys.argv = original_argv
        assert (root / "CONTINUAL_REPORT.md").exists()
        decision = (root / "continual_analysis.json").read_text(encoding="utf-8")
        assert '"selected_pass": true' in decision


if __name__ == "__main__":
    test_detects_acc_forgetting_and_sor_retention()
    print("continual analysis tests passed")
