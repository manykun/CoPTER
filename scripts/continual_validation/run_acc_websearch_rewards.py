#!/usr/bin/env python3
"""Two concurrent WebSearch ACC reward arms. Online training only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from run_aba import load_config, render_runtime_conf, sha256_file, atomic_json
from run_acc_reward_action_pilot import run_variant

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("acc_paper", "acc_3816bc6_aligned")


def prepare_inputs(seed):
    folder = ROOT / "simulation/mix/aba" / f"websearch_rewards_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    generator = ROOT / "tools/traffic/acc_validation/realistic_websearch.json"
    cdf = ROOT / "tools/traffic/pattern/WebSearch.txt"
    template = ROOT / "simulation/mix/aba/webserver_cachefollower/task_a_webserver.conf"
    flow, conf = folder / "task_websearch.flow", folder / "task_websearch.conf"
    manifest_file = folder / "manifest.json"
    sources = {str(path.relative_to(ROOT)): sha256_file(path) for path in (generator, cdf, template, ROOT / "tools/traffic/TraGen.py", ROOT / "simulation/mix/256hosts.topo")}
    registration = {"seed": seed, "bandwidth": "10G", "sources": sources}
    if manifest_file.exists():
        saved = json.loads(manifest_file.read_text())
        if saved["registration"] != registration or saved["flow_sha256"] != sha256_file(flow) or saved["conf_sha256"] != sha256_file(conf):
            raise RuntimeError("canonical WebSearch inputs changed; refusing reuse")
        return conf, flow
    if flow.exists() or conf.exists():
        raise RuntimeError(f"incomplete input preparation in {folder}; inspect before retrying")
    subprocess.run([sys.executable, str(ROOT / "tools/traffic/TraGen.py"), "--config", str(generator), "--seed", str(seed), "--bandwidth", "10G", "--output-dir", str(folder), "--name", "task_websearch", "--no-json"], cwd=ROOT, check=True)
    render_runtime_conf(template, flow, ROOT / "simulation/output/aba/websearch_rewards_template", conf, 400)
    atomic_json(manifest_file, {"registration": registration, "flow_sha256": sha256_file(flow), "conf_sha256": sha256_file(conf)})
    return conf, flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "smoke", "all", "run", "analyze"), default="all")
    parser.add_argument("--run-id-prefix", default="acc_websearch_rewards_s1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--port-base", type=int, default=8200)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.stop_after = args.stop_after or args.epochs
    if not 1 <= args.stop_after <= args.epochs <= 100:
        parser.error("require 1 <= stop-after <= epochs <= 100")
    if not 1024 <= args.port_base <= 65534:
        parser.error("port-base must leave two usable ports")
    if Path(args.run_id_prefix).name != args.run_id_prefix or args.run_id_prefix in (".", ".."):
        parser.error("run-id-prefix must be a directory name")
    conf, flow = prepare_inputs(args.seed)
    base = load_config(ROOT / "configs/aba/webserver_cachefollower.yaml")
    base.update({"route": "WWW", "runtime_root": str(conf.parent.relative_to(ROOT) / "runtime")})
    for slot in ("a", "b"):
        base.update({f"task_{slot}": "realistic_websearch", f"task_{slot}_conf": str(conf.relative_to(ROOT)), f"task_{slot}_flow": str(flow.relative_to(ROOT))})
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_variant, args, base, arm, index) for index, arm in enumerate(ARMS)]
        for future in futures:
            future.result()
    if args.stage in ("all", "analyze", "smoke"):
        prefix = args.run_id_prefix + ("_smoke" if args.stage == "smoke" else "")
        subprocess.run([sys.executable, str(ROOT / "tools/analysis/analyze_acc_reward_action_pilot.py"), "--run-id-prefix", prefix, "--variants", ",".join(ARMS)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
