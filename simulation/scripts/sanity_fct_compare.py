#!/usr/bin/env python3
"""对比 force_action sanity 实验三组固定动作的 FCT（仅公共完成流）。"""
import os
import sys

def load_fct(path):
    flows = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            key = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            flows[key] = (int(parts[6]), int(parts[7]))
    return flows

def pct(vals, p):
    idx = min(len(vals) - 1, int(round(p / 100.0 * (len(vals) - 1))))
    return vals[idx]

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "output/sanity"
    tags = ["good", "bad", "aggr"]
    data = {}
    for t in tags:
        p = os.path.join(base, f"fct_{t}.fct")
        if os.path.exists(p):
            data[t] = load_fct(p)
    common = set.intersection(*[set(d.keys()) for d in data.values()])
    print(f"公共完成流: {len(common)}")
    print(f"{'tag':6s} {'avgFCT(us)':>11s} {'p50(us)':>9s} {'p95(us)':>9s} {'p99(us)':>10s} {'avgSlow':>8s}")
    results = {}
    for t, d in data.items():
        fcts = sorted(d[k][0] for k in common)
        slows = sorted(d[k][0] / max(d[k][1], 1) for k in common)
        n = len(fcts)
        st = {
            "avg": sum(fcts) / n / 1e3,
            "p50": pct(fcts, 50) / 1e3,
            "p95": pct(fcts, 95) / 1e3,
            "p99": pct(fcts, 99) / 1e3,
            "slow": sum(slows) / n,
        }
        results[t] = st
        print(f"{t:6s} {st['avg']:>11.1f} {st['p50']:>9.1f} {st['p95']:>9.1f} {st['p99']:>10.1f} {st['slow']:>8.2f}")
    g = results.get("good")
    for t in ["bad", "aggr"]:
        if t in results and g:
            r = results[t]
            print(f"  {t} vs good: avgFCT {100*(r['avg']-g['avg'])/g['avg']:+.1f}%  p95 {100*(r['p95']-g['p95'])/g['p95']:+.1f}%  p99 {100*(r['p99']-g['p99'])/g['p99']:+.1f}%")

if __name__ == "__main__":
    main()
