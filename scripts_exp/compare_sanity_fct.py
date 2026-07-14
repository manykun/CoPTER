#!/usr/bin/env python3
"""对比sanity实验中三组固定动作(good/bad/aggr)的FCT, 仅统计公共完成流."""
import sys

BASE = "/root/paddlejob/workspace/yangziwen/CoPTER/simulation/output/sanity"


def load(path):
    flows = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            flows[tuple(p[:6])] = (int(p[6]), int(p[7]))
    return flows


def pct(v, p):
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def main():
    tags = ["good", "bad", "aggr"]
    data = {t: load(f"{BASE}/fct_{t}.fct") for t in tags}
    common = set.intersection(*[set(d) for d in data.values()])
    print("common flows:", len(common))
    print(f"{'tag':6s} {'avgFCT(us)':>11s} {'p50':>8s} {'p95':>9s} {'p99':>10s} {'avgSlow':>8s} {'p99Slow':>8s}")
    res = {}
    for t in tags:
        d = data[t]
        f = sorted(d[k][0] for k in common)
        s = sorted(d[k][0] / max(d[k][1], 1) for k in common)
        n = len(f)
        res[t] = (sum(f) / n / 1e3, pct(f, 50) / 1e3, pct(f, 95) / 1e3, pct(f, 99) / 1e3, sum(s) / n, pct(s, 99))
        r = res[t]
        print(f"{t:6s} {r[0]:>11.1f} {r[1]:>8.1f} {r[2]:>9.1f} {r[3]:>10.1f} {r[4]:>8.2f} {r[5]:>8.2f}")
    g = res["good"]
    for t in ["bad", "aggr"]:
        r = res[t]
        print(f"{t} vs good: avgFCT {100*(r[0]-g[0])/g[0]:+.1f}%  p95 {100*(r[2]-g[2])/g[2]:+.1f}%  p99 {100*(r[3]-g[3])/g[3]:+.1f}%")


if __name__ == "__main__":
    main()
