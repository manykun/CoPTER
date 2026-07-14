#!/usr/bin/env python3
"""中期FCT对比：仅比较在同一场景所有参数组下都完成的相同流。

FCT文件格式: src_ip dst_ip sport dport size(bytes) start_ns fct_ns standalone_fct_ns
流的唯一键: (src_ip, dst_ip, sport, dport, size, start_ns)
"""
import sys
import os

def load_fct(path):
    flows = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            key = (parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
            fct = int(parts[6])
            sfct = int(parts[7])
            flows[key] = (fct, sfct, int(parts[4]))
    return flows

def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]

def stats(flows, keys):
    fcts = sorted(flows[k][0] for k in keys)
    slowdowns = sorted(flows[k][0] / max(flows[k][1], 1) for k in keys)
    n = len(fcts)
    return {
        "n": n,
        "avg_fct_us": sum(fcts) / n / 1e3,
        "p50_fct_us": pct(fcts, 50) / 1e3,
        "p95_fct_us": pct(fcts, 95) / 1e3,
        "p99_fct_us": pct(fcts, 99) / 1e3,
        "avg_slowdown": sum(slowdowns) / n,
        "p99_slowdown": pct(slowdowns, 99),
    }

def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "output/static"
    for scen in ["scenA", "scenB"]:
        paths = {s: os.path.join(base, f"static_{scen}_{s}.fct") for s in ["secn0", "secn1", "secn2"]}
        data = {s: load_fct(p) for s, p in paths.items() if os.path.exists(p)}
        if len(data) < 2:
            print(f"[{scen}] 数据不足，跳过")
            continue
        common = set.intersection(*[set(d.keys()) for d in data.values()])
        print(f"\n===== {scen} 公共完成流: {len(common)} =====")
        header = f"{'param':8s} {'n':>6s} {'avgFCT(us)':>11s} {'p50(us)':>9s} {'p95(us)':>9s} {'p99(us)':>10s} {'avgSlow':>8s} {'p99Slow':>9s}"
        print(header)
        results = {}
        for s, d in data.items():
            st = stats(d, common)
            results[s] = st
            print(f"{s:8s} {st['n']:>6d} {st['avg_fct_us']:>11.1f} {st['p50_fct_us']:>9.1f} {st['p95_fct_us']:>9.1f} {st['p99_fct_us']:>10.1f} {st['avg_slowdown']:>8.2f} {st['p99_slowdown']:>9.2f}")
        if "secn0" in results:
            b = results["secn0"]
            for s in ["secn1", "secn2"]:
                if s in results:
                    r = results[s]
                    print(f"  {s} vs secn0: avgFCT {100*(r['avg_fct_us']-b['avg_fct_us'])/b['avg_fct_us']:+.1f}%  p95 {100*(r['p95_fct_us']-b['p95_fct_us'])/b['p95_fct_us']:+.1f}%  p99 {100*(r['p99_fct_us']-b['p99_fct_us'])/b['p99_fct_us']:+.1f}%")

if __name__ == "__main__":
    main()
