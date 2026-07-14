#!/usr/bin/env python3
"""Parse sanity agent logs, extract PortObservation stats per run, and
recompute reward components to see which one lacks discrimination."""
import re
import sys
import math

PAT = re.compile(r"queue_length_norm=([0-9.e-]+), tx_rate_norm=([0-9.e-]+), ecn_rate_norm=([0-9.e-]+)")

def stats(path):
    q, tx, ecn = [], [], []
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                q.append(float(m.group(1)))
                tx.append(float(m.group(2)))
                ecn.append(float(m.group(3)))
    n = len(q)
    # congested subset: q > 0.005 or ecn > 1e-5
    cong = [(a, b, c) for a, b, c in zip(q, tx, ecn) if a > 0.005 or c > 1e-5]
    def mean(x): return sum(x) / len(x) if x else 0.0
    qs = sorted(q)
    def pct(vals, p):
        if not vals: return 0.0
        return vals[min(len(vals)-1, int(p/100*(len(vals)-1)))]
    print(f"  samples={n} congested={len(cong)} ({100.0*len(cong)/max(n,1):.0f}%)")
    print(f"  all:  qlen mean={mean(q):.4f} p95={pct(qs,95):.4f} max={max(q) if q else 0:.4f}  tx mean={mean(tx):.4f}  ecn mean={mean(ecn):.6f} max={max(ecn) if ecn else 0:.6f}")
    cq = [x[0] for x in cong]; ct = [x[1] for x in cong]; ce = [x[2] for x in cong]
    print(f"  cong: qlen mean={mean(cq):.4f}  tx mean={mean(ct):.4f}  ecn mean={mean(ce):.6f}")
    # reward components with current formula (window ~ single sample approx)
    r_q = mean([math.exp(-6.0 * x) for x in cq])
    r_t = mean([min(1.0, max(0.0, x)) for x in ct])
    r_e = mean([0.6 + 0.4*(e/0.03) if e <= 0.03 else math.exp(-7.0*(e-0.03)) for e in ce])
    print(f"  approx components on congested: r_queue={r_q:.4f} r_tput={r_t:.4f} r_ecn={r_e:.4f}")
    print(f"  weighted total = {0.45*r_q + 0.30*r_t + 0.25*r_e:.4f}")

for tag in ["good", "bad", "aggr"]:
    print(f"=== {tag} ===")
    stats(f"/root/paddlejob/workspace/yangziwen/CoPTER/simulation/output/sanity/agent_{tag}.log")
