#!/usr/bin/env python3
"""
综合分析脚本：对比 acc 与 sor 实验在 Hadoop_Shuffle 负载下的输出结果。

适配以下分析工具：
  1. analysis_fct_slowdown.py  - FCT slowdown 统计
  2. com_fct.py                - FCT 综合对比图 (以 acc 为基准)
  3. fct_time.py               - FCT 随时间变化
  4. queue_time.py             - 队列大小分析
  5. rate_time.py              - TxRate / ECNRate 分析
  6. analysis_throughput.py    - 吞吐量分析

输出目录：/root/paddlejob/workspace/yangziwen/CoPTER/tools/analysis/sor_acc
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import re
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
from datetime import datetime

# ====================================================================
# 全局配置
# ====================================================================
SIM_OUTPUT_DIR  = "/root/paddlejob/workspace/yangziwen/CoPTER/simulation/output/Hadoop_Shuffle"
ANALYSIS_DIR    = "/root/paddlejob/workspace/yangziwen/CoPTER/tools/analysis"
OUTPUT_DIR      = os.path.join(ANALYSIS_DIR, "sor_acc_new")
EXPERIMENT      = "Hadoop_Shuffle"
METHODS         = ["acc", "sor"]
BASELINE        = "acc"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- 样式 ----
NAME_MAPPING = {
    "acc":    "ACC",
    "sor":    "SOR",
    "copter": "CoPT",
    "m3":     "SCoPE",
    "m4":     "m4",
    "dcqcn":  "SECN1",
    "hpcc":   "SECN2",
}
COLOR_MAP = {
    "acc":    "#00CC66",
    "sor":    "#0066FF",
    "copter": "#FF6B00",
    "m3":     "#9933FF",
    "m4":     "#FF3333",
    "dcqcn":  "#9933FF",
    "hpcc":   "#FF3333",
}
MARKERS = {
    "acc":    "^",
    "sor":    "s",
    "copter": "o",
}
LINE_STYLES = {
    "acc":    "-",
    "sor":    "--",
    "copter": "-",
}
HATCHES = {
    "acc":    "\\\\",
    "sor":    "||",
    "copter": "//",
}

FONT_SIZE   = 24
LEGEND_SIZE = 20
FIG_SIZE    = (14, 8)

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'DejaVu Serif'],
    'font.size':          FONT_SIZE,
    'axes.labelsize':     FONT_SIZE,
    'axes.titlesize':     FONT_SIZE,
    'legend.fontsize':    LEGEND_SIZE,
    'xtick.labelsize':    FONT_SIZE,
    'ytick.labelsize':    FONT_SIZE,
    'axes.unicode_minus': False,
    'axes.linewidth':     1.5,
    'grid.linestyle':     '--',
    'grid.alpha':         0.6,
    'figure.dpi':         150,
    'text.usetex':        False,
})


def sim_file(method, ext):
    return os.path.join(SIM_OUTPUT_DIR, f"{method}_{EXPERIMENT}.{ext}")


def out_file(name):
    return os.path.join(OUTPUT_DIR, name)


def method_style(m):
    return (
        NAME_MAPPING.get(m, m),
        COLOR_MAP.get(m, "#888888"),
        MARKERS.get(m, "o"),
        LINE_STYLES.get(m, "-"),
        HATCHES.get(m, "x"),
    )


# ====================================================================
# 1. FCT Slowdown Analysis  (analysis_fct_slowdown.py)
# ====================================================================

def get_pctl(a, p):
    i = int(len(a) * p)
    i = min(i, len(a) - 1)
    return a[i]


def process_fct_file(file_path):
    """读取 .fct 文件，返回 [(slowdown, flow_size), ...] 按 flow_size 排序。"""
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 8:
                continue
            flow_size   = int(fields[4])
            fct         = int(fields[6])
            ideal_fct   = int(fields[7])
            slowdown    = max(1.0, fct / ideal_fct)
            result.append([slowdown, flow_size])
    result.sort(key=lambda x: x[1])
    return result


def analyze_fct_file(file_path, step=5):
    data    = process_fct_file(file_path)
    n_flows = len(data)

    fct_all = sorted(x[0] for x in data)
    all_result = [
        np.average(fct_all),
        get_pctl(fct_all, 0.5),
        get_pctl(fct_all, 0.95),
        get_pctl(fct_all, 0.99),
    ]

    step_result = [[i / 100.0] for i in range(0, 100, step)]
    for i in range(0, 100, step):
        l = i * n_flows // 100
        r = (i + step) * n_flows // 100
        if l >= n_flows or r > n_flows:
            continue
        d = data[l:r]
        fct_sub = sorted(x[0] for x in d)
        flow_size = d[-1][1]
        idx = i // step
        step_result[idx] += [flow_size,
                              np.average(fct_sub),
                              get_pctl(fct_sub, 0.5),
                              get_pctl(fct_sub, 0.95),
                              get_pctl(fct_sub, 0.99)]
    return all_result, step_result


def analyze_size_groups(file_path):
    data = process_fct_file(file_path)
    small, large = [], []
    small_1, large_1 = [], []
    for f in data:
        if f[1] < 100_000:
            small.append(f[0])
        if f[1] > 10_000_000:
            large.append(f[0])
        if f[1] < 10_000:
            small_1.append(f[0])
        if f[1] > 1_000_000:
            large_1.append(f[0])

    def stats(lst):
        if not lst:
            return [0, 0, 0, 0]
        s = sorted(lst)
        return [np.average(s), get_pctl(s, 0.5), get_pctl(s, 0.95), get_pctl(s, 0.99)]

    return stats(small), stats(large), stats(small_1), stats(large_1)


def run_fct_slowdown():
    """
    阶段1：处理 .fct 文件，将 slowdown 统计摘要写入 OUTPUT_DIR/<method>_Hadoop_Shuffle.fct
    供 com_fct.py 逻辑读取。
    """
    print("\n" + "="*60)
    print("【阶段1】FCT Slowdown 统计")
    print("="*60)

    for method in METHODS:
        fct_path = sim_file(method, "fct")
        if not os.path.exists(fct_path):
            print(f"  [跳过] 文件不存在: {fct_path}")
            continue

        all_result, step_result = analyze_fct_file(fct_path)
        small, large, small_1, large_1 = analyze_size_groups(fct_path)

        out_path = out_file(f"{method}_{EXPERIMENT}.fct")
        with open(out_path, 'w') as f:
            f.write("Overall FCT: \tAvg\tMid\t95th\t99th\n")
            f.write(f"\t\t{all_result[0]:.3f}\t{all_result[1]:.3f}\t"
                    f"{all_result[2]:.3f}\t{all_result[3]:.3f}\t\n")
            f.write("Percent\tSize\tAvg\tMid\t95th\t99th\n")
            for item in step_result:
                if len(item) < 6:
                    continue
                f.write(f"{item[0]:.3f}\t{item[1]:2.3f}\t{item[2]:.3f}\t"
                        f"{item[3]:.3f}\t{item[4]:.3f}\t{item[5]:.3f}\t\n")
            f.write(f"< 100KB: Avg: {small[0]:.3f}, Mid: {small[1]:.3f}, "
                    f"95th: {small[2]:.3f}, 99th: {small[3]:.3f}\n")
            f.write(f"> 10MB: Avg: {large[0]:.3f}, Mid: {large[1]:.3f}, "
                    f"95th: {large[2]:.3f}, 99th: {large[3]:.3f}\n")
            f.write(f"< 10KB: Avg: {small_1[0]:.3f}, Mid: {small_1[1]:.3f}, "
                    f"95th: {small_1[2]:.3f}, 99th: {small_1[3]:.3f}\n")
            f.write(f"> 1MB: Avg: {large_1[0]:.3f}, Mid: {large_1[1]:.3f}, "
                    f"95th: {large_1[2]:.3f}, 99th: {large_1[3]:.3f}\n")

        print(f"  [OK] {method}: Avg={all_result[0]:.3f}, p95={all_result[2]:.3f}, "
              f"p99={all_result[3]:.3f}  ->  {out_path}")


# ====================================================================
# 2. FCT Comparison Charts  (com_fct.py, baseline=acc)
# ====================================================================

def _clean(s):
    return s.strip().strip(',').strip()


def parse_fct_summary(file_path):
    """解析 run_fct_slowdown() 输出的摘要文件，返回 dict。"""
    with open(file_path, 'r') as f:
        content = f.read()
    lines = [l.strip() for l in content.split('\n') if l.strip()]

    # Overall
    m = re.search(
        r"Overall FCT:\s+Avg\s+Mid\s+95th\s+99th\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",
        content, re.IGNORECASE
    )
    if not m:
        for i, line in enumerate(lines):
            if "Overall FCT:" in line and "Avg" in line:
                if i + 1 < len(lines):
                    vals = re.findall(r"\S+", lines[i + 1])
                    if len(vals) >= 4:
                        m = type('m', (), {'groups': lambda s=vals: tuple(s[:4])})()
                        break
    overall = {
        "Avg":  float(_clean(m.groups()[0])),
        "Mid":  float(_clean(m.groups()[1])),
        "95th": float(_clean(m.groups()[2])),
        "99th": float(_clean(m.groups()[3])),
    }

    # Percent data
    ph = None
    for i, line in enumerate(lines):
        if "Percent" in line and "Size" in line and "Avg" in line:
            ph = i; break
    markers_sz = ["< 100KB", "> 10MB", "< 10KB", "> 1MB"]
    pe = None
    if ph is not None:
        for i in range(ph + 1, len(lines)):
            if any(k in lines[i] for k in markers_sz):
                pe = i; break
    pe = pe or len(lines)

    percent_data = []
    if ph is not None:
        for line in lines[ph + 1:pe]:
            vals = re.findall(r"\d+\.?\d*", line)
            if len(vals) < 6:
                continue
            percent_data.append({
                "Percent": float(vals[0]), "Size": float(vals[1]),
                "Avg": float(vals[2]),     "Mid":  float(vals[3]),
                "95th": float(vals[4]),    "99th": float(vals[5]),
            })

    # Size groups
    sz_lines = [l for l in lines if any(k in l for k in markers_sz)]
    size_groups = {}
    for line in sz_lines:
        for k in markers_sz:
            if k in line:
                vals = re.findall(r"\d+\.\d+", line)
                if len(vals) >= 4:
                    size_groups[k] = {
                        "Avg": float(vals[0]), "Mid":  float(vals[1]),
                        "95th": float(vals[2]), "99th": float(vals[3]),
                    }
                break

    return {"overall": overall,
            "percent_data": pd.DataFrame(percent_data),
            "size_groups": size_groups}


def run_fct_comparison():
    """阶段2：FCT 综合对比图（以 acc 为基准）。"""
    print("\n" + "="*60)
    print("【阶段2】FCT 综合对比图")
    print("="*60)

    results = {}
    for method in METHODS:
        summary_path = out_file(f"{method}_{EXPERIMENT}.fct")
        if not os.path.exists(summary_path):
            print(f"  [跳过] 摘要文件不存在: {summary_path}（请先运行阶段1）")
            continue
        results[method] = parse_fct_summary(summary_path)
        print(f"  [OK] 解析 {method}")

    if BASELINE not in results:
        print(f"  [错误] 基准方法 {BASELINE} 数据缺失，跳过对比图")
        return
    if len(results) < 2:
        print("  [错误] 有效数据不足2个方法，跳过对比图")
        return

    baseline = results[BASELINE]

    # ---- 2a. 整体 FCT 对比柱状图 ----
    metrics = ["Avg", "95th", "99th"]
    x = np.arange(len(metrics))
    width = 0.35
    n_methods = len(results)
    offsets = np.linspace(-(n_methods - 1) * width / 2,
                           (n_methods - 1) * width / 2, n_methods)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0] * 1.5, FIG_SIZE[1]))

    for ax_idx, (ylabel, value_key) in enumerate([
        ("FCT Slowdown", "Value"),
        ("Normalized FCT Slowdown (vs ACC)", "Relative"),
    ]):
        ax = axes[ax_idx]
        for i, (method, res) in enumerate(results.items()):
            display, color, _, _, hatch = method_style(method)
            vals = [res["overall"][m] for m in metrics]
            if value_key == "Relative":
                vals = [v / baseline["overall"][m] for v, m in zip(vals, metrics)]
            bars = ax.bar(x + offsets[i], vals, width * 0.9,
                          label=display,
                          edgecolor=color, facecolor='none',
                          linewidth=2, hatch=hatch)
        if value_key == "Relative":
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_file("fct_overall_comparison.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] fct_overall_comparison.pdf")

    # ---- 2b. 按百分位分组 FCT 曲线 ----
    pct_vals = results[BASELINE]["percent_data"]["Percent"].values
    for metric in ["Avg", "95th", "99th"]:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for method, res in results.items():
            display, color, marker, ls, _ = method_style(method)
            df = res["percent_data"]
            if len(df) == len(pct_vals):
                ax.plot(pct_vals, df[metric],
                        label=display, color=color,
                        linestyle=ls, marker=marker,
                        markersize=6, linewidth=2)
        ax.set_xlabel("Flow Percentile (%)")
        ax.set_ylabel(f"{metric} FCT Slowdown")
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        fn = f"fct_percent_{metric.lower()}.pdf"
        plt.savefig(out_file(fn), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {fn}")

    # ---- 2c. 按流大小分组 FCT ----
    size_groups = ["< 10KB", "< 100KB", "> 1MB", "> 10MB"]
    sg_metrics = ["Avg", "95th", "99th"]
    x2 = np.arange(len(sg_metrics))
    fig, axes = plt.subplots(1, len(size_groups), figsize=(6 * len(size_groups), FIG_SIZE[1]))
    if len(size_groups) == 1:
        axes = [axes]
    for ax, sg in zip(axes, size_groups):
        for i, (method, res) in enumerate(results.items()):
            if sg not in res["size_groups"]:
                continue
            display, color, _, _, hatch = method_style(method)
            vals = [res["size_groups"][sg][m] for m in sg_metrics]
            ax.bar(x2 + offsets[i], vals, width * 0.9,
                   label=display, edgecolor=color, facecolor='none',
                   linewidth=2, hatch=hatch)
        ax.set_title(sg)
        ax.set_xticks(x2)
        ax.set_xticklabels(sg_metrics)
        ax.set_ylabel("FCT Slowdown")
        ax.legend(frameon=False, fontsize=14)
        ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("fct_size_group_comparison.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] fct_size_group_comparison.pdf")


# ====================================================================
# 3. FCT Time Series  (fct_time.py)
# ====================================================================

def run_fct_time():
    """阶段3：FCT 随时间变化曲线（avg / p99 FCT slowdown over time）。"""
    print("\n" + "="*60)
    print("【阶段3】FCT 时间序列分析")
    print("="*60)

    bucket_ns = 1_000_000   # 1ms bucket

    all_avgs, all_p99s = {}, {}

    for method in METHODS:
        fct_path = sim_file(method, "fct")
        if not os.path.exists(fct_path):
            print(f"  [跳过] {fct_path}")
            continue

        flows = []
        with open(fct_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                flow_size    = int(parts[4])
                start_ns     = int(parts[5])
                actual_fct   = int(parts[6])
                ideal_fct    = int(parts[7])
                slowdown     = max(1.0, actual_fct / ideal_fct)
                flows.append((start_ns, slowdown))

        buckets = {}
        for start_ns, slowdown in flows:
            bi = start_ns // bucket_ns
            buckets.setdefault(bi, []).append(slowdown)

        avg_pts, p99_pts = [], []
        for bi, vals in sorted(buckets.items()):
            t_s = bi * bucket_ns / 1e9
            avg_pts.append((t_s, np.mean(vals)))
            p99_pts.append((t_s, np.percentile(vals, 99)))

        all_avgs[method] = np.array(avg_pts)
        all_p99s[method] = np.array(p99_pts)
        print(f"  [OK] {method}: {len(flows)} flows, {len(buckets)} buckets")

    if not all_avgs:
        print("  [错误] 无有效数据")
        return

    for label, data_dict in [("avg", all_avgs), ("p99", all_p99s)]:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        for method, arr in data_dict.items():
            display, color, marker, ls, _ = method_style(method)
            ax.plot(arr[:, 0], arr[:, 1],
                    label=display, color=color,
                    linestyle=ls, linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{'Average' if label == 'avg' else '99th Pct'} FCT Slowdown")
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        fn = f"fct_time_{label}.pdf"
        plt.savefig(out_file(fn), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  [OK] {fn}")

    # Combined plot
    fig, axes = plt.subplots(1, 2, figsize=(FIG_SIZE[0] * 1.5, FIG_SIZE[1]))
    for ax, (label, data_dict) in zip(axes, [("avg", all_avgs), ("p99", all_p99s)]):
        for method, arr in data_dict.items():
            display, color, marker, ls, _ = method_style(method)
            ax.plot(arr[:, 0], arr[:, 1],
                    label=display, color=color,
                    linestyle=ls, linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{'Average' if label == 'avg' else '99th Pct'} FCT Slowdown")
        ax.legend(frameon=False)
        ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("fct_time_combined.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] fct_time_combined.pdf")


# ====================================================================
# 4. Queue Analysis  (queue_time.py)
# ====================================================================

@dataclass
class PortQueueData:
    switch_id:       int
    switch_buffer:   int
    port_id:         int
    queue_size:      int
    monitor_time_s:  float


def parse_queue_file(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                records.append(PortQueueData(
                    switch_id=int(parts[0]),
                    switch_buffer=int(parts[1]),
                    port_id=int(parts[2]),
                    queue_size=int(parts[3]),
                    monitor_time_s=float(parts[4]),
                ))
            except ValueError:
                pass
    return records


def process_queue_file(file_path):
    records = parse_queue_file(file_path)
    if not records:
        return None

    all_sizes = [r.queue_size for r in records]
    overall_avg = np.mean(all_sizes)
    overall_std = np.std(all_sizes)

    time_buckets = {}
    for r in records:
        time_buckets.setdefault(r.monitor_time_s, []).append(r)

    avg_q, p99_q = [], []
    for t, bucket in time_buckets.items():
        sizes = [b.queue_size for b in bucket]
        avg_q.append((t, np.mean(sizes)))
        p99_q.append((t, np.percentile(sizes, 99)))

    avg_q.sort()
    p99_q.sort()
    return np.array(avg_q), np.array(p99_q), overall_avg, overall_std


def run_queue_analysis():
    """阶段4：队列大小随时间变化及归一化对比。"""
    print("\n" + "="*60)
    print("【阶段4】队列大小分析")
    print("="*60)

    file_results = {}
    for method in METHODS:
        q_path = sim_file(method, "queue")
        if not os.path.exists(q_path):
            print(f"  [跳过] {q_path}")
            continue
        res = process_queue_file(q_path)
        if res:
            file_results[method] = res
            avg_q, _, oa, os_ = res
            print(f"  [OK] {method}: avg_queue={oa:.1f} B, std={os_:.1f} B")

    if not file_results:
        print("  [错误] 无有效队列数据")
        return

    # ---- 4a. 绝对队列大小曲线 ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for method, (avg_arr, p99_arr, _, _) in file_results.items():
        display, color, marker, ls, _ = method_style(method)
        ax.plot(avg_arr[:, 0], avg_arr[:, 1],
                label=display, color=color, linestyle=ls, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg Queue Size (Bytes)")
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("queue_avg_time.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] queue_avg_time.pdf")

    # ---- 4b. 归一化队列大小（以 acc 为基准）----
    if BASELINE not in file_results:
        print(f"  [警告] 基准 {BASELINE} 不在结果中，跳过归一化图")
        return

    base_arr = file_results[BASELINE][0]
    base_times = base_arr[:, 0]
    base_vals  = base_arr[:, 1]

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for method, (avg_arr, _, _, _) in file_results.items():
        display, color, marker, ls, _ = method_style(method)
        interp = np.interp(base_times, avg_arr[:, 0], avg_arr[:, 1])
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_vals = np.where(base_vals != 0, interp / base_vals, 1.0)
        ax.plot(base_times, norm_vals,
                label=display, color=color, linestyle=ls, linewidth=2)
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Queue Size (vs ACC)")
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("queue_normalized.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] queue_normalized.pdf")

    # ---- 4c. 统计汇总 ----
    print("\n  队列统计:")
    print(f"  {'Method':<8}  {'AvgQueue(B)':>14}  {'Std(B)':>12}")
    print("  " + "-"*38)
    for method, (_, _, oa, os_) in sorted(file_results.items(),
                                           key=lambda x: x[1][2]):
        display = NAME_MAPPING.get(method, method)
        print(f"  {display:<8}  {oa:>14.2f}  {os_:>12.2f}")


# ====================================================================
# 5. Rate Analysis  (rate_time.py)
# ====================================================================

@dataclass
class PortMonitor:
    switch_id:       int
    port_id:         int
    maxrate:         int
    txrate:          float
    ecnrate:         float
    monitor_time_s:  float


def parse_rate_file(file_path, skip_initial=2):
    records = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                records.append(PortMonitor(
                    switch_id=int(parts[0]),
                    port_id=int(parts[1]),
                    maxrate=int(parts[2]),
                    txrate=float(parts[3]),
                    ecnrate=float(parts[4]),
                    monitor_time_s=float(parts[5]),
                ))
            except ValueError:
                pass

    if len(records) <= skip_initial:
        return None

    time_buckets = {}
    for r in records:
        time_buckets.setdefault(r.monitor_time_s, []).append(r)

    avg_tx, p99_tx, avg_ecn, p99_ecn = [], [], [], []
    for t, bucket in time_buckets.items():
        txs  = [b.txrate  for b in bucket]
        ecns = [b.ecnrate for b in bucket]
        avg_tx.append((t,  np.mean(txs)))
        p99_tx.append((t,  np.percentile(txs, 99)))
        avg_ecn.append((t, np.mean(ecns)))
        p99_ecn.append((t, np.percentile(ecns, 99)))

    for lst in (avg_tx, p99_tx, avg_ecn, p99_ecn):
        lst.sort()

    return (np.array(avg_tx)[skip_initial:],
            np.array(p99_tx)[skip_initial:],
            np.array(avg_ecn)[skip_initial:],
            np.array(p99_ecn)[skip_initial:])


def run_rate_analysis():
    """阶段5：TxRate / ECNRate 随时间变化及归一化对比。"""
    print("\n" + "="*60)
    print("【阶段5】速率（TxRate / ECNRate）分析")
    print("="*60)

    file_results = {}
    for method in METHODS:
        tr_path = sim_file(method, "txrate")
        if not os.path.exists(tr_path):
            print(f"  [跳过] {tr_path}")
            continue
        res = parse_rate_file(tr_path)
        if res:
            file_results[method] = res
            avg_tx = res[0]
            print(f"  [OK] {method}: mean_txrate={np.mean(avg_tx[:, 1]):.4f}")

    if not file_results:
        print("  [错误] 无有效速率数据")
        return

    # ---- 5a. TxRate & ECNRate 曲线（2 subplots）----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_SIZE[0], FIG_SIZE[1] * 1.5))
    for method, (avg_tx, _, avg_ecn, _) in file_results.items():
        display, color, marker, ls, _ = method_style(method)
        ax1.plot(avg_tx[:, 0], avg_tx[:, 1],
                 label=display, color=color, linestyle=ls, linewidth=2)
        ax2.plot(avg_ecn[:, 0], avg_ecn[:, 1],
                 label=display, color=color, linestyle=ls, linewidth=2)
    ax1.set_ylabel("Avg TxRate (normalized)")
    ax1.legend(frameon=False)
    ax1.grid(axis='y', alpha=0.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Avg ECNRate (normalized)")
    ax2.legend(frameon=False)
    ax2.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("rate_time.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] rate_time.pdf")

    # ---- 5b. 归一化速率柱状图（以 acc 为基准）----
    if BASELINE not in file_results:
        print(f"  [警告] 基准 {BASELINE} 不在结果中，跳过归一化柱状图")
        return

    base_avg_tx  = np.mean(file_results[BASELINE][0][:, 1])
    base_avg_ecn = np.mean(file_results[BASELINE][2][:, 1])

    labels_r  = [NAME_MAPPING.get(m, m) for m in file_results]
    norm_tx_  = [np.mean(v[0][:, 1]) / base_avg_tx
                 for v in file_results.values()]
    norm_ecn_ = [np.mean(v[2][:, 1]) / base_avg_ecn
                 for v in file_results.values()]

    x = np.arange(len(labels_r))
    w = 0.35
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    bars1 = ax.bar(x - w/2, norm_tx_,  w, label='Avg TxRate',
                   color=[COLOR_MAP.get(m, "#888") for m in file_results],
                   alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + w/2, norm_ecn_, w, label='Avg ECNRate',
                   color=[COLOR_MAP.get(m, "#888") for m in file_results],
                   alpha=0.4, edgecolor='black', hatch='//')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_r)
    ax.set_ylabel("Normalized Rate (vs ACC)")
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("rate_normalized_bar.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] rate_normalized_bar.pdf")


# ====================================================================
# 6. Throughput Analysis  (analysis_throughput.py)
# ====================================================================

def load_throughput(file_path):
    try:
        df = pd.read_csv(
            file_path, sep=r'\s+', header=None,
            names=['switch_id', 'node_id', 'throughput_bps', 'timestamp', 'max_port_rate']
        )
        df['throughput_mbps']    = df['throughput_bps']  / 1e6
        df['max_port_rate_mbps'] = df['max_port_rate']   / 1e6
        return df
    except Exception as e:
        print(f"  [错误] 加载 {file_path}: {e}")
        return None


def run_throughput_analysis():
    """阶段6：吞吐量时序、分布及端口利用率。"""
    print("\n" + "="*60)
    print("【阶段6】吞吐量分析")
    print("="*60)

    data_dict = {}
    stats_list = []
    for method in METHODS:
        tp_path = sim_file(method, "throughput")
        if not os.path.exists(tp_path):
            print(f"  [跳过] {tp_path}")
            continue
        df = load_throughput(tp_path)
        if df is not None and not df.empty:
            data_dict[method] = df
            stats_list.append({
                'label':           NAME_MAPPING.get(method, method),
                'avg_mbps':        df['throughput_mbps'].mean(),
                'max_mbps':        df['throughput_mbps'].max(),
                'p95_mbps':        df['throughput_mbps'].quantile(0.95),
                'p99_mbps':        df['throughput_mbps'].quantile(0.99),
                'unique_switches': df['switch_id'].nunique(),
            })
            print(f"  [OK] {method}: avg={df['throughput_mbps'].mean():.2f} Mbps")

    if not data_dict:
        print("  [错误] 无有效吞吐量数据")
        return

    # ---- 6a. 时序曲线 ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for method, df in data_dict.items():
        display, color, _, ls, _ = method_style(method)
        time_avg = (df.sort_values('timestamp')
                      .groupby('timestamp')['throughput_mbps']
                      .mean()
                      .reset_index())
        ax.plot(time_avg['timestamp'], time_avg['throughput_mbps'],
                label=display, color=color, linestyle=ls, linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg Throughput (Mbps)")
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("throughput_time_series.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] throughput_time_series.pdf")

    # ---- 6b. 分布箱线图 ----
    fig, ax = plt.subplots(figsize=(max(6, 3 * len(data_dict)), FIG_SIZE[1]))
    plot_data   = []
    plot_labels = []
    for method, df in data_dict.items():
        nz = df[df['throughput_mbps'] > 0]['throughput_mbps']
        plot_data.append(nz)
        plot_labels.append(NAME_MAPPING.get(method, method))
    bp = ax.boxplot(plot_data, labels=plot_labels, showfliers=False,
                    patch_artist=True)
    for patch, method in zip(bp['boxes'], data_dict.keys()):
        patch.set_facecolor(COLOR_MAP.get(method, "#888888"))
        patch.set_alpha(0.6)
    ax.set_ylabel("Throughput (Mbps)")
    ax.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("throughput_distribution.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] throughput_distribution.pdf")

    # ---- 6c. 端口利用率散点图 ----
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for method, df in data_dict.items():
        display, color, marker, _, _ = method_style(method)
        port_stats = df.groupby(['switch_id', 'node_id']).agg(
            throughput_mbps=('throughput_mbps', 'mean'),
            max_port_rate_mbps=('max_port_rate_mbps', 'first'),
        ).reset_index()
        port_stats['util_pct'] = (
            port_stats['throughput_mbps'] / port_stats['max_port_rate_mbps'] * 100
        )
        ax.scatter(port_stats.index, port_stats['util_pct'],
                   alpha=0.6, label=f"{display} Utilization",
                   color=color, marker=marker)
    ax.axhline(100, color='red', linestyle='--', linewidth=1.5, label='100% limit')
    ax.set_xlabel("Port Index")
    ax.set_ylabel("Utilization (%)")
    ax.legend(frameon=False, fontsize=14)
    ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_file("throughput_port_utilization.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] throughput_port_utilization.pdf")

    # ---- 6d. 统计报告 ----
    report_path = out_file("throughput_statistics.txt")
    with open(report_path, 'w') as f:
        f.write(f"Throughput Analysis Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        for s in stats_list:
            f.write(f"Dataset: {s['label']}\n")
            f.write(f"  Avg Throughput:  {s['avg_mbps']:.2f} Mbps\n")
            f.write(f"  Max Throughput:  {s['max_mbps']:.2f} Mbps\n")
            f.write(f"  p95 Throughput:  {s['p95_mbps']:.2f} Mbps\n")
            f.write(f"  p99 Throughput:  {s['p99_mbps']:.2f} Mbps\n")
            f.write(f"  Unique Switches: {s['unique_switches']}\n\n")
    print(f"  [OK] throughput_statistics.txt")

    # 打印到控制台
    print("\n  吞吐量统计:")
    print(f"  {'Method':<8}  {'Avg(Mbps)':>12}  {'Max(Mbps)':>12}  {'p99(Mbps)':>12}")
    print("  " + "-"*48)
    for s in stats_list:
        print(f"  {s['label']:<8}  {s['avg_mbps']:>12.2f}  {s['max_mbps']:>12.2f}  {s['p99_mbps']:>12.2f}")


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"CoPTER 分析：ACC vs SOR — {EXPERIMENT}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"{'='*60}")

    run_fct_slowdown()       # 1. analysis_fct_slowdown.py
    run_fct_comparison()     # 2. com_fct.py  (acc 为基准)
    run_fct_time()           # 3. fct_time.py
    run_queue_analysis()     # 4. queue_time.py
    run_rate_analysis()      # 5. rate_time.py
    run_throughput_analysis()# 6. analysis_throughput.py

    print(f"\n{'='*60}")
    print(f"全部分析完成！结果保存至：{OUTPUT_DIR}")
    print(f"{'='*60}")
    print("\n生成的文件列表：")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")
