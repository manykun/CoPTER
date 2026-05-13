import numpy as np
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
from tqdm import tqdm
import seaborn as sns
import pandas as pd  # 补充导入pandas
from pathlib import Path

# ------------------------------
# 原有代码保持不变（数据结构、解析、绘图函数等）
# ------------------------------
@dataclass
class PortMonitor:
    switch_id: int          # 交换机ID
    port_id: int            # 端口ID
    maxrate: int            # 最大速率
    txrate: float           # 发送速率（归一化）
    ecnrate: float          # ECN标记速率（归一化）
    monitor_time_s: float   # 监控时间（秒）

def parse_rate_line(line: str, line_num: int) -> Union[PortMonitor, None]:
    stripped_line = line.strip()
    if not stripped_line:
        print(f"警告：第{line_num}行是空行，已跳过")
        return None
    
    parts = stripped_line.split()
    if len(parts) < 6:
        print(f"警告：第{line_num}行字段不足6个（实际{len(parts)}个），内容：{stripped_line}")
        return None
    if len(parts) > 6:
        print(f"警告：第{line_num}行字段超过6个（实际{len(parts)}个），将使用前6个字段，内容：{stripped_line}")
    
    try:
        return PortMonitor(
            switch_id=int(parts[0]),
            port_id=int(parts[1]),
            maxrate=int(parts[2]),
            txrate=float(parts[3]),
            ecnrate=float(parts[4]),
            monitor_time_s=float(parts[5])
        )
    except ValueError as e:
        print(f"警告：第{line_num}行数值转换失败 - {str(e)}，内容：{stripped_line}")
        return None

def process_single_rate_file(
    file_path: str,
    skip_initial_points: int = 2
) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None]:
    monitor_records: List[PortMonitor] = []
    
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            record = parse_rate_line(line, line_num)
            if record:
                monitor_records.append(record)
    
    if len(monitor_records) == 0:
        print(f"警告：文件无有效数据 -> {file_path}")
        return None
    if len(monitor_records) <= skip_initial_points:
        print(f"警告：文件数据量不足（{len(monitor_records)}条），无法跳过{skip_initial_points}个初始点 -> {file_path}")
        return None

    time_buckets: Dict[float, List[PortMonitor]] = {}
    for record in monitor_records:
        time_key = record.monitor_time_s
        if time_key not in time_buckets:
            time_buckets[time_key] = []
        time_buckets[time_key].append(record)

    avg_txrate = []
    p99_txrate = []
    avg_ecnrate = []
    p99_ecnrate = []
    
    for time_key, bucket_data in tqdm(time_buckets.items(), desc=f"处理 {os.path.basename(file_path)}"):
        tx_rates = [item.txrate for item in bucket_data]
        ecn_rates = [item.ecnrate for item in bucket_data]
        
        avg_txrate.append((time_key, np.mean(tx_rates)))
        p99_txrate.append((time_key, np.percentile(tx_rates, 99)))
        avg_ecnrate.append((time_key, np.mean(ecn_rates)))
        p99_ecnrate.append((time_key, np.percentile(ecn_rates, 99)))

    avg_txrate.sort(key=lambda x: x[0])
    p99_txrate.sort(key=lambda x: x[0])
    avg_ecnrate.sort(key=lambda x: x[0])
    p99_ecnrate.sort(key=lambda x: x[0])
    
    avg_txrate_arr = np.array(avg_txrate)[skip_initial_points:]
    p99_txrate_arr = np.array(p99_txrate)[skip_initial_points:]
    avg_ecnrate_arr = np.array(avg_ecnrate)[skip_initial_points:]
    p99_ecnrate_arr = np.array(p99_ecnrate)[skip_initial_points:]

    return avg_txrate_arr, p99_txrate_arr, avg_ecnrate_arr, p99_ecnrate_arr

def plot_rate_comparison(
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None
):
    plt.figure(figsize=(14, 10))
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    line_style_tx = '-'
    line_style_ecn = '--'

    plt.subplot(2, 1, 1)
    for idx, (filename, (avg_tx, _, avg_ecn, _)) in enumerate(file_results.items()):
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        plt.plot(
            avg_tx[:, 0], avg_tx[:, 1],
            color=color,
            linestyle=line_style_tx,
            linewidth=2,
            label=f"{file_label} - Avg TxRate"
        )
        
        plt.plot(
            avg_ecn[:, 0], avg_ecn[:, 1],
            color=color,
            linestyle=line_style_ecn,
            linewidth=2,
            label=f"{file_label} - Avg ECNRate"
        )
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Average Rate (normalized)', fontsize=12)
    plt.title(f'{title}\n(Average Rates)', fontsize=13, pad=15)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)

    plt.subplot(2, 1, 2)
    for idx, (filename, (_, p99_tx, _, p99_ecn)) in enumerate(file_results.items()):
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        plt.plot(
            p99_tx[:, 0], p99_tx[:, 1],
            color=color,
            linestyle=line_style_tx,
            linewidth=2,
            label=f"{file_label} - P99 TxRate"
        )
        
        plt.plot(
            p99_ecn[:, 0], p99_ecn[:, 1],
            color=color,
            linestyle=line_style_ecn,
            linewidth=2,
            label=f"{file_label} - P99 ECNRate"
        )
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('99th Percentile Rate (normalized)', fontsize=12)
    plt.title(f'{title}\n(99th Percentile Rates)', fontsize=13, pad=15)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 速率对比图保存：{output_path}")

def plot_rate_vs_baseline(
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    baseline_filename: str,
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None
):
    if baseline_filename not in file_results:
        print(f"警告：基准文件 {baseline_filename} 不在分析结果中，跳过基准对比图")
        return
    
    baseline_avg_tx, baseline_p99_tx, baseline_avg_ecn, baseline_p99_ecn = file_results[baseline_filename]
    baseline_times = baseline_avg_tx[:, 0]

    plt.figure(figsize=(14, 10))
    color_list = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']

    plt.subplot(2, 1, 1)
    for idx, (filename, (avg_tx, _, avg_ecn, _)) in enumerate(file_results.items()):
        if filename == baseline_filename:
            continue
        
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        interp_avg_tx = np.interp(baseline_times, avg_tx[:, 0], avg_tx[:, 1])
        interp_avg_ecn = np.interp(baseline_times, avg_ecn[:, 0], avg_ecn[:, 1])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tx_diff_pct = (interp_avg_tx - baseline_avg_tx[:, 1]) / baseline_avg_tx[:, 1] * 100
            ecn_diff_pct = (interp_avg_ecn - baseline_avg_ecn[:, 1]) / baseline_avg_ecn[:, 1] * 100
            tx_diff_pct[baseline_avg_tx[:, 1] == 0] = 0 if np.all(interp_avg_tx[baseline_avg_tx[:, 1] == 0] == 0) else 100
            ecn_diff_pct[baseline_avg_ecn[:, 1] == 0] = 0 if np.all(interp_avg_ecn[baseline_avg_ecn[:, 1] == 0] == 0) else 100
        
        plt.plot(
            baseline_times, tx_diff_pct,
            color=color,
            linestyle='-',
            linewidth=2,
            label=f"{file_label} - Avg TxRate vs Baseline"
        )
        
        plt.plot(
            baseline_times, ecn_diff_pct,
            color=color,
            linestyle='--',
            linewidth=2,
            label=f"{file_label} - Avg ECNRate vs Baseline"
        )
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Difference from Baseline (%)', fontsize=12)
    plt.title(f'{title}\n(Average Rate Difference)', fontsize=13, pad=15)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)

    plt.subplot(2, 1, 2)
    for idx, (filename, (_, p99_tx, _, p99_ecn)) in enumerate(file_results.items()):
        if filename == baseline_filename:
            continue
        
        color = color_list[idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        
        interp_p99_tx = np.interp(baseline_times, p99_tx[:, 0], p99_tx[:, 1])
        interp_p99_ecn = np.interp(baseline_times, p99_ecn[:, 0], p99_ecn[:, 1])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tx_diff_pct = (interp_p99_tx - baseline_p99_tx[:, 1]) / baseline_p99_tx[:, 1] * 100
            ecn_diff_pct = (interp_p99_ecn - baseline_p99_ecn[:, 1]) / baseline_p99_ecn[:, 1] * 100
            tx_diff_pct[baseline_p99_tx[:, 1] == 0] = 0 if np.all(interp_p99_tx[baseline_p99_tx[:, 1] == 0] == 0) else 100
            ecn_diff_pct[baseline_p99_ecn[:, 1] == 0] = 0 if np.all(interp_p99_ecn[baseline_p99_ecn[:, 1] == 0] == 0) else 100
        
        plt.plot(
            baseline_times, tx_diff_pct,
            color=color,
            linestyle='-',
            linewidth=2,
            label=f"{file_label} - P99 TxRate vs Baseline"
        )
        
        plt.plot(
            baseline_times, ecn_diff_pct,
            color=color,
            linestyle='--',
            linewidth=2,
            label=f"{file_label} - P99 ECNRate vs Baseline"
        )
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Difference from Baseline (%)', fontsize=12)
    plt.title(f'{title}\n(99th Percentile Rate Difference)', fontsize=13, pad=15)
    plt.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    if xlim:
        plt.xlim(*xlim)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 基准对比图保存：{output_path}")

# ------------------------------
# 修复：TxRate/ECNRate 归一化对比柱状图（PDF格式）
# ------------------------------
def plot_normalized_rate_bar_chart(
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    baseline_filename: str,
    output_dir: str
):
    """
    生成TxRate和ECNRate平均值的归一化对比柱状图（以copter为基准）
    样式与参考代码保持一致，保存为PDF格式
    """
    # 配置全局样式（与参考代码一致）
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Computer Modern Roman'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.unicode_minus': False,
        'axes.linewidth': 1.0,
        'grid.linestyle': '--',
        'grid.alpha': 0.6,
        'figure.dpi': 300,
        'text.usetex': False,
    })
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")

    # 样式配置（与参考代码完全一致）
    color_map = {
        "copter": "#FF6B00",
        "m4": "#E60023",
        "m3": "#0066FF",
        "acc": "#00CC66",
        "dcqcn": "#9933FF",
        "hpcc": "#FFCC00"
    }
    hatches = {
        "copter": '||', "m4": 'xx', "m3": '++', "acc": '\\', "dcqcn": 'x', "hpcc": '+'
    }
    name_mapping = {
        "copter": "CoPTER",
        "m3": "m3",
        "m4": "m4",
        "acc": "ACC",
        "dcqcn": r"$SECN_1$",
        "hpcc": r"$SECN_2$"
    }

    # 1. 单独提取基准文件的平均速率（确保先初始化基准值）
    baseline_avg_tx = None
    baseline_avg_ecn = None
    baseline_method = baseline_filename.split('_')[0].lower()
    
    if baseline_filename in file_results:
        avg_tx_arr, _, avg_ecn_arr, _ = file_results[baseline_filename]
        baseline_avg_tx = np.mean(avg_tx_arr[:, 1])
        baseline_avg_ecn = np.mean(avg_ecn_arr[:, 1])
        print(f"📊 基准文件（{baseline_method}）统计：Avg TxRate={baseline_avg_tx:.4f}, Avg ECNRate={baseline_avg_ecn:.4f}")
    else:
        print(f"⚠️  基准文件 {baseline_filename} 未找到，无法生成柱状图")
        return
    
    # 检查基准值有效性
    if baseline_avg_tx is None or baseline_avg_ecn is None:
        print(f"⚠️  基准值获取失败，无法生成柱状图")
        return
    
    # 2. 计算所有文件的平均速率和归一化值
    rate_data = []
    for filename, (avg_tx_arr, _, avg_ecn_arr, _) in file_results.items():
        method_name = filename.split('_')[0].lower()
        if method_name not in color_map:
            method_name = "unknown"
            print(f"⚠️  未知方法名：{filename}，使用默认样式")
        
        # 计算整体平均速率
        overall_avg_tx = np.mean(avg_tx_arr[:, 1])
        overall_avg_ecn = np.mean(avg_ecn_arr[:, 1])
        
        # 计算归一化值
        norm_tx = overall_avg_tx / baseline_avg_tx if baseline_avg_tx != 0 else 0.0
        norm_ecn = overall_avg_ecn / baseline_avg_ecn if baseline_avg_ecn != 0 else 0.0
        
        # 添加数据
        rate_data.append({
            "Method": method_name,
            "Rate Type": "Avg TxRate",
            "Value": overall_avg_tx,
            "Normalized Value": norm_tx
        })
        # rate_data.append({
        #     "Method": method_name,
        #     "Rate Type": "Avg ECNRate",
        #     "Value": overall_avg_ecn,
        #     "Normalized Value": norm_ecn
        # })

    # -------------------------- 修复核心：转换为DataFrame --------------------------
    df_rate = pd.DataFrame(rate_data)  # 列表转换为DataFrame
    
    # 3. 绘制柱状图（PDF格式，适配论文双栏）
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    sns.barplot(
        x="Rate Type", 
        y="Normalized Value", 
        hue="Method", 
        data=df_rate,  # 传入DataFrame
        palette=color_map,
        ax=ax,
        edgecolor='black'
    )

    # 应用样式（空心+边框+填充图案）
    for i, bar in enumerate(ax.containers):
        method_name = bar.get_label().lower()
        # 匹配颜色和图案
        color = color_map.get(method_name, "#999999")
        hatch = hatches.get(method_name, '')
        
        for patch in bar.patches:
            patch.set_facecolor('none')
            patch.set_edgecolor(color)
            patch.set_linewidth(2)
            patch.set_hatch(hatch)
            patch.set_alpha(1.0)

    # 4. 图表美化
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='CoPTER Baseline')
    ax.set_ylabel("Normalized Value (vs CoPTER)", fontsize=12)
    ax.set_xlabel("")
    ax.set_title("")
    ax.grid(axis='y', linestyle='', alpha=0.7)

    # 替换图例
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == 'CoPTER Baseline':
            new_labels.append(label)
        else:
            new_labels.append(name_mapping.get(label.lower(), label))
    ax.legend(handles=handles, labels=new_labels, title="", loc='upper left', frameon=False)

    # 5. 保存PDF
    output_path = os.path.join(output_dir, "normalized_rate_comparison.pdf")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 归一化速率对比柱状图（PDF）保存：{output_path}")
    plt.close(fig)

# ------------------------------
# 批量分析入口（保持不变）
# ------------------------------
def batch_analyze_rate_files(
    file_dir: str,
    file_list: List[str],
    output_dir: str = "rate_analysis_results",
    skip_initial_points: int = 2,
    start_time: float = 2.0,
    window_size: float = 0.02,
    baseline_filename: str = None
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录：{output_dir}")

    # 1. 批量处理所有文件
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for filename in file_list:
        file_path = os.path.join(file_dir, filename)
        result = process_single_rate_file(file_path, skip_initial_points)
        if result:
            file_results[filename] = result
            print(f"✅ 处理完成：{filename}")
        else:
            print(f"❌ 跳过文件：{filename}")

    if not file_results:
        print("❌ 无有效数据，程序退出")
        return

    # 2. 生成原有对比图
    full_title = 'Port Rate Comparison (TxRate & ECNRate)'
    full_output = os.path.join(output_dir, "full_time_rate_comparison.png")
    plot_rate_comparison(file_results, full_title, full_output)

    end_time = start_time + window_size
    window_title = f'Port Rate Comparison\n({window_size*1000:.0f}ms Window: {start_time:.3f}-{end_time:.3f}s)'
    window_output = os.path.join(
        output_dir,
        f"window_{start_time:.3f}_{end_time:.3f}_rate_comparison.png"
    )
    plot_rate_comparison(file_results, window_title, window_output, xlim=(start_time, end_time))

    # 3. 生成基准对比图和柱状图
    if baseline_filename and baseline_filename in file_results:
        baseline_full_title = f'Rate Comparison Against Baseline\n({os.path.splitext(baseline_filename)[0]})'
        baseline_full_output = os.path.join(output_dir, "baseline_comparison_full_time.png")
        plot_rate_vs_baseline(file_results, baseline_filename, baseline_full_title, baseline_full_output)

        baseline_window_title = f'Rate Comparison Against Baseline\n({os.path.splitext(baseline_filename)[0]} - {window_size*1000:.0f}ms Window)'
        baseline_window_output = os.path.join(
            output_dir,
            f"baseline_comparison_window_{start_time:.3f}_{end_time:.3f}.png"
        )
        plot_rate_vs_baseline(file_results, baseline_filename, baseline_window_title, baseline_window_output, xlim=(start_time, end_time))
        
        # 生成归一化对比柱状图（PDF）
        plot_normalized_rate_bar_chart(file_results, baseline_filename, output_dir)
    elif baseline_filename:
        print(f"⚠️  基准文件 {baseline_filename} 未在有效文件列表中，跳过基准对比图和柱状图")

# ------------------------------
# 主函数（保持不变）
# ------------------------------
if __name__ == "__main__":
    # 配置参数
    FILE_DIR = "/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_cachefollower_random"
    FILE_LIST = [
        "acc_thesis_mix_webserver_websearch_cachefollower_random.txrate",
        "copter_thesis_mix_webserver_websearch_cachefollower_random.txrate",
        "m3_thesis_mix_webserver_websearch_cachefollower_random.txrate",
        # "m4_thesis_mix_webserver_websearch_cachefollower_random.txrate"
        "dcqcn_thesis_mix_webserver_websearch_cachefollower_random.txrate",
        "hpcc_thesis_mix_webserver_websearch_cachefollower_random.txrate",
    ]
    OUTPUT_DIR = "rate_analysis_results/thesis_mix_webserver_websearch_cachefollower_random"
    SKIP_INITIAL_POINTS = 2
    START_TIME = 2.00
    WINDOW_SIZE = 0.01
    BASELINE_FILENAME = "copter_thesis_mix_webserver_websearch_cachefollower_random.txrate"

    # 执行批量分析
    batch_analyze_rate_files(
        file_dir=FILE_DIR,
        file_list=FILE_LIST,
        output_dir=OUTPUT_DIR,
        skip_initial_points=SKIP_INITIAL_POINTS,
        start_time=START_TIME,
        window_size=WINDOW_SIZE,
        baseline_filename=BASELINE_FILENAME
    )

    print("\n🎉 批量分析完成！结果已保存至：", OUTPUT_DIR)