import numpy as np
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict
import time

# -------------------------- 全局绘图样式配置（沿用论文风格） --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 60,
    'axes.labelsize': 60,
    'axes.titlesize': 60,
    'legend.fontsize': 50,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.0,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    'text.usetex': False,
})

# -------------------------- 样式映射（保持与FCT脚本一致） --------------------------
name_mapping = {
    "copter": "CoPT",
    "m3": "SCoPE",
    "m4": "m4",
    "acc": "ACC",
    "dcqcn": r"$SECN_1$",  # 内置解析器支持$包裹+_表示下标
    "hpcc": r"$SECN_2$"
}
color_map = {
    "copter": "#FF6B00",
    "acc": "#00CC66",
    "m4": "#0066FF",
    "dcqcn": "#9933FF",
    "hpcc": "#FF3333"
}
markers = {
    "copter": '',
    "acc": '',
    "m4": '',
    "dcqcn": 'D',
    "hpcc": 'v'
}
line_styles = {
    "copter": '-',
    "acc": '-',
    "m4": '-',
    "dcqcn": '-',
    "hpcc": '-'
}

# 标记点间隔（解决密集问题：每N个点显示一个标记）
marker_interval = 50

@dataclass
class PortQueueData:
    switch_id: int          
    switch_buffer: int      
    port_id: int            
    queue_size: int         
    monitor_time_s: float   

def parse_queue_line(line: str, line_num: int) -> Union[PortQueueData, None]:
    stripped_line = line.strip()
    if not stripped_line:
        print(f"警告：第{line_num}行是空行，已跳过")
        return None
        
    parts = stripped_line.split()
    if len(parts) < 5:
        print(f"警告：第{line_num}行字段不足5个（实际{len(parts)}个），内容：{stripped_line}")
        return None
    if len(parts) > 5:
        print(f"警告：第{line_num}行字段超过5个（实际{len(parts)}个），将使用前5个字段，内容：{stripped_line}")
    
    try:
        return PortQueueData(
            switch_id=int(parts[0]),
            switch_buffer=int(parts[1]),
            port_id=int(parts[2]),
            queue_size=int(parts[3]),
            monitor_time_s=float(parts[4])
        )
    except ValueError as e:
        print(f"警告：第{line_num}行数值转换失败 - {str(e)}，内容：{stripped_line}")
        return None

def process_single_queue_file(file_path: str) -> Union[Tuple[np.ndarray, np.ndarray, float, float], None]:
    """
    处理单个队列文件，返回平均队列数组、P99队列数组、整体平均队列长度和标准差
    """
    queue_records: List[PortQueueData] = []
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return None
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  
        for line_num, line in enumerate(f, 1):  
            record = parse_queue_line(line, line_num)
            if record:
                queue_records.append(record)
    
    if len(queue_records) == 0:
        print(f"警告：文件无有效数据 -> {file_path}")
        return None

    # 计算整体平均队列长度和标准差（替换方差为标准差）
    all_queue_sizes = [record.queue_size for record in queue_records]
    overall_avg_queue = np.mean(all_queue_sizes)
    overall_queue_std = np.std(all_queue_sizes)  # 关键修改：用np.std计算标准差

    time_buckets = {}
    for record in queue_records:
        time_key = record.monitor_time_s
        if time_key not in time_buckets:
            time_buckets[time_key] = []
        time_buckets[time_key].append(record)

    avg_queue = []
    p99_queue = []
    for time_key, bucket_data in time_buckets.items():
        queue_sizes = [item.queue_size for item in bucket_data]
        avg_size = np.mean(queue_sizes)
        p99_size = np.percentile(queue_sizes, 99)
        avg_queue.append((time_key, avg_size))
        p99_queue.append((time_key, p99_size))

    avg_queue.sort(key=lambda x: x[0])
    p99_queue.sort(key=lambda x: x[0])

    # 返回值中替换方差为标准差
    return np.array(avg_queue), np.array(p99_queue), overall_avg_queue, overall_queue_std

def plot_queue(
    file_results: dict,  # {filename: (avg_array, p99_array, overall_avg, overall_std)}
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None
):
    plt.figure(figsize=(18,12))  # 调整为FCT图表一致的尺寸

    for idx, (filename, (avg_array, p99_array, _, _)) in enumerate(file_results.items()):
        # 提取方法名称（从文件名前缀获取）
        method_name = os.path.splitext(filename)[0].split('_')[0]
        # 映射显示名称、颜色、标记和线型
        display_name = name_mapping.get(method_name, method_name)
        color = color_map.get(method_name, color_map["copter"])
        marker = markers.get(method_name, 'o')
        linestyle = line_styles.get(method_name, '-')

        # 采样标记点（解决密集问题）
        x_data = avg_array[:, 0]
        y_data = avg_array[:, 1]
        # 每marker_interval个点取一个标记
        sample_indices = list(range(0, len(x_data), marker_interval))
        
        # 绘制曲线（所有曲线宽度一致）
        alpha = 0.8 if method_name != "copter" else 1.0  # 基准曲线不透明
        linewidth = 1.5  # 统一线宽
        
        plt.plot(
            x_data, y_data,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=f"{display_name}"
        )
        
        # 绘制标记点（只在采样位置显示）
        plt.plot(
            x_data[sample_indices], y_data[sample_indices],
            color=color,
            marker=marker,
            markersize=5 if method_name == "copter" else 4,  # 基准标记更大
            linestyle='',  # 只显示标记，不显示额外线条
            alpha=alpha
        )
        
        # 绘制99th Percentile队列（如需启用，取消注释）
        # plt.plot(
        #     p99_array[:, 0], p99_array[:, 1],
        #     color=color,
        #     linestyle='--',
        #     linewidth=1.5,
        #     marker=marker,
        #     markersize=4,
        #     label=f"{display_name} (99th)"
        # )

    if xlim:
        plt.xlim(*xlim)
    plt.xlabel('Time (s)', fontsize=60)
    plt.ylabel('Queue Size (Bytes)', fontsize=60)
    plt.title(title, fontsize=60, pad=15)
    plt.legend(
        frameon=False,
        framealpha=0.9,
        shadow=False,
        edgecolor='black',
        facecolor='white',
        labelspacing=0.8,
        handlelength=2.0,
        handletextpad=0.8,
        fontsize=60,
        loc='upper left'
    )
    plt.grid(True, axis='y', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')  # 保存为PDF格式
    print(f"✅ 图表保存：{output_path}")

def plot_comparison_against_baseline(
    file_results: Dict[str, Tuple[np.ndarray, np.ndarray, float, float]],
    baseline_filename: str,
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None
):
    if baseline_filename not in file_results:
        print(f"警告：基准文件 {baseline_filename} 不在分析结果中，无法生成对比图")
        return
    
    # 获取基准数据
    baseline_avg, _, _, _ = file_results[baseline_filename]
    baseline_times = baseline_avg[:, 0]
    baseline_values = baseline_avg[:, 1]
    
    plt.figure(figsize=(18,12))  # 调整为FCT图表一致的尺寸
    
    # 先处理基准文件（CoPTER），确保它在图例中显示
    baseline_method = os.path.splitext(baseline_filename)[0].split('_')[0]
    baseline_display = name_mapping.get(baseline_method, baseline_method)
    baseline_color = color_map.get(baseline_method, color_map["copter"])
    baseline_marker = markers.get(baseline_method, 'o')
    baseline_linestyle = line_styles.get(baseline_method, '-')
    
    # 基准的归一化值始终为1.0
    baseline_normalized = np.ones_like(baseline_times)
    
    # 采样标记点
    sample_indices = list(range(0, len(baseline_times), marker_interval))
    
    # 绘制基准曲线（线宽与其他曲线一致）
    plt.plot(
        baseline_times, baseline_normalized,
        color=baseline_color,
        linestyle=baseline_linestyle,
        linewidth=6,  # 统一线宽
        alpha=1.0,  # 完全不透明（保持基准辨识度）
        label=f"{baseline_display}"
    )
    
    # 绘制基准标记点
    plt.plot(
        baseline_times[sample_indices], baseline_normalized[sample_indices],
        color=baseline_color,
        marker=baseline_marker,
        markersize=5,  # 基准标记稍大（保持辨识度）
        linestyle='',
        alpha=1.0
    )
    
    # 处理其他方法
    for idx, (filename, (avg_array, _, _, _)) in enumerate(file_results.items()):
        if filename == baseline_filename:
            continue
            
        # 提取方法名称
        method_name = os.path.splitext(filename)[0].split('_')[0]
        display_name = name_mapping.get(method_name, method_name)
        color = color_map.get(method_name, color_map["copter"])
        marker = markers.get(method_name, 'o')
        linestyle = line_styles.get(method_name, '-')
        
        # 插值匹配基准时间点
        interp_values = np.interp(baseline_times, avg_array[:, 0], avg_array[:, 1])
        
        # 计算归一化值（当前值 / 基准值）
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_values = interp_values / baseline_values
            # 处理基准值为0的特殊情况（避免无穷大）
            normalized_values[baseline_values == 0] = 0 if np.all(interp_values[baseline_values == 0] == 0) else 1

        # 采样标记点
        sample_indices = list(range(0, len(baseline_times), marker_interval))
        
        # 样式设置（与基准保持一致的线宽）
        alpha = 0.8
        linewidth = 6
        markersize = 4
        
        # 绘制归一化曲线
        plt.plot(
            baseline_times, normalized_values,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=f"{display_name}"
        )
        
        # 绘制标记点
        plt.plot(
            baseline_times[sample_indices], normalized_values[sample_indices],
            color=color,
            marker=marker,
            markersize=markersize,
            linestyle='',
            alpha=alpha
        )
    
    # 添加归一化基准线（y=1，作为辅助线）
    # plt.axhline(y=1, color='gray', linestyle='--', alpha=0.4, linewidth=1.0)
    
    if xlim:
        plt.xlim(*xlim)
    
    plt.xlabel('Time (s)', fontsize=60)
    plt.ylabel('Normalized Queue length', fontsize=60)
    # plt.title(title, fontsize=14, pad=15)
    plt.legend(
        frameon=False,
        framealpha=0.9,
        shadow=False,
        edgecolor='black',
        facecolor='white',
        labelspacing=0.4,
        handlelength=2.0,
        handletextpad=0.8,
        fontsize=50,
        loc='upper right'
    )
    # plt.grid(False, axis='y', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ 基准归一化对比图表保存：{output_path}")

def batch_analyze_queue_files(
    file_dir: str, 
    file_list: List[str], 
    main_output_dir: str = "queue_analysis_results",
    custom_subfolder: str = "webserver_load0.7_202405",
    start_time: float = 2.00,
    window_size: float = 0.02,
    baseline_filename: str = "copter_webserver_t0.05_l0.7_co.queue"
):
    # 创建输出目录
    final_output_dir = os.path.join(main_output_dir, custom_subfolder)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📁 所有图表将保存到：{final_output_dir}")

    # 1. 处理所有文件，收集数据
    file_results = {}
    overall_avg_stats = []  # 存储整体平均队列长度和标准差统计
    for filename in file_list:
        file_path = os.path.join(file_dir, filename)
        result = process_single_queue_file(file_path)
        if result:
            avg_array, p99_array, overall_avg, overall_std = result  # 接收标准差
            file_results[filename] = (avg_array, p99_array, overall_avg, overall_std)
            # 记录统计信息（替换方差为标准差）
            method_name = os.path.splitext(filename)[0].split('_')[0]
            display_name = name_mapping.get(method_name, method_name)
            overall_avg_stats.append({
                "Method": display_name,
                "Filename": filename,
                "Overall_Average_Queue_Size(Bytes)": overall_avg,
                "Queue_Size_Std(Bytes)": overall_std  # 关键修改：存储标准差
            })
            print(f"✅ 处理完成：{filename}")
        else:
            print(f"❌ 跳过文件：{filename}")

    if not file_results:
        print("❌ 无有效数据，程序退出")
        return

    # 2. 输出整体平均队列长度和标准差统计（更新打印格式）
    print("\n" + "="*120)
    print("📊 整体平均队列长度与标准差统计")
    print("="*120)
    # 按平均队列长度排序
    for stats in sorted(overall_avg_stats, key=lambda x: x["Overall_Average_Queue_Size(Bytes)"]):
        print(
            f"{stats['Method']:<10} | {stats['Filename']:<40} | "
            f"平均队列长度: {stats['Overall_Average_Queue_Size(Bytes)']:.2f} Bytes | "
            f"标准差: {stats['Queue_Size_Std(Bytes)']:.2f} Bytes"  # 标注单位Bytes
        )
    print()

    # 3. 生成完整时间跨度图
    full_title = 'Port Queue Size Comparison'
    full_output = os.path.join(final_output_dir, "full_time_queue_comparison.pdf")
    plot_queue(file_results, full_title, full_output)

    # 4. 生成指定时间窗口图
    end_time = start_time + window_size
    window_title = f'Port Queue Size Comparison ({window_size*1000:.0f}ms Window: {start_time:.3f}-{end_time:.3f}s)'
    window_output = os.path.join(
        final_output_dir, 
        f"window_{start_time:.3f}_{end_time:.3f}_queue_comparison.pdf"
    )
    plot_queue(file_results, window_title, window_output, xlim=(start_time, end_time))
    
    # 5. 生成基准归一化对比图（完整时间）
    baseline_method = os.path.splitext(baseline_filename)[0].split('_')[0]
    baseline_display = name_mapping.get(baseline_method, baseline_method)
    baseline_full_title = f'Normalized Queue Size Comparison vs {baseline_display}'
    baseline_full_output = os.path.join(final_output_dir, "baseline_comparison_full_time.pdf")
    plot_comparison_against_baseline(file_results, baseline_filename, baseline_full_title, baseline_full_output)
    
    # 6. 生成基准归一化对比图（指定时间窗口）
    baseline_window_title = f'Normalized Queue Size Comparison vs {baseline_display} ({window_size*1000:.0f}ms Window)'
    baseline_window_output = os.path.join(
        final_output_dir, 
        f"baseline_comparison_window_{start_time:.3f}_{end_time:.3f}.pdf"
    )
    plot_comparison_against_baseline(file_results, baseline_filename, baseline_window_title, baseline_window_output, xlim=(start_time, end_time))

if __name__ == "__main__":
    # -------------------------- 配置参数（用户可根据需求修改） --------------------------
    QUEUE_FILE_DIR = "/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_cachefollower_random"
    QUEUE_FILE_LIST = [
        "acc_thesis_mix_webserver_websearch_cachefollower_random.queue",
        "copter_thesis_mix_webserver_websearch_cachefollower_random.queue",
        "m3_thesis_mix_webserver_websearch_cachefollower_random.queue",
        # "m4_thesis_mix_webserver_websearch_cachefollower_random.queue",
        "dcqcn_thesis_mix_webserver_websearch_cachefollower_random.queue",
        "hpcc_thesis_mix_webserver_websearch_cachefollower_random.queue"
    ]
    MAIN_OUTPUT_DIR = "queue_analysis_results"
    CUSTOM_SUBFOLDER = "thesis_mix_webserver_websearch_cachefollower_random"
    START_TIME = 2.03
    WINDOW_SIZE = 0.02
    BASELINE_FILENAME = "copter_thesis_mix_webserver_websearch_cachefollower_random.queue"
    # --------------------------------------------------------------------------------

    batch_analyze_queue_files(
        file_dir=QUEUE_FILE_DIR,
        file_list=QUEUE_FILE_LIST,
        main_output_dir=MAIN_OUTPUT_DIR,
        custom_subfolder=CUSTOM_SUBFOLDER,
        start_time=START_TIME,
        window_size=WINDOW_SIZE,
        baseline_filename=BASELINE_FILENAME
    )