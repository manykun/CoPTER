import numpy as np
import os
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict

@dataclass
class PortQueueData:
    switch_id: int          
    switch_buffer: int      
    port_id: int            # 端口标识：对应数据中的 connected_node_id
    queue_size: int         
    monitor_time_s: float   


def parse_queue_line(line: str, line_num: int) -> Union[PortQueueData, None]:
    """解析单条队列监控数据"""
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


def process_single_queue_file(file_path: str) -> Union[Tuple[Dict, Dict], None]:
    """处理单个队列文件，返回端口级和交换机级数据"""
    queue_records: List[PortQueueData] = []
    if not os.path.exists(file_path):
        print(f"错误：文件不存在 -> {file_path}")
        return None
    
    # 读取所有有效记录
    total_records = 0
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:  
        for line_num, line in enumerate(f, 1):  
            total_records += 1
            record = parse_queue_line(line, line_num)
            if record:
                queue_records.append(record)
    
    if len(queue_records) == 0:
        print(f"警告：文件无有效数据 -> {file_path}")
        return None

    # 构建端口级数据
    port_level_data: Dict[int, Dict[int, List[Tuple[float, int]]]] = {}
    for record in queue_records:
        if record.switch_id not in port_level_data:
            port_level_data[record.switch_id] = {}
        if record.port_id not in port_level_data[record.switch_id]:
            port_level_data[record.switch_id][record.port_id] = []
        port_level_data[record.switch_id][record.port_id].append(
            (record.monitor_time_s, record.queue_size)
        )
    
    # 对每个端口的时间序列按时间排序
    for switch_id in port_level_data:
        for port_id in port_level_data[switch_id]:
            port_level_data[switch_id][port_id].sort(key=lambda x: x[0])

    # 构建交换机级总数据
    time_switch_buckets = {}
    for record in queue_records:
        rounded_time = round(record.monitor_time_s, 6)
        key = (rounded_time, record.switch_id)
        if key not in time_switch_buckets:
            time_switch_buckets[key] = []
        time_switch_buckets[key].append(record)

    switch_total_queue = []
    switch_total_occupancy = []
    zero_queue_switch_count = 0

    for (time_key, switch_id), bucket_data in time_switch_buckets.items():
        all_queue_sizes = [item.queue_size for item in bucket_data]
        has_non_zero_queue = any(qs != 0 for qs in all_queue_sizes)
        if not has_non_zero_queue:
            zero_queue_switch_count += 1
            continue
        
        switch_buffer = bucket_data[0].switch_buffer
        if switch_buffer == 0:
            print(f"警告：时间{time_key:.3f}s 交换机{switch_id}的buffer为0，跳过计算")
            continue
        
        port_ids = [item.port_id for item in bucket_data]
        if len(port_ids) != len(set(port_ids)):
            print(f"警告：时间{time_key:.6f}s 交换机{switch_id}存在重复端口记录，端口列表：{port_ids}")
        
        total_queue_size = sum(all_queue_sizes)
        total_occupancy = (total_queue_size / switch_buffer) * 100

        if total_occupancy > 100:
            print(f"调试：时间{time_key:.6f}s 交换机{switch_id}占用率超100%！总队列：{total_queue_size}，Buffer：{switch_buffer}，占用率：{total_occupancy:.2f}%")
        elif total_occupancy < 0:
            print(f"调试：时间{time_key:.6f}s 交换机{switch_id}占用率为负！总队列：{total_queue_size}，Buffer：{switch_buffer}，占用率：{total_occupancy:.2f}%")

        switch_total_queue.append((time_key, switch_id, total_queue_size))
        switch_total_occupancy.append((time_key, switch_id, total_occupancy))

    if switch_total_queue and switch_total_occupancy:
        switch_total_queue.sort(key=lambda x: x[0])
        switch_total_occupancy.sort(key=lambda x: x[0])

    # 打印处理日志
    valid_switch_count = len(time_switch_buckets) - zero_queue_switch_count
    print(f"📊 文件{os.path.basename(file_path)}处理统计：")
    print(f"   - 总记录数：{total_records}")
    print(f"   - 有效记录数：{len(queue_records)}")
    print(f"   - 涉及交换机数：{len(port_level_data)}")
    print(f"   - 有非零队列的交换机组数：{valid_switch_count}")

    return (
        port_level_data,
        (np.array(switch_total_queue) if switch_total_queue else np.array([]), 
         np.array(switch_total_occupancy) if switch_total_occupancy else np.array([]))
    )


# -------------------------- 平滑处理核心函数 --------------------------
def smooth_series(data: np.ndarray, window: int, method: str) -> np.ndarray:
    """
    对时序数据进行平滑处理
    :param data: 原始数据数组
    :param window: 平滑窗口大小（建议3-10）
    :param method: 平滑方法："moving_average"（移动平均）或"ewma"（指数加权平滑）
    :return: 平滑后的数据数组
    """
    if len(data) < window or window < 2:
        return data  # 数据量不足或窗口过小，返回原始数据
    
    data = np.asarray(data, dtype=np.float64)  # 确保浮点型
    
    if method == "moving_average":
        # 移动平均：窗口内等权平均（边缘用镜像填充避免截断）
        kernel = np.ones(window) / window
        # 边缘处理：前后镜像填充，避免平滑后数据长度缩短
        padded = np.pad(data, pad_width=window//2, mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        # 确保输出长度与输入一致
        return smoothed[:len(data)] if len(smoothed) > len(data) else smoothed
    
    elif method == "ewma":
        # 指数加权平滑：近期数据权重更高
        alpha = 2 / (window + 1)  # 平滑系数，窗口越大alpha越小
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]  # 初始值
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    else:
        return data  # 无效方法返回原始数据


# -------------------------- 端口级绘图函数（带平滑） --------------------------
def plot_switch_single_port_queues(
    file_port_data: Dict[int, Dict[int, List[Tuple[float, int]]]],
    target_switch_id: int,
    filename: str,
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None,
    smooth_window: int = 5,
    smooth_method: str = "moving_average",
    show_raw: bool = False  # 是否显示原始数据（用于对比）
):
    """绘制指定交换机所有端口的队列曲线（支持平滑）"""
    if target_switch_id not in file_port_data:
        print(f"警告：文件{filename}中无交换机{target_switch_id}的数据，跳过")
        return
    
    target_port_data = file_port_data[target_switch_id]
    if not target_port_data:
        print(f"警告：交换机{target_switch_id}无端口数据，跳过")
        return
    
    plt.figure(figsize=(14, 8))
    color_list = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
        '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5'
    ]
    line_style = '-'
    port_count = len(target_port_data)
    print(f"   交换机{target_switch_id}共{port_count}个端口，绘制平滑曲线...")
    
    for port_idx, (port_id, time_qlen_list) in enumerate(sorted(target_port_data.items())):
        time_data = np.array([t for t, q in time_qlen_list])
        queue_data = np.array([q for t, q in time_qlen_list])
        
        # 应用平滑
        smoothed_queue = smooth_series(queue_data, smooth_window, smooth_method)
        
        color = color_list[port_idx % len(color_list)]
        file_label = os.path.splitext(filename)[0]
        plot_label = f"{file_label} - Port {port_id}"
        
        # 可选：显示原始数据（灰色透明）
        if show_raw:
            plt.plot(
                time_data, queue_data,
                color='gray',
                linestyle='--',
                linewidth=1.0,
                alpha=0.5,
                label=f"Raw - Port {port_id}" if port_idx == 0 else ""  # 仅第一个端口显示原始数据标签
            )
        
        # 绘制平滑后的数据
        plt.plot(
            time_data, smoothed_queue,
            color=color,
            linestyle=line_style,
            linewidth=2.0,
            alpha=0.8,
            label=plot_label
        )

    if xlim:
        plt.xlim(*xlim)
    plt.title(f"Switch {target_switch_id}: {title}\n(Smooth: {smooth_method}, Window: {smooth_window})", fontsize=14, pad=20)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Port Queue Size (Bytes)', fontsize=12)
    plt.legend(
        fontsize=9, 
        loc='upper left', 
        bbox_to_anchor=(1, 1),
        ncol=2 if port_count > 10 else 1
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 端口队列平滑图保存：{output_path}")


# -------------------------- 交换机级总占用率绘图函数（带平滑） --------------------------
def plot_switch_total_occupancy(
    file_results: dict,
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None,
    target_switch_id: Union[int, None] = None,
    smooth_window: int = 5,
    smooth_method: str = "moving_average",
    show_raw: bool = False  # 是否显示原始数据
):
    plt.figure(figsize=(14, 8))
    color_list = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    line_style = '-'

    for file_idx, (filename, (_, (_, total_occ_array))) in enumerate(file_results.items()):
        if len(total_occ_array) == 0:
            print(f"警告：文件{filename}无交换机占用率数据，跳过")
            continue
        
        all_switch_ids = np.unique(total_occ_array[:, 1])
        if target_switch_id is not None:
            if target_switch_id not in all_switch_ids:
                print(f"警告：文件{filename}中无交换机{target_switch_id}的数据，跳过")
                continue
            plot_switch_ids = [target_switch_id]
        else:
            plot_switch_ids = all_switch_ids
        
        for switch_idx, switch_id in enumerate(plot_switch_ids):
            switch_data = total_occ_array[total_occ_array[:, 1] == switch_id]
            time_data = switch_data[:, 0]
            occ_data = switch_data[:, 2].astype(float)
            
            # 应用平滑
            smoothed_occ = smooth_series(occ_data, smooth_window, smooth_method)
            
            color = color_list[(file_idx * len(plot_switch_ids) + switch_idx) % len(color_list)]
            file_label = os.path.splitext(filename)[0]
            plot_label = f"{file_label} - Switch {int(switch_id)}"
            
            # 显示原始数据（可选）
            if show_raw:
                plt.plot(
                    time_data, occ_data,
                    color='gray',
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.5,
                    label=f"Raw - {file_label}" if (file_idx == 0 and switch_idx == 0) else ""
                )
            
            # 绘制平滑后的数据
            plt.plot(
                time_data, smoothed_occ,
                color=color,
                linestyle=line_style,
                linewidth=2.5,
                alpha=1.0,
                label=plot_label
            )

    if xlim:
        plt.xlim(*xlim)
    plt.title(f"{title}\n(Smooth: {smooth_method}, Window: {smooth_window})", fontsize=14, pad=20)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Switch Queue Occupancy (%)', fontsize=12)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 交换机占用率平滑图保存：{output_path}")


# -------------------------- 其他保留函数（交换机级总队列/基准对比） --------------------------
def plot_switch_total_queue(
    file_results: dict,
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None,
    target_switch_id: Union[int, None] = None,
    smooth_window: int = 5,
    smooth_method: str = "moving_average",
    show_raw: bool = False
):
    plt.figure(figsize=(14, 8))
    color_list = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    line_style = '-'

    for file_idx, (filename, (_, (total_queue_array, _))) in enumerate(file_results.items()):
        if len(total_queue_array) == 0:
            print(f"警告：文件{filename}无交换机总队列数据，跳过")
            continue
        
        all_switch_ids = np.unique(total_queue_array[:, 1])
        if target_switch_id is not None:
            if target_switch_id not in all_switch_ids:
                print(f"警告：文件{filename}中无交换机{target_switch_id}的数据，跳过")
                continue
            plot_switch_ids = [target_switch_id]
        else:
            plot_switch_ids = all_switch_ids
        
        for switch_idx, switch_id in enumerate(plot_switch_ids):
            switch_data = total_queue_array[total_queue_array[:, 1] == switch_id]
            time_data = switch_data[:, 0]
            queue_data = switch_data[:, 2].astype(float)
            
            # 应用平滑
            smoothed_queue = smooth_series(queue_data, smooth_window, smooth_method)
            
            color = color_list[(file_idx * len(plot_switch_ids) + switch_idx) % len(color_list)]
            file_label = os.path.splitext(filename)[0]
            plot_label = f"{file_label} - Switch {int(switch_id)}"
            
            # 显示原始数据（可选）
            if show_raw:
                plt.plot(
                    time_data, queue_data,
                    color='gray',
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.5,
                    label=f"Raw - {file_label}" if (file_idx == 0 and switch_idx == 0) else ""
                )
            
            # 绘制平滑后的数据
            plt.plot(
                time_data, smoothed_queue,
                color=color,
                linestyle=line_style,
                linewidth=2.5,
                alpha=1.0,
                label=plot_label
            )

    if xlim:
        plt.xlim(*xlim)
    plt.title(f"{title}\n(Smooth: {smooth_method}, Window: {smooth_window})", fontsize=14, pad=20)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Switch Total Queue Size (Bytes)', fontsize=12)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 交换机总队列平滑图保存：{output_path}")


def plot_comparison_against_baseline(
    file_results: Dict[str, Tuple[Dict, Tuple[np.ndarray, np.ndarray]]],
    baseline_filename: str,
    title: str,
    output_path: str,
    xlim: Tuple[float, float] = None,
    target_switch_id: Union[int, None] = None,
    smooth_window: int = 5,
    smooth_method: str = "moving_average"
):
    if baseline_filename not in file_results:
        print(f"警告：基准文件 {baseline_filename} 不在分析结果中，无法生成对比图")
        return
    
    _, (baseline_total_queue, _) = file_results[baseline_filename]
    if len(baseline_total_queue) == 0:
        print(f"警告：基准文件 {baseline_filename} 无有效数据，无法生成对比图")
        return
    
    baseline_all_switch_ids = np.unique(baseline_total_queue[:, 1])
    if target_switch_id is not None:
        if target_switch_id not in baseline_all_switch_ids:
            print(f"警告：基准文件{baseline_filename}中无交换机{target_switch_id}的数据，无法对比")
            return
        baseline_switch_ids = [target_switch_id]
    else:
        baseline_switch_ids = baseline_all_switch_ids

    plt.figure(figsize=(14, 8))
    color_list = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    for switch_id in baseline_switch_ids:
        baseline_switch_data = baseline_total_queue[baseline_total_queue[:, 1] == switch_id]
        baseline_times = baseline_switch_data[:, 0]
        baseline_values = baseline_switch_data[:, 2].astype(float)
        # 基准数据也平滑
        smoothed_baseline = smooth_series(baseline_values, smooth_window, smooth_method)
        
        for file_idx, (filename, (_, (total_queue_array, _))) in enumerate(file_results.items()):
            if filename == baseline_filename:
                continue
            if len(total_queue_array) == 0:
                print(f"警告：文件 {filename} 无有效数据，跳过基准对比")
                continue
            if switch_id not in total_queue_array[:, 1]:
                print(f"警告：文件 {filename} 无交换机{int(switch_id)}的数据，跳过")
                continue
            
            file_switch_data = total_queue_array[total_queue_array[:, 1] == switch_id]
            file_times = file_switch_data[:, 0]
            file_values = file_switch_data[:, 2].astype(float)
            # 对比数据平滑
            smoothed_file = smooth_series(file_values, smooth_window, smooth_method)
            
            # 插值使时间对齐
            interp_smoothed = np.interp(baseline_times, file_times, smoothed_file)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                diff_percent = (interp_smoothed - smoothed_baseline) / smoothed_baseline * 100
                diff_percent[smoothed_baseline == 0] = 0
            
            color = color_list[file_idx % len(color_list)]
            file_label = os.path.splitext(filename)[0]
            plot_label = f"{file_label} - Switch {int(switch_id)} vs Baseline (%)"
            
            plt.plot(
                baseline_times, diff_percent,
                color=color,
                linewidth=2,
                alpha=1.0,
                label=plot_label
            )
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='Zero Difference')
    if xlim:
        plt.xlim(*xlim)
    plt.title(f"{title}\n(Smooth: {smooth_method}, Window: {smooth_window})", fontsize=14, pad=20)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Difference from Baseline (%)', fontsize=12)
    plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 基准对比平滑图保存：{output_path}")


# -------------------------- 批量分析入口函数 --------------------------
def batch_analyze_queue_files(
    file_dir: str, 
    file_list: List[str], 
    main_output_dir: str = "queue_analysis_results", 
    custom_subfolder: str = "webserver_load0.7_202405", 
    start_time: float = 2.00, 
    window_size: float = 0.02, 
    baseline_filename: str = "copter_webserver_t0.05_l0.7_co.queue",
    target_switch_id: Union[int, None] = None,
    plot_port_level: bool = True,
    # 平滑参数（核心新增）
    smooth_window: int = 5,
    smooth_method: str = "moving_average",
    show_raw: bool = False  # 是否在图中显示原始数据（用于对比平滑效果）
):
    final_output_dir = os.path.join(main_output_dir, custom_subfolder)
    os.makedirs(final_output_dir, exist_ok=True)
    
    if target_switch_id is not None:
        print(f"\n📁 交换机{target_switch_id}的分析结果将保存到：{final_output_dir}\n")
    else:
        print(f"\n📁 所有交换机的分析结果将保存到：{final_output_dir}\n")

    file_results: Dict[str, Tuple[Dict, Tuple[np.ndarray, np.ndarray]]] = {}
    for filename in file_list:
        print(f"🔄 正在处理文件：{filename}")
        file_path = os.path.join(file_dir, filename)
        result = process_single_queue_file(file_path)
        if result:
            file_results[filename] = result
            print(f"✅ {filename} 处理完成\n")
        else:
            print(f"❌ {filename} 无有效数据，跳过\n")

    if not file_results:
        print("❌ 无任何有效数据，程序退出")
        return

    end_time = start_time + window_size

    # 绘制端口级平滑曲线
    if plot_port_level and target_switch_id is not None:
        print(f"\n🎯 开始生成交换机{target_switch_id}的端口队列平滑曲线...")
        # 全时间范围
        port_full_title = 'All Ports Queue Size vs Time (Smoothed)\n(Incast Scenario)'
        port_full_output = os.path.join(
            final_output_dir, 
            f"switch_{target_switch_id}_all_ports_full_time_smoothed.png"
        )
        for filename, (port_data, _) in file_results.items():
            plot_switch_single_port_queues(
                file_port_data=port_data,
                target_switch_id=target_switch_id,
                filename=filename,
                title=port_full_title,
                output_path=port_full_output,
                smooth_window=smooth_window,
                smooth_method=smooth_method,
                show_raw=show_raw
            )
        
        # 时间窗口内
        port_window_title = f'All Ports Queue Size vs Time (Smoothed)\n(Incast Scenario - Window: {start_time:.3f}-{end_time:.3f}s)'
        port_window_output = os.path.join(
            final_output_dir, 
            f"switch_{target_switch_id}_all_ports_window_{start_time:.3f}_{end_time:.3f}_smoothed.png"
        )
        for filename, (port_data, _) in file_results.items():
            plot_switch_single_port_queues(
                file_port_data=port_data,
                target_switch_id=target_switch_id,
                filename=filename,
                title=port_window_title,
                output_path=port_window_output,
                xlim=(start_time, end_time),
                smooth_window=smooth_window,
                smooth_method=smooth_method,
                show_raw=show_raw
            )

    # 绘制交换机级平滑曲线
    print(f"\n📊 开始生成交换机级平滑曲线...")
    # 总队列
    full_queue_title = 'Switch-Level Total Queue Size (Smoothed)\n(Incast Scenario)'
    full_queue_output = os.path.join(
        final_output_dir, 
        f"switch_{target_switch_id}_total_queue_full_time_smoothed.png" if target_switch_id else "switch_total_queue_full_time_smoothed.png"
    )
    plot_switch_total_queue(
        file_results=file_results,
        title=full_queue_title,
        output_path=full_queue_output,
        target_switch_id=target_switch_id,
        smooth_window=smooth_window,
        smooth_method=smooth_method,
        show_raw=show_raw
    )

    window_queue_title = f'Switch-Level Total Queue Size (Smoothed)\n(Incast Scenario - Window: {start_time:.3f}-{end_time:.3f}s)'
    window_queue_output = os.path.join(
        final_output_dir, 
        f"switch_{target_switch_id}_total_queue_window_{start_time:.3f}_{end_time:.3f}_smoothed.png" if target_switch_id else f"switch_total_queue_window_{start_time:.3f}_{end_time:.3f}_smoothed.png"
    )
    plot_switch_total_queue(
        file_results=file_results,
        title=window_queue_title,
        output_path=window_queue_output,
        xlim=(start_time, end_time),
        target_switch_id=target_switch_id,
        smooth_window=smooth_window,
        smooth_method=smooth_method,
        show_raw=show_raw
    )

    # 总占用率
    full_occ_title = 'Switch-Level Queue Occupancy (Smoothed)\n(Incast Scenario)'
    full_occ_output = os.path.join(
        final_output_dir, 
        f"switch_{target_switch_id}_occupancy_full_time_smoothed.png" if target_switch_id else "switch_occupancy_full_time_smoothed.png"
    )
    plot_switch_total_occupancy(
        file_results=file_results,
        title=full_occ_title,
        output_path=full_occ_output,
        target_switch_id=target_switch_id,
        smooth_window=smooth_window,
        smooth_method=smooth_method,
        show_raw=show_raw
    )

    window_occ_title = f'Switch-Level Queue Occupancy (Smoothed)\n(Incast Scenario - Window: {start_time:.3f}-{end_time:.3f}s)'
    window_occ_output = os.path.join(
        final_output_dir, 
        f"switch_{target_switch_id}_occupancy_window_{start_time:.3f}_{end_time:.3f}_smoothed.png" if target_switch_id else f"switch_occupancy_window_{start_time:.3f}_{end_time:.3f}_smoothed.png"
    )
    plot_switch_total_occupancy(
        file_results=file_results,
        title=window_occ_title,
        output_path=window_occ_output,
        xlim=(start_time, end_time),
        target_switch_id=target_switch_id,
        smooth_window=smooth_window,
        smooth_method=smooth_method,
        show_raw=show_raw
    )

    # 基准对比
    if baseline_filename in file_results:
        print(f"\n📊 开始生成基准对比平滑曲线...")
        baseline_full_title = f'Switch-Level Queue Size vs Baseline (Smoothed)\n(Baseline: {os.path.splitext(baseline_filename)[0]})'
        baseline_full_output = os.path.join(
            final_output_dir, 
            f"switch_{target_switch_id}_baseline_comparison_full_time_smoothed.png" if target_switch_id else "switch_baseline_comparison_full_time_smoothed.png"
        )
        plot_comparison_against_baseline(
            file_results=file_results,
            baseline_filename=baseline_filename,
            title=baseline_full_title,
            output_path=baseline_full_output,
            target_switch_id=target_switch_id,
            smooth_window=smooth_window,
            smooth_method=smooth_method
        )

        baseline_window_title = f'Switch-Level Queue Size vs Baseline (Smoothed)\n(Window: {start_time:.3f}-{end_time:.3f}s, Baseline: {os.path.splitext(baseline_filename)[0]})'
        baseline_window_output = os.path.join(
            final_output_dir, 
            f"switch_{target_switch_id}_baseline_comparison_window_{start_time:.3f}_{end_time:.3f}_smoothed.png" if target_switch_id else f"switch_baseline_comparison_window_{start_time:.3f}_{end_time:.3f}_smoothed.png"
        )
        plot_comparison_against_baseline(
            file_results=file_results,
            baseline_filename=baseline_filename,
            title=baseline_window_title,
            output_path=baseline_window_output,
            xlim=(start_time, end_time),
            target_switch_id=target_switch_id,
            smooth_window=smooth_window,
            smooth_method=smooth_method
        )

    print(f"\n🎉 所有分析完成！结果已保存至：{final_output_dir}")


if __name__ == "__main__":
    # -------------------------- 配置参数（根据需求修改） --------------------------
    QUEUE_FILE_DIR = "/home/ame/copter/simulation/output/mix_webserver_websearch_hadoop_clusters"  # 队列监控文件目录
    QUEUE_FILE_LIST = [
        # "acc_webserver_incast.queue",
        # "copter_webserver_incast.queue",
        # "copter_webserver_incast_like_acc.queue",
        # "copter_webserver_incast_m3.queue"
        "acc_mix_webserver_websearch_hadoop_clusters.queue",
        "copter_mix_webserver_websearch_hadoop_clusters.queue",
        "m3_mix_webserver_websearch_hadoop_clusters.queue"
    ]
    MAIN_OUTPUT_DIR = "queue_analysis_results"  # 结果总目录
    CUSTOM_SUBFOLDER = "mix_webserver_websearch_hadoop_clusters"  # 自定义子目录
    START_TIME = 2.00  # 聚焦起始时间（如Incast发生时间）
    WINDOW_SIZE = 0.05  # 聚焦时间窗口（如0.05s=50ms）
    BASELINE_FILENAME = "copter_mix_webserver_websearch_hadoop_clusters.queue"  # 基准文件名（可选）
    TARGET_SWITCH_ID = 264  # 目标交换机ID（如Incast场景的264）
    PLOT_PORT_LEVEL = True  # 是否绘制端口级曲线

    # -------------------------- 平滑参数（核心调优点） --------------------------
    SMOOTH_WINDOW = 10  # 平滑窗口大小（3-10，值越大越平滑）
    SMOOTH_METHOD = "moving_average"  # 平滑方法："moving_average"或"ewma"
    SHOW_RAW = True  # 是否显示原始数据（用于对比平滑效果）

    # -------------------------- 执行分析 --------------------------
    batch_analyze_queue_files(
        file_dir=QUEUE_FILE_DIR,
        file_list=QUEUE_FILE_LIST,
        main_output_dir=MAIN_OUTPUT_DIR,
        custom_subfolder=CUSTOM_SUBFOLDER,
        start_time=START_TIME,
        window_size=WINDOW_SIZE,
        baseline_filename=BASELINE_FILENAME,
        target_switch_id=TARGET_SWITCH_ID,
        plot_port_level=PLOT_PORT_LEVEL,
        smooth_window=SMOOTH_WINDOW,
        smooth_method=SMOOTH_METHOD,
        show_raw=SHOW_RAW
    )
