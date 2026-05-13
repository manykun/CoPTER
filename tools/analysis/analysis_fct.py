import argparse
import numpy as np
import os


def get_pctl(a, p):
    """计算指定百分位值"""
    i = int(len(a) * p)
    return a[i] if i < len(a) else a[-1]  # 避免索引越界


def process_file(file_path):
    """处理文件并返回按流量大小排序的 [fct, flow_size] 列表（替换原slowdown为fct）"""
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 8:  # 确保行有足够的字段
                continue
            flow_size = int(fields[4])        # 第5个字段：流量大小
            fct = int(fields[6])              # 第7个字段：实际FCT值（直接保存该值）
            result.append([fct, flow_size])   # 存储[实际FCT, 流量大小]
    result.sort(key=lambda x: x[1])  # 按流量大小排序
    return result


def analyze_file(file_path, step):
    """分析单个文件，返回整体和分位数结果（基于FCT值）"""
    data = process_file(file_path)
    n_flows = len(data)
    if n_flows == 0:
        return [], []

    # 整体结果（基于FCT值）
    fct_all = sorted([x[0] for x in data])
    all_result = [
        np.average(fct_all),
        get_pctl(fct_all, 0.5),
        get_pctl(fct_all, 0.95),
        get_pctl(fct_all, 0.99)
    ]

    # 按步长计算分位数结果
    step_result = [[i / 100.0] for i in range(0, 100, step)]
    for i in range(0, 100, step):
        l = i * n_flows // 100
        r = (i + step) * n_flows // 100
        if l >= n_flows or r > n_flows:  # 防止越界
            continue

        d = data[l:r]
        fct = sorted([x[0] for x in d])  # 提取当前区间的FCT值并排序
        flow_size = d[-1][1]  # 区间内最大流量大小

        idx = i // step
        step_result[idx].append(flow_size)
        step_result[idx].append(np.average(fct))
        step_result[idx].append(get_pctl(fct, 0.5))
        step_result[idx].append(get_pctl(fct, 0.95))
        step_result[idx].append(get_pctl(fct, 0.99))

    return all_result, step_result


def analyze_small_and_large(file_path):
    """分析单个文件的小流量（<100KB）和大流量（>10MB）的FCT结果"""
    data = process_file(file_path)  # data = [fct, flow_size], 按flow_size排序
    n_flows = len(data)
    if n_flows == 0:
        return [0,0,0,0], [0,0,0,0]

    fct_small = []  # <100KB的FCT
    fct_large = []  # >10MB的FCT

    for flow in data:
        if flow[1] < 100000:  # 100KB = 100,000字节
            fct_small.append(flow[0])
        elif flow[1] > 10000000:  # 10MB = 10,000,000字节
            fct_large.append(flow[0])
    
    fct_small = sorted(fct_small)
    fct_large = sorted(fct_large)

    # 小流量结果
    small_result = [
        np.average(fct_small) if fct_small else 0,
        get_pctl(fct_small, 0.5) if fct_small else 0,
        get_pctl(fct_small, 0.95) if fct_small else 0,
        get_pctl(fct_small, 0.99) if fct_small else 0
    ]
    # 大流量结果
    large_result = [
        np.average(fct_large) if fct_large else 0,
        get_pctl(fct_large, 0.5) if fct_large else 0,
        get_pctl(fct_large, 0.95) if fct_large else 0,
        get_pctl(fct_large, 0.99) if fct_large else 0
    ]

    return small_result, large_result


def analyze_small_and_large_1(file_path):
    """分析单个文件的小流量（<10KB）和大流量（>1MB）的FCT结果"""
    data = process_file(file_path)  # data = [fct, flow_size], 按flow_size排序
    n_flows = len(data)
    if n_flows == 0:
        return [0,0,0,0], [0,0,0,0]

    fct_small = []  # <10KB的FCT
    fct_large = []  # >1MB的FCT

    for flow in data:
        if flow[1] < 10000:  # 10KB = 10,000字节
            fct_small.append(flow[0])
        elif flow[1] > 1000000:  # 1MB = 1,000,000字节
            fct_large.append(flow[0])
    
    fct_small = sorted(fct_small)
    fct_large = sorted(fct_large)

    # 小流量结果
    small_result = [
        np.average(fct_small) if fct_small else 0,
        get_pctl(fct_small, 0.5) if fct_small else 0,
        get_pctl(fct_small, 0.95) if fct_small else 0,
        get_pctl(fct_small, 0.99) if fct_small else 0
    ]
    # 大流量结果
    large_result = [
        np.average(fct_large) if fct_large else 0,
        get_pctl(fct_large, 0.5) if fct_large else 0,
        get_pctl(fct_large, 0.95) if fct_large else 0,
        get_pctl(fct_large, 0.99) if fct_large else 0
    ]

    return small_result, large_result


def overall():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='分析FCT文件，输出整体和分位数结果（基于实际FCT值）')
    parser.add_argument('-d', dest='directory', action='store', default="/home/ame/copter/simulation/output", 
                        help="指定包含fct文件的目录")
    parser.add_argument('-p', dest='prefix', action='store', default='', 
                        help="指定fct文件的前缀")
    parser.add_argument('-r', dest='surfix', action='store', default='',
                        help='指定fct文件的后缀')
    parser.add_argument('-s', dest='step', action='store', default='5', 
                        help="分位数计算的步长")
    args = parser.parse_args()

    directory = args.directory
    prefix = args.prefix
    surfix = args.surfix
    step = int(args.step)

    fct_files = [f for f in os.listdir(directory) 
                if f.startswith(prefix) and f.endswith(f"{surfix}.fct")]
    if not fct_files:
        print(f"在目录 {directory} 中未找到前缀为 '{prefix}' 且后缀为 '{surfix}.fct' 的文件")
        return

    # 创建结果目录
    if not os.path.exists("result"):
        os.makedirs("result")

    overall_output_file = os.path.join("result", f"{prefix}_load{surfix}_overall.csv")
    with open(overall_output_file, 'w') as overall_f:
        overall_f.write("File,Average FCT,Median FCT,95th FCT,99th FCT\n")

        for fct_file in fct_files:
            file_path = os.path.join(directory, fct_file)
            all_result, step_result = analyze_file(file_path, step)
            if not all_result:
                continue

            # 写入整体结果
            overall_f.write(f"{fct_file},{all_result[0]:.3f},{all_result[1]:.3f},{all_result[2]:.3f},{all_result[3]:.3f}\n")

            # 写入分位数结果到单独文件
            output_file = os.path.join("result", f"{fct_file}.csv")
            with open(output_file, 'w') as out_f:
                out_f.write("Percentile,Size,Average FCT,Median FCT,95th FCT,99th FCT\n")
                for item in step_result:
                    if len(item) < 6:  # 跳过无效数据
                        continue
                    out_f.write(f"{item[0]:.3f},{item[1]},{item[2]:.3f},{item[3]:.3f},{item[4]:.3f},{item[5]:.3f}\n")


def partial():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='分析FCT文件，输出小流量和大流量的FCT结果（基于实际FCT值）')
    parser.add_argument('-d', dest='directory', action='store', default="/home/ame/copter/simulation/output", 
                        help="指定包含fct文件的目录")
    parser.add_argument('-p', dest='prefix', action='store', default='', 
                        help="指定fct文件的前缀")
    parser.add_argument('-r', dest='surfix', action='store', default='',
                        help='指定fct文件的后缀')
    parser.add_argument('-s', dest='step', action='store', default='5', 
                        help="分位数计算的步长")
    args = parser.parse_args()

    directory = args.directory
    prefix = args.prefix
    surfix = args.surfix
    step = int(args.step)

    fct_files = [f for f in os.listdir(directory) 
                if f.startswith(prefix) and f.endswith(f"{surfix}.fct")]
    if not fct_files:
        print(f"在目录 {directory} 中未找到前缀为 '{prefix}' 且后缀为 '{surfix}.fct' 的文件")
        return

    # 创建结果目录
    if not os.path.exists("result"):
        os.makedirs("result")

    overall_output_file = os.path.join("result", f"{prefix}_load{surfix}_partial.csv")
    with open(overall_output_file, 'w') as overall_f:
        overall_f.write("File,Size,Average FCT,Median FCT,95th FCT,99th FCT\n")
        for fct_file in fct_files:
            file_path = os.path.join(directory, fct_file)
            small_result, large_result = analyze_small_and_large(file_path)
            overall_f.write(f"{fct_file},< 100KB,{small_result[0]:.3f},{small_result[1]:.3f},{small_result[2]:.3f},{small_result[3]:.3f}\n")
            overall_f.write(f"{fct_file},> 10MB,{large_result[0]:.3f},{large_result[1]:.3f},{large_result[2]:.3f},{large_result[3]:.3f}\n")


def partial_1():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='分析FCT文件，输出小流量和大流量的FCT结果（基于实际FCT值）')
    parser.add_argument('-d', dest='directory', action='store', default="/home/ame/copter/simulation/output", 
                        help="指定包含fct文件的目录")
    parser.add_argument('-p', dest='prefix', action='store', default='', 
                        help="指定fct文件的前缀")
    parser.add_argument('-r', dest='surfix', action='store', default='',
                        help='指定fct文件的后缀')
    parser.add_argument('-s', dest='step', action='store', default='5', 
                        help="分位数计算的步长")
    args = parser.parse_args()

    directory = args.directory
    prefix = args.prefix
    surfix = args.surfix
    step = int(args.step)

    fct_files = [f for f in os.listdir(directory) 
                if f.startswith(prefix) and f.endswith(f"{surfix}.fct")]
    if not fct_files:
        print(f"在目录 {directory} 中未找到前缀为 '{prefix}' 且后缀为 '{surfix}.fct' 的文件")
        return

    # 创建结果目录
    if not os.path.exists("result"):
        os.makedirs("result")

    overall_output_file = os.path.join("result", f"{prefix}_load{surfix}_partial_1.csv")
    with open(overall_output_file, 'w') as overall_f:
        overall_f.write("File,Size,Average FCT,Median FCT,95th FCT,99th FCT\n")
        for fct_file in fct_files:
            file_path = os.path.join(directory, fct_file)
            small_result, large_result = analyze_small_and_large_1(file_path)
            overall_f.write(f"{fct_file},< 10KB,{small_result[0]:.3f},{small_result[1]:.3f},{small_result[2]:.3f},{small_result[3]:.3f}\n")
            overall_f.write(f"{fct_file},> 1MB,{large_result[0]:.3f},{large_result[1]:.3f},{large_result[2]:.3f},{large_result[3]:.3f}\n")


if __name__ == "__main__":
    file_dir = "/home/ame/copter/simulation/output/acc_dcqcn_06load_hadoop_5paramStep/"
    file_list = [
        "_dcqcn_06load_hadoop_5paramStep.fct"
        # "acc_webserver_incast.fct" # 可修改为需要分析的文件
    ]

    output_path = "acc_dcqcn_06load_hadoop_5paramStep"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for file in file_list:
        file_path = os.path.join(file_dir, file)
        result_path = os.path.join(output_path, file)

        all_result, step_result = analyze_file(file_path, 5)
        small_result, large_result = analyze_small_and_large(file_path)
        small_result_1, large_result_1 = analyze_small_and_large_1(file_path)

        with open(result_path, 'w') as out_f:
            out_f.write("Overall FCT: \tAvg\tMid\t95th\t99th\n")
            if all_result:
                out_f.write(f"\t\t{all_result[0]:.3f}\t{all_result[1]:.3f}\t{all_result[2]:.3f}\t{all_result[3]:.3f}\t\n")
            else:
                out_f.write("\t\t0.000\t0.000\t0.000\t0.000\t\n")
            
            out_f.write("Percent\tSize\tAvg\tMid\t95th\t99th\n")
            for item in step_result:
                if len(item) < 6:
                    continue
                out_f.write(f"{item[0]:.3f}\t{item[1]}\t{item[2]:.3f}\t{item[3]:.3f}\t{item[4]:.3f}\t{item[5]:.3f}\n")
            
            out_f.write(f"< 100KB: Avg: {small_result[0]:.3f}, Mid: {small_result[1]:.3f}, 95th: {small_result[2]:.3f}, 99th: {small_result[3]:.3f}\n")
            out_f.write(f"> 10MB: Avg: {large_result[0]:.3f}, Mid: {large_result[1]:.3f}, 95th: {large_result[2]:.3f}, 99th: {large_result[3]:.3f}\n")
            out_f.write(f"< 10KB: Avg: {small_result_1[0]:.3f}, Mid: {small_result_1[1]:.3f}, 95th: {small_result_1[2]:.3f}, 99th: {small_result_1[3]:.3f}\n")
            out_f.write(f"> 1MB: Avg: {large_result_1[0]:.3f}, Mid: {large_result_1[1]:.3f}, 95th: {large_result_1[2]:.3f}, 99th: {large_result_1[3]:.3f}\n")

        print(f"已处理 {file}，结果保存至 {result_path}")