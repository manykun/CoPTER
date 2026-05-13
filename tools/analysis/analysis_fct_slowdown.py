import argparse
import numpy as np
import os


def get_pctl(a, p):
    """计算指定百分位值"""
    i = int(len(a) * p)
    return a[i]

def process_file(file_path):
    """处理文件并返回按流量大小排序的 [slowdown, flow_size] 列表"""
    result = []
    with open(file_path, 'r') as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) < 8:  # 确保行有足够的字段
                continue
            flow_size = int(fields[4])        # $5
            fct = int(fields[6])              # $7
            ideal_fct = int(fields[7])        # $8
            slowdown = fct / ideal_fct
            if slowdown < 1:
                slowdown = 1
            result.append([slowdown, flow_size])
    result.sort(key=lambda x: x[1])
    return result


def analyze_file(file_path, step):
    """分析单个文件，返回整体和分位数结果"""
    data = process_file(file_path)
    n_flows = len(data)

    # 整体结果
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
        fct = sorted([x[0] for x in d])
        flow_size = d[-1][1]

        idx = i // step
        step_result[idx].append(flow_size)
        step_result[idx].append(np.average(fct))
        step_result[idx].append(get_pctl(fct, 0.5))
        step_result[idx].append(get_pctl(fct, 0.95))
        step_result[idx].append(get_pctl(fct, 0.99))

    return all_result, step_result


def analyze_small_and_large(file_path):
    """分析单个文件，返回整体和分位数结果"""
    data = process_file(file_path)  # data = [slowdown, flow_size], 按flow_size排序
    n_flows = len(data)

    fct_small = []
    fct_large = []

    for flow in data:
        if flow[1] < 100000:
            fct_small.append(flow[0])
        elif flow[1] > 10000000:
            fct_large.append(flow[0])
    
    fct_small = sorted(fct_small)
    fct_large = sorted(fct_large)

    small_result = [
        np.average(fct_small),
        get_pctl(fct_small, 0.5),
        get_pctl(fct_small, 0.95),
        get_pctl(fct_small, 0.99)
    ]
    if len(fct_large) != 0:
        large_result = [
            np.average(fct_large),
            get_pctl(fct_large, 0.5),
            get_pctl(fct_large, 0.95),
            get_pctl(fct_large, 0.99)
        ]
    else:
        large_result = [0, 0, 0, 0]

    return small_result, large_result



def analyze_small_and_large_1(file_path):
    """分析单个文件，返回整体和分位数结果"""
    data = process_file(file_path)  # data = [slowdown, flow_size], 按flow_size排序
    n_flows = len(data)

    fct_small = []
    fct_large = []

    for flow in data:
        if flow[1] < 10000:
            fct_small.append(flow[0])
        elif flow[1] > 1000000:
            fct_large.append(flow[0])
    
    fct_small = sorted(fct_small)
    fct_large = sorted(fct_large)

    small_result = [
        np.average(fct_small),
        get_pctl(fct_small, 0.5),
        get_pctl(fct_small, 0.95),
        get_pctl(fct_small, 0.99)
    ]
    large_result = [
        np.average(fct_large),
        get_pctl(fct_large, 0.5),
        get_pctl(fct_large, 0.95),
        get_pctl(fct_large, 0.99)
    ]


    return small_result, large_result


def overall():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', dest='directory', action='store', default="/home/ame/copter/simulation/output", 
                        help="Specify the directory containing fct files")
    parser.add_argument('-p', dest='prefix', action='store', default='', 
                        help="Specify the prefix of the fct files")
    parser.add_argument('-r', dest='surfix', action='store', default='',
                        help='Specify the surfix of the fct files')
    parser.add_argument('-s', dest='step', action='store', default='5', 
                        help="Step size for percentile calculation")
    args = parser.parse_args()

    directory = args.directory
    prefix = args.prefix
    surfix = args.surfix
    step = int(args.step)

    fct_files = [f for f in os.listdir(directory) 
                if f.startswith(prefix) and f.endswith(f"{surfix}.fct")]
    if not fct_files:
        print(f"No files found with prefix '{prefix}' and suffix '.fct' in {directory}")
        return

    overall_output_file = os.path.join("result", f"{prefix}_load{surfix}_overall.csv")
    with open(overall_output_file, 'w') as overall_f:
        overall_f.write("File,Average FCT,Median FCT,95th FCT,99th FCT\n")

        for fct_file in fct_files:
            file_path = os.path.join(directory, fct_file)
            all_result, step_result = analyze_file(file_path, step)

            # 写入整体结果
            overall_f.write(f"{fct_file},{all_result[0]:.3f},{all_result[1]:.3f},{all_result[2]:.3f},{all_result[3]:.3f}\n")

            # 写入分位数结果到单独文件
            output_file = os.path.join("result", f"{fct_file}.csv")
            with open(output_file, 'w') as out_f:
                out_f.write("Percentile,Size,Average FCT,Median FCT,95th FCT,99th FCT\n")
                for item in step_result:
                    out_f.write(f"{item[0]:.3f},{item[1]},{item[2]:.3f},{item[3]:.3f},{item[4]:.3f},{item[5]:.3f}\n")


def partial():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', dest='directory', action='store', default="/home/ame/copter/simulation/output", 
                        help="Specify the directory containing fct files")
    parser.add_argument('-p', dest='prefix', action='store', default='', 
                        help="Specify the prefix of the fct files")
    parser.add_argument('-r', dest='surfix', action='store', default='',
                        help='Specify the surfix of the fct files')
    parser.add_argument('-s', dest='step', action='store', default='5', 
                        help="Step size for percentile calculation")
    args = parser.parse_args()

    directory = args.directory
    prefix = args.prefix
    surfix = args.surfix
    step = int(args.step)

    fct_files = [f for f in os.listdir(directory) 
                if f.startswith(prefix) and f.endswith(f"{surfix}.fct")]
    if not fct_files:
        print(f"No files found with prefix '{prefix}' and suffix '.fct' in {directory}")
        return

    overall_output_file = os.path.join("result", f"{prefix}_load{surfix}_partial.csv")
    with open(overall_output_file, 'w') as overall_f:
        overall_f.write("File,Size,Median FCT,95th FCT,99th FCT\n")
        for fct_file in fct_files:
            file_path = os.path.join(directory, fct_file)
            small_result, large_result = analyze_small_and_large(file_path)
            overall_f.write(f"{fct_file},< 100KB, {small_result[0]:.3f},{small_result[1]:.3f},{small_result[2]:.3f},{small_result[3]:.3f}\n")
            overall_f.write(f"{fct_file},> 10MB, {large_result[0]:.3f},{large_result[1]:.3f},{large_result[2]:.3f},{large_result[3]:.3f}\n")

def partial_1():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', dest='directory', action='store', default="/home/ame/copter/simulation/output", 
                        help="Specify the directory containing fct files")
    parser.add_argument('-p', dest='prefix', action='store', default='', 
                        help="Specify the prefix of the fct files")
    parser.add_argument('-r', dest='surfix', action='store', default='',
                        help='Specify the surfix of the fct files')
    parser.add_argument('-s', dest='step', action='store', default='5', 
                        help="Step size for percentile calculation")
    args = parser.parse_args()

    directory = args.directory
    prefix = args.prefix
    surfix = args.surfix
    step = int(args.step)

    fct_files = [f for f in os.listdir(directory) 
                if f.startswith(prefix) and f.endswith(f"{surfix}.fct")]
    if not fct_files:
        print(f"No files found with prefix '{prefix}' and suffix '.fct' in {directory}")
        return

    overall_output_file = os.path.join("result", f"{prefix}_load{surfix}_partial_1.csv")
    with open(overall_output_file, 'w') as overall_f:
        overall_f.write("File,Size,Median FCT,95th FCT,99th FCT\n")
        for fct_file in fct_files:
            file_path = os.path.join(directory, fct_file)
            small_result, large_result = analyze_small_and_large_1(file_path)
            overall_f.write(f"{fct_file},< 10KB, {small_result[0]:.3f},{small_result[1]:.3f},{small_result[2]:.3f},{small_result[3]:.3f}\n")
            overall_f.write(f"{fct_file},> 1MB, {large_result[0]:.3f},{large_result[1]:.3f},{large_result[2]:.3f},{large_result[3]:.3f}\n")

if __name__ == "__main__":
    # file = "/home/ame/copter/simulation/output/GoogleRPCHPCC_SECN_load0.9.fct"
    
    # file_dir = "/home/ame/copter/simulation/output/"
    file_dir = "/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_cachefollower_random"
    file_list = [
        # "WebServerDCQCN_SECN_load0.7.fct"
        "acc_thesis_mix_webserver_websearch_cachefollower_random.fct",
        "copter_thesis_mix_webserver_websearch_cachefollower_random.fct",
        "m3_thesis_mix_webserver_websearch_cachefollower_random.fct",
        "dcqcn_thesis_mix_webserver_websearch_cachefollower_random.fct",
        "hpcc_thesis_mix_webserver_websearch_cachefollower_random.fct",
        # "m4_thesis_mix_webserver_websearch_cachefollower_random.fct"
        # "m3_mix_webserver_websearch_hadoop_clusters.fct"
        # "copter_webserver_t0.05_l0.7 copy.fct"
        # "_dcqcn_06load_hadoop_5paramStep.fct"
        # "copter_Hadoop_n256_t0.05_l0.9.fct"
        # "acc_webserver_incast.fct"
        # "copter_webserver_incast_m3.fct"
        # "copter_webserver_incast_like_acc.fct"
        # "copter_webserver_incast.fct"
        # "acc_Hadoop_n256_t0.05_l0.9.fct"
        # "new_copter_webserver_t0.05_l0.7.fct"
        # "acc_mix_webserver_websearch_hadoop_short.fct",
        # "copter_mix_webserver_websearch_hadoop_short.fct"
    ]
    # file_list = [
    #     "GoogleRPCHPCC_SECN_load0.3.fct",
    #     "GoogleRPCHPCC_SECN_load0.5.fct",
    #     "GoogleRPCHPCC_SECN_load0.7.fct",
    #     "GoogleRPCHPCC_SECN_load0.9.fct",
    #     "GoogleRPCDCQCN_SECN_load0.3.fct",
    #     "GoogleRPCDCQCN_SECN_load0.5.fct",
    #     "GoogleRPCDCQCN_SECN_load0.7.fct",
    #     "GoogleRPCDCQCN_SECN_load0.9.fct",
    #     "WebServerHPCC_SECN_load0.3.fct",
    #     "WebServerHPCC_SECN_load0.5.fct",
    #     "WebServerHPCC_SECN_load0.7.fct",
    #     "WebServerHPCC_SECN_load0.9.fct",
    #     "WebServerDCQCN_SECN_load0.3.fct",
    #     "WebServerDCQCN_SECN_load0.5.fct",
    #     "WebServerDCQCN_SECN_load0.7.fct",
    #     "WebServerDCQCN_SECN_load0.9.fct",
    #     "HadoopHPCC_SECN_load0.3.fct",
    #     "HadoopHPCC_SECN_load0.5.fct",
    #     "HadoopHPCC_SECN_load0.7.fct",
    #     "HadoopHPCC_SECN_load0.9.fct",
    #     "HadoopDCQCN_SECN_load0.3.fct",
    #     "HadoopDCQCN_SECN_load0.5.fct",
    #     "HadoopDCQCN_SECN_load0.7.fct",
    #     "HadoopDCQCN_SECN_load0.9.fct",
    # ]

    output_path = "thesis_mix_webserver_websearch_cachefollower_random"
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
            out_f.write(f"\t\t{all_result[0]:.3f}\t{all_result[1]:.3f}\t{all_result[2]:.3f}\t{all_result[3]:.3f}\t\n")
            out_f.write("Percent\tSize\tAvg\tMid\t95th\t99th\n")
            for item in step_result:
                out_f.write(f"{item[0]:.3f}\t{item[1]:2.3f}\t{item[2]:.3f}\t{item[3]:.3f}\t{item[4]:.3f}\t{item[5]:.3f}\t\n")
            
            out_f.write(f"< 100KB: Avg: {small_result[0]:.3f}, Mid: {small_result[1]:.3f}, 95th: {small_result[2]:.3f}, 99th: {small_result[3]:.3f}\n")
            out_f.write(f"> 10MB: Avg: {large_result[0]:.3f}, Mid: {large_result[1]:.3f}, 95th: {large_result[2]:.3f}, 99th: {large_result[3]:.3f}\n")
            out_f.write(f"< 10KB: Avg: {small_result_1[0]:.3f}, Mid: {small_result_1[1]:.3f}, 95th: {small_result_1[2]:.3f}, 99th: {small_result_1[3]:.3f}\n")
            out_f.write(f"> 1MB: Avg: {large_result_1[0]:.3f}, Mid: {large_result_1[1]:.3f}, 95th: {large_result_1[2]:.3f}, 99th: {large_result_1[3]:.3f}\n")

        print(f"Processed {file} and saved results to {result_path}")

    # overall()
    # partial()
    # partial_1()
    # data = process_file(file)
    # n_flows = len(data)

    # # Get Overall Result
    # fct_all = sorted([x[0] for x in data])
    # all_result = []
    # all_result.append(np.average(fct_all))
    # all_result.append(get_pctl(fct_all, 0.5))
    # all_result.append(get_pctl(fct_all, 0.95))
    # all_result.append(get_pctl(fct_all, 0.99))
    
    # print("Overall FCT: \tAvg\tMid\t95th\t99th")
    # print(f"\t\t{all_result[0]:.3f}\t{all_result[1]:.3f}\t{all_result[2]:.3f}\t{all_result[3]:.3f}\t")

    # # Get Percentile Result with Steps
    # step_result = [[i / 100.0] for i in range(0, 100, step)]
    # for i in range(0, 100, step):
    #     l = i * n_flows // 100
    #     r = (i + step) * n_flows // 100
    #     if l >= n_flows or r > n_flows:  # 防止越界
    #         continue

    #     # 提取当前区间的子集
    #     d = data[l:r]
    #     fct = sorted([x[0] for x in d])  # 提取 slowdown 并排序
    #     flow_size = d[-1][1]  # 区间内最大流量大小

    #     # 添加统计结果
    #     idx = i // step
    #     step_result[idx].append(flow_size)         # 流量大小
    #     step_result[idx].append(np.average(fct))
    #     step_result[idx].append(get_pctl(fct, 0.5))  # 中位 FCT
    #     step_result[idx].append(get_pctl(fct, 0.95)) # 95 分位 FCT
    #     step_result[idx].append(get_pctl(fct, 0.99)) # 99 分位 FCT

    # # 输出结果
    # print("Percent\tSize\tAvg\tMid\t95th\t99th")
    # for item in step_result:
    #     print(f"{item[0]:.3f}\t{item[1]:2.3f}\t{item[2]:.3f}\t{item[3]:.3f}\t{item[4]:.3f}\t{item[5]:.3f}\t")