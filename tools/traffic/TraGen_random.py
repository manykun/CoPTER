# TraGen.py - Traffic Generator for Network Simulation (modified)
import sys
import os
import random
import math
import heapq
import json
from optparse import OptionParser


class CustomRand:
    def __init__(self):
        pass

    def testCdf(self, cdf):
        if cdf[0][1] != 0 or cdf[-1][1] != 100:
            return False
        for i in range(1, len(cdf)):
            if cdf[i][1] <= cdf[i - 1][1] or cdf[i][0] <= cdf[i - 1][0]:
                return False
        return True

    def setCdf(self, cdf):
        if not self.testCdf(cdf):
            return False
        self.cdf = cdf
        return True

    def getAvg(self):
        s = 0
        last_x, last_y = self.cdf[0]
        for x, y in self.cdf[1:]:
            s += (x + last_x) / 2.0 * (y - last_y)
            last_x, last_y = x, y
        return s / 100

    def rand(self):
        r = random.random() * 100
        return self.getValueFromPercentile(r)

    def getValueFromPercentile(self, y):
        for i in range(1, len(self.cdf)):
            if y <= self.cdf[i][1]:
                x0, y0 = self.cdf[i - 1]
                x1, y1 = self.cdf[i]
                return x0 + (x1 - x0) / (y1 - y0) * (y - y0)


def translate_bandwidth(b):
    if not b or not isinstance(b, str):
        return None
    if b.endswith('G'):
        return float(b[:-1]) * 1e9
    if b.endswith('M'):
        return float(b[:-1]) * 1e6
    if b.endswith('K'):
        return float(b[:-1]) * 1e3
    return float(b)


def poisson(lam):
    return -math.log(1 - random.random()) * lam


def load_cdf(file_path):
    with open(file_path, "r") as f:
        return [list(map(float, line.strip().split())) for line in f if line.strip()]


def ensure_unique_start(host_busy_times, src, start_ns):
    """
    如果 start_ns 与该 src 已有记录冲突，则微调 start_ns（+1 ns）直到不冲突。
    同时将 start_ns 记录到 host_busy_times[src]。
    """
    if src not in host_busy_times:
        host_busy_times[src] = set()
    # 微调直到唯一
    while start_ns in host_busy_times[src]:
        start_ns += 1
    host_busy_times[src].add(start_ns)
    return start_ns


def main():
    # 固定随机种子（如需完全随机可注释掉）
    random.seed(48)

    parser = OptionParser()
    parser.add_option("-b", "--bandwidth", dest="bandwidth", default="10G",
                      help="bandwidth of host link (G/M/K), default 10G")
    parser.add_option("-c", "--config", dest="group_config", default=None,
                      help="JSON file specifying flow generation groups")
    options, _ = parser.parse_args()

    if not options.group_config:
        print("Usage: --flow-groups <group_file.json> required")
        sys.exit(1)

    bandwidth = translate_bandwidth(options.bandwidth)
    config_dir = os.path.dirname(options.group_config) or "."
    config_name = os.path.basename(options.group_config)[:-7] if options.group_config.endswith('.json') else os.path.basename(options.group_config)

    # 输出路径
    output_txt = os.path.join("result", config_name + ".flow")
    output_json = os.path.join("result", config_name + "_flows.json")

    if bandwidth is None:
        print("Bandwidth format incorrect")
        sys.exit(1)

    # 创建result目录（如果不存在）
    os.makedirs("result", exist_ok=True)

    with open(options.group_config, 'r') as f:
        group_config = json.load(f)

    # 可选的随机选择集（按你的要求）
    LOAD_CHOICES = [0.6, 0.7, 0.8, 0.9]
    CDF_CHOICES = ["WebServer.txt", "WebSearch.txt", "cachefollower-all.txt"]

    flow_list = []  # 存储所有流，用于全局排序
    flow_count = 0

    # 记录每个源主机已占用的开始时间（纳秒），用于避免同一时间同一主机向多个目的主机发送
    host_busy_times = {}

    for idx, group in enumerate(group_config):
        # 随机覆盖 load 和 cdf（如果组中已显式指定则优先使用组中指定值）
        chosen_load = random.choice(LOAD_CHOICES)
        chosen_cdf = random.choice(CDF_CHOICES)

        src_hosts = group["src_hosts"]
        dst_hosts = group["dst_hosts"]
        cdf_path = os.path.join(config_dir, chosen_cdf)
        # 如果随机选择的 cdf 不存在，退回使用组内原始 cdf（如果有），否则报错并跳过该组
        if not os.path.isfile(cdf_path):
            alt = group.get("cdf", None)
            if alt:
                alt_path = os.path.join(config_dir, alt)
                if os.path.isfile(alt_path):
                    cdf_path = alt_path
                    print(f"[WARN] chosen cdf {chosen_cdf} not found, using group's cdf {alt}")
                else:
                    print(f"[ERROR] neither chosen cdf {chosen_cdf} nor group's cdf {alt} exists. Skipping group {idx}")
                    continue
            else:
                print(f"[ERROR] chosen cdf {chosen_cdf} not found and group has no fallback. Skipping group {idx}")
                continue

        start_time = int(group.get("start_time_s", 2) * 1e9)  # 纳秒
        duration = int(group.get("duration_s", 10) * 1e9)  # 纳秒
        load = float(chosen_load)
        pattern = group.get("pattern", "poisson_random")
        period = float(group.get("period_s", 1)) * 1e9  # 纳秒
        incast_dst_count = int(group.get("incast_dst_count", 1))
        reduce_group_size = int(group.get("reduce_group_size", 8))

        # 打印该组的最终选择，便于调试
        print(f"Processing group {idx}: src={src_hosts[0]}-{src_hosts[-1]} | chosen_cdf={os.path.basename(cdf_path)} | load={load} | pattern={pattern}")

        # 加载CDF文件
        try:
            cdf = load_cdf(cdf_path)
        except Exception as e:
            print(f"Failed to load CDF {cdf_path}: {e}")
            continue

        customRand = CustomRand()
        if not customRand.setCdf(cdf):
            print(f"Invalid CDF in {cdf_path}, skipping group {idx}")
            continue
        avg_size = customRand.getAvg()  # 流大小的平均值（字节）

        ###########################################################################
        # 1. 泊松随机模式（poisson_random / poisson_random）
        ###########################################################################
        if pattern == "poisson_random" or pattern == "poisson_random".lower():
            # 计算平均到达间隔（纳秒）
            avg_inter_arrival = 1 / (bandwidth * load / 8. / avg_size) * 1e9

            # 初始化小根堆：每个源主机的第一个流时间
            host_heap = [(start_time + int(poisson(avg_inter_arrival)), h) for h in src_hosts]
            heapq.heapify(host_heap)  # 最小堆用于主机事件

            while host_heap:
                t, src = host_heap[0]
                inter_t = int(poisson(avg_inter_arrival))
                dst = random.choice(dst_hosts)
                # 避免自环
                while dst == src:
                    dst = random.choice(dst_hosts)

                # 检查是否在时间窗口内
                if t + inter_t > start_time + duration:
                    heapq.heappop(host_heap)
                else:
                    size = max(1, int(customRand.rand()))
                    start_ns = int(t)
                    # 确保该 src 在该时间点没有重复发送（若冲突则微调）
                    start_ns = ensure_unique_start(host_busy_times, src, start_ns)
                    flow_list.append({
                        "id": flow_count,
                        "src": int(src),
                        "dst": int(dst),
                        "size": int(size),
                        "start_ns": int(start_ns),  # 纳秒时间，用于排序
                        "start_s": start_ns * 1e-9,  # 秒时间，用于写入
                        "pg": 3,
                        "dport": 100
                    })
                    flow_count += 1
                    # 更新堆
                    heapq.heapreplace(host_heap, (t + inter_t, src))

        ###########################################################################
        # 2. 泊松入向汇聚模式（poisson_incast）
        ###########################################################################
        elif pattern == "poisson_incast":
            # 校验汇聚目的数
            if incast_dst_count > len(dst_hosts):
                print(f"incast_dst_count ({incast_dst_count}) > len(dst_hosts) ({len(dst_hosts)})")
                continue

            # 选择固定的汇聚目的主机
            incast_dsts = random.sample(dst_hosts, k=incast_dst_count)
            print(f"Poisson Incast: {len(src_hosts)} sources → {incast_dst_count} destinations ({incast_dsts})")

            # 计算平均到达间隔
            avg_inter_arrival = 1 / (bandwidth * load / 8. / avg_size) * 1e9

            # 初始化小根堆
            host_heap = [(start_time + int(poisson(avg_inter_arrival)), h) for h in src_hosts]
            heapq.heapify(host_heap)

            while host_heap:
                t, src = host_heap[0]
                inter_t = int(poisson(avg_inter_arrival))
                # 从固定汇聚目的中选一个
                dst = random.choice(incast_dsts)
                while dst == src:
                    dst = random.choice(incast_dsts)

                # 检查时间窗口
                if t + inter_t > start_time + duration:
                    heapq.heappop(host_heap)
                else:
                    size = max(1, int(customRand.rand()))
                    start_ns = int(t)
                    start_ns = ensure_unique_start(host_busy_times, src, start_ns)
                    # 添加到流列表
                    flow_list.append({
                        "id": flow_count,
                        "src": int(src),
                        "dst": int(dst),
                        "size": int(size),
                        "start_ns": int(start_ns),
                        "start_s": start_ns * 1e-9,
                        "pg": 3,
                        "dport": 100
                    })
                    flow_count += 1
                    heapq.heapreplace(host_heap, (t + inter_t, src))

        ###########################################################################
        # 3. 全归约模式（all_reduce）
        ###########################################################################
        elif pattern == "all_reduce":
            # 划分All-Reduce组
            reduce_groups = []
            for i in range(0, len(src_hosts), reduce_group_size):
                group_hosts = src_hosts[i:i+reduce_group_size]
                if len(group_hosts) >= 2:  # 组内至少2台主机
                    reduce_groups.append(group_hosts)
            if not reduce_groups:
                print("No valid All-Reduce groups (need at least 2 hosts per group)")
                continue
            print(f"All-Reduce: {len(reduce_groups)} groups, each up to {reduce_group_size} hosts")

            # 计算周期数
            max_cycle = int(duration // period) if period > 0 else 1
            for cycle in range(max_cycle):
                cycle_start = start_time + cycle * int(period)
                if cycle_start > start_time + duration:
                    break

                # 遍历每个All-Reduce组
                for reduce_group in reduce_groups:
                    for src in reduce_group:
                        for dst in reduce_group:
                            if src == dst:
                                continue  # 跳过自环

                            # 流启动时间：周期起始时间 + 微秒级偏移
                            time_offset = random.randint(0, 1000)  # 0-1微秒偏移
                            flow_start = cycle_start + time_offset
                            size = max(1, int(customRand.rand()))

                            start_ns = int(flow_start)
                            start_ns = ensure_unique_start(host_busy_times, src, start_ns)

                            # 添加到流列表
                            flow_list.append({
                                "id": flow_count,
                                "src": int(src),
                                "dst": int(dst),
                                "size": int(size),
                                "start_ns": int(start_ns),
                                "start_s": start_ns * 1e-9,
                                "pg": 3,
                                "dport": 100
                            })
                            flow_count += 1

        ###########################################################################
        # 4. 全对全模式（all_to_all）
        ###########################################################################
        elif pattern == "all_to_all":
            # 校验源和目的集合
            valid_src = [s for s in src_hosts if s not in dst_hosts]
            if not valid_src:
                valid_src = src_hosts  # 允许源在目的集合中（但会跳过自环）
            if not src_hosts or not dst_hosts:
                print("src_hosts or dst_hosts is empty for all_to_all")
                continue
            print(f"All-to-All: {len(src_hosts)} sources → {len(dst_hosts)} destinations")

            # 计算平均到达间隔
            avg_inter_arrival = 1 / (bandwidth * load / 8. / avg_size) * 1e9

            # 初始化小根堆
            host_heap = []
            for src in src_hosts:
                first_time = start_time + int(poisson(avg_inter_arrival))
                host_heap.append((first_time, src, 0))
            heapq.heapify(host_heap)

            while host_heap:
                t, src, dst_idx = host_heap[0]
                inter_t = int(poisson(avg_inter_arrival))
                # 按索引选择目的
                dst = dst_hosts[dst_idx % len(dst_hosts)]
                # 跳过自环
                while dst == src:
                    dst_idx += 1
                    dst = dst_hosts[dst_idx % len(dst_hosts)]

                # 检查时间窗口
                if t + inter_t > start_time + duration:
                    heapq.heappop(host_heap)
                else:
                    size = max(1, int(customRand.rand()))
                    start_ns = int(t)
                    start_ns = ensure_unique_start(host_busy_times, src, start_ns)
                    # 添加到流列表
                    flow_list.append({
                        "id": flow_count,
                        "src": int(src),
                        "dst": int(dst),
                        "size": int(size),
                        "start_ns": int(start_ns),
                        "start_s": start_ns * 1e-9,
                        "pg": 3,
                        "dport": 100
                    })
                    flow_count += 1
                    # 更新堆
                    next_dst_idx = dst_idx + 1
                    heapq.heapreplace(host_heap, (t + inter_t, src, next_dst_idx))

        else:
            print(f"Unknown pattern: {pattern}")
            continue

    # 所有流按开始时间（纳秒）排序
    flow_list.sort(key=lambda x: x["start_ns"])

    # 写入TXT文件（排序后）
    with open(output_txt, "w") as txt_file:
        txt_file.write(f"{flow_count}\n")  # 写入流总数
        for flow in flow_list:
            txt_file.write(
                f"{flow['src']} {flow['dst']} {flow['pg']} {flow['dport']} {flow['size']} {flow['start_s']:.9f}\n"
            )

    # 准备JSON输出数据
    json_output = [
        {
            "id": flow["id"],
            "src": flow["src"],
            "dst": flow["dst"],
            "size": flow["size"],
            "start": flow["start_ns"]
        }
        for flow in flow_list
    ]

    # 写入JSON文件
    with open(output_json, "w") as json_file:
        json.dump(
            json_output,
            json_file,
            indent=2,
            default=lambda x: int(x) if isinstance(x, float) and x.is_integer() else x
        )

    print(f"Generated {flow_count} flows. Saved to {output_txt} and {output_json}.")


if __name__ == "__main__":
    main()
