import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Flow:
    flow_id: int
    flow_size: int
    start_time_ns: int
    bucket_time_ns: int
    ideal_fct_ns: int
    actual_fct_ns: int
    slowdown: float


def parse_fct_line(line, flow_id):
    """
    Parse a line from the FCT file. The input format will be like:
    Source Destination Field_1 Field_2 FlowSize StartTime IdealFCT ActualFCT
    """
    parts = line.strip().split()
    if len(parts) < 8:
        return None
    source, destination, field_1, field_2, flow_size, start_time, actual_fct, ideal_fct = parts[:8]
    flow_size = int(flow_size)
    start_time_ns = int(start_time)
    ideal_fct_ns = int(ideal_fct)
    actual_fct_ns = int(actual_fct)
    slowdown = max(1, actual_fct_ns / ideal_fct_ns)
    bucket_time_ns = int(start_time_ns + actual_fct_ns/2)
    return Flow(flow_id, flow_size, start_time_ns, bucket_time_ns, ideal_fct_ns, actual_fct_ns, slowdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-f', dest='filepath', action='store', default=None, help="Specify the path of the fct files")
    args = parser.parse_args()

    if args.filepath is None or not os.path.exists(args.filepath):
        print("Please specify a valid file path.")
        exit(1)
    
    filepath = args.filepath

    flows = []
    total_flows = 0

    with open(filepath, 'r') as f:
        for line in f:
            flow = parse_fct_line(line, total_flows)
            if flow is not None:
                flows.append(flow)
                total_flows += 1
    
    # Calculate average FCT and 99th percentile FCT in buckets, the bucket size is 100us = 100000 ns
    bucket_size_ns = 1000000
    buckets = {}
    for flow in flows:
        bucket_index = flow.start_time_ns // bucket_size_ns
        if bucket_index not in buckets:
            buckets[bucket_index] = []
        buckets[bucket_index].append(flow)
    avg_fct = []
    p99_fct = []
    for bucket_index, bucket_flows in buckets.items():
        if not bucket_flows:
            continue
        avg_fct_value = np.mean([flow.slowdown for flow in bucket_flows])
        p99_fct_value = np.percentile([flow.slowdown for flow in bucket_flows], 99)
        avg_fct.append((bucket_index * bucket_size_ns, avg_fct_value))
        p99_fct.append((bucket_index * bucket_size_ns, p99_fct_value))

    avg_fct.sort(key=lambda x: x[0])  # Sort by time
    p99_fct.sort(key=lambda x: x[0])  # Sort by time

    # Draw the "AverageFCT - bucket" and "99thFCT - bucket" curves
    avg_fct = np.array(avg_fct)
    p99_fct = np.array(p99_fct)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_fct[:, 0], avg_fct[:, 1], label='Average FCT', color='blue')
    plt.plot(p99_fct[:, 0], p99_fct[:, 1], label='99th Percentile FCT', color='red')
    plt.xlabel('Time (ns)') 
    plt.ylabel('FCT (ns)')
    plt.title('Average and 99th Percentile FCT over Time')
    plt.legend()
    plt.grid()
    plt.savefig('fct_analysis.png')
    # plt.show()

    