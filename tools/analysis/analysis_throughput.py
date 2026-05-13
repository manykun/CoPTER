import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Data file paths
file_paths = {
    'acc': '/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_random/acc_thesis_mix_webserver_websearch_random.throughput',
    'copter': '/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_random/copter_thesis_mix_webserver_websearch_random.throughput',
    'm3': '/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_random/m3_thesis_mix_webserver_websearch_random.throughput',
    'm4': '/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_random/m4_thesis_mix_webserver_websearch_random.throughput',
    'dcqcn': '/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_random/dcqcn_thesis_mix_webserver_websearch_random.throughput',
    'hpcc': '/home/ame/copter/simulation/output/thesis_mix_webserver_websearch_random/hpcc_thesis_mix_webserver_websearch_random.throughput'
}

# Output directory - updated to match the path from your error message
output_dir = '/home/ame/copter/tools/analysis/thesis_mix_webserver_websearch_random/throughput_analysis'
os.makedirs(output_dir, exist_ok=True)

def load_throughput_data(file_path):
    """Load and preprocess throughput data from file"""
    try:
        columns = ['switch_id', 'node_id', 'throughput_bps', 'timestamp', 'max_port_rate']
        
        df = pd.read_csv(
            file_path, 
            sep='\s+',
            header=None, 
            names=columns
        )
        
        df['throughput_mbps'] = df['throughput_bps'] / 1e6
        df['max_port_rate_mbps'] = df['max_port_rate'] / 1e6
        
        return df
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None

def analyze_throughput(df, label):
    """Calculate key statistics for throughput data"""
    if df is None or df.empty:
        return None
    
    stats = {
        'label': label,
        'total_records': len(df),
        'avg_throughput_mbps': df['throughput_mbps'].mean(),
        'max_throughput_mbps': df['throughput_mbps'].max(),
        'min_throughput_mbps': df['throughput_mbps'].min(),
        'p95_throughput_mbps': df['throughput_mbps'].quantile(0.95),
        'p99_throughput_mbps': df['throughput_mbps'].quantile(0.99),
        'unique_switches': df['switch_id'].nunique(),
        'unique_nodes': df['node_id'].nunique()
    }
    
    return stats

def plot_time_series(data_dict, output_dir):
    """Plot throughput trend over time"""
    plt.figure(figsize=(12, 6))
    
    for label, df in data_dict.items():
        if df is not None and not df.empty:
            df_sorted = df.sort_values('timestamp')
            time_avg = df_sorted.groupby('timestamp')['throughput_mbps'].mean().reset_index()
            plt.plot(time_avg['timestamp'], time_avg['throughput_mbps'], label=label)
    
    plt.title('Throughput Trend Over Time', fontsize=14, pad=15)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Average Throughput (Mbps)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'throughput_time_series.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Time series plot saved to: {output_path}")

def plot_distribution(data_dict, output_dir):
    """Box plot comparison of throughput distributions"""
    plt.figure(figsize=(10, 6))
    
    plot_data = []
    plot_labels = []
    for label, df in data_dict.items():
        if df is not None and not df.empty:
            non_zero_throughput = df[df['throughput_mbps'] > 0]['throughput_mbps']
            plot_data.append(non_zero_throughput)
            plot_labels.append(label)
    
    # Create box plot without custom styling that caused error
    plt.boxplot(plot_data, labels=plot_labels, showfliers=False)
    
    # Alternative styling that works with most matplotlib versions
    plt.title('Throughput Distribution Comparison', fontsize=14, pad=15)
    plt.ylabel('Throughput (Mbps)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'throughput_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution box plot saved to: {output_path}")

def plot_top_switches(data_dict, output_dir, top_n=5):
    """Bar plot of top N switches by average throughput"""
    # 获取有效数据的数量
    valid_count = sum(1 for df in data_dict.values() if df is not None and not df.empty)
    
    # 根据数据数量调整子图布局
    if valid_count == 3:
        # 3个数据时使用1行3列布局
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    else:
        # 默认使用1行2列布局
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 确保axes是数组形式，方便索引
    if valid_count == 1:
        axes = [axes]
    
    for idx, (label, df) in enumerate(data_dict.items()):
        if df is not None and not df.empty and idx < len(axes):
            ax = axes[idx]
            
            switch_avg = df.groupby('switch_id')['throughput_mbps'].mean().sort_values(ascending=False)
            top_switches = switch_avg.head(top_n)
            
            bars = ax.bar(top_switches.index.astype(str), top_switches.values, color='skyblue')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f'{label} - Top {top_n} Switches by Throughput', fontsize=12, pad=12)
            ax.set_xlabel('Switch ID', fontsize=11)
            ax.set_ylabel('Average Throughput (Mbps)', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            ax.tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'top_switches.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Top switches plot saved to: {output_path}")

def plot_throughput_vs_max_rate(data_dict, output_dir):
    """Scatter plot of port utilization"""
    plt.figure(figsize=(12, 6))
    
    for label, df in data_dict.items():
        if df is not None and not df.empty:
            port_stats = df.groupby(['switch_id', 'node_id']).agg({
                'throughput_mbps': 'mean',
                'max_port_rate_mbps': 'first'
            }).reset_index()
            
            port_stats['utilization_pct'] = (port_stats['throughput_mbps'] / port_stats['max_port_rate_mbps']) * 100
            
            plt.scatter(
                port_stats.index,
                port_stats['utilization_pct'],
                alpha=0.6,
                label=f'{label} Port Utilization (%)'
            )
    
    plt.axhline(y=100, color='red', linestyle='--', linewidth=1.5, label='100% Utilization')
    
    plt.title('Port Utilization Distribution', fontsize=14, pad=15)
    plt.xlabel('Port Index', fontsize=12)
    plt.ylabel('Utilization (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'port_utilization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Port utilization plot saved to: {output_path}")

def save_statistics(stats_list, output_dir):
    """Save key statistics to a text report"""
    output_path = os.path.join(output_dir, 'throughput_statistics.txt')
    
    with open(output_path, 'w') as f:
        f.write(f"Throughput Data Analysis Report\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for stats in stats_list:
            if stats:
                f.write(f"Dataset: {stats['label']}\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Records: {stats['total_records']}\n")
                f.write(f"Number of Unique Switches: {stats['unique_switches']}\n")
                f.write(f"Number of Unique Nodes: {stats['unique_nodes']}\n")
                f.write(f"Average Throughput: {stats['avg_throughput_mbps']:.2f} Mbps\n")
                f.write(f"Maximum Throughput: {stats['max_throughput_mbps']:.2f} Mbps\n")
                f.write(f"Minimum Throughput: {stats['min_throughput_mbps']:.2f} Mbps\n")
                f.write(f"95th Percentile Throughput: {stats['p95_throughput_mbps']:.2f} Mbps\n")
                f.write(f"99th Percentile Throughput: {stats['p99_throughput_mbps']:.2f} Mbps\n")
                f.write("\n")
    
    print(f"Statistics report saved to: {output_path}")

def main():
    """Main function to load data, run analysis, and generate visualizations"""
    data_dict = {}
    stats_list = []
    
    print("Starting throughput data analysis...\n")
    for label, file_path in file_paths.items():
        print(f"Loading {label} dataset from: {file_path}")
        df = load_throughput_data(file_path)
        data_dict[label] = df
        
        stats = analyze_throughput(df, label)
        if stats:
            stats_list.append(stats)
    
    print("\nGenerating visualizations...")
    plot_time_series(data_dict, output_dir)
    plot_distribution(data_dict, output_dir)
    plot_top_switches(data_dict, output_dir, top_n=5)
    plot_throughput_vs_max_rate(data_dict, output_dir)
    
    save_statistics(stats_list, output_dir)
    
    print("\nAnalysis completed successfully!")
    print(f"All results saved to: {output_dir}")

if __name__ == "__main__":
    main()
    