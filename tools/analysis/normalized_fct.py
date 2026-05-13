import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

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

# -------------------------- 样式映射（保持与原脚本一致） --------------------------
name_mapping = {
    "copter": "CoPT", 
    "acc": "ACC",
    "m4": "SCoPE" 
}
color_map = {
    "copter": "#FF6B00",
    "acc": "#00CC66",
    "m4": "#0066FF",
}
markers = {
    "copter": '^',
    "acc": 'o',
    "m4": 's',
}
line_styles = {
    "copter": '-',
    "acc": '-',
    "m4": '-',
}

# -------------------------- 辅助函数：清理数值字符串（去除逗号和空白字符） --------------------------
def clean_numeric_str(s):
    """去除字符串中的逗号、空白字符，确保能转换为float"""
    return s.strip().strip(',').strip()

# -------------------------- 数据解析函数：修复尺寸指标匹配冲突 --------------------------
def parse_overall_fct(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # 1. 解析Overall FCT（原有逻辑保留）
            overall_pattern = r"Overall FCT:\s+Avg\s+Mid\s+95th\s+99th\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
            match_overall = re.search(overall_pattern, content, re.IGNORECASE | re.DOTALL)
            overall_data = None
            
            if match_overall:
                overall_data = {
                    "Avg": float(clean_numeric_str(match_overall.group(1))),
                    "Mid": float(clean_numeric_str(match_overall.group(2))),
                    "95th": float(clean_numeric_str(match_overall.group(3))),
                    "99th": float(clean_numeric_str(match_overall.group(4)))
                }
            else:
                for i, line in enumerate(lines):
                    if "Overall FCT:" in line and "Avg" in line and "Mid" in line and "95th" in line and "99th" in line:
                        if i + 1 < len(lines):
                            vals = re.findall(r"\S+", lines[i+1])
                            if len(vals) >= 4:
                                overall_data = {
                                    "Avg": float(clean_numeric_str(vals[0])),
                                    "Mid": float(clean_numeric_str(vals[1])),
                                    "95th": float(clean_numeric_str(vals[2])),
                                    "99th": float(clean_numeric_str(vals[3]))
                                }
            
            # 2. 解析<100KB尺寸（精准匹配，避免混淆）
            small100kb_data = None
            # 正则添加单词边界\b，确保匹配"100KB"而不是包含它的字符串
            small100kb_pattern = r"<\s*100KB\b:\s+Avg:\s*(\S+),\s+Mid:\s*\S+,\s+95th:\s*\S+,\s+99th:\s*(\S+)"
            match_small = re.search(small100kb_pattern, content, re.IGNORECASE)
            
            if match_small:
                small100kb_data = {
                    "Small100KB_Avg": float(clean_numeric_str(match_small.group(1))),
                    "Small100KB_99th": float(clean_numeric_str(match_small.group(2)))
                }
            else:
                # 行遍历精准匹配
                for line in lines:
                    # 确保行以"< 100KB:"开头（或包含完整匹配）
                    if line.startswith("< 100KB:") or re.match(r"^\s*<\s*100KB\s*:", line):
                        avg_match = re.search(r"Avg:\s*(\S+)", line)
                        p99_match = re.search(r"99th:\s*(\S+)", line)
                        if avg_match and p99_match:
                            small100kb_data = {
                                "Small100KB_Avg": float(clean_numeric_str(avg_match.group(1))),
                                "Small100KB_99th": float(clean_numeric_str(p99_match.group(1)))
                            }
                            break  # 找到后立即退出，避免重复匹配
            
            # 3. 解析>1MB尺寸（精准匹配，避免与>1MB混淆）
            large1MB_data = None
            # 正则添加单词边界\b，确保匹配"1MB"而不是"1MB"
            large1MB_pattern = r">\s*1MB\b:\s+Avg:\s*(\S+),\s+Mid:\s*\S+,\s+95th:\s*\S+,\s+99th:\s*(\S+)"
            match_large = re.search(large1MB_pattern, content, re.IGNORECASE)
            
            if match_large:
                large1MB_data = {
                    "Large1MB_Avg": float(clean_numeric_str(match_large.group(1))),
                    "Large1MB_99th": float(clean_numeric_str(match_large.group(2)))
                }
            else:
                # 行遍历精准匹配
                for line in lines:
                    # 确保行以"> 1MB:"开头（或包含完整匹配）
                    if line.startswith("> 1MB:") or re.match(r"^\s*>\s*1MB\s*:", line):
                        avg_match = re.search(r"Avg:\s*(\S+)", line)
                        p99_match = re.search(r"99th:\s*(\S+)", line)
                        if avg_match and p99_match:
                            large1MB_data = {
                                "Large1MB_Avg": float(clean_numeric_str(avg_match.group(1))),
                                "Large1MB_99th": float(clean_numeric_str(p99_match.group(1)))
                            }
                            break  # 找到后立即退出，避免重复匹配
            
            # 验证数据完整性
            if overall_data and small100kb_data and large1MB_data:
                # 打印解析结果（便于调试验证）
                print(f"✅ 解析成功 {file_path.name}:")
                print(f"  <100KB - Avg: {small100kb_data['Small100KB_Avg']:.3f}, 99th: {small100kb_data['Small100KB_99th']:.3f}")
                print(f"  >1MB  - Avg: {large1MB_data['Large1MB_Avg']:.3f}, 99th: {large1MB_data['Large1MB_99th']:.3f}")
                return {**overall_data, **small100kb_data, **large1MB_data}
            
            print(f"⚠️  警告：文件 {file_path.name} 缺少部分数据")
            return None
    except Exception as e:
        print(f"❌ 解析文件 {file_path.name} 失败: {e}")
        return None

# -------------------------- 路径与核心配置 --------------------------
load_values = [0.2, 0.4, 0.6, 0.8]  # 待对比的负载值
methods = ["copter", "acc", "m4"]   # 待对比的方法
base_metrics = ["Avg", "Mid", "95th", "99th"]
size_metrics = ["Small100KB_Avg", "Small100KB_99th", "Large1MB_Avg", "Large1MB_99th"]
all_metrics = base_metrics + size_metrics

base_dir = Path("/home/ame/copter/tools/analysis")  # 数据根目录
output_dir = Path("/home/ame/copter/tools/analysis/normalized_fct_plots/thesis_websearch_0.05t")  # 输出文件夹
output_dir.mkdir(parents=True, exist_ok=True)  # 自动创建文件夹（不存在时）

# -------------------------- 数据收集：加载所有方法-负载-指标数据 --------------------------
all_data = []
print("="*80)
print("📥 开始解析所有FCT文件...")
print("="*80)

for load in load_values:
    load_folder = base_dir / f"thesis_websearch_0.05t_{load}load"
    print(f"\n📂 正在处理负载: {load*100}%")
    for method in methods:
        fct_file = load_folder / f"{method}_thesis_websearch_0.05t_{load}load.fct"
        fct_metrics = parse_overall_fct(fct_file)
        if fct_metrics:
            all_data.append({
                "Load": load,
                "Method": method,
                "Avg": fct_metrics["Avg"],
                "Mid": fct_metrics["Mid"],
                "95th": fct_metrics["95th"],
                "99th": fct_metrics["99th"],
                "Small100KB_Avg": fct_metrics["Small100KB_Avg"],
                "Small100KB_99th": fct_metrics["Small100KB_99th"],
                "Large1MB_Avg": fct_metrics["Large1MB_Avg"],
                "Large1MB_99th": fct_metrics["Large1MB_99th"]
            })

# 转换为DataFrame，便于数据处理
df = pd.DataFrame(all_data)
if df.empty:
    print("\n❌ 错误：未收集到任何有效数据，请检查文件路径和格式")
    exit(1)

# ========================== 打印原始数据（含新增指标） ==========================
print("\n" + "="*120)
print("📊 解析后的原始 FCT 数据（单位：通常为ms）")
print("="*120)
df_sorted = df.sort_values(by=["Method", "Load"]).reset_index(drop=True)
pd.options.display.float_format = '{:.3f}'.format
display_cols = ["Load", "Method", "Avg", "Mid", "95th", "99th", 
                "Small100KB_Avg", "Small100KB_99th", "Large1MB_Avg", "Large1MB_99th"]
print(df_sorted[display_cols].to_string(index=False))
print()

# -------------------------- 归一化处理：含新增尺寸指标 --------------------------
copter_baseline = df[df["Method"] == "copter"].set_index("Load")

for metric in all_metrics:
    df[f"Normalized_{metric}"] = df.apply(
        lambda row: row[metric] / copter_baseline.loc[row["Load"], metric],
        axis=1
    )

# ========================== 打印归一化数据（含新增指标） ==========================
print("="*120)
print("📈 归一化后的数据（以 CoPTER 为基准，值越小越优）")
print("="*120)
normalized_cols = ["Load", "Method"] + [f"Normalized_{m}" for m in all_metrics]
df_normalized = df[normalized_cols].sort_values(by=["Method", "Load"]).reset_index(drop=True)
col_rename = {
    "Normalized_Avg": "Norm_Avg",
    "Normalized_Mid": "Norm_Mid",
    "Normalized_95th": "Norm_95th",
    "Normalized_99th": "Norm_99th",
    "Normalized_Small100KB_Avg": "Norm_Small100KB_Avg",
    "Normalized_Small100KB_99th": "Norm_Small100KB_99th",
    "Normalized_Large1MB_Avg": "Norm_Large1MB_Avg",
    "Normalized_Large1MB_99th": "Norm_Large1MB_99th"
}
df_normalized.rename(columns=col_rename, inplace=True)
print(df_normalized.to_string(index=False))
print()

# -------------------------- 图表配置：指标-标题-文件名映射 --------------------------
metric_config = {
    "Avg": ("Average FCT", "avg_fct",{"loc":"upper left"}),
    "Mid": ("Median FCT", "mid_fct",{"loc":"upper left"}),
    "95th": ("95th Percentile FCT", "95th_fct",{"loc":"upper right","bbox_to_anchor":(1.02,1.05)}),
    "99th": ("99th Percentile FCT", "99th_fct",{"loc":"upper right"}),
    "Small100KB_Avg": ("Average FCT (<100KB)", "small100KB_avg_fct",{"loc":"upper right"}),
    "Small100KB_99th": ("99th Percentile FCT (<100KB)", "small100KB_99th_fct",{"loc":"upper right"}),
    "Large1MB_Avg": ("Average FCT (>1MB)", "large1MB_avg_fct",{"loc":"upper left"}),
    "Large1MB_99th": ("99th Percentile FCT (>1MB)", "large1MB_99th_fct",{"loc":"upper left"})
}

# -------------------------- 批量绘制：含新增尺寸指标图表 --------------------------
print("="*80)
print("🎨 开始生成图表...")
print("="*80)

for metric in all_metrics:
    plt.figure(figsize=(18, 12))
    
    for method in methods:
        method_data = df[df["Method"] == method]
        plt.plot(
            method_data["Load"],
            method_data[f"Normalized_{metric}"],
            marker=markers[method],
            linestyle=line_styles[method],
            color=color_map[method],
            linewidth=6,
            markersize=20,
            label=name_mapping[method]
        )
    
    # 图表细节配置
    title, filename, legend_config = metric_config[metric]
    plt.xlabel("Load(%)", fontsize=60)
    plt.ylabel("Normalized FCT", fontsize=60)
    # plt.title(title, fontsize=14, pad=15)
    plt.grid(True, axis='y', linestyle='', alpha=0.6)
    
    # 图例配置
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
        title=None,
        **legend_config
    )
    
    plt.xticks(load_values, [f"{int(load*100)}" for load in load_values])
    plt.ylim(bottom=0.98)
    
    # 保存PDF文件
    pdf_filename = f"normalized_{metric_config[metric][1]}.pdf"
    pdf_path = output_dir / pdf_filename
    plt.tight_layout()
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"✅ 已保存：{pdf_filename}")

# -------------------------- 输出汇总信息 --------------------------
print(f"\n" + "="*80)
print(f"📁 所有图表保存路径：{output_dir}")
print(f"✅ 共生成 {len(all_metrics)} 个图表文件")
print("="*80)

print(f"\n💡 尺寸分片指标图表：")
for metric in size_metrics:
    print(f"  - normalized_{metric_config[metric][1]}.pdf")