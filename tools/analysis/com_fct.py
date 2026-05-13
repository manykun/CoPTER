import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# 设置全局字体和样式（适配英文论文，关闭LaTeX依赖）
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],  # 论文标准字体
    'font.size': 60,
    'axes.labelsize': 60,  # 双栏图字体适当缩小，避免拥挤
    'axes.titlesize': 60,
    'legend.fontsize': 50,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,
    'axes.unicode_minus': False,
    'axes.linewidth': 1.0, 
    'grid.linestyle': '',
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    # 关键修改：关闭LaTeX渲染，改用matplotlib内置数学表达式解析
    'text.usetex': False,
    # 启用matplotlib内置的下标解析（默认已启用，无需额外配置）
    'mathtext.fontset': 'custom',  # 自定义数学文本字体
    'mathtext.rm': 'Times New Roman',  # 数学文本中的罗马字体（普通文本）
    'mathtext.it': 'Times New Roman:italic',  # 数学文本中的斜体
    'mathtext.bf': 'Times New Roman:bold',  # 数学文本中的粗体
})

sns.set_style("white")
sns.set_palette("colorblind")

# -------------------------- 文件夹配置（保持不变） --------------------------
custom_folder_name = "thesis_mix_webserver_websearch_random"
main_output_dir = Path("/home/ame/copter/tools/analysis/fct_analysis_plots")
output_dir = main_output_dir / custom_folder_name
output_dir.mkdir(parents=True, exist_ok=True)
print(f"图表将保存到: {output_dir}")

# -------------------------- 数据路径配置（保持不变，键名仍为原名称以匹配文件） --------------------------
file_paths = {
    "copter": "/home/ame/copter/tools/analysis/thesis_mix_webserver_websearch_random/copter_thesis_mix_webserver_websearch_random.fct",
    "m3": "/home/ame/copter/tools/analysis/thesis_mix_webserver_websearch_random/m4_thesis_mix_webserver_websearch_random.fct",
    # "m4": "/home/ame/copter/tools/analysis/thesis_mix_webserver_websearch_random/m4_thesis_mix_webserver_websearch_random.fct",
    "acc": "/home/ame/copter/tools/analysis/thesis_mix_webserver_websearch_random/acc_thesis_mix_webserver_websearch_random.fct",
    "dcqcn": "/home/ame/copter/tools/analysis/thesis_mix_webserver_websearch_random/dcqcn_thesis_mix_webserver_websearch_random.fct",
    "hpcc": "/home/ame/copter/tools/analysis/thesis_mix_webserver_websearch_random/hpcc_thesis_mix_webserver_websearch_random.fct"
}

# -------------------------- 名称映射（原名称→带下标显示名称，语法不变） --------------------------
name_mapping = {
    "copter": "CoPT",
    "m3": "SCoPE",
    "m4": "m4",
    "acc": "ACC",
    "dcqcn": r"$SECN_1$",  # 内置解析器支持$包裹+_表示下标
    "hpcc": r"$SECN_2$"
}

# 样式配置（键名=原名称，用于匹配数据）
color_map = {
    "copter": "#FF6B00",
    "m4": "#E60023",
    "m3": "#0066FF",
    "acc": "#00CC66",
    "dcqcn": "#9933FF",
    "hpcc": "#FFCC00"
}
line_styles = {
    "copter":'-', "m4": '-', "m3": '-', "acc": '-', "dcqcn": '-', "hpcc": '-'
}
markers = {
    "copter": 'x', "m4": 'o', "m3": 's', "acc": '^', "dcqcn": 'D', "hpcc": 'p'
}
hatches = {
    "copter": '||', "m4": 'xx', "m3": '+', "acc": '\\', "dcqcn": 'x', "hpcc": '++'
}

# -------------------------- 数据解析函数（保持不变） --------------------------
def parse_fct_file(file_path, debug=False):
    data = {"overall": None, "percent_data": None, "size_groups": {}}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        # 解析整体FCT
        overall_pattern = r"Overall FCT:\s+Avg\s+Mid\s+95th\s+99th\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)"
        overall_match = re.search(overall_pattern, content, re.IGNORECASE)
        if overall_match:
            overall_vals = overall_match.groups()
        else:
            overall_lines = [line for line in lines if "Overall FCT:" in line]
            overall_nums = []
            for line in overall_lines + [lines[i+1] if i+1 < len(lines) else '' for i, l in enumerate(lines) if "Overall FCT:" in l]:
                overall_nums.extend(re.findall(r"\d+\.?\d*", line))
            overall_vals = overall_nums[:4]
        
        data["overall"] = {
            "Avg": float(overall_vals[0]), "Mid": float(overall_vals[1]),
            "95th": float(overall_vals[2]), "99th": float(overall_vals[3])
        }
        
        # 解析百分比分组数据
        percent_header = None
        for i, line in enumerate(lines):
            if "Percent" in line and "Size" in line and ("Avg" in line or "Average" in line):
                percent_header = i
                break
        if percent_header is None:
            raise ValueError("No percent data header found")
        
        size_group_markers = ["< 100KB", "> 10MB", "< 10KB", "> 1MB"]
        percent_end = None
        for i in range(percent_header + 1, len(lines)):
            if any(marker in lines[i] for marker in size_group_markers):
                percent_end = i
                break
        percent_end = percent_end if percent_end else len(lines)
        
        percent_data = []
        for line in lines[percent_header+1:percent_end]:
            if not line:
                continue
            vals = re.findall(r"\d+\.?\d*", line)
            if len(vals) < 6:
                continue
            percent_data.append({
                "Percent": float(vals[0]), "Size": float(vals[1]),
                "Avg": float(vals[2]), "Mid": float(vals[3]),
                "95th": float(vals[4]), "99th": float(vals[5])
            })
        data["percent_data"] = pd.DataFrame(percent_data)
        
        # 解析按大小分组数据
        size_group_lines = [line for line in lines if any(s in line for s in size_group_markers)]
        for line in size_group_lines:
            group_name = None
            for marker in size_group_markers:
                if marker in line:
                    group_name = marker
                    break
            if not group_name:
                continue
            vals = re.findall(r"\d+\.\d+", line)
            if len(vals) >= 4:
                data["size_groups"][group_name] = {
                    "Avg": float(vals[0]), "Mid": float(vals[1]),
                    "95th": float(vals[2]), "99th": float(vals[3])
                }
        
        return data
    except Exception as e:
        print(f"Error parsing file: {str(e)}")
        raise

# -------------------------- 解析数据（保持不变，日志显示映射后的名称） --------------------------
results = {}
for name, path in file_paths.items():
    try:
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_file():
            print(f"Error: File {path} does not exist or is not a file")
            continue
        debug_mode = (name == list(file_paths.keys())[0])
        results[name] = parse_fct_file(path, debug=debug_mode)
        print(f"Successfully parsed {name_mapping[name]} data")
    except Exception as e:
        print(f"Error parsing {name_mapping[name]}: {e}")

# 设置基准（保持不变）
if "copter" not in results:
    print("Error: Failed to parse copter data as baseline, exiting")
    exit(1)
baseline = results["copter"]

# -------------------------- 1. 整体性能对比（原有PNG + 优化PDF + 新增归一化PDF） --------------------------
def plot_overall_comparison(results):
    metrics_png = ["Avg", "Mid", "95th", "99th"]
    data_png = []
    for name, res in results.items():
        for metric in metrics_png:
            data_png.append({
                "Scheme": name,  # 保留原名称用于匹配样式
                "Metric": metric,
                "Value": res["overall"][metric],
                "Relative Value": res["overall"][metric] / baseline["overall"][metric]
            })
    df_png = pd.DataFrame(data_png)
    
    fig_png, axes_png = plt.subplots(1, 2, figsize=(18,12))
    # 绝对数值PNG
    sns.barplot(x="Metric", y="Value", hue="Scheme", data=df_png, 
                palette=color_map, ax=axes_png[0], edgecolor='black')
    for i, bar in enumerate(axes_png[0].containers):
        scheme_name = list(results.keys())[i]
        for patch in bar.patches:
            patch.set_facecolor('none')
            patch.set_edgecolor(color_map[scheme_name])
            patch.set_linewidth(2)
            patch.set_hatch(hatches[scheme_name])
            patch.set_alpha(1.0)
    # 替换图例为带下标名称
    handles, labels = axes_png[0].get_legend_handles_labels()
    new_labels = [name_mapping[label] for label in labels]
    axes_png[0].legend(handles=handles, labels=new_labels, 
                       title="Scheme", title_fontsize=60, loc='upper left', frameon=False)
    axes_png[0].set_title("Overall FCT Performance Comparison", fontsize=60, pad=20)
    axes_png[0].set_ylabel("FCT slowdown", fontsize=60)
    axes_png[0].set_xlabel("Performance Metric", fontsize=60)
    axes_png[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 相对数值PNG
    sns.barplot(x="Metric", y="Relative Value", hue="Scheme", data=df_png, 
                palette=color_map, ax=axes_png[1], edgecolor='black')
    for i, bar in enumerate(axes_png[1].containers):
        scheme_name = list(results.keys())[i]
        for patch in bar.patches:
            patch.set_facecolor('none')
            patch.set_edgecolor(color_map[scheme_name])
            patch.set_linewidth(2)
            patch.set_hatch(hatches[scheme_name])
            patch.set_alpha(1.0)
    axes_png[1].axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='copter baseline')
    # 替换图例
    handles, labels = axes_png[1].get_legend_handles_labels()
    new_labels = [name_mapping[label] if label != 'copter baseline' else label for label in labels]
    axes_png[1].legend(handles=handles, labels=new_labels, 
                       title="Scheme", title_fontsize=60, loc='upper left', frameon=False)
    axes_png[1].set_title("Overall FCT Relative to copter", fontsize=60, pad=20)
    axes_png[1].set_ylabel("Relative Value", fontsize=60)
    axes_png[1].set_xlabel("Performance Metric", fontsize=60)
    axes_png[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存PNG
    plt.tight_layout()
    plt.savefig(output_dir / "overall_fct_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: overall_fct_comparison.png")
    plt.show()
    plt.close(fig_png)
    
    # 优化PDF：仅95th/99th指标（原有）
    metrics_pdf = ["Avg","95th", "99th"]
    # metrics_pdf = ["95th", "99th"]
    data_pdf = df_png[df_png["Metric"].isin(metrics_pdf)].copy()
    data_pdf["Metric"] = data_pdf["Metric"].replace({"Avg":"Avg","95th": "p95", "99th": "p99"})
    # data_pdf["Metric"] = data_pdf["Metric"].replace({"95th": "p95", "99th": "p99"})
    
    fig_pdf, ax_pdf = plt.subplots(1, 1, figsize=(18, 12))
    sns.barplot(x="Metric", y="Value", hue="Scheme", data=data_pdf, 
                palette=color_map, ax=ax_pdf, edgecolor='black')
    
    # 样式设置
    for i, bar in enumerate(ax_pdf.containers):
        scheme_name = list(results.keys())[i]
        for patch in bar.patches:
            patch.set_facecolor('none')
            patch.set_edgecolor(color_map[scheme_name])
            patch.set_linewidth(6)
            patch.set_hatch(hatches[scheme_name])
            patch.set_alpha(1.0)
    
    # 替换PDF图例
    handles, labels = ax_pdf.get_legend_handles_labels()
    new_labels = [name_mapping[label] for label in labels]
    legend_pdf = ax_pdf.legend(handles=handles, labels=new_labels,
                               title="", title_fontsize=60, loc='upper left', frameon=False,bbox_to_anchor=(0.01, 1.06),labelspacing=0.1)
    for text in legend_pdf.get_texts():
        text.set_fontname('Times New Roman')
    
    ax_pdf.set_title("")
    ax_pdf.set_ylabel("FCT slowdown", fontsize=60, fontname='Times New Roman')
    ax_pdf.set_xlabel("", fontname='Times New Roman')
    for label in ax_pdf.get_xticklabels():
        label.set_fontname('Times New Roman')
    for label in ax_pdf.get_yticklabels():
        label.set_fontname('Times New Roman')

    # 加粗坐标轴线条
    ax_pdf.spines['top'].set_linewidth(2.0)
    ax_pdf.spines['top'].set_color('black')  # 新增：外框线黑色
    ax_pdf.spines['right'].set_linewidth(2.0)
    ax_pdf.spines['right'].set_color('black')  # 新增：外框线黑色
    ax_pdf.spines['bottom'].set_linewidth(2.0)
    ax_pdf.spines['bottom'].set_color('black')  # 新增：外框线黑色
    ax_pdf.spines['left'].set_linewidth(2.0)
    ax_pdf.spines['left'].set_color('black')  # 新增：外框线黑色
    ax_pdf.grid(axis='y', linestyle='', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_fct_95_99.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_pdf)
    print("PDF saved: overall_fct_95_99.pdf")
    
    # -------------------------- 新增：归一化后的PDF（以copter为基准） --------------------------
    data_relative_pdf = data_pdf.copy()
    # 计算每个scheme相对于copter的归一化值
    for scheme in results.keys():
        scheme_mask = data_relative_pdf["Scheme"] == scheme
        # 获取copter对应指标的基准值
        for metric in ["Avg", "p95", "p99"]:
            original_metric = "95th" if metric == "p95" else "99th" if metric == "p99" else "Avg"
            baseline_val = baseline["overall"][original_metric]
            metric_mask = data_relative_pdf["Metric"] == metric
            data_relative_pdf.loc[scheme_mask & metric_mask, "Value"] = (
                results[scheme]["overall"][original_metric] / baseline_val
            )
    
    # 创建归一化PDF图表
    fig_relative_pdf, ax_relative_pdf = plt.subplots(1, 1, figsize=(18,12))
    sns.barplot(x="Metric", y="Value", hue="Scheme", data=data_relative_pdf,
                palette=color_map, ax=ax_relative_pdf, edgecolor='black')
    
    # 保持相同样式设置
    for i, bar in enumerate(ax_relative_pdf.containers):
        scheme_name = list(results.keys())[i]
        for patch in bar.patches:
            patch.set_facecolor('none')
            patch.set_edgecolor(color_map[scheme_name])
            patch.set_linewidth(6)
            patch.set_hatch(hatches[scheme_name])
            patch.set_alpha(1.0)
    
    # 添加基准线（y=1.0）
    # ax_relative_pdf.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)
    
    # 替换图例（与原有PDF一致）
    handles, labels = ax_relative_pdf.get_legend_handles_labels()
    new_labels = [name_mapping[label] for label in labels]
    legend_relative = ax_relative_pdf.legend(
        handles=handles, labels=new_labels,
        title="",  # 无图例标题（与优化PDF一致）
        title_fontsize=60,  # 按尺寸比例适配（优化PDF是60）
        loc='upper center',  # 位置一致
        frameon=False,  # 无边框（与优化PDF一致）
        bbox_to_anchor=(0.5, 1.05),   # ✔ 图例放到图顶部
        ncol=3,         # ✔ 图例一行排列
        columnspacing=0.5,            # ✔ 调整列间距
        # bbox_to_anchor=(0.01, 1),  # 图例上移（与优化PDF一致）
        labelspacing=0.8,  # 图例项间距（与优化PDF一致）
        fontsize=50  # 图例字体大小（按尺寸比例适配）
    )
    # 图例字体：Times New Roman（包括数学文本）
    for text in legend_relative.get_texts():
        text.set_fontname('Times New Roman')
        text.set_math_fontfamily('custom')  # 保障SECN₁/SECN₂字体一致
    # 保持相同的轴标签和样式
    ax_relative_pdf.set_title("")
    ax_relative_pdf.set_ylabel("Normalized FCT", fontsize=60,fontname='Times New Roman')
    ax_relative_pdf.set_xlabel("", fontname='Times New Roman')
        # 轴刻度字体：Times New Roman（与优化PDF一致）
    for label in ax_relative_pdf.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(60)  # 按尺寸比例适配
    for label in ax_relative_pdf.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(60)  # 按尺寸比例适配
    
    # 坐标轴线条：与优化PDF完全一致（黑色+粗细2.0）
    ax_relative_pdf.spines['top'].set_linewidth(2.0)
    ax_relative_pdf.spines['top'].set_color('black')
    ax_relative_pdf.spines['right'].set_linewidth(2.0)
    ax_relative_pdf.spines['right'].set_color('black')
    ax_relative_pdf.spines['bottom'].set_linewidth(2.0)
    ax_relative_pdf.spines['bottom'].set_color('black')
    ax_relative_pdf.spines['left'].set_linewidth(2.0)
    ax_relative_pdf.spines['left'].set_color('black')
    ax_relative_pdf.grid(axis='y', linestyle='', alpha=0.7)
    
    # 调整y轴范围，使图表更美观
    ax_relative_pdf.set_ylim(0.8, 1.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_fct_normalized_95_99.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_relative_pdf)
    print("PDF saved: overall_fct_normalized_95_99.pdf")

# -------------------------- 2. 按百分比分组对比（原有PNG，无PDF需求） --------------------------
def plot_percent_comparison(results):
    metrics = ["Avg", "95th", "99th"]
    percent_values = results["copter"]["percent_data"]["Percent"].values
    
    # -------------------------- 原有PNG生成（保持不变） --------------------------
    # 绝对数值PNG
    fig, axes = plt.subplots(len(metrics), 1, figsize=(18,12))
    for i, metric in enumerate(metrics):
        for name, res in results.items():
            if len(res["percent_data"]) == len(percent_values):
                axes[i].plot(percent_values, res["percent_data"][metric], 
                             label=name_mapping[name],  # 直接用带下标名称
                             color=color_map[name], 
                             linestyle=line_styles[name], 
                             marker=markers[name], markersize=6)
        axes[i].set_title(f"{metric} FCT vs. Flow Percentile", fontsize=16, pad=15)
        axes[i].set_xlabel("Flow Percentile (%)", fontsize=14)
        axes[i].set_ylabel(f"{metric} FCT slowdown", fontsize=14)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend(title="Scheme", title_fontsize=12, loc='upper left', frameon=False)
        y_min, y_max = axes[i].get_ylim()
        axes[i].set_ylim(y_min, y_max * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "percent_fct_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: percent_fct_comparison.png")
    
    # 相对值PNG
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 18))
    for i, metric in enumerate(metrics):
        for name, res in results.items():
            if name == "copter":
                continue
            if len(res["percent_data"]) == len(baseline["percent_data"]):
                relative_vals = res["percent_data"][metric] / baseline["percent_data"][metric]
                axes[i].plot(percent_values, relative_vals, 
                             label=name_mapping[name], 
                             color=color_map[name], 
                             linestyle=line_styles[name], 
                             marker=markers[name], markersize=6)
        axes[i].axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='copter baseline')
        axes[i].set_title(f"Relative {metric} FCT vs. Flow Percentile", fontsize=16, pad=15)
        axes[i].set_xlabel("Flow Percentile (%)", fontsize=14)
        axes[i].set_ylabel(f"Relative {metric} FCT", fontsize=14)
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].legend(title="Scheme", title_fontsize=12, loc='upper left', frameon=False)
        y_min, y_max = axes[i].get_ylim()
        axes[i].set_ylim(max(0, y_min * 0.95), y_max * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "percent_fct_relative_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: percent_fct_relative_comparison.png")
    plt.show()
    plt.close('all')
    
    # -------------------------- 新增PDF生成（每个子图单独生成，与归一化PDF风格一致） --------------------------
    print("\nGenerating percentile comparison PDFs...")
    
    # 定义PDF通用样式配置
    pdf_font_size = 60
    legend_font_size = 50
    line_width = 6
    marker_size = 4
    fig_size = (18,12)
    x_tick_count = 10  # x轴目标刻度数量
    y_tick_count = 5  # y轴目标刻度数量
    
    # 1. 绝对数值PDF（每个指标单独生成）
    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        
        # 绘制所有方案的曲线
        for name, res in results.items():
            if len(res["percent_data"]) == len(percent_values):
                ax.plot(percent_values, res["percent_data"][metric], 
                       label=name_mapping[name],
                       color=color_map[name],
                       linestyle=line_styles[name],
                       marker=markers[name],
                       markersize=marker_size,
                       linewidth=line_width)
        
        # 设置样式（与归一化PDF保持一致）
        ax.set_title("")
        ax.set_xlabel("Flow Percentile (%)", fontsize=pdf_font_size, fontname='Times New Roman')
        ax.set_ylabel(f"{metric} FCT slowdown", fontsize=pdf_font_size, fontname='Times New Roman')
        
        # 轴刻度字体
        for label in ax.get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(pdf_font_size)
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(pdf_font_size)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=x_tick_count)) 
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_tick_count))
        
        # 坐标轴线条（黑色粗边框）
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_linewidth(2.0)
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['left'].set_color('black')
        
        # 网格设置（无网格，与归一化PDF一致）
        # ax.grid(axis='y', linestyle='', alpha=0.7)
        
        # 图例设置
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles=handles, labels=labels,
            title="",
            title_fontsize=legend_font_size,
            loc='upper left',
            frameon=False,
            bbox_to_anchor=(0.01, 1.0),
            labelspacing=0.8,
            fontsize=legend_font_size
        )
        # 图例字体设置
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')
            text.set_math_fontfamily('custom')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存PDF
        filename = f"percent_fct_{metric.lower()}_absolute.pdf"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"PDF saved: {filename}")
    
    # 2. 相对数值PDF（每个指标单独生成，以copter为基准）
    for metric in metrics:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)
        
        # 绘制所有对比方案的相对值曲线
        for name, res in results.items():
            if name == "copter":
                continue  # 基准方案不重复绘制
            if len(res["percent_data"]) == len(baseline["percent_data"]):
                relative_vals = res["percent_data"][metric] / baseline["percent_data"][metric]
                ax.plot(percent_values, relative_vals,
                       label=name_mapping[name],
                       color=color_map[name],
                       linestyle=line_styles[name],
                       marker=markers[name],
                       markersize=marker_size,
                       linewidth=line_width)
        
        # 添加基准线（y=1.0）
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2.5, label='CoPT (baseline)')
        
        # 设置样式（与归一化PDF保持一致）
        ax.set_title("")
        ax.set_xlabel("Flow Percentile (%)", fontsize=pdf_font_size, fontname='Times New Roman')
        ax.set_ylabel(f"Relative {metric} FCT", fontsize=pdf_font_size, fontname='Times New Roman')
        
        # 轴刻度字体
        for label in ax.get_xticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(pdf_font_size)
        for label in ax.get_yticklabels():
            label.set_fontname('Times New Roman')
            label.set_fontsize(pdf_font_size)
        
        # 坐标轴线条（黑色粗边框）
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_linewidth(2.0)
        ax.spines['right'].set_color('black')
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_linewidth(2.0)
        ax.spines['left'].set_color('black')
        
        # 网格设置（无网格，与归一化PDF一致）
        # ax.grid(axis='y', linestyle='', alpha=0.7)
        
        # 调整y轴范围（根据数据自动调整，保持美观）
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(max(0.8, y_min * 0.95), y_max * 1.05)
        
        # 图例设置
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles=handles, labels=labels,
            title="",
            title_fontsize=legend_font_size,
            loc='upper left',
            frameon=False,
            bbox_to_anchor=(0.01, 1.0),
            labelspacing=0.8,
            fontsize=legend_font_size
        )
        # 图例字体设置
        for text in legend.get_texts():
            text.set_fontname('Times New Roman')
            text.set_math_fontfamily('custom')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存PDF
        filename = f"percent_fct_{metric.lower()}_relative.pdf"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"PDF saved: {filename}")


# -------------------------- 3. 按流大小分组对比（原有PNG + 优化PDF） --------------------------
def plot_size_group_comparison(results):
    size_groups = ["< 10KB", "< 100KB", "> 1MB", "> 10MB"]
    metrics_png = ["Avg", "Mid", "95th", "99th"]
    
    data_png = []
    for group in size_groups:
        for name, res in results.items():
            if group not in res["size_groups"]:
                print(f"Warning: No data for {group} in {name_mapping[name]}")
                continue
            baseline_group_val = baseline["size_groups"][group]
            for metric in metrics_png:
                if baseline_group_val[metric] == 0:
                    relative_val = np.nan
                    print(f"Warning: Baseline (copter) {metric} value is 0 for {group} flows, skip relative value calculation")
                else:
                    relative_val = res["size_groups"][group][metric] / baseline_group_val[metric]
                
                data_png.append({
                    "Flow Size Group": group,
                    "Scheme": name,  # 保留原名称匹配样式
                    "Metric": metric,
                    "Value": res["size_groups"][group][metric],
                    "Relative Value": relative_val
                })
    df_png = pd.DataFrame(data_png)
    if df_png.empty:
        print("Error: Insufficient size group data for plotting")
        return
    
    # 绝对数值PNG
    g_png = sns.catplot(x="Metric", y="Value", hue="Scheme", col="Flow Size Group",
                        data=df_png, kind="bar", palette=color_map, col_wrap=2, height=5, edgecolor='black')
    for i, ax in enumerate(g_png.axes):
        for j, bar in enumerate(ax.containers):
            scheme_name = list(results.keys())[j]
            for patch in bar.patches:
                patch.set_facecolor('none')
                patch.set_edgecolor(color_map[scheme_name])
                patch.set_linewidth(2)
                patch.set_hatch(hatches[scheme_name])
                patch.set_alpha(1.0)
        # 替换子图图例
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [name_mapping[label] for label in labels]
        ax.legend(handles=handles, labels=new_labels, title="Scheme", loc='upper left', frameon=False)
    g_png.set_titles("{col_name} Flows", fontsize=14)
    g_png.set_axis_labels("Performance Metric", "FCT slowdown")
    # for ax in g_png.axes:
    #     ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "size_group_fct_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: size_group_fct_comparison.png")
    
    # 相对值PNG
    rel_df_png = df_png[df_png["Scheme"] != "copter"]
    g_png_rel = sns.catplot(x="Metric", y="Relative Value", hue="Scheme", col="Flow Size Group",
                            data=rel_df_png, kind="bar", palette=color_map, col_wrap=2, height=5, edgecolor='black')
    for i, ax in enumerate(g_png_rel.axes):
        for j, bar in enumerate(ax.containers):
            scheme_name = list(results.keys())[j+1]  # 跳过copter
            for patch in bar.patches:
                patch.set_facecolor('none')
                patch.set_edgecolor(color_map[scheme_name])
                patch.set_linewidth(2)
                patch.set_hatch(hatches[scheme_name])
                patch.set_alpha(1.0)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)
        # 替换子图图例
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [name_mapping[label] for label in labels]
        ax.legend(handles=handles, labels=new_labels, title="Scheme", loc='upper left', frameon=False)
    g_png_rel.set_titles("{col_name} Flows", fontsize=14)
    g_png_rel.set_axis_labels("Performance Metric", "Relative FCT")
    # for ax in g_png_rel.axes:
    #     ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "size_group_fct_relative_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: size_group_fct_relative_comparison.png")
    plt.show()
    plt.close(g_png.fig)
    plt.close(g_png_rel.fig)
    
    # 优化PDF：仅95th/99th指标
    metrics_pdf = ["95th", "99th"]
    data_pdf = df_png[df_png["Metric"].isin(metrics_pdf)].copy()
    
    g_pdf = sns.catplot(x="Metric", y="Value", hue="Scheme", col="Flow Size Group",
                        data=data_pdf, kind="bar", palette=color_map, col_wrap=2,
                        height=3.5, aspect=0.8, edgecolor='black')
    
    for i, ax in enumerate(g_pdf.axes):
        for j, bar in enumerate(ax.containers):
            scheme_name = list(results.keys())[j]
            for patch in bar.patches:
                patch.set_facecolor('none')
                patch.set_edgecolor(color_map[scheme_name])
                patch.set_linewidth(2)
                patch.set_hatch(hatches[scheme_name])
                patch.set_alpha(1.0)
        # 替换PDF子图图例
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [name_mapping[label] for label in labels]
        ax.legend(handles=handles, labels=new_labels, title="", loc='upper left', frameon=False)
        # ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    g_pdf.legend.remove()  # 移除全局重复图例
    g_pdf.set_titles("")
    g_pdf.set_axis_labels("", "FCT slowdown")
    g_pdf.set_ylabels("FCT slowdown", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "size_group_fct_95_99.pdf", dpi=300, bbox_inches="tight")
    plt.close(g_pdf.fig)
    print("PDF saved: size_group_fct_95_99.pdf")

# -------------------------- 执行所有可视化（原有逻辑不变） --------------------------
if __name__ == "__main__":
    if len(results) < 2:
        print("Error: Insufficient data parsed, need at least 2 schemes for comparison")
        exit(1)
    
    print("Generating overall performance comparison plot...")
    plot_overall_comparison(results)
    
    print("\nGenerating percentile group comparison plots...")
    plot_percent_comparison(results)
    
    print("\nGenerating flow size group comparison plots...")
    plot_size_group_comparison(results)
    
    print(f"\nAll files saved to: {output_dir}")
    print("PNG files (original): overall_fct_comparison.png, percent_fct_comparison.png, size_group_fct_comparison.png, etc.")
    print("PDF files (new, double-column): overall_fct_95_99.pdf, overall_fct_normalized_95_99.pdf, size_group_fct_95_99.pdf")
    print("Scheme name mapping: dcqcn→$SECN_1$, hpcc→$SECN_2$")