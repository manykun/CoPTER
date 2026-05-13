import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# -------------------------- 全局配置（新增刻度+外框设置） --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.unicode_minus': False,
    'axes.linewidth': 0.8,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    'text.usetex': False,
    'xtick.major.pad': 4,
    'ytick.major.pad': 4,
    'axes.labelpad': 6,
    # 新增：刻度属性配置
    'xtick.direction': 'in',  # 刻度向内
    'ytick.direction': 'in',
    'xtick.major.size': 3,  # 主刻度长度
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,  # 次刻度长度
    'ytick.minor.size': 1.5,
    'xtick.major.width': 0.8,  # 刻度线宽度
    'ytick.major.width': 0.8,
    # 新增：外框默认黑色（后续每个ax单独强化）
    'axes.edgecolor': 'black',
})

sns.set_style("whitegrid")
sns.set_palette("colorblind")

# -------------------------- 文件夹/数据路径/样式配置（保持不变） --------------------------
custom_folder_name = "thesis_websearch_0.05t_0.8load"
main_output_dir = Path("/home/ame/copter/tools/analysis/fct_analysis_plots")
output_dir = main_output_dir / custom_folder_name
output_dir.mkdir(parents=True, exist_ok=True)
print(f"图表将保存到: {output_dir}")

file_paths = {
    "copter": "/home/ame/copter/tools/analysis/thesis_websearch_0.05t_0.8load/copter_thesis_websearch_0.05t_0.8load.fct",
    "m3": "/home/ame/copter/tools/analysis/thesis_websearch_0.05t_0.8load/m4_thesis_websearch_0.05t_0.8load.fct",
    "acc": "/home/ame/copter/tools/analysis/thesis_websearch_0.05t_0.8load/acc_thesis_websearch_0.05t_0.8load.fct",
}

name_mapping = {
    "copter": "CoPTER",
    "m3": "m3",
    "m4": "m4",
    "acc": "ACC",
    "dcqcn": r"$SECN_1$",
    "hpcc": r"$SECN_2$"
}

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

# -------------------------- 解析数据（保持不变） --------------------------
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

# -------------------------- 工具函数：统一设置图表样式（新增） --------------------------
def set_plot_style(ax):
    # 1. 设置外框为黑色（强化四个边框）
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.8)  # 外框线条宽度与轴一致
    
    # 2. 强化刻度属性（覆盖全局配置，确保生效）
    ax.tick_params(
        direction='in',  # 刻度向内
        which='major',  # 主刻度
        length=3, width=0.8, pad=4,
        labelsize=8, labelcolor='black'
    )
    ax.tick_params(
        direction='in',  # 次刻度也向内
        which='minor',
        length=1.5, width=0.8
    )
    
    # 3. 网格样式优化（不遮挡数据）
    ax.grid(True, axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.grid(False, axis='x')  # 隐藏x轴网格，更整洁

# -------------------------- 1. 整体性能对比（核心优化） --------------------------
def plot_overall_comparison(results):
    metrics_png = ["Avg", "Mid", "95th", "99th"]
    data_png = []
    for name, res in results.items():
        for metric in metrics_png:
            data_png.append({
                "Scheme": name,
                "Metric": metric,
                "Value": res["overall"][metric],
                "Relative Value": res["overall"][metric] / baseline["overall"][metric]
            })
    df_png = pd.DataFrame(data_png)
    
    # 原有PNG绘制（应用统一样式）
    fig_png, axes_png = plt.subplots(1, 2, figsize=(16, 6))
    # 左图
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
    handles, labels = axes_png[0].get_legend_handles_labels()
    new_labels = [name_mapping[label] for label in labels]
    # 图例内嵌：右上角，不遮挡柱状图
    axes_png[0].legend(handles=handles, labels=new_labels,
                       title="Scheme", title_fontsize=12, loc='upper right',
                       frameon=True, framealpha=0.9,  # 带半透明背景
                       bbox_to_anchor=(0.98, 0.98), handlelength=1.5, labelspacing=0.3)
    axes_png[0].set_title("Overall FCT Performance Comparison", fontsize=16, pad=20)
    axes_png[0].set_ylabel("FCT slowdown", fontsize=14)
    axes_png[0].set_xlabel("Performance Metric", fontsize=14)
    set_plot_style(axes_png[0])  # 应用统一样式
    
    # 右图
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
    handles, labels = axes_png[1].get_legend_handles_labels()
    new_labels = [name_mapping[label] if label != 'copter baseline' else label for label in labels]
    axes_png[1].legend(handles=handles, labels=new_labels,
                       title="Scheme", title_fontsize=12, loc='upper right',
                       frameon=True, framealpha=0.9,
                       bbox_to_anchor=(0.98, 0.98), handlelength=1.5, labelspacing=0.3)
    axes_png[1].set_title("Overall FCT Relative to copter", fontsize=16, pad=20)
    axes_png[1].set_ylabel("Relative Value", fontsize=14)
    axes_png[1].set_xlabel("Performance Metric", fontsize=14)
    set_plot_style(axes_png[1])  # 应用统一样式
    
    plt.tight_layout()
    plt.savefig(output_dir / "overall_fct_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: overall_fct_comparison.png")
    plt.show()
    plt.close(fig_png)
    
    # 双栏PDF：整体FCT（优化图例+样式）
    metrics_pdf = ["Avg","95th", "99th"]
    data_pdf = df_png[df_png["Metric"].isin(metrics_pdf)].copy()
    data_pdf["Metric"] = data_pdf["Metric"].replace({"Avg":"Avg","95th": "p95", "99th": "p99"})
    
    fig_pdf, ax_pdf = plt.subplots(1, 1, figsize=(3.3, 2.8))
    sns.barplot(x="Metric", y="Value", hue="Scheme", data=data_pdf, 
                palette=color_map, ax=ax_pdf, edgecolor='black', linewidth=0.8)
    
    for i, bar in enumerate(ax_pdf.containers):
        scheme_name = list(results.keys())[i]
        for patch in bar.patches:
            patch.set_facecolor('none')
            patch.set_edgecolor(color_map[scheme_name])
            patch.set_linewidth(1.0)
            patch.set_hatch(hatches[scheme_name])
            patch.set_alpha(1.0)
    
    # 图例内嵌：左上角，紧贴边框
    handles, labels = ax_pdf.get_legend_handles_labels()
    new_labels = [name_mapping[label] for label in labels]
    ax_pdf.legend(handles=handles, labels=new_labels,
                  title="", title_fontsize=8, loc='upper left',
                  frameon=True, framealpha=0.9,  # 半透明背景防遮挡
                  bbox_to_anchor=(0.02, 0.98), handlelength=1.2, labelspacing=0.2,
                  bordercolor='black', borderwidth=0.5)  # 图例边框黑色
    
    ax_pdf.set_title("")
    ax_pdf.set_ylabel("FCT slowdown", fontsize=10)
    ax_pdf.set_xlabel("")
    set_plot_style(ax_pdf)  # 应用统一样式
    
    plt.tight_layout(pad=0.3)
    plt.savefig(output_dir / "overall_fct_95_99.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_pdf)
    print("PDF saved: overall_fct_95_99.pdf")
    
    # 归一化PDF（同风格优化）
    data_relative_pdf = data_pdf.copy()
    for scheme in results.keys():
        scheme_mask = data_relative_pdf["Scheme"] == scheme
        for metric in ["Avg", "p95", "p99"]:
            original_metric = "95th" if metric == "p95" else "99th" if metric == "p99" else "Avg"
            baseline_val = baseline["overall"][original_metric]
            metric_mask = data_relative_pdf["Metric"] == metric
            data_relative_pdf.loc[scheme_mask & metric_mask, "Value"] = (
                results[scheme]["overall"][original_metric] / baseline_val
            )
    
    fig_relative_pdf, ax_relative_pdf = plt.subplots(1, 1, figsize=(3.3, 2.8))
    sns.barplot(x="Metric", y="Value", hue="Scheme", data=data_relative_pdf,
                palette=color_map, ax_relative_pdf, edgecolor='black', linewidth=0.8)
    
    for i, bar in enumerate(ax_relative_pdf.containers):
        scheme_name = list(results.keys())[i]
        for patch in bar.patches:
            patch.set_facecolor('none')
            patch.set_edgecolor(color_map[scheme_name])
            patch.set_linewidth(1.0)
            patch.set_hatch(hatches[scheme_name])
            patch.set_alpha(1.0)
    
    # 图例内嵌
    handles, labels = ax_relative_pdf.get_legend_handles_labels()
    new_labels = [name_mapping[label] for label in labels]
    ax_relative_pdf.legend(handles=handles, labels=new_labels,
                           title="", title_fontsize=8, loc='upper left',
                           frameon=True, framealpha=0.9,
                           bbox_to_anchor=(0.02, 0.98), handlelength=1.2, labelspacing=0.2,
                           bordercolor='black', borderwidth=0.5)
    
    ax_relative_pdf.set_title("")
    ax_relative_pdf.set_ylabel("Normalized FCT", fontsize=10)
    ax_relative_pdf.set_xlabel("")
    ax_relative_pdf.set_ylim(0.8, 1.2)
    set_plot_style(ax_relative_pdf)  # 应用统一样式
    
    plt.tight_layout(pad=0.3)
    plt.savefig(output_dir / "overall_fct_normalized_95_99.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig_relative_pdf)
    print("PDF saved: overall_fct_normalized_95_99.pdf")

# -------------------------- 2. 按百分比分组对比（优化样式） --------------------------
def plot_percent_comparison(results):
    metrics = ["Avg", "95th", "99th"]
    percent_values = results["copter"]["percent_data"]["Percent"].values
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 18))
    for i, metric in enumerate(metrics):
        for name, res in results.items():
            if len(res["percent_data"]) == len(percent_values):
                axes[i].plot(percent_values, res["percent_data"][metric], 
                             label=name_mapping[name],
                             color=color_map[name], 
                             linestyle=line_styles[name], 
                             marker=markers[name], markersize=4)  # 缩小标记点
        # 图例内嵌：右上角
        axes[i].legend(title="Scheme", title_fontsize=12, loc='upper right',
                       frameon=True, framealpha=0.9,
                       bbox_to_anchor=(0.98, 0.95), handlelength=1.5, labelspacing=0.3)
        axes[i].set_title(f"{metric} FCT vs. Flow Percentile", fontsize=16, pad=15)
        axes[i].set_xlabel("Flow Percentile (%)", fontsize=14)
        axes[i].set_ylabel(f"{metric} FCT slowdown", fontsize=14)
        set_plot_style(axes[i])  # 应用统一样式
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
                             marker=markers[name], markersize=4)
        axes[i].axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5, label='copter baseline')
        # 图例内嵌：右上角
        axes[i].legend(title="Scheme", title_fontsize=12, loc='upper right',
                       frameon=True, framealpha=0.9,
                       bbox_to_anchor=(0.98, 0.95), handlelength=1.5, labelspacing=0.3)
        axes[i].set_title(f"Relative {metric} FCT vs. Flow Percentile", fontsize=16, pad=15)
        axes[i].set_xlabel("Flow Percentile (%)", fontsize=14)
        axes[i].set_ylabel(f"Relative {metric} FCT", fontsize=14)
        set_plot_style(axes[i])  # 应用统一样式
        y_min, y_max = axes[i].get_ylim()
        axes[i].set_ylim(max(0, y_min * 0.95), y_max * 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / "percent_fct_relative_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: percent_fct_relative_comparison.png")
    plt.show()

# -------------------------- 3. 按流大小分组对比（核心优化） --------------------------
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
                    "Scheme": name,
                    "Metric": metric,
                    "Value": res["size_groups"][group][metric],
                    "Relative Value": relative_val
                })
    df_png = pd.DataFrame(data_png)
    if df_png.empty:
        print("Error: Insufficient size group data for plotting")
        return
    
    # 原有PNG绘制（应用样式）
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
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [name_mapping[label] for label in labels]
        # 图例内嵌
        ax.legend(handles=handles, labels=new_labels, title="Scheme", loc='upper right',
                  frameon=True, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
        set_plot_style(ax)  # 应用统一样式
    g_png.set_titles("{col_name} Flows", fontsize=14)
    g_png.set_axis_labels("Performance Metric", "FCT slowdown")
    plt.tight_layout()
    plt.savefig(output_dir / "size_group_fct_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: size_group_fct_comparison.png")
    
    # 相对值PNG（应用样式）
    rel_df_png = df_png[df_png["Scheme"] != "copter"]
    g_png_rel = sns.catplot(x="Metric", y="Relative Value", hue="Scheme", col="Flow Size Group",
                            data=rel_df_png, kind="bar", palette=color_map, col_wrap=2, height=5, edgecolor='black')
    for i, ax in enumerate(g_png_rel.axes):
        for j, bar in enumerate(ax.containers):
            scheme_name = list(results.keys())[j+1]
            for patch in bar.patches:
                patch.set_facecolor('none')
                patch.set_edgecolor(color_map[scheme_name])
                patch.set_linewidth(2)
                patch.set_hatch(hatches[scheme_name])
                patch.set_alpha(1.0)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1.5)
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [name_mapping[label] for label in labels]
        ax.legend(handles=handles, labels=new_labels, title="Scheme", loc='upper right',
                  frameon=True, framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
        set_plot_style(ax)  # 应用统一样式
    g_png_rel.set_titles("{col_name} Flows", fontsize=14)
    g_png_rel.set_axis_labels("Performance Metric", "Relative FCT")
    plt.tight_layout()
    plt.savefig(output_dir / "size_group_fct_relative_comparison.png", dpi=300, bbox_inches="tight")
    print("PNG saved: size_group_fct_relative_comparison.png")
    plt.show()
    plt.close(g_png.fig)
    plt.close(g_png_rel.fig)
    
    # 双栏PDF（核心优化）
    metrics_pdf = ["95th", "99th"]
    data_pdf = df_png[df_png["Metric"].isin(metrics_pdf)].copy()
    data_pdf["Metric"] = data_pdf["Metric"].replace({"95th": "p95", "99th": "p99"})
    
    g_pdf = sns.catplot(x="Metric", y="Value", hue="Scheme", col="Flow Size Group",
                        data=data_pdf, kind="bar", palette=color_map, col_wrap=2,
                        height=2.2, aspect=0.9, edgecolor='black', linewidth=0.8)
    
    for i, ax in enumerate(g_pdf.axes):
        for j, bar in enumerate(ax.containers):
            scheme_name = list(results.keys())[j]
            for patch in bar.patches:
                patch.set_facecolor('none')
                patch.set_edgecolor(color_map[scheme_name])
                patch.set_linewidth(1.0)
                patch.set_hatch(hatches[scheme_name])
                patch.set_alpha(1.0)
        
        # 图例内嵌：左上角，带黑色边框
        handles, labels = ax.get_legend_handles_labels()
        new_labels = [name_mapping[label] for label in labels]
        ax.legend(handles=handles, labels=new_labels, title="", loc='upper left',
                  frameon=True, framealpha=0.9, bordercolor='black', borderwidth=0.5,
                  bbox_to_anchor=(0.02, 0.98), handlelength=1.0, labelspacing=0.1, fontsize=7)
        
        set_plot_style(ax)  # 应用统一样式
        ax.set_xlabel("", fontsize=8)
        ax.set_ylabel("FCT slowdown", fontsize=8)
        ax.tick_params(labelsize=7)
    
    g_pdf.legend.remove()
    g_pdf.set_titles("{col_name}", fontsize=9)
    g_pdf.set_axis_labels("", "FCT slowdown")
    
    plt.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.6)
    plt.savefig(output_dir / "size_group_fct_95_99.pdf", dpi=300, bbox_inches="tight")
    plt.close(g_pdf.fig)
    print("PDF saved: size_group_fct_95_99.pdf")

# -------------------------- 执行所有可视化（保持不变） --------------------------
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
    print("优化特性：图例内嵌（带半透明背景防遮挡）、黑色外框、刻度向内、网格整洁")