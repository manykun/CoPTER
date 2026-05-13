import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Patch

# -------------------------- 1. 全局样式设置 (复刻参考代码) --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 60,
    'axes.labelsize': 60,
    'axes.titlesize': 60,
    'legend.fontsize': 40,  # 稍微调小以适应图例
    'xtick.labelsize': 50,  # 坐标轴标签大小
    'ytick.labelsize': 50,
    'axes.unicode_minus': False,
    'axes.linewidth': 2.0,  # 加粗边框
    'grid.linestyle': '',
    'grid.alpha': 0.6,
    'figure.dpi': 300,
    'text.usetex': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
})

sns.set_style("white")

# -------------------------- 2. 数据准备 --------------------------
# 从提供的文本报告中提取的数据 (单位: Mbps -> 转换为 Gbps)
raw_data = {
    "acc":    {"Avg": 6662.99, "Max": 40004.48},
    "copter": {"Avg": 6641.00, "Max": 40000.96},
    # "m3":     {"Avg": 6710.82, "Max": 40006.40},
    "m4":     {"Avg": 6606.84, "Max": 40008.64}
}

# 构造DataFrame
data_list = []
for scheme, values in raw_data.items():
    data_list.append({
        "Scheme": scheme,
        "Metric": "Average",
        "Value": values["Avg"] / 1000.0  # Mbps -> Gbps
    })
    data_list.append({
        "Scheme": scheme,
        "Metric": "Maximum",
        "Value": values["Max"] / 1000.0  # Mbps -> Gbps
    })

df = pd.DataFrame(data_list)

# -------------------------- 3. 样式映射配置 --------------------------
# 名称映射（用于图例显示）
name_mapping = {
    "copter": "CoPT",
    "m4": "SCoPE",
    "m3": "m3",
    "acc": "ACC"
}

# 颜色映射 (参考代码中的定义)
color_map = {
    "copter": "#FF6B00",
    "m3": "#E60023",
    "m4": "#0066FF",
    "acc": "#00CC66"
}

# 纹理映射
hatches = {
    "copter": 'o', 
    "m4": '.', 
    "m3": '*', 
    "acc": '-'
}

# 排序：确保绘图顺序一致
schemes_order = ["copter","m4","acc"]

# -------------------------- 4. 绘图执行 --------------------------
fig, ax = plt.subplots(figsize=(18, 12))

# 使用 Seaborn 绘制柱状图
# X轴为指标(Avg/Max)，Hue为方案(acc/copter...)
sns.barplot(
    x="Metric", 
    y="Value", 
    hue="Scheme", 
    data=df, 
    hue_order=schemes_order,
    palette=color_map, 
    ax=ax, 
    edgecolor='black',
    width=0.7 # 调整柱子总宽度
)

# -------------------------- 5. 应用自定义样式 (空心柱 + 纹理) --------------------------
# Seaborn返回的patches顺序通常是：所有方案的第1组bar，然后所有方案的第2组bar...
# 或者按hue分组。最安全的方法是直接遍历所有containers。

for i, container in enumerate(ax.containers):
    # container对应一个hue类别（即一种Scheme）
    scheme_name = schemes_order[i]
    color = color_map[scheme_name]
    hatch = hatches[scheme_name]
    
    for patch in container:
        # 设置空心样式：面色透明，边框色为指定颜色
        patch.set_facecolor('none')
        patch.set_edgecolor(color)
        patch.set_linewidth(4.0) # 柱子边框加粗
        patch.set_hatch(hatch)
        patch.set_alpha(1.0)

# -------------------------- 6. 添加自动标注 (Δ < x) --------------------------
# 计算每个Metric组的统计数据以确定标注位置和内容
metrics = ["Average", "Maximum"]
y_max_limit = df["Value"].max() * 1.25 # 预留上方空间

for idx, metric in enumerate(metrics):
    # 获取该Metric下的所有值
    sub_df = df[df["Metric"] == metric]
    values = sub_df["Value"].values
    
    # 计算变化率 (Max-Min)/Mean
    delta = (values.max() - values.min()) / values.mean()
    mean_val = values.mean()
    max_val = values.max()
    
    text_label = r'$\Delta < {:.4f}$'.format(delta)
    
    # 计算标注的X坐标 (该组X轴的中心)
    x_pos = idx 
    
    # 计算标注的Y坐标 (该组最高柱子上方)
    y_pos = max_val + (y_max_limit * 0.02)
    
    # 绘制标注线和文字
    # 绘制一条横跨该组的线 (可选，这里直接写字)
    ax.text(x_pos, y_pos, text_label, 
            ha='center', va='bottom', 
            fontsize=50, color='#CC0000', fontname='Times New Roman')

# -------------------------- 7. 坐标轴与图例调整 --------------------------
# 设置轴标签
ax.set_xlabel("", fontsize=60)
ax.set_ylabel("Throughput (Gbps)", fontsize=60, fontname='Times New Roman')
ax.set_ylim(0, y_max_limit)

# 设置刻度字体
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')

# 加粗坐标轴外框
ax.spines['top'].set_linewidth(3.0)
ax.spines['top'].set_color('black')
ax.spines['right'].set_linewidth(3.0)
ax.spines['right'].set_color('black')
ax.spines['bottom'].set_linewidth(3.0)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(3.0)
ax.spines['left'].set_color('black')

# 自定义图例
# 创建自定义的Legend handles以匹配空心柱样式
legend_handles = []
for scheme in schemes_order:
    patch = Patch(
        facecolor='none',
        edgecolor=color_map[scheme],
        hatch=hatches[scheme],
        linewidth=3.0,
        label=name_mapping[scheme]
    )
    legend_handles.append(patch)

# 放置图例
legend = ax.legend(
    handles=legend_handles,
    loc='upper left',
    # bbox_to_anchor=(0.5, 1.0), # 图例置于顶部中间
    # ncol=4,                    # 一行显示
    frameon=False,             # 无边框
    columnspacing=1.5,
    handletextpad=0.5,
    prop={'family': 'Times New Roman', 'size': 40}
)

# -------------------------- 8. 保存输出 --------------------------
output_file = "throughput_analysis.pdf"
plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Chart saved to: {output_file}")
plt.show()