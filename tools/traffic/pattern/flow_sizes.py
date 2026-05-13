import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator

# --------------------------
# 全局字体配置：Times New Roman
# --------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
    'axes.labelsize': 60,
    'axes.titlesize': 60,
    'xtick.labelsize': 60,
    'ytick.labelsize': 60,
    'legend.fontsize': 60,
    'text.usetex': False,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'Times New Roman',
    'mathtext.it': 'Times New Roman:italic',
    'mathtext.bf': 'Times New Roman:bold',
    'axes.linewidth': 1.2  # 边框粗细
})

def read_flow_data(file_path, label):
    """Read flow size data from TXT file and add dataset label"""
    df = pd.read_csv(file_path, sep='\s+', header=None, names=['x', 'y'])
    df.loc[:, 'kind'] = label
    return df

# --------------------------
# User Configuration（明确每条曲线的样式）
# --------------------------
# 数据集配置：路径、标签、标记、颜色一一对应
datasets = [
    {
        'path': '/home/ame/copter/tools/traffic/pattern/WebSearch.txt',
        'label': 'WebSearch',
        'marker': '^',  # 三角标记
        'color': '#1f77b4'  # 蓝色（论文常用配色）
    },
    {
        'path': '/home/ame/copter/tools/traffic/pattern/WebServer.txt',
        'label': 'WebServer',
        'marker': 's',  # 方形标记
        'color': '#ff7f0e'  # 橙色（论文常用配色）
    },
    {
        'path': '/home/ame/copter/tools/traffic/pattern/cachefollower-all.txt',
        'label': 'CacheFollower',
        'marker': 'o',  # 圆形标记
        'color': '#2ca02c'  # 绿色（论文常用配色）
    }
]

# 标记通用配置（核心修改：实心+无边框）
marker_size = 20  # 标记大小（确保可见）
_linewidth = 10  # 曲线加粗线宽
markevery = 0.05  # 每5%数据点显示一个标记（密集但不拥挤）

# --------------------------
# Data Loading
# --------------------------
data_dict = {}
for cfg in datasets:
    df = read_flow_data(cfg['path'], cfg['label'])
    # 按x排序（确保CDF曲线连续）
    df = df.sort_values('x').reset_index(drop=True)
    data_dict[cfg['label']] = df

# --------------------------
# 手动绘制CDF曲线（实心标记+无边框）
# --------------------------
plt.figure(figsize=(18, 12))
_fontsize = 55

# 循环绘制每条曲线
for cfg in datasets:
    df = data_dict[cfg['label']]
    plt.plot(
        df['x'], df['y'],
        color=cfg['color'],
        linewidth=_linewidth,
        marker=cfg['marker'],  # 强制设置标记
        markersize=marker_size,
        # 核心修改1：标记实心（填充色=曲线颜色）
        markerfacecolor=cfg['color'],
        # 核心修改2：移除黑色边框（边框色=透明/无）
        markeredgecolor='none',
        markevery=markevery,
        label=cfg['label']  # 图例标签
    )

# --------------------------
# 图表样式配置
# --------------------------
# 四面边框（黑色+加粗）
ax = plt.gca()  # 获取当前坐标轴
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.2)

# x轴对数刻度
plt.xscale('log')

# 设置x轴对数刻度密度：每1个数量级显示5个刻度（可调整numticks参数）
ax.xaxis.set_major_locator(LogLocator(numticks=20))  # 主刻度（多显示关键刻度）
ax.xaxis.set_minor_locator(LogLocator(numticks=40, subs='auto'))  # 次刻度（补充细分刻度）
# 显示次刻度标签（默认隐藏，打开后刻度更密集）
ax.minorticks_on()
# 调整刻度标签样式（避免重叠）
plt.tick_params(
    axis='x', 
    which='both',  # 同时设置主刻度和次刻度
    labelsize=_fontsize-5,  # 次刻度标签可略小（避免拥挤）
    pad=15,  # 刻度标签与轴的距离（防止重叠）
    labelbottom=True
)

# 轴标签
plt.xlabel('Flow size (KB)', fontsize=_fontsize)
plt.ylabel('CDF (%)', fontsize=_fontsize)

# 刻度大小
plt.tick_params(labelsize=_fontsize)

# 图例配置（确保标记与曲线一致）
legend = plt.legend(
    loc='upper center',
    frameon=False,
    ncol=1,
    bbox_to_anchor=(0.76, 0.5),
    fontsize=_fontsize-5
)

# 优化图例样式（同步实心+无边框）
for i, handle in enumerate(legend.legend_handles):
    handle.set_linewidth(_linewidth)
    handle.set_markersize(marker_size)
    # 图例标记同步：实心+无边框
    handle.set_markerfacecolor(datasets[i]['color'])
    handle.set_markeredgecolor('none')

# --------------------------
# 保存与显示
# --------------------------
plt.savefig('flow_size_cdf.pdf', bbox_inches='tight', dpi=300)
plt.show()