# This Python Script is to analyze FCT in different ECN threshold settings under the same workload

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# 读取 CSV 数据
data = pd.read_csv('result/256hosts_flow_dcqcn_load0.7_overall.fct.csv')  # 替换为你的 CSV 文件路径

# 提取 Kmin 和 Kmax 从文件名
data['Kmin'] = data['File'].str.extract(r'kmin(\d+)').astype(int)
data['Kmax'] = data['File'].str.extract(r'kmax(\d+)').astype(int)
data['Average FCT'] = data['Average FCT'].astype(float)
data['99th FCT'] = data['99th FCT'].astype(float)

# 创建网格
kmin_range = np.linspace(data['Kmin'].min(), data['Kmin'].max(), 100)
kmax_range = np.linspace(data['Kmax'].min(), data['Kmax'].max(), 100)
kmin_grid, kmax_grid = np.meshgrid(kmin_range, kmax_range)

# 插值
fct_grid = griddata(
    (data['Kmin'], data['Kmax']),
    data['Average FCT'],
    (kmin_grid, kmax_grid),
    method='cubic'  # 可选 'linear', 'nearest'
)

fct_95th_grid = griddata(
    (data['Kmin'], data['Kmax']),
    data['99th FCT'],
    (kmin_grid, kmax_grid),
    method='cubic'  # 可选 'linear', 'nearest'
)

# 创建子图，左右并排显示
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

# 绘制 Average FCT 热图
contour1 = ax1.contourf(kmin_grid, kmax_grid, fct_grid, cmap='viridis', levels=20)
fig.colorbar(contour1, ax=ax1, label='Average FCT')
ax1.set_xlabel('Kmin')
ax1.set_ylabel('Kmax')
ax1.set_title('Average FCT vs Kmin and Kmax')
ax1.grid(True)

# 绘制 99th FCT 热图
contour2 = ax2.contourf(kmin_grid, kmax_grid, fct_95th_grid, cmap='viridis', levels=20)
fig.colorbar(contour2, ax=ax2, label='99th FCT')
ax2.set_xlabel('Kmin')
# ax2.set_ylabel('Kmax')  # 共享 Y 轴，无需重复标签
ax2.set_title('99th FCT vs Kmin and Kmax')
ax2.grid(True)

# 调整布局
plt.tight_layout()

# # 保存图像（可选）
# plt.savefig('heatmap_fct_comparison.png')

# 显示图像
plt.show()