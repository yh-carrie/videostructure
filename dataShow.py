import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
font_path = '/usr/share/fonts/myfonts/SIMHEI.TTF'
prop = font_manager.FontProperties(fname=font_path)
prop.set_size(22)

# 示例数据配置（请替换为实际数据）
beta_values = [0.1, 0.3, 0.5, 0.7, 0.9]
models = {
    "PA100k": [86.9, 87.5, 88.1, 87.0, 86.6],  # 蓝色线条
    "PETA": [88.3, 89.0, 88.7, 88.2, 87.8],  # 橙色线条
}
# models = {
#     "PA100k": [90.1, 90.3, 90.9, 90.0, 89.6],  # 蓝色线条
#     "PETA": [89.1, 89.8, 89.5, 89.0, 88.7],  # 橙色线条
# }

# 创建画布
plt.figure(figsize=(10, 6), facecolor='white')
plt.grid(True,
         color='#999999',        # 改为中灰色
         linestyle=(0, (2,2)),   # 短虚线（2pt线段+2pt间隔）
         linewidth=0.8,
         alpha=0.5)

# 绘制折线
colors = {'PA100k': 'blue', 'PETA': 'orange'}
for model, values in models.items():
    plt.plot(beta_values, values,
             marker='o',
             color=colors[model],
             linewidth=2.5,
             markersize=8,
             label=model)

# 设置坐标轴
plt.xticks(beta_values, fontsize=20)
plt.yticks(np.arange(85, 91, 1), fontsize=20)  # y轴60-74，步长2
plt.xlabel("β的取值", fontsize=22, labelpad=10,fontproperties=prop)
plt.ylabel("mA值（%）", fontsize=22, labelpad=10,fontproperties=prop)
plt.ylim(85, 91)  # 留出顶部空间显示标签

# 添加图例
plt.legend(loc='lower right', fontsize=12, frameon=False,prop=prop)

# # 显示数据标签（可选）
# for model, values in models.items():
#     for x, y in zip(beta_values, values):
#         plt.text(x, y+0.8, f'{y}',
#                  ha='center',
#                  color=colors[model],
#                  fontsize=10)

# 图表美化
plt.gca().spines['top'].set_visible(True)
plt.gca().spines['right'].set_visible(True)


plt.tight_layout()
plt.show()