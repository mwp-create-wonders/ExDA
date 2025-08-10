import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 创建混淆矩阵数据
data = {
    'name': ['BigGAN', 'CycleGAN', 'DALLE', 'Flux', 'Glide', 'LatentDM', 'Midjouney', 'SD', 'Real'],
    'BigGAN': [997, 0, 7, 0, 12, 4, 0, 0, 0],
    'CycleGAN': [0, 1020, 0, 0, 0, 0, 0, 0, 0],
    'DALLE': [2, 0, 959, 5, 26, 20, 8, 0, 0],
    'Flux': [0, 0, 0, 965, 0, 0, 0, 21, 34],
    'Glide': [0, 0, 12, 0, 991, 17, 0, 0, 0],
    'LatentDM': [6, 0, 13, 0, 33, 958, 10, 0, 0],
    'Midjouney': [0, 0, 24, 0, 6, 6, 984, 0, 0],
    'SD': [0, 0, 0, 0, 0, 0, 0, 1020, 0],
    'Real': [10, 0, 0, 0, 8, 28, 0, 0, 974]
}


# 将数据转换为 DataFrame
df_cm = pd.DataFrame(data).set_index('name')

# 计算每行的总和
row_sums = df_cm.sum(axis=0)
print(row_sums)

# 将每个单元格的值除以其所在行的总和，并乘以100得到百分比
df_cm_normalized = df_cm.div(row_sums, axis=0) * 100
print(df_cm_normalized)
df_cm_normalized = df_cm_normalized.T

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(df_cm_normalized, annot=True, fmt='.2f', cmap='Oranges', cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5, linecolor='gray', ).xaxis.tick_top()

# 添加标题和标签
plt.xlabel('Predicted Models', fontsize=16)
plt.ylabel('True Models', fontsize=16)


# 显示图形
plt.tight_layout()
# plt.show()
plt.savefig("confusion.png",dpi=800)
