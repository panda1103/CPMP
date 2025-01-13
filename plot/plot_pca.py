import numpy as np
import pandas as pd
import os
import sys

df1 = pd.read_csv('../caco2_uff_ig_true/CycPeptMPDB_Peptide_Assay_Caco2.csv', low_memory=False)
df1['Label'] = 'Caco2'
df2 = pd.read_csv('../rrck_uff_ig_true/CycPeptMPDB_Peptide_Assay_RRCK.csv', low_memory=False)
df2['Label'] = 'RRCK'
df3 = pd.read_csv('../mdck_uff_ig_true/CycPeptMPDB_Peptide_Assay_MDCK.csv', low_memory=False)
df3['Label'] = 'MDCK'

df = pd.concat([df1, df2, df3])[['SMILES', 'Label', 'PC1', 'PC2']]
df.to_csv('label_pca.csv', index=False)

import matplotlib.pyplot as plt


colors = [
    (78, 98, 171),
    (203, 233, 157),
    (214, 64, 78)
]
markers = ['o', 's', '^']
color_map = {label: (r/255, g/255, b/255) for label, (r, g, b) in zip(df['Label'].unique(), colors)}
marker_map = { label: x for label, x in zip(df['Label'].unique(), markers)}
# 创建散点图
plt.figure(figsize=(8, 8))

for label in df['Label'].unique():
    subset = df[df['Label'] == label]
    plt.scatter(subset['PC1'], subset['PC2'],
                color=color_map[label],
                marker=marker_map[label],  # 圆形标记
                facecolors='none',  # 空心填充
                edgecolors=color_map[label],  # 边框颜色
                linewidths=1.5,  # 边框宽度
                label=label)

# 添加图例
plt.legend(title='', fontsize=14)

# 添加标题和轴标签
plt.title('PCA of Cycle Peptides from 3 Datasets', fontsize=18)
plt.xlabel('PC1 (20.7%)', fontsize=16)
plt.ylabel('PC2 (9.3%)', fontsize=16)

# 保存图形为 PNG 文件
plt.savefig('pca.png')
plt.savefig('pca.pdf')

# 显示图形
plt.show()
