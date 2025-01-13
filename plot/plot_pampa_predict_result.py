import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

data = pd.read_csv('../../train_pampa/repeat_1_uff_ig_true_h64_N6_N_dense2_slope0.16_batch_size32.predict.csv')

# 计算R², MSE, MAE
r2 = r2_score(data['y'], data['predict'])
mse = mean_squared_error(data['y'], data['predict'])
mae = mean_absolute_error(data['y'], data['predict'])

# 绘制散点图
plt.figure(figsize=(8, 8))
point_size = 70
plt.scatter(data['predict'], data['y'], color=(214/255, 64/255, 78/255), label=f'R²={r2:.2f}, MSE={mse:.2f}, MAE={mae:.2f}', edgecolors='white', s=point_size)

# 绘制y=x的虚线对角线
plt.plot([-10, -4], [-10, -4], '--',  color='gray')

plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
plt.grid(which='both', linestyle='--', color='lightgray', alpha=0.5)

# 设置图例
plt.legend(fontsize=14)
# 设置横轴和纵轴范围
plt.xlim(-10, -4)
plt.ylim(-10, -4)

# 设置标签
plt.xlabel('True Permeability', fontsize=16)
plt.ylabel('Predicted Permeability', fontsize=16)

# 设置标题
plt.title('PAMPA Permeability Prediction', fontsize=18)
plt.tick_params(axis='both', labelsize=14)

# 显示图形
plt.savefig('pampa.png')
plt.savefig('pampa.pdf')

# 显示图形
plt.show()
