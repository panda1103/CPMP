import pandas as pd

df = pd.read_csv('rf_baselines_result.csv')
#df = pd.read_csv('svm_baselines_result.csv')
df_sorted = df.sort_values(by='mse', ascending=True)

# 提取 mse 最低的三行
top = df_sorted.head(1)['model'].values[0]
df_select = df[df['model']==top].head(3)
print(df_select)
# 计算 r2, mse, mae 的平均值
mean_r2 = df_select['r2'].mean()
mean_mse = df_select['mse'].mean()
mean_mae = df_select['mae'].mean()

# 计算偏差（标准差）
std_r2 = df_select['r2'].std()
std_mse = df_select['mse'].std()
std_mae = df_select['mae'].std()

# 打印结果
print(f"Mean r2: {mean_r2}, Std r2: {std_r2}")
print(f"Mean mse: {mean_mse}, Std mse: {std_mse}")
print(f"Mean mae: {mean_mae}, Std mae: {std_mae}")
