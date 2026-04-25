import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# 1. 读取上一步合并的数据
df = pd.read_csv("multimodal_dataset_raw.csv")

# 2. 定义参与插值的特征（坚决排除 Group 标签，防止数据泄露）
features_for_imputation = ['Age', 'Sex', 'PTEDUCAT', 'ApoE4_Count', 'MMSCORE', 'CDGLOBAL', 'CDRSB']
df_impute = df[features_for_imputation].copy()

# 3. 类别特征编码：KNN 只能处理数值，需要将 Sex 转换为 0 和 1 (M=0, F=1)
df_impute['Sex'] = df_impute['Sex'].map({'M': 0, 'F': 1})

# 4. 执行 KNN 插值
# n_neighbors=5 是一般医学插值的经验值
# weights='distance' 表示特征越相似的患者，其投票权重越大
print("正在执行 KNN 距离加权插值...")
imputer = KNNImputer(n_neighbors=5, weights='distance')
imputed_array = imputer.fit_transform(df_impute)

# 将插值后的 numpy 数组转回 DataFrame
df_imputed = pd.DataFrame(imputed_array, columns=features_for_imputation)

# ---------------------------------------------------------
# 5. 后处理：生理学对齐（重点亮点）
# 机器学习插出的值带有小数，不符合真实临床量表的离散性质，需要进行修正
# ---------------------------------------------------------
print("正在进行临床分数离散化校准...")

# ApoE4 携带等位基因数量只能是 0, 1, 2，四舍五入并限制边界
df_imputed['ApoE4_Count'] = df_imputed['ApoE4_Count'].round().clip(0, 2)

# MMSCORE 满分 30 分，必须是整数
df_imputed['MMSCORE'] = df_imputed['MMSCORE'].round().clip(0, 30)

# CDGLOBAL 和 CDRSB 通常具有 0.5 的步进（如 0.5, 1.0, 1.5, 2.0）
# 优雅的 0.5 舍入算法：乘以 2 -> 四舍五入 -> 除以 2
df_imputed['CDGLOBAL'] = (df_imputed['CDGLOBAL'] * 2).round() / 2
df_imputed['CDRSB'] = (df_imputed['CDRSB'] * 2).round() / 2

# 6. 将处理好的特征替换回原数据集
for col in ['PTEDUCAT', 'ApoE4_Count', 'MMSCORE', 'CDGLOBAL', 'CDRSB']:
    df[col] = df_imputed[col]

# 为了后续送入深度学习全连接层，我们顺便把数值化后的性别保存下来
df['Sex_encoded'] = df_imputed['Sex']

# 7. 最终检查与保存
print("\n=== 插值后数据集缺失值统计 ===")
print(df[['Age', 'Sex_encoded', 'PTEDUCAT', 'ApoE4_Count', 'MMSCORE', 'CDGLOBAL', 'CDRSB']].isnull().sum())

df.to_csv("multimodal_dataset_final.csv", index=False)
print("\n已成功保存至 multimodal_dataset_final.csv ！这是最终可以喂给网络的表格。")