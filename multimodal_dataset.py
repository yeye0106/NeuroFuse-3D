import pandas as pd
import numpy as np

# ================= 配置参数 =================
TIME_WINDOW_DAYS = 90  # 黄金标准时间窗
# ============================================

# 1. 读取数据 (此时的 label.csv 已经是千锤百炼的金标准数据，且 1:1:1 均衡)
df_label = pd.read_csv("label.csv")
df_apoe = pd.read_csv("Labels/apoeres.csv")
df_cdr = pd.read_csv("Labels/cdr.csv")
df_mmse = pd.read_csv("Labels/mmse.csv")
df_ptdemog = pd.read_csv("Labels/ptdemog.csv")

# 统一日期格式
df_label['Acq Date'] = pd.to_datetime(df_label['Acq Date'])
df_cdr['VISDATE'] = pd.to_datetime(df_cdr['VISDATE'], errors='coerce')
df_mmse['VISDATE'] = pd.to_datetime(df_mmse['VISDATE'], errors='coerce')

# ---------------------------------------------------------
# 步骤一：处理静态特征 (基因、教育年限)
# 这些特征人一生不变，不需要时间窗，直接用 PTID 最新记录拼接
# ---------------------------------------------------------
print("正在拼接静态特征 (ApoE4 基因, 教育年限)...")
df_apoe_clean = df_apoe[['PTID', 'GENOTYPE']].dropna().drop_duplicates(subset=['PTID'], keep='last').copy()
df_apoe_clean['ApoE4_Count'] = df_apoe_clean['GENOTYPE'].apply(lambda x: str(x).count('4'))

df_edu_clean = df_ptdemog[['PTID', 'PTEDUCAT']].dropna().drop_duplicates(subset=['PTID'], keep='last').copy()

df_merged = pd.merge(df_label, df_apoe_clean[['PTID', 'ApoE4_Count']], left_on='Subject', right_on='PTID', how='left').drop(columns=['PTID'])
df_merged = pd.merge(df_merged, df_edu_clean, left_on='Subject', right_on='PTID', how='left').drop(columns=['PTID'])


# ---------------------------------------------------------
# 步骤二：处理动态特征 (MMSE, CDR) - 严苛的 90天时间对齐策略
# ---------------------------------------------------------
def get_closest_score(subject, mri_date, clinical_df, score_cols, time_window_days):
    patient_records = clinical_df[clinical_df['PTID'] == subject].copy()
    if patient_records.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    patient_records['time_diff'] = (patient_records['VISDATE'] - mri_date).dt.days.abs()
    closest_record = patient_records.loc[patient_records['time_diff'].idxmin()]

    if closest_record['time_diff'] > time_window_days:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    return closest_record[score_cols]

# 提取 MMSE
print(f"正在对齐 MMSE 时间序列 (允许最大时间窗: {TIME_WINDOW_DAYS} 天)...")
mmse_features = df_merged.apply(lambda row: get_closest_score(row['Subject'], row['Acq Date'], df_mmse, ['MMSCORE'], TIME_WINDOW_DAYS), axis=1)
df_merged = pd.concat([df_merged, mmse_features], axis=1)

# 提取 CDR
print(f"正在对齐 CDR 时间序列 (允许最大时间窗: {TIME_WINDOW_DAYS} 天)...")
cdr_features = df_merged.apply(lambda row: get_closest_score(row['Subject'], row['Acq Date'], df_cdr, ['CDGLOBAL', 'CDRSB'], TIME_WINDOW_DAYS), axis=1)
df_merged = pd.concat([df_merged, cdr_features], axis=1)

# ---------------------------------------------------------
# 步骤三：检查与保存
# ---------------------------------------------------------
print("\n=== 最终金标准数据集分组统计 (期望维持完美均衡) ===")
print(df_merged['Group'].value_counts())

print(f"\n=== 缺失值统计 (待 KNN 插值处理) ===")
print(f"注：量表如果在 {TIME_WINDOW_DAYS} 天内没找到匹配，则设为空值")
print(df_merged[['Age', 'Sex', 'PTEDUCAT', 'ApoE4_Count', 'MMSCORE', 'CDGLOBAL', 'CDRSB']].isnull().sum())

# 保存为新的 CSV
df_merged.to_csv("multimodal_dataset_raw.csv", index=False)
print("\n✅ 已成功保存至 multimodal_dataset_raw.csv")