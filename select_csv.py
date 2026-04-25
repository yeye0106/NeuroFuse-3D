import pandas as pd
import numpy as np
import warnings

# 忽略 pandas 的一些赋值警告，保持终端干净
warnings.filterwarnings('ignore')

# ================= 配置参数 =================
INPUT_CSV = 'cleaned_all.csv'
DXSUM_CSV = 'Labels/dxsum.csv'  # 引入金标准诊断表
OUTPUT_CSV = 'label.csv'
RANDOM_SEED = 42
BALANCE_CLASSES = True  # 是否开启强制样本均衡 (1:1:1)
time_window_days = 90


# ============================================

def calculate_quality_score(description):
    if pd.isna(description):
        return -1

    desc = str(description).lower().strip()
    reject_keywords = ['scout', 'cal', 'b1', 'mapping', 'vwip', 'phase', 'localizer', 'aahead']
    if any(keyword in desc for keyword in reject_keywords):
        return -1

    score = 10
    if 'mprage' in desc or 'mp-rage' in desc or 'mp rage' in desc:
        score += 50
    if 'spgr' in desc:
        score += 50
    if 'accelerated' in desc or 'grappa' in desc or 'sense' in desc or 'accel' in desc:
        score += 20
    if 'repeat' in desc or 'repet' in desc:
        score += 10
    if 'no angle' in desc or '_nd' in desc:
        score -= 5

    return score


def parse_true_diagnosis(row):
    """解析 dxsum.csv 中的权威诊断"""
    diag = str(row.get('DIAGNOSIS', '')).strip()
    if diag in ['1', '1.0']: return 'CN'
    if diag in ['2', '2.0']: return 'MCI'
    if diag in ['3', '3.0']: return 'AD'

    dxnorm = str(row.get('DXNORM', '')).strip()
    if dxnorm in ['1', '1.0']: return 'CN'

    dxmci = str(row.get('DXMCI', '')).strip()
    if dxmci in ['1', '1.0']: return 'MCI'

    dxad = str(row.get('DXAD', '')).strip()
    if dxad in ['1', '1.0']: return 'AD'

    return np.nan


def get_true_diagnosis(subject, mri_date, df_dxsum, time_window_days=time_window_days):
    """为 MRI 匹配最近的临床诊断"""
    records = df_dxsum[df_dxsum['PTID'] == subject].copy()
    if records.empty:
        return np.nan

    records['time_diff'] = (records['EXAMDATE'] - mri_date).dt.days.abs()
    closest = records.loc[records['time_diff'].idxmin()]

    if closest['time_diff'] > time_window_days:
        return np.nan

    return closest['True_Diagnosis']


def main():
    print(f"正在读取原始影像数据: {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)

    # 基础清理
    if 'Subject' in df.columns:
        df['Subject'] = df['Subject'].astype(str).str.strip()
    df['Acq Date'] = pd.to_datetime(df['Acq Date'], errors='coerce')
    df = df.dropna(subset=['Acq Date'])

    # ---------------------------------------------------------
    # 核心新增：在源头执行金标准标签洗礼
    # ---------------------------------------------------------
    print(f"正在加载金标准临床诊断表: {DXSUM_CSV} ...")
    df_dxsum = pd.read_csv(DXSUM_CSV, dtype=str)
    df_dxsum['EXAMDATE'] = pd.to_datetime(df_dxsum['EXAMDATE'], errors='coerce')
    df_dxsum['True_Diagnosis'] = df_dxsum.apply(parse_true_diagnosis, axis=1)
    df_dxsum = df_dxsum.dropna(subset=['True_Diagnosis'])

    print(f"正在为所有影像进行时间轴对齐，纠正错误标签 (时间窗: {time_window_days}天)...")
    # 这一步稍微耗时，但极度保证了数据的纯洁性
    df['True_Group'] = df.apply(lambda row: get_true_diagnosis(row['Subject'], row['Acq Date'], df_dxsum), axis=1)

    # 丢弃找不到真实诊断的影像
    df_valid = df.dropna(subset=['True_Group']).copy()
    df_valid['Group'] = df_valid['True_Group']  # 覆盖旧的脏标签
    df_valid = df_valid.drop(columns=['True_Group'])
    print(f"成功匹配到金标准诊断的影像剩余: {len(df_valid)} 条")

    # ---------------------------------------------------------
    # 继续原本的筛选和打分流程
    # ---------------------------------------------------------
    df_valid['Quality_Score'] = df_valid['Description'].apply(calculate_quality_score)
    df_filtered = df_valid[df_valid['Quality_Score'] > 0].copy()

    df_sorted = df_filtered.sort_values(by=['Subject', 'Acq Date', 'Quality_Score'],
                                        ascending=[True, True, False])

    df_unique = df_sorted.drop_duplicates(subset=['Subject'], keep='first').copy()
    print(f"严格去重后 (每位患者保留1张高质量确诊基线): {len(df_unique)} 条")

    # 执行抽样，此时抽选的是绝对干净的数据，比例绝不会再被破坏！
    if BALANCE_CLASSES:
        print("\n正在执行类别均衡 (匹配少数类样本数量)...")
        min_class_count = df_unique['Group'].value_counts().min()

        balanced_dfs = []
        for group_name in df_unique['Group'].unique():
            df_group = df_unique[df_unique['Group'] == group_name]
            df_group_sampled = df_group.sample(n=min_class_count, random_state=RANDOM_SEED)
            balanced_dfs.append(df_group_sampled)

        df_final = pd.concat(balanced_dfs).reset_index(drop=True)
    else:
        df_final = df_unique.reset_index(drop=True)

    df_final = df_final.sort_values(by=['Subject'], ascending=[True])

    df_final = df_final.drop(columns=['Quality_Score'])
    df_final['Acq Date'] = df_final['Acq Date'].dt.strftime('%m/%d/%Y')

    print(f"\n============= 最终输出结果 =============")
    print(f"总样本数: {len(df_final)}")
    print("最终分组样本如下 (完美均衡的金标准数据):")
    for group_name, count in df_final['Group'].value_counts().items():
        print(f"  - {group_name}: {count}")

    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 处理完成！高质量且排版整齐的数据已保存至 {OUTPUT_CSV}")


if __name__ == "__main__":
    main()