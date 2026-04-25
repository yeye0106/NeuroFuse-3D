import pandas as pd
import os

CSV_PATH = "label.csv"
DATA_ROOT = "dataset_nii"

def build_image_id_index(data_root):
    """
    递归遍历数据根目录，找到所有以 'I' 开头且后跟数字的文件夹，
    将其作为 Image Data ID，记录完整路径。
    """
    id_to_path = {}
    print("正在扫描数据目录，建立索引...")
    for root, dirs, files in os.walk(data_root):
        for d in dirs:
            if d.startswith('I') and d[1:].isdigit():  # 典型的 ADNI IID 如 I13710
                full_path = os.path.join(root, d)
                id_to_path[d] = full_path
    print(f"索引建立完成，共找到 {len(id_to_path)} 个 Image ID 文件夹。")
    return id_to_path

def check_with_index(csv_path, data_root):
    df = pd.read_csv(csv_path)
    print("CSV 列名：", df.columns.tolist())
    print(f"总记录数：{len(df)}")

    # 确定 Image ID 列
    id_col = 'Image Data ID'
    if id_col not in df.columns:
        for col in df.columns:
            if 'image' in col.lower() and 'id' in col.lower():
                id_col = col
                break
    print(f"使用的 Image ID 列名：{id_col}")

    # 建立索引
    id_index = build_image_id_index(data_root)

    # 检查每条记录
    exists = []
    missing = []
    for idx, row in df.iterrows():
        img_id = str(row[id_col])
        if img_id in id_index:
            exists.append((idx, img_id, id_index[img_id]))
        else:
            missing.append((idx, img_id))

    print("\n" + "=" * 60)
    print(f"存在数据：{len(exists)} 条")
    print(f"缺失数据：{len(missing)} 条")
    print("=" * 60)

    if missing:
        print("\n缺失记录示例（前10条）：")
        for i, (idx, img_id) in enumerate(missing[:10]):
            print(f"  {idx}: {img_id}")
        missing_df = pd.DataFrame(missing, columns=['Index', 'Image Data ID'])
        missing_out = os.path.join(os.path.dirname(csv_path), 'missing_by_id.csv')
        missing_df.to_csv(missing_out, index=False)
        print(f"\n所有缺失记录已保存至：{missing_out}")

    return exists, missing

if __name__ == "__main__":
    check_with_index(CSV_PATH, DATA_ROOT)