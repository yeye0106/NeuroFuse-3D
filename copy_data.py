import pandas as pd
import os
import shutil
from pathlib import Path

# ----------------------------- 配置参数 ---------------------------------
CSV_PATH = "label.csv"   # 你的CSV文件
DATA_ROOT = r"D:\desktop\dataset\dataset"                          # 原始数据根目录
TARGET_ROOT = "dataset"                                    # 目标文件夹（相对于脚本运行目录）
# ----------------------------------------------------------------------

def build_image_id_index(data_root):
    """建立 Image Data ID 到源文件夹路径的索引"""
    print("正在扫描原始数据目录，建立索引...")
    id_to_path = {}
    for root, dirs, files in os.walk(data_root):
        for d in dirs:
            if d.startswith('I') and d[1:].isdigit():
                full_path = os.path.join(root, d)
                id_to_path[d] = full_path
    print(f"索引建立完成，共找到 {len(id_to_path)} 个 Image ID 文件夹。")
    return id_to_path

def copy_dataset(csv_path, data_root, target_root):
    # 读取CSV
    df = pd.read_csv(csv_path)
    id_col = 'Image Data ID'
    image_ids = df[id_col].astype(str).tolist()
    print(f"CSV中共有 {len(image_ids)} 条记录。")

    # 建立源数据索引
    src_index = build_image_id_index(data_root)

    # 创建目标根目录
    os.makedirs(target_root, exist_ok=True)

    copied_count = 0
    missing_ids = []
    skip_existing = True  # 如果目标已存在，是否跳过（避免重复复制）

    for idx, img_id in enumerate(image_ids):
        if img_id not in src_index:
            missing_ids.append(img_id)
            print(f"警告：未找到源文件夹 - {img_id}")
            continue

        src_path = src_index[img_id]
        dst_path = os.path.join(target_root, img_id)

        # 如果目标文件夹已存在，跳过或覆盖（根据需求）
        if os.path.exists(dst_path):
            if skip_existing:
                print(f"[{idx+1}/{len(image_ids)}] {img_id} 目标已存在，跳过。")
                continue
            else:
                shutil.rmtree(dst_path)

        # 执行复制
        try:
            shutil.copytree(src_path, dst_path)
            copied_count += 1
            print(f"[{idx+1}/{len(image_ids)}] 已复制 {img_id}")
        except Exception as e:
            print(f"[{idx+1}/{len(image_ids)}] 复制 {img_id} 时出错: {e}")

    # 输出汇总
    print("\n" + "=" * 60)
    print(f"复制完成！成功复制 {copied_count} 个文件夹到 {os.path.abspath(target_root)}")
    if missing_ids:
        print(f"缺失 {len(missing_ids)} 个ID，已保存至 missing_ids.txt")
        with open("missing_ids.txt", "w") as f:
            for mid in missing_ids:
                f.write(mid + "\n")
    print("=" * 60)

if __name__ == "__main__":
    copy_dataset(CSV_PATH, DATA_ROOT, TARGET_ROOT)