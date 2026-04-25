import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "dataset_preprocessed"      # 上一步剥骨后的输出目录（内含 Ixxxx 子文件夹）
OUTPUT_DIR = "dataset_final_128"        # 最终输出目录
TARGET_SHAPE = (128, 128, 128)          # 目标 3D 尺寸
MAX_WORKERS = 16                         # 线程数

os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_all_nifti_files(root_dir):
    """
    递归遍历 root_dir，收集所有 .nii.gz 文件的路径，
    返回列表: [(full_path, relative_path), ...]
    """
    nii_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.endswith('.nii.gz'):
                full_path = os.path.join(dirpath, f)
                rel_path = os.path.relpath(full_path, root_dir)
                nii_files.append((full_path, rel_path))
    return nii_files


def process_single_image(input_path, rel_path):
    output_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        return True, f"已存在，跳过: {rel_path}"

    try:
        # 1. 读取 NIfTI 图像数据
        img = nib.load(input_path)
        data = img.get_fdata()

        # 2. 紧凑裁剪 (Crop) - 去除全黑背景
        coords = np.array(np.nonzero(data))
        if coords.size == 0:
            return False, f"空白图像: {rel_path}"

        min_coords = coords.min(axis=1)
        max_coords = coords.max(axis=1) + 1
        cropped_data = data[min_coords[0]:max_coords[0],
                           min_coords[1]:max_coords[1],
                           min_coords[2]:max_coords[2]]

        # 3. 缩放 (Resize) 到目标尺寸
        zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, cropped_data.shape)]
        resized_data = zoom(cropped_data, zoom_factors, order=1)  # 双线性插值

        # 4. 归一化 (Min-Max to 0-1)
        min_val = resized_data.min()
        max_val = resized_data.max()
        if max_val > min_val:
            normalized_data = (resized_data - min_val) / (max_val - min_val)
        else:
            normalized_data = resized_data

        normalized_data = normalized_data.astype(np.float32)

        # 5. 保存（仿射矩阵设为单位矩阵，因为空间已对齐）
        new_img = nib.Nifti1Image(normalized_data, np.eye(4))
        nib.save(new_img, output_path)

        return True, f"成功: {rel_path}"

    except Exception as e:
        return False, f"失败 [{rel_path}]: {str(e)}"


def main():
    nii_files = find_all_nifti_files(INPUT_DIR)
    print(f"共发现 {len(nii_files)} 个 NIfTI 文件。开始裁剪→缩放→归一化...")

    success_count = 0
    fail_list = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_single_image, full_path, rel_path): rel_path
            for full_path, rel_path in nii_files
        }

        for future in tqdm(as_completed(future_to_file), total=len(nii_files), desc="Final Processing"):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_list.append(msg)

    print("\n" + "=" * 40)
    print("图像终极预处理结束！")
    print(f"成功: {success_count} | 失败: {len(fail_list)}")

    if fail_list:
        print("\n=== 失败详情 (前10个) ===")
        for msg in fail_list[:10]:
            print(msg)


if __name__ == "__main__":
    main()