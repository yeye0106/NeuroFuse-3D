import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "dataset_preprocessed"  # 刚刚做完剥骨的文件夹
OUTPUT_DIR = "dataset_final_128"  # 最终可以喂给神经网络的文件夹
TARGET_SHAPE = (128, 128, 128)  # 目标 3D 尺寸
MAX_WORKERS = 8  # 线程数

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================

def process_single_image(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(output_path):
        return True, f"已存在，跳过: {filename}"

    try:
        # 1. 读取 NIfTI 图像数据
        img = nib.load(input_path)
        data = img.get_fdata()

        # 2. 紧凑裁剪 (Crop) - 去除全黑背景
        # 找到所有非零体素的坐标
        coords = np.array(np.nonzero(data))
        if coords.size == 0:
            return False, f"空白图像: {filename}"

        min_coords = coords.min(axis=1)  # 最小边界 (x, y, z)
        max_coords = coords.max(axis=1) + 1  # 最大边界 (x, y, z)

        # 按照边界切片，剔除冗余黑边
        cropped_data = data[min_coords[0]:max_coords[0],
        min_coords[1]:max_coords[1],
        min_coords[2]:max_coords[2]]

        # 3. 缩放 (Resize) 到 128x128x128
        # 计算三个维度的缩放比例
        zoom_factors = [t / s for t, s in zip(TARGET_SHAPE, cropped_data.shape)]
        # order=1 表示双线性插值，速度快且不易产生人工伪影
        resized_data = zoom(cropped_data, zoom_factors, order=1)

        # 4. 归一化 (Min-Max Normalization to 0-1)
        min_val = resized_data.min()
        max_val = resized_data.max()
        if max_val > min_val:
            normalized_data = (resized_data - min_val) / (max_val - min_val)
        else:
            normalized_data = resized_data

        # 5. 转换为单精度浮点数 (float32) 以节省显存，并保存
        normalized_data = normalized_data.astype(np.float32)

        # 对于深度学习，仿射矩阵我们只需要一个单位矩阵即可，因为空间位置已经对齐
        new_img = nib.Nifti1Image(normalized_data, np.eye(4))
        nib.save(new_img, output_path)

        return True, f"成功: {filename}"

    except Exception as e:
        return False, f"失败 [{filename}]: {str(e)}"


def main():
    nii_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.nii.gz')]
    print(f"共发现 {len(nii_files)} 个待处理图像。开始 裁剪 -> 缩放 -> 归一化...")

    success_count = 0
    fail_list = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_image, f): f for f in nii_files}

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