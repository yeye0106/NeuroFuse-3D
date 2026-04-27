import os
import ants
from nilearn import datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "dataset_nii"
OUTPUT_DIR = "dataset_preprocessed"
MAX_WORKERS = 2  # 建议不要太大，配准非常吃 CPU 和内存

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("正在自动下载/加载 MNI152 标准脑模板...")
# nilearn 会自动下载 2009c 非线性对称 MNI 模板（学术界公认标准）
mni_dataset = datasets.fetch_icbm152_2009()
MNI_T1_PATH = mni_dataset.t1  # 包含颅骨的 MNI 模板
MNI_MASK_PATH = mni_dataset.mask  # 仅脑实质的 Mask (0和1)

# 全局加载模板，避免多线程重复加载
MNI_T1 = ants.image_read(MNI_T1_PATH)
MNI_MASK = ants.image_read(MNI_MASK_PATH)


# ============================================

def process_single_mri(nii_filename):
    """
    处理单个 MRI 图像的核心逻辑
    """
    input_path = os.path.join(INPUT_DIR, nii_filename)
    output_path = os.path.join(OUTPUT_DIR, nii_filename)

    if os.path.exists(output_path):
        return True, f"已存在，跳过: {nii_filename}"

    try:
        # 1. 读取图像
        img = ants.image_read(input_path)

        # 2. N4 偏置场校正 (极其重要：消除磁场不均匀导致的伪影)
        img_n4 = ants.n4_bias_field_correction(img)

        # 3. 空间配准 (核心：只用 Affine，保留萎缩特征)
        # fixed: 目标模板；moving: 要移动的图像
        reg = ants.registration(
            fixed=MNI_T1,
            moving=img_n4,
            type_of_transform='Affine'  # 论文中的方法学核心词汇
        )
        warped_img = reg['warpedmovout']

        # 4. 颅骨剥离 (Mask 投影法)
        # 因为现在图像已经和 MNI 模板完全对齐，直接用 MNI 的掩膜乘上去即可
        # warped_img 是连续体素值，MNI_MASK 是 0/1 矩阵，相乘后背景清零
        stripped_img = warped_img * MNI_MASK

        # 5. 保存结果
        ants.image_write(stripped_img, output_path)
        return True, f"成功: {nii_filename}"

    except Exception as e:
        return False, f"失败 [{nii_filename}]: {str(e)}"


def main():
    # 获取所有待处理的 nii.gz 文件
    nii_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.nii.gz')]

    print(f"共发现 {len(nii_files)} 个 NIfTI 文件。开始高级预处理流水线...")

    success_count = 0
    fail_list = []

    # 启动多线程
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(process_single_mri, f): f for f in nii_files}

        for future in tqdm(as_completed(future_to_file), total=len(nii_files), desc="MRI 配准与剥骨"):
            success, msg = future.result()
            if success:
                success_count += 1
            else:
                fail_list.append(msg)

    print("\n" + "=" * 40)
    print("高级预处理任务结束！")
    print(f"成功: {success_count} | 失败: {len(fail_list)}")

    if fail_list:
        print("\n=== 失败详情 (前10个) ===")
        for msg in fail_list[:10]:
            print(msg)


if __name__ == "__main__":
    main()