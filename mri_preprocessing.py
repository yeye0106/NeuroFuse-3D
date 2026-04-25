import os
import ants
from nilearn import datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "dataset_nii"          # 第一个脚本的输出目录（内含 Ixxxx 子文件夹）
OUTPUT_DIR = "dataset_preprocessed"
MAX_WORKERS = 2                    # 配准非常消耗 CPU/内存，不宜过大

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("正在自动下载/加载 MNI152 标准脑模板...")
# nilearn 下载 2009c 非线性对称 MNI 模板（学术界公认标准）
mni_dataset = datasets.fetch_icbm152_2009()
MNI_T1_PATH = mni_dataset.t1         # 带颅骨的 MNI 模板
MNI_MASK_PATH = mni_dataset.mask     # 仅脑实质的 Mask（用于颅骨剥离）

# 全局加载模板，避免多线程重复加载
MNI_T1 = ants.image_read(MNI_T1_PATH)
MNI_MASK = ants.image_read(MNI_MASK_PATH)


def find_all_nifti_files(root_dir):
    """
    递归遍历 root_dir，收集所有 .nii.gz 文件的路径，
    并计算相对于 root_dir 的相对路径（用于保留子文件夹结构）。
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


def process_single_mri(input_path, rel_path):
    """
    处理单个 MRI 图像
    input_path : 完整的输入文件路径
    rel_path   : 相对于 INPUT_DIR 的路径（例如 "I12345/I12345.nii.gz"）
    输出文件将保存在 OUTPUT_DIR 下的相同相对路径位置
    """
    output_path = os.path.join(OUTPUT_DIR, rel_path)
    # 确保输出子文件夹存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        return True, f"已存在，跳过: {rel_path}"

    try:
        # 1. 读取图像
        img = ants.image_read(input_path)

        # 2. N4 偏置场校正（消除磁场不均匀导致的伪影）
        img_n4 = ants.n4_bias_field_correction(img)

        # 3. 空间配准：仅使用仿射变换（保留萎缩特征，不进行非线性变形）
        reg = ants.registration(
            fixed=MNI_T1,
            moving=img_n4,
            type_of_transform='Affine'
        )
        warped_img = reg['warpedmovout']

        # 4. 颅骨剥离：用 MNI 掩膜直接乘法（此时图像已与 MNI 模板对齐）
        stripped_img = warped_img * MNI_MASK

        # 5. 保存结果
        ants.image_write(stripped_img, output_path)
        return True, f"成功: {rel_path}"

    except Exception as e:
        return False, f"失败 [{rel_path}]: {str(e)}"


def main():
    # 获取所有待处理的 NIfTI 文件（包含相对路径）
    nii_files = find_all_nifti_files(INPUT_DIR)
    print(f"共发现 {len(nii_files)} 个 NIfTI 文件。开始高级预处理流水线...")

    success_count = 0
    fail_list = []

    # 启动多线程
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(process_single_mri, full_path, rel_path): rel_path
            for full_path, rel_path in nii_files
        }

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
        print("\n=== 失败详情（前10个） ===")
        for msg in fail_list[:10]:
            print(msg)


if __name__ == "__main__":
    main()