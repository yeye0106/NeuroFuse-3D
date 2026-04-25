import os
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区域 =================
INPUT_DIR = "dataset_final_128"
TARGET_SHAPE = (128, 128, 128)
MAX_WORKERS = 8  # 读取文件较慢，多线程加速


# ============================================

def qc_single_image(filename):
    filepath = os.path.join(INPUT_DIR, filename)
    report = {
        "filename": filename,
        "status": "PASS",
        "errors": [],
        "warnings": []
    }

    try:
        # 1. 加载图像测试
        img = nib.load(filepath)
        data = img.get_fdata()

        # 2. 形状检测
        if data.shape != TARGET_SHAPE:
            report["status"] = "FAIL"
            report["errors"].append(f"形状错误: {data.shape}，期望 {TARGET_SHAPE}")

        # 3. 数据损坏与坏点检测
        if np.isnan(data).any():
            report["status"] = "FAIL"
            report["errors"].append("发现 NaN (缺失值) 坏点")
        if np.isinf(data).any():
            report["status"] = "FAIL"
            report["errors"].append("发现 Inf (无穷大) 坏点")

        # 4. 归一化边界检测 [0, 1] (允许极小的浮点误差)
        min_val = data.min()
        max_val = data.max()
        if min_val < -1e-5 or max_val > 1 + 1e-5:
            report["status"] = "FAIL"
            report["errors"].append(f"未归一化: Min={min_val:.4f}, Max={max_val:.4f}")

        # 5. 专家级检测：背景冗余度 (Zero Ratio)
        # 计算完全为 0 的背景体素占整个 128^3 空间的比例
        zero_ratio = np.sum(data == 0) / data.size

        # 对于剥骨后的 128^3 图像，正常大脑的背景比例一般在 30% ~ 70% 之间
        if zero_ratio < 0.10:
            report["warnings"].append(f"背景太少 ({zero_ratio:.1%}): 可能未成功剥骨或裁剪异常")
        elif zero_ratio > 0.85:
            report["warnings"].append(f"背景太多 ({zero_ratio:.1%}): 图像可能萎缩极度严重或存在大片空白")

        # 检查是否全部是 0 (纯黑图像)
        if zero_ratio == 1.0:
            report["status"] = "FAIL"
            report["errors"].append("致命错误：全黑/空白图像")

    except Exception as e:
        report["status"] = "FAIL"
        report["errors"].append(f"读取崩溃: {str(e)}")

    return report


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"找不到文件夹: {INPUT_DIR}")
        return

    nii_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.nii.gz')]
    total_files = len(nii_files)
    print(f"🚀 开始执行质检，共发现 {total_files} 个文件...")

    qc_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(qc_single_image, f): f for f in nii_files}

        for future in tqdm(as_completed(future_to_file), total=total_files, desc="QC Checking"):
            qc_results.append(future.result())

    # 分析结果
    failed_files = [r for r in qc_results if r["status"] == "FAIL"]
    warned_files = [r for r in qc_results if len(r["warnings"]) > 0 and r["status"] == "PASS"]

    print("\n" + "=" * 50)
    print("📊 质量控制 (QC) 报告")
    print("=" * 50)
    print(f"总检查数 : {total_files}")
    print(f"完美通过 : {total_files - len(failed_files) - len(warned_files)}")
    print(f"通过带警告: {len(warned_files)}")
    print(f"❌ 质检失败: {len(failed_files)}")

    if failed_files:
        print("\n=== ❌ 失败文件详情 (必须修复或剔除) ===")
        for report in failed_files:
            err_str = " | ".join(report["errors"])
            print(f"[{report['filename']}] -> {err_str}")

    if warned_files:
        # 只打印前 10 个警告，避免刷屏
        print(f"\n=== ⚠️ 警告文件详情 (共 {len(warned_files)} 个，建议抽查前 10 个) ===")
        for report in warned_files[:10]:
            warn_str = " | ".join(report["warnings"])
            print(f"[{report['filename']}] -> {warn_str}")


if __name__ == "__main__":
    main()