import os
import re
import json
import logging
import pandas as pd
import pydicom
import dicom2nifti
import dicom2nifti.settings as settings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- 配置区 ---
origin_root = r'C:\all'  # 下载的原始 DICOM 根目
csv_path = 'cleaned_all.csv'  # ADNI 的 CSV 路径录
output_root = 'dataset_nii'  # 转换后的根目录
MAX_WORKERS = 8  # 根据 CPU 核心数调整线程数

# 启用压缩并禁用一些严格检查（ADNI 数据有时会有切片间距微小差异，建议开启）
settings.enable_resampling()
settings.set_resample_spline_interpolation_order(1)
settings.set_resample_padding(0)


def extract_metadata_to_json(dicom_folder, json_path):
    """提取 DICOM 头部信息并保存为 JSON"""
    try:
        # 获取文件夹中第一个 DICOM 文件
        dcm_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm') or f.startswith('I')]
        if not dcm_files:
            return False

        sample_dcm = pydicom.dcmread(os.path.join(dicom_folder, dcm_files[0]))

        # 提取常用字段，可以根据需要增加
        metadata = {
            "PatientID": getattr(sample_dcm, "PatientID", "Unknown"),
            "SeriesDescription": getattr(sample_dcm, "SeriesDescription", "Unknown"),
            "StudyDate": getattr(sample_dcm, "StudyDate", "Unknown"),
            "Modality": getattr(sample_dcm, "Modality", "Unknown"),
            "Manufacturer": getattr(sample_dcm, "Manufacturer", "Unknown"),
            "MagneticFieldStrength": getattr(sample_dcm, "MagneticFieldStrength", "Unknown"),
            "EchoTime": float(getattr(sample_dcm, "EchoTime", 0)),
            "RepetitionTime": float(getattr(sample_dcm, "RepetitionTime", 0)),
            "FlipAngle": float(getattr(sample_dcm, "FlipAngle", 0)),
            "PixelSpacing": [float(x) for x in getattr(sample_dcm, "PixelSpacing", [0, 0])],
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        return str(e)


def convert_single_patient(img_id, dicom_folder, patient_out_dir):
    """处理单个样本：创建文件夹、转 NII、写 JSON、写 Log"""
    # 1. 创建该 ID 的专属文件夹
    os.makedirs(patient_out_dir, exist_ok=True)

    nii_path = os.path.join(patient_out_dir, f"{img_id}.nii.gz")
    json_path = os.path.join(patient_out_dir, f"{img_id}.json")
    log_path = os.path.join(patient_out_dir, f"{img_id}.txt")

    log_content = []
    success = True

    # 2. 转换 NIfTI
    try:
        if not os.path.exists(nii_path):
            dicom2nifti.dicom_series_to_nifti(dicom_folder, nii_path, reorient_nifti=True)
            log_content.append(f"NIfTI Conversion: SUCCESS")
        else:
            log_content.append(f"NIfTI Conversion: SKIPPED (Already exists)")
    except Exception as e:
        success = False
        log_content.append(f"NIfTI Conversion: FAILED - {str(e)}")

    # 3. 提取 JSON 元数据
    json_res = extract_metadata_to_json(dicom_folder, json_path)
    if json_res is True:
        log_content.append("Metadata Extraction: SUCCESS")
    else:
        log_content.append(f"Metadata Extraction: FAILED/EMPTY - {json_res}")

    # 4. 写入日志文件
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(log_content))

    return success, img_id


def build_id_mapping(root_dir):
    """扫描文件夹，建立 ImageID -> 路径的映射"""
    mapping = {}
    id_pattern = re.compile(r'^I\d+$')
    for dirpath, dirnames, _ in os.walk(root_dir):
        for d in dirnames:
            if id_pattern.match(d):
                if d not in mapping:
                    mapping[d] = os.path.join(dirpath, d)
    return mapping


def main():
    # 读取数据
    df = pd.read_csv(csv_path)
    df['Image Data ID'] = df['Image Data ID'].astype(str)

    # 建立路径映射
    id_to_folder = build_id_mapping(origin_root)

    tasks = []
    for img_id in df['Image Data ID'].unique():
        dcm_dir = id_to_folder.get(img_id)
        if dcm_dir:
            # 目标文件夹：output_root/I12345/
            patient_out_dir = os.path.join(output_root, img_id)
            tasks.append((img_id, dcm_dir, patient_out_dir))

    print(f"开始转换，总计任务数: {len(tasks)}")

    # 多线程执行
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(convert_single_patient, tid, tdir, tout): tid
            for tid, tdir, tout in tasks
        }

        for future in tqdm(as_completed(future_to_id), total=len(tasks), desc="Processing ADNI"):
            res, tid = future.result()

    print("\n所有任务处理完成！请检查输出目录。")


if __name__ == "__main__":
    main()