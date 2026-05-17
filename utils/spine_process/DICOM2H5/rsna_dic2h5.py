import os
import re
import numpy as np
import pydicom
import h5py
from pydicom.misc import is_dicom
import pandas as pd

def natural_sort_key(filename):
    """对 IM0, IM1, ..., IM11 进行自然排序"""
    match = re.search(r'IM(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def normalize_volume(volume, modality):
    """
    根据模态选择归一化方法
    返回 normalized_volume, norm_params (dict)
    """
    if modality in ["T1", "T2", "FS", "AX"]:
        # Min-Max 归一化到 [0, 1]
        min_val = np.min(volume)
        max_val = np.max(volume)
        if max_val - min_val == 0:
            normalized = np.zeros_like(volume)
        else:
            normalized = (volume - min_val) / (max_val - min_val)
        return normalized, {
            'norm_method': 'min_max',
            'min': min_val,
            'max': max_val
        }

def process_dicom_folder_to_h5(dicom_dir, modality_name, bg_threshold=0.0):
    """
    处理一个 DICOM 文件夹（如 AX/），生成 {modality}.h5
    """
    # 检查是否已经存在 h5 文件，如果存在则跳过
    h5_path = os.path.join(dicom_dir, f"{modality_name}.h5")
    if os.path.exists(h5_path):
        print(f"跳过已处理的文件: {h5_path}")
        return True  # 返回 True 表示已完成（虽然跳过了）
    
    # 获取所有文件（仅文件，非目录）
    all_files = [f for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]

    # 筛选 DICOM 文件
    dicom_paths = []
    for f in all_files:
        path = os.path.join(dicom_dir, f)
        if is_dicom(path):
            dicom_paths.append(path)

    if not dicom_paths:
        print(f"跳过 {dicom_dir}：未找到 DICOM 文件")
        return False

    # 按 IM0, IM1, ... 自然排序
    dicom_paths.sort(key=lambda p: natural_sort_key(os.path.basename(p)))

    # 读取像素数据
    slices = []
    for path in dicom_paths:
        try:
            ds = pydicom.dcmread(path)
            slices.append(ds.pixel_array.astype(np.float32))
        except Exception as e:
            print(f"读取失败 {path}: {e}")
            return False

    volume = np.stack(slices, axis=0)  # (N, H, W)

    # 根据模态进行归一化
    normalized_volume, norm_params = normalize_volume(volume, modality_name)

    # 保存到 H5（保存在当前 dicom_dir 下）
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('slices', data=normalized_volume, dtype=np.float32, compression='gzip')
        # 保存归一化参数
        for key, val in norm_params.items():
            f.attrs[key] = val
        f.attrs['num_slices'] = normalized_volume.shape[0]
        f.attrs['height'] = normalized_volume.shape[1]
        f.attrs['width'] = normalized_volume.shape[2]
        f.attrs['modality'] = modality_name

    print(f"已生成: {h5_path}")
    return True

def map_series_desc_to_modality(series_desc):
    """将 series_description 映射为模态缩写"""
    if series_desc == "Sagittal T2/STIR":
        return "T2"
    elif series_desc == "Sagittal T1":
        return "T1"
    elif series_desc == "Axial T2":
        return "AX"
    else:
        return None  # 忽略其他类型

def main_from_csv(csv_path, base_image_dir="/data4/whz/rsna/train_images"):
    """
    从 CSV 文件读取 study_id, series_id, series_description，
    处理每个 series 文件夹下的 DICOM 图像并生成 .h5 文件。
    """
    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        study_id = str(row['study_id'])
        series_id = str(row['series_id'])
        series_desc = row['series_description']

        modality = map_series_desc_to_modality(series_desc)
        if modality is None:
            print(f"跳过未知序列描述: {series_desc}")
            continue

        dicom_dir = os.path.join(base_image_dir, study_id, series_id)
        if not os.path.isdir(dicom_dir):
            print(f"路径不存在，跳过: {dicom_dir}")
            continue

        print(f"处理: {study_id}/{series_id} ({series_desc} -> {modality})")
        try:
            success = process_dicom_folder_to_h5(dicom_dir, modality)
            if not success:
                print(f"处理失败: {dicom_dir}")
        except Exception as e:
            print(f"异常: {dicom_dir}, 错误: {e}")

if __name__ == "__main__":
    CSV_PATH = "/data4/whz/rsna/train_series_descriptions.csv"  # 替换为你的 CSV 路径
    main_from_csv(CSV_PATH)