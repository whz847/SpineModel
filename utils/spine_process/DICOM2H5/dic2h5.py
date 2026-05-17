import os
import re
import numpy as np
import pydicom
import h5py
from pydicom.misc import is_dicom


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
    # elif modality == "AX":
    #     # Z-score 归一化
    #     mean = np.mean(volume)
    #     std = np.std(volume)
    #     if std == 0:
    #         normalized = np.zeros_like(volume)
    #     else:
    #         normalized = (volume - mean) / std
    #     return normalized, {
    #         'norm_method': 'z_score',
    #         'mean': mean,
    #         'std': std
    #     }
    # else:
    #     # 默认 fallback：Z-score
    #     mean = np.mean(volume)
    #     std = np.std(volume)
    #     if std == 0:
    #         normalized = np.zeros_like(volume)
    #     else:
    #         normalized = (volume - mean) / std
    #     return normalized, {
    #         'norm_method': 'z_score_fallback',
    #         'mean': mean,
    #         'std': std
    #     }


def process_dicom_folder_to_h5(dicom_dir, modality_name, bg_threshold=0.0):
    """
    处理一个 DICOM 文件夹（如 AX/），生成 {modality}.h5
    """
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
    h5_path = os.path.join(dicom_dir, f"{modality_name}.h5")
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


# ====== 主流程 ======

def main(root_dir="AIS"):
    """
    遍历 AIS/ 下所有病例，处理 LUMBAR_MRI 中的 AX/FS/T1/T2
    """
    target_modalities = {"AX", "FS", "T1", "T2"}

    if not os.path.isdir(root_dir):
        raise ValueError(f"根目录不存在: {root_dir}")

    # 遍历 AIS/ 下的每个病人文件夹（如 2845803PA13X）
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        lumbar_path = os.path.join(patient_path, "LUMBAR_MRI")

        if not os.path.isdir(lumbar_path):
            continue  # 跳过没有 LUMBAR_MRI 的

        print(f"\n处理患者: {patient_id}")

        # 遍历 LUMBAR_MRI 下的所有子文件夹
        for modality in os.listdir(lumbar_path):
            modality_path = os.path.join(lumbar_path, modality)
            if not os.path.isdir(modality_path):
                continue
            if modality not in target_modalities:
                continue

            print(f" 处理模态: {modality}")
            try:
                process_dicom_folder_to_h5(modality_path, modality)
            except Exception as e:
                print(f" 处理 {modality_path} 时出错: {e}")

    print("\n所有任务完成！")


# ======================
# 运行入口
# ======================
if __name__ == "__main__":
    # 修改为你实际的 AIS 根路径
    AIS_ROOT = "/data3/whz/spinedataset/first_XMRI/valid"  # 绝对路径ut
    main(AIS_ROOT)