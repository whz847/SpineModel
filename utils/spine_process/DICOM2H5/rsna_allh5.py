import os
import h5py
import numpy as np


def merge_series_h5_to_study(study_dir, target_modalities=("AX", "FS", "T1", "T2")):
    """
    从 study_dir 下的所有 series_id 文件夹中收集 AX.h5, T1.h5, T2.h5，
    用 AX 代替 FS，生成 study.h5。
    """
    # Step 0: 检查是否已经存在 study.h5 文件，如果存在则跳过
    output_path = os.path.join(study_dir, "study.h5")
    if os.path.exists(output_path):
        print(f"  跳过已处理的文件: {output_path}")
        return True  # 返回 True 表示已完成（虽然跳过了）
    
    # Step 1: 扫描所有 series_id 子目录，收集各模态的 .h5 路径
    found_modalities = {}
    for series_id in os.listdir(study_dir):
        series_path = os.path.join(study_dir, series_id)
        if not os.path.isdir(series_path):
            continue

        for mod in ["AX", "T1", "T2"]:
            h5_file = os.path.join(series_path, f"{mod}.h5")
            if os.path.isfile(h5_file) and mod not in found_modalities:
                found_modalities[mod] = h5_file

    # Step 2: 如果 AX 存在，且 FS 不存在，则用 AX 作为 FS
    if "AX" in found_modalities and "FS" not in found_modalities:
        found_modalities["FS"] = found_modalities["AX"]  # 复用路径，后面会复制数据

    # Step 3: 创建 study.h5
    with h5py.File(output_path, 'w') as f_out:
        for mod in target_modalities:
            if mod not in found_modalities:
                print(f"  跳过缺失模态: {mod} (无对应 .h5 文件)")
                continue

            h5_path = found_modalities[mod]
            try:
                with h5py.File(h5_path, 'r') as f_in:
                    # 创建模态组
                    mod_group = f_out.create_group(mod)

                    # 复制 slices 数据
                    slices_data = f_in['slices'][:]
                    mod_group.create_dataset(
                        'slices',
                        data=slices_data,
                        dtype=slices_data.dtype,
                        compression='gzip'
                    )

                    # 复制属性
                    for key, val in f_in.attrs.items():
                        mod_group.attrs[key] = val

                print(f"  合并模态: {mod} <- {h5_path}")
            except Exception as e:
                print(f"  合并 {mod} 失败 ({h5_path}): {e}")

    print(f"  已生成: {output_path}")


def main_merge_studies(root_dir="/data4/whz/rsna/train_images"):
    """
    遍历所有 study_id 文件夹，合并其下的 series .h5 文件为 study.h5
    """
    study_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    total = len(study_ids)
    for i, study_id in enumerate(study_ids, 1):
        study_path = os.path.join(root_dir, study_id)
        print(f"\n[{i}/{total}] 处理 study_id: {study_id}")
        merge_series_h5_to_study(study_path)


if __name__ == "__main__":
    main_merge_studies()