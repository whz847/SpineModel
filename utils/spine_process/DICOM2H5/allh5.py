import os
import h5py
import numpy as np

def merge_modalities_to_study_h5(lumbar_dir, modalities=("AX", "FS", "T1", "T2")):
    """
    将 LUMBAR_MRI/ 下的 AX.h5, T2.h5 等合并为一个 study.h5
    """
    output_path = os.path.join(lumbar_dir, "study.h5")
    
    with h5py.File(output_path, 'w') as f_out:
        for mod in modalities:
            h5_path = os.path.join(lumbar_dir, mod, f"{mod}.h5")
            if not os.path.isfile(h5_path):
                print(f"跳过缺失模态: {mod}")
                continue

            try:
                with h5py.File(h5_path, 'r') as f_in:
                    # 创建模态分组
                    mod_group = f_out.create_group(mod)

                    # 复制 slices 数据集
                    slices_data = f_in['slices']
                    mod_group.create_dataset(
                        'slices',
                        data=slices_data[:],  # 加载到内存再写入（也可用更高效方式）
                        dtype=slices_data.dtype,
                        compression='gzip',
                        compression_opts=9
                    )

                    # 复制属性
                    for key, val in f_in.attrs.items():
                        mod_group.attrs[key] = val

                print(f"合并模态: {mod}")
            except Exception as e:
                print(f"合并 {mod} 失败: {e}")

    print(f" 已生成整合文件: {output_path}")

def main_merge_all(root_dir="AIS"):
    """
    遍历所有患者，合并其 LUMBAR_MRI 下的模态 H5 文件
    """
    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        lumbar_path = os.path.join(patient_path, "LUMBAR_MRI")
        
        if not os.path.isdir(lumbar_path):
            continue

        print(f"\n合并患者 {patient_id} 的多模态数据...")
        merge_modalities_to_study_h5(lumbar_path)

if __name__ == "__main__":
    AIS_ROOT = "/data3/whz/spinedataset/first_XMRI/train"
    main_merge_all(AIS_ROOT)