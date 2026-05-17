import os
import numpy as np
import pydicom
from pydicom.misc import is_dicom
from PIL import Image

def dicom_to_jpg_spine_xray(dicom_path, output_jpg_path):
    """
    将脊柱 X 光 DICOM 转为 JPG，适配临床显示习惯
    """
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    # === 1. 获取窗宽窗位 ===
    ww, wc = None, None
    if 'WindowWidth' in ds and 'WindowCenter' in ds:
        try:
            ww = float(ds.WindowWidth)
            wc = float(ds.WindowCenter)
        except (TypeError, ValueError):
            pass

    # 若无有效 WW/WL，使用脊柱默认值
    if ww is None or wc is None:
        ww, wc = 2000.0, 400.0  # 脊柱 X 光常用值

    # === 2. 应用窗宽窗位 ===
    img_min = wc - ww / 2.0
    img_max = wc + ww / 2.0
    img = np.clip(img, img_min, img_max)

    # === 3. 归一化到 [0, 255] ===
    img = (img - img_min) / (img_max - img_min) * 255.0
    img = np.clip(img, 0, 255)  # 再次确保范围
    img_uint8 = img.astype(np.uint8)

    # === 4. 处理 MONOCHROME1（X 光常见）===
    photometric = ds.get('PhotometricInterpretation', 'MONOCHROME2')
    if photometric == 'MONOCHROME1':
        img_uint8 = 255 - img_uint8  # 反转：骨骼变白

    # === 5. 保存为 JPG ===
    Image.fromarray(img_uint8).save(output_jpg_path, quality=95)
    print(f"已保存脊柱 X 光图像: {output_jpg_path}")

def process_spine_aplate(root_dir="AIS"):
    """
    处理 AIS/<ID>/LUMBAR_APLAP/AP 和 LAP 中的脊柱 X 光 DICOM
    """
    views = ["AP", "LAP"]

    for patient_id in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient_id)
        aplats_path = os.path.join(patient_path, "LUMBAR_APLAP")

        if not os.path.isdir(aplats_path):
            continue

        print(f"\n处理患者 {patient_id} 的脊柱 X 光 (AP/LAP)...")

        for view in views:
            view_dir = os.path.join(aplats_path, view)
            if not os.path.isdir(view_dir):
                print(f" 跳过缺失视图: {view}")
                continue

            # 查找 DICOM 文件（无后缀，仅文件）
            dicom_files = []
            for f in os.listdir(view_dir):
                fp = os.path.join(view_dir, f)
                if os.path.isfile(fp) and is_dicom(fp):
                    dicom_files.append(fp)

            if not dicom_files:
                print(f"{view_dir} 中未找到 DICOM 文件")
                continue
            elif len(dicom_files) > 1:
                print(f"{view_dir} 中有多个文件，仅处理第一个")

            dicom_path = dicom_files[0]
            jpg_path = os.path.join(view_dir, f"{view}.jpg")

            try:
                dicom_to_jpg_spine_xray(dicom_path, jpg_path)
            except Exception as e:
                print(f" 处理 {view} 失败: {e}")

    print("\n所有脊柱 X 光 JPG 转换完成！")

# ======================
if __name__ == "__main__":
    AIS_ROOT = "/data3/whz/spinedataset/first_XMRI/valid"  # 可替换为绝对路径
    process_spine_aplate(AIS_ROOT)