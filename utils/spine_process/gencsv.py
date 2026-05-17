# import os
# import csv

# def generate_csv_for_split(aix_root="AIS", split_name="train", csv_output_dir="AIS/csv"):
    # """
    # 为 train 或 valid 分割生成 CSV 文件，包含三列路径：
      # - study.h5 路径（LUMBAR_MRI/study.h5）
      # - AP.jpg 路径（LUMBAR_APLAP/AP/AP.jpg）
      # - LAP.jpg 路径（LUMBAR_APLAP/LAP/LAP.jpg）

    # 所有路径以 "AIS/..." 开头（相对路径，便于跨机器共享）

    # Args:
        # aix_root (str): AIS 根目录，默认 "AIS"
        # split_name (str): 子集名称，如 "train" 或 "valid"
        # csv_output_dir (str): CSV 文件保存目录
    # """
    # # 构建输入目录：AIS/train 或 AIS/valid
    # split_dir = os.path.join(aix_root, split_name)
    # if not os.path.isdir(split_dir):
        # print(f"跳过 {split_name}：目录不存在 {split_dir}")
        # return

    # # 确保 CSV 输出目录存在
    # os.makedirs(csv_output_dir, exist_ok=True)
    # csv_path = os.path.join(csv_output_dir, f"{split_name}.csv")

    # rows_written = 0
    # with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        # writer = csv.writer(f)
        # writer.writerow(["study_h5_path", "ap_jpg_path", "lap_jpg_path"])

        # for patient_id in os.listdir(split_dir):
            # patient_path = os.path.join(split_dir, patient_id)
            # if not os.path.isdir(patient_path):
                # continue

            # # 构造相对路径（从 AIS 开始）
            # study_h5_rel = os.path.join(split_name, patient_id, "LUMBAR_MRI", "study.h5")
            # ap_jpg_rel   = os.path.join(split_name, patient_id, "LUMBAR_APLAP", "AP", "AP.jpg")
            # lap_jpg_rel  = os.path.join(split_name, patient_id, "LUMBAR_APLAP", "LAP", "LAP.jpg")

            # # 转为绝对路径用于检查是否存在
            # study_h5_abs = os.path.join(aix_root, study_h5_rel)
            # ap_jpg_abs   = os.path.join(aix_root, ap_jpg_rel)
            # lap_jpg_abs  = os.path.join(aix_root, lap_jpg_rel)

            # if os.path.isfile(study_h5_abs) and os.path.isfile(ap_jpg_abs) and os.path.isfile(lap_jpg_abs):
                # # 写入 **相对路径（从 AIS 开始）**
                # writer.writerow([study_h5_rel, ap_jpg_rel, lap_jpg_rel])
                # rows_written += 1
            # else:
                # missing = []
                # if not os.path.isfile(study_h5_abs): missing.append("study.h5")
                # if not os.path.isfile(ap_jpg_abs):   missing.append("AP.jpg")
                # if not os.path.isfile(lap_jpg_abs):  missing.append("LAT.jpg")
                # print(f"{patient_id} 缺失文件: {', '.join(missing)}")

    # print(f"已生成 {split_name}.csv，共 {rows_written} 行，保存至: {csv_path}")


# def main():
    # AIX_ROOT = "C:\\Users\\dell\\Desktop\\spine\\4.脊柱侧弯AIS"
    # CSV_OUTPUT_DIR = "C:\\Users\\dell\\Desktop\\spine\\4.脊柱侧弯AIS"  # 你可以修改这个路径！

    # generate_csv_for_split(AIX_ROOT, "train", CSV_OUTPUT_DIR)
    # generate_csv_for_split(AIX_ROOT, "valid", CSV_OUTPUT_DIR)


# if __name__ == "__main__":
    # main()

import os
import csv


def generate_csv_for_split(aix_root="AIS", split_name="train", csv_output_dir="AIS/csv"):
    split_dir = os.path.join(aix_root, split_name)
    if not os.path.isdir(split_dir):
        print(f"跳过 {split_name}：目录不存在 {split_dir}")
        return

    os.makedirs(csv_output_dir, exist_ok=True)
    csv_path = os.path.join(csv_output_dir, f"{split_name}.csv")

    rows_written = 0
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["study_h5_path", "ap_jpg_path", "lap_jpg_path"])

        for patient_id in os.listdir(split_dir):
            patient_path = os.path.join(split_dir, patient_id)
            if not os.path.isdir(patient_path):
                continue

            # 构建相对路径（这些路径将写入CSV）
            study_h5_rel = f"first_XMRI/{split_name}/{patient_id}/LUMBAR_MRI/study.h5"
            ap_jpg_rel = f"first_XMRI/{split_name}/{patient_id}/LUMBAR_APLAP/AP/AP.jpg"
            lap_jpg_rel = f"first_XMRI/{split_name}/{patient_id}/LUMBAR_APLAP/LAP/LAP.jpg"

            # 构建绝对路径用于检查文件是否存在
            study_h5_abs = os.path.join(aix_root, split_name, patient_id, "LUMBAR_MRI", "study.h5")
            ap_jpg_abs = os.path.join(aix_root, split_name, patient_id, "LUMBAR_APLAP", "AP", "AP.jpg")
            lap_jpg_abs = os.path.join(aix_root, split_name, patient_id, "LUMBAR_APLAP", "LAP", "LAP.jpg")

            # 初始化CSV行数据
            csv_row = [study_h5_rel, ap_jpg_rel, lap_jpg_rel]
            missing = []  # 记录缺失的文件，仅用于打印信息

            # 检查每个文件是否存在，如果不存在则对应位置设为"NULL"
            if not os.path.isfile(study_h5_abs):
                csv_row[0] = "NULL"  # 将study_h5路径设为NULL
                missing.append("study.h5")

            if not os.path.isfile(ap_jpg_abs):
                csv_row[1] = "NULL"  # 将ap_jpg路径设为NULL
                missing.append("AP.jpg")

            if not os.path.isfile(lap_jpg_abs):
                csv_row[2] = "NULL"  # 将lap_jpg路径设为NULL
                missing.append("LAP.jpg")

            # 写入CSV行（无论文件是否完整）
            writer.writerow(csv_row)
            rows_written += 1

            # 如果有文件缺失，打印信息（可选）
            if missing:
                print(f"{patient_id} 缺失: {', '.join(missing)}，在CSV中标记为NULL")

    print(f"已生成 {split_name}.csv，共 {rows_written} 行，保存至: {csv_path}")


def main():
    AIX_ROOT = "/data3/whz/spinedataset/first_XMRI" #写到train、valid目录的上一级
    CSV_OUTPUT_DIR = "/data3/whz/spinedataset/first_XMRI"

    generate_csv_for_split(AIX_ROOT, "train", CSV_OUTPUT_DIR)
    generate_csv_for_split(AIX_ROOT, "valid", CSV_OUTPUT_DIR)


if __name__ == "__main__":
    main()