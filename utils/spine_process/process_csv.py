import os
import pandas as pd

def read_test_csv_safe(test_csv_path):
    """安全读取 test.csv，并移除 Unnamed 列"""
    for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
        try:
            df = pd.read_csv(test_csv_path, dtype=str, encoding=encoding)
            # 移除 Unnamed 列（列名以 'Unnamed:' 开头）
            df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
            print(f"使用编码 '{encoding}' 成功读取 test.csv，剩余列: {list(df.columns)}")
            return df
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"无法读取 {test_csv_path}，请检查文件编码或格式")

def extract_patient_id_from_path(study_h5_path):
    parts = study_h5_path.replace('\\', '/').split('/')
    return parts[-3] if len(parts) >= 3 else None

def merge_csv_with_test(test_csv_path, input_csv_path, output_csv_path):
    # 1. 安全读取并清理 test.csv
    test_df = read_test_csv_safe(test_csv_path)
    if 'case_id_raw' not in test_df.columns:
        raise ValueError("test.csv 必须包含 'case_id_raw' 列")

    # 构建映射字典（只包含有效列）
    test_dict = {}
    for _, row in test_df.iterrows():
        case_id = str(row['case_id_raw']).strip()
        if case_id:
            # 只保留非 NaN 且列名有效的字段（可选）
            test_dict[case_id] = row.to_dict()

    print(f"从 {test_csv_path} 加载了 {len(test_dict)} 条有效记录")

    # 2. 读取 train/valid.csv（假设是 UTF-8）
    input_df = pd.read_csv(input_csv_path, dtype=str)

    # 3. 合并
    merged_rows = []
    matched_count = 0
    for _, row in input_df.iterrows():
        patient_id = extract_patient_id_from_path(row['study_h5_path'])
        new_row = row.to_dict()

        if patient_id and patient_id in test_dict:
            test_row = test_dict[patient_id]
            for key, val in test_row.items():
                if key != 'case_id_raw' and key not in new_row:  # 可选：跳过 case_id_raw
                    new_row[key] = val
            matched_count += 1

        merged_rows.append(new_row)

    # 4. 保存（用 utf-8-sig 避免 Excel 乱码）
    merged_df = pd.DataFrame(merged_rows)
    merged_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"合并完成: {matched_count}/{len(input_df)} 行匹配，输出至 {output_csv_path}")

# ======================
if __name__ == "__main__":
    TEST_CSV_PATH = r"/data3/whz/spinedataset/first_XMRI/all.csv"
    CSV_DIR = r"/data3/whz/spinedataset/first_XMRI/"

    merge_csv_with_test(
        test_csv_path=TEST_CSV_PATH,
        input_csv_path=os.path.join(CSV_DIR, "train.csv"),
        output_csv_path=os.path.join(CSV_DIR, "train_with_process.csv")
    )

    merge_csv_with_test(
        test_csv_path=TEST_CSV_PATH,
        input_csv_path=os.path.join(CSV_DIR, "valid.csv"),
        output_csv_path=os.path.join(CSV_DIR, "valid_with_process.csv")
    )