import os
import shutil
import pandas as pd
import json


def extract_chexpert_data(
    csv_filepath,
    jsonl_filepath,
    class_filepath="labels.json",
    filepath_prefix="/mnt/K/WHZ/datasets/CheXpert/valid",
    type="train",
    is_output_classname=True,
):
    """
    This function is an example that used to process the metadata csv file of  dataset into our given jsonl format.

    For other datasets, you can refer to this.

    """
    df = pd.read_csv(csv_filepath)
    count = 0

    if is_output_classname:
        categories = df.columns[3:].tolist()
        print(categories)
        categories_data = {"labels_name": categories}

    jsonl_data = []
    for index, row in df.iterrows():
        if type == "train" or type == "val":
            patient_id = row["study_h5_path"].split("/")[-3]
            study_h5_path = row["study_h5_path"].replace(filepath_prefix, type)
            ap_jpg_path = row["ap_jpg_path"].replace(filepath_prefix, type)
            lap_jpg_path = row["lap_jpg_path"].replace(filepath_prefix, type)
            labels = [
                row[column] if pd.notna(row[column]) else 0.0
                for column in df.columns[3:]
            ]
        else:
            patient_id = row["study_h5_path"].split("/")[-3]
            study_h5_path = row["study_h5_path"].replace("test", type)
            ap_jpg_path = row["ap_jpg_path"].replace("test", type)
            lap_jpg_path = row["lap_jpg_path"].replace("test", type)
            labels = [
                row[column] if pd.notna(row[column]) else 0.0
                for column in df.columns[1:]
            ]

        data = {
            "id": patient_id,
            "study_h5_path": study_h5_path,
            "ap_jpg_path": ap_jpg_path,
            "lap_jpg_path": lap_jpg_path,
            "text": "",
            "labels": labels,
        }

        jsonl_data.append(data)

    with open(jsonl_filepath, "w", encoding="utf-8") as jsonl_file:
        for data in jsonl_data:
            jsonl_file.write(json.dumps(data) + "\n")
    if is_output_classname:
        with open(class_filepath, "w", encoding="utf-8") as categories_file:
            json.dump(categories_data, categories_file)

    print("Conversion completed.")

if __name__=='__main__':
    extract_chexpert_data(csv_filepath='/data3/whz/spinedataset/first_XMRI/train_with_process.csv',
                          jsonl_filepath='/data3/whz/SpineModel-main/datasets/spine_train_labels.json',
                          class_filepath="/data3/whz/SpineModel-main/datasets/spine_train_label.json", #可不写具体的路径，只是为了显示数据集中的类别，没什么用
                          filepath_prefix='/data3/whz/spinedataset/first_XMRI/train',
                          type='train')#val