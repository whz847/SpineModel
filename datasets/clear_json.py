#
# """删除CheXpert-v1.0-small/"""
# import json
#
# input_file = "/mnt/K/WHZ/SpineModel-main/datasets/train_labels.json"   # 替换为你的输入文件路径
# output_file = "/mnt/K/WHZ/SpineModel-main/datasets/train_labelss.json" # 替换为你想保存的输出文件路径
#
# with open(input_file, 'r', encoding='utf-8') as fin, \
#      open(output_file, 'w', encoding='utf-8') as fout:
#
#     for line in fin:
#         line = line.strip()
#         if not line:
#             continue  # 跳过空行
#         data = json.loads(line)
#         # 移除 filepath 中的 "CheXpert-v1.0-small/"
#         data["filepath"] = data["filepath"].replace("CheXpert-v1.0-small/", "", 1)
#         # 写回新行（保持 JSONL 格式）
#         fout.write(json.dumps(data, ensure_ascii=False) + "\n")


"""将-1.0替换为0.0"""
import json

input_file = "/mnt/K/WHZ/SpineModel-main/datasets/valid_labels.json"   # 替换为你的输入文件路径
output_file = "/mnt/K/WHZ/SpineModel-main/datasets/valid_labelss.json" # 替换为你想保存的输出文件路径

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue  # 跳过空行
        data = json.loads(line)

        # 将 labels 中的 -1.0 替换为 0.0
        data["labels"] = [0.0 if x == -1.0 else x for x in data["labels"]]

        # 写回一行（保持 JSONL 格式）
        fout.write(json.dumps(data, ensure_ascii=False) + "\n")