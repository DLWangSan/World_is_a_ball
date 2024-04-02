import re
import json

file = "../data/annotations/trainval.txt"
with open(file, "r") as f:
    lines = f.readlines()

# 使用字典来存储提取的键值对
result_dict = {}

# 分割文本为行，并提取所需信息
for line in lines:
    parts = line.split()
    if len(parts) >= 2:
        # 使用正则表达式移除名称中的数字部分
        name_part = re.sub(r'_[0-9]+$', '', parts[0])
        key = int(parts[1]) - 1
        result_dict[key] = name_part

# 将字典保存为JSON文件
with open("./extracted_dict.json", "w") as json_file:
    json.dump(result_dict, json_file, indent=4)

