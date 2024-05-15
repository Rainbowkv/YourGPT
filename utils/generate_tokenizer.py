import json
import os.path

file = 'tokenizer/i2s_tokenizer.json'
if not os.path.exists(os.path.dirname(file)):
    os.makedirs(os.path.dirname(file))

# 加载数据集
input_file_path = "data/input.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# tokenizer生成
chars = sorted(list(set(data)))
i2s = {i: ch for i, ch in enumerate(chars)}

# 保存i2s字典到JSON文件
with open(file, 'w', encoding='utf-8') as f:
    json.dump(i2s, f, ensure_ascii=False, indent=4)
