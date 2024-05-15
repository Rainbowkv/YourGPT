from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import TrainDataSet
from models import UltimateModel


@dataclass
# 训练、评估、预测设置
class TrainConfig(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 65
    # 训练设置
    train_data_proportion = 0.9
    batch_size = 64
    iterations = 5000
    learning_rate = 3e-4
    eval_interval = 500
    eval_iters = 200
    max_tokens = 500


model = UltimateModel(TrainConfig)
# 加载数据集
input_file_path = "data/input.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# tokenizer生成
chars = sorted(list(set(data)))

s2i = {ch: i for i, ch in enumerate(chars)}
i2s = {i: ch for i, ch in enumerate(chars)}
encoder = lambda s: [s2i[c] for c in s]
decoder = lambda nums: "".join([i2s[num] for num in nums])

# 序列化并划分数据集
input_sequence = torch.tensor(encoder(data), dtype=torch.long)
n = int(TrainConfig.train_data_proportion * len(input_sequence))
train_data = input_sequence[:n]
val_data = input_sequence[n:]

train_data = TrainDataSet(train_data, model.ModelStruct.block_size)
train_loader = DataLoader(train_data, batch_size=64)
for x, y in train_loader:
    print(x)
    print(y)
    exit()