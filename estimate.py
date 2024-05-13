import torch
from dataclasses import dataclass

from models import *
from utils import *


@dataclass
# 评估设置
class EvalConfig:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 65
    batch_size = 128
    train_data_proportion = 0.9
    eval_interval = 500
    eval_iters = 200


torch.manual_seed(1337)

# 加载数据集
input_file_path = "data/input.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# tokenizer生成
chars = sorted(list(set(data)))

s2i = {ch: i for i, ch in enumerate(chars)}
encoder = lambda s: [s2i[c] for c in s]

# 序列化并划分数据集
input_sequence = torch.tensor(encoder(data), dtype=torch.long)
n = int(EvalConfig.train_data_proportion * len(input_sequence))
train_data = input_sequence[:n]
val_data = input_sequence[n:]

model = UltimateModel(EvalConfig).to(EvalConfig.device)  # 模型实例
model.load_state_dict(torch.load("checkpoint/2024-05-12-23-23-08-params-10873409.pth"))
# 计算参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量：{total_params}.")
print(f"评估设备：device={EvalConfig.device}")

all_loss = estimate_loss(train_data, val_data, model, EvalConfig)
print(f"train_loss = {all_loss[0]}, val_loss = {all_loss[1]}")
