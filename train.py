import torch
import torch.nn as nn
from models.RainbowGPT import RainbowGPT
from models.UltimateModel import UltimateModel
from utils.utils import get_batch, estimate_loss

from dataclasses import dataclass
import time
from datetime import datetime
from tqdm import tqdm


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


torch.manual_seed(1337)

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

model = UtilmateModel(TrainConfig).to(TrainConfig.device)  # 模型实例
# 计算参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量：{total_params}.")
print(f"训练设备：device={TrainConfig.device}")

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=TrainConfig.learning_rate)  # 优化器实例（梯度下降策略，不计算梯度，依赖loss.backward()计算的梯度值）

start_time = time.time()

# 训练
for cur_iter in tqdm(range(TrainConfig.iterations + 1), desc="Trainning"):
    xb, yb = get_batch(train_data, model.ModelStruct.block_size, TrainConfig.batch_size, TrainConfig.device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if cur_iter % TrainConfig.eval_interval == 0:
        all_loss = estimate_loss(train_data, val_data, model, TrainConfig)
        print(f"cur_iter = {cur_iter}, train_loss = {all_loss[0]}, val_loss = {all_loss[1]}")

print(f"训练耗时：{time.time() - start_time} s")
# 预测
with open('outcome/fake_shakespeare_2.txt', 'w', encoding='utf-8') as file:
    file.write(
        decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=TrainConfig.device), 7777)[0].tolist()))
# 仅保存模型参数
torch.save(model.state_dict(),
           "checkpoint/" + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + f"-params-{total_params}" + ".pth")
print(f"模型保存成功.")
