import torch
import torch.nn as nn
from models.RainbowGPT import RainbowGPT

from dataclasses import dataclass
import time
from datetime import datetime
from tqdm import tqdm

torch.manual_seed(1337)

# 加载数据集
input_file_path = "data/input.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# tokenizer生成
chars = sorted(list(set(data)))
vocab_size = len(chars)


@dataclass
# 训练、评估、预测设置
class GPTConfig(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = len(chars)
    # 超参数设置
    train_data_proportion = 0.9
    batch_size = 64
    block_size = 256
    n_embd = 384
    num_heads = 6
    iterations = 5000
    eval_interval = 500
    eval_iters = 200
    max_tokens = 500
    learning_rate = 3e-4
    n_blocks = 6
    att_dropout = 0.2  # 正则化
    res_dropout = 0.2
    fw_dropout = 0.2


s2i = {ch: i for i, ch in enumerate(chars)}
i2s = {i: ch for i, ch in enumerate(chars)}
encoder = lambda s: [s2i[c] for c in s]
decoder = lambda nums: "".join([i2s[num] for num in nums])

# 序列化并划分数据集
input_sequence = torch.tensor(encoder(data), dtype=torch.long)
n = int(GPTConfig.train_data_proportion * len(input_sequence))
train_data = input_sequence[:n]
val_data = input_sequence[n:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - GPTConfig.block_size, (GPTConfig.batch_size,))
    x = torch.stack([data[i:i + GPTConfig.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + GPTConfig.block_size + 1] for i in ix])
    return x.to(GPTConfig.device), y.to(GPTConfig.device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "eval"]:
        losses = torch.zeros(GPTConfig.eval_iters)
        for k in range(GPTConfig.eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = RainbowGPT(GPTConfig).to(GPTConfig.device)  # 模型实例
# 计算参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量：{total_params}.")
print(f"训练设备：device={GPTConfig.device}")

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=GPTConfig.learning_rate)  # 优化器实例（梯度下降策略，不计算梯度，依赖loss.backward()计算的梯度值）
start_time = time.time()

# 训练
for cur_iter in tqdm(range(GPTConfig.iterations + 1), desc="Trainning"):
    if cur_iter % GPTConfig.eval_interval == 0:
        all_loss = estimate_loss()
        print(f"cur_iter = {cur_iter}, train_loss = {all_loss['train']}, val_loss = {all_loss['eval']}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"训练耗时：{time.time() - start_time} s")
# 预测
with open('outcome/fake_shakespeare_2.txt', 'w', encoding='utf-8') as file:
    file.write(
        decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=GPTConfig.device), 7777)[0].tolist()))
# 仅保存模型参数
torch.save(model.state_dict(), datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + ".pth")
print(f"模型保存成功, 参数量：{model.parameters()}.")
