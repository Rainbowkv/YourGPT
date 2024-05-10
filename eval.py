from dataclasses import dataclass
import torch
import torch.nn as nn

from models.TransformerModel import TransformerModel

# from models.RainbowGPT import RainbowGPT

torch.manual_seed(1337)

# 加载数据集
input_file_path = "data/input.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# tokenizer生成
chars = sorted(list(set(data)))
i2s = {i: ch for i, ch in enumerate(chars)}
decoder = lambda nums: "".join([i2s[num] for num in nums])


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
    n_blocks = 3
    att_dropout = 0.2  # 正则化
    res_dropout = 0.2
    fw_dropout = 0.2


if GPTConfig.device == "cuda":
    torch.cuda.manual_seed(47)

# 获取mini_batch的函数
model = TransformerModel(GPTConfig).to(GPTConfig.device)  # 模型实例
model.load_state_dict(torch.load("checkpoint/trisa_ffwd_res_layerNorm_Model.pth"))

# 预测
with open('outcome/fake_shakespeare.txt', 'w', encoding='utf-8') as file:
    file.write(
        decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=GPTConfig.device), 7777)[0].tolist()))
