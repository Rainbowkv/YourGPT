from dataclasses import dataclass
import torch
import torch.nn as nn
import json

from models.TransformerModel import TransformerModel
from models.RainbowGPT import RainbowGPT


# torch.manual_seed(1337)

# 从文件加载i2s字典
with open('tokenizer/i2s_tokenizer.json', 'r', encoding='utf-8') as f:
    i2s = json.load(f)

decoder = lambda nums: "".join([i2s[str(num)] for num in nums]) if isinstance(nums, list) else i2s[str(nums)]
# 此时可以像之前一样使用decoder
print(decoder(45))  # 假设i2s.json文件中含有合适的映射

@dataclass
# 训练、评估、预测设置
class EvalConfig(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    eval_interval = 500
    eval_iters = 200
    max_tokens = 10000


if EvalConfig.device == "cuda":
    # torch.cuda.manual_seed(47)
    print("GPU上运行...")

# 获取mini_batch的函数
model = RainbowGPT(EvalConfig).to(EvalConfig.device)  # 模型实例
model.load_state_dict(torch.load("checkpoint/2024-05-11-00-30-09.pth"))
# model.load_state_dict(torch.load("checkpoint/trisa_ffwd_res_layerNorm_Model.pth"))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量：{total_params}.")

# 即时打印，不保存
input = torch.zeros((1, 1), dtype=torch.long, device=EvalConfig.device)
model.generate_and_print_token(input, EvalConfig.max_tokens, decoder)

# 写入文件
# with open('outcome/fake_shakespeare_2.txt', 'w', encoding='utf-8') as file:
#     file.write(
#         decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=GPTConfig.device), 7777)[0].tolist()))
#
# print("预测完成.")

