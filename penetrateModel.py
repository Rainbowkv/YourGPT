from dataclasses import dataclass
import torch
import torch.nn as nn
import json

from models import *

# torch.manual_seed(1337)
# 从文件加载i2s字典
with open('tokenizer/en_i2s_tokenizer.json', 'r', encoding='utf-8') as f:
    i2s = json.load(f)

decoder = lambda nums: "".join([i2s[str(num)] for num in nums]) if isinstance(nums, list) else i2s[str(nums)]


@dataclass
# 训练、评估、预测设置
class DemoConfig(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"


if DemoConfig.device == "cuda":
    # torch.cuda.manual_seed(47)
    print("GPU上运行...")

# 获取mini_batch的函数
model = DevModel(DemoConfig).to(DemoConfig.device)  # 模型实例
# model.load_state_dict(torch.load("checkpoint/2024-05-12-05-16-50-params-42783809.pth"))
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型参数量：{total_params}.")
for param in model.parameters():
    print(f"param总大小：{param.size()}, 含有的元素个数：{param.nelement()}, 每个元素大小：{param.element_size()}")
