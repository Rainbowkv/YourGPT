import torch
from transformer_4d import TransformerModel

# 训练、评估、预测设置
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.manual_seed(47)

# 加载数据集
input_file_path = "data/input.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# tokenizer生成
chars = sorted(list(set(data)))
i2s = {i: ch for i, ch in enumerate(chars)}
decoder = lambda nums: "".join([i2s[num] for num in nums])
model = TransformerModel().to(device)  # 模型实例
model.load_state_dict(torch.load("checkpoint/trisa_ffwd_res_layerNorm_Model.pth"))

# 预测
print(decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 500)[0].tolist()))
