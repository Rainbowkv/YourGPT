import torch
import torch.nn as nn
from torch.nn import functional as F

# 训练、评估、预测设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(47)
train_data_proportion = 0.9
batch_size = 1024
block_size = 8
iterations = 7000
eval_interval = 300
eval_iters = 200
max_tokens = 500
# 超参数设置
learning_rate = 1e-3
# ------------------------------------------------------

# 加载数据集
input_file_path = "input.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()

# tokenizer生成
chars = sorted(list(set(data)))
vocab_size = len(chars)
s2i = {ch: i for i, ch in enumerate(chars)}
i2s = {i: ch for i, ch in enumerate(chars)}
encoder = lambda s: [s2i[c] for c in s]
decoder = lambda nums: "".join([i2s[num] for num in nums])

# 序列化并划分数据集
input_sequence = torch.tensor(encoder(data), dtype=torch.long)
n = int(train_data_proportion * len(input_sequence))
train_data = input_sequence[:n]
val_data = input_sequence[n:]


# 获取mini_batch的函数
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 模型结构（模型 = 结构 + 参数）
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, target=None):
        logits = self.token_embedding_table(idx)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size).to(device)  # 模型实例
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 优化器实例（梯度下降策略，不计算梯度，依赖loss.backward()计算的梯度值）

# 训练
for cur_iter in range(iterations):
    if cur_iter % eval_interval == 0:
        all_loss = estimate_loss()
        print(f"cur_iter = {cur_iter}, train_loss = {all_loss['train']}, val_loss = {all_loss['eval']}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss)
# 预测
print(decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 500)[0].tolist()))
# 仅保存模型参数
torch.save(model.state_dict(), "bigramModel.pth")
print("模型保存成功")

