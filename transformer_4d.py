import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# 训练、评估、预测设置
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
torch.manual_seed(1337)
if device == "cuda":
    torch.cuda.manual_seed(47)
train_data_proportion = 0.9
# 超参数设置
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
dropout = 0.2  # 正则化
# ------------------------------------------------------

# 加载数据集
input_file_path = "data/input.txt"
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


class MultiAttentionHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, n_embd * 3, bias=False)
        # 解码时，block_size是t的上限，不要误以为相等。预测其实相当于解码。
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.proj = nn.Linear(n_embd, n_embd, bias=False)  # 投影回残差路径的层
        self.W_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, X):  # (B, T, C) -> (B, T, head_size) 因此只要单头时，head_size必须=C否则无法输入lm_head
        """
        四维矩阵相乘，并行处理所有头。
        :param X: 单个头的输入X(b, t, c)  所有注意力头都拿到完整的真实输入X，不是平分！
        :return: 单个头的输出OUT(b, t, head_size)  所有注意力头拼接真实输出OUT: head_size = c//head_nums
        """
        b, t, c = X.shape  # 无论单头还是多头都接受完整的X输入，这样多头才有明显的优势，如果不是完整的输入，多头就退化成单头的批处理模式了。
        K, Q, V = self.c_attn(X).split(n_embd, dim=2)  # (B, T, C)
        K = K.view(b, t, num_heads, n_embd // num_heads).transpose(1, 2)
        Q = Q.view(b, t, num_heads, n_embd // num_heads).transpose(1, 2)
        V = V.view(b, t, num_heads, n_embd // num_heads).transpose(1, 2)

        W = Q @ K.transpose(-2, -1) * c ** -0.5  # (B, nh, T, T)
        # 解码时，block_size是t的上限，不要误以为相等。预测其实相当于解码。
        W = W.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
        W = F.softmax(W, dim=-1)
        W = self.W_dropout(W)
        OUT = W @ V  # (B, nh, T, hs)
        OUT = OUT.transpose(1, 2).contiguous().view(b, t, c)
        OUT = self.proj(OUT)
        OUT = self.out_dropout(OUT)
        return OUT


class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),  # 这里本来是n_embd, n_embd，这里是为了复现A I A Y N论文
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),  # 投影回残差路径的层
            nn.Dropout(dropout)  # 正则化
        )

    def forward(self, X):
        return self.net(X)


class Block(nn.Module):

    def __init__(self):
        super().__init__()
        self.sa = MultiAttentionHead()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, X):
        X = X + self.sa(self.ln1(X))  # 优化：加入残差，层规范（类似于BatchNorm)
        X = X + self.ffwd(self.ln2(X))
        return X


# 模型结构（模型 = 结构 + 参数）
class TransformerModel(nn.Module):

    def __init__(self):  # vocal_size是上面的全局参数，不需要传入构造函数。
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # (B, T, vocab_size)->(B, T, n_embd)
        self.pos_embedding_table = nn.Embedding(block_size, n_embd)  # pos和token编码长度必须一致，下面要加起来。

        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_blocks)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.net = nn.Sequential(  # 构建网络
            self.blocks,
            self.ln_f
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)  # 这里不再像二元模型时 编码维度==词汇量大小，因为解码不那么容易，需要明确的中间层。

    def forward(self, idx, target=None):
        tok_emd = self.token_embedding_table(idx)  # tok_emd.shape = (B, T, C), C = n_embd
        pos_emd = self.pos_embedding_table(
            torch.arange(idx.shape[1], device=device))  # 不能写block_size, 它是T=idx.shape[1]的上限
        X = tok_emd + pos_emd  # (T, C) + (B, T, C)

        X = self.net(X)  # 进入网络

        logits = self.lm_head(X)  # (B, T, n_embd)->(B, T, vocab_size)

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
            content = idx[:, -block_size:]  # 有位置嵌入表后，输入的长度必须限制，否则查表会越界。ps:刚好可以输入最大值的上下文长度block，而不是block-1
            logits, loss = self(content)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = TransformerModel().to(device)  # 模型实例
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # 优化器实例（梯度下降策略，不计算梯度，依赖loss.backward()计算的梯度值）

start_time = time.time()
# 训练
for cur_iter in range(iterations + 1):
    if cur_iter % eval_interval == 0:
        all_loss = estimate_loss()
        print(f"cur_iter = {cur_iter}, train_loss = {all_loss['train']}, val_loss = {all_loss['eval']}")
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"训练耗时：{time.time()-start_time} s")
# 预测
print(decoder(model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 500)[0].tolist()))
# 仅保存模型参数
torch.save(model.state_dict(), "trisa_ffwd_res_layerNorm_Model.pth")
print("模型保存成功")
