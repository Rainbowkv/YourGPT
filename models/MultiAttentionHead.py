import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiAttentionHead(nn.Module):

    def __init__(self, config):
        assert config.n_embd % config.num_heads == 0, \
            f"错误：n_embd 必须能被 num_heads 整除。当前 n_embd: {config.n_embd}, num_heads: {config.num_heads}"
        super().__init__()
        self.num_heads = config.num_heads
        self.head_size = config.n_embd // config.num_heads
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(self.n_embd, self.n_embd * 3, bias=False)
        # 解码时，block_size是t的上限，不要误以为相等。预测其实相当于解码。
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)  # 投影回残差路径的层
        self.W_dropout = nn.Dropout(config.att_dropout)
        self.out_dropout = nn.Dropout(config.res_dropout)

    def forward(self, X):  # (B, T, C) -> (B, T, head_size) 因此只要单头时，head_size必须=C否则无法输入lm_head
        """
        四维矩阵相乘，并行处理所有头。
        :param X: 单个头的输入X(b, t, c)  所有注意力头都拿到完整的真实输入X，不是平分！
        :return: 单个头的输出OUT(b, t, head_size)  所有注意力头拼接真实输出OUT: head_size = c//head_nums
        """
        b, t, c = X.shape  # 无论单头还是多头都接受完整的X输入，这样多头才有明显的优势，如果不是完整的输入，多头就退化成单头的批处理模式了。
        K, Q, V = self.c_attn(X).split(self.n_embd, dim=2)  # (B, T, C)
        K = K.view(b, t, self.num_heads, self.n_embd // self.num_heads).transpose(1, 2)
        Q = Q.view(b, t, self.num_heads, self.n_embd // self.num_heads).transpose(1, 2)
        V = V.view(b, t, self.num_heads, self.n_embd // self.num_heads).transpose(1, 2)

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
