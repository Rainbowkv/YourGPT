import torch
import torch.nn as nn
from torch.nn import functional as F

from .DecoderBlock import DecoderBlock


class HugeModel(nn.Module):
    class ModelStruct:
        precision = torch.float32
        vocab_size = 65
        block_size = 256
        n_embd = 768
        n_blocks = 6
        num_heads = 6
        att_dropout = 0.25
        res_dropout = 0.25
        fw_dropout = 0.25

    def __init__(self, config):  # vocal_size是上面的全局参数，不需要传入构造函数。
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(self.ModelStruct.vocab_size,
                                                  self.ModelStruct.n_embd, dtype=self.ModelStruct.precision)  # (B, T, vocab_size)->(B, T, n_embd)
        self.pos_embedding_table = nn.Embedding(self.ModelStruct.block_size,
                                                self.ModelStruct.n_embd, dtype=self.ModelStruct.precision)  # pos和token编码长度必须一致，下面要加起来。

        self.blocks = nn.Sequential(
            *[DecoderBlock(self.ModelStruct) for _ in range(self.ModelStruct.n_blocks)]
        )
        self.ln_f = nn.LayerNorm(self.ModelStruct.n_embd, dtype=self.ModelStruct.precision)
        self.net = nn.Sequential(  # 构建网络
            self.blocks,
            self.ln_f
        )

        self.lm_head = nn.Linear(self.ModelStruct.n_embd,
                                 self.ModelStruct.vocab_size,
                                 dtype=self.ModelStruct.precision)  # 这里不再像二元模型时 编码维度==词汇量大小，因为解码不那么容易，需要明确的中间层。

    def forward(self, idx, target=None):
        tok_emd = self.token_embedding_table(idx)  # tok_emd.shape = (B, T, C), C = n_embd
        pos_emd = self.pos_embedding_table(
            torch.arange(idx.shape[1], device=self.config.device))  # 不能写block_size, 它是T=idx.shape[1]的上限
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
            content = idx[:,
                      -self.ModelStruct.block_size:]  # 有位置嵌入表后，输入的长度必须限制，否则查表会越界。ps:刚好可以输入最大值的上下文长度block，而不是block-1
            logits, loss = self(content)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def generate_and_print_token(self, idx, max_tokens, decoder):
        for _ in range(max_tokens):
            # 有位置嵌入表后，输入的长度必须限制，否则查表会越界。ps:刚好可以输入最大值的上下文长度block，而不是block-1
            content = idx[:, -self.ModelStruct.block_size:]
            logits, loss = self(content)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            print(decoder(idx_next.item()), end='', flush=True)
            # time.sleep(0.1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
