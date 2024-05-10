import torch.nn as nn
from models.MultiAttentionHead import MultiAttentionHead
from models.FeedForward import FeedForward


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.sa = MultiAttentionHead(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)

    def forward(self, X):
        X = X + self.sa(self.ln1(X))  # 优化：加入残差，层规范（类似于BatchNorm)
        X = X + self.ffwd(self.ln2(X))
        return X
