import torch.nn as nn
from .MultiSA import MultiSA
from .FeedForward import FeedForward


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, dtype=config.precision)
        self.sa = MultiSA(config)
        self.ln2 = nn.LayerNorm(config.n_embd, dtype=config.precision)
        self.ffwd = FeedForward(config)

    def forward(self, X):
        X = X + self.sa(self.ln1(X))  # 优化：加入残差，层规范（类似于BatchNorm)
        X = X + self.ffwd(self.ln2(X))
        return X
