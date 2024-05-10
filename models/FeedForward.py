import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),  # 这里本来是n_embd, n_embd，这里是为了复现A I A Y N论文
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),  # 投影回残差路径的层
            nn.Dropout(config.fw_dropout)  # 正则化
        )

    def forward(self, X):
        return self.net(X)
