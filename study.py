import torch
from torch import nn

# 创建最大词个数为10，每个词用维度为4表示
embedding = nn.Embedding(10, 4)

# 将第一个句子填充0，与第二个句子长度对齐
in_vector = torch.LongTensor([[1, 2, 3, 4, 0, 0], [1, 2, 3, 4, 0, 0], [1, 2, 5, 6, 5, 7]])
out_emb = embedding(in_vector)
print(in_vector.shape)
print((out_emb.shape))
print(out_emb)
print(embedding.weight)

