import argparse
import random
from dataclasses import dataclass
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from models import DevModel
from data import TrainDataSet


@dataclass
# 训练、评估、预测设置
class TrainConfig(nn.Module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 65
    # 训练设置
    train_data_proportion = 0.9
    batch_size = 64
    iterations = 5000
    learning_rate = 3e-4
    eval_interval = 500
    eval_iters = 200
    max_tokens = 500


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)
device = torch.device('cuda', args.local_rank)
torch.distributed.init_process_group(backend='nccl')

# 固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

batch_size = 64

model = DevModel(TrainConfig)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)

# 加载数据集
input_file_path = "data/input.txt"
with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
train_data = TrainDataSet(data, model.ModelStruct.block_size)
# training!
if args.local_rank == 0:
    tb_writer = SummaryWriter(comment='ddp-3')

train_sampler = DistributedSampler(train_data)
train_loader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                  find_unused_parameters=True)

for i, (inputs, labels) in enumerate(train_loader):
    # forward
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # log
    if args.local_rank == 0 and i % 5 == 0:
        tb_writer.add_scalar('loss', loss.item(), i)

if args.local_rank == 0:
    tb_writer.close()
