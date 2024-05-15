from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import torch.multiprocessing as mp
import torch


class CustomDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + block_size]
        y = self.data[idx + 1: idx + block_size + 1]
        return x, y


...  # 省略部分不变的代码 ...


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)  # 初始化进程组，它们之间要通信。


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size, data, block_size, iterations, eval_interval):
    setup(rank, world_size)

    model = TransformerModel().to(device)  # 模型实例
    model = DDP(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_set = CustomDataset(data, block_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=world_size, rank=rank)
    loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler)

    start_time = time.time()
    for epoch in range(iterations // len(loader) + 1):
        for i, (xb, yb) in enumerate(loader):
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if i % eval_interval == 0:
                print('epoch {}, iter {}, loss {}'.format(epoch, i, loss.item()))

    print(f'训练耗时：{time.time() - start_time} s')


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    mp.spawn(worker, args=(n_gpus, data, block_size, iterations, eval_interval), nprocs=n_gpus, join=True)