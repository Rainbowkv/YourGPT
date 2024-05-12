import torch

from tqdm import tqdm


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(train_data, val_data, model, config):
    out = []
    model.eval()
    for data in tqdm([train_data, val_data], desc="evaluating outer"):
        losses = torch.zeros(config.eval_iters)
        for k in tqdm(range(config.eval_iters), desc="evaluating inner"):
            x, y = get_batch(data, model.ModelStruct.block_size, config.batch_size, config.device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out.append(losses.mean())
    model.train()
    return out
