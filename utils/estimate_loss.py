import torch
from tqdm import tqdm

from .get_batch import get_batch


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
