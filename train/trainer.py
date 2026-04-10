import torch
from tqdm import tqdm
from utils.metrics import calculate_metrics
from config import cfg

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for img, mask in tqdm(loader, desc="Training", leave=False):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        out = model(img)

        if isinstance(out, (list, tuple)):
            loss = 0
            for o in out:
                loss += loss_fn(o, mask)
        else:
            loss = loss_fn(out, mask)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    total_pre = 0.0
    total_rec = 0.0
    cnt = len(loader)

    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        out = model(img)
        loss = loss_fn(out, mask)
        total_loss += loss.item()
        d, i, p, r = calculate_metrics(out, mask)
        total_dice += d
        total_iou += i
        total_pre += p
        total_rec += r

    return total_loss/cnt, total_dice/cnt, total_iou/cnt, total_pre/cnt, total_rec/cnt
