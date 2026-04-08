import torch
from tqdm import tqdm
from utils.metrics import dice_score, iou_score
from config import cfg

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for img, mask in tqdm(loader):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        pred = model(img)
        loss = loss_fn(pred, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    for img, mask in loader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = loss_fn(pred, mask)
        total_loss += loss.item()
        total_dice += dice_score(pred, mask).item()
        total_iou += iou_score(pred, mask).item()
    return (total_loss/len(loader),
            total_dice/len(loader),
            total_iou/len(loader))