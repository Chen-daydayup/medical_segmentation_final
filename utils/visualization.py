import matplotlib.pyplot as plt
import torch
import os
from config import cfg

def visualize_result(model, loader, device, save_path):
    model.eval()
    with torch.no_grad():
        img, mask = next(iter(loader))
        img, mask = img.to(device), mask.to(device)
        pred = torch.sigmoid(model(img)) > 0.5

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img[0].cpu().permute(1,2,0))
        axes[0].set_title("Image")
        axes[1].imshow(mask[0].cpu().squeeze(), cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[2].imshow(pred[0].cpu().squeeze(), cmap="gray")
        axes[2].set_title("Prediction")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()