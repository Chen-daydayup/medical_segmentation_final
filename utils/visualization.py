import matplotlib.pyplot as plt
import torch
import os
import cv2
import numpy as np
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

def overlay(img, gt, pred, alpha=0.5):
    overlay = img.copy()
    H, W = img.shape[:2]
    gt = cv2.resize(gt, (W, H))
    pred = cv2.resize(pred, (W, H))
    mask_gt = (gt > 0.5)
    mask_pred = (pred > 0.5)
    overlay[mask_gt] = (overlay[mask_gt] * (1 - alpha) + np.array([0, 255, 0]) * alpha).astype(np.uint8)
    overlay[mask_pred] = (overlay[mask_pred] * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
    return overlay

def draw_20_images(loader, dataset_name, save_subdir, models):
    save_dir = os.path.join(cfg.FIGURE_DIR, save_subdir)
    os.makedirs(save_dir, exist_ok=True)
    cnt = 0
    with torch.no_grad():
        for img, gt in loader:
            if cnt >= 20:
                break
            img = img.to(cfg.DEVICE)
            pred_u = torch.sigmoid(models["unet"](img)).cpu().numpy()[0, 0]
            pred_r = torch.sigmoid(models["resunetpp"](img)).cpu().numpy()[0, 0]
            pred_o = torch.sigmoid(models["ours"](img)).cpu().numpy()[0, 0]

            img_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gt_np = gt[0, 0].numpy()

            ov_u = overlay(img_np, gt_np, pred_u)
            ov_r = overlay(img_np, gt_np, pred_r)
            ov_o = overlay(img_np, gt_np, pred_o)

            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(ov_u)
            plt.title("UNet")
            plt.axis("off")
            plt.subplot(1, 3, 2)
            plt.imshow(ov_r)
            plt.title("ResUNet++")
            plt.axis("off")
            plt.subplot(1, 3, 3)
            plt.imshow(ov_o)
            plt.title("Ours (ResUNet++-Transformer)")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{dataset_name}_3model_{cnt+1:02d}.png"), dpi=300, bbox_inches="tight")
            plt.close()
            cnt += 1
    print(f"✅ {dataset_name} 对比图生成完成")
