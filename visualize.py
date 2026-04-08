import os
import cv2
import glob
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from config import cfg
from data.dataset import KvasirDataset
from torch.utils.data import DataLoader

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def get_etis_loader():
    test_img = sorted(glob.glob("./data/ETIS-LaribPolypDB/images/*.png"))
    test_mask = sorted(glob.glob("./data/ETIS-LaribPolypDB/masks/*.png"))
    paired = list(zip(test_img, test_mask))
    random.shuffle(paired)
    test_img, test_mask = zip(*paired)
    test_ds = KvasirDataset(test_img, test_mask, augment=False)
    return DataLoader(test_ds, batch_size=1, shuffle=False)

def get_cvc_loader():
    test_img = sorted(glob.glob("./data/CVC-ColonDB/images/*.png"))
    test_mask = sorted(glob.glob("./data/CVC-ColonDB/masks/*.png"))
    paired = list(zip(test_img, test_mask))
    random.shuffle(paired)
    test_img, test_mask = zip(*paired)
    test_ds = KvasirDataset(test_img, test_mask, augment=False)
    return DataLoader(test_ds, batch_size=1, shuffle=False)

def load_model(model_name, weight_path):
    if model_name == "unet":
        from models.unet import UNet
        model = UNet()
    elif model_name == "resunetpp":
        from models.resunetpp import ResUNetPP
        model = ResUNetPP()
    elif model_name == "resunetpp_transformer":
        from models.resunetpp_transformer import ResUNetPPTransformer
        model = ResUNetPPTransformer()
    else:
        raise ValueError("only unet / resunetpp / resunetpp_transformer")

    model.load_state_dict(torch.load(weight_path, map_location=cfg.DEVICE))
    model.eval().to(cfg.DEVICE)
    return model

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
            plt.savefig(os.path.join(save_dir, f"{dataset_name}_3model_rand_{cnt + 1:02d}.png"), dpi=300, bbox_inches="tight")
            plt.close()
            cnt += 1
    print(f"✅ {dataset_name} 20 张三模型对比图已生成！")


if __name__ == "__main__":
    unet = load_model("unet", os.path.join(cfg.WEIGHTS_DIR, "unet_dice_bce.pth"))
    resunetpp = load_model("resunetpp", os.path.join(cfg.WEIGHTS_DIR, "resunetpp_dice_bce.pth"))
    ours = load_model("resunetpp_transformer", os.path.join(cfg.WEIGHTS_DIR, "resunetpp_transformer_dice_bce.pth"))

    models = {
        "unet": unet,
        "resunetpp": resunetpp,
        "ours": ours
    }

    etis_loader = get_etis_loader()
    draw_20_images(etis_loader, "etis", "etis_comparison", models)

    cvc_loader = get_cvc_loader()
    draw_20_images(cvc_loader, "cvc", "cvc_comparison", models)

    print("\n🎉 全部完成！ETIS + CVC-ColonDB 共 40 张对比图已生成！")