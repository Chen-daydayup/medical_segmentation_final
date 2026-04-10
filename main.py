import os
import glob
import random
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import cfg
from data.dataset import KvasirDataset
from losses.loss_factory import get_loss
from torch.utils.data import DataLoader
from utils import count_parameters, calculate_metrics,save_comparison_table
from train import train_one_epoch, validate

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(cfg.SEED)

kvasir_img = sorted(glob.glob("./data/Kvasir/images/*.jpg"))
kvasir_mask = sorted(glob.glob("./data/Kvasir/masks/*.jpg"))
cvc_clinic_img = sorted(glob.glob("./data/CVC-ClinicDB_PNG_datasets/Original/*.png"))
cvc_clinic_mask = sorted(glob.glob("./data/CVC-ClinicDB_PNG_datasets/Truth/*.png"))

all_img = kvasir_img + cvc_clinic_img
all_mask = kvasir_mask + cvc_clinic_mask

train_img, val_img, train_mask, val_mask = train_test_split(
    all_img, all_mask, test_size=0.2, random_state=cfg.SEED
)

test_img_etis = sorted(glob.glob("./data/ETIS-LaribPolypDB/images/*.png"))
test_mask_etis = sorted(glob.glob("./data/ETIS-LaribPolypDB/masks/*.png"))
test_img_cvc = sorted(glob.glob("./data/CVC-ColonDB/images/*.png"))
test_mask_cvc = sorted(glob.glob("./data/CVC-ColonDB/masks/*.png"))

train_ds = KvasirDataset(train_img, train_mask, augment=True)
val_ds = KvasirDataset(val_img, val_mask, augment=False)
test_ds_etis = KvasirDataset(test_img_etis, test_mask_etis, augment=False)
test_ds_cvc = KvasirDataset(test_img_cvc, test_mask_cvc, augment=False)

train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)
test_loader_etis = DataLoader(test_ds_etis, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)
test_loader_cvc = DataLoader(test_ds_cvc, batch_size=1, shuffle=False, num_workers=cfg.NUM_WORKERS)

def build_model(name):
    if name == "unet":
        from models.unet import UNet
        return UNet()
    elif name == "resunet":
        from models.resunet import ResUNet
        return ResUNet()
    elif name == "unetpp":
        from models.unetpp import UNetPP
        return UNetPP()
    elif name == "resunetpp":
        from models.resunetpp import ResUNetPP
        return ResUNetPP()
    elif name == "resunetpp_transformer":
        from models.resunetpp_transformer import ResUNetPPTransformer
        return ResUNetPPTransformer()

results = []
all_val_loss = []
all_val_dice = []

for model_name in cfg.MODEL_NAMES:
    loss_type = "dice_bce"
    print(f"\n===== {model_name} | {loss_type} =====")
    model = build_model(model_name).to(cfg.DEVICE)
    total_p, trainable_p = count_parameters(model)
    print(f"[PARAMS] Total: {total_p / 1e6:.2f}M | Trainable: {trainable_p / 1e6:.2f}M")

    loss_fn = get_loss(loss_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    best_dice = 0
    best_iou = 0
    best_pre = 0
    best_rec = 0
    val_loss_history = []
    val_dice_history = []

    for epoch in range(cfg.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, cfg.DEVICE)
        val_loss, val_dice, val_iou, val_pre, val_rec = validate(model, val_loader, loss_fn, cfg.DEVICE)

        val_loss_history.append(val_loss)
        val_dice_history.append(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            best_iou = val_iou
            best_pre = val_pre
            best_rec = val_rec
            torch.save(model.state_dict(), os.path.join(cfg.WEIGHTS_DIR, f"{model_name}_{loss_type}.pth"))

        print(f"Ep {epoch+1:2d} | TrainLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | Dice {val_dice:.4f}")

    results.append({
        "model": model_name,
        "params(M)": round(total_p / 1e6, 2),
        "best_val_dice": round(best_dice, 4),
        "best_val_iou": round(best_iou, 4)
    })
    all_val_loss.append(val_loss_history)
    all_val_dice.append(val_dice_history)
    save_comparison_table(results, "train_results.csv")

print("\n" + "="*80)
print(" 🔥 最终评估：双独立测试集综合验证 🔥")
print("="*80)

def test_general_dataset(loader, model_name, loss_type, device):
    model = build_model(model_name).to(device)
    weight_path = os.path.join(cfg.WEIGHTS_DIR, f"{model_name}_{loss_type}.pth")
    if not os.path.exists(weight_path):
        print(f"⚠️ 未找到 {model_name} 权重，跳过测试！")
        return None
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    total_dice, total_iou, total_pre, total_rec = 0,0,0,0
    cnt = len(loader)
    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(device), mask.to(device)
            out = model(img)
            d,i,p,r = calculate_metrics(out, mask)
            total_dice += d
            total_iou += i
            total_pre += p
            total_rec += r
    return {
        "dice": round(total_dice/cnt,4),
        "iou": round(total_iou/cnt,4),
        "precision": round(total_pre/cnt,4),
        "recall": round(total_rec/cnt,4)
    }

final_results = []
for model_name in cfg.MODEL_NAMES:
    loss_type = "dice_bce"
    print(f"\n>>> Testing {model_name} ...")
    etis_res = test_general_dataset(test_loader_etis, model_name, loss_type, cfg.DEVICE)
    cvc_res = test_general_dataset(test_loader_cvc, model_name, loss_type, cfg.DEVICE)
    if etis_res and cvc_res:
        print(f"ETIS | Dice:{etis_res['dice']:.4f} | IoU:{etis_res['iou']:.4f} | Recall:{etis_res['recall']:.4f}")
        print(f"CVC  | Dice:{cvc_res['dice']:.4f} | IoU:{cvc_res['iou']:.4f} | Recall:{cvc_res['recall']:.4f}")
        final_results.append({
            "model": model_name,
            "etis_dice": etis_res['dice'], "etis_iou": etis_res['iou'], "etis_recall": etis_res['recall'],
            "cvc_dice": cvc_res['dice'], "cvc_iou": cvc_res['iou'], "cvc_recall": cvc_res['recall']
        })

save_comparison_table(final_results, "final_test_results.csv")

if all_val_loss:
    plt.figure(figsize=(10,5))
    for i, label in enumerate(cfg.MODEL_NAMES):
        plt.plot(range(1, cfg.EPOCHS+1), all_val_loss[i], label=label)
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(cfg.CURVE_DIR, "val_loss.png"), dpi=300)
    plt.close()

if all_val_dice:
    plt.figure(figsize=(10,5))
    for i, label in enumerate(cfg.MODEL_NAMES):
        plt.plot(range(1, cfg.EPOCHS+1), all_val_dice[i], label=label)
    plt.title("Validation Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(cfg.CURVE_DIR, "val_dice.png"), dpi=300)
    plt.close()

print("\n🎉 训练与评估全部完成！")
