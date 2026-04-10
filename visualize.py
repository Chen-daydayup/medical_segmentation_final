import os
import glob
import random
import numpy as np
import torch
from config import cfg
from data.dataset import KvasirDataset
from torch.utils.data import DataLoader
from utils import draw_20_images, overlay

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
        raise ValueError("仅支持 unet / resunetpp / resunetpp_transformer")

    model.load_state_dict(torch.load(weight_path, map_location=cfg.DEVICE))
    model.eval().to(cfg.DEVICE)
    return model

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

    print("\n🎉 全部可视化完成：ETIS + CVC 共 40 张三模型对比图！")
