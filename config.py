import torch
import os

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    EPOCHS = 50
    BATCH_SIZE = 8
    IMAGE_SIZE = (256, 256)
    NUM_WORKERS = 4

    DATA_ROOT = "./data/"
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")
    MASK_DIR = os.path.join(DATA_ROOT, "masks")

    LR = 1e-4
    WEIGHT_DECAY = 1e-5

    LOSS_TYPES = ["dice_bce"]

    MODEL_NAMES = [
        "unet",
        "resunet",
        "unetpp",
        "resunetpp",
        "resunetpp_transformer"
    ]

    SAVE_DIR = "./results"
    LOG_DIR = os.path.join(SAVE_DIR, "logs")
    WEIGHTS_DIR = os.path.join(SAVE_DIR, "weights")
    FIGURE_DIR = os.path.join(SAVE_DIR, "figures")
    CURVE_DIR = os.path.join(SAVE_DIR, "curves")

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)
    os.makedirs(CURVE_DIR, exist_ok=True)

cfg = Config()