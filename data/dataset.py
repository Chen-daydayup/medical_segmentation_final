import os
import cv2
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class KvasirDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.image_size = (256, 256)

        # ✅ 训练集增强（翻转、旋转、缩放）
        self.train_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

        # ✅ 验证集不增强
        self.val_transform = A.Compose([
            A.Resize(height=256, width=256),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图片和mask（保证一一对应）
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # 数据增强
        if self.augment:
            augmented = self.train_transform(image=image, mask=mask)
        else:
            augmented = self.val_transform(image=image, mask=mask)

        image = augmented["image"].float()
        mask = augmented["mask"].unsqueeze(0).float() / 255.0

        return image, mask