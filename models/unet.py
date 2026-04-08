import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.c1 = DoubleConv(in_ch, 32)
        self.c2 = DoubleConv(32, 64)
        self.c3 = DoubleConv(64, 128)
        self.c4 = DoubleConv(128, 256)
        self.c5 = DoubleConv(256, 256)

        self.c6 = DoubleConv(256+256, 256)
        self.c7 = DoubleConv(256+128, 128)
        self.c8 = DoubleConv(128+64, 64)
        self.c9 = DoubleConv(64+32, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))
        c4 = self.c4(self.pool(c3))
        c5 = self.c5(self.pool(c4))

        x = self.up(c5)
        x = torch.cat([x, c4], dim=1)
        x = self.c6(x)

        x = self.up(x)
        x = torch.cat([x, c3], dim=1)
        x = self.c7(x)

        x = self.up(x)
        x = torch.cat([x, c2], dim=1)
        x = self.c8(x)

        x = self.up(x)
        x = torch.cat([x, c1], dim=1)
        x = self.c9(x)
        return self.out(x)