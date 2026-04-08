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

class UNetPP(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        f = [32, 64, 128, 256, 256]

        self.c00 = DoubleConv(in_ch, f[0])
        self.c10 = DoubleConv(f[0], f[1])
        self.c20 = DoubleConv(f[1], f[2])
        self.c30 = DoubleConv(f[2], f[3])
        self.c40 = DoubleConv(f[3], f[4])

        self.c01 = DoubleConv(f[0]+f[1], f[0])
        self.c11 = DoubleConv(f[1]+f[2], f[1])
        self.c21 = DoubleConv(f[2]+f[3], f[2])
        self.c31 = DoubleConv(f[3]+f[4], f[3])

        self.c02 = DoubleConv(f[0]*2 + f[1], f[0])
        self.c12 = DoubleConv(f[1]*2 + f[2], f[1])
        self.c22 = DoubleConv(f[2]*2 + f[3], f[2])

        self.c03 = DoubleConv(f[0]*3 + f[1], f[0])
        self.c13 = DoubleConv(f[1]*3 + f[2], f[1])

        self.c04 = DoubleConv(f[0]*4 + f[1], f[0])
        self.out = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        x00 = self.c00(x)
        x10 = self.c10(self.pool(x00))
        x20 = self.c20(self.pool(x10))
        x30 = self.c30(self.pool(x20))
        x40 = self.c40(self.pool(x30))

        x01 = self.c01(torch.cat([x00, self.up(x10)], 1))
        x11 = self.c11(torch.cat([x10, self.up(x20)], 1))
        x21 = self.c21(torch.cat([x20, self.up(x30)], 1))
        x31 = self.c31(torch.cat([x30, self.up(x40)], 1))

        x02 = self.c02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.c12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.c22(torch.cat([x20, x21, self.up(x31)], 1))

        x03 = self.c03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        x13 = self.c13(torch.cat([x10, x11, x12, self.up(x22)], 1))

        x04 = self.c04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))
        return self.out(x04)