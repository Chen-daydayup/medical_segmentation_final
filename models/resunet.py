import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)

class ResUNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.c1 = ResBlock(in_ch, 32)
        self.c2 = ResBlock(32, 64)
        self.c3 = ResBlock(64, 128)
        self.c4 = ResBlock(128, 256)
        self.c5 = ResBlock(256, 256)
        self.c6 = ResBlock(256+256, 256)
        self.c7 = ResBlock(256+128, 128)
        self.c8 = ResBlock(128+64, 64)
        self.c9 = ResBlock(64+32, 32)
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(self.pool(c1))
        c3 = self.c3(self.pool(c2))
        c4 = self.c4(self.pool(c3))
        c5 = self.c5(self.pool(c4))
        x = self.up(c5); x=torch.cat([x,c4],1); x=self.c6(x)
        x = self.up(x); x=torch.cat([x,c3],1); x=self.c7(x)
        x = self.up(x); x=torch.cat([x,c2],1); x=self.c8(x)
        x = self.up(x); x=torch.cat([x,c1],1); x=self.c9(x)
        return self.out(x)