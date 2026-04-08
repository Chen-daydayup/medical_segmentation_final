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

class ResUNetPP(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        f = [32,64,128,256,256]

        self.c00 = ResBlock(in_ch, f[0])
        self.c10 = ResBlock(f[0], f[1])
        self.c20 = ResBlock(f[1], f[2])
        self.c30 = ResBlock(f[2], f[3])
        self.c40 = ResBlock(f[3], f[4])

        self.c01 = ResBlock(f[0]+f[1], f[0])
        self.c11 = ResBlock(f[1]+f[2], f[1])
        self.c21 = ResBlock(f[2]+f[3], f[2])
        self.c31 = ResBlock(f[3]+f[4], f[3])

        self.c02 = ResBlock(f[0]*2 + f[1], f[0])
        self.c12 = ResBlock(f[1]*2 + f[2], f[1])
        self.c22 = ResBlock(f[2]*2 + f[3], f[2])

        self.c03 = ResBlock(f[0]*3 + f[1], f[0])
        self.c13 = ResBlock(f[1]*3 + f[2], f[1])

        self.c04 = ResBlock(f[0]*4 + f[1], f[0])
        self.out = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        x00 = self.c00(x)
        x10 = self.c10(self.pool(x00))
        x20 = self.c20(self.pool(x10))
        x30 = self.c30(self.pool(x20))
        x40 = self.c40(self.pool(x30))

        x01 = self.c01(torch.cat([x00, self.up(x10)],1))
        x11 = self.c11(torch.cat([x10, self.up(x20)],1))
        x21 = self.c21(torch.cat([x20, self.up(x30)],1))
        x31 = self.c31(torch.cat([x30, self.up(x40)],1))

        x02 = self.c02(torch.cat([x00,x01,self.up(x11)],1))
        x12 = self.c12(torch.cat([x10,x11,self.up(x21)],1))
        x22 = self.c22(torch.cat([x20,x21,self.up(x31)],1))

        x03 = self.c03(torch.cat([x00,x01,x02,self.up(x12)],1))
        x13 = self.c13(torch.cat([x10,x11,x12,self.up(x22)],1))

        x04 = self.c04(torch.cat([x00,x01,x02,x03,self.up(x13)],1))
        return self.out(x04)