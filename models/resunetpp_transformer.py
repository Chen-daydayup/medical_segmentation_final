import torch
import torch.nn as nn
from .resunetpp import ResBlock

class TransformerBlock(nn.Module):
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )
        self.drop_path = nn.Dropout(0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class ResUNetPPTransformer(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        f = [32, 64, 128, 256, 256]

        # Encoder
        self.c00 = ResBlock(in_ch, f[0])
        self.c10 = ResBlock(f[0], f[1])
        self.c20 = ResBlock(f[1], f[2])
        self.c30 = ResBlock(f[2], f[3])
        self.c40 = ResBlock(f[3], f[4])

        # Transformer 瓶颈
        self.transformer = TransformerBlock(dim=f[4])
        self.bottleneck_conv = ResBlock(f[4], f[4])

        # Decoder
        self.c01 = ResBlock(f[0] + f[1], f[0])
        self.c11 = ResBlock(f[1] + f[2], f[1])
        self.c21 = ResBlock(f[2] + f[3], f[2])
        self.c31 = ResBlock(f[3] + f[4], f[3])

        self.c02 = ResBlock(f[0] * 2 + f[1], f[0])
        self.c12 = ResBlock(f[1] * 2 + f[2], f[1])
        self.c22 = ResBlock(f[2] * 2 + f[3], f[2])

        self.c03 = ResBlock(f[0] * 3 + f[1], f[0])
        self.c13 = ResBlock(f[1] * 3 + f[2], f[1])

        self.c04 = ResBlock(f[0] * 4 + f[1], f[0])

        # Deep Supervision 输出头
        self.out1 = nn.Conv2d(f[0], num_classes, 1)
        self.out2 = nn.Conv2d(f[0], num_classes, 1)
        self.out3 = nn.Conv2d(f[0], num_classes, 1)
        self.out4 = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        x00 = self.c00(x)
        x10 = self.c10(self.pool(x00))
        x20 = self.c20(self.pool(x10))
        x30 = self.c30(self.pool(x20))
        x40 = self.c40(self.pool(x30))

        # Transformer 全局建模
        x40 = self.transformer(x40)
        x40 = self.bottleneck_conv(x40)

        # 解码器
        x01 = self.c01(torch.cat([x00, self.up(x10)], dim=1))
        x11 = self.c11(torch.cat([x10, self.up(x20)], dim=1))
        x21 = self.c21(torch.cat([x20, self.up(x30)], dim=1))
        x31 = self.c31(torch.cat([x30, self.up(x40)], dim=1))

        # UNet++ 密集连接
        x02 = self.c02(torch.cat([x00, x01, self.up(x11)], 1))
        x12 = self.c12(torch.cat([x10, x11, self.up(x21)], 1))
        x22 = self.c22(torch.cat([x20, x21, self.up(x31)], 1))

        x03 = self.c03(torch.cat([x00, x01, x02, self.up(x12)], 1))
        x13 = self.c13(torch.cat([x10, x11, x12, self.up(x22)], 1))

        x04 = self.c04(torch.cat([x00, x01, x02, x03, self.up(x13)], 1))

        # 深度监督
        if self.training:
            y1 = self.out1(x01)
            y2 = self.out2(x02)
            y3 = self.out3(x03)
            y4 = self.out4(x04)
            return y1, y2, y3, y4
        else:
            return self.out4(x04)