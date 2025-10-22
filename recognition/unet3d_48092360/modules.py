import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Two stacked 3x3 convolutions with BN + ReLU
    Args:
        in_channels (int): Num of input channels
        out_channels (int): Num of output channels
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """2D UNet with 4 downs and transpose conv ups"""
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base: int = 64):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, base) # 64
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.down4 = Down(base*8, base*16)
        self.bot = DoubleConv(base*16, base*16) # 1024

        # Decoder
        self.up1 = Up(base*16, base*8) # 1024 -> 512
        self.up2 = Up(base*8, base*4)
        self.up3 = Up(base*4, base*2)
        self.up4 = Up(base*2, base) # 128 -> 64
        self.outc = OutConv(base, out_channels)

    def forward(self, x):
        x1 = self.inc(x) # 128x128
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.bot(x5) # 8x8

        y = self.up1(x6, x4) # 16x16
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1) # 128x128
        return self.outc(y)


class Down(nn.Module):
    """
    Downsampling block: MaxPool(2) to DoubleConv
    Args:
        in_ch (int): Input channels
        out_ch (int): Output channels after DoubleConv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    """
    Upsampling block: ConvTranspose2d to concat skip to DoubleConv.
    Uses learned transpose convolution to upsample the deep feature map,
    concatenates it with the corresponding encoder skip then refines with DoubleConv
    Args:
        in_ch (int): Channels entering block (from bottleneck or prev up stage)
        out_ch (int): Output channels after DoubleConv
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        # Upsample deep feature
        # (N, in_ch//2, H*2, W*2)
        x = self.up(x)

        # Center crop skip to match x if shapes differ (handles odd/even dims)
        if skip.size(2) != x.size(2) or skip.size(3) != x.size(3):
            # if skip is smaller (odd dims), pad first to avoid empty crop
            if skip.size(2) < x.size(2) or skip.size(3) < x.size(3):
                ph = x.size(2) - skip.size(2)
                pw = x.size(3) - skip.size(3)
                skip = F.pad(skip, (pw//2, pw - pw//2, ph//2, ph - ph//2))
            dh = (skip.size(2) - x.size(2)) // 2
            dw = (skip.size(3) - x.size(3)) // 2
            skip = skip[:, :, dh:dh + x.size(2), dw:dw + x.size(3)]

        # Concat along channels then refine
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final 1x1 convolution mapping features to class logits
    Args:
        in_ch (int): Input channels from the last decoder stage
        out_ch (int): Num of output channels
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Placeholder for 3D UNet
    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super().__init__()
        self.head = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.head(x)
