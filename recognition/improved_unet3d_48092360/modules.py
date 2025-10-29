# ================================================================================================
# File: recognition/improved_unet3d_48092360/modules.py
# Author: Roman Bek (48092360)
# Brief: Isensee style Improved UNet3D for prostate MRI segmentation (6 classes)
#   with pre-activation residual context blocks, learnable downsampling,
#   nearest-neighbor upsampling + 3x3x3 conv, localization blocks and
#   summed deep-supervision heads.
# Refs: Isensee (https://arxiv.org/abs/1802.10508v1), nnU-Net (https://arxiv.org/abs/1809.10486)
# Last date modified: 29/10/2025
# =================================================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    """
    Improved 3D UNet, network structure from (Isensee, 2018):
    - ContextBlock3d (pre-activation residual + dropout)
    - 3x3x3 stride 2 downsampling
    - nearest neighbor upsample + 3x3x3 conv
    - LocalizationBlock3d after skip concat
    - Deep supervision: sum of mult -scale segmentation heads
    - out_chanels = 6 for HipMRI: background + 5 organs

    REF: (Isensee 2018)
    https://arxiv.org/abs/1802.10508v1
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 6, base: int = 16, deep_supervision: bool = True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Encoder (context)
        self.inc = ContextBlock3d(in_channels, base, pdrop=0.3)
        self.down1 = Down3d(base, base*2)
        self.down2 = Down3d(base*2, base*4)
        self.down3 = Down3d(base*4, base*8)
        self.down4 = Down3d(base*8, base*16)
        self.bot = ContextBlock3d(base*16, base*16, pdrop=0.3)

        # Decoder (localization)
        self.up1 = Up3d(base*16, base*8, skip_ch=base*8)
        self.up2 = Up3d(base*8, base*4, skip_ch=base*4)
        self.up3 = Up3d(base*4, base*2, skip_ch=base*2)
        self.up4 = Up3d(base*2, base, skip_ch=base)
        self.outc = OutConv3d(base, out_channels)

        # Deep supervision heads (summed to final)
        self.seg2 = nn.Conv3d(base*4, out_channels, kernel_size=1)
        self.seg3 = nn.Conv3d(base*2, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path: context features at progressively lower resolutions
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.bot(x5)

        # Decoder path: upsample, align+concat skip, then localize/refine
        y1 = self.up1(x6, x4) # base*8
        y2 = self.up2(y1, x3) # base*4
        y3 = self.up3(y2, x2) # base*2
        y4 = self.up4(y3, x1) # base

        # Final classifier, if enabled add deep supervision heads (upsampled) to logits
        # REF: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
        logits = self.outc(y4)
        if self.deep_supervision:
            logits = (
                logits
                + F.interpolate(self.seg3(y3), size=logits.shape[2:], mode="nearest")
                + F.interpolate(self.seg2(y2), size=logits.shape[2:], mode="nearest")
            )
        return logits


class ContextBlock3d(nn.Module):
    """
    Pre activation residual context module with dropout
    IN, LeakyReLU, 3x3x3, Dropout3d, IN, LeakyReLU, 3x3x3, with residual
    If channels change uses 1x1x1 for the skip path

    REF: nnU-Net (LeakyReLU)
    https://arxiv.org/abs/1809.10486
    REF: (Isensee, 2018)
    https://arxiv.org/abs/1802.10508
    """
    def __init__(self, in_ch: int, out_ch: int, pdrop: float = 0.3):
        super().__init__()
        self.in1 = nn.InstanceNorm3d(in_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.drop = nn.Dropout3d(p=pdrop)
        self.in2 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act2 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        # Pre activation -> conv -> dropout -> pre-activation -> conv -> residual add
        res = x
        y = self.act1(self.in1(x))
        y = self.conv1(y)
        y = self.drop(y)
        y = self.act2(self.in2(y))
        y = self.conv2(y)
        return y + self.skip(res)


class LocalizationBlock3d(nn.Module):
    """
    Localization block: 3x3x3 conv (+IN+LeakyReLU), 1x1x1 conv
    Used after concatenating skip and upsampled features to reduce/aggregate channels
    
    REF: nnU-Net (LeakyReLU)
    https://arxiv.org/abs/1809.10486
    REF: (Isensee, 2018)
    https://arxiv.org/abs/1802.10508v1
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.act1 = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        # Reduce+refine fused (skip+upsampled) features
        x = self.conv1(x)
        x = self.act1(self.in1(x))
        x = self.conv2(x)
        return x
    

class Down3d(nn.Module):
    """
    Improved downsampling (Isensee et al.): 3x3x3 stride-2 conv -> ContextBlock3d.
    Replaces max-pooling with learnable downsampling

    REF: (Isensee, 2018)
    https://arxiv.org/abs/1802.10508v1
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.block = ContextBlock3d(out_ch, out_ch, pdrop=0.3)

    def forward(self, x):
        # Strided conv for downsampling then context processing
        return self.block(self.down(x))


class Up3d(nn.Module):
    """
    Improved upsampling: nearest neighbor upsample x2, 3x3x3 conv (+IN+LeakyReLU),
    concat with aligned skip then LocalizationBlock3d. Avoids checkerboard artifacts

    REF: (Isensee, 2018) nearest upsample+conv vs transpose conv
    https://arxiv.org/abs/1802.10508v1
    """
    def __init__(self, in_ch: int, out_ch: int, skip_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.reduce = nn.Conv3d(in_ch, in_ch // 2, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm3d(in_ch // 2, affine=True)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        self.loc = LocalizationBlock3d(in_ch // 2 + skip_ch, out_ch)

    def _align(self, skip, x):
        # Pad/crop the skip so spatial dims match the upsampled tensor (D, H, W)
        pd = max(0, x.size(2) - skip.size(2))
        ph = max(0, x.size(3) - skip.size(3))
        pw = max(0, x.size(4) - skip.size(4))
        if pd or ph or pw:
            # W, H, D
            skip = F.pad(skip, (pw//2, pw - pw//2,  ph//2, ph - ph//2, pd//2, pd - pd//2))
        dd = (skip.size(2) - x.size(2)) // 2
        dh = (skip.size(3) - x.size(3)) // 2
        dw = (skip.size(4) - x.size(4)) // 2
        return skip[:, :, dd:dd + x.size(2), dh:dh + x.size(3), dw:dw + x.size(4)]

    def forward(self, x, skip):
        # Upsample -> channel reduce -> align+concat skip -> localization refine
        x = self.up(x)
        x = self.act(self.in1(self.reduce(x)))
        skip = self._align(skip, x)
        x = torch.cat([skip, x], dim=1)
        return self.loc(x)


class OutConv3d(nn.Module):
    """
    Final 1x1x1 convolution mapping features to class logits (3D)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        # Map decoder features to raw class logits
        return self.conv(x)
