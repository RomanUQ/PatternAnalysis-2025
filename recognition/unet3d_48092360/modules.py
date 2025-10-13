import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """Two stacked 3x3 convs with BN+ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1 ,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet2D(nn.Module):
    """Temporary stub so train/predict run"""
    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.head = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(x)

class UNet2D: pass
class UNet3D: pass