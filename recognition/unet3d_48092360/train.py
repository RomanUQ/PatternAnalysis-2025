from modules import UNet2D
import torch

# just to test run Unet2D for now
if __name__ == "__main__":
    m = UNet2D(in_channels=1, out_channels=1)
    x = torch.zeros(2,1,128,128)
    y = m(x)
    print("UNet2D forward OK:", tuple(y.shape))