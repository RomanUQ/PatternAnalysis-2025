# recognition/unet3d_48092360/train.py
import time, torch
from torch import nn
from recognition.unet3d_48092360.dataset import make_loaders
from recognition.unet3d_48092360.modules import UNet2D

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters (simple constants, no argparse)
DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-3
NUM_WORKERS = 2
