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

# Data
train_loader, val_loader = make_loaders(
    DATA_ROOT, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Model / Optim / Loss
model = UNet2D(in_channels=1, out_channels=1, base=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

def train_one_epoch():
    """Run one training epoch over train_loader (forward, loss, backward, step)"""
    model.train()
    running, n = 0.0, 0
    for batch in train_loader:
        # Shape: [B,1,H,W]
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / n