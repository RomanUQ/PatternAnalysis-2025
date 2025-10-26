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
DATA_ROOT = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\unet3d_48092360\.gitignore\2d_dataset"
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-3
NUM_WORKERS = 0

# Data
train_loader, val_loader = make_loaders(
    DATA_ROOT, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Model / Optim / Loss
model = UNet2D(in_channels=1, out_channels=1, base=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

use_amp = torch.cuda.is_available()
scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

def train_one_epoch():
    """Run one training epoch over train_loader (forward, loss, backward, step)"""
    model.train()
    running, n = 0.0, 0
    for batch in train_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).to(torch.float32)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
        n += x.size(0)
    return running / max(n,1)

@torch.no_grad()
def evaluate():
    """
    Evaluate on val_loader and return mean loss
    Returns:
        float: Average validation loss over all samples
    """
    model.eval()
    running, n = 0.0, 0
    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        running += criterion(model(x), y).item() * x.size(0)
        n += x.size(0)
    return running / n

# Run
best = float('inf'); t0 = time.time()
for ep in range(1, EPOCHS + 1):
    tr_loss = train_one_epoch()
    va_loss = evaluate()
    best = min(best, va_loss)
    print(f"Epoch {ep:02d}/{EPOCHS} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} | best {best:.4f}")
print(f"Total time: {time.time()-t0:.1f}s")
