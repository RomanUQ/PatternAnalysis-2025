# recognition/unet3d_48092360/train.py
import time, torch, os
from torch import nn
from recognition.unet3d_48092360.dataset import make_loaders, make_loaders_3d
from recognition.unet3d_48092360.modules import UNet2D, UNet3D
import matplotlib.pyplot as plt

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print("Warning CUDA not Found. Using CPU")

# MODE toggle
MODE = "3d"  # strictly "2d" or "3d"

# Hyper-parameters (simple constants, no argparse)
# DATA ROOT is set by mode below
DATA_ROOT_2D = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\unet3d_48092360\data\2d_dataset"
DATA_ROOT_3D = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\unet3d_48092360\data\3d_dataset"

if MODE == "2d":
    DATA_ROOT = DATA_ROOT_2D
    EPOCHS = 10
    BATCH_SIZE = 4
    LR = 1e-3
    NUM_WORKERS = 0
else:
    DATA_ROOT = DATA_ROOT_3D
    EPOCHS = 20
    BATCH_SIZE = 1
    LR = 1e-3
    NUM_WORKERS = 0

# Data
if MODE == "2d":
    train_loader, val_loader = make_loaders(
        DATA_ROOT, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
else:
    train_loader, val_loader = make_loaders_3d(
        DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Model / Optim / Loss
if MODE == "2d":
    model = UNet2D(in_channels=1, out_channels=1, base=64).to(device)
else:
    model = UNet3D(in_channels=1, out_channels=1, base=16, deep_supervision=True).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

def dice_coef(logits, target, eps: float = 1e-6):
    pred = (torch.sigmoid(logits) > 0.5).float()
    # auto select dims: 2D is (1,2,3), 3D is (1,2,3,4)
    if pred.dim() == 4:
        dims = (1,2,3)
    else:
        dims = (1,2,3,4)
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims) + eps
    return (2.0 * inter / denom).mean().item()

def train_one_epoch():
    """Run one training epoch over train_loader (forward, loss, backward, step)"""
    model.train()
    running, n = 0.0, 0
    for i, batch in enumerate(train_loader):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * x.size(0)
        n += x.size(0)
        if (i + 1) % 500 == 0:
            print(f"  step {i+1}/{len(train_loader)}  loss={loss.item():.4f}", flush=True)
    return running / max(n,1)

@torch.no_grad()
def evaluate():
    """
    Evaluate on val_loader and return mean loss
    Returns:
        float: Average validation loss over all samples
    """
    model.eval()
    loss_sum, dice_sum, n = 0.0, 0.0, 0
    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        dice_sum += dice_coef(logits, y) * x.size(0)
        n += x.size(0)
    return loss_sum / n, dice_sum / n

if __name__ == "__main__":

    # Run + track metrics
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    train_losses, val_losses, val_dice = [], [], []
    best = float('inf'); t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        tr_loss = train_one_epoch()
        va_loss, va_d = evaluate()
        train_losses.append(tr_loss); val_losses.append(va_loss); val_dice.append(va_d)

        # save best checkpoint (by val loss)
        if va_loss <= best:
            best = va_loss
            torch.save(model.state_dict(), os.path.join("checkpoints", "best.pt"))

        print(f"Epoch {ep:02d}/{EPOCHS} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} | dice {va_d:.4f} | best {best:.4f}")

    # plots
    plt.figure(); plt.plot(train_losses, label="train"); plt.plot(val_losses, label="val"); plt.legend(); plt.title("Loss"); plt.savefig("plots/loss_curve.png", dpi=150); plt.close()
    plt.figure(); plt.plot(val_dice, label="val dice"); plt.legend(); plt.title("Dice"); plt.savefig("plots/dice_curve.png", dpi=150); plt.close()
    print(f"Saved plots to {os.path.abspath('plots')}")
    print(f"Total time: {time.time()-t0:.1f}s")
