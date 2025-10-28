# recognition/improved_unet3d_48092360/train.py 
import time, torch, os
from torch import nn
from recognition.improved_unet3d_48092360.dataset import make_loaders, make_loaders_3d
from recognition.improved_unet3d_48092360.modules import UNet2D, UNet3D
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
if device.type == 'cpu':
    print("Warning CUDA not Found. Using CPU")

# MODE toggle
MODE = "3d"  # strictly "2d" or "3d"

NUM_CLASSES = 6

# bg small, c4/c5 higher
CLASS_WEIGHTS = torch.tensor([0.05, 1.0, 1.0, 1.0, 2.0, 2.0])

# Hyper-parameters (simple constants, no argparse)
# DATA ROOT is set by mode below
DATA_ROOT_2D = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\improved_unet3d_48092360\data\2d_dataset"
DATA_ROOT_3D = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\improved_unet3d_48092360\data\3d_dataset"

if MODE == "2d":
    DATA_ROOT = DATA_ROOT_2D
    EPOCHS = 10
    BATCH_SIZE = 4
    LR = 1e-3
    NUM_WORKERS = 0
else:
    DATA_ROOT = DATA_ROOT_3D
    EPOCHS = 30
    BATCH_SIZE = 1
    LR = 5e-4
    NUM_WORKERS = 2

# Data
if MODE == "2d":
    train_loader, val_loader = make_loaders(
        DATA_ROOT, split="train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
else:
    # use train/val/test
    train_loader, val_loader, test_loader = make_loaders_3d(
        DATA_ROOT, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# Model / Optim / Loss
if MODE == "2d":
    model = UNet2D(in_channels=1, out_channels=1, base=64).to(device)
    criterion = nn.BCEWithLogitsLoss()
else:
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES, base=16, deep_supervision=True).to(device)
    w = CLASS_WEIGHTS.to(device).float()
    criterion = nn.CrossEntropyLoss(weight=w)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

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

def soft_dice_loss(logits, target, eps=1e-6):
    C = logits.size(1)
    t = torch.nn.functional.one_hot(target.squeeze(1), C).permute(0,4,1,2,3).float()
    p = torch.softmax(logits, dim=1)
    inter = (p * t).sum(dim=(0,2,3,4))
    den = (p + t).sum(dim=(0,2,3,4)) + eps
    dice_per_c = 2*inter/den
    return 1.0 - dice_per_c.mean()

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
            if MODE == "3d":
                ce = criterion(logits, y.squeeze(1).long())
                loss = ce + 0.5 * soft_dice_loss(logits, y)
            else:
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

    # accumulators for multiclass Dice (3D)
    if MODE == "3d":
        inter_sums = torch.zeros(NUM_CLASSES, device=device)
        den_sums = torch.zeros(NUM_CLASSES, device=device)
        gt_sums = torch.zeros(NUM_CLASSES, device=device)

    for batch in val_loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(x)
            if MODE == "3d":
                ce = criterion(logits, y.squeeze(1).long())
                loss = ce + 0.5 * soft_dice_loss(logits, y)
            else:
                loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        n += x.size(0)

        if MODE == "3d":
            # compute classwise dice components
            t = y.squeeze(1).long()
            p = logits.softmax(1).argmax(1)
            for c in range(NUM_CLASSES):
                pz = (p == c).float()
                tz = (t == c).float()
                inter_sums[c] += (pz * tz).sum()
                den_sums[c] += pz.sum() + tz.sum()
                gt_sums[c] += tz.sum()
        else:
            dice_sum += dice_coef(logits, y) * x.size(0)

    if MODE == "3d":
        dice_per_class = (2.0 * inter_sums / (den_sums + 1e-6)).tolist()
        # ignore classes never present in GT to avoid degenerate denominators
        gt_list = gt_sums.tolist()
        valid = []
        for i, cnt in enumerate(gt_list):
            if cnt > 0:
                valid.append(i)
        if valid:
            min_val = None
            for i in valid:
                val_i = dice_per_class[i]
                if (min_val is None) or (val_i < min_val):
                    min_val = val_i
            min_dice = min_val
        else:
            min_dice = 0.0
        # compact printout
        parts = []
        for i in valid:
            parts.append(f"c{i}:{dice_per_class[i]:.3f}")
        short = " ".join(parts)
        print(f"    [val dice per class] {short}")
        return loss_sum / max(n,1), float(min_dice)
    else:
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

    # Final test evaluation
    if MODE == "3d":
        model.eval()
        loss_sum, n = 0.0, 0
        inter_sums = torch.zeros(NUM_CLASSES, device=device)
        den_sums = torch.zeros(NUM_CLASSES, device=device)
        gt_sums = torch.zeros(NUM_CLASSES, device=device)
        with torch.no_grad():
            for batch in test_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y.squeeze(1).long())
                loss_sum += loss.item() * x.size(0)
                n += x.size(0)
                t = y.squeeze(1).long()
                p = logits.softmax(1).argmax(1)
                for c in range(NUM_CLASSES):
                    pz = (p == c).float()
                    tz = (t == c).float()
                    inter_sums[c] += (pz * tz).sum()
                    den_sums[c] += pz.sum() + tz.sum()
                    gt_sums[c] += tz.sum()

        test_loss = loss_sum / max(n,1)
        dice_per_class = (2.0 * inter_sums / (den_sums + 1e-6)).tolist()
        gt_list_test = gt_sums.tolist()
        valid = []
        for i, cnt in enumerate(gt_list_test):
            if cnt > 0:
                valid.append(i)
        if valid:
            min_val = None
            for i in valid:
                val_i = dice_per_class[i]
                if (min_val is None) or (val_i < min_val):
                    min_val = val_i
            min_dice = min_val
        else:
            min_dice = 0.0
        parts = []
        for i in valid:
            parts.append(f"c{i}:{dice_per_class[i]:.3f}")
        short = " ".join(parts)
        print(f"[TEST] loss {test_loss:.4f} | min dice {min_dice:.4f}")
        print(f"[TEST] dice per class: {short}")
