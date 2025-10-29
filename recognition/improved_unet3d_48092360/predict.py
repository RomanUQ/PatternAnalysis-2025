# ==================================================================================
# File: recognition/improved_unet3d_48092360/predict.py
# Author: Roman Bek (48092360)
# Brief: Loads best.pt, runs inference on one 3D volume, reports perclass Dice,
#   and saves a segmentation slice (image/GT/pred) to ./outputs.
# Last date modified: 2025-10-29
# ==================================================================================

import os, torch
import matplotlib.pyplot as plt
from recognition.improved_unet3d_48092360.modules import UNet3D
from recognition.improved_unet3d_48092360.dataset import build_dataset_3d

# Select CUDA if available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 6

# Data roots
DATA_ROOT_3D = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\improved_unet3d_48092360\data\3d_dataset"

# Output dir and checkpoint path (if present)
OUTDIR = "recognition/improved_unet3d_48092360/outputs"

# loads if file exists
CKPT = os.path.join(OUTDIR, "best.pt")
THRESH = 0.5


@torch.no_grad()
def main():
    """ Run inference on a single 3D volume and save a central-slice visualisation"""
    # Ensure output directory exists
    os.makedirs(OUTDIR, exist_ok=True)

    # Build model and (optionally) load best checkpoint
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES, base=16, deep_supervision=True).to(DEVICE)
    if os.path.isfile(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print(f"Loaded weights: {CKPT}")
    else:
        print("No checkpoint found running with random 3D model")

    model.eval()

    # Build dataset and take the first volume (index 0)
    ds = build_dataset_3d(DATA_ROOT_3D)
    sample = ds[0]

    # [1,1,D,H,W]
    x = sample["image"].unsqueeze(0).to(DEVICE)
    y = sample["mask"].unsqueeze(0).to(DEVICE)

    # Forward pass and argmax over classes to predicted labels
    logits = model(x)
    probs = logits.softmax(1)
    # (class IDs)
    pred = probs.argmax(1)

    # Accumulate per class Dice numerators/denominators
    inter_sums = torch.zeros(NUM_CLASSES, device=DEVICE)
    den_sums = torch.zeros(NUM_CLASSES, device=DEVICE)

    # [1,D,H,W]
    t = y.squeeze(1).long()
    p = pred

    # Perclass loop (0..NUM_CLASSES-1)
    c_idx = 0
    while c_idx < NUM_CLASSES:
        pz = (p == c_idx).float()
        tz = (t == c_idx).float()
        inter_sums[c_idx] += (pz * tz).sum()
        den_sums[c_idx] += pz.sum() + tz.sum()
        c_idx += 1

    # Compute Dice per class only print classes present in this volume
    dice_per_class = (2.0 * inter_sums / (den_sums + 1e-6)).tolist()

    # console message for classes that appear
    msg_parts = []
    i = 0
    while i < NUM_CLASSES:
        if den_sums[i].item() > 0:
            msg_parts.append(f"c{i}:{dice_per_class[i]:.3f}")
        i += 1
    print("[3D] Dice per class (volume 0): " + " ".join(msg_parts))

    # Extract central axial slice for image/mask/pred visualisation
    # [D,H,W]
    img = x[0,0].cpu()
    msk = y[0,0].cpu()
    pred_vol = pred[0].cpu()
    D = img.shape[0]
    mid = D // 2

    # Slice arrays at z=mid
    img_s = img[mid].numpy()
    msk_s = msk[mid].numpy()
    pred_s = pred_vol[mid].numpy()

    # Render and save figure
    plt.figure(figsize=(9,3))
    for i, (title, arr) in enumerate([("image[z=mid]", img_s), ("mask[z=mid]", msk_s), ("pred[z=mid]", pred_s)]):
        plt.subplot(1,3,i+1); plt.imshow(arr, cmap="gray"); plt.title(title); plt.axis("off")
    out_path = os.path.abspath(os.path.join(OUTDIR, "predict_example_3d.png"))
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
