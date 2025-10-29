# recognition/improved_unet3d_48092360/predict.py
import os, torch
import matplotlib.pyplot as plt
from recognition.improved_unet3d_48092360.modules import UNet3D
from recognition.improved_unet3d_48092360.dataset import build_dataset_3d

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUM_CLASSES = 6

# Data roots
DATA_ROOT_3D = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\improved_unet3d_48092360\data\3d_dataset"

OUTDIR = "recognition/improved_unet3d_48092360/outputs"
CKPT = os.path.join(OUTDIR, "best.pt") # loads if file exists
THRESH = 0.5


@torch.no_grad()
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # 3D path (multiclass)
    model = UNet3D(in_channels=1, out_channels=NUM_CLASSES, base=16, deep_supervision=True).to(DEVICE)
    if os.path.isfile(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print(f"Loaded weights: {CKPT}")
    else:
        print("No checkpoint found running with random 3D model")

    model.eval()

    ds = build_dataset_3d(DATA_ROOT_3D)
    sample = ds[0]
    x = sample["image"].unsqueeze(0).to(DEVICE) # [1,1,D,H,W]
    y = sample["mask"].unsqueeze(0).to(DEVICE) # [1,1,D,H,W]

    logits = model(x) # [1,C,D,H,W]
    probs = logits.softmax(1) # [1,C,D,H,W]
    pred = probs.argmax(1) # [1,D,H,W] (class IDs)

    # Per class Dice on this volume
    inter_sums = torch.zeros(NUM_CLASSES, device=DEVICE)
    den_sums = torch.zeros(NUM_CLASSES, device=DEVICE)
    t = y.squeeze(1).long() # [1,D,H,W]
    p = pred # [1,D,H,W]

    c_idx = 0
    while c_idx < NUM_CLASSES:
        pz = (p == c_idx).float()
        tz = (t == c_idx).float()
        inter_sums[c_idx] += (pz * tz).sum()
        den_sums[c_idx] += pz.sum() + tz.sum()
        c_idx += 1

    dice_per_class = (2.0 * inter_sums / (den_sums + 1e-6)).tolist()

    # Print only classes that appear in real labels for this volume
    msg_parts = []
    i = 0
    while i < NUM_CLASSES:
        if den_sums[i].item() > 0:
            msg_parts.append(f"c{i}:{dice_per_class[i]:.3f}")
        i += 1
    print("[3D] Dice per class (volume 0): " + " ".join(msg_parts))

    # Visualise a central axial slice
    img = x[0,0].cpu() # [D,H,W]
    msk = y[0,0].cpu() # [D,H,W] (int labels)
    pred_vol = pred[0].cpu() # [D,H,W] (int labels)
    D = img.shape[0]
    mid = D // 2

    img_s = img[mid].numpy()
    msk_s = msk[mid].numpy()
    pred_s = pred_vol[mid].numpy()

    plt.figure(figsize=(9,3))
    for i, (title, arr) in enumerate([("image[z=mid]", img_s), ("mask[z=mid]", msk_s), ("pred[z=mid]", pred_s)]):
        plt.subplot(1,3,i+1); plt.imshow(arr, cmap="gray"); plt.title(title); plt.axis("off")
    out_path = os.path.abspath(os.path.join(OUTDIR, "predict_example_3d.png"))
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
