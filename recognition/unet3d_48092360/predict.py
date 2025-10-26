# recognition/unet3d_48092360/predict.py
import os, torch
import matplotlib.pyplot as plt
from recognition.unet3d_48092360.modules import UNet2D
from recognition.unet3d_48092360.dataset import build_dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\unet3d_48092360\.gitignore\2d_dataset"
CKPT = "checkpoints/best.pt" # loads if file exists
THRESH = 0.5

@torch.no_grad()
def dice_coef(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Batch Dice over tensors shaped [B,1,H,W]."""
    pred  = (torch.sigmoid(logits) > THRESH).float()
    inter = (pred * target).sum(dim=(1,2,3))
    denom = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    return (2.0 * inter / denom).mean().item()

def main():
    """Load one val sample, run UNet2D, print Dice, save image/mask/pred figure."""
    model = UNet2D(in_channels=1, out_channels=1, base=64).to(DEVICE)
    if os.path.isfile(CKPT):
        model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
        print(f"Loaded weights: {CKPT}")
    else:
        print("No checkpoint found running with random model")

    model.eval()

    ds = build_dataset(DATA_ROOT, split="validate")
    sample = ds[0]
    print("sample shapes:", sample["image"].shape, sample["mask"].shape)
    x = sample["image"].unsqueeze(0).to(DEVICE) # [1,1,H,W]
    y = sample["mask"].unsqueeze(0).to(DEVICE) # [1,1,H,W]

    logits = model(x)
    prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    pred = (prob > THRESH).astype("float32")
    img = x[0,0].cpu().numpy()
    msk = y[0,0].cpu().numpy()

    d = dice_coef(logits, y)
    print(f"Dice(sample 0): {d:.4f}")

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(9,3))
    for i, (title, arr) in enumerate([("image", img), ("mask", msk), ("pred", pred)]):
        plt.subplot(1,3,i+1); plt.imshow(arr, cmap="gray"); plt.title(title); plt.axis("off")
    plt.tight_layout(); plt.savefig("plots/predict_example.png", dpi=150); plt.close()
    print("Saved:", os.path.abspath("plots/predict_example.png"))

if __name__ == "__main__":
    main()
