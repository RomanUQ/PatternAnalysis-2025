# recognition/improved_unet3d_48092360/predict.py
import os, torch
import matplotlib.pyplot as plt
from recognition.unet3d_48092360.modules import UNet2D, UNet3D
from recognition.unet3d_48092360.dataset import build_dataset, build_dataset_3d

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODE toggle
MODE = "3d"  # strictly "2d" or "3d"

# Data roots
DATA_ROOT_2D = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\unet3d_48092360\data\2d_dataset"
DATA_ROOT_3D = r"C:\Users\roman\Desktop\COMP3710_REPORT\PatternAnalysis-2025\recognition\unet3d_48092360\data\3d_dataset"

CKPT = "checkpoints/best.pt" # loads if file exists
THRESH = 0.5


@torch.no_grad()
def dice_coef(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Batch Dice for 2D or 3D tensors
    2D expects [B,1,H,W]; 3D expects [B,1,D,H,W]
    """
    pred = (torch.sigmoid(logits) > THRESH).float()
    dims = (1,2,3) if pred.dim() == 4 else (1,2,3,4)
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims) + eps
    return (2.0 * inter / denom).mean().item()


@torch.no_grad()
def main():
    os.makedirs("plots", exist_ok=True)

    if MODE == "2d":
        # 2D path
        model = UNet2D(in_channels=1, out_channels=1, base=64).to(DEVICE)
        if os.path.isfile(CKPT):
            model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
            print(f"Loaded weights: {CKPT}")
        else:
            print("No checkpoint found, running with random 2D model")

        model.eval()

        ds = build_dataset(DATA_ROOT_2D, split="validate")
        sample = ds[0]
        x = sample["image"].unsqueeze(0).to(DEVICE) # [1,1,H,W]
        y = sample["mask"].unsqueeze(0).to(DEVICE) # [1,1,H,W]

        logits = model(x)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
        pred = (prob > THRESH).astype("float32")
        img = x[0,0].cpu().numpy()
        msk = y[0,0].cpu().numpy()

        d = dice_coef(logits, y)
        print(f"[2D] Dice(sample 0): {d:.4f}")

        plt.figure(figsize=(9,3))
        for i, (title, arr) in enumerate([("image", img), ("mask", msk), ("pred", pred)]):
            plt.subplot(1,3,i+1); plt.imshow(arr, cmap="gray"); plt.title(title); plt.axis("off")
        plt.tight_layout(); plt.savefig("plots/predict_example_2d.png", dpi=150); plt.close()
        print("Saved:", os.path.abspath("plots/predict_example_2d.png"))

    else:
        # #D path
        model = UNet3D(in_channels=1, out_channels=1, base=16, deep_supervision=True).to(DEVICE)
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

        logits = model(x)
        d = dice_coef(logits, y)
        print(f"[3D] Dice(volume 0): {d:.4f}")

        # Visualise a central axial slice
        prob = torch.sigmoid(logits)[0,0].cpu() # [D,H,W]
        img = x[0,0].cpu()
        msk = y[0,0].cpu()
        D = img.shape[0]
        mid = D // 2

        img_s  = img[mid].numpy()
        msk_s  = msk[mid].numpy()
        pred_s = (prob[mid].numpy() > THRESH).astype("float32")

        plt.figure(figsize=(9,3))
        for i, (title, arr) in enumerate([("image[z=mid]", img_s), ("mask[z=mid]", msk_s), ("pred[z=mid]", pred_s)]):
            plt.subplot(1,3,i+1); plt.imshow(arr, cmap="gray"); plt.title(title); plt.axis("off")
        plt.tight_layout(); plt.savefig("plots/predict_example_3d.png", dpi=150); plt.close()
        print("Saved:", os.path.abspath("plots/predict_example_3d.png"))


if __name__ == "__main__":
    main()
