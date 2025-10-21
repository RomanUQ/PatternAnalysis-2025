import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize an array to zero mean and unit variance (z-score)
    Args:
        x: Input image/array
        eps: Constant for numerical stability
    Returns:
        np.ndarray: z-scored array
    """
    x = x.astype(np.float32, copy=False)
    m, s = float(x.mean()), float(x.std())
    return (x - m) / (s if s > eps else eps)

class HipMRISlicesDataset(Dataset):
    """
    2D NIfTI slices for UNet2D
    Uses folders:
      <root>/keras_slices_<split>/     (images)
      <root>/keras_slices_seg_<split>/ (masks)
    Returns dict: {'image': np.ndarray(H,W), 'mask': np.ndarray(H,W)}
    """
    def __init__(self, root: str, split: str = "train",
                 transform: transforms.Compose | None = None,
                 binarize_mask: bool = True):
        self.img_dir = os.path.join(root, f"keras_slices_{split}")
        self.msk_dir = os.path.join(root, f"keras_slices_seg_{split}")
        self.transform = transform
        self.binarize = binarize_mask

        self.pairs = []

    def __len__(self):
        return len(self.pairs)
