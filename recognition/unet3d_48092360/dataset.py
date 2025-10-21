import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
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

        imgs = sorted(os.listdir(self.img_dir))
        mset = set(os.listdir(self.msk_dir))
        self.pairs = []
        for f in imgs:
            if f in mset:
                self.pairs.append((os.path.join(self.img_dir, f), os.path.join(self.msk_dir, f)))

    def __len__(self):
        return len(self.pairs)
    
    def _load_2d(self, path: str) -> np.ndarray:
        """
        Load a 2D NIfTI slice as float32. Accepts (H, W) or 
        (H, W, 1)/(1, H, W) and just squeezes any size-1 axes

        Args:
            path (str): Filepath to a NIfTI file
        Returns:
            numpy.ndarray: 2D array of shape (H, W) with dtype float32
        """
        arr = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
        if arr.ndim == 2:
            return arr
        return np.squeeze(arr)
    
    def __getitem__(self, i: int):
        """
        Get sample i: load image + mask, squeeze to 2D, binarize mask, apply transform
        Args:
            i (int): Zero-based sample index
        Returns:
            dict: {'image': np.ndarray(H,W) or torch.FloatTensor[1,H,W],
                'mask':  same type/shape as 'image'}
        """
        ip, mp = self.pairs[i]
        img = self._load_2d(ip)
        msk = self._load_2d(mp)

        # convert any label >0 to 1.0
        if self.binarize:
            msk = (msk != 0).astype(np.float32)
        sample = {"image": img, "mask": msk}

        if self.transform:
            return self.transform(sample)
        return sample

class ZScore:
    """Normalize image to zero mean/unit std, leave mask unchanged"""
    def __call__(self, sample):
        return {"image": _zscore(sample["image"]), "mask": sample["mask"]}

class ToTensor:
    """Convert numpy arrays to torch tensors and add channel dim"""
    def __call__(self, sample):
        # shape: (1,H,W)
        img = torch.from_numpy(sample["image"]).unsqueeze(0).float()
        msk = torch.from_numpy(sample["mask"]).unsqueeze(0).float()
        return {"image": img, "mask": msk}

def build_dataset(data_root: str, split: str = "train"):
    """
    Build the HipMRISlicesDataset with standard transforms (ZScore; ToTensor)
    Args:
        data_root (str): Root path containing keras_slices_* and keras_slices_seg_* folders
        split (str): Dataset split to load ("train", "validate", or "test")
    Returns:
        HipMRISlicesDataset: Dataset giving dicts with image/mask, transformed to tensors
    """
    tfm = transforms.Compose([ZScore(), ToTensor()])
    return HipMRISlicesDataset(data_root, split=split, transform=tfm, binarize_mask=True)

def make_loaders(data_root: str, split: str = "train", batch_size: int = 8, num_workers: int = 2):
    """
    Create DataLoaders for the chosen split (train) and the validation split
    Args:
        data_root (str): Root path to the dataset
        split (str): Split for the training loader ("train" or "test")
        batch_size (int): Batch size for both loaders
        num_workers (int): Num of worker processes per DataLoader
    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    train_ds = build_dataset(data_root, split=split)
    val_ds = build_dataset(data_root, split="validate")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
