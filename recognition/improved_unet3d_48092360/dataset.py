# ================================================================================
# File: recognition/improved_unet3d_48092360/dataset.py
# Author: Roman Bek (48092360)
# Brief: HipMRI 3D dataset utilities: load NIfTI volumes, z-score normalization,
#   simple axis flips, numpy->tensor conversion, padding collate, and
#   deterministic 80/10/10 DataLoaders (train/val/test).
# Refs: PyTorch DataLoader docs (https://docs.pytorch.org/docs/stable/data.html)
# Last date modified: 2025-10-29
# ================================================================================

import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch.nn.functional as F


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Helper tp normalise an array to zero mean and unit variance (z-score)
    Args:
        x: Input image/array
        eps: Constant for numerical stability
    Returns:
        np.ndarray: z-scored array
    """
    x = x.astype(np.float32, copy=False)
    m, s = float(x.mean()), float(x.std())
    return (x - m) / (s if s > eps else eps)


class HipMRI3DVolumes(Dataset):
    """
    3D NIfTI volumes for UNet3D
    Returns dict: {'image': np.ndarray(H,W,D), 'mask': np.ndarray(H,W,D)}
    """
    def __init__(self, root: str, transform=None, binarize_mask: bool = True):
        """
        Initialize dataset paths and pair images with masks.
        Args:
            root (str): Dataset root containing 'semantic_MRs' and 'semantic_labels_only'
        """
        self.img_dir = os.path.join(root, "semantic_MRs")
        self.msk_dir = os.path.join(root, "semantic_labels_only")
        self.transform = transform
        self.binarize = binarize_mask

        # helper to normalize file stem to a comparable ID
        def _id(name: str) -> str:
            # strip extension
            if name.endswith(".nii.gz"):
                base = name[:-7]
            elif name.endswith(".nii"):
                base = name[:-4]
            else:
                base = os.path.splitext(name)[0]

            # Keep only <...>_Week<...>
            parts = base.split("_")
            if len(parts) >= 2 and parts[1].lower().startswith("week"):
                # Normalize capitalization of Week
                week = "Week" + parts[1][4:]
                return parts[0] + "_" + week

            # return the whole stem if it didnt match
            return base

        # list and sort filenames for deterministic pairing
        imgs = os.listdir(self.img_dir); imgs.sort()
        msks = os.listdir(self.msk_dir); msks.sort()

        # build a map from normalized ID -> mask filename
        m_by_id = {}
        for f in msks:
            k = _id(f)
            m_by_id[k] = f

        # create (image_path, mask_path) pairs where IDs match
        self.pairs = []
        for f in imgs:
            k = _id(f)
            if k in m_by_id:
                self.pairs.append((
                    os.path.join(self.img_dir, f),
                    os.path.join(self.msk_dir, m_by_id[k])
                ))

    def __len__(self):
        """Number of paired volumes in the dataset."""
        return len(self.pairs)

    def _load_3d(self, path: str) -> np.ndarray:
        """
        Load a 3D NIfTI volume as float32. Accepts (H,W,D) or 
        (H, W, D, 1) and just squeezes any size 1 axes

        Args:
            path (str): Filepath to a NIfTI file
        Returns:
            numpy.ndarray: 3D array of shape (H, W, D)
        """
        vol = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
        if vol.ndim == 4 and 1 in vol.shape:
            vol = np.squeeze(vol)
        return vol # (H,W,D)

    def __getitem__(self, i: int):
        """
        Get sample i: load volume + mask, squeeze to 3D, apply transform
        Args:
            i (int): Zero based sample index
        Returns:
            dict: {'image': np.ndarray(H,W,D) or torch.FloatTensor[1,D,H,W],
                'mask':  same type/shape as 'image'}
        """
        # resolve paths for i-th pair
        ip, mp = self.pairs[i]

        # load image volume (float32)
        vol = self._load_3d(ip)

        # load mask as integer class IDs (keep multiclass labels)
        msk = nib.as_closest_canonical(nib.load(mp)).get_fdata()
        if msk.ndim == 4 and 1 in msk.shape:
            msk = np.squeeze(msk)
        msk = np.asarray(msk, dtype=np.int64)

        # assemble sample and optionally transform
        sample = {"image": vol, "mask": msk}
        if self.transform:
            return self.transform(sample)
        return sample


class ZScore3D:
    """Normalize volume to zero mean/unit std; leave mask unchanged"""
    def __call__(self, sample):
        """
        Apply z-score normalization to 'image' only.
        Args:
            sample (dict): {'image': np.ndarray, 'mask': np.ndarray}
        Returns:
            dict: normalized image with original mask
        """
        return {"image": _zscore(sample["image"]), "mask": sample["mask"]}


class RandomFlip3D:
    """Random flips along each axis with prob p per axis"""
    def __init__(self, p: float = 0.5):
        """
        Args:
            p (float): Probability of applying a flip per axis
        """
        self.p = p
    def __call__(self, sample):
        """
        Randomly flip image and mask along axes 0/1/2 with prob p each
        Args:
            sample (dict): {'image': np.ndarray, 'mask': np.ndarray}
        """
        img, msk = sample["image"], sample["mask"]
        # flip along axis 0 (H)
        if random.random() < self.p:
            img = np.flip(img, 0); msk = np.flip(msk, 0)
        # flip along axis 1 (W)
        if random.random() < self.p:
            img = np.flip(img, 1); msk = np.flip(msk, 1)
        # flip along axis 2 (D)
        if random.random() < self.p:
            img = np.flip(img, 2); msk = np.flip(msk, 2)
        # ensure memory contiguity after flips
        return {"image": np.ascontiguousarray(img), "mask": np.ascontiguousarray(msk)}


class ToTensor3D:
    """Convert numpy (H,W,D) to torch tensors [1,D,H,W] for 3D convs"""
    def __call__(self, sample):
        """
        Convert arrays to tensors and permute to channel first depth first order
        Args:
            sample (dict): {'image': np.ndarray(H,W,D), 'mask': np.ndarray(H,W,D)}
        """
        img = torch.from_numpy(sample["image"]).float().permute(2, 0, 1).unsqueeze(0)
        msk = torch.from_numpy(sample["mask"]).long().permute(2, 0, 1).unsqueeze(0)
        return {"image": img, "mask": msk}

def pad_collate3d(batch):
    """
    Zero pad each [1,D,H,W] to the batch max size then stack
    Args:
        batch (list[dict]): items with 'image' and 'mask' tensors shaped [1,D,H,W]
    Returns:
        dict: {'image': torch.FloatTensor[B,1,D,H,W], 'mask': torch.LongTensor[B,1,D,H,W]}
    """
    # gather tensors and compute per-batch max spatial dims
    imgs = [b["image"] for b in batch]
    msks = [b["mask"] for b in batch]
    D = max(t.shape[1] for t in imgs)
    H = max(t.shape[2] for t in imgs)
    W = max(t.shape[3] for t in imgs)

    # pad each sample to (D,H,W) then stack
    pimgs, pmsks = [], []
    for x, y in zip(imgs, msks):
        pd = D - x.shape[1]
        ph = H - x.shape[2]
        pw = W - x.shape[3]
        x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="constant", value=0.0)
        y = F.pad(y, (0, pw, 0, ph, 0, pd), mode="constant", value=0)
        pimgs.append(x); pmsks.append(y)

    # stack into batch tensors
    return {"image": torch.stack(pimgs, 0), "mask": torch.stack(pmsks, 0)}

def build_dataset_3d(data_root: str, augment: bool = False):
    """
    Build the HipMRI3DVolumes dataset with standard transforms
    Args:
        data_root (str): Root path containing semantic_MRs and semantic_labels_only
        augment (bool): If True, include random flips
    Returns:
        HipMRI3DVolumes: Dataset giving dicts with volume/mask, transformed to tensors
    """
    # assemble transform pipeline (optional flips -> z-score -> to tensor)
    tfms = [ZScore3D()]
    if augment:
        tfms.insert(0, RandomFlip3D(p=0.5))
    tfms.append(ToTensor3D())
    return HipMRI3DVolumes(data_root, transform=transforms.Compose(tfms), binarize_mask=True)

def make_loaders_3d(data_root: str, batch_size: int = 1, num_workers: int = 0, split_ratio: float = 0.9):
    """
    Create DataLoaders for the 3D dataset with a deterministic 80/10/10 split (train/val/test)
    Args:
        data_root (str): Root path to the 3D dataset
        batch_size (int): Batch size for all loaders
        num_workers (int): Num of worker processes per DataLoader
        split_ratio (float): Unused; kept for compatibility with earlier versions
    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: (train_loader, val_loader, test_loader)

    REF: researched torch.Generator() and Dataloader()
    REF: https://docs.pytorch.org/docs/stable/data.html
    """
    # deterministic train/val/test via index permutation
    base = HipMRI3DVolumes(data_root, transform=None, binarize_mask=True)
    n = len(base)
    assert n >= 3, f"Not enough 3D volumes under {data_root}"
    n_tr = max(1, int(n * 0.8))
    n_va = max(1, int(n * 0.1))
    n_te = max(1, n - n_tr - n_va)

    # fixed-seed permutation for reproducible splits
    g = torch.Generator().manual_seed(67)
    perm = torch.randperm(n, generator=g).tolist()
    tr_idx = perm[:n_tr]; va_idx = perm[n_tr:n_tr+n_va]; te_idx = perm[n_tr+n_va:]

    # build transformed datasets (augmented for train, clean for eval/test)
    ds_train = build_dataset_3d(data_root, augment=True)
    ds_eval = build_dataset_3d(data_root, augment=False)

    # subset views for each split
    train_ds = Subset(ds_train, tr_idx)
    val_ds = Subset(ds_eval, va_idx)
    test_ds = Subset(ds_eval, te_idx)

    # wrap subsets in DataLoaders (pin_memory if CUDA available)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, collate_fn=pad_collate3d)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, collate_fn=pad_collate3d)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, collate_fn=pad_collate3d)
    return train_loader, val_loader, test_loader
