# recognition/improved_unet3d_48092360/dataset.py
import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

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

# ----------------------------- 2D DATASET -----------------------------

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

        def _id(name: str) -> str:
            # remove extension
            if name.endswith(".nii.gz"):
                base = name[:-7]
            elif name.endswith(".nii"):
                base = name[:-4]
            else:
                base = os.path.splitext(name)[0]
            # strip prefixes so case_ and seg_ match
            for pref in ("case_", "seg_"):
                if base.startswith(pref):
                    return base[len(pref):]
            return base

        imgs = os.listdir(self.img_dir)
        imgs.sort()

        msks = os.listdir(self.msk_dir)
        msks.sort()

        self.pairs = []
        m_by_id = {}
        for f in msks:
            k = _id(f)
            m_by_id[k] = f

        for f in imgs:
            k = _id(f)
            if k in m_by_id:
                self.pairs.append((
                    os.path.join(self.img_dir, f),
                    os.path.join(self.msk_dir, m_by_id[k])))

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

def pad_collate(batch):
    """
    Pad to max H,W in the batch, then stack.
    Expects each item: {'image': Tensor[1,H,W], 'mask': Tensor[1,H,W]}.
    Pads with zeros (OK for image + mask).
    """
    imgs = [b["image"] for b in batch]
    msks = [b["mask"] for b in batch]
    H = max(t.shape[1] for t in imgs)
    W = max(t.shape[2] for t in imgs)

    pad_imgs, pad_msks = [], []
    for x, y in zip(imgs, msks):
        ph = H - x.shape[1]
        pw = W - x.shape[2]
        x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
        y = F.pad(y, (0, pw, 0, ph), mode="constant", value=0.0)
        pad_imgs.append(x)
        pad_msks.append(y)

    return {"image": torch.stack(pad_imgs, 0), "mask": torch.stack(pad_msks, 0)}

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
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=pin, collate_fn=pad_collate)
    return train_loader, val_loader

# ----------------------------- 3D DATASET -----------------------------

class HipMRI3DVolumes(Dataset):
    """
    3D NIfTI volumes for UNet3D
    Returns dict: {'image': np.ndarray(H,W,D), 'mask': np.ndarray(H,W,D)}
    """
    def __init__(self, root: str, transform=None, binarize_mask: bool = True):
        self.img_dir = os.path.join(root, "semantic_MRs")
        self.msk_dir = os.path.join(root, "semantic_labels_only")
        self.transform = transform
        self.binarize = binarize_mask

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

        # pairing images with masks
        imgs = os.listdir(self.img_dir); imgs.sort()
        msks = os.listdir(self.msk_dir); msks.sort()

        m_by_id = {}
        for f in msks:
            k = _id(f)
            m_by_id[k] = f

        self.pairs = []
        for f in imgs:
            k = _id(f)
            if k in m_by_id:
                self.pairs.append((
                    os.path.join(self.img_dir, f),
                    os.path.join(self.msk_dir, m_by_id[k])
                ))

    def __len__(self):
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
        ip, mp = self.pairs[i]
        vol = self._load_3d(ip)
        # keep raw integer class IDs (multiclass)
        msk = nib.as_closest_canonical(nib.load(mp)).get_fdata()
        if msk.ndim == 4 and 1 in msk.shape:
            msk = np.squeeze(msk)
        msk = np.asarray(msk, dtype=np.int64)
        sample = {"image": vol, "mask": msk}
        if self.transform:
            return self.transform(sample)
        return sample

class ZScore3D:
    """Normalize volume to zero mean/unit std; leave mask unchanged"""
    def __call__(self, sample):
        return {"image": _zscore(sample["image"]), "mask": sample["mask"]}

class ToTensor3D:
    """Convert numpy (H,W,D) to torch tensors [1,D,H,W] for 3D convs"""
    def __call__(self, sample):
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
    imgs = [b["image"] for b in batch]
    msks = [b["mask"] for b in batch]
    D = max(t.shape[1] for t in imgs)
    H = max(t.shape[2] for t in imgs)
    W = max(t.shape[3] for t in imgs)

    pimgs, pmsks = [], []
    for x, y in zip(imgs, msks):
        pd = D - x.shape[1]
        ph = H - x.shape[2]
        pw = W - x.shape[3]
        x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="constant", value=0.0)
        y = F.pad(y, (0, pw, 0, ph, 0, pd), mode="constant", value=0)
        pimgs.append(x); pmsks.append(y)

    return {"image": torch.stack(pimgs, 0), "mask": torch.stack(pmsks, 0)}

def build_dataset_3d(data_root: str):
    """
    Build the HipMRI3DVolumes dataset with standard transforms
    Args:
        data_root (str): Root path containing semantic_MRs and semantic_labels_only
    Returns:
        HipMRI3DVolumes: Dataset giving dicts with volume/mask, transformed to tensors
    """
    tfm = transforms.Compose([ZScore3D(), ToTensor3D()])
    return HipMRI3DVolumes(data_root, transform=tfm, binarize_mask=True)

def make_loaders_3d(data_root: str, batch_size: int = 1, num_workers: int = 0, split_ratio: float = 0.9):
    """
    Create DataLoaders for the 3D dataset with a deterministic 90/10 split
    Args:
        data_root (str): Root path to the 3D dataset
        batch_size (int): Batch size for both loaders
        num_workers (int): Num of worker processes per DataLoader
        split_ratio (float): Fraction for the training split (remainder is validation)
    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)

    REF: researched torch.Generator() and Dataloader()
    REF: https://docs.pytorch.org/docs/stable/data.html
    """
    full = build_dataset_3d(data_root)
    n = len(full)
    n_tr = max(1, int(n * split_ratio))
    gen = torch.Generator().manual_seed(67)
    train_ds, val_ds = torch.utils.data.random_split(full, [n_tr, n - n_tr], generator=gen)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, collate_fn=pad_collate3d)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin, collate_fn=pad_collate3d)
    return train_loader, val_loader
