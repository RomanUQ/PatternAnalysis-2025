# COMP3710 — Project 7: Improved UNet3D for Prostate MRI Segmentation
> Result highlight: All six labels achieve Dice >= 0.70 on the test set.
> Current run: min per class Dice = 0.8255 (test).
___

![Model Diagram](recognition/improved_unet3d_48092360/outputs/predict_example_3d.PNG)

---

**Author:** Roman Bek (48092360)
___
## Scope
This repository implements only the 3D solution for Project 7 (Improved UNet3D). Previously explored a 2D UNet as a warm up, but per course guidance only 3D materials are included here and the 2D work is mentioned for context only and not part of the submission however, it can be seen in the commit history.
---
## Problem and Goal
**Task:** Segment downsampled 3D prostate MRI volumes into semantic classes (background + 5 organs), meeting the project requirement that all labels achieve Dice >= 0.70 on the test set.

**Dataset format:** NIfTI volumes (.nii.gz) with matching image/label stems.
---
## Method

The model implemented is the Improved UNet3D inspired by nnU-Net/Isensee:

* Context blocks (pre activation residual): IN; LeakyReLU; 3x3x3; Dropout3d; IN; LeakyReLU; 3x3x with residual/1x1x1 skip.
    - these keep gradients healthy in deep paths; 3x3x3 kernels capture compact 3D context efficiently; light dropout regularises without disrupting volumetric structure
* Learnable downsampling: 3x3x3 stride 2 conv
    - strided convolutions learn the downsampling filter (anti aliasing + feature selection) instead of discarding detail like max pool, preserving semantics at lower resolutions
* Upsampling that avoids checkerboard: nearest-neighbor x2, then 3x3x3 conv.
    - nearest neighbor avoids checkerboard artifacts common with transposed conv; the following 3x3x3 refines edges and restores fine detail
* Localisation blocks after skip concatenation.
    - After fusing high-res encoder features with deep decoder context a small conv stack rebalances channels and sharpens boundaries while also controlling memory
* Deep supervision: two auxiliary 1x1x1 heads (from decoder stages) upsampled and summed into the final logits.
    - multi scale losses shorten gradient paths and encourage consistent predictions across scales which speeds convergence and especially helps small/rare organs
* Normalization: InstanceNorm3d(affine=True) for small batch stability.
    - With batch size = 1, BatchNorm is unstable and InstanceNorm is batch independent and also mitigates MR intensity variability across scans.
* Loss: CrossEntropy(weight=class_weights) + 0.5 * soft_dice_loss(logits, y)
    - CE provides calibrated class probabilities, soft dice directly optimises region overlap. Class weights down weight background and up weight small organs to combat imbalance. This hybrid objective follows successful practice in medical segmentation (Isensee, 2018).

**Algorithm:** The encoder captures 3D context via residual ContextBlock3d modules with learnable downsampling to keep discriminative signal at depth. The decoder upsamples without transpose conv artifacts, fuses aligned skips and uses Localisation blocks to refine features. Deep supervision improves gradient flow and multi scale consistency. Class weighted CE + soft Dice balances calibrated probabilities with boundary/overlap quality while InstanceNorm3d decouples stats from small batches.
---
## Model Details
* Improved UNet3D (Isensee style)
* Classes: NUM_CLASSES = 6 (mapping: 0=Background, 1=Body, 2=Bone, 3=Bladder, 4=Rectum, 5=Prostate)
* Deep supervision: two aux heads summed into the final logits
* Normalisation: InstanceNorm3d
* Activations: LeakyReLU(0.01)
* Regularisation: light Dropout3d in context blocks
---
## Data and Preprocessing
### Transforms
* Z-score normalization per volume (images only)
    - Standardizes intensity scale per scan so the network doesn't chase absolute MR intensity
* Augmentation (train only): random flips on each axis (p = 0.5 per axis).
    - Pelvic structures are about symmetric so flips expand data diversity cheaply and reduce overfitting
* Tensor shapes: images to [1, D, H, W], labels to [1, D, H, W] (long, class ids). Where, 1: channels (grayscale); D: depth; H: height; W: width
* Batching: zero-pad each sample to batch maxima in D/H/W.
---
### Split and reproducibility
* Deterministic 80/10/10 train/val/test via a fixed generator seed (67). This maximizes training data while retaining a non tiny validation set for stable model selection.
* 80% maximizes learning signal and 10% val is large enough for stable model selection and fixed seed makes runs comparable
## Training Setup
* Loss: CrossEntropy(weight=class_weights) + 0.5 * soft_dice_loss(logits, y)
* Class weights: [0.05, 1.0, 1.0, 1.0, 2.0, 2.0] (down weight background, emphasize small organs)
* Optimizer: AdamW(lr = 5e-4)
* Precision: Mixed precision (torch.amp) enabled when CUDA is available
* Schedule: epochs = 30, batch_size = 1 (3D memory constraints)
* Metrics: per-class Dice on val each epoch; min per-class Dice shown as a conservative summary; test reported at end

**Justification:** Batch size = 1 to fit full 3D volumes in VRAM while remaining stable with InstanceNorm3d; 30 epochs because val loss/Dice could plateau around 25–30 and already meet the >= 0.7 per class target; LR = 5e-4 (AdamW) as a stable mid range choice (between 1e-4 and 1e-3) that converges reliably with deep supervision under AMP.
---
## Environment and Dependencies
### Versions
* Python 3.10.6
* PyTorch >= 2.1
* NumPy 1.26.4
* NiBabel >= 5.3
* Matplotlib >= 3.9
* (TorchVision for transforms wrapper)

### Hardware and runtime (original run)
* OS: Windows (paths like C:\Users...)
* Epochs: 30, batch size: 1
* Total time: ~ 10,953.7 s (~ 3 h 02 m)
* Device is selected automatically: CUDA if available, otherwise CPU
---
## Usage
### 1) Place data
**Put 3D data under:**
- recognition/improved_unet3d_48092360/data/3d_dataset/semantic_MRs/
- recognition/improved_unet3d_48092360/data/3d_dataset/semantic_labels_only/
(OR change DATA_ROOT_3D variable path to data in train.py and predict.py)

### 2) Train
**Run:**
- python recognition/improved_unet3d_48092360/train.py

**Aartifacts are saved in:**
- recognition/improved_unet3d_48092360/outputs/best.pt          (best by validation loss)
- recognition/improved_unet3d_48092360/outputs/loss_curve.png
- recognition/improved_unet3d_48092360/outputs/dice_curve.png

### 3) Predict (single volume)
**Run:**
- python recognition/improved_unet3d_48092360/predict.py

**Outputs:**
- Console: per class Dice for a held out example
- recognition/improved_unet3d_48092360/outputs/predict_example_3d.png    (central axial slice (image, GT, pred))
---
## Results
### Validation (during training)
Loss and "min perclass Dice" per epoch are printed; the two plots are saved automatically to recognition/improved_unet3d_48092360/outputs

### Test set (final)
**30 epoch run, final console out:**
loss 0.0622 | min dice 0.8255
dice per class: c0:0.989 c1:0.972 c2:0.897 c3:0.902 c4:0.826 c5:0.836

where c0: Background, c1: Body, c2: Bone, c3: Bladder, c4: Rectum, c5: Prostate
**so Minimum per class Dice (test): 0.8255 whcich meets >= 0.70 for all classes.**

## Plot of Loss and Plot of Dice

![Train vs Val Loss](recognition/improved_unet3d_48092360/outputs/loss_curve.png)

---

![Validation Dice per Class per Epoch](recognition/improved_unet3d_48092360/outputs/dice_curve.png)
