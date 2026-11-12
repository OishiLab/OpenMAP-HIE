# OpenMAP-HIE

Pathology-informed score-based diffusion model for unsupervised neonatal hypoxic–ischemic encephalopathy (HIE) lesion localization on apparent diffusion coefficient (ADC) images.

Manuscript: [OpenMAP-HIE](https://arxiv.org/html/2610.29322v1)

This code is built on Aggregated Normative Diffusion (ANDi):

- Paper: [Medical Image Analysis](https://www.sciencedirect.com/science/article/pii/S1361841525004414)
- Code: [AlexanderFrotscher/ANDi](https://github.com/AlexanderFrotscher/ANDi)

This repository currently contains the training code. Pretrained model weights and the evaluation code will be available soon.

## What the model does

OpenMAP-HIE localizes acute HIE lesions without voxel-wise lesion labels. Training uses only healthy neonatal ADC slices.

1. A denoising diffusion model learns the healthy ADC distribution. Its U-Net predicts the noise added at each diffusion step, which is a score of that healthy distribution.
2. A ControlNet branch is then trained on top of the frozen U-Net. The observed ADC image is the spatial condition, so the model can keep patient-specific anatomy instead of rebuilding a generic healthy brain. During this stage the condition is dropped out at random, which leaves an unconditional (healthy-prior) prediction available at test time.

At inference, the same noisy ADC slice is scored twice: once with the subject image as the condition, and once with no condition. The anomaly signal is the difference between those two noise predictions, not the residual of a finished pseudo-healthy reconstruction. Because acute HIE lowers ADC, only differences that point in that direction are kept. Those maps are combined across noise levels with a geometric mean. An optional population ADC Z-map can refine the result; the core model needs only the subject’s ADC image.

## Files

| File | Purpose |
| --- | --- |
| `train_bonbid.py` | Stage 1. Trains the healthy-prior DDPM U-Net on lesion-free BONBID-HIE slices. |
| `train_bonbid_affine_double_space.yml` | Config for stage 1 (CSV, image size, noise schedule, pyramid noise, checkpoint interval). |
| `train_controlnet_bonbid.py` | Stage 2. Loads a stage-1 checkpoint, freezes the U-Net, and trains the ControlNet branch. |
| `train_controlnet_bonbid_affine_double_space.yml` | Config for stage 2. `pretrained_unet_path` must point at a stage-1 `{epoch}_ema_ckpt.pt`. |
| `modules.py` | U-Net, EMA, and the learning-rate schedule used by both stages. |
| `control_net.py` | ControlNet wrapper: frozen base U-Net plus a trainable control encoder with zero-initialized residual connections. |
| `diffusion.py` | Forward noising, sampling, and the conditional–unconditional noise-prediction discrepancy used for anomaly maps. |
| `datasets/bonbid2023.py` | Loads axial slices listed in the CSV column of lesion-free indices. Returns images for stage 1 and `(image, control)` for stage 2. |
| `utils.py` | Shared helpers from ANDi: image saving, pyramid noise, checkpoint directories, and volume I/O. |
| `environment.yml` | Conda environment used to run training. |
| `README_BONBID2023.md` | CSV columns and the two training commands. |
| `README_ControlNet.md` | How the ControlNet branch, conditioning columns, and classifier-free guidance are wired. |

## Training

Set `csv_path` in both YAML files to a BONBID-HIE CSV that lists NIfTI paths and the lesion-free axial slice indices. `channels` must equal the number of entries in `dataset_channels`.

Stage 1 writes checkpoints to `models/<run_name>/{epoch}_ckpt.pt` and `{epoch}_ema_ckpt.pt`.

```bash
python train_bonbid.py
```

Stage 2 loads the EMA checkpoint from stage 1 (`pretrained_unet_path`) and trains only the control branch.

```bash
python train_controlnet_bonbid.py
```

More detail is in `README_BONBID2023.md` and `README_ControlNet.md`.

The OpenMAP-HIE is a publication of The Johns Hopkins University and copyright © 2026 The Johns Hopkins University. All rights reserved.
