## ControlNet for ANDi DDPM

This repository includes a ControlNet-style conditioning wrapper for the existing ANDi DDPM U-Net.

### Files

- **`control_net.py`**
  - `ControlNetUNet`: wraps a frozen pretrained `modules.UNet` and injects control residuals.
- **`train_controlnet.py`**
  - Trains ControlNet on top of a **frozen** pretrained DDPM U-Net checkpoint.
- **`utils.py`**
  - `Volume3DControlDataset`: loads **image** channels and **control** channels from different CSV columns.
  - `Volume3D_Control(conf)`: dataloader that returns `(images, control)` batches.
- **`diffusion.py`**
  - `Diffusion.sample_controlnet(...)`: sampling loop that calls the ControlNet model with CFG.
  - `Diffusion.normative_diffusion_controlnet(...)`: ANDi-style per-timestep deviations when the model is `ControlNetUNet` (requires `control` each step).
- **`notebooks/controlnet_eval.ipynb`**
  - Loads ControlNet, runs conditional sampling, accumulated norm map, SSIM, Dice (set `MASK_LESION`), and Fréchet (FID-style) on a batch.
- **Config**
  - `conf/train_controlnet_volume3d.yml`: example config for Volume3D + separate control columns (includes optional `eval_norm_*` for the notebook).

### Data interface (important)

For ControlNet training, the condition is loaded **separately** from the input image channels:

- **Images**: come from CSV columns listed in `dataset_channels` (these form the DDPM/U-Net input).
- **Control**: comes from CSV columns listed in `control_columns` (these form the ControlNet conditioning input).

So the dataloader yields:

- `images`: `(B, C_img, H, W)` (when `return_slices: True`)
- `control`: `(B, C_ctl, H, W)` where `C_ctl = len(control_columns)`

This avoids ambiguity and makes it explicit what is used as conditioning.

### Model: `ControlNetUNet`

- **Frozen base**: a pretrained `modules.UNet` is loaded from `pretrained_unet_path` and kept frozen.
- **Trainable control branch**: a control encoder produces residual features.
- **Zero-initialized connections**: the residual-to-U-Net connectors are initialized to zeros, so the initial behavior matches the base model.
- **Injection points**:
  - Multiple encoder-scale injections
  - A **mid injection after `bot1`** (512 channels in this U-Net)
- **Resolution weighting**:
  - Each injected residual is multiplied by \(w_i = 64 / h_i\) where \(h_i\) is the spatial resolution of that block.

### CFG (Classifier-Free Guidance)

CFG is applied on the **predicted noise**:

\[
\epsilon = \epsilon_{\text{uncond}} + s \cdot (\epsilon_{\text{cond}} - \epsilon_{\text{uncond}})
\]

- **Training-time CFG dropout**: `cfg_dropout` in the config drops the control tensor to zeros with probability \(p\).
- **Inference-time CFG scale**: `cfg_scale` is used in `Diffusion.sample_controlnet(...)`.

### Training (Volume3D)

1. Train a base DDPM U-Net first (same `size` and `channels` as your ControlNet run).
2. Edit `conf/train_controlnet_volume3d.yml`:
   - `dataset_channels`: image CSV columns (U-Net input)
   - `control_columns`: control CSV columns (ControlNet condition)
   - `pretrained_unet_path`: checkpoint path to the trained base U-Net
3. Run:

```bash
python train_controlnet.py
```

### Sampling

At sampling time, you need to provide a `control` tensor with shape `(n, C_ctl, H, W)`:

```python
samples = diffusion.sample_controlnet(
    model=controlnet_unet,
    n=n,
    channels=channels,
    control=control,
    cfg_scale=7.5,
    pyramid=True,
    discount=0.8,
)
```

### Notes / gotchas

- `pretrained_unet_path` must match the base U-Net architecture (same `channels`, `size`).
- `control` must have the same spatial size as the diffusion input.
- Each entry in `control_columns` must be a valid CSV column with a path to a NIfTI volume.

