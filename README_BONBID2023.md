## BONBID2023 training (healthy slices only)

The clinical CSV (e.g. `BONBID2023_clinicaldata_train_no_lesion_slices.csv`) lists, per subject:

- NIfTI paths: `adc_affine`, `zmap_affine`, `lesion_affine`, etc.
- **`axial_slices_no_lesion`**: comma-separated slice indices (axial) **without** lesion — used for training.
- **`axial_slices_lesion`**: slices with lesion — **not** used for normative training.

### Dataset module

- **`datasets/bonbid2023.py`**
  - `BONBID2023HealthySliceDataset`: loads only slices from `axial_slices_no_lesion`.
  - `bonbid2023_train_dataloader(conf)`: images only (DDPM).
  - `bonbid2023_control_dataloader(conf)`: `(images, control)` for ControlNet.

### Training commands

1. **DDPM** (healthy slices):

```bash
python train_bonbid.py
```

Config: `train_bonbid_affine_double_space.yml` — set `csv_path`, `dataset_channels`, `channels`, `size`. Checkpoints are written to `models/<run_name>/{epoch}_ckpt.pt` and `{epoch}_ema_ckpt.pt`.

2. **ControlNet** (same slice filter; provide `control_columns`, e.g. `zmap_affine`):

```bash
python train_controlnet_bonbid.py
```

Config: `train_controlnet_bonbid_affine_double_space.yml` — set `pretrained_unet_path` to an EMA checkpoint from step 1 (`{epoch}_ema_ckpt.pt`). The frozen base U-Net is loaded from that file; only the control branch is trained.

### Evaluation (volume-level AUPR / Dice)

- **Notebook:** `notebooks/eval_bonbid_diffusion.ipynb` — loads `Volume3D_Eval` with a **test CSV** (image columns + lesion mask column), runs `normative_diffusion`, saves metrics like `eval.py`.
- **Config template:** `conf/eval_bonbid.yml` — set `csv_path`, `mask_column` (e.g. `lesion_affine`), `model` (prefer `*_ema_ckpt.pt`), and match training hyperparameters (`dataset_channels`, `image_size`, `pyramid`, etc.).

### Notes

- Update **file paths** in the CSV if your data live elsewhere.
- `channels` must match `len(dataset_channels)`.
- Horizontal flip is shared between image and control tensors when both are loaded.
