"""
BONBID2023 CSV: one row per subject with NIfTI paths and comma-separated axial slice lists.

Uses only slices listed in ``axial_slices_no_lesion`` (healthy / no lesion) for training
the DDPM and ControlNet, as in the clinical CSV (e.g. ``BONBID2023_clinicaldata_train_no_lesion_slices.csv``).

Expected columns (configurable):
- Image channels: e.g. ``adc_affine``, optionally multiple modalities.
- Optional control channels: e.g. ``zmap_affine`` for ControlNet conditioning.
- Slice list: ``axial_slices_no_lesion`` — comma-separated integer slice indices along the
  depth axis (same convention as TorchIO / Volume3D after ``(C, D, H, W)``).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import torchio as tio
except ImportError as e:
    raise ImportError(
        "datasets.bonbid2023 requires torchio. Install with: pip install torchio"
    ) from e


def _parse_slice_indices(cell) -> List[int]:
    """Parse '0,1,2,58,59' or similar into a list of ints."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            try:
                out.append(int(part))
            except ValueError:
                continue
    return out


class BONBID2023HealthySliceDataset(Dataset):
    """
    2D slices from BONBID volumes, **only** from indices in the no-lesion column.

    Returns either:
    - ``tensor (C_img, H, W)`` if ``control_columns`` is None
    - ``(images, control)`` each ``(C, H, W)`` if ``control_columns`` is set
    """

    def __init__(
        self,
        csv_path: str,
        image_size: Union[int, Sequence[int]],
        image_columns: Sequence[str],
        control_columns: Optional[Sequence[str]] = None,
        slices_no_lesion_column: str = "axial_slices_no_lesion",
        normalize: Union[bool, str] = True,
        horizontal_flip: float = 0.0,
        torchio_transforms=None,
    ):
        super().__init__()
        self.csv_path = csv_path
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)
        self.image_columns = list(image_columns)
        self.control_columns = (
            list(control_columns) if control_columns is not None else None
        )
        self.slices_column = slices_no_lesion_column
        self.normalize = normalize
        self.horizontal_flip_prob = float(horizontal_flip or 0.0)
        self.df = pd.read_csv(csv_path)

        for c in self.image_columns:
            if c not in self.df.columns:
                raise ValueError(f"Missing image column '{c}' in CSV. Have: {self.df.columns.tolist()}")
        if self.control_columns:
            for c in self.control_columns:
                if c not in self.df.columns:
                    raise ValueError(
                        f"Missing control column '{c}' in CSV. Have: {self.df.columns.tolist()}"
                    )
        if self.slices_column not in self.df.columns:
            raise ValueError(
                f"Missing slices column '{self.slices_column}'. Have: {self.df.columns.tolist()}"
            )
        
        self.transforms = self._build_transforms(torchio_transforms)

        # Rows with existing files and at least one no-lesion slice
        self.valid_indices: List[int] = []
        self.slice_indices: List[tuple] = []  # (idx_into_valid, slice_z)

        for idx in range(len(self.df)):
            if not all(
                os.path.exists(str(self.df.loc[idx, ch])) for ch in self.image_columns
            ):
                continue
            if self.control_columns and not all(
                os.path.exists(str(self.df.loc[idx, ch]))
                for ch in self.control_columns
            ):
                continue
            zs = _parse_slice_indices(self.df.loc[idx, self.slices_column])
            if not zs:
                continue
            vol_i = len(self.valid_indices)
            self.valid_indices.append(idx)
            for z in zs:
                self.slice_indices.append((vol_i, z))

        if len(self.slice_indices) == 0:
            raise ValueError(
                "No valid (volume, slice) pairs. Check paths and "
                f"'{self.slices_column}' slice lists."
            )

        print(f"BONBID2023: {len(self.valid_indices)} subjects, {len(self.slice_indices)} healthy slices")
        print(f"  image columns: {self.image_columns}")
        if self.control_columns:
            print(f"  control columns: {self.control_columns}")

    def _build_transforms(self, additional_transforms=None):
        # (H, W, -1): resize in-plane; axial slice indices stay aligned with third dim
        """
        transform_list = [
            tio.Resize(target_shape=(self.image_size[0], self.image_size[1], -1), 
                       image_interpolation='nearest', #"linear", "bspline"   # Custom interpolation for ScalarImage  # stk: suggested from bonbid paper for ADC map and z-score from ADC map
                        label_interpolation='nearest'     # Explicitly setting Nearest Neighbor for LabelMap
                      )
        ]"""
        # Define the dynamic operation
        resize_xy_lambda = tio.Lambda(
            lambda subject: tio.Resize(target_shape=(self.image_size[0], self.image_size[1], subject.shape[-1]),
                                       image_interpolation='nearest', #"linear", "bspline"   # Custom interpolation for ScalarImage  # stk: suggested from bonbid paper for ADC map and z-score from ADC map
                        label_interpolation='nearest'     # Explicitly setting Nearest Neighbor for LabelMap
                                      )(subject)
        )
        # Insert it directly into your global preprocessing block
        transform_list = [
            resize_xy_lambda,  # Changes to 256x256, leaves third dimension untouched
        ]
        
        if additional_transforms:
            transform_list.extend(additional_transforms)
        return tio.Compose(transform_list)

    def _normalize_volume_percentile(self, volume: torch.Tensor) -> torch.Tensor:
        normalized = volume.clone()
        for c in range(volume.shape[0]):
            channel_data = volume[c].reshape(-1)
            foreground = channel_data[channel_data > 0]
            if len(foreground) > 0:
                p99 = torch.quantile(foreground.float(), 0.99)
                if p99 > 0:
                    normalized[c] = normalized[c] / p99
        return normalized

    def _load_stack(
        self, row_idx: int, columns: List[str], flip_width: bool
    ) -> torch.Tensor:
        subject_dict = {ch: tio.ScalarImage(str(self.df.loc[row_idx, ch])) for ch in columns}
        subject = tio.Subject(subject_dict)
        if self.transforms is not None:
            subject = self.transforms(subject)

        vols = []
        for ch in columns:
            vol_data = subject[ch].data
            if vol_data.ndim == 4:
                vol_data = vol_data.squeeze(0)
            elif vol_data.ndim == 5:
                vol_data = vol_data.squeeze(0).squeeze(-1)
            vols.append(vol_data)
        volume = torch.stack(vols, dim=0)
        volume = volume.permute(0, 3, 1, 2)

        if self.normalize:
            if self.normalize in ("zscore", "z-score"):
                for c in range(volume.shape[0]):
                    chd = volume[c]
                    std = chd.std()
                    if std > 0:
                        volume[c] = (chd - chd.mean()) / std
            else:
                volume = self._normalize_volume_percentile(volume)

        if flip_width:
            volume = torch.flip(volume, dims=[3])
        return volume

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, index: int):
        vol_i, z = self.slice_indices[index]
        row_idx = self.valid_indices[vol_i]

        flip_w = (
            self.horizontal_flip_prob > 0
            and torch.rand(1).item() < self.horizontal_flip_prob
        )

        img_vol = self._load_stack(row_idx, self.image_columns, flip_w)
        sl = img_vol[:, z, :, :]

        if self.control_columns is None:
            return sl

        ctl_vol = self._load_stack(row_idx, self.control_columns, flip_w)
        ctl = ctl_vol[:, z, :, :]
        return sl, ctl


def bonbid2023_train_dataloader(conf) -> DataLoader:
    """DDPM training: yields only image tensors (healthy slices)."""
    dataset_channels = conf.get("dataset_channels")
    if not dataset_channels:
        raise ValueError("conf['dataset_channels'] must list CSV columns for the image (e.g. adc_affine).")

    ds = BONBID2023HealthySliceDataset(
        csv_path=conf["csv_path"],
        image_size=conf["image_size"],
        image_columns=dataset_channels,
        control_columns=None,
        slices_no_lesion_column=conf.get(
            "bonbid_slices_no_lesion_column", "axial_slices_no_lesion"
        ),
        normalize=conf.get("normalize", True),
        horizontal_flip=conf.get("horizontal_flip", 0.0),
        torchio_transforms=conf.get("torchio_transforms", None),
    )
    return DataLoader(
        ds,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=True,
    )


def bonbid2023_control_dataloader(conf) -> DataLoader:
    """ControlNet training: yields (images, control) on healthy slices only."""
    dataset_channels = conf.get("dataset_channels")
    control_columns = conf.get("control_columns")
    if not dataset_channels:
        raise ValueError("conf['dataset_channels'] is required.")
    if not control_columns:
        raise ValueError("conf['control_columns'] is required for ControlNet.")

    ds = BONBID2023HealthySliceDataset(
        csv_path=conf["csv_path"],
        image_size=conf["image_size"],
        image_columns=dataset_channels,
        control_columns=control_columns,
        slices_no_lesion_column=conf.get(
            "bonbid_slices_no_lesion_column", "axial_slices_no_lesion"
        ),
        normalize=conf.get("normalize", True),
        horizontal_flip=conf.get("horizontal_flip", 0.0),
        torchio_transforms=conf.get("torchio_transforms", None),
    )
    return DataLoader(
        ds,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=True,
    )
