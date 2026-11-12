__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import os
import pickle
import random

import lmdb
import nibabel as nib
import numpy as np
import pandas as pd
import skimage.exposure as ex
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as tf
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation, grey_dilation, median_filter
from scipy.signal import medfilt2d
from skimage.measure import label, regionprops
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# TorchIO imports for 3D medical image preprocessing
try:
    import torchio as tio
    TORCHIO_AVAILABLE = True
except ImportError:
    TORCHIO_AVAILABLE = False
    print("Warning: torchio not available. Install with: pip install torchio")


def plot_images(images, mode="RGB"):
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu(),
            cmap="gray",
        )
        plt.show()
    else:
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu()
        )
        plt.show()


def save_images(images, path, mode="RGB", **kwargs):
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)


def upload_images(images, mode="RGB", **kwargs):
    """Creates a numpy array to upload the images to wandb.

    Parameters
    ----------
    images : tensor
        The tensor containing the images
    mode : str, optional
        flag that decides if the images are meant to be RBG or gray scale, by default "RGB"

    Returns
    -------
    numpy.array
        The array that can be uploaded to wandb
    """
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    return ndarr


class LMDBDataset(Dataset):
    """To load slices saved in a Lightning Memory-Mapped Database (LMDB).
    The numpy files in the LMDB are created by the split_healthy.py script.
    These are all preprocessed slices, i.e. they are normalized. Used for training.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(self, directory: str, my_transforms: transforms):
        super().__init__()
        self.directory = directory
        self.transforms = my_transforms
        env = lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(
            self.directory,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not hasattr(self, "txn"):
            self.open_lmdb()

        byteflow = self.txn.get(f"{index:08}".encode("ascii"))
        unpacked = pickle.loads(byteflow)
        tensor = torch.from_numpy(unpacked).float()
        tensor = self.transforms(tensor)
        return tensor


class MRIDataVolume(Dataset):
    """The data set class to load and normalize complete volumes. Used for evaluation.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(
        self,
        df: pd.DataFrame,
        dataset_path: str,
        image_size: int,
        hist: bool,
        shift: bool,
    ):
        self.df = df
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.hist = hist
        self.shift = shift
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if self.shift == True:
            id_ = self.df.loc[idx, self.df.columns[0]]
            patient_number = id_.split("_")[-1]
        else:
            id_ = self.df.loc[idx, self.df.columns[0]]
            patient_number = id_
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, patient_number + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            images.append(img)

        mask_path = os.path.join(self.dataset_path, id_, patient_number + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=float)
        mask[mask > 0.5] = (
            1  # for data sets where the gt masks need to be registered a thr needs to be decided
        )
        mask[mask < 1] = 0
        mask = torch.from_numpy(mask)
        mask = F.interpolate(mask[None, None], [128, 128, 155], mode="nearest-exact")
        mask = mask[0, 0].type(torch.bool)
        if self.hist == True:
            img = np.stack([x for x in images])
            img = hist_norm(img)
        else:
            img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
            img = normalize_volume(img.float())

        volume = torch.zeros(
            img.shape[0], self.image_size, self.image_size, mask.shape[2]
        )
        my_transform = transforms.Resize(self.image_size, antialias=True)
        for i in range(mask.shape[2]):
            volume[:, :, :, i] = my_transform(img[None, :, :, :, i])

        return volume, mask


class Volume3DEvalDataset(Dataset):
    """Dataset class for loading 3D medical volumes with masks for evaluation.
    
    This dataset loads 3D volumes from a CSV file using TorchIO for preprocessing.
    It returns full volumes and optional masks for evaluation purposes.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing volume paths
    image_size : int or tuple
        Target spatial size (height, width) for resizing volumes
    channels : list of str, optional
        List of column names in CSV to use as channels
    mask_column : str, optional
        Column name in CSV containing mask paths. If None, returns dummy masks.
    normalize : bool or str, optional
        Normalization method. Options: True/'percentile', 'zscore', or False
    torchio_transforms : list, optional
        Additional TorchIO transforms to apply
    """
    
    def __init__(
        self,
        csv_path,
        image_size,
        channels=None,
        mask_column=None,
        normalize=True,
        torchio_transforms=None,
    ):
        super().__init__()
        
        if not TORCHIO_AVAILABLE:
            raise ImportError("torchio is required for Volume3DEvalDataset. Install with: pip install torchio")
        
        self.csv_path = csv_path
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.normalize = normalize
        self.mask_column = mask_column
        
        # Load CSV file
        self.df = pd.read_csv(csv_path)
        
        # Determine channels to use
        if channels is None:
            possible_channels = ['B0', 'FA_rigid', 'MD_rigid', 'FA_similar', 'MD_similar', 'B1000']
            self.channels = [ch for ch in possible_channels if ch in self.df.columns]
            if len(self.channels) == 0:
                raise ValueError(f"No valid channel columns found in CSV. Available columns: {self.df.columns.tolist()}")
        else:
            # Handle case where channels might be an integer (from model config)
            if isinstance(channels, int):
                # If it's an integer, ignore it and use auto-detection
                possible_channels = ['B0', 'FA_rigid', 'MD_rigid', 'FA_similar', 'MD_similar', 'B1000']
                self.channels = [ch for ch in possible_channels if ch in self.df.columns]
                if len(self.channels) == 0:
                    raise ValueError(f"No valid channel columns found in CSV. Available columns: {self.df.columns.tolist()}")
            elif isinstance(channels, list):
                self.channels = channels
                missing = [ch for ch in self.channels if ch not in self.df.columns]
                if missing:
                    raise ValueError(f"Channel columns not found in CSV: {missing}")
            else:
                raise ValueError(f"channels must be a list of column names or None, got {type(channels)}")
        
        self.num_channels = len(self.channels)
        
        # Check if mask column exists
        self.has_masks = mask_column is not None and mask_column in self.df.columns
        
        # Filter out rows with missing files
        self.valid_indices = []
        for idx in range(len(self.df)):
            if all(os.path.exists(str(self.df.loc[idx, ch])) for ch in self.channels):
                if self.has_masks:
                    mask_path = str(self.df.loc[idx, mask_column])
                    if os.path.exists(mask_path):
                        self.valid_indices.append(idx)
                else:
                    self.valid_indices.append(idx)
        
        if len(self.valid_indices) == 0:
            raise ValueError("No valid volume files found in CSV. Please check file paths.")
        
        # Build TorchIO preprocessing pipeline
        self.transforms = self._build_transforms(torchio_transforms)
        
        print(f"Loaded {len(self.valid_indices)} valid volumes from {csv_path}")
        print(f"Using {self.num_channels} channels: {self.channels}")
        if self.has_masks:
            print(f"Loading masks from column: {mask_column}")
        else:
            print("No masks provided - will return dummy masks")
    
    def _build_transforms(self, additional_transforms=None):
        """Build TorchIO transform pipeline."""
        transform_list = []
        
        # Resize in-plane H×W; keep third spatial dim (axial) unchanged
        """
        transform_list = [
            tio.Resize(target_shape=(self.image_size[0], self.image_size[1], -1), 
                       image_interpolation='nearest', #"linear", "bspline"   # Custom interpolation for ScalarImage  # stk: suggested from bonbid paper for ADC map and z-score from ADC map
                        label_interpolation='nearest'     # Explicitly setting Nearest Neighbor for LabelMap
                      )
        ]"""
        # Define the dynamic operation
        resize_xy_lambda = tio.Lambda(
            lambda subject: tio.Resize(target_shape=(self.image_size[0], self.image_size[1], subject.shape[-1]), # stk: tio.lambda directly operates on the object inside subject, which is a tensor here.
                                       image_interpolation='nearest', #"linear", "bspline"   # Custom interpolation for ScalarImage  # stk: suggested from bonbid paper for ADC map and z-score from ADC map
                        label_interpolation='nearest'     # Explicitly setting Nearest Neighbor for LabelMap
                                      )(subject)
        )
        # Insert it directly into your global preprocessing block
        transform_list = [
            resize_xy_lambda,  # Changes to 256x256, leaves third dimension untouched
        ]
        
        # Add any additional transforms
        if additional_transforms:
            transform_list.extend(additional_transforms)
        
        return tio.Compose(transform_list) if transform_list else None
    
    def normalize_volume_percentile(self, volume):
        """Normalize volume using 99th percentile normalization per channel."""
        normalized = volume.clone()
        for c in range(volume.shape[0]):
            channel_data = volume[c].reshape(-1)
            foreground = channel_data[channel_data > 0]
            if len(foreground) > 0:
                p99 = torch.quantile(foreground.float(), 0.99)
                if p99 > 0:
                    normalized[c] = normalized[c] / p99
        return normalized
    
    def load_volume(self, volume_idx):
        """Load a full 3D volume for the given index using TorchIO."""
        # Create TorchIO subject with all channels
        subject_dict = {}
        for ch in self.channels:
            vol_path = str(self.df.loc[volume_idx, ch])
            subject_dict[ch] = tio.ScalarImage(vol_path)
        
        subject = tio.Subject(subject_dict)
        
        # Apply transforms if available
        if self.transforms is not None:
            subject = self.transforms(subject)
        
        # Extract data from all channels and stack
        volumes = []
        for ch in self.channels:
            vol_data = subject[ch].data  # Shape: (1, H, W, D)
            if vol_data.ndim == 4:
                vol_data = vol_data.squeeze(0)  # (H, W, D)
            elif vol_data.ndim == 5:
                vol_data = vol_data.squeeze(0).squeeze(-1)  # (H, W, D)
            volumes.append(vol_data)
        
        # Stack channels: (channels, H, W, D)
        volume = torch.stack(volumes, dim=0)
        
        # Apply normalization
        if self.normalize:
            if self.normalize == 'zscore' or self.normalize == 'z-score':
                for c in range(volume.shape[0]):
                    channel_data = volume[c]
                    mean = channel_data.mean()
                    std = channel_data.std()
                    if std > 0:
                        volume[c] = (channel_data - mean) / std
            else:
                volume = self.normalize_volume_percentile(volume)
        
        return volume
    
    def load_mask(self, volume_idx):
        """Load mask for the given volume index."""
        if not self.has_masks:
            # Return dummy mask (all zeros)
            volume = self.load_volume(volume_idx)
            mask = torch.zeros(volume.shape[1], volume.shape[2], volume.shape[3], dtype=torch.bool)
            return mask
        
        mask_path = str(self.df.loc[volume_idx, self.mask_column])
        mask_subject = tio.Subject(mask=tio.LabelMap(mask_path))
        
        # Apply same transforms as volume
        if self.transforms is not None:
            mask_subject = self.transforms(mask_subject)
        
        mask_data = mask_subject.mask.data  # Shape: (1, H, W, D)
        if mask_data.ndim == 4:
            mask_data = mask_data.squeeze(0)  # (H, W, D)
        elif mask_data.ndim == 5:
            mask_data = mask_data.squeeze(0).squeeze(-1)  # (H, W, D)
        
        # Convert to boolean mask (threshold at 0.5)
        mask = (mask_data > 0.5).type(torch.bool)
        
        return mask
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, index):
        volume_idx = self.valid_indices[index]
        volume = self.load_volume(volume_idx)
        mask = self.load_mask(volume_idx)
        
        # Return in format: (channels, height, width, depth), (height, width, depth)
        return volume, mask


class Volume3DControlEvalDataset(Dataset):
    """3D volumes for ControlNet evaluation: aligned image + control NIfTIs and optional mask.

    Same TorchIO resize/normalization conventions as ``Volume3DEvalDataset`` and
    ``Volume3DControlDataset`` (no random flip — eval only).
    Returns ``(volume, control, mask)`` with shapes ``(C_img,H,W,D)``, ``(C_ctl,H,W,D)``, ``(H,W,D)``.
    """

    def __init__(
        self,
        csv_path,
        image_size,
        image_channels,
        control_columns,
        mask_column=None,
        normalize=True,
        torchio_transforms=None,
    ):
        super().__init__()

        if not TORCHIO_AVAILABLE:
            raise ImportError(
                "torchio is required for Volume3DControlEvalDataset. Install with: pip install torchio"
            )
        if not image_channels or not isinstance(image_channels, (list, tuple)):
            raise ValueError("image_channels must be a non-empty list of CSV column names")
        if not control_columns or not isinstance(control_columns, (list, tuple)):
            raise ValueError("control_columns must be a non-empty list of CSV column names")

        self.csv_path = csv_path
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.normalize = normalize
        self.mask_column = mask_column
        self.channels = list(image_channels)
        self.control_columns = list(control_columns)
        self.num_channels = len(self.channels)
        self.num_control_channels = len(self.control_columns)

        self.df = pd.read_csv(csv_path)
        for ch in self.channels:
            if ch not in self.df.columns:
                raise ValueError(f"Image column '{ch}' not in CSV. Have: {self.df.columns.tolist()}")
        for ch in self.control_columns:
            if ch not in self.df.columns:
                raise ValueError(f"Control column '{ch}' not in CSV. Have: {self.df.columns.tolist()}")

        self.has_masks = mask_column is not None and mask_column in self.df.columns

        self.valid_indices = []
        for idx in range(len(self.df)):
            if not all(os.path.exists(str(self.df.loc[idx, ch])) for ch in self.channels):
                continue
            if not all(os.path.exists(str(self.df.loc[idx, ch])) for ch in self.control_columns):
                continue
            if self.has_masks:
                if not os.path.exists(str(self.df.loc[idx, mask_column])):
                    continue
            self.valid_indices.append(idx)

        if len(self.valid_indices) == 0:
            raise ValueError(
                "No valid rows with existing image, control"
                + (" and mask" if self.has_masks else "")
                + " files."
            )

        self.transforms = self._build_transforms(torchio_transforms)

        print(f"Loaded {len(self.valid_indices)} valid volumes (image+control) from {csv_path}")
        print(f"  image channels ({self.num_channels}): {self.channels}")
        print(f"  control channels ({self.num_control_channels}): {self.control_columns}")
        if self.has_masks:
            print(f"  masks: {mask_column}")
        else:
            print("  no mask column — dummy masks for metrics")
    
    def _build_transforms(self, additional_transforms=None):
        """
        transform_list = [
            tio.Resize(target_shape=(self.image_size[0], self.image_size[1], -1), 
                       image_interpolation='nearest', #"linear", "bspline"   # Custom interpolation for ScalarImage  # stk: suggested from bonbid paper for ADC map and z-score from ADC map
                        label_interpolation='nearest'     # Explicitly setting Nearest Neighbor for LabelMap
                      )
        ]"""
        # Define the dynamic operation
        resize_xy_lambda = tio.Lambda(
            lambda subject: tio.Resize(target_shape=(self.image_size[0], self.image_size[1], subject.shape[-1]), # stk: tio.lambda directly operates on the object inside subject, which is a tensor here.
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
        return tio.Compose(transform_list) if transform_list else None

    def normalize_volume_percentile(self, volume):
        normalized = volume.clone()
        for c in range(volume.shape[0]):
            channel_data = volume[c].reshape(-1)
            foreground = channel_data[channel_data > 0]
            if len(foreground) > 0:
                p99 = torch.quantile(foreground.float(), 0.99)
                if p99 > 0:
                    normalized[c] = normalized[c] / p99
        return normalized

    def _extract_hw_d(self, vol_data):
        if vol_data.ndim == 4:
            return vol_data.squeeze(0)
        if vol_data.ndim == 5:
            return vol_data.squeeze(0).squeeze(-1)
        raise ValueError(f"Unexpected volume ndim {vol_data.ndim}")

    def _load_aligned(self, volume_idx):
        subject_dict = {}
        for ch in self.channels:
            subject_dict[ch] = tio.ScalarImage(str(self.df.loc[volume_idx, ch]))
        for ch in self.control_columns:
            subject_dict[ch] = tio.ScalarImage(str(self.df.loc[volume_idx, ch]))
        subject = tio.Subject(subject_dict)
        if self.transforms is not None:
            subject = self.transforms(subject)

        img_vols = []
        for ch in self.channels:
            img_vols.append(self._extract_hw_d(subject[ch].data))
        ctl_vols = []
        for ch in self.control_columns:
            ctl_vols.append(self._extract_hw_d(subject[ch].data))

        volume = torch.stack(img_vols, dim=0)
        control = torch.stack(ctl_vols, dim=0)

        if self.normalize:
            if self.normalize == "zscore" or self.normalize == "z-score":
                for vol in (volume, control):
                    for c in range(vol.shape[0]):
                        channel_data = vol[c]
                        mean = channel_data.mean()
                        std = channel_data.std()
                        if std > 0:
                            vol[c] = (channel_data - mean) / std
            else:
                volume = self.normalize_volume_percentile(volume)
                control = self.normalize_volume_percentile(control)

        return volume, control

    def _load_mask(self, volume_idx, ref_shape_cdhw):
        """ref_shape_cdhw: (C,H,W,D) from image volume."""
        if not self.has_masks:
            _, h, w, d = ref_shape_cdhw[0], ref_shape_cdhw[1], ref_shape_cdhw[2], ref_shape_cdhw[3]
            return torch.zeros(h, w, d, dtype=torch.bool)

        mask_path = str(self.df.loc[volume_idx, self.mask_column])
        mask_subject = tio.Subject(mask=tio.LabelMap(mask_path))
        if self.transforms is not None:
            mask_subject = self.transforms(mask_subject)
        mask_data = self._extract_hw_d(mask_subject.mask.data)
        return (mask_data > 0.5).type(torch.bool)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        volume_idx = self.valid_indices[index]
        volume, control = self._load_aligned(volume_idx)
        mask = self._load_mask(volume_idx, volume.shape)
        return volume, control, mask
    
    
def Brats21(conf):
    if conf["horizontal_flip"] != None:
        my_transforms = transforms.Compose(
            [transforms.RandomHorizontalFlip(conf["horizontal_flip"])]
        )
    else:
        my_transforms = transforms.Compose([])
    dataset = LMDBDataset(conf["dataset_path"], my_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=True,
    )
    return dataloader


def MRI_Volume(conf, hist=False, shift=False):
    df = pd.read_csv(conf["path_to_csv"])
    dataset = MRIDataVolume(
        df, conf["dataset_path"], conf["size"], hist=hist, shift=shift
    )
    dataloader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=False,
    )
    return dataloader


class Volume3DDataset(Dataset):
    """Dataset class for loading 3D medical volumes from NIfTI files for training.
    
    This dataset loads 3D volumes from a CSV file using TorchIO for preprocessing
    and can return either:
    1. Individual 2D slices (for compatibility with 2D models)
    2. Full 3D volumes (if model supports 3D)
    
    The CSV file should contain columns with paths to NIfTI files for different
    modalities/channels (e.g., B0, FA_rigid, MD_rigid, etc.)
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing volume paths
    image_size : int or tuple
        Target spatial size (height, width) for resizing volumes. If int, assumes square.
    channels : list of str, optional
        List of column names in CSV to use as channels. If None, uses:
        ['B0', 'FA_rigid', 'MD_rigid'] or available columns
    return_slices : bool, optional
        If True, returns individual 2D slices. If False, returns full 3D volumes.
        Default: True (for compatibility with existing 2D models)
    normalize : bool or str, optional
        Normalization method. Options:
        - True or 'percentile': 99th percentile normalization per channel
        - 'zscore': Z-score normalization
        - False: No normalization
        Default: True
    horizontal_flip : float, optional
        Probability of random horizontal flip (0.0-1.0). Default: 0.0
    torchio_transforms : list, optional
        Additional TorchIO transforms to apply. Default: None
    """
    
    def __init__(
        self,
        csv_path,
        image_size,
        channels=None,
        return_slices=True,
        normalize=True,
        horizontal_flip=0.0,
        torchio_transforms=None,
    ):
        super().__init__()
        
        if not TORCHIO_AVAILABLE:
            raise ImportError("torchio is required for Volume3DDataset. Install with: pip install torchio")
        
        self.csv_path = csv_path
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.normalize = normalize
        self.horizontal_flip_prob = horizontal_flip
        self.return_slices = return_slices
        
        # Load CSV file
        self.df = pd.read_csv(csv_path)
        
        # Determine channels to use
        if channels is None:
            # Default: try to use common medical imaging modalities
            possible_channels = ['B0', 'FA_rigid', 'MD_rigid', 'FA_similar', 'MD_similar', 'B1000']
            self.channels = [ch for ch in possible_channels if ch in self.df.columns]
            if len(self.channels) == 0:
                raise ValueError(f"No valid channel columns found in CSV. Available columns: {self.df.columns.tolist()}")
        else:
            self.channels = channels
            # Verify all channels exist in CSV
            missing = [ch for ch in self.channels if ch not in self.df.columns]
            if missing:
                raise ValueError(f"Channel columns not found in CSV: {missing}")
        
        self.num_channels = len(self.channels)
        
        # Filter out rows with missing files and build index
        self.valid_indices = []
        self.slice_indices = []  # For return_slices=True: (volume_idx_in_valid_list, slice_idx)
        
        for idx in range(len(self.df)):
            if all(os.path.exists(str(self.df.loc[idx, ch])) for ch in self.channels):
                # Store the index in valid_indices list before appending
                volume_idx_in_valid = len(self.valid_indices)
                self.valid_indices.append(idx)
                
                if return_slices:
                    # Load one volume to get depth using torchio
                    vol_path = str(self.df.loc[idx, self.channels[0]])
                    subject = tio.Subject(
                        image=tio.ScalarImage(vol_path)
                    )
                    vol_data = subject.image.data  # Shape: (1, H, W, D) or (1, H, W, D, T)
                    # Get depth dimension
                    if vol_data.ndim == 4:
                        depth = vol_data.shape[3]  # (1, H, W, D)
                    elif vol_data.ndim == 5:
                        depth = vol_data.shape[3]  # (1, H, W, D, T) - use D
                    else:
                        depth = vol_data.shape[-1]  # Fallback: use last dimension
                    # Add all slices for this volume
                    # volume_idx_in_valid is the index into self.valid_indices list
                    for slice_idx in range(depth):
                        self.slice_indices.append((volume_idx_in_valid, slice_idx))
        
        if len(self.valid_indices) == 0:
            raise ValueError("No valid volume files found in CSV. Please check file paths.")
        
        # Build TorchIO preprocessing pipeline
        self.transforms = self._build_transforms(torchio_transforms)
        
        print(f"Loaded {len(self.valid_indices)} valid volumes from {csv_path}")
        print(f"Using {self.num_channels} channels: {self.channels}")
        if return_slices:
            print(f"Total slices available: {len(self.slice_indices)}")
    
    def _build_transforms(self, additional_transforms=None):
        """Build TorchIO transform pipeline."""
        transform_list = []
        
        # Resize in-plane H×W; keep third spatial dim (axial) unchanged
        # TorchIO spatial order is (s0, s1, s2); use (H, W, -1) for sagittal×coronal×axial
        """
        transform_list = [
            tio.Resize(target_shape=(self.image_size[0], self.image_size[1], -1), 
                       image_interpolation='nearest', #"linear", "bspline"   # Custom interpolation for ScalarImage  # stk: suggested from bonbid paper for ADC map and z-score from ADC map
                        label_interpolation='nearest'     # Explicitly setting Nearest Neighbor for LabelMap
                      )
        ]"""
        # Define the dynamic operation
        resize_xy_lambda = tio.Lambda(
            lambda subject: tio.Resize(target_shape=(self.image_size[0], self.image_size[1], subject.shape[-1]), # stk: tio.lambda directly operates on the object inside subject, which is a tensor here.
                                       image_interpolation='nearest', #"linear", "bspline"   # Custom interpolation for ScalarImage  # stk: suggested from bonbid paper for ADC map and z-score from ADC map
                        label_interpolation='nearest'     # Explicitly setting Nearest Neighbor for LabelMap
                                      )(subject)
        )
        # Insert it directly into your global preprocessing block
        transform_list = [
            resize_xy_lambda,  # Changes to 256x256, leaves third dimension untouched
        ]
        
        # Note: Normalization will be handled after loading since we need per-channel
        # percentile normalization which TorchIO doesn't provide directly
        
        # Add any additional transforms
        if additional_transforms:
            transform_list.extend(additional_transforms)
        
        return tio.Compose(transform_list) if transform_list else None
    
    def normalize_volume_percentile(self, volume):
        """Normalize volume using 99th percentile normalization per channel."""
        normalized = volume.clone()
        for c in range(volume.shape[0]):
            channel_data = volume[c].reshape(-1)
            # Only normalize using foreground (non-zero) voxels
            foreground = channel_data[channel_data > 0]
            if len(foreground) > 0:
                p99 = torch.quantile(foreground.float(), 0.99)
                if p99 > 0:
                    normalized[c] = normalized[c] / p99
        return normalized
    
    def load_volume(self, volume_idx):
        """Load a full 3D volume for the given index using TorchIO."""
        # Create TorchIO subject with all channels
        subject_dict = {}
        for ch in self.channels:
            vol_path = str(self.df.loc[volume_idx, ch])
            subject_dict[ch] = tio.ScalarImage(vol_path)
        
        subject = tio.Subject(subject_dict)

        # Apply transforms if available
        if self.transforms is not None:
            subject = self.transforms(subject)
        
        # Extract data from all channels and stack
        volumes = []
        for ch in self.channels:
            vol_data = subject[ch].data  # Shape: (1, H, W, D) or (1, H, W, D, T)
            # Remove channel dimension: (H, W, D) or (H, W, D, T)
            if vol_data.ndim == 4:
                vol_data = vol_data.squeeze(0)  # (H, W, D)
            elif vol_data.ndim == 5:
                vol_data = vol_data.squeeze(0).squeeze(-1)  # (H, W, D) - remove time dimension
            volumes.append(vol_data)
        
        # Stack channels: (channels, H, W, D)
        volume = torch.stack(volumes, dim=0)
        
        # Reorder to (channels, depth, height, width) for easier processing
        # volume is (channels, H, W, D) -> (channels, D, H, W)
        volume = volume.permute(0, 3, 1, 2)
        
        # Apply normalization
        if self.normalize:
            if self.normalize == 'zscore' or self.normalize == 'z-score':
                # Z-score normalization per channel
                for c in range(volume.shape[0]):
                    channel_data = volume[c]
                    mean = channel_data.mean()
                    std = channel_data.std()
                    if std > 0:
                        volume[c] = (channel_data - mean) / std
            else:
                # Percentile normalization (default)
                volume = self.normalize_volume_percentile(volume)
        
        # Apply random horizontal flip if enabled
        if self.horizontal_flip_prob > 0 and torch.rand(1) < self.horizontal_flip_prob:
            volume = torch.flip(volume, dims=[3])  # Flip along width dimension
        return volume
    
    def __len__(self):
        if self.return_slices:
            return len(self.slice_indices)
        else:
            return len(self.valid_indices)
    
    def __getitem__(self, index):
        if self.return_slices:
            # Return a single 2D slice
            volume_idx_in_valid, slice_idx = self.slice_indices[index]
            volume_idx = self.valid_indices[volume_idx_in_valid]
            volume = self.load_volume(volume_idx)
            
            # Extract the slice: (channels, depth, height, width) -> (channels, height, width)
            slice_data = volume[:, slice_idx, :, :]
            
            return slice_data
        else:
            # Return full 3D volume
            volume = self.load_volume(self.valid_indices[index])
            return volume


class Volume3DControlDataset(Dataset):
    """Dataset for 3D volumes with separate image and control columns.

    - Image channels come from one set of CSV columns.
    - Control channels come from a separate set of CSV columns.
    Both are loaded with the same TorchIO transforms so they remain aligned.
    """

    def __init__(
        self,
        csv_path,
        image_size,
        image_channels,
        control_columns,
        return_slices=True,
        normalize=True,
        horizontal_flip=0.0,
        torchio_transforms=None,
    ):
        super().__init__()

        if not TORCHIO_AVAILABLE:
            raise ImportError(
                "torchio is required for Volume3DControlDataset. "
                "Install with: pip install torchio"
            )

        if image_channels is None or len(image_channels) == 0:
            raise ValueError("image_channels must be a non-empty list of column names")
        if control_columns is None or len(control_columns) == 0:
            raise ValueError("control_columns must be a non-empty list of column names")

        self.csv_path = csv_path
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.normalize = normalize
        self.horizontal_flip_prob = horizontal_flip
        self.return_slices = return_slices

        self.df = pd.read_csv(csv_path)

        # Verify columns
        missing_img = [ch for ch in image_channels if ch not in self.df.columns]
        missing_ctl = [ch for ch in control_columns if ch not in self.df.columns]
        if missing_img:
            raise ValueError(
                f"Image channel columns not found in CSV: {missing_img}"
            )
        if missing_ctl:
            raise ValueError(
                f"Control channel columns not found in CSV: {missing_ctl}"
            )

        self.image_channels = list(image_channels)
        self.control_columns = list(control_columns)
        self.num_image_channels = len(self.image_channels)
        self.num_control_channels = len(self.control_columns)

        # Build index of valid rows (all image + control files must exist)
        self.valid_indices = []
        self.slice_indices = []

        for idx in range(len(self.df)):
            if all(os.path.exists(str(self.df.loc[idx, ch])) for ch in self.image_channels) and all(
                os.path.exists(str(self.df.loc[idx, ch])) for ch in self.control_columns
            ):
                volume_idx_in_valid = len(self.valid_indices)
                self.valid_indices.append(idx)

                if return_slices:
                    # Use first image channel to determine depth
                    vol_path = str(self.df.loc[idx, self.image_channels[0]])
                    subject = tio.Subject(image=tio.ScalarImage(vol_path))
                    vol_data = subject.image.data
                    if vol_data.ndim == 4:
                        depth = vol_data.shape[3]
                    elif vol_data.ndim == 5:
                        depth = vol_data.shape[3]
                    else:
                        depth = vol_data.shape[-1]
                    for slice_idx in range(depth):
                        self.slice_indices.append((volume_idx_in_valid, slice_idx))

        if len(self.valid_indices) == 0:
            raise ValueError(
                "No valid volume files found for image/control columns in CSV."
            )

        # Shared TorchIO preprocessing
        self.transforms = self._build_transforms(torchio_transforms)

        print(f"Loaded {len(self.valid_indices)} valid volumes with control from {csv_path}")
        print(f"Using {self.num_image_channels} image channels: {self.image_channels}")
        print(f"Using {self.num_control_channels} control channels: {self.control_columns}")
        if return_slices:
            print(f"Total slices available: {len(self.slice_indices)}")

    def _build_transforms(self, additional_transforms=None):
        # (H, W, -1): resize in-plane; axial (third dim) unchanged
        """
        transform_list = [
            tio.Resize(target_shape=(self.image_size[0], self.image_size[1], -1), 
                       image_interpolation='nearest', #"linear", "bspline"   # Custom interpolation for ScalarImage  # stk: suggested from bonbid paper for ADC map and z-score from ADC map
                        label_interpolation='nearest'     # Explicitly setting Nearest Neighbor for LabelMap
                      )
        ]"""
        # Define the dynamic operation
        resize_xy_lambda = tio.Lambda(
            lambda subject: tio.Resize(target_shape=(self.image_size[0], self.image_size[1], subject.shape[-1]), # stk: tio.lambda directly operates on the object inside subject, which is a tensor here.
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
        return tio.Compose(transform_list) if transform_list else None

    def _normalize_volume_percentile(self, volume):
        normalized = volume.clone()
        for c in range(volume.shape[0]):
            channel_data = volume[c].reshape(-1)
            foreground = channel_data[channel_data > 0]
            if len(foreground) > 0:
                p99 = torch.quantile(foreground.float(), 0.99)
                if p99 > 0:
                    normalized[c] = normalized[c] / p99
        return normalized

    def _load_image_and_control(self, volume_idx):
        # Build TorchIO subject with both image and control channels
        subject_dict = {}
        for ch in self.image_channels:
            subject_dict[ch] = tio.ScalarImage(str(self.df.loc[volume_idx, ch]))
        for ch in self.control_columns:
            subject_dict[ch] = tio.ScalarImage(str(self.df.loc[volume_idx, ch]))

        subject = tio.Subject(subject_dict)
        if self.transforms is not None:
            subject = self.transforms(subject)

        # Extract image channels
        img_vols = []
        for ch in self.image_channels:
            vol_data = subject[ch].data
            if vol_data.ndim == 4:
                vol_data = vol_data.squeeze(0)
            elif vol_data.ndim == 5:
                vol_data = vol_data.squeeze(0).squeeze(-1)
            img_vols.append(vol_data)

        # Extract control channels
        ctl_vols = []
        for ch in self.control_columns:
            vol_data = subject[ch].data
            if vol_data.ndim == 4:
                vol_data = vol_data.squeeze(0)
            elif vol_data.ndim == 5:
                vol_data = vol_data.squeeze(0).squeeze(-1)
            ctl_vols.append(vol_data)

        img = torch.stack(img_vols, dim=0)      # (C_img, H, W, D)
        ctl = torch.stack(ctl_vols, dim=0)      # (C_ctl, H, W, D)

        # Reorder to (C, D, H, W)
        img = img.permute(0, 3, 1, 2)
        ctl = ctl.permute(0, 3, 1, 2)

        # Normalize (percentile or z-score) independently for image and control
        if self.normalize:
            if self.normalize == "zscore" or self.normalize == "z-score":
                for vol in (img, ctl):
                    for c in range(vol.shape[0]):
                        channel_data = vol[c]
                        mean = channel_data.mean()
                        std = channel_data.std()
                        if std > 0:
                            vol[c] = (channel_data - mean) / std
            else:
                img = self._normalize_volume_percentile(img)
                ctl = self._normalize_volume_percentile(ctl)

        # Random horizontal flip (same for image and control)
        if self.horizontal_flip_prob > 0 and torch.rand(1) < self.horizontal_flip_prob:
            img = torch.flip(img, dims=[3])
            ctl = torch.flip(ctl, dims=[3])

        return img, ctl

    def __len__(self):
        if self.return_slices:
            return len(self.slice_indices)
        else:
            return len(self.valid_indices)

    def __getitem__(self, index):
        if self.return_slices:
            volume_idx_in_valid, slice_idx = self.slice_indices[index]
            volume_idx = self.valid_indices[volume_idx_in_valid]
            img, ctl = self._load_image_and_control(volume_idx)
            img_slice = img[:, slice_idx, :, :]
            ctl_slice = ctl[:, slice_idx, :, :]
            return img_slice, ctl_slice
        else:
            img, ctl = self._load_image_and_control(self.valid_indices[index])
            return img, ctl


def Volume3D(conf):
    """Create a DataLoader for 3D volumes from CSV file using TorchIO.
    
    Parameters
    ----------
    conf : dict
        Configuration dictionary with keys:
        - csv_path: Path to CSV file with volume paths
        - image_size: Target spatial size (int or tuple)
        - channels: List of channel column names (optional)
        - return_slices: Whether to return slices or full volumes (default: True)
        - normalize: Normalization method - True/'percentile' for 99th percentile,
                     'zscore' for z-score, or False for no normalization (default: True)
        - horizontal_flip: Probability of horizontal flip (0.0-1.0) (default: 0.0)
        - torchio_transforms: Additional TorchIO transforms to apply (optional)
        - batch_size: Batch size
        - workers: Number of data loader workers
    """
    horizontal_flip_prob = conf.get("horizontal_flip", 0.0)
    if horizontal_flip_prob is not None and horizontal_flip_prob == "":
        horizontal_flip_prob = 0.0
    if horizontal_flip_prob is None:
        horizontal_flip_prob = 0.0
    
    # Use dataset_channels if specified, otherwise try channels (but check if it's a list)
    dataset_channels = conf.get("dataset_channels", None)
    if dataset_channels is None:
        # Check if channels is a list (dataset channels) or int (model channels)
        channels_val = conf.get("channels", None)
        if isinstance(channels_val, list):
            dataset_channels = channels_val
        else:
            dataset_channels = None  # Will use auto-detection
    
    dataset = Volume3DDataset(
        csv_path=conf["csv_path"],
        image_size=conf["image_size"],
        channels=dataset_channels,
        return_slices=conf.get("return_slices", True),
        normalize=conf.get("normalize", True),
        horizontal_flip=horizontal_flip_prob,
        torchio_transforms=conf.get("torchio_transforms", None),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=True,
    )
    return dataloader


def Volume3D_Control(conf):
    """Create a DataLoader for 3D volumes with separate control columns.

    Uses Volume3DControlDataset and returns (images, control) batches:
    - images: (B, C_img, H, W) or (B, C_img, D, H, W) depending on return_slices
    - control: (B, C_ctl, H, W) or (B, C_ctl, D, H, W)
    """
    horizontal_flip_prob = conf.get("horizontal_flip", 0.0)
    if horizontal_flip_prob is not None and horizontal_flip_prob == "":
        horizontal_flip_prob = 0.0
    if horizontal_flip_prob is None:
        horizontal_flip_prob = 0.0

    dataset_channels = conf.get("dataset_channels", None)
    if dataset_channels is None:
        channels_val = conf.get("channels", None)
        if isinstance(channels_val, list):
            dataset_channels = channels_val
        else:
            dataset_channels = None
    control_columns = conf.get("control_columns", None)

    dataset = Volume3DControlDataset(
        csv_path=conf["csv_path"],
        image_size=conf["image_size"],
        image_channels=dataset_channels,
        control_columns=control_columns,
        return_slices=conf.get("return_slices", True),
        normalize=conf.get("normalize", True),
        horizontal_flip=horizontal_flip_prob,
        torchio_transforms=conf.get("torchio_transforms", None),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=True,
    )
    return dataloader


def Volume3D_Eval(conf):
    """Create a DataLoader for 3D volumes with masks for evaluation using TorchIO.
    
    Parameters
    ----------
    conf : dict
        Configuration dictionary with keys:
        - csv_path: Path to CSV file with volume paths
        - image_size: Target spatial size (int or tuple)
        - dataset_channels: List of channel column names (optional)
        - mask_column: Column name in CSV containing mask paths (optional)
        - normalize: Normalization method - True/'percentile' for 99th percentile,
                     'zscore' for z-score, or False for no normalization (default: True)
        - torchio_transforms: Additional TorchIO transforms to apply (optional)
        - batch_size: Batch size (typically 1 for evaluation)
        - workers: Number of data loader workers
    """
    # Use dataset_channels if specified, otherwise try channels (but check if it's a list)
    dataset_channels = conf.get("dataset_channels", None)
    if dataset_channels is None:
        # Check if channels is a list (dataset channels) or int (model channels)
        channels_val = conf.get("channels", None)
        if isinstance(channels_val, list):
            dataset_channels = channels_val
        else:
            dataset_channels = None  # Will use auto-detection
    
    dataset = Volume3DEvalDataset(
        csv_path=conf["csv_path"],
        image_size=conf["image_size"],
        channels=dataset_channels,
        mask_column=conf.get("mask_column", None),
        normalize=conf.get("normalize", True),
        torchio_transforms=conf.get("torchio_transforms", None),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=False,  # No shuffling for evaluation
    )
    return dataloader


def Volume3D_Control_Eval(conf):
    """DataLoader for ControlNet evaluation: (image volume, control volume, mask) per subject."""
    dataset_channels = conf.get("dataset_channels")
    control_columns = conf.get("control_columns")
    if not dataset_channels:
        raise ValueError("conf['dataset_channels'] is required for Volume3D_Control_Eval.")
    if not control_columns:
        raise ValueError("conf['control_columns'] is required for ControlNet evaluation.")

    dataset = Volume3DControlEvalDataset(
        csv_path=conf["csv_path"],
        image_size=conf["image_size"],
        image_channels=dataset_channels,
        control_columns=control_columns,
        mask_column=conf.get("mask_column", None),
        normalize=conf.get("normalize", True),
        torchio_transforms=conf.get("torchio_transforms", None),
    )
    return DataLoader(
        dataset,
        batch_size=conf["batch_size"],
        num_workers=conf["workers"],
        shuffle=False,
    )


def normalize_volume(images):
    """Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.

    Parameters
    ----------
    images : tensor
        Tensor containing the volumes

    Returns
    -------
    tensor
        The tensor containing the normalized volumes.
    """
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :].reshape(-1)
        i_ = i_[i_ > 0]
        p_99 = torch.quantile(i_, 0.99)
        images[modality, :, :, :] /= p_99

    return images


def hist_norm(images):
    """Applies histogram normalization to the volumes.

    Parameters
    ----------
    images : tensor
        Tensor containing the volumes

    Returns
    -------
    tensor
        The tensor with the histogram normalized.
    """
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :]
        mask = np.zeros_like(i_)
        mask[i_ > 0] = 1
        i_ = i_ / np.max(i_)
        i_ = ex.equalize_hist(i_.astype(np.float32), mask=mask, nbins=256)
        i_ *= mask
        images[modality, :, :, :] = i_
    return torch.Tensor(images)


def dice_stitch(pred, target):
    pred_sum = pred.view(-1).sum()
    target_sum = target.view(-1).sum()
    intersection = pred.view(-1).float() @ target.view(-1).float()
    dice = (2 * intersection) / (pred_sum + target_sum)
    return dice


def dice(pred, truth):
    num = 2 * ((pred * truth).sum(dim=(1, 2, 3)).type(torch.float))
    den = (pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))).type(torch.float)
    return num / den


def coarse_noise(n, channels, device, noise_size=16, noise_std=0.2, image_size=128):
    noise = torch.normal(
        mean=torch.zeros(n, channels, noise_size, noise_size), std=noise_std
    ).to(device)
    noise = F.interpolate(
        noise,
        size=(image_size, image_size),
        mode="bilinear",
        antialias=False,
        align_corners=True,
    )
    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(image_size))
    roll_y = random.choice(range(image_size))
    noise = torch.roll(noise, shifts=[roll_x, roll_y], dims=[2, 3])
    return noise


def pyramid_noise_like(n, channels, image_size, discount, device):
    u = transforms.Resize(image_size, antialias=True)
    noise = torch.randn((n, channels, image_size, image_size)).to(device)
    w = image_size
    h = image_size
    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(n, channels, w, h).to(device)) * discount**i
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


def random_transform_vectorized(tensor):
    # Random rotations for the entire batch
    angles = torch.randint(0, 24, (tensor.size(0),)) * 15
    sliced_tensor = torch.stack(
        [tf.rotate(tensor[i, :, :, :], angles[i].item()) for i in range(tensor.size(0))]
    )

    # Random horizontal flip for the entire batch
    flip_h = torch.rand(sliced_tensor.size(0)) < 0.5
    for i in range(sliced_tensor.size(0)):
        if flip_h[i]:
            sliced_tensor[i, :, :, :] = tf.hflip(sliced_tensor[i, :, :, :])

    # Random vertical flip for the entire batch
    flip_v = torch.rand(sliced_tensor.size(0)) < 0.5
    for i in range(sliced_tensor.size(0)):
        if flip_v[i]:
            sliced_tensor[i, :, :, :] = tf.vflip(sliced_tensor[i, :, :, :])

    return sliced_tensor


def median_filter_2D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        for j in range(volume.shape[1]):
            volume[i, j, :, :] = medfilt2d(volume[i, j, :, :], kernel_size=kernelsize)
    return torch.Tensor(volume)


def median_filter_3D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)


def connected_components_3d(volume):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Connected components")
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)
        for prop in props:
            if prop["filled_area"] <= 20:
                volume[i, cc_volume == prop["label"]] = 0
    return torch.Tensor(volume)


def gray_dilation(volume, kernelsize=3):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Gray Dilation")
    for i in pbar:
        volume[i] = grey_dilation(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)


def bin_dilation(volume, structure):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Binray Dilation")
    for i in pbar:
        volume[i] = binary_dilation(volume[i], structure=structure)
    return torch.Tensor(volume)


def norm_tensor(tensor):
    my_max = torch.max(tensor)
    my_min = torch.min(tensor)
    my_tensor = (tensor - my_min) / (my_max - my_min)
    return my_tensor


def gmean(input_x, dim, keepdim=False):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim, keepdim=keepdim))


class BratsDataset(Dataset):
    """DO NOT USE DEPRECATED CODE.
    The data set class to load individual slices from the files containing the volumes.
    A .csv is requirred that specifies the slices to load from the volume.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """

    def __init__(
        self,
        df: pd.DataFrame,
        my_transforms: transforms,
        dataset_path: str,
        image_size: int,
        hist: bool,
    ):
        self.df = df
        self.transforms = my_transforms
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.hist = hist
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, self.df.columns[0]]
        images = []
        slice = self.df.loc[idx, "Slice"]
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, id_ + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            images.append(img)

        mask_path = os.path.join(self.dataset_path, id_, id_ + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj[:, :, slice], dtype=int)
        mask[mask >= 1] = 1  # mask contains the labels 1, 2, and 4 for BraTS
        mask = torch.from_numpy(mask)
        mask = mask[None, :, :]
        if self.hist == True:
            img = np.stack([x for x in images])
            img = hist_norm(img)
        else:
            img = torch.stack([torch.from_numpy(x) for x in images], dim=0)
            img = normalize_volume(img.float())
        img = img[:, :, :, slice]
        img = self.transforms(img)
        my_transform = transforms.Resize(self.image_size, antialias=True)
        mask = my_transform(mask)
        return img, mask


def make_dicts(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
