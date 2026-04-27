"""
task2_dataset.py — Data loading and preprocessing for 3D Brain Tumor Segmentation.

This module handles:
1. Downloading the BraTS 2020 dataset via kagglehub.
2. Creating a PyTorch Dataset for 3D NIfTI volumes.
3. Implementing 3D data augmentations (rotations, flips, elastic deformations).
4. Extracting randomized 3D patches to manage memory constraints.
"""

import os
import glob
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import config as cfg

# MONAI for 3D Medical Imaging augmentations and transforms
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandAffined,
    RandFlipd,
    RandRotate90d,
    Rand3DElasticd,
    RandGaussianNoised,
    RandAdjustContrastd,
    EnsureTyped,
    ConcatItemsd,
    MapTransform,
)

class ConvertBraTSLabelsd(MapTransform):
    """
    BraTS labels are typically:
    0: Background
    1: Necrotic and non-enhancing tumor core (NCR/NET)
    2: Peritumoral edema (ED)
    4: GD-enhancing tumor (ET)
    
    We convert these to a simpler multi-class or binary structure based on needs.
    Here we map: 0->0, 1->1, 2->2, 4->3 to have continuous class indices [0, 1, 2, 3].
    """
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # Map label 4 to 3
            d[key][d[key] == 4] = 3
        return d

def get_brats_data_dicts(data_dir):
    """
    Scans the downloaded dataset directory and creates a list of dictionaries
    for MONAI dataset, combining the 4 modalities (FLAIR, T1, T1ce, T2) and the mask.
    """
    # Find all patient folders in the training directory
    train_dir = os.path.join(data_dir, 'BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData')
    
    # Fallback if structure is slightly different
    if not os.path.exists(train_dir):
        # Look for the first directory that contains 'TrainingData'
        found = False
        for root, dirs, files in os.walk(data_dir):
            if 'MICCAI_BraTS2020_TrainingData' in root:
                train_dir = root
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Could not find BraTS training data in {data_dir}")

    patient_folders = sorted(glob.glob(os.path.join(train_dir, "BraTS20*")))
    
    data_dicts = []
    for patient_folder in patient_folders:
        patient_id = os.path.basename(patient_folder)
        # Check if all required files exist
        flair = os.path.join(patient_folder, f"{patient_id}_flair.nii")
        t1 = os.path.join(patient_folder, f"{patient_id}_t1.nii")
        t1ce = os.path.join(patient_folder, f"{patient_id}_t1ce.nii")
        t2 = os.path.join(patient_folder, f"{patient_id}_t2.nii")
        seg = os.path.join(patient_folder, f"{patient_id}_seg.nii")
        
        if all(os.path.exists(f) for f in [flair, t1, t1ce, t2, seg]):
            data_dicts.append({
                "flair": flair,
                "t1": t1,
                "t1ce": t1ce,
                "t2": t2,
                "label": seg
            })
            
    return data_dicts

def download_and_prepare_dataset():
    """
    Downloads the BraTS 2020 dataset using kagglehub and prepares the data dictionaries.
    """
    print("Downloading BraTS 2020 Dataset (this may take a while, ~4GB)...")
    path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")
    print(f"Dataset downloaded to: {path}")
    
    data_dicts = get_brats_data_dicts(path)
    print(f"Found {len(data_dicts)} valid patient volumes.")
    
    # Split into Train / Val / Test (80 / 10 / 10)
    np.random.seed(42)
    np.random.shuffle(data_dicts)
    
    num_total = len(data_dicts)
    num_train = int(num_total * 0.8)
    num_val = int(num_total * 0.1)
    
    train_files = data_dicts[:num_train]
    val_files = data_dicts[num_train:num_train+num_val]
    test_files = data_dicts[num_train+num_val:]
    
    return train_files, val_files, test_files

def get_transforms(patch_size=(96, 96, 96)):
    """
    Defines the data preprocessing and augmentation pipelines for training and validation.
    The augmentation meets the brief's requirement: rotations, flips, elastic deformations.
    
    Memory-optimised: uses num_samples=1 and reduced elastic deformation probability
    to avoid OOM on AMD iGPUs with shared system RAM.
    """
    keys = ["flair", "t1", "t1ce", "t2", "label"]
    image_keys = ["flair", "t1", "t1ce", "t2"]
    
    # Base transforms applied to all splits
    base_transforms = [
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        ConvertBraTSLabelsd(keys=["label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # Normalize each modality independently
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Ensure the image is at least the size of the patch before random cropping
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
    ]
    
    # Training augmentations (memory-optimised)
    train_transforms = Compose(base_transforms + [
        # Extract 1 patch per volume (reduced from 2 to save memory)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1, neg=1, # 50% chance to center on tumor, 50% on background
            num_samples=1, # Single patch per volume to save memory
            image_key="image",
            image_threshold=0,
        ),
        # Brief requirements: flips, rotations, elastic deformations
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        # Elastic deformation — increased probability for stronger regularization
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(3, 5),
            magnitude_range=(30, 80),
            prob=0.2,
            mode=("bilinear", "nearest"),
        ),
        # Intensity augmentations to prevent memorization of pixel values
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1),
        RandAdjustContrastd(keys=["image"], prob=0.2, gamma=(0.5, 2.0)),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    # Validation transforms (No augmentation, just crop to fixed size or use sliding window)
    val_transforms = Compose(base_transforms + [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=1, neg=0, # Always center on tumor for validation patches
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    return train_transforms, val_transforms

def get_dataloaders(batch_size=1, patch_size=(64, 64, 64), num_workers=cfg.TASK2_NUM_WORKERS):
    """
    Creates and returns the PyTorch DataLoaders.
    num_workers defaults to cfg.TASK2_NUM_WORKERS to speed up data fetching.
    """

    from monai.data import CacheDataset, DataLoader, Dataset
    
    train_files, val_files, test_files = download_and_prepare_dataset()
    train_transforms, val_transforms = get_transforms(patch_size)
    
    print("Creating datasets (caching metadata)...")
    # Utilizing CacheDataset to eliminate the CPU IO bottleneck (sawtooth GPU utilization)
    train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=num_workers)
    val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    test_ds = CacheDataset(data=test_files, transform=val_transforms, cache_rate=1.0, num_workers=num_workers)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=False)  # pin_memory=False for iGPUs with shared RAM
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=False)
    
    return train_loader, val_loader, test_loader, train_files, val_files, test_files

if __name__ == "__main__":
    print("Testing Task 2 Data Pipeline...")
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(batch_size=1, num_workers=0)
    
    for batch in train_loader:
        img = batch["image"]
        lbl = batch["label"]
        print(f"Image shape: {img.shape}")
        print(f"Label shape: {lbl.shape}")
        print(f"Unique labels in patch: {torch.unique(lbl)}")
        break
    print("Pipeline test successful!")
