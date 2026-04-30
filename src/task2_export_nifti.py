"""
task2_export_nifti.py — Export 3D brain tumor predictions to NIfTI format for 3D Slicer.

This script runs Sliding Window Inference on full patient volumes and saves
the output as .nii.gz files.
"""

import os
import sys
import torch
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MONAI imports
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ConcatItemsd,
    Resized, ScaleIntensityRangePercentilesd, EnsureTyped
)

sys.path.insert(0, os.path.dirname(__file__))
from task2_dataset import download_and_prepare_dataset
from task2_model import Custom3DUNet
import config as cfg

def get_full_volume_transforms():
    """Transforms for loading the full 3D volume and resizing to 128x128x128."""
    keys = ["flair", "t1", "t1ce", "t2"]
    return Compose([
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        ConcatItemsd(keys=keys, name="image", dim=0),
        # Resize directly to 128x128x128 to prevent memory crashes and match expected model input
        Resized(keys=["image"], spatial_size=(128, 128, 128)),
        # Normalize
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])

def export_nifti_volumes(num_patients=3):
    print("\n" + "█" * 60)
    print("  TASK 2: Exporting 3D NIfTI Volumes for 3D Slicer")
    print("█" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: CUDA/ROCm is not available! The script is falling back to CPU, which may be slow and cause memory issues.")

    # 1. Load Model Architecture
    print("Loading model architecture...")
    model = Custom3DUNet(in_channels=4, out_classes=4, init_features=cfg.TASK2_INIT_FEATURES).to(device)

    # 2. Load Weights
    weight_path = 'task2_best_model.pth'
    if not os.path.exists(weight_path):
        print(f"Error: Could not find {weight_path}. Make sure you have trained the model.")
        return

    print(f"Loading trained weights from {weight_path}...")
    state_dict = torch.load(weight_path, map_location=device)
    
    # Handle DataParallel 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Load Test Data
    print("\nPreparing test dataset...")
    _, _, test_files = download_and_prepare_dataset()
    
    # Limit to specified number of patients to save time
    test_files = test_files[:num_patients]
    
    transforms = get_full_volume_transforms()
    test_ds = Dataset(data=test_files, transform=transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    out_dir = "predictions"
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nCreated output directory: {out_dir}/")

    # 4. Inference and Export
    print(f"\nStarting Sliding Window Inference on {len(test_files)} patients...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            patient_path = test_files[i]['flair']
            patient_id = os.path.basename(os.path.dirname(patient_path))
            
            print(f"\n[{i+1}/{len(test_files)}] Processing {patient_id}...")
            inputs = batch["image"].to(device)
            
            # Extract affine matrix to preserve real-world coordinates
            if hasattr(inputs, "meta") and "affine" in inputs.meta:
                affine = inputs.meta["affine"][0].cpu().numpy()
            else:
                affine = np.eye(4) # Fallback

            # Run direct 3D model inference on the 128x128x128 volume
            print("  Running 3D model directly on GPU with mixed precision...")
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(inputs)
            
            torch.cuda.empty_cache()
            
            # Get predicted classes
            preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            
            # Remap class 3 back to 4 to match standard BraTS format 
            # (0: Background, 1: Necrotic, 2: Edema, 4: Enhancing Tumor)
            preds[preds == 3] = 4
            
            # Save as NIfTI (.nii.gz)
            preds = preds.astype(np.uint8)
            nii_img = nib.Nifti1Image(preds, affine)
            
            out_file = os.path.join(out_dir, f"{patient_id}_prediction.nii.gz")
            nib.save(nii_img, out_file)
            print(f"  Saved 3D volume -> {out_file}")

            # Generate PNG Screenshots for the report
            img_dir = "predictions_images"
            os.makedirs(img_dir, exist_ok=True)
            
            # Extract FLAIR volume for background (channel 0 from inputs)
            flair_vol = inputs[0, 0].cpu().numpy()
            
            # Get middle slices
            mid_z = preds.shape[2] // 2
            mid_y = preds.shape[1] // 2
            mid_x = preds.shape[0] // 2
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Axial (Z)
            axes[0].imshow(flair_vol[:, :, mid_z].T, cmap='gray', origin='lower')
            axes[0].imshow(preds[:, :, mid_z].T, cmap='nipy_spectral', alpha=0.4, origin='lower')
            axes[0].set_title("Axial View")
            axes[0].axis('off')
            
            # Coronal (Y)
            axes[1].imshow(flair_vol[:, mid_y, :].T, cmap='gray', origin='lower')
            axes[1].imshow(preds[:, mid_y, :].T, cmap='nipy_spectral', alpha=0.4, origin='lower')
            axes[1].set_title("Coronal View")
            axes[1].axis('off')
            
            # Sagittal (X)
            axes[2].imshow(flair_vol[mid_x, :, :].T, cmap='gray', origin='lower')
            axes[2].imshow(preds[mid_x, :, :].T, cmap='nipy_spectral', alpha=0.4, origin='lower')
            axes[2].set_title("Sagittal View")
            axes[2].axis('off')
            
            plt.tight_layout()
            png_file = os.path.join(img_dir, f"{patient_id}_slices.png")
            plt.savefig(png_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved screenshots -> {png_file}")

    print("\n" + "█" * 60)
    print("  EXPORT COMPLETE!")
    print("  You can now drag and drop the .nii.gz files from the")
    print(f"  '{out_dir}' folder directly into 3D Slicer.")
    print("█" * 60 + "\n")

if __name__ == "__main__":
    export_nifti_volumes(num_patients=10)
