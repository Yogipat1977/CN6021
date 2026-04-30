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
import csv
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

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

def calc_metrics(pred, gt, class_val):
    p = (pred == class_val)
    g = (gt == class_val)
    
    tp = np.sum(p & g)
    fp = np.sum(p & ~g)
    fn = np.sum(~p & g)
    tn = np.sum(~p & ~g)
    
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else (1.0 if np.sum(g) == 0 else 0.0)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else (1.0 if np.sum(g) == 0 else 0.0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if np.sum(g) == 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 1.0
    
    # HD95
    if np.sum(p) == 0 and np.sum(g) == 0:
        hd95 = 0.0
    elif np.sum(p) == 0 or np.sum(g) == 0:
        hd95 = np.nan
    else:
        try:
            p_border = p ^ binary_erosion(p)
            g_border = g ^ binary_erosion(g)
            p_pts = np.argwhere(p_border)
            g_pts = np.argwhere(g_border)
            if len(p_pts) == 0 or len(g_pts) == 0:
                hd95 = np.nan
            else:
                tree_g = cKDTree(g_pts)
                dists_p_to_g, _ = tree_g.query(p_pts)
                tree_p = cKDTree(p_pts)
                dists_g_to_p, _ = tree_p.query(g_pts)
                hd95 = np.percentile(np.concatenate([dists_p_to_g, dists_g_to_p]), 95)
        except Exception:
            hd95 = np.nan

    return dice, iou, sensitivity, specificity, hd95

def get_full_volume_transforms():
    """Transforms for loading the full 3D volume and resizing to 128x128x128."""
    image_keys = ["flair", "t1", "t1ce", "t2"]
    all_keys = image_keys + ["label"]
    return Compose([
        LoadImaged(keys=all_keys),
        EnsureChannelFirstd(keys=all_keys),
        ConcatItemsd(keys=image_keys, name="image", dim=0),
        # Resize directly to 128x128x128 to prevent memory crashes and match expected model input
        Resized(keys=["image"], spatial_size=(128, 128, 128), mode="trilinear"),
        Resized(keys=["label"], spatial_size=(128, 128, 128), mode="nearest"),
        # Normalize
        ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"]),
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

    csv_file = os.path.join(out_dir, "predictions_metrics.csv")
    csv_header = ["PatientID", "Class", "Dice", "IoU", "Sensitivity", "Specificity", "HD95"]
    csv_rows = []

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
            
            # Ground truth
            gt = batch["label"].squeeze(0).squeeze(0).cpu().numpy()
            
            # Remap class 3 back to 4 to match standard BraTS format 
            # (0: Background, 1: Necrotic, 2: Edema, 4: Enhancing Tumor)
            preds[preds == 3] = 4
            
            # Evaluate classes: 1, 2, 4
            for class_val, class_name in [(1, "NCR/NET"), (2, "ED"), (4, "ET")]:
                d, iou, sens, spec, hd = calc_metrics(preds, gt, class_val)
                csv_rows.append([patient_id, class_name, f"{d:.4f}", f"{iou:.4f}", f"{sens:.4f}", f"{spec:.4f}", f"{hd:.4f}"])
            
            # Create a dedicated folder for this specific patient
            patient_out_dir = os.path.join(out_dir, patient_id)
            os.makedirs(patient_out_dir, exist_ok=True)

            # Save Prediction as NIfTI (.nii.gz)
            preds = preds.astype(np.uint8)
            nib.save(nib.Nifti1Image(preds, affine), os.path.join(patient_out_dir, f"{patient_id}_prediction.nii.gz"))
            
            # Save Ground Truth as NIfTI
            gt = gt.astype(np.uint8)
            nib.save(nib.Nifti1Image(gt, affine), os.path.join(patient_out_dir, f"{patient_id}_gt.nii.gz"))
            
            # Save T2-FLAIR as NIfTI
            flair_vol = inputs[0, 0].cpu().numpy()
            nib.save(nib.Nifti1Image(flair_vol, affine), os.path.join(patient_out_dir, f"{patient_id}_t2f.nii.gz"))
            
            print(f"  Saved Prediction, GT, and T2F 3D volumes to {patient_out_dir}/")

    # Write CSV
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        writer.writerows(csv_rows)
    print(f"\n  Saved metrics CSV -> {csv_file}")

    print("\n" + "█" * 60)
    print("  EXPORT COMPLETE!")
    print("  You can now drag and drop the .nii.gz files from the")
    print(f"  '{out_dir}' folder directly into 3D Slicer.")
    print("█" * 60 + "\n")

if __name__ == "__main__":
    export_nifti_volumes(num_patients=10)
