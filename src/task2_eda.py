"""
task2_eda.py — Exploratory Data Analysis for 3D Brain Tumor Segmentation.

Generates 3D slice visualisations, intensity distributions, and class 
imbalance analysis, adhering to the Task 1 Dark Theme.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib

sys.path.insert(0, os.path.dirname(__file__))
from eda import (BG_DARK, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY,
                 BLUE_LIGHT, BLUE_MID, BLUE_DARK, BLUE_PALE,
                 GRID_COLOR, save_fig)
from task2_dataset import download_and_prepare_dataset

# Apply theme
plt.rcParams.update({
    'figure.facecolor': BG_DARK, 'axes.facecolor': BG_PANEL,
    'axes.edgecolor': GRID_COLOR, 'axes.labelcolor': TEXT_PRIMARY,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'text.color': TEXT_PRIMARY,
    'xtick.color': TEXT_SECONDARY, 'ytick.color': TEXT_SECONDARY,
    'legend.facecolor': BG_PANEL, 'legend.edgecolor': GRID_COLOR,
    'legend.fontsize': 9, 'legend.labelcolor': TEXT_PRIMARY,
    'grid.color': GRID_COLOR, 'grid.linestyle': '--', 'grid.alpha': 0.5,
    'font.family': 'sans-serif', 'figure.dpi': 150,
    'savefig.dpi': 200, 'savefig.facecolor': BG_DARK,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.3,
})

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_3d_slices(data_dict, prefix="eda"):
    """
    Plots the middle axial, coronal, and sagittal slices of the 4 modalities and mask.
    """
    print("   Generating 3D Slice Visualizations...")
    
    # Load data
    flair = nib.load(data_dict["flair"]).get_fdata()
    t1 = nib.load(data_dict["t1"]).get_fdata()
    t1ce = nib.load(data_dict["t1ce"]).get_fdata()
    t2 = nib.load(data_dict["t2"]).get_fdata()
    seg = nib.load(data_dict["label"]).get_fdata()
    
    # Map label 4 to 3 for plotting
    seg[seg == 4] = 3
    
    # Find middle slices
    x_mid, y_mid, z_mid = flair.shape[0]//2, flair.shape[1]//2, flair.shape[2]//2
    
    modalities = [
        ("FLAIR", flair),
        ("T1", t1),
        ("T1ce", t1ce),
        ("T2", t2),
        ("Ground Truth", seg)
    ]
    
    views = [
        ("Axial", lambda v: v[:, :, z_mid]),
        ("Coronal", lambda v: v[:, y_mid, :]),
        ("Sagittal", lambda v: v[x_mid, :, :])
    ]
    
    fig, axes = plt.subplots(len(views), len(modalities), figsize=(15, 10))
    fig.patch.set_facecolor(BG_DARK)
    
    for i, (view_name, slice_fn) in enumerate(views):
        for j, (mod_name, vol) in enumerate(modalities):
            ax = axes[i, j]
            slice_data = slice_fn(vol)
            
            # Rotate for proper anatomical orientation display
            slice_data = np.rot90(slice_data)
            
            if mod_name == "Ground Truth":
                cmap = matplotlib.colors.ListedColormap(['black', BLUE_DARK, BLUE_MID, BLUE_LIGHT])
                im = ax.imshow(slice_data, cmap=cmap, vmin=0, vmax=3)
            else:
                im = ax.imshow(slice_data, cmap="gray")
                
            ax.axis("off")
            if i == 0:
                ax.set_title(mod_name, color=TEXT_PRIMARY)
            if j == 0:
                ax.text(-0.1, 0.5, view_name, va='center', ha='right', 
                        rotation=90, transform=ax.transAxes, color=TEXT_PRIMARY, fontweight='bold', fontsize=12)

    fig.suptitle('BraTS 2020 Multi-Modal MRI Slices', fontsize=16, fontweight='bold', color=TEXT_PRIMARY)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_3d_slices.png')

def plot_intensity_distributions(data_dicts, num_samples=5, prefix="eda"):
    """
    Plots the voxel intensity distributions for each modality across random samples.
    """
    print("   Generating Intensity Distributions...")
    np.random.seed(42)
    sample_dicts = np.random.choice(data_dicts, min(num_samples, len(data_dicts)), replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    modalities = ["flair", "t1", "t1ce", "t2"]
    colors = [BLUE_LIGHT, BLUE_MID, BLUE_DARK, BLUE_PALE]
    
    for i, mod in enumerate(modalities):
        ax = axes[i]
        all_voxels = []
        for d in sample_dicts:
            vol = nib.load(d[mod]).get_fdata()
            # Ignore background (0) for intensity distribution
            voxels = vol[vol > 0]
            if len(voxels) > 0:
                # Subsample to avoid massive arrays
                sub_voxels = np.random.choice(voxels, min(100000, len(voxels)), replace=False)
                all_voxels.append(sub_voxels)
        
        if all_voxels:
            all_voxels = np.concatenate(all_voxels)
            ax.hist(all_voxels, bins=50, color=colors[i], alpha=0.7, edgecolor=BG_DARK, density=True)
            
        ax.set_title(f'{mod.upper()} Intensity Distribution')
        ax.set_xlabel('Voxel Intensity')
        ax.set_ylabel('Density')
        ax.grid(alpha=0.3)
        
    fig.suptitle('Voxel Intensity Distribution (Foreground Only)', fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_intensity_distributions.png')

def plot_class_imbalance(data_dicts, num_samples=10, prefix="eda"):
    """
    Analyzes and plots the class imbalance (Background vs Tumor regions).
    """
    print("   Generating Class Imbalance Analysis...")
    np.random.seed(42)
    sample_dicts = np.random.choice(data_dicts, min(num_samples, len(data_dicts)), replace=False)
    
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    for d in sample_dicts:
        seg = nib.load(d["label"]).get_fdata()
        seg[seg == 4] = 3
        unique, counts = np.unique(seg, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[int(u)] += c
            
    total_voxels = sum(class_counts.values())
    tumor_voxels = class_counts[1] + class_counts[2] + class_counts[3]
    
    labels = ['Background', 'NCR/NET (Core)', 'Edema', 'Enhancing Tumor']
    counts = [class_counts[0], class_counts[1], class_counts[2], class_counts[3]]
    percentages = [c / total_voxels * 100 for c in counts]
    
    print(f"      Background: {percentages[0]:.4f}%")
    print(f"      Tumor Total: {(tumor_voxels / total_voxels * 100):.4f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart showing massive imbalance (Background vs Tumor)
    ax1.pie([class_counts[0], tumor_voxels], 
            labels=['Background', 'All Tumor Classes'], 
            autopct='%1.2f%%', 
            colors=[BG_PANEL, BLUE_MID],
            explode=(0, 0.1),
            textprops={'color': TEXT_PRIMARY, 'fontweight': 'bold'},
            wedgeprops={'edgecolor': GRID_COLOR})
    ax1.set_title('Severe Class Imbalance', fontsize=14, fontweight='bold', color=TEXT_PRIMARY)
    
    # Bar chart for tumor subclasses only
    tumor_labels = ['NCR/NET', 'Edema', 'Enhancing']
    tumor_counts = [class_counts[1], class_counts[2], class_counts[3]]
    ax2.bar(tumor_labels, tumor_counts, color=[BLUE_DARK, BLUE_MID, BLUE_LIGHT], edgecolor=BG_DARK)
    ax2.set_title('Tumor Subclasses Breakdown', fontsize=14, fontweight='bold', color=TEXT_PRIMARY)
    ax2.set_ylabel('Voxel Count')
    ax2.grid(axis='y', alpha=0.3)
    
    fig.suptitle('3D Semantic Segmentation Class Imbalance', fontsize=16, fontweight='bold', y=1.05)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_class_imbalance.png')

def run_task2_eda():
    print("\n" + "▓" * 60)
    print("  TASK 2: 3D MRI Exploratory Data Analysis")
    print("▓" * 60)
    
    train_files, val_files, test_files = download_and_prepare_dataset()
    if len(train_files) > 0:
        plot_3d_slices(train_files[0], prefix="task2_eda")
        plot_intensity_distributions(train_files, prefix="task2_eda")
        plot_class_imbalance(train_files, prefix="task2_eda")
        print("\n   EDA complete! All 3D figures saved to figures/")
    else:
        print("   Error: No data found.")

if __name__ == "__main__":
    run_task2_eda()
