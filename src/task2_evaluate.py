"""
task2_evaluate.py — Evaluation Metrics and Final Visualizations.

Calculates Dice Score, IoU, and Hausdorff Distance.
Generates overlay visualizations comparing predictions to ground truth
using the Task 1 Dark Theme.
"""

import os
import sys
import gc
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff

sys.path.insert(0, os.path.dirname(__file__))
from eda import (BG_DARK, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY,
                 BLUE_LIGHT, BLUE_MID, BLUE_DARK, BLUE_PALE,
                 GRID_COLOR, save_fig)
from task2_dataset import get_test_dataloader
import config as cfg

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

def calculate_metrics_numpy(y_pred, y_true):
    """
    Calculates Dice, IoU, and Hausdorff Distance for a single 3D volume.
    Ignores background (class 0).
    """
    metrics = {'dice': [], 'iou': [], 'hausdorff': []}
    
    # We evaluate for classes 1, 2, 3
    for c in range(1, 4):
        pred_c = (y_pred == c).astype(np.float32)
        true_c = (y_true == c).astype(np.float32)
        
        intersection = np.sum(pred_c * true_c)
        union = np.sum(pred_c) + np.sum(true_c)
        
        # Dice
        if union == 0:
            dice = 1.0 if np.sum(pred_c) == 0 else 0.0
        else:
            dice = (2. * intersection) / union
        metrics['dice'].append(dice)
        
        # IoU
        iou_union = union - intersection
        if iou_union == 0:
            iou = 1.0 if np.sum(pred_c) == 0 else 0.0
        else:
            iou = intersection / iou_union
        metrics['iou'].append(iou)
        
        # Hausdorff Distance (computationally expensive for 3D, we sample edge points)
        pred_coords = np.argwhere(pred_c > 0)
        true_coords = np.argwhere(true_c > 0)
        
        if len(pred_coords) > 0 and len(true_coords) > 0:
            # Subsample to speed up Hausdorff calculation
            np.random.seed(42)
            if len(pred_coords) > 1000: pred_coords = pred_coords[np.random.choice(len(pred_coords), 1000, replace=False)]
            if len(true_coords) > 1000: true_coords = true_coords[np.random.choice(len(true_coords), 1000, replace=False)]
            
            hd1 = directed_hausdorff(pred_coords, true_coords)[0]
            hd2 = directed_hausdorff(true_coords, pred_coords)[0]
            hd = max(hd1, hd2)
        else:
            hd = 0.0 if len(pred_coords) == len(true_coords) else float('inf') # Arbitrary penalty
        
        if hd != float('inf'):
            metrics['hausdorff'].append(hd)

    # Return mean across classes
    return {
        'dice': np.mean(metrics['dice']),
        'iou': np.mean(metrics['iou']),
        'hausdorff': np.mean(metrics['hausdorff']) if metrics['hausdorff'] else 0.0
    }

def plot_predictions(image, y_true, y_pred, prefix='task2', idx=0):
    """
    Visualizes the prediction vs ground truth overlaid on the FLAIR MRI slice.
    """
    # Extract middle axial slice
    z_mid = image.shape[2] // 2
    
    # image shape: [B, C, D, H, W] -> get FLAIR (idx 0 for channel)
    img_slice = image[0, 0, z_mid, :, :].cpu().numpy()
    true_slice = y_true[0, 0, z_mid, :, :].cpu().numpy()
    pred_slice = y_pred[0, 0, z_mid, :, :].cpu().numpy()
    
    img_slice = np.rot90(img_slice)
    true_slice = np.rot90(true_slice)
    pred_slice = np.rot90(pred_slice)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG_DARK)
    
    cmap = matplotlib.colors.ListedColormap(['black', BLUE_DARK, BLUE_MID, BLUE_LIGHT])
    
    # Base Image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title('FLAIR MRI', color=TEXT_PRIMARY)
    axes[0].axis('off')
    
    # Ground Truth Overlay
    axes[1].imshow(img_slice, cmap='gray')
    true_masked = np.ma.masked_where(true_slice == 0, true_slice)
    axes[1].imshow(true_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    axes[1].set_title('Ground Truth', color=TEXT_PRIMARY)
    axes[1].axis('off')
    
    # Prediction Overlay
    axes[2].imshow(img_slice, cmap='gray')
    pred_masked = np.ma.masked_where(pred_slice == 0, pred_slice)
    axes[2].imshow(pred_masked, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    axes[2].set_title('Model Prediction', color=TEXT_PRIMARY)
    axes[2].axis('off')
    
    fig.suptitle('3D Segmentation: Prediction vs Ground Truth', fontsize=16, fontweight='bold', color=TEXT_PRIMARY)
    fig.tight_layout()
    save_fig(fig, f'{prefix}_prediction_overlay_{idx}.png')

def evaluate_model(model, patch_size=(64, 64, 64)):
    print("\n" + "▓" * 60)
    print("  TASK 2: Evaluating 3D U-Net")
    print("▓" * 60)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.eval()
    test_loader, _ = get_test_dataloader(batch_size=1, patch_size=patch_size)
    
    if len(test_loader.dataset) == 0:
        print("   No test data found. Exiting evaluation.")
        return
        
    all_dice = []
    all_iou = []
    all_hd = []
    
    print(f"   Evaluating on {len(test_loader.dataset)} test patches...")
    
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)
            
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1, keepdim=True)
            
            # Move to CPU for metric calculation (saves GPU memory)
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            metrics = calculate_metrics_numpy(preds_np, labels_np)
            all_dice.append(metrics['dice'])
            all_iou.append(metrics['iou'])
            all_hd.append(metrics['hausdorff'])
            
            # Save visuals for the first 3 samples
            if i < 3:
                plot_predictions(inputs, labels, preds, prefix='task2', idx=i)
            
            # Memory cleanup
            del inputs, labels, logits, probs, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    mean_hd = np.mean(all_hd)
    
    print("\n   ===============================================")
    print("   FINAL EVALUATION METRICS (Tumor Classes)")
    print("   ===============================================")
    print(f"   Mean Dice Score:       {mean_dice:.4f}")
    print(f"   Mean IoU:              {mean_iou:.4f}")
    print(f"   Mean Hausdorff (95%):  {mean_hd:.4f} voxels")
    print("   ===============================================")
    print("   Visualizations saved to figures/ directory.")
    
    return {'dice': mean_dice, 'iou': mean_iou, 'hausdorff': mean_hd}

if __name__ == "__main__":
    print("Running Evaluation independently...")
    from task2_model import Custom3DUNet
    
    # Load model architecture
    model = Custom3DUNet(in_channels=4, out_classes=4, init_features=cfg.TASK2_INIT_FEATURES)
    
    # Load trained weights
    weight_path = 'task2_best_model.pth'
    if os.path.exists(weight_path):
        print(f"Loading trained weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    else:
        print(f"Warning: {weight_path} not found. Evaluating with random initialization.")
        
    evaluate_model(model, patch_size=cfg.TASK2_PATCH_SIZE)
