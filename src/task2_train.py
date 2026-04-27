"""
task2_train.py — Training Loop, Loss Functions, and Monitoring for 3D Segmentation.

Addresses class imbalance via a combined Dice + Focal Loss.
Implements a memory-efficient custom training loop with ROCm/CUDA/MPS/CPU support.
Generates training monitoring visualisations adhering to the Dark Theme.
"""

import os
import sys
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from eda import (BG_DARK, BG_PANEL, TEXT_PRIMARY, TEXT_SECONDARY,
                 BLUE_LIGHT, BLUE_MID, BLUE_DARK, BLUE_PALE,
                 GRID_COLOR, save_fig)
from task2_model import Custom3DUNet, inflate_2d_to_3d_weights
from task2_dataset import get_dataloaders
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

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================================
# LOSS FUNCTIONS (Addressing Severe Class Imbalance)
# ============================================================================

class DiceLoss(nn.Module):
    """
    Computes generalized Dice Loss for multi-class segmentation.
    Directly maximizes spatial overlap between predictions and ground truth,
    which is robust to the massive background class imbalance.
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: [B, C, D, H, W]
        # targets: [B, 1, D, H, W]
        
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.long(), 1)
        
        # Calculate intersection and union over spatial dimensions
        dims = (2, 3, 4)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        union = torch.sum(probs, dim=dims) + torch.sum(targets_one_hot, dim=dims)
        
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Mean across classes and batch
        loss = 1.0 - torch.mean(dice_score)
        return loss

class FocalLoss(nn.Module):
    """
    Focal Loss down-weights well-classified examples (e.g., massive background)
    and focuses on hard, misclassified examples (e.g., small tumor boundaries).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, logits, targets):
        # targets should be [B, D, H, W] for CrossEntropy
        ce_loss = self.ce_loss(logits, targets.squeeze(1).long())
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss

class DiceFocalLoss(nn.Module):
    """
    Combines the overlap maximization of Dice with the hard-example mining of Focal Loss.
    """
    def __init__(self, lambda_dice=1.0, lambda_focal=1.0):
        super(DiceFocalLoss, self).__init__()
        self.dice = DiceLoss()
        # Alpha weights background less, tumor classes more
        alpha = torch.tensor([0.1, 1.0, 1.0, 1.0])
        self.focal = FocalLoss(alpha=alpha, gamma=2.0)
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

    def forward(self, logits, targets):
        # Alpha needs to be sent to device dynamically
        if self.focal.ce_loss.weight is not None and self.focal.ce_loss.weight.device != logits.device:
            self.focal.ce_loss.weight = self.focal.ce_loss.weight.to(logits.device)
            
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)
        return self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

# ============================================================================
# MONITORING VISUALIZATIONS
# ============================================================================

def plot_training_monitoring(history, prefix='task2'):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs, history['train_loss'], color=BLUE_MID, linewidth=2, label='Training Loss (Dice+Focal)')
    ax.plot(epochs, history['val_loss'], color=BLUE_LIGHT, linewidth=2, label='Validation Loss', linestyle='--')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Monitoring (3D Segmentation)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    save_fig(fig, f'{prefix}_training_monitoring.png')

# ============================================================================
# HELPER: Detect GPU Type
# ============================================================================

def _get_device():
    """
    Detects the best available device with accurate naming.
    ROCm (AMD GPUs) reports via torch.cuda but is NOT NVIDIA — detect this properly.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        # ROCm-backed GPUs still use torch.cuda API but are AMD
        is_amd = "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper() or "GFX" in gpu_name.upper()
        if is_amd:
            print(f"   Using AMD ROCm GPU: {gpu_name}")
            print(f"   (AMP disabled — ROCm GradScaler is unstable on iGPUs)")
        else:
            print(f"   Using NVIDIA CUDA GPU: {gpu_name}")
        return device, is_amd
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("   Using Apple Silicon MPS GPU.")
        return torch.device("mps"), False
    else:
        print("   Using CPU (Warning: 3D training will be very slow).")
        return torch.device("cpu"), False

# ============================================================================
# TRAINING LOOP (ROCm / CUDA / MPS / CPU)
# ============================================================================

def train_model(epochs=30, batch_size=1, patch_size=(96, 96, 96), patience=10):
    print("\n" + "▓" * 60)
    print("  TASK 2: Training 3D U-Net Model")
    print("▓" * 60)
    
    # 1. Hardware Selection
    device, is_amd = _get_device()
    
    # AMP GradScaler only for real NVIDIA GPUs — it crashes on ROCm iGPUs
    use_amp = (device.type == 'cuda' and not is_amd)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("   Enabling Mixed Precision (AMP) for faster training.")
    
    # 2. DataLoaders
    train_loader, val_loader, _, _, _, _ = get_dataloaders(batch_size=batch_size, patch_size=patch_size)
    if len(train_loader.dataset) == 0:
        print("   No data found. Exiting training.")
        return None, None
        
    print(f"   Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # 3. Model & Transfer Learning Initialization
    model = Custom3DUNet(in_channels=4, out_classes=4, init_features=cfg.TASK2_INIT_FEATURES)
    model = inflate_2d_to_3d_weights(model)
    model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"   ⚡ Activating multi-GPU training on {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    
    # 4. Optimizer, Scheduler, and Loss
    criterion = DiceFocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TASK2_LEARNING_RATE, weight_decay=cfg.TASK2_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_weights = None
    patience_counter = 0
    
    # Gradient Accumulation to simulate larger effective batch size
    accumulation_steps = cfg.TASK2_ACCUMULATION_STEPS
    
    print(f"\n   Config: batch={batch_size}, patch={patch_size}, accum={accumulation_steps}, AMP={use_amp}")
    print("   Starting Training Loop...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx == 0 and epoch == 0 and device.type == 'cuda':
                print("      (Note: The very first batch may take several minutes as ROCm compiles 3D convolution kernels. Please be patient...)")
            
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)
            
            # Forward pass (with AMP only on real NVIDIA GPUs)
            try:
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(inputs)
                        loss = criterion(logits, labels)
                else:
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                
                # Backward pass (with gradient accumulation)
                loss_scaled = loss / accumulation_steps
                
                if scaler is not None:
                    scaler.scale(loss_scaled).backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss_scaled.backward()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                train_loss += loss.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"      ⚠ OOM on batch {batch_idx}! Clearing cache and skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
            
            # Periodically clear GPU cache to prevent fragmentation on iGPUs
            del inputs, labels, logits, loss, loss_scaled
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        train_loss /= max(len(train_loader), 1)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs = batch_data["image"].to(device, non_blocking=True)
                labels = batch_data["label"].to(device, non_blocking=True)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                del inputs, labels, logits, loss
                
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        elapsed = time.time() - start_time
        print(f"   Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {elapsed:.1f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_model_weights, 'task2_best_model.pth')
            print(f"      → Saved new best model (Val Loss: {best_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"      → No improvement in validation loss (Patience: {patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n   🛑 Early stopping triggered at Epoch {epoch+1}!")
                break
        
        # Memory cleanup between epochs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    plot_training_monitoring(history)
    print("\n   Training Complete! Monitoring visualizations saved to figures/")
    
    # Restore best weights before returning
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    return model, history

if __name__ == "__main__":
    train_model(epochs=2, batch_size=1, patch_size=(64, 64, 64)) # Tiny run for testing script execution
