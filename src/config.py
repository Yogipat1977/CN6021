"""
config.py — Centralized Configuration for CN6021 Pipeline.

Use this file to easily tune hyperparameters and hardware settings
without digging through the individual scripts.
"""

# ===================================================================
# Task 2: 3D Brain Tumor Segmentation Configuration
# ===================================================================

# Hardware & Data Loading
TASK2_NUM_WORKERS = 4         # Increase for fast GPUs (e.g., 4 or 8). Set to 0 for shared RAM/iGPUs.
TASK2_BATCH_SIZE = 4          # Batch size per step. Decrease if Out of Memory (OOM) occurs.

# Model Architecture
TASK2_INIT_FEATURES = 32      # Base capacity. 16 = ~5.6M params, 32 = ~22.5M params.
TASK2_PATCH_SIZE = (128, 128, 128) # 3D crop size during training and eval.

# Training Loop
TASK2_EPOCHS = 50             # Maximum number of epochs to train.
TASK2_PATIENCE = 10           # Early stopping patience (epochs without improvement).
TASK2_ACCUMULATION_STEPS = 4  # Gradient accumulation steps to simulate larger batch sizes.

# Optimization
TASK2_LEARNING_RATE = 1e-3
TASK2_WEIGHT_DECAY = 1e-3
