"""
task2_run.py — Single entry-point to run the complete Task 2 pipeline.

Steps:
1. Run EDA → generate 3D data exploration figures
2. Train Model → 3D U-Net with Transfer Learning and Focal/Dice Loss
3. Evaluate Model → generate prediction visuals and metric reports
"""

import os
import sys
import gc
import torch

sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("\n" + "█" * 60)
    print("  CN6021 — Task 2: 3D Brain Tumor Segmentation")
    print("  Group: Hard Joshi, Jayrup Nakawala, Yogi Patel")
    print("█" * 60)

    # ---- Step 1: EDA ----
    from task2_eda import run_task2_eda
    run_task2_eda()

    # Free EDA memory before training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- Step 2: Train Model ----
    from task2_train import train_model
    # Scaled up for multi-GPU 48GB VRAM setup (2x RTX 3090s)
    model, history = train_model(epochs=50, batch_size=4, patch_size=(128, 128, 128))
    
    if model is None:
        return

    # ---- Step 3: Evaluate Model ----
    from task2_evaluate import evaluate_model
    metrics = evaluate_model(model, patch_size=(128, 128, 128))


    # ---- Summary ----
    print("\n\n" + "█" * 60)
    print("  TASK 2 PIPELINE COMPLETE — Summary")
    print("█" * 60)
    print(f"\n  Final Metrics:")
    if metrics:
        print(f"    Mean Dice Score:  {metrics['dice']:.4f}")
        print(f"    Mean IoU:         {metrics['iou']:.4f}")
        print(f"    Hausdorff (95%):  {metrics['hausdorff']:.4f}")
    print(f"\n  All figures saved to: figures/")
    print("█" * 60 + "\n")

if __name__ == '__main__':
    main()
