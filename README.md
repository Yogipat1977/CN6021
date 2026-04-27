# CN6021: 3D Brain Tumor Segmentation
**Group Project — Task 2: Advanced Medical Imaging Pipeline**

This repository contains a complete deep learning pipeline for 3D Brain Tumor Segmentation using the **BraTS 2020 Dataset**. The project implements a custom 3D U-Net with a weight inflation strategy from pre-trained 2D ResNet models to mitigate the constraints of limited annotated 3D medical data.

## 🚀 Key Features
- **Custom 3D U-Net Architecture:** Specifically designed for multi-modal MRI input (FLAIR, T1, T1ce, T2).
- **2D-to-3D Transfer Learning:** Implements weight inflation from pre-trained ResNet18 encoders.
- **Robust Loss Functions:** Combined **Dice Loss** and **Focal Loss** to handle extreme class imbalance (99% background).
- **Hardware Optimized:** Memory-safe training loops designed for AMD ROCm (Radeon iGPUs) and Apple Silicon (MPS).
- **Medical Data Augmentations:** Advanced 3D transforms including elastic deformations, flips, and intensity jittering via the MONAI framework.

## 🛠️ Project Structure
```text
├── src/
│   ├── task2_run.py       # Main entry point (Run the whole pipeline)
│   ├── task2_train.py     # Training loop & memory-safe GPU logic
│   ├── task2_model.py     # 3D U-Net architecture & inflation logic
│   ├── task2_dataset.py   # MONAI Data loaders & 3D Augmentation
│   └── task2_evaluate.py  # Metrics (Dice, IoU, Hausdorff) & Visuals
├── figures/               # Automatically generated EDA and prediction overlays
├── report/                # Coursework report source files (Typst)
└── presentation/          # Quarto/Reveal.js presentation source
```

## ⚙️ Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Arch Linux / ROCm Configuration
For AMD GPU users on Arch Linux, this project requires the official `python-pytorch-rocm` package. Re-create the environment with system site packages to link the ROCm drivers:
```bash
python -m venv venv --system-site-packages
```

## 📈 Usage
The entire pipeline—from data download to final evaluation—can be executed with a single command:
```bash
python src/task2_run.py
```

## 📊 Outputs & Monitoring
- **EDA Figures:** `figures/task2_eda_*`
- **Training Curves:** `figures/task2_training_monitoring.png`
- **Segmentation Overlays:** `figures/task2_prediction_overlay_*.png`
- **Model Weights:** Saved automatically as `task2_best_model.pth`.

---
**Team Members:** Hard Joshi, Jayrup Nakawala, Yogi Patel
