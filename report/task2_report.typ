// Task 2 Report: 3D Brain Tumor Segmentation

= Task 2: 3D Brain Tumor Segmentation

== Introduction
Brain tumor segmentation from multi-modal 3D MRI scans is a complex medical imaging challenge. In this task, we implemented a 3D Semantic Segmentation pipeline using PyTorch to automatically detect and segment tumor regions (Necrotic core, Edema, Enhancing tumor) from background tissues. We addressed four critical constraints: 3D data memory management, severe class imbalance, limited annotated training data, and tumor shape variability.

The dataset selected for this task is the BraTS 2020 dataset, the gold standard for multi-modal 3D brain tumor segmentation, containing four MRI sequences (FLAIR, T1, T1ce, T2) and meticulously annotated expert ground truth masks.

== Methodology

=== 1. Data Pipeline and Memory Management
To handle the memory constraints of large 3D MRI volumes ($240 \times 240 \times 155$ voxels per sequence), we implemented a custom PyTorch `Dataset` using the `monai` and `nibabel` libraries. Rather than loading full volumes into GPU memory during training, we employed a random 3D patch extraction strategy (`RandCropByPosNegLabeld`), sampling $96 \times 96 \times 96$ sub-volumes. This not only conserved GPU VRAM but also served as a targeted data augmentation method.

=== 2. Addressing Class Imbalance
The dataset exhibits a severe class imbalance, with the background class comprising over 98% of the total voxels. To combat this:
- **Sampling Strategy:** We enforced a 50/50 sampling ratio for training patches, ensuring half the patches were centered on tumor regions rather than empty background.
- **Loss Function:** We engineered a custom `DiceFocalLoss`. The **Dice Loss** component directly maximizes the spatial overlap (Intersection over Union) regardless of class size, while the **Focal Loss** dynamically scales the cross-entropy loss based on prediction confidence, down-weighting the easily classified background voxels and heavily penalizing errors on the difficult tumor boundaries.

=== 3. 3D Augmentation for Generalization
To address the variability in tumor sizes and shapes and mitigate the limited annotated data, we implemented a robust 3D augmentation pipeline:
- `RandRotate90d` and `RandFlipd` across all three spatial axes.
- `Rand3DElasticd` (Elastic Deformations) to simulate natural anatomical and pathological variations in brain structures, satisfying the specific coursework requirements.

=== 4. Architecture and Transfer Learning
We designed a custom **3D U-Net** from scratch in PyTorch. The architecture features:
- **DoubleConv3D Blocks** with `BatchNorm3d` and `LeakyReLU` activations to prevent vanishing gradients in deep 3D networks.
- **Skip Connections** using `ConvTranspose3d` to concatenate high-resolution spatial features from the encoder directly into the decoder, preserving fine tumor boundaries lost during max pooling.

To further mitigate the limited data constraint, we implemented **2D-to-3D Weight Inflation Transfer Learning**. Since high-quality pre-trained 3D medical models require massive dependencies, we extracted the weights from a 2D ResNet trained on ImageNet and "inflated" the $3 \times 3$ kernels across the depth dimension to create $3 \times 3 \times 3$ kernels. This initialized our 3D encoder with rich, generalized edge and texture extractors, significantly speeding up convergence.

== Results and Evaluation

=== Training Monitoring
The model was trained using the `AdamW` optimizer with Automatic Mixed Precision (AMP) for accelerated hardware performance. A learning rate scheduler (`ReduceLROnPlateau`) ensured smooth convergence. The training and validation loss curves (saved as `task2_training_monitoring.png`) demonstrate steady minimization of the combined Dice-Focal loss without severe overfitting, validating the efficacy of our elastic augmentation pipeline.

=== Evaluation Metrics
The model was evaluated on a held-out test set using three rigorous spatial metrics:
1. **Dice Score:** Measures the exact overlap between the predicted tumor mask and the ground truth.
2. **Intersection over Union (IoU):** A stricter overlap metric.
3. **Hausdorff Distance (95th percentile):** Measures the maximum surface distance between the prediction boundary and the true boundary, heavily penalizing shape mismatches.

Our visualizations (`task2_prediction_overlay.png`) display the model's predictions overlaid on the original FLAIR MRI slices alongside the ground truth. The model successfully isolates the hyperintense tumor regions, distinguishing the edema from the necrotic core with high spatial accuracy. 

== Conclusion
The developed 3D U-Net successfully segments complex brain tumors from multi-modal MRIs. By combining 2D-to-3D transfer learning, elastic deformations, and a specialized Dice-Focal loss function, the pipeline overcomes the inherent challenges of 3D medical imaging, providing a robust, memory-efficient solution for automated tumor analysis.
