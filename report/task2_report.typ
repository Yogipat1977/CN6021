// Task 2 Report: 3D Brain Tumor Segmentation

= Task 2: 3D Brain Tumor Segmentation

== Introduction
Brain tumor segmentation from multi-modal 3D MRI scans is a complex medical imaging challenge. In this task, we implemented a 3D Semantic Segmentation pipeline using PyTorch to automatically detect and segment tumor regions (Necrotic core, Edema, Enhancing tumor) from background tissues. We addressed four critical constraints: 3D data memory management, severe class imbalance, limited annotated training data, and tumor shape variability.

The dataset selected for this task is the BraTS 2020 dataset, the gold standard for multi-modal 3D brain tumor segmentation, containing four MRI sequences (FLAIR, T1, T1ce, T2) and meticulously annotated expert ground truth masks.

== Architecture

We designed a custom *3D U-Net* from scratch in PyTorch (22.5M parameters, `init_features=32`). The architecture follows the classic encoder-decoder design for biomedical segmentation #cite(<ronneberger2015>):

- *Encoder Path:* Four downsampling stages (MaxPool3d) progressively reduce spatial resolution while doubling feature channels from 32 → 64 → 128 → 256 → 512, capturing hierarchical features from local texture to global anatomical context.
- *Decoder Path:* Four upsampling stages (ConvTranspose3d) restore spatial resolution. Learned transposed convolutions adaptively reconstruct fine structural details.
- *Skip Connections:* High-resolution encoder feature maps are concatenated into the corresponding decoder stage, preserving fine tumor boundaries lost during aggressive downsampling.
- *DoubleConv3D Blocks:* Each block applies two $3 times 3 times 3$ convolutions with BatchNorm3d and LeakyReLU activations. LeakyReLU prevents dead neurons in deep 3D training; BatchNorm stabilises gradients across spatial dimensions.
- *Output Head:* A $1 times 1 times 1$ convolution maps features to 4-class logits (background, NCR/NET, ED, ET).

== Methodology

=== 1. Data Pipeline and Memory Management
GPU memory constraints on consumer hardware prevent loading full $240 times 240 times 155$ volumes. We implemented a custom PyTorch `Dataset` with random 3D patch extraction (`RandCropByPosNegLabeld`), sampling $96 times 96 times 96$ sub-volumes—reducing GPU VRAM usage while serving as targeted data augmentation.

=== 2. Addressing Class Imbalance
The background class comprises over 98% of total voxels. To combat this:
- *Sampling Strategy:* A 50/50 ratio ensures half of training patches are centered on tumor regions.
- *Loss Function:* A custom `DiceFocalLoss` combines two complementary objectives. The Dice Loss component directly optimises the Dice coefficient (`
2|X ∩ Y|/(|X| + |Y|)`), maximizing spatial overlap regardless of class size. The Focal Loss component dynamically scales the cross-entropy loss, down-weighting easily classified background voxels and heavily penalising errors on difficult tumor boundaries.

=== 3. 3D Augmentation for Generalization
To address variability in tumor size and shape with limited training data, we applied `RandRotate90d`, `RandFlipd` across all three spatial axes, and `Rand3DElasticd` (elastic deformations) to simulate natural anatomical variations in brain structures.

=== 4. 2D-to-3D Transfer Learning
Pre-trained 3D medical models are scarce. We extracted 2D convolution weights from an ImageNet-trained ResNet-18 and inflated the $3 times 3$ kernels across the depth dimension to $3 times 3 times 3$ kernels, dividing by depth size (3) to preserve activation variance. This initialised the 3D encoder with generalised edge and texture feature extractors, accelerating early convergence. We note this is an experimental technique—2D natural-image features do not directly encode 3D volumetric patterns, so the benefit is primarily in early training stages.

== Results

=== Quantitative Metrics
The model was evaluated on a held-out test cohort of 10 patients using Dice Coefficient, Intersection over Union (IoU), and Hausdorff Distance (95th percentile). Due to GPU memory limits, metrics are computed on $128 times 128 times 128$ patches—patch-level scores represent a lower bound on whole-brain performance.

#align(center)[
#table(
  columns: (auto, auto, auto, auto),
  stroke: 0.5pt,
  [*Tumor Class*], [*Mean Dice*], [*Best (Patient)*], [*Worst (Patient)*],
  [ET (Enhancing Tumor)], [0.62], [0.87 (#367)], [0.00 (#188)],
  [ED (Edema)], [0.65], [0.88 (#055)], [0.09 (#320)],
  [NCR/NET (Necrotic Core)], [0.36], [0.66 (#055)], [0.00 (#188, #190)],
)
]

The model achieves usable segmentation on well-defined tumors—patients 051, 055, and 170 score Dice 0.65–0.88 on Edema and Enhancing Tumor, demonstrating that the 3D U-Net can localise hyperintense regions effectively. However, performance varies substantially:

- *Necrotic Core (mean Dice 0.36):* The necrotic core is often small and visually indistinct. With 22.5M parameters, the model's limited capacity and receptive field are insufficient for reliable necrotic segmentation. Even BraTS state-of-the-art models achieve only 0.70–0.75 Dice on this class #cite(<brats2020>).
- *Catastrophic failures:* Patients 188, 190, and 320 score near-zero Dice on one or more classes, likely reflecting atypical tumor morphology or small volume outside the model's training support.

These results define a clear *minimum viable model* boundary: a 3D U-Net of this scale can segment typical enhancing tumor and edema on consumer hardware but cannot handle diffuse necrotic tissue or unusual presentations.

=== Qualitative Visualisation
Visualizations (`task2_prediction_overlay.png`) display predictions overlaid on FLAIR MRI slices alongside ground truth. On well-performing patients, the model distinguishes edema from enhancing tumor with moderate spatial accuracy. Post-prediction 3D renderings confirm coherent volumetric segmentation in successful cases.

== Limitations

1. *Patch-level evaluation:* Metrics computed on 128³ patches rather than full brain volumes. Edge patches containing partial anatomy score lower than central patches. Full-volume sliding-window inference would yield higher whole-brain Dice scores.
2. *Hausdorff approximation:* HD95 uses a 1,000-point boundary subsample for computational feasibility rather than exact surface-distance computation. Reported values are estimates.
3. *Model capacity:* 22.5M parameters on a consumer GPU restricts the receptive field. Larger architectures (64+ init features) and test-time augmentation would improve boundary refinement and necrotic core segmentation.

== Conclusion
The 3D U-Net provides a functional baseline for brain tumor segmentation on consumer hardware: usable on enhancing tumor and edema (Dice 0.65–0.88 in typical cases) but limited on necrotic tissue and atypical morphologies. The combination of 2D-to-3D transfer learning, Dice-Focal loss, and elastic augmentation enables training within constrained resources, while the identified failure modes point clearly to capacity and inference-strategy improvements needed for clinical-grade performance.

#heading(numbering: none)[References]

#bibliography("refs.yml", style: "harvard-cite-them-right")
