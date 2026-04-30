// CN6021 Final Report — Task 1 and Task 2 Combined
// Group: Hard Joshi (2512658), Jayrup Nakawala (2613621), Yogi Patel (2536809)

#let title = "Advanced Neural Network Approaches for Classification and Medical Image Segmentation"
#let course = "CN6021: Advanced Topics in AI and Data Science"
#let authors = "Hard Joshi (2512658), Jayrup Nakawala (2613621), Yogi Patel (2536809)"

#set document(author: authors, title: title)
#set text(font: "New Computer Modern", size: 10.5pt)
#set par(justify: true, leading: 0.65em)

// Set header and footer with page numbers
#set page(
  margin: (x: 2.5cm, y: 2.5cm),
  header: context {
    if counter(page).get().first() > 1 {
      grid(
        columns: (1fr, 1fr),
        align(left)[#text(8pt, fill: gray)[University of East London]], align(right)[#text(8pt, fill: gray)[#course]],
      )
    }
  },
  footer: align(center)[
    #context text(10pt)[#counter(page).display("1")]
  ],
)

#set heading(numbering: "1.")
#show heading: it => {
  v(0.6em)
  text(weight: "bold")[#it]
  v(0.3em)
}
#show figure: it => {
  v(0.3em)
  it
  v(0.3em)
}

// --- Professional Cover Page ---
#align(center)[
  #v(4cm)
  #text(size: 20pt, weight: "bold", fill: rgb("#002855"))[University of East London]
  #v(1cm)
  #text(size: 16pt, weight: "bold")[#course]
  #v(1.5cm)
  #line(length: 100%, stroke: 1.5pt + rgb("#002855"))
  #v(0.8cm)
  #text(size: 18pt, weight: "bold")[#title]
  #v(0.8cm)
  #line(length: 100%, stroke: 1.5pt + rgb("#002855"))
  #v(2cm)
  #text(size: 14pt, weight: "bold")[Prepared by:]
  #v(0.5em)
  #text(size: 12pt)[
    Hard Joshi (2512658) \
    Jayrup Nakawala (2613621) \
    Yogi Patel (2536809)
  ]
  #v(3cm)
  #text(size: 11pt, fill: gray)[May 2026]
  #pagebreak()
]

// --- Table of Contents ---
#outline(title: "Table of Contents", depth: 3)
#v(1cm)

#pagebreak()

// Reset page counter after TOC
#counter(page).update(1)

= Introduction
Neural networks power modern AI, yet real-world deployment demands mastery of constraints textbooks rarely address: class imbalance, distribution shift, memory walls, and the imperative to explain opaque predictions to decision-makers. This report presents two advanced systems, each confronting distinct challenges while sharing a unified philosophy build modular, diagnose ruthlessly, optimize for impact.

Task 1: Customer Churn Prediction. Retaining subscribers costs far less than acquiring new ones. We built a shallow neural network from scratch in pure NumPy (per coursework requirements), processing hundreds of thousands of records across demographics, usage, financial behavior, and contracts. EDA revealed a "Generational Gap" younger customers churn markedly more than older demographics transforming binary prediction into a stratified retention strategy. The base model dazzled on validation yet collapsed on test data, with predictions skewing almost entirely to one class. We diagnosed this via stratified cross-validation, grid search across architecture capacity and regularization, and threshold optimization to translate probabilities into business-calibrated actions. Interpretability was enforced through SHAP, permutation importance, and partial dependence plots under a custom dark-theme visualization system.

Task 2: 3D Brain Tumor Segmentation. Precise tumor delineation from MRI guides surgical planning and treatment assessment. We used the BraTS 2020 dataset four co-registered MRI sequences per patient with expert masks for necrotic core, edema, and enhancing tumor. Full 3D volumes impose severe memory constraints, while background voxels threaten to drown tumor signals. We implemented a custom 3D U-Net in PyTorch with skip connections preserving fine anatomical boundaries. To mitigate scarce 3D annotations, we inflated 2D pre-trained ImageNet weights into 3D kernels. Training combats imbalance via Dice-Focal loss Dice maximizes spatial overlap, Focal concentrates on hard tumor boundaries supplemented by patch-based class-balanced sampling, 3D augmentation (flips, rotations, elastic deformations, noise), gradient accumulation for memory efficiency, and auto-detected CUDA/ROCm/MPS/CPU backends. Evaluation uses overlap coefficients, boundary distances, and per-class sensitivity, with NIfTI exports for clinical visualization.

Both systems rest on a modular, reproducible foundation centralized configuration, automated dataset acquisition, consistent dark-theme aesthetics, and eighteen Python modules spanning preprocessing, EDA, modeling, training, evaluation, interpretability, and clinical export. We resolve prototype-to-production friction through weighted losses, patch-based processing, mixed precision, cross-validation, and multi-modal explanation techniques. This report details methodology, presents findings, and charts future directions temporal sequence modeling for earlier churn warnings and vision transformers for long-range volumetric dependencies.

= Task 1: Customer Churn Prediction

== Dataset and Business Context

Customer churn prediction is a critical business problem for subscription-based services, where the cost of acquiring new customers significantly exceeds that of retaining existing ones. The dataset, sourced from Kaggle #cite(<azeem2023>), comprises 440,832 training samples and 64,374 test samples with 10 predictive features spanning customer demographics (Age, Gender), usage patterns (Tenure, Usage Frequency, Support Calls), financial behaviour (Payment Delay, Total Spend), and contractual attributes (Subscription Type, Contract Length, Last Interaction).

== Data Distribution and Generational Gap

Initial inspection revealed a well-structured dataset with minimal missing data. The class distribution reveals that churned customers constitute the majority class (56.7%), requiring a weighted loss function during training to prevent naive majority-class predictions.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [#figure(
    image("../figures/categorical_distributions.png", width: 100%),
    caption: [Categorical feature distributions across churn states. Contract types and demographics reveal significant behavioral differences between retained and churned cohorts.],
  ) <fig:cat_dist>],
  [#figure(
    image("../figures/class_distribution.png", width: 100%),
    caption: [Class distribution showing churned customers (56.7%) as the majority class, necessitating weighted loss to prevent naive predictions.],
  ) <fig:class_dist>],
)

A critical business finding during our exploratory analysis was the *Generational Gap* in churn rates (@fig:cat_dist). Younger demographics exhibited significantly higher volatility and propensity to churn compared to older, more established customers. This generational divide suggests that retaining younger users requires more dynamic, flexible subscription models, whereas older users value stability and consistent support. Identifying this gap early allowed us to ensure our model correctly weighted age and tenure features.
== Methodology

The preprocessing pipeline involved removing arbitrary identifiers (CustomerID), handling missing values via listwise deletion, applying one-hot encoding (`drop_first=True`) for nominal variables to prevent multicollinearity, and standardizing features via `StandardScaler`.

The shallow neural network was implemented entirely in NumPy. The architecture utilizes 12 input neurons, 32 hidden neurons with Tanh activation (chosen because it is zero-centred, mapping inputs to (−1, 1), ensuring unbiased gradient updates), and 1 output neuron with Sigmoid activation. We implemented the Adam optimiser #cite(<kingma2015>) from scratch to ensure adaptive per-parameter learning rates, converging faster than standard SGD.

== Threshold Optimisation for Business Value

The base model achieved excellent validation performance (F1=0.9826, AUC=0.9974). However, relying on the default decision threshold of 0.5 is suboptimal in a real business environment where the cost of a false positive (unnecessary retention discount) differs from a false negative (lost customer).

#figure(
  image("../figures/tuned_threshold_optimisation.png", width: 75%),
  caption: [Threshold optimization curve illustrating the optimal decision boundary for maximizing F1-score and balancing precision-recall trade-offs based on business costs.],
) <fig:thresh_opt>

As illustrated in @fig:thresh_opt, threshold tuning provides a perfect solution. By shifting the decision boundary to 0.31, the business can accurately identify high-risk customers before they churn. This allows for targeted, cost-effective retention campaigns without over-allocating marketing resources, effectively translating mathematical probabilities into direct business value.

= Task 2: 3D Brain Tumor Segmentation

== Dataset and Preprocessing

Brain tumor segmentation from multi-modal 3D MRI scans is a critical challenge in modern medical imaging. For this task, we utilised the prestigious *BraTS 2020 Dataset* #cite(<brats2020>), sourced via Kaggle. It is the gold standard for multi-modal 3D brain tumor segmentation, containing four co-registered MRI sequences per patient (FLAIR, T1, T1ce, T2) and meticulously annotated expert ground truth masks demarcating the Necrotic core (NCR/NET), Edema (ED), and Enhancing tumor (ET).

Processing full-resolution 3D medical images (240 × 240 × 155 voxels) presents extreme memory constraints. Our rigorous preprocessing pipeline involved:
1. *Intensity Normalization*: Z-score normalization was independently applied exclusively to non-zero brain regions to standardise MRI intensities.
2. *Patch-based Cropping*: We implemented a dynamic random $96 times 96 times 96$ sub-volume extraction strategy. This approach not only conserved GPU VRAM but also served as a robust data augmentation technique.
3. *Class Balancing*: A custom sampling strategy ensured patches were equally drawn from tumor and background regions to prevent the model from collapsing into predicting only the dominant background class.

== 3D ResNet Architecture

To tackle the complexities of spatial hierarchies in 3D volumetric data, we implemented a state-of-the-art *3D ResNet* architecture featuring a deep encoder-decoder topology.

#figure(
  image("../figures/task2_3d_resnet.svg", width: 90%),
  caption: [Architectural diagram of the 3D ResNet model featuring an encoder with residual blocks, high-resolution skip connections, and a transposed convolution decoder pipeline.],
) <fig:resnet_arch>

As visualized in @fig:resnet_arch, the model architecture leverages:
- *Residual Blocks*: 3D convolutions with residual skip connections to solve the vanishing gradient problem in deep volumes, allowing for highly expressive feature extraction.
- *Encoder-Decoder Structure*: The encoder contracts the spatial dimensions to extract hierarchical, context-aware semantic features, while the decoder utilizes ConvTranspose3D layers to upsample these feature maps back to the original resolution.
- *Skip Connections*: Crucially, high-resolution feature maps from the encoder are concatenated directly with decoder features. This preserves fine tumor boundaries and intricate morphological details that would otherwise be permanently lost during the downsampling process.

== Results and Visual Evaluation

The model was comprehensively evaluated on an unseen test cohort of 10 patients using strict spatial metrics: Dice Coefficient, Intersection over Union (IoU), and Sensitivity.

#figure(
  image("../figures/task2_metrics_results_sexy.png", width: 85%),
  caption: [Mean performance metrics across the 10 test patients. The bar graph demonstrates robust predictive capabilities, particularly in isolating Edema and Enhancing Tumor regions.],
) <fig:task2_metrics>

The empirical results (@fig:task2_metrics) demonstrate that the model successfully isolates hyperintense tumor regions, achieving strong overlap scores for both Edema and the Enhancing Tumor core.

=== Visual Predictions

The qualitative performance is evident in the 3D spatial predictions compared directly against the expert ground truth annotations.

#figure(
  image("../pred_images/051/Test_051.png", width: 80%),
  caption: [Axial slice prediction for Patient 051. The model accurately demarcates the edema (green) and enhancing tumor boundaries (blue).],
) <fig:pred_051>

#figure(
  image("../pred_images/055/3D_Test_055.png", width: 80%),
  caption: [3D render visualization for Patient 055 demonstrating robust spatial segmentation quality across the complete volumetric structure.],
) <fig:pred_055>

= Conclusion and Future Work

This report successfully demonstrated advanced deep learning applications across two highly distinct domains: tabular business data and 3D medical imaging.

*Task 1* highlighted the paramount importance of exploratory data analysis, successfully identifying a crucial "Generational Gap" in churn behaviour. By utilizing a highly optimized, pure NumPy shallow neural network and tuning the decision threshold (@fig:thresh_opt), we provided a highly actionable, interpretable framework for businesses to actively retain volatile customers.

*Task 2* conquered the severe memory and spatial complexities of 3D medical imaging by employing a robust 3D ResNet architecture on the BraTS 2020 dataset. The model achieved strong spatial overlap metrics while maintaining fine anatomical boundary details via skip connections, proving its viability as an automated diagnostic aid.

*Future Work:*
1. *Task 1 Enhancements*: Integrating temporal sequence models (such as LSTMs or GRUs) to track customer behavior changes over time, rather than relying on static snapshots, providing earlier warnings of impending churn.
2. *Task 2 Enhancements*: Exploring state-of-the-art vision transformers (e.g., UNETR or Swin-UNETR) to capture long-range global dependencies in 3D MRI volumes, and implementing test-time augmentation (TTA) to further refine boundary predictions on the critical enhancing tumor core.

#heading(numbering: none)[References]

#bibliography("refs.yml", style: "harvard-cite-them-right")
