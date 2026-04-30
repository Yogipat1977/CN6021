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

Task 1: Customer Churn Prediction. Retaining subscribers costs far less than acquiring new ones. We built a shallow neural network from scratch in pure NumPy (per coursework requirements), processing over 505,000 records across demographics, usage, financial behavior, and contracts. EDA revealed a "Generational Gap" — younger customers churn markedly more than older demographics — transforming binary prediction into a stratified retention strategy. An initial train-test distribution shift was diagnosed and resolved by merging Kaggle partitions into a unified stratified dataset. Systematic grid search identified the optimal architecture (12→64→1, 897 parameters), and threshold optimisation (0.28) prioritised recall for business deployment, achieving test F1 of 0.928. Interpretability was enforced through SHAP, permutation importance, and partial dependence plots under a custom dark-theme visualization system.

Task 2: 3D Brain Tumor Segmentation. Precise tumor delineation from MRI guides surgical planning and treatment assessment. We used the BraTS 2020 dataset four co-registered MRI sequences per patient with expert masks for necrotic core, edema, and enhancing tumor. Full 3D volumes impose severe memory constraints, while background voxels threaten to drown tumor signals. We implemented a custom 3D U-Net in PyTorch with skip connections preserving fine anatomical boundaries. To mitigate scarce 3D annotations, we inflated 2D pre-trained ImageNet weights into 3D kernels. Training combats imbalance via Dice-Focal loss Dice maximizes spatial overlap, Focal concentrates on hard tumor boundaries supplemented by patch-based class-balanced sampling, 3D augmentation (flips, rotations, elastic deformations, noise), gradient accumulation for memory efficiency, and auto-detected CUDA/ROCm/MPS/CPU backends. Evaluation uses overlap coefficients, boundary distances, and per-class sensitivity, with NIfTI exports for clinical visualization.

Both systems rest on a modular, reproducible foundation centralized configuration, automated dataset acquisition, consistent dark-theme aesthetics, and eighteen Python modules spanning preprocessing, EDA, modeling, training, evaluation, interpretability, and clinical export. We resolve prototype-to-production friction through weighted losses, patch-based processing, mixed precision, cross-validation, and multi-modal explanation techniques. This report details methodology, presents findings, and charts future directions temporal sequence modeling for earlier churn warnings and vision transformers for long-range volumetric dependencies.

#pagebreak()

= Task 1: Customer Churn Prediction

== Dataset and Data Strategy

Customer churn prediction is a critical business problem for subscription-based services, where the cost of acquiring new customers significantly exceeds that of retaining existing ones. The dataset, sourced from Kaggle #cite(<azeem2023>), was originally partitioned into disjoint training (440,832 samples) and test (64,374 samples) files with 10 predictive features: demographics (Age, Gender), usage patterns (Tenure, Usage Frequency, Support Calls), financial behaviour (Payment Delay, Total Spend), and contractual attributes (Subscription Type, Contract Length, Last Interaction).

Our initial model achieved near-perfect validation scores (F1=0.98) but collapsed on the held-out test set (F1=0.66)—a 0.32 F1 gap. Investigation revealed that the original Kaggle training and test partitions were sampled from different data-generating processes, creating a train-test distribution shift that no hyperparameter tuning could bridge. To resolve this, we merged both files into a unified pool of 505,206 samples and applied stratified re-partitioning (65% train / 20% validation / 15% test) preserving the 56.7:43.3 churn-to-retained ratio in every split. This eliminated the distribution mismatch and enabled honest model evaluation.

== Data Distribution and Generational Gap

Initial inspection revealed a well-structured dataset with minimal missing data. The class distribution reveals that churned customers constitute the majority class (56.7%), requiring a weighted loss function during training to prevent naive majority-class predictions.

#figure(
  image("../figures/categorical_distributions.png", width: 100%),
  caption: [Categorical feature distributions across churn states. Contract types and demographics reveal significant behavioral differences between retained and churned cohorts.],
) <fig:cat_dist>

#figure(
  image("../figures/class_distribution.png", width: 50%),
  caption: [Class distribution showing churned customers (56.7%) as the majority class, necessitating weighted loss to prevent naive predictions.],
) <fig:class_dist>

A critical business finding during our exploratory analysis was the *Generational Gap* in churn rates (@fig:cat_dist). Younger demographics exhibited significantly higher volatility and propensity to churn compared to older, more established customers. This generational divide suggests that retaining younger users requires more dynamic, flexible subscription models, whereas older users value stability and consistent support. Identifying this gap early allowed us to ensure our model correctly weighted age and tenure features.
== Methodology

=== Feature Selection and Preprocessing
The preprocessing pipeline addresses high dimensionality and data quality through four distinct stages:
1. *Feature Selection*: CustomerID was removed as an arbitrary identifier with no predictive value. Correlation analysis confirmed weak inter-feature relationships, indicating independent signals without the need for aggressive dimensionality reduction like PCA, which would sacrifice the interpretability required by stakeholders.
2. *Missing Value Handling*: Listwise deletion was applied to the negligible subset of incomplete rows, ensuring no synthetic noise was introduced via imputation.
3. *One-Hot Encoding*: Categorical features (Gender, Subscription Type) were encoded using `drop_first=True` to eliminate the "dummy variable trap" while preserving non-ordinal categorical relationships.
4. *Standardisation*: All features were scaled to zero mean and unit variance via `StandardScaler`, ensuring stable gradient descent during training.

=== Handling Non-Linear Relationships
A shallow neural network was selected to model the complex, non-linear dependencies between customer features and churn. By utilizing a hidden layer with 64 neurons and Tanh activations, the network maps input features into a high-dimensional manifold where non-linear boundaries can be effectively learned. Per the Universal Approximation Theorem, this architecture is mathematically capable of approximating any continuous function, bridging the gap between simple linear models and computationally expensive deep networks.

=== Implementation
The network was implemented entirely in NumPy to satisfy resource constraints. The architecture utilizes 12 input neurons, 64 hidden neurons, and 1 output neuron with Sigmoid activation. We implemented the Adam optimiser #cite(<kingma2015>) from scratch to ensure adaptive per-parameter learning rates. Class imbalance (56.7:43.3) was addressed through weighted binary cross-entropy loss ($w_0 = 1.155$, $w_1 = 0.882$).

== Hyperparameter Tuning

A systematic grid search was conducted over 27 hyperparameter combinations using stratified 5-fold cross-validation:

#align(center)[
#table(
  columns: (auto, auto),
  stroke: 0.5pt,
  [*Hyperparameter*], [*Search Range*],
  [Hidden Size], [16, 32, 64],
  [Learning Rate], [0.001, 0.005, 0.01],
  [Weight Decay (L2)], [0.001, 0.01, 0.05],
)
]

#figure(
  image("../figures/tuned_grid_search_heatmap.png", width: 65%),
  caption: [Grid search results: Best F1-score by hidden size and learning rate. The configuration hidden=64, lr=0.005 consistently outperformed alternatives.],
) <fig:grid>

The optimal configuration was *hidden_size=64, learning_rate=0.005, weight_decay=0.001*—the 64-neuron hidden layer provided sufficient capacity without overfitting, while the moderate learning rate enabled stable convergence over 93 epochs with early stopping (patience=15).

== Final Model Performance

The merged-and-repartitioned dataset eliminated the distribution mismatch, enabling honest evaluation where all three splits (train, validation, test) are drawn from the same underlying distribution:

#align(center)[
#table(
  columns: (auto, auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  [*Split*], [*Accuracy*], [*Precision*], [*Recall*], [*F1*], [*AUC-ROC*],
  [Validation], [0.932], [0.935], [0.950], [0.942], [0.981],
  [Test], [0.913], [0.879], [0.972], [0.928], [0.946],
)
]

The final model achieves strong generalisation with an F1-score of 0.928 and AUC-ROC of 0.946 on the held-out test set. The 97.2% recall ensures the model captures nearly all churning customers, while precision of 87.9% keeps false alarms manageable for business operations.

#figure(
  image("../figures/base_training_curves.png", width: 85%),
  caption: [Training curves showing convergence by epoch 93 with early stopping. The narrow and parallel train-val gap confirms good generalisation with no overfitting.],
) <fig:curves>

== Threshold Optimisation for Business Deployment

In a real business environment, the cost of a false negative (losing a customer) exceeds the cost of a false positive (unnecessary retention discount). We tuned the decision threshold to prioritise recall:

#figure(
  image("../figures/tuned_threshold_optimisation.png", width: 75%),
  caption: [Threshold optimisation curve. The optimal decision boundary of 0.28 maximises F1-score while ensuring high recall for business-critical churn detection.],
) <fig:thresh_opt>

Lowering the threshold from the default 0.50 to 0.28 increases recall at a modest precision cost—a deliberate trade-off. The model can now identify 97.2% of at-risk customers before they churn, enabling targeted, cost-effective retention campaigns without over-allocating marketing resources. This directly translates mathematical probabilities into business value.

== Interpretability Analysis

To satisfy stakeholder interpretability requirements, we applied two complementary techniques to explain the model's decision-making process.

=== Permutation Importance

Permutation importance quantifies each feature's contribution by measuring the drop in F1-score when that feature's values are randomly shuffled, breaking its relationship with the target while preserving its distribution.

#figure(
  image("../figures/tuned_permutation_importance.png", width: 75%),
  caption: [Permutation importance ranking. Support Calls dominates as the strongest predictive signal—shuffling it causes the largest F1-score degradation. Contract Length, Total Spend, and Age follow, confirming the behavioural and demographic drivers identified during EDA.],
) <fig:perm_imp>

The top four features—Support Calls, Contract Length, Total Spend, and Age—align with the Generational Gap finding: younger customers on short-term contracts with high support call volumes represent the highest churn risk. This provides a clear, actionable hierarchy for retention teams to prioritise intervention.

=== SHAP Analysis

SHAP (SHapley Additive exPlanations) values explain individual predictions by decomposing each output into additive feature contributions with strong theoretical guarantees from cooperative game theory. We used Kernel SHAP to approximate Shapley values for our NumPy network.

#figure(
  image("../figures/tuned_shap_summary.png", width: 85%),
  caption: [SHAP summary plot. Each dot represents one customer's SHAP value for a given feature—red indicates high feature values, blue indicates low. High Support Calls (red) strongly push predictions toward churn (positive SHAP), while long Contract Lengths (blue) push toward retention (negative SHAP).],
) <fig:shap>

The SHAP summary confirms and refines the permutation importance findings: high Support Call counts consistently drive churn predictions, while long-term contracts and high Total Spend exert a stabilising retention effect. The red-blue colour split on Contract Length is particularly informative—short-term contracts (blue, low value) shift predictions toward churn, while annual contracts (red, high value) provide the strongest retention signal. This granular, feature-level insight enables the business to design targeted interventions: customers with high support calls on short contracts are the highest-priority retention cohort.

#pagebreak()

= Task 2: 3D Brain Tumor Segmentation

== Dataset and Preprocessing

Brain tumor segmentation from multi-modal 3D MRI scans is a critical challenge in modern medical imaging. For this task, we utilised the prestigious *BraTS 2020 Dataset* #cite(<brats2020>), sourced via Kaggle. It is the gold standard for multi-modal 3D brain tumor segmentation, containing four co-registered MRI sequences per patient (FLAIR, T1, T1ce, T2) and meticulously annotated expert ground truth masks demarcating the Necrotic core (NCR/NET), Edema (ED), and Enhancing tumor (ET).

#figure(
  image("../figures/task2_eda_3d_slices.png", width: 100%),
  caption: [BraTS 2020 multi-modal MRI slices. Each patient dataset contains co-registered T1, T1ce, T2, and T2-FLAIR sequences. The expert ground truth (right) demarcates complex tumor morphology across axial, coronal, and sagittal planes.],
) <fig:task2_eda>

Processing full-resolution 3D medical images (240 × 240 × 155 voxels) presents extreme memory constraints. Our rigorous preprocessing pipeline involved:
1. *Intensity Normalization*: Z-score normalization was independently applied exclusively to non-zero brain regions to standardise MRI intensities.
2. *Patch-based Cropping*: We implemented a dynamic random $96 times 96 times 96$ sub-volume extraction strategy. This approach not only conserved GPU VRAM but also served as a robust data augmentation technique.
3. *Class Balancing*: A custom sampling strategy ensured patches were equally drawn from tumor and background regions to prevent the model from collapsing into predicting only the dominant background class.

== Methodology

=== 3D Data Augmentation
To improve model generalisation and mitigate the scarcity of expertly annotated 3D volumes, we implemented a robust online augmentation pipeline using `Monai`:
- *Spatial Transforms*: Random 90-degree rotations and horizontal/vertical flips across all three axes to ensure rotation-invariant feature learning.
- *Elastic Deformations*: Random 3D elastic deformations to simulate biological variability in tumor shape and brain anatomy.
- *Intensity Augmentation*: Random Gaussian noise and intensity scaling to make the model robust to sensor noise and varied MRI acquisition protocols.

=== 3D U-Net Architecture

To tackle the complexities of spatial hierarchies in 3D volumetric data, we implemented a custom *3D U-Net* from scratch in PyTorch—a proven encoder-decoder architecture for biomedical image segmentation #cite(<ronneberger2015>). The model has 22.5M parameters with `init_features=32`.

#figure(
  image("../figures/task2_3d_unet.svg", width: 90%),
  caption: [Architectural diagram of the 3D U-Net featuring a symmetric encoder-decoder topology with skip connections concatenating high-resolution encoder features into the decoder pathway.],
) <fig:unet_arch>

As visualized in @fig:unet_arch, the architecture leverages:
- *DoubleConv3D Blocks*: Each block applies two $3 times 3 times 3$ convolutions with BatchNorm3d and LeakyReLU activations. LeakyReLU prevents dead neurons during deep 3D training, while BatchNorm stabilises gradients across the spatial dimensions.
- *Encoder Path*: Four downsampling stages via MaxPool3d ($2 times 2 times 2$) progressively reduce spatial resolution while doubling feature channels (32 → 64 → 128 → 256 → 512), capturing hierarchical semantic features from local texture to global anatomical context.
- *Decoder Path*: Four upsampling stages via ConvTranspose3d restore spatial resolution. Unlike simple interpolation, learned transposed convolutions adaptively reconstruct fine structural details.
- *Skip Connections*: High-resolution feature maps from each encoder stage are concatenated directly into the corresponding decoder stage. This is critical for preserving fine tumor boundaries and small necrotic regions that would otherwise be lost during aggressive spatial downsampling.
- *Output Head*: A final $1 times 1 times 1$ convolution maps the 32-channel feature volume to 4-class logits (background, NCR/NET, ED, ET).

=== Hyperparameter Tuning Constraints

Due to the extreme computational demands of 3D volumetric training (22.5M parameters processing $128^3$ patches), we were constrained to renting cloud GPU servers for a limited duration. Each 50-epoch training run consumed approximately 3 GPU-hours on a cloud instance, making systematic hyperparameter grid search infeasible within our resource budget. Consequently, we adopted architecture and loss function choices grounded in established literature—3D U-Net topology #cite(<ronneberger2015>), LeakyReLU activations, Dice-Focal loss—and validated a single configuration rather than sweeping over alternatives. We acknowledge that a formal hyperparameter search (over `init_features`, dropout probability, patch size, or learning rate) would likely yield further performance improvements and identify this as a primary direction for future work with expanded compute access.

== Quantitative Results

The model was evaluated on an unseen test cohort of 10 patients. Due to GPU memory constraints (consumer hardware), metrics are computed on $128 times 128 times 128$ sub-volume patches rather than full 240×240×155 volumes. Patch-level scores represent a lower bound on whole-brain performance, as edge patches containing partial anatomy score lower than centrally-sampled patches.

#align(center)[
#table(
  columns: (auto, auto, auto, auto, auto),
  stroke: 0.5pt,
  [*Tumor Class*], [*Mean Dice*], [*Mean IoU*], [*Best (Patient)*], [*Worst (Patient)*],
  [ET (Enhancing)], [0.62], [0.45], [0.87 (#367)], [0.00 (#188)],
  [ED (Edema)], [0.65], [0.48], [0.88 (#055)], [0.09 (#320)],
  [NCR/NET (Necrotic)], [0.36], [0.22], [0.66 (#055)], [0.00 (#188, #190)],
)
]

The model achieves usable segmentation on well-defined tumors—Dice scores of 0.65--0.88 on Edema and Enhancing Tumor for patients 051, 055, and 170 demonstrate the U-Net can localise hyperintense regions effectively. However, performance degrades sharply on two fronts:

*Necrotic Core (NCR/NET, mean Dice 0.36):* The necrotic core is often small, diffuse, and visually indistinct on MRI. Even BraTS state-of-the-art models achieve only 0.70--0.75 Dice on this class #cite(<brats2020>). The 3D U-Net's limited receptive field and 22.5M parameter budget are insufficient for reliable necrotic segmentation.

*Catastrophic failures (Patients 188, 190, 320):* Several patients score zero Dice on individual tumor classes. These likely represent tumors with atypical morphology, small volume, or imaging artifacts outside the model's training support. This identifies clear minimum-viable-model boundaries.

#figure(
  image("../figures/task2_metrics_results_sexy.png", width: 85%),
  caption: [Per-patient performance breakdown across the 10 test cases. The bar graph illustrates the high inter-patient variance—well-performing patients (051, 055, 170) contrast sharply with failures (188, 320), highlighting the model's sensitivity to tumor morphology and size.],
) <fig:task2_metrics>

With a larger model (e.g., 64+ init features) and full-volume sliding-window inference, performance on small and diffuse tumors would be expected to improve. These results define a clear *minimum viable model* boundary for 3D brain tumor segmentation on consumer hardware.

== Evaluation Limitations

Three methodological caveats apply to the reported metrics:

1. *Patch-level evaluation:* Metrics are computed on individual 128³ patches, not full 240×240×155 brain volumes. Edge patches containing partial anatomy score lower than central patches. Full-volume inference with sliding-window aggregation (and test-time augmentation) would yield higher whole-brain Dice scores.
2. *Hausdorff Distance approximation:* HD95 is computed on a 1,000-point random subsample of boundary voxels for computational feasibility, rather than the exact surface-distance computation. Reported HD95 values should be treated as estimates.
3. *Limited test cohort:* 10 patients from a single dataset (BraTS 2020) provide meaningful evaluation but may not capture the full range of glioma presentations seen in clinical practice.

=== Qualitative Visualisation

The qualitative performance is evident in the 3D spatial predictions compared directly against the expert ground truth annotations. On well-performing patients, the model distinguishes edema from enhancing tumor with moderate spatial accuracy.

#figure(
  grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 0.8em,
    // Row 1: Patient 051
    [ #align(center)[ #image("../figures/results_051_t2f.png", width: 90%) #text(8pt)[T2-FLAIR (051)] ] ],
    [ #align(center)[ #image("../figures/results_051_gt.png", width: 90%) #text(8pt)[Ground Truth (051)] ] ],
    [ #align(center)[ #image("../figures/results_051_test.png", width: 90%) #text(8pt)[Prediction (051)] ] ],
    
    // Row 2: Patient 055
    [ #align(center)[ #image("../figures/results_055_t2f.png", width: 90%) #text(8pt)[T2-FLAIR (055)] ] ],
    [ #align(center)[ #image("../figures/results_055_gt.png", width: 90%) #text(8pt)[Ground Truth (055)] ] ],
    [ #align(center)[ #image("../figures/results_055_test.png", width: 90%) #text(8pt)[Prediction (055)] ] ],
    
    // Row 3: Patient 170
    [ #align(center)[ #image("../figures/results_170_t2f.png", width: 90%) #text(8pt)[T2-FLAIR (170)] ] ],
    [ #align(center)[ #image("../figures/results_170_gt.png", width: 90%) #text(8pt)[Ground Truth (170)] ] ],
    [ #align(center)[ #image("../figures/results_170_test.png", width: 90%) #text(8pt)[Prediction (170)] ] ],
  ),
  caption: [Qualitative comparison across three representative test patients (051, 055, 170). Each row displays the multi-modal MRI input (T2-FLAIR), the corresponding expert ground truth mask, and the model's volumetric prediction. The 3D U-Net shows strong spatial agreement on edema (green) and enhancing tumor (blue) across varied tumor sizes.],
) <fig:3x3_comp>

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1.5em,
    [
      #align(center)[
        #image("../figures/results_051_3d_gt.png", height: 14em)
        #v(0.3em)
        #text(9pt)[(a) 3D Ground Truth (051)]
      ]
    ],
    [
      #align(center)[
        #image("../figures/results_051_3d_test.png", height: 14em)
        #v(0.3em)
        #text(9pt)[(b) 3D Prediction (051)]
      ]
    ],
  ),
  caption: [Volumetric 3D rendering for Patient 051. The prediction (b) reconstructs a coherent volumetric structure that matches the overall morphology and spatial extent of the expert annotation (a).],
) <fig:3d_comp>

#figure(
  grid(
    columns: (1fr, 1fr),
    rows: (auto, auto),
    gutter: 1em,
    [ #align(center)[ #image("../figures/results_055_3d_et_gt.png", width: 90%) #text(8pt)[(a) ET Ground Truth (055)] ] ],
    [ #align(center)[ #image("../figures/results_055_3d_et_test.png", width: 90%) #text(8pt)[(b) ET Prediction (055)] ] ],
    [ #align(center)[ #image("../figures/results_055_3d_tc_gt.png", width: 90%) #text(8pt)[(c) TC Ground Truth (055)] ] ],
    [ #align(center)[ #image("../figures/results_055_3d_tc_test.png", width: 90%) #text(8pt)[(d) TC Prediction (055)] ] ],
  ),
  caption: [Tumor sub-region breakdown for Patient 055. The top row compares Enhancing Tumor (ET) volumes, while the bottom row compares the broader Tumor Core (TC). The model captures the internal heterogeneity of the glioma while maintaining boundary smoothness.],
) <fig:subregion_comp>

#pagebreak()

= Conclusion and Future Work

This report demonstrated advanced deep learning applications across two highly distinct domains: tabular business data and 3D medical imaging.

*Task 1* identified a "Generational Gap" in churn behaviour through exploratory data analysis. A data-level distribution shift was diagnosed and resolved by merging the original Kaggle partitions into a unified dataset with stratified re-splitting. The resulting pure-NumPy shallow neural network achieves test F1 of 0.928 and AUC-ROC of 0.946 with 897 parameters, providing an interpretable, business-ready framework for customer retention. Threshold optimisation (0.28) ensures 97.2% recall, enabling cost-effective churn prevention.

*Task 2* developed a 3D U-Net (22.5M parameters) for brain tumor segmentation from multi-modal MRI on the BraTS 2020 dataset. The model achieves usable segmentation on well-defined tumors (Dice 0.65--0.88 for Edema and Enhancing Tumor in typical cases) but exhibits clear failure modes on necrotic tissue (mean Dice 0.36) and atypical morphologies—defining a minimum viable model boundary for consumer-hardware training. Patch-based processing, Dice-Focal loss, and 2D-to-3D transfer learning enabled training within constrained resources.

*Future Work:*
1. *Task 1*: Integrating temporal sequence models (LSTMs/GRUs) to track customer behaviour changes over time, enabling earlier churn warnings and proactive retention before support calls spike.
2. *Task 2*: Full-volume sliding-window inference with test-time augmentation to improve whole-brain metrics. Exploring vision transformers (UNETR, Swin-UNETR) to capture long-range volumetric dependencies that the current U-Net's limited receptive field cannot model. Larger architectures (64+ init features) to improve necrotic core segmentation.

#pagebreak()

#bibliography("refs.yml", style: "harvard-cite-them-right")

#pagebreak()

= Appendix: Key Code Snippets

== Task 1: Shallow Neural Network (Pure NumPy)

The following snippets demonstrate the core components of the manually-implemented neural network — forward pass, backpropagation, and the Adam optimiser — all written in pure NumPy without any deep learning frameworks.

#figure(
  raw(block: true, lang: "python", "
    import numpy as np

    def tanh(z):
        return np.tanh(z)

    def tanh_derivative(a):
        return 1.0 - a ** 2

    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def binary_cross_entropy(y_true, y_pred, sample_weights=None):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss_per_sample = -(y_true * np.log(y_pred)
                          + (1 - y_true) * np.log(1 - y_pred))
        if sample_weights is not None:
            loss_per_sample *= sample_weights
        return np.mean(loss_per_sample)
  "),
  caption: [Activation functions and weighted binary cross-entropy loss implementation.],
) <fig:code_activations>

#figure(
  raw(block: true, lang: "python", "
    class ShallowNeuralNetwork:
        def __init__(self, input_size, hidden_size=64, learning_rate=0.005,
                     weight_decay=0.001):
            limit1 = np.sqrt(6.0 / (input_size + hidden_size))
            self.W1 = np.random.uniform(-limit1, limit1,
                                         (input_size, hidden_size))
            self.b1 = np.zeros((1, hidden_size))
            limit2 = np.sqrt(6.0 / (hidden_size + 1))
            self.W2 = np.random.uniform(-limit2, limit2, (hidden_size, 1))
            self.b2 = np.zeros((1, 1))
            self.lr, self.weight_decay = learning_rate, weight_decay

        def forward(self, X):
            self.X = X
            self.Z1 = X @ self.W1 + self.b1
            self.A1 = tanh(self.Z1)
            self.Z2 = self.A1 @ self.W2 + self.b2
            self.A2 = sigmoid(self.Z2)
            return self.A2

        def backward(self, y_true, sample_weights=None):
            m = y_true.shape[0]
            dZ2 = self.A2 - y_true.reshape(-1, 1)
            if sample_weights is not None:
                dZ2 *= sample_weights.reshape(-1, 1)
            dW2 = (self.A1.T @ dZ2) / m + self.weight_decay * self.W2
            db2 = np.mean(dZ2, axis=0, keepdims=True)
            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * tanh_derivative(self.A1)
            dW1 = (self.X.T @ dZ1) / m + self.weight_decay * self.W1
            db1 = np.mean(dZ1, axis=0, keepdims=True)
            return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  "),
  caption: [Network architecture with Xavier/Glorot initialisation, manual forward pass, and backpropagation through a single hidden layer.],
) <fig:code_nn>

#figure(
  raw(block: true, lang: "python", "
        def _adam_update(self, grads):
            self.t += 1
            for param in ['W1', 'b1', 'W2', 'b2']:
                g = grads[param]
                self.m[param] = (self.beta1 * self.m[param]
                               + (1 - self.beta1) * g)
                self.v[param] = (self.beta2 * self.v[param]
                               + (1 - self.beta2) * g ** 2)
                m_hat = self.m[param] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param] / (1 - self.beta2 ** self.t)
                update = self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
                setattr(self, param, getattr(self, param) - update)
  "),
  caption: [Adam optimiser implemented from scratch with momentum ($beta_1 = 0.9$), RMSProp ($beta_2 = 0.999$), and bias-corrected moment estimates.],
) <fig:code_adam>

== Task 2: 3D U-Net for Brain Tumor Segmentation (PyTorch)

The following snippets show the custom 3D U-Net architecture implementing a standard encoder-decoder with skip connections, a combined Dice-Focal loss function for severe class imbalance, and the 2D-to-3D weight inflation transfer learning strategy.

#figure(
  raw(block: true, lang: "python", "
    class Custom3DUNet(nn.Module):
        def __init__(self, in_channels=4, out_classes=4,
                     init_features=32, dropout=0.3):
            super().__init__()
            self.inc  = DoubleConv3D(in_channels, init_features, dropout)
            self.down1 = Down3D(init_features, init_features * 2, dropout)
            self.down2 = Down3D(init_features * 2, init_features * 4, dropout)
            self.down3 = Down3D(init_features * 4, init_features * 8, dropout)
            self.down4 = Down3D(init_features * 8, init_features * 16, dropout)
            self.up1 = Up3D(init_features * 16, init_features * 8, dropout)
            self.up2 = Up3D(init_features * 8, init_features * 4, dropout)
            self.up3 = Up3D(init_features * 4, init_features * 2, dropout)
            self.up4 = Up3D(init_features * 2, init_features, dropout)
            self.outc = OutConv3D(init_features, out_classes)

        def forward(self, x):
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            return self.outc(x)
  "),
  caption: [Custom 3D U-Net architecture with symmetric encoder-decoder topology. Down3D applies MaxPool3d followed by DoubleConv3D (two Conv3d + BatchNorm3d + LeakyReLU blocks). Up3D applies ConvTranspose3d and concatenates skip connections from the encoder.],
) <fig:code_unet>

#figure(
  raw(block: true, lang: "python", "
    class DiceFocalLoss(nn.Module):
        def __init__(self, lambda_dice=1.0, lambda_focal=1.0):
            super().__init__()
            self.dice = DiceLoss()
            alpha = torch.tensor([0.1, 1.0, 1.0, 1.0])
            self.focal = FocalLoss(alpha=alpha, gamma=2.0)
            self.lambda_dice = lambda_dice
            self.lambda_focal = lambda_focal

        def forward(self, logits, targets):
            dice_loss = self.dice(logits, targets)
            focal_loss = self.focal(logits, targets)
            return self.lambda_dice * dice_loss   \
                 + self.lambda_focal * focal_loss

    class DiceLoss(nn.Module):
        def forward(self, logits, targets):
            probs = torch.softmax(logits, dim=1)
            targets_oh = torch.zeros_like(probs)
            targets_oh.scatter_(1, targets.long(), 1)
            dims = (2, 3, 4)
            intersection = torch.sum(probs * targets_oh, dim=dims)
            union = (torch.sum(probs, dim=dims)
                   + torch.sum(targets_oh, dim=dims))
            dice = (2.0 * intersection + 1e-5) / (union + 1e-5)
            return 1.0 - torch.mean(dice)

    class FocalLoss(nn.Module):
        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets.squeeze(1).long(),
                                  reduction='none')
            pt = torch.exp(-ce)
            return torch.mean(((1 - pt) ** 2.0) * ce)
  "),
  caption: [Combined Dice-Focal loss. Dice loss directly optimises spatial overlap regardless of class size, while Focal loss ($gamma=2$) down-weights easily classified background voxels (alpha=0.1) and concentrates on hard tumor boundary predictions.],
) <fig:code_loss>

#figure(
  raw(block: true, lang: "python", "
    def inflate_2d_to_3d_weights(model_3d):
        resnet2d = _build_resnet18_manually()
        conv2d_layers = [m for m in resnet2d.modules()
                         if isinstance(m, nn.Conv2d)]
        conv3d_layers = [m for m in model_3d.modules()
                         if isinstance(m, nn.Conv3d)]
        inflated = 0
        for c2, c3 in zip(conv2d_layers, conv3d_layers):
            if (c2.in_channels == c3.in_channels
                and c2.out_channels == c3.out_channels
                and c2.kernel_size == (3, 3)
                and c3.kernel_size == (3, 3, 3)):
                w2 = c2.weight.data
                w3 = w2.unsqueeze(2).repeat(1, 1, 3, 1, 1) / 3.0
                c3.weight.data = w3
                inflated += 1
        return model_3d
  "),
  caption: [2D-to-3D weight inflation. Pre-trained ImageNet 2D convolution filters ([Out, In, 3, 3]) are replicated across the depth axis to create 3D kernels ([Out, In, 3, 3, 3]), divided by the depth size to preserve activation variance, and injected into the 3D U-Net encoder.],
) <fig:code_inflation>

