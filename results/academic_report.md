# Bird Species Classification Using a Custom Convolutional Neural Network in PyTorch

**Academic Project Report**  
**Framework**: PyTorch 2.14 (CPU execution)  
**Dataset**: Hugging Face `chriamue/bird-species-dataset` (Gerald Piosenka, 2023)  

---

## 1. Title and Project Overview

Fine-grained visual categorization (FGVC) involves distinguishing visually similar subcategories within a broader taxonomic class. In avian classification, differences between species are often subtle—manifested in beak curvature, plumage patterns, or crest formations—while differences within a single species can be pronounced due to lighting, background foliage, and avian posture.

This project implements an end-to-end, reproducible deep learning pipeline in PyTorch to classify bird species from RGB photographic images. Rather than relying on transfer learning from large pretrained vision transformers or ResNets, a custom 4-stage Convolutional Neural Network (CNN) with Batch Normalization, Dropout, and Adaptive Average Pooling was engineered and trained from scratch. The system includes automated data preparation with SHA-256 integrity verification, rigorous evaluation on an isolated held-out test split, learning curve diagnostics, a command-line interface, and an interactive Streamlit demonstration application.

---

## 2. Abstract

This study demonstrates the application of a custom Convolutional Neural Network (CNN) to the classification of 10 bird taxa selected from the Hugging Face `chriamue/bird-species-dataset`. A total of 1,701 colour images were extracted and partitioned into a newly constructed stratified split: 70% training ($1,186$ images), 15% validation ($255$ images), and 15% testing ($260$ images) using a fixed random seed ($42$). Complete disjointness was verified via SHA-256 content hashing ($0$ duplicate hashes across splits). The custom CNN, comprising 4 convolutional blocks followed by Adaptive Average Pooling and a regularized dense head ($p=0.4$ Dropout), was trained using AdamW and a `ReduceLROnPlateau` scheduler. Across 12 scheduled training epochs on CPU hardware, the model reached its minimum validation loss of $0.8883$ at Epoch 11 ($67.06\%$ validation accuracy). Evaluated on the held-out test partition of 260 unseen images, the model achieved an overall accuracy of **68.08%**, a macro-averaged precision of **0.7202**, a macro-averaged recall of **0.6753**, and a macro-averaged F1-score of **0.6703**. These results provide an interpretable baseline illustrating feature convergence, regularization dynamics, and the limitations of closed-world vision models without pretrained representations.

---

## 3. Problem Statement and Objectives

### 3.1 Problem Statement
Avian species identification in the wild is traditionally performed by experienced ornithologists. Automating this task with computer vision is hindered by:
1. High intra-class variance due to flight angles, molting stages, lighting conditions, and cluttered natural foliage backgrounds.
2. Low inter-class variance among morphologically similar species (e.g., hornbill species sharing dark bodies and elongated casques).
3. The risk of spatial overfitting when training deep neural architectures from scratch on moderate sample sizes.

### 3.2 Objectives
1. **Pipeline Engineering**: Construct an automated data pipeline using selective HTTP archive streaming to extract only the target species from Hugging Face archives without transferring the full 1.78 GB dataset.
2. **Data Integrity & Partitioning**: Formulate an approximately balanced 10-class dataset with verified zero-leakage partitions confirmed via SHA-256 cryptographic hashing.
3. **Architecture Design**: Implement a modular, interpretable 4-stage CNN architecture operating directly on $224 \times 224$ RGB inputs.
4. **Regularization & Optimization**: Mitigate over-adaptation using spatial data augmentation, Batch Normalization, weight decay ($10^{-4}$), and dense Dropout ($p=0.4$).
5. **Rigorous Held-Out Evaluation**: Evaluate the frozen optimal checkpoint exclusively on held-out test data, reporting precision, recall, F1-score, support, and confusion matrix diagnostics.

---

## 4. Dataset Source and Selection of 10 Species

### 4.1 Primary Data Source
The primary dataset is the Hugging Face repository [`chriamue/bird-species-dataset`](https://huggingface.co/datasets/chriamue/bird-species-dataset), derived from Gerald Piosenka's Kaggle 525 Bird Species collection. The original repository contains 525 taxa with over 89,000 photographic images.

### 4.2 Deterministic Selection of 10 Species
To create a computationally tractable and reproducible academic benchmark, 10 species were selected deterministically based on alphabetical ordering and a minimum threshold of $\ge 130$ usable images per class:

| Class Index | Exact Dataset Label | Common Name | Morphological Characteristics |
| :---: | :--- | :--- | :--- |
| `0` | `ABBOTTS BABBLER` | Abbott's Babbler | Olive-brown plumage, pale throat, short tail |
| `1` | `ABBOTTS BOOBY` | Abbott's Booby | White seabird, black eye patch, large bill |
| `2` | `ABYSSINIAN GROUND HORNBILL` | Abyssinian Ground Hornbill | Terrestrial black bird, blue/red facial skin, large bill |
| `3` | `AFRICAN CROWNED CRANE` | African Crowned Crane | Distinctive golden stiff-feathered head crest |
| `4` | `AFRICAN EMERALD CUCKOO` | African Emerald Cuckoo | Iridescent emerald-green body, yellow breast |
| `5` | `AFRICAN FIREFINCH` | African Firefinch | Scarlet breast and face, small crimson finch |
| `6` | `AFRICAN OYSTER CATCHER` | African Oystercatcher | Uniform black plumage, bright red eye-ring and bill |
| `7` | `AFRICAN PIED HORNBILL` | African Pied Hornbill | Black body with white belly, curved horn bill |
| `8` | `AFRICAN PYGMY GOOSE` | African Pygmy Goose | Small duck, green-black back, orange flanks |
| `9` | `ALBATROSS` | Albatross | Pelagic seabird, expansive white wingspan, hooked bill |

---

## 5. Data Preprocessing and Augmentation

To preserve diagnostic plumage features while promoting geometric invariance, separate transformations were specified:

### 5.1 Training Augmentation Pipeline
- **RGB Standardization**: Converts input to 3-channel RGB, handling variations in JPEG color encoding.
- **Resize**: Rescales input to $256 \times 256$ pixels.
- **Random Resized Crop**: Randomly crops to $224 \times 224$ pixels with scale $[0.85, 1.0]$ and aspect ratio $[0.9, 1.1]$.
- **Random Horizontal Flip**: Applied with probability $p = 0.5$.
- **Random Rotation**: Rotation sampled uniformly from $[-15^{\circ}, +15^{\circ}]$.
- **Tensor Conversion & Normalization**: Scales pixel intensities to $[0.0, 1.0]$ and normalizes with ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 5.2 Evaluation and Inference Pipeline
Deterministic transformations with no random distortions:
- RGB Standardization $\to$ Bilinear Resize to $224 \times 224$ pixels $\to$ Tensor Conversion $\to$ ImageNet Normalization.

---

## 6. Dataset Partition Methodology and Class-Count Table

### 6.1 Partition Methodology
Because the original Hugging Face dataset split allocated only 5 images per class to validation and test partitions (insufficient for stable evaluation support), all validated images for each of the 10 selected species were consolidated across source archives and partitioned into a newly constructed stratified split:
- **70% Training** per species ($1,186$ images total)
- **15% Validation** per species ($255$ images total)
- **15% Held-Out Testing** per species ($260$ images total)

This procedure does not preserve the original Kaggle/HuggingFace train/valid/test partition boundaries; rather, it constructs a newly stratified, reproducible benchmark with fixed seed $42$.

### 6.2 Partition Balance and Image Counts
The dataset is **approximately balanced**, ranging from $142$ total images (`ALBATROSS`) to $197$ total images (`AFRICAN PIED HORNBILL`):

| Species Label | Train (70%) | Validation (15%) | Held-Out Test (15%) | Total Images |
| :--- | :---: | :---: | :---: | :---: |
| `ABBOTTS BABBLER` | 121 | 26 | 26 | 173 |
| `ABBOTTS BOOBY` | 133 | 28 | 29 | 190 |
| `ABYSSINIAN GROUND HORNBILL` | 132 | 28 | 29 | 189 |
| `AFRICAN CROWNED CRANE` | 101 | 22 | 22 | 145 |
| `AFRICAN EMERALD CUCKOO` | 114 | 25 | 25 | 164 |
| `AFRICAN FIREFINCH` | 102 | 22 | 23 | 147 |
| `AFRICAN OYSTER CATCHER` | 115 | 25 | 25 | 165 |
| `AFRICAN PIED HORNBILL` | 137 | 30 | 30 | 197 |
| `AFRICAN PYGMY GOOSE` | 132 | 28 | 29 | 189 |
| `ALBATROSS` | 99 | 21 | 22 | 142 |
| **Total** | **1,186** | **255** | **260** | **1,701** |

### 6.3 Cryptographic Leakage Verification
Every image file across the three partitions was processed through SHA-256 content hashing (`scripts/prepare_data.py --verify-only`):
- **Total files analyzed**: $1,701$
- **Unique SHA-256 hashes**: $1,701$
- **Cross-split duplicate count**: $0$
- **Intra-split duplicate count**: $0$
- **Verified Disjointness**: Complete data isolation with zero shared images.

---

## 7. CNN Architecture and Tensor Dimensions

The model (`BirdClassifierCNN`) was designed for academic interpretability and computational efficiency on CPU hardware.

### 7.1 Structural Progression Table

| Stage | Operation | Input Shape | Output Shape | Output Parameters |
| :--- | :--- | :---: | :---: | :--- |
| **Input** | RGB Image Batch | — | $[B, 3, 224, 224]$ | 3 channels |
| **Block 1** | Conv2d(3, 32, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2) | $[B, 3, 224, 224]$ | $[B, 32, 112, 112]$ | 864 weights |
| **Block 2** | Conv2d(32, 64, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2) | $[B, 32, 112, 112]$ | $[B, 64, 56, 56]$ | 18,432 weights |
| **Block 3** | Conv2d(64, 128, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2) | $[B, 64, 56, 56]$ | $[B, 128, 28, 28]$ | 73,728 weights |
| **Block 4** | Conv2d(128, 256, k=3, p=1) + BatchNorm2d + ReLU + MaxPool2d(2, 2) | $[B, 128, 28, 28]$ | $[B, 256, 14, 14]$ | 294,912 weights |
| **Pool** | AdaptiveAvgPool2d((4, 4)) | $[B, 256, 14, 14]$ | $[B, 256, 4, 4]$ | 0 (non-parametric) |
| **Flatten** | Flatten(start_dim=1) | $[B, 256, 4, 4]$ | $[B, 4096]$ | 0 |
| **Dense** | Linear(4096, 256) + ReLU | $[B, 4096]$ | $[B, 256]$ | 1,048,576 weights |
| **Regularize**| Dropout($p=0.40$) | $[B, 256]$ | $[B, 256]$ | 0 (stochastic mask) |
| **Output** | Linear(256, 10) | $[B, 256]$ | $[B, 10]$ | 2,560 weights |

- **Logits Output**: Outputs unnormalized logits directly into `torch.nn.CrossEntropyLoss`. Softmax is omitted from the internal network graph to preserve numerical stability.

---

## 8. Loss Function, Optimizer, Learning-Rate Scheduler, Regularization, and Early Stopping

### 8.1 Loss Function
Multi-class Cross-Entropy Loss:
$$\mathcal{L}_{\text{CE}} = - \frac{1}{B} \sum_{i=1}^{B} \log \left( \frac{\exp(z_{i, y_i})}{\sum_{j=1}^{C} \exp(z_{i, j})} \right)$$
where $z$ represents the unnormalized logits, $y_i$ is the target class index, and $C = 10$.

### 8.2 Optimizer and Learning Rate Scheduler
- **Optimizer**: AdamW with $\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and weight decay $\lambda = 10^{-4}$.
- **Scheduler**: `ReduceLROnPlateau` monitoring validation loss ($\text{mode} = \text{'min'}$, $\text{factor} = 0.5$, $\text{patience} = 2$, $\eta_{\min} = 10^{-5}$).

### 8.3 Regularization Stack
1. **Dropout ($p = 0.40$)**: Deactivates 40% of hidden activations in the classification head during training (`model.train()`), forcing neurons to learn non-redundant representations; disabled during evaluation (`model.eval()`).
2. **Weight Decay ($10^{-4}$)**: Penalizes large weight vectors via decoupled $L_2$ decay in AdamW.
3. **Data Augmentation**: Disables memorization of image positions and orientations.

### 8.4 Early Stopping Configuration
- **Monitored Metric**: Validation loss ($\mathcal{L}_{\text{val}}$)
- **Patience**: 4 epochs
- **Minimum Improvement Threshold**: $\Delta_{\min} = 0.001$

---

## 9. Experimental Setup and Reproducibility Details

- **Hardware Platform**: Windows x86_64, Multi-core CPU
- **Environment**: Python 3.12, PyTorch 2.14.1+cpu, Torchvision 0.29.1+cpu, Scikit-Learn 1.2+
- **Batch Size**: 32 images per batch
- **Epoch Count**: 12 scheduled epochs
- **Random Seeding**: Fixed random seed $42$ applied across Python `random`, NumPy, and PyTorch (`torch.manual_seed(42)`).

---

## 10. Training/Validation Curves and Their Interpretation

Empirical epoch-by-epoch history from `results/training_history.csv`:

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Learning Rate | Elapsed Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3.1520 | 23.10% | 1.7563 | 37.65% | 0.001 | 88.8s |
| 2 | 1.7734 | 37.02% | 1.4805 | 54.51% | 0.001 | 80.3s |
| 3 | 1.5560 | 44.10% | 1.2609 | 60.39% | 0.001 | 79.1s |
| 4 | 1.4214 | 50.76% | 1.1441 | 56.47% | 0.001 | 73.9s |
| 5 | 1.2631 | 55.48% | 1.1303 | 61.57% | 0.001 | 77.4s |
| 6 | 1.2272 | 57.84% | 1.2702 | 60.00% | 0.001 | 71.9s |
| 7 | 1.2147 | 57.59% | 1.1453 | 59.22% | 0.001 | 70.4s |
| 8 | 1.1154 | 61.30% | 0.9700 | 67.84% | 0.001 | 75.0s |
| 9 | 1.1108 | 60.12% | 0.9456 | 69.80% | 0.001 | 79.8s |
| 10 | 1.0361 | 65.01% | 0.9285 | 68.24% | 0.001 | 80.8s |
| **11\*** | **1.0144** | **62.73%** | **0.8883** | **67.06%** | **0.001** | **79.0s** |
| 12 | 0.9039 | 68.72% | 0.9006 | 72.16% | 0.001 | 79.6s |

*\*Epoch 11 achieved the global minimum validation loss and was chosen as the best checkpoint.*

### 10.1 Diagnostic Interpretation of Curves
1. **Convergence vs. Underfitting**:
   - Training loss dropped by **71.3%** ($3.1520 \to 0.9039$), and training accuracy surged from $23.10\% \to 68.72\%$ ($+45.62\%$ improvement).
   - This substantial reduction and sustained progress above the 10.0% chance level indicate strong feature convergence, showing that underfitting is not a primary concern.
2. **Optimal Checkpoint vs. Accuracy Peak**:
   - The predefined model selection criterion was **minimum validation loss**, which was attained at **Epoch 11** ($\mathcal{L}_{\text{val}} = 0.8883$, $\text{Acc}_{\text{val}} = 67.06\%$).
   - In Epoch 12, validation accuracy achieved its numerical maximum of $72.16\%$, but validation loss increased marginally to $0.9006$ ($+0.0123$ increase) while training loss fell to $0.9039$ ($-0.1105$).
   - This divergence is consistent with the initial onset of sample-specific fitting, demonstrating why validation loss is a reliable criterion for checkpoint selection.
3. **Generalization and Train/Val Gap**:
   - In multiple early-to-mid epochs (e.g., Epochs 1, 2, 8, 9, 12), validation accuracy matched or exceeded training accuracy.
   - This is consistent with the expected effects of active Dropout ($p=0.4$) and data augmentation during training: during training, 40% of hidden neurons are dropped and images are perturbed, while evaluation runs deterministically without dropout or image distortion.
4. **Early Stopping Outcome**:
   - As recorded in `results/training_summary.json`, early stopping did not trigger because only 1 epoch elapsed after the minimum validation loss was achieved at Epoch 11, leaving the patience counter at $1/4$ when the scheduled 12 epochs concluded.

---

## 11. Test Results and Per-Class Metrics

The saved Epoch 11 checkpoint was evaluated on the **260 unseen held-out test images**:

### 11.1 Aggregate Metrics
- **Overall Test Accuracy**: **68.08%** (177 correct predictions out of 260)
- **Macro-Averaged Precision**: **0.7202**
- **Macro-Averaged Recall**: **0.6753**
- **Macro-Averaged F1-Score**: **0.6703**

### 11.2 Per-Class Classification Report

| Species Label | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `ABBOTTS BABBLER` | 0.7407 | 0.7692 | 0.7547 | 26 |
| `ABBOTTS BOOBY` | 0.6154 | 0.8276 | 0.7059 | 29 |
| `ABYSSINIAN GROUND HORNBILL` | 0.5714 | 0.4138 | 0.4800 | 29 |
| `AFRICAN CROWNED CRANE` | 0.5417 | 0.5909 | 0.5652 | 22 |
| `AFRICAN EMERALD CUCKOO` | 0.8077 | 0.8400 | 0.8235 | 25 |
| `AFRICAN FIREFINCH` | 0.8182 | 0.7826 | 0.8000 | 23 |
| `AFRICAN OYSTER CATCHER` | 0.7143 | 0.8000 | 0.7547 | 25 |
| `AFRICAN PIED HORNBILL` | 0.5227 | 0.7667 | 0.6216 | 30 |
| `AFRICAN PYGMY GOOSE` | 0.8696 | 0.6897 | 0.7692 | 29 |
| `ALBATROSS` | **1.0000** | 0.2727 | 0.4286 | 22 |
| **Macro Average** | **0.7202** | **0.6753** | **0.6703** | **260** |
| **Weighted Average** | **0.7129** | **0.6808** | **0.6719** | **260** |

---

## 12. Confusion Matrix Interpretation

Examining `results/confusion_matrix.png` and `results/confusion_matrix.csv`:

1. **High Visual Distinction Classes**:
   - `AFRICAN EMERALD CUCKOO` achieved an **F1-score of 0.8235** (84.0% recall, 80.8% precision) owing to its vivid iridescent green plumage and yellow breast.
   - `AFRICAN FIREFINCH` achieved an **F1-score of 0.8000** (78.3% recall, 81.8% precision) due to its distinctive crimson body and estrildid finch morphology.
2. **Morphological Confusion Patterns**:
   - `ABYSSINIAN GROUND HORNBILL` and `AFRICAN PIED HORNBILL` exhibited noticeable mutual misclassification: both species feature large dark silhouettes and prominent bills, leading to shared feature activations.
   - `ALBATROSS` attained **1.0000 precision** (0 false positives), but had lower recall ($27.3\%$) because several in-flight marine images were classified as `ABBOTTS BOOBY` (which also presents white plumage and oceanic flight poses).

---

## 13. Limitations

1. **Closed-World Constraint**: The model is trained exclusively on 10 bird taxa. It cannot identify bird species outside these 10 categories; any out-of-distribution bird image will still be assigned to one of the 10 classes.
2. **Uncalibrated Confidence Scores**: Softmax output values represent relative class likelihoods under a closed-set prior and are not calibrated Bayesian posterior probabilities.
3. **Hardware Constraints**: The architecture was trained for 12 epochs on CPU hardware. While sufficient for strong feature learning (68.08% accuracy), deeper models with attention mechanisms trained on GPU could extract finer feather micro-textures.
4. **Resolution Cap**: The $224 \times 224$ input resolution limits the network's ability to discern ultra-fine plumage striations and subtle eye-ring differences.

---

## 14. Conclusion

A custom 4-stage Convolutional Neural Network trained from scratch on 10 bird species achieves **68.08% test accuracy** and a **0.6703 macro F1-score** across 260 unseen test images. The pipeline demonstrates:
- The feasibility of training custom CNN architectures from random initialization on domain-specific vision tasks without pretrained weights.
- The utility of cryptographic SHA-256 verification in ensuring strict split isolation ($0$ duplicates across $1,701$ samples).
- The stabilizing effect of Batch Normalization, Dropout ($p=0.4$), and data augmentation in preventing overfitting.

---

## 15. Future Improvements

1. **Fine-Grained Attention**: Integrating Squeeze-and-Excitation (SE) blocks or Convolutional Block Attention Modules (CBAM) to adaptively highlight informative anatomical regions (beaks, crests).
2. **Resolution Scaling**: Training on $384 \times 384$ or $448 \times 448$ inputs to resolve micro-plumage textures.
3. **Self-Supervised Pretraining**: Using Masked Autoencoding (MAE) or SimCLR on unlabeled bird images prior to supervised fine-tuning.
4. **Confidence Calibration**: Applying temperature scaling or Platt scaling on the validation set to produce calibrated probability outputs.

---

## 16. References and Citations

1. Piosenka, G. (2023). *BIRDS 525 SPECIES - Image Classification*. Kaggle Dataset Repository.
2. Hugging Face Dataset Card: `chriamue/bird-species-dataset` ([https://huggingface.co/datasets/chriamue/bird-species-dataset](https://huggingface.co/datasets/chriamue/bird-species-dataset)).
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE CVPR, 770-778.
4. Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). *Dropout: A simple way to prevent neural networks from overfitting*. JMLR, 15(1), 1929-1958.
5. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. International Conference on Learning Representations (ICLR).
