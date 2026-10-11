# Bird Species Classification Using a Custom Convolutional Neural Network in PyTorch

An academic, production-grade Deep Learning system for fine-grained bird species classification using a custom Convolutional Neural Network (CNN) built in PyTorch from scratch without pretrained weights.

---

## 1. Project Overview & Features

Fine-grained visual categorization (FGVC) presents a challenging computer vision task because inter-class variance is subtle (similar bird silhouettes and anatomy) while intra-class variance is large (diverse poses, lighting conditions, and natural foliage backgrounds). This project implements an interpretable, modular 4-stage CNN trained from scratch on a curated 10-species subset of the Hugging Face Bird Species Dataset, satisfying all rigorous academic benchmarks and reproducible ML standards.

### Key Features
- **Custom Modular CNN Architecture**: 4-stage convolutional feature extractor with Batch Normalization, ReLU activations, Max Pooling, and an Adaptive Average Pooling bridge to a dense classification head.
- **Cryptographic Split Integrity**: SHA-256 content hashing across all dataset partitions ensures verified zero data leakage ($0$ duplicates).
- **Reproducible Pipeline**: Deterministic random seeding ($42$) across Python, NumPy, and PyTorch.
- **Robust Regularization Stack**: Spatial data augmentation, weight decay ($10^{-4}$), and Dropout ($p=0.40$).
- **Multi-Faceted Diagnostics**: Held-out test evaluation, confusion matrix visualizations, training/validation loss and accuracy curves, and empirical learning analysis.
- **Dual Prediction Interfaces**: Interactive Streamlit web application with in-memory inference and a command-line interface (CLI) with top-$k$ ranking.

---

## 2. Dataset Source & Selected Classes

- **Primary Source**: [Hugging Face `chriamue/bird-species-dataset`](https://huggingface.co/datasets/chriamue/bird-species-dataset) (based on Gerald Piosenka's 525 Bird Species dataset).
- **Format**: 3-channel RGB colour images ($\sim 224 \times 224$ pixels).
- **Selected Species (Exact Dataset Labels)**:
  1. `ABBOTTS BABBLER`
  2. `ABBOTTS BOOBY`
  3. `ABYSSINIAN GROUND HORNBILL`
  4. `AFRICAN CROWNED CRANE`
  5. `AFRICAN EMERALD CUCKOO`
  6. `AFRICAN FIREFINCH`
  7. `AFRICAN OYSTER CATCHER`
  8. `AFRICAN PIED HORNBILL`
  9. `AFRICAN PYGMY GOOSE`
  10. `ALBATROSS`

### Dataset Preparation & Splitting Strategy
- **Selective HTTP Streaming**: Downloads and extracts only the target species from Hugging Face tar archives without downloading the entire 1.78 GB dataset.
- **Stratified Partitioning**: 70% Training ($1,186$ images), 15% Validation ($255$ images), 15% Held-Out Testing ($260$ images) across all 10 species ($1,701$ total images).
- **Disjoint Partitions**: Verified zero sample overlap using SHA-256 content checksums ($0$ cross-split duplicates, $0$ intra-split duplicates). Detailed in `data/metadata.json`.

---

## 3. CNN Architecture

The model (`BirdClassifierCNN`) avoids pretrained model dependencies and uses a modular 4-stage feature hierarchy:

```
Input: [Batch, 3, 224, 224]
  ├── Stage 1: Conv2d(3 -> 32, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2)   => [B, 32, 112, 112]
  ├── Stage 2: Conv2d(32 -> 64, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2)  => [B, 64, 56, 56]
  ├── Stage 3: Conv2d(64 -> 128, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2) => [B, 128, 28, 28]
  ├── Stage 4: Conv2d(128 -> 256, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2)=> [B, 256, 14, 14]
  ├── Aggregation: AdaptiveAvgPool2d((4, 4))                               => [B, 256, 4, 4]
  ├── Flatten:                                                             => [B, 4096]
  ├── Dense: Linear(4096 -> 256) + ReLU
  ├── Regularization: Dropout(p=0.40)
  └── Classification Head: Linear(256 -> 10)                                => [B, 10]
```

- **Output Representation**: Unnormalized raw logits `[batch_size, 10]`. No internal Softmax is applied to prevent numerical instability with `nn.CrossEntropyLoss`.

---

## 4. Empirical Evaluation Results

Evaluated on the held-out test split of **260 unseen images** using the best checkpoint saved at **Epoch 11**:

| Metric | Result | Description |
| :--- | :---: | :--- |
| **Test Accuracy** | **68.08%** | 177 / 260 correctly classified instances |
| **Macro Precision** | **0.7202** | Unweighted average precision across all 10 species |
| **Macro Recall** | **0.6753** | Unweighted average recall across all 10 species |
| **Macro F1-Score** | **0.6703** | Harmonic mean of macro precision and recall |
| **Optimal Checkpoint** | **Epoch 11** | Minimum validation loss = 0.8883 (Validation Accuracy = 67.06%) |

### Species-Wise Performance Breakdown

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
| `ALBATROSS` | 1.0000 | 0.2727 | 0.4286 | 22 |

---

## 5. System Requirements & Windows Setup

- **Operating System**: Windows 10/11 (or Linux / macOS)
- **Python Version**: Python 3.10 – 3.12 (Tested on Python 3.12.10)
- **Hardware**: Compatible with CPU execution (CUDA GPU detected automatically if available)

### Installation
Open PowerShell in the project directory:
```powershell
cd C:\Users\vivan\.gemini\antigravity-ide\scratch\bird-species-classifier

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

---

## 6. Commands to Prepare, Train, Evaluate, and Predict

### 6.1 Data Preparation & Integrity Check
```powershell
# Option A: Full download, extraction, and stratified splitting
python scripts/prepare_data.py configs/config.yaml

# Option B: Run standalone SHA-256 cryptographic verification on existing dataset
python scripts/prepare_data.py --verify-only
```

### 6.2 Model Training
```powershell
# Trains CNN with automatic checkpointing and history logging
python src/train.py configs/config.yaml
```

### 6.3 Evaluation & Visualization
```powershell
# Evaluates on held-out test set and generates confusion matrix & learning curves
python src/evaluate.py configs/config.yaml
```

### 6.4 Prediction (CLI)
```powershell
# Classify an individual image file with top-k predictions
python src/predict.py --image "data/processed/test/AFRICAN FIREFINCH/train_3_005.jpg" --top_k 3
```

### 6.5 Interactive Streamlit Web UI
```powershell
# Launch Streamlit web demonstration
streamlit run app/streamlit_app.py
```

### 6.6 Run Unit Tests
```powershell
# Run the 11 unit tests covering model, transforms, early stopping, and data integrity
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## 7. Checkpoint Placement & Key File Paths

- **Trained Model Checkpoint**: `models/best_model.pth`
- **Configuration File**: `configs/config.yaml`
- **Dataset Partitions**: `data/processed/{train,val,test}`
- **Dataset Metadata**: `data/metadata.json`
- **Training Epoch History**: `results/training_history.csv`
- **Training Summary & Early Stopping Outcome**: `results/training_summary.json`
- **Test Metrics JSON**: `results/test_metrics.json`
- **Classification Report CSV**: `results/classification_report.csv`
- **Confusion Matrix Plot**: `results/confusion_matrix.png`
- **Loss Curve**: `results/loss_curve.png`
- **Accuracy Curve**: `results/accuracy_curve.png`
- **Learning Analysis**: `results/learning_analysis.md`
- **Formal Academic Report**: `academic_report.md` (and `results/academic_report.md`)

---

## 8. Troubleshooting

- **Windows OpenMP DLL Initialization (`WinError 1114`)**: On Windows, PyTorch's OpenMP runtime can conflict if other C-extensions are loaded first. In all project scripts, `import torch` is positioned at the top of imports to ensure clean DLL initialization.
- **Worker Processes on Windows (`BrokenPipeError`)**: When running PyTorch `DataLoader` on Windows, `num_workers` must be set to `0` (or wrapped in `if __name__ == '__main__':`). The default configuration uses `num_workers: 0` for safe multiprocessing.
- **Missing Checkpoint in Streamlit**: If `models/best_model.pth` is missing, the Streamlit app displays an informational banner instructing the user to run `python src/train.py` first.

---

## 9. Academic Integrity & Limitations Note

- **Closed-World Constraint**: The model is trained exclusively on 10 bird taxa. It does not classify open-world bird species outside these 10 categories.
- **Softmax Score Interpretation**: Output percentages represent relative normalized Softmax probabilities under a closed-set prior, not calibrated Bayesian posterior confidence.
- **Data Partitions**: The dataset partitions are a newly constructed stratified 70/15/15 split ($seed=42$) created from consolidated raw samples, with verified zero content overlap across splits confirmed via SHA-256 hashing.

---

## 10. References

1. Piosenka, G. (2023). *BIRDS 525 SPECIES - Image Classification*. Kaggle Dataset Repository.
2. Hugging Face Dataset: `chriamue/bird-species-dataset` ([https://huggingface.co/datasets/chriamue/bird-species-dataset](https://huggingface.co/datasets/chriamue/bird-species-dataset)).
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. Proceedings of the IEEE CVPR.
4. Srivastava, N., et al. (2014). *Dropout: A simple way to prevent neural networks from overfitting*. JMLR.
5. Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.
