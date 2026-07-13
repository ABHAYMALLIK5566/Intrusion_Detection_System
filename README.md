# 🛡️ Intrusion Detection System — Residual MLP on NSL-KDD

A production-grade deep learning system for **network intrusion detection** trained on the NSL-KDD benchmark dataset.

## Architecture

**Multi-task Residual MLP** (1.75M parameters):

```
Input (122 features)
      │
  [BatchNorm]
      │
  ┌───┴──────────────────────────┐
  │   6 × Residual Blocks        │
  │   [256 → 256 → 512 → 512     │
  │    → 256 → 128]              │
  │   (Linear → BN → GELU → skip)│
  └───────────────────────────────┘
              │
         [Dropout 0.3]
              │
    ┌─────────┴──────────┐
    │                    │
[Binary Head]     [Category Head]
  Dense(64)→BN     Dense(128)→BN
   sigmoid           softmax(5)
      │                   │
Normal / Attack    normal / DoS /
                  Probe / R2L / U2R
```

### Key Design Decisions

| Feature | Choice | Why |
|---------|--------|-----|
| Skip connections | ResNet-style | Prevents vanishing gradients in deep MLPs |
| Activation | GELU | Smoother gradients than ReLU |
| Normalisation | BatchNorm per layer | Stable training, faster convergence |
| Loss (binary) | BCE + label smoothing | Prevents overconfident predictions |
| Loss (category) | Focal Loss (γ=2) | Handles severe class imbalance (U2R: 52 samples) |
| LR schedule | Cosine decay + warm-up | No manual LR tuning needed |
| Class weights | Inverse frequency | Forces model to learn rare attack types |
| Multi-task | Binary + 5-class | Richer representations, learns *what* attack, not just *is* attack |

## Results (evaluated on `KDDTest.txt`)

### Binary Classification — Normal vs Attack

| Metric | Value |
|--------|-------|
| Accuracy | 82.36% |
| F1 Score | 82.30% |
| ROC-AUC | **0.8605** |
| Val AUC (best epoch) | **0.9994** |

### 5-Class Attack Category

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| normal | 68% | **97%** | 80% |
| DoS | **93%** | 81% | **87%** |
| Probe | 81% | 59% | 69% |
| R2L | 80% | 10% | 17% |
| U2R | 53% | 28% | 37% |
| **Macro avg** | **75%** | **55%** | **58%** |

> **Note on the test set gap**: The NSL-KDD `KDDTest.txt` contains **17 attack types never seen during training** (apache2, httptunnel, sqlattack, etc.), mostly in R2L and U2R categories. This is a known property of the benchmark — it intentionally tests out-of-distribution generalisation. The validation AUC of 0.9994 reflects in-distribution performance on the training distribution.

## Project Structure

```
.
├── train.py              # Main training entry point  (python train.py --help)
├── requirements.txt
├── src/
│   └── ids/
│       ├── __init__.py
│       └── deep_model.py # Model architecture, FocalLoss, class weight helpers
├── data/
│   └── raw/              # Place KDDTrain.txt and KDDTest.txt here (gitignored)
└── artifacts/            # Generated models & plots (gitignored)
    ├── model_resnet_ids.keras
    ├── model_resnet_ids_best.keras
    ├── scaler.pkl
    ├── feature_columns.pkl
    ├── training_history.pkl
    └── plots/
        ├── training_curves.png
        ├── loss_curves.png
        ├── binary_cm.png
        ├── category_cm.png
        ├── per_class_f1.png
        └── roc_curve.png
```

## Setup

**Python 3.10+ recommended**

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## Data

Download the NSL-KDD dataset and place the files as:
- `data/raw/KDDTrain.txt`
- `data/raw/KDDTest.txt`

**Direct download:**
```bash
curl -L "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTrain%2B.txt" \
     -o data/raw/KDDTrain.txt
curl -L "https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/master/KDDTest%2B.txt" \
     -o data/raw/KDDTest.txt
```

## Training

```bash
# Default: 60 epochs, batch size 512
python train.py

# Custom
python train.py --epochs 100 --batch 256

# Skip plot generation (faster)
python train.py --no-plots
```

Training takes **~20 minutes on CPU**, ~3 minutes on a GPU.  
The best model checkpoint is saved automatically via `ModelCheckpoint`.

## Artifacts

After training, `artifacts/` will contain:

| File | Description |
|------|-------------|
| `model_resnet_ids.keras` | Final model weights |
| `model_resnet_ids_best.keras` | Best checkpoint (by val AUC) |
| `scaler.pkl` | Fitted `RobustScaler` for inference |
| `feature_columns.pkl` | Column names for inference alignment |
| `training_history.pkl` | Epoch-by-epoch metrics dict |
| `plots/` | Training curves, confusion matrices, ROC, F1 bar chart |

## Inference

```python
import numpy as np
import joblib
import tensorflow as tf

model   = tf.keras.models.load_model("artifacts/model_resnet_ids.keras")
scaler  = joblib.load("artifacts/scaler.pkl")
columns = joblib.load("artifacts/feature_columns.pkl")

# Prepare a sample (preprocess the same way as training)
# X: (n_samples, n_features) float32 array
bin_prob, cat_prob = model.predict(X)
is_attack   = (bin_prob.squeeze() >= 0.5).astype(int)
attack_type = np.argmax(cat_prob, axis=1)  # 0=normal,1=DoS,2=Probe,3=R2L,4=U2R
```

## Dependencies

| Package | Version |
|---------|---------|
| TensorFlow | ≥ 2.14 |
| scikit-learn | ≥ 1.3 |
| pandas | ≥ 2.0 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.7 |
| seaborn | ≥ 0.12 |
| joblib | ≥ 1.3 |
