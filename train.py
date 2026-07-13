"""
train.py  —  Intrusion Detection System (NSL-KDD)
==================================================
End-to-end training pipeline for the Residual MLP IDS model.

Usage
-----
    python train.py [--epochs 60] [--batch 512] [--no-plots]

Steps
-----
  1. Load & preprocess NSL-KDD data
  2. Build multi-task labels  (binary + 5-class attack category)
  3. Compute class weights & focal loss for severe imbalance
  4. Build ResidualIDS model  (6 residual blocks, 1.75M params)
  5. Train with cosine-decay LR + warm-up, EarlyStopping, ModelCheckpoint
  6. Evaluate: accuracy, F1, ROC-AUC, per-class report
  7. Save model, scaler, column list, and all plots to artifacts/
"""

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, accuracy_score,
    roc_curve, ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ids.deep_model import (
    build_ids_model, FocalLoss,
    compute_binary_class_weights, compute_category_class_weights,
    get_attack_category, CATEGORY_NAMES,
)

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train Residual MLP IDS on NSL-KDD")
parser.add_argument("--epochs",    type=int,  default=60,    help="Max epochs")
parser.add_argument("--batch",     type=int,  default=512,   help="Batch size")
parser.add_argument("--no-plots",  action="store_true",      help="Skip plot generation")
args = parser.parse_args()

REPO          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(REPO, "data", "raw")
ARTIFACTS_DIR = os.path.join(REPO, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

COLUMNS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins","logged_in",
    "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
    "num_shells","num_access_files","num_outbound_cmds","is_host_login",
    "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
    "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","outcome","level",
]

# Raw numeric columns — scaled before one-hot encoding
RAW_NUM_COLS = [
    "duration","src_bytes","dst_bytes","wrong_fragment","urgent","hot",
    "num_failed_logins","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate",
    "dst_host_count","dst_host_srv_count","dst_host_same_srv_rate",
    "dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate",
    "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
]


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def preprocess(df: pd.DataFrame, scaler=None, fit: bool = True):
    """
    Preprocess NSL-KDD dataframe.

    Returns
    -------
    df_out   : preprocessed feature DataFrame
    y_bin    : binary labels  (0=normal, 1=attack)
    y_cat    : category labels (0=normal,1=DoS,2=Probe,3=R2L,4=U2R)
    scaler   : fitted RobustScaler (returned only when fit=True)
    """
    df = df.copy()
    y_bin = (df["outcome"] != "normal").astype(int).values
    y_cat = df["outcome"].apply(get_attack_category).values
    df.drop(columns=["outcome", "level"], inplace=True)

    # Scale numerics BEFORE one-hot encoding so scaler only sees stable columns
    if fit:
        scaler = RobustScaler()
        df[RAW_NUM_COLS] = scaler.fit_transform(df[RAW_NUM_COLS])
    else:
        df[RAW_NUM_COLS] = scaler.transform(df[RAW_NUM_COLS])

    df = pd.get_dummies(df, columns=["protocol_type", "service", "flag"])
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    df[bool_cols] = df[bool_cols].astype(int)
    return df, y_bin, y_cat, scaler


def make_dataset(X, y_bin, y_cat_oh, sw, batch_size, shuffle=False):
    """Wrap arrays into a tf.data.Dataset with sample weights."""
    ds = tf.data.Dataset.from_tensor_slices((
        X,
        {"binary": y_bin.astype(np.float32),
         "category": y_cat_oh.astype(np.float32)},
        sw.astype(np.float32),
    ))
    if shuffle:
        ds = ds.shuffle(buffer_size=10_000, seed=42)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ═══════════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ═══════════════════════════════════════════════════════════════════════════
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up followed by cosine annealing."""

    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=1e-6):
        super().__init__()
        self.peak_lr      = peak_lr
        self.warmup_steps = float(warmup_steps)
        self.total_steps  = float(total_steps)
        self.min_lr       = min_lr

    def __call__(self, step):
        step      = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * (step / self.warmup_steps)
        cosine_lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
            1 + tf.cos(np.pi * (step - self.warmup_steps) /
                       (self.total_steps - self.warmup_steps))
        )
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return dict(peak_lr=self.peak_lr, warmup_steps=self.warmup_steps,
                    total_steps=self.total_steps, min_lr=self.min_lr)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  Residual MLP IDS  —  NSL-KDD  (Multi-task)")
    print("=" * 65)

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n[1/7] Loading datasets …")
    train_path = os.path.join(DATA_DIR, "KDDTrain.txt")
    test_path  = os.path.join(DATA_DIR, "KDDTest.txt")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Place KDDTrain.txt and KDDTest.txt in data/raw/ before training.\n"
            "Download from: https://github.com/HoaNP/NSL-KDD-DataSet"
        )
    df_train = pd.read_csv(train_path, header=None, names=COLUMNS)
    df_test  = pd.read_csv(test_path,  header=None, names=COLUMNS)
    print(f"  Train: {df_train.shape}   Test: {df_test.shape}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    print("\n[2/7] Preprocessing …")
    df_train_p, y_bin_train, y_cat_train, scaler = preprocess(df_train, fit=True)
    df_test_p,  y_bin_test,  y_cat_test,  _      = preprocess(df_test,  scaler=scaler, fit=False)

    # Align test columns with train (handle unseen service/flag dummies)
    for c in set(df_train_p.columns) - set(df_test_p.columns):
        df_test_p[c] = 0
    df_test_p = df_test_p[df_train_p.columns]

    X_train = df_train_p.values.astype(np.float32)
    X_test  = df_test_p.values.astype(np.float32)
    INPUT_DIM = X_train.shape[1]
    print(f"  Feature dim : {INPUT_DIM}")
    print(f"  Binary  — train: {np.bincount(y_bin_train)}   test: {np.bincount(y_bin_test)}")
    print(f"  Category— train: {np.bincount(y_cat_train)}   test: {np.bincount(y_cat_test)}")

    joblib.dump(scaler,              os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    joblib.dump(df_train_p.columns,  os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))

    # Train/val split (stratified on category for rare class coverage)
    X_tr, X_val, y_bin_tr, y_bin_val, y_cat_tr, y_cat_val = train_test_split(
        X_train, y_bin_train, y_cat_train,
        test_size=0.15, random_state=42, stratify=y_cat_train,
    )
    print(f"  Train={X_tr.shape[0]}  Val={X_val.shape[0]}  Test={X_test.shape[0]}")

    # ── 3. Class weights ──────────────────────────────────────────────────
    print("\n[3/7] Computing class weights …")
    cat_weights = compute_category_class_weights(y_cat_tr, n_classes=5)
    print(f"  Category weights: { {k: f'{v:.2f}' for k,v in cat_weights.items()} }")

    sw_tr  = np.array([cat_weights[c] for c in y_cat_tr],  dtype=np.float32)
    sw_val = np.array([cat_weights[c] for c in y_cat_val], dtype=np.float32)

    y_cat_tr_oh  = tf.keras.utils.to_categorical(y_cat_tr,  num_classes=5)
    y_cat_val_oh = tf.keras.utils.to_categorical(y_cat_val, num_classes=5)
    y_cat_test_oh= tf.keras.utils.to_categorical(y_cat_test, num_classes=5)

    train_ds = make_dataset(X_tr,  y_bin_tr,  y_cat_tr_oh,  sw_tr,  args.batch, shuffle=True)
    val_ds   = make_dataset(X_val, y_bin_val, y_cat_val_oh, sw_val, args.batch)

    # ── 4. Build model ────────────────────────────────────────────────────
    print("\n[4/7] Building Residual MLP …")
    model = build_ids_model(
        input_dim      = INPUT_DIM,
        n_classes      = 5,
        block_units    = [256, 256, 512, 512, 256, 128],
        shared_dropout = 0.3,
        l2             = 1e-4,
    )
    model.summary()

    STEPS = len(X_tr) // args.batch
    lr_schedule = WarmupCosineDecay(
        peak_lr      = 3e-4,
        warmup_steps = 5 * STEPS,
        total_steps  = args.epochs * STEPS,
    )
    model.compile(
        optimizer = keras.optimizers.Adam(lr_schedule, clipnorm=1.0),
        loss = {
            "binary"  : keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            "category": FocalLoss(gamma=2.0, alpha=0.25),
        },
        loss_weights = {"binary": 1.0, "category": 0.6},
        metrics = {
            "binary"  : [keras.metrics.BinaryAccuracy(name="acc"),
                         keras.metrics.AUC(name="auc")],
            "category": [keras.metrics.CategoricalAccuracy(name="acc")],
        },
    )

    # ── 5. Train ──────────────────────────────────────────────────────────
    print(f"\n[5/7] Training  ({args.epochs} epochs, batch={args.batch}) …")
    ckpt_path = os.path.join(ARTIFACTS_DIR, "model_resnet_ids_best.keras")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_binary_auc",
            mode="max", save_best_only=True, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_binary_auc", mode="max",
            patience=12, restore_best_weights=True, verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data = val_ds,
        epochs          = args.epochs,
        callbacks       = callbacks,
        verbose         = 1,
    )

    model.save(os.path.join(ARTIFACTS_DIR, "model_resnet_ids.keras"))
    joblib.dump(history.history, os.path.join(ARTIFACTS_DIR, "training_history.pkl"))
    print("  ✅ Model saved: model_resnet_ids.keras")

    # ── 6. Evaluate ───────────────────────────────────────────────────────
    print("\n[6/7] Evaluating on held-out test set …")
    bin_prob, cat_prob = model.predict(X_test, batch_size=1024, verbose=0)
    bin_pred = (bin_prob.squeeze() >= 0.5).astype(int)
    cat_pred = np.argmax(cat_prob, axis=1)

    bin_acc = accuracy_score(y_bin_test, bin_pred)
    bin_f1  = f1_score(y_bin_test, bin_pred, average="binary")
    bin_auc = roc_auc_score(y_bin_test, bin_prob.squeeze())
    cat_acc = accuracy_score(y_cat_test, cat_pred)
    cat_f1  = f1_score(y_cat_test, cat_pred, average="macro")

    print(f"\n  ── Binary (Normal vs Attack) ──")
    print(f"  Accuracy : {bin_acc*100:.3f}%")
    print(f"  F1 Score : {bin_f1*100:.3f}%")
    print(f"  ROC-AUC  : {bin_auc:.5f}")
    print(f"\n  ── 5-Class Category ──")
    print(f"  Accuracy (macro): {cat_acc*100:.3f}%")
    print(f"  F1 (macro)      : {cat_f1*100:.3f}%")
    print(f"\n  Per-class report:")
    print(classification_report(y_cat_test, cat_pred,
                                target_names=CATEGORY_NAMES, zero_division=0))

    # ── 7. Plots ──────────────────────────────────────────────────────────
    if not args.no_plots:
        print("[7/7] Generating plots …")
        H = history.history
        plots_dir = os.path.join(ARTIFACTS_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Training curves
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Residual MLP IDS — Training History", fontsize=14, fontweight="bold")
        axes[0].plot(H["binary_acc"],       label="Train"); axes[0].plot(H["val_binary_acc"],     label="Val", ls="--"); axes[0].set_title("Binary Accuracy");    axes[0].legend(); axes[0].grid(True)
        axes[1].plot(H["binary_auc"],       label="Train"); axes[1].plot(H["val_binary_auc"],     label="Val", ls="--"); axes[1].set_title("Binary ROC-AUC");      axes[1].legend(); axes[1].grid(True)
        axes[2].plot(H["category_acc"],     label="Train"); axes[2].plot(H["val_category_acc"],   label="Val", ls="--"); axes[2].set_title("5-Class Accuracy");    axes[2].legend(); axes[2].grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=120); plt.close()

        # Loss curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Residual MLP IDS — Loss Curves", fontsize=14, fontweight="bold")
        axes[0].plot(H["binary_loss"],      label="Train"); axes[0].plot(H["val_binary_loss"],    label="Val", ls="--"); axes[0].set_title("Binary Head Loss");    axes[0].legend(); axes[0].grid(True)
        axes[1].plot(H["category_loss"],    label="Train"); axes[1].plot(H["val_category_loss"],  label="Val", ls="--"); axes[1].set_title("Category Focal Loss"); axes[1].legend(); axes[1].grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "loss_curves.png"), dpi=120); plt.close()

        # Binary confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix(y_bin_test, bin_pred),
                               display_labels=["Normal", "Attack"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"Binary CM  (Acc={bin_acc*100:.2f}%, AUC={bin_auc:.4f})"); ax.grid(False)
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "binary_cm.png"), dpi=120); plt.close()

        # 5-class confusion matrix
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(confusion_matrix(y_cat_test, cat_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=CATEGORY_NAMES, yticklabels=CATEGORY_NAMES, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"5-Class CM  (F1-macro={cat_f1*100:.2f}%)")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "category_cm.png"), dpi=120); plt.close()

        # Per-class F1 bar chart
        per_class_f1 = f1_score(y_cat_test, cat_pred, average=None, labels=list(range(5)), zero_division=0)
        colors = ["#4CAF50","#2196F3","#FF9800","#E91E63","#9C27B0"]
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.bar(CATEGORY_NAMES, per_class_f1 * 100, color=colors, edgecolor="white")
        for bar, val in zip(bars, per_class_f1):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val*100:.1f}%", ha="center", va="bottom", fontweight="bold")
        ax.set_ylim(0, 115); ax.set_ylabel("F1 Score (%)"); ax.grid(axis="y", alpha=0.4)
        ax.set_title("Per-Class F1 Score — Residual MLP IDS")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "per_class_f1.png"), dpi=120); plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_bin_test, bin_prob.squeeze())
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, lw=2, color="#2196F3", label=f"ResidualIDS  (AUC={bin_auc:.5f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve — Binary IDS Classifier")
        ax.legend(loc="lower right"); ax.grid(True, alpha=0.4)
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=120); plt.close()

        print(f"  All plots saved to: {plots_dir}/")

    print("\n" + "=" * 65)
    print("  DONE!")
    print(f"  Binary  — Acc={bin_acc*100:.3f}%  F1={bin_f1*100:.3f}%  AUC={bin_auc:.5f}")
    print(f"  5-Class — Acc={cat_acc*100:.3f}%  F1-macro={cat_f1*100:.3f}%")
    print(f"  Artifacts: {ARTIFACTS_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
