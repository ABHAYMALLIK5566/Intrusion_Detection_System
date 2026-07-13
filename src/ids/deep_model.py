"""
src/ids/deep_model.py
=====================
Production-grade Residual MLP for Network Intrusion Detection.

Architecture
------------
- Input BatchNorm
- N × Residual Blocks  (Linear → BN → GELU → Linear → BN → skip-add → GELU)
- Shared representation dropout
- Binary head   : Normal vs Attack  (BCE + class weights)
- 5-Class head  : normal / DoS / Probe / R2L / U2R  (Focal Loss + class weights)

Focal Loss focuses training on hard / rare examples (critical for U2R/R2L detection).
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ---------------------------------------------------------------------------
# Attack category mapping
# ---------------------------------------------------------------------------
ATTACK_CATEGORIES = {
    "normal":           0,
    # DoS
    "back":             1, "land":       1, "neptune":   1, "pod":         1,
    "smurf":            1, "teardrop":   1, "apache2":   1, "udpstorm":    1,
    "processtable":     1, "worm":       1, "mailbomb":  1,
    # Probe
    "ipsweep":          2, "nmap":       2, "portsweep": 2, "satan":       2,
    "mscan":            2, "saint":      2,
    # R2L
    "ftp_write":        3, "guess_passwd": 3, "imap":     3, "multihop":   3,
    "phf":              3, "spy":          3, "warezclient": 3, "warezmaster": 3,
    "sendmail":         3, "named":        3, "snmpgetattack": 3, "snmpguess": 3,
    "xlock":            3, "xsnoop":       3, "httptunnel": 3,
    # U2R
    "buffer_overflow":  4, "loadmodule": 4, "perl":      4, "rootkit":     4,
    "sqlattack":        4, "xterm":      4, "ps":        4,
}

CATEGORY_NAMES = ["normal", "DoS", "Probe", "R2L", "U2R"]


def get_attack_category(label: str) -> int:
    """Map raw attack label → integer category (0–4)."""
    return ATTACK_CATEGORIES.get(label, 1)   # unknown → DoS bucket


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for multi-class classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Focuses training on hard, misclassified examples.
    gamma=2  is the standard choice from the original paper.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 name: str = "focal_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        weight = self.alpha * y_true * tf.pow(1.0 - y_pred, self.gamma)
        loss = weight * ce
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha})
        return cfg


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------
def residual_block(x, units: int, dropout_rate: float = 0.2,
                   l2: float = 1e-4) -> tf.Tensor:
    """
    ResNet-style residual block for tabular data.

    Layout:
        h = Dense(units) → BatchNorm → GELU
            → Dense(units) → BatchNorm
        out = GELU(h + skip)

    A 1×1 projection is added when the input width ≠ units.
    """
    reg = regularizers.L2(l2)

    # Main path
    h = layers.Dense(units, kernel_regularizer=reg, use_bias=False)(x)
    h = layers.BatchNormalization()(h)
    h = layers.Activation("gelu")(h)
    h = layers.Dropout(dropout_rate)(h)

    h = layers.Dense(units, kernel_regularizer=reg, use_bias=False)(h)
    h = layers.BatchNormalization()(h)

    # Skip / projection
    if x.shape[-1] != units:
        x = layers.Dense(units, kernel_regularizer=reg, use_bias=False)(x)
        x = layers.BatchNormalization()(x)

    out = layers.Add()([h, x])
    out = layers.Activation("gelu")(out)
    return out


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_ids_model(
    input_dim: int,
    n_classes: int = 5,
    block_units: list[int] = None,
    shared_dropout: float = 0.3,
    l2: float = 1e-4,
) -> keras.Model:
    """
    Build the multi-task Residual MLP IDS model.

    Parameters
    ----------
    input_dim     : number of input features after preprocessing
    n_classes     : number of attack categories (default 5)
    block_units   : list of hidden sizes for each residual block
    shared_dropout: dropout rate on the shared representation
    l2            : L2 weight decay

    Returns
    -------
    Compiled Keras model with two outputs:
        'binary'  – sigmoid output for Normal/Attack
        'category'– softmax output for 5-class attack type
    """
    if block_units is None:
        block_units = [256, 256, 512, 512, 256, 128]

    inp = keras.Input(shape=(input_dim,), name="features")

    # Input normalisation
    x = layers.BatchNormalization(name="input_bn")(inp)

    # Stack of residual blocks
    for i, units in enumerate(block_units):
        x = residual_block(x, units, dropout_rate=0.2, l2=l2)

    # Shared representation
    x = layers.Dropout(shared_dropout, name="shared_dropout")(x)

    # ── Binary head (Normal vs Attack) ──────────────────────────────────────
    b = layers.Dense(64, activation="gelu", name="binary_dense")(x)
    b = layers.BatchNormalization(name="binary_bn")(b)
    binary_out = layers.Dense(1, activation="sigmoid", name="binary")(b)

    # ── 5-Class head (attack category) ──────────────────────────────────────
    c = layers.Dense(128, activation="gelu", name="cat_dense")(x)
    c = layers.BatchNormalization(name="cat_bn")(c)
    category_out = layers.Dense(n_classes, activation="softmax", name="category")(c)

    model = keras.Model(inputs=inp, outputs=[binary_out, category_out],
                        name="ResidualIDS")
    return model


# ---------------------------------------------------------------------------
# Class weight helpers
# ---------------------------------------------------------------------------
def compute_binary_class_weights(y_binary: np.ndarray) -> dict:
    """Inverse-frequency class weights for binary labels."""
    n_total = len(y_binary)
    n_pos = y_binary.sum()
    n_neg = n_total - n_pos
    w_pos = n_total / (2 * n_pos)
    w_neg = n_total / (2 * n_neg)
    return {0: w_neg, 1: w_pos}


def compute_category_class_weights(y_category: np.ndarray,
                                   n_classes: int = 5) -> dict:
    """Inverse-frequency class weights for the 5-class head."""
    counts = np.bincount(y_category, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)          # avoid div-by-zero
    weights = len(y_category) / (n_classes * counts)
    return {i: float(weights[i]) for i in range(n_classes)}
