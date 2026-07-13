"""
IDS package — Residual MLP for Network Intrusion Detection.
"""

from .deep_model import (
    build_ids_model,
    FocalLoss,
    compute_binary_class_weights,
    compute_category_class_weights,
    get_attack_category,
    CATEGORY_NAMES,
    ATTACK_CATEGORIES,
)

__all__ = [
    "build_ids_model",
    "FocalLoss",
    "compute_binary_class_weights",
    "compute_category_class_weights",
    "get_attack_category",
    "CATEGORY_NAMES",
    "ATTACK_CATEGORIES",
]
