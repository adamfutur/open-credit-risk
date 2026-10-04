"""
M4 — imbalance handling.

Real default rate here is ~6.7% (matches the brief's "real-world 2-5%" ballpark
closely enough to be a legitimate example). Provides both SMOTE oversampling and
class-weighting so models.py can compare which one the evaluation metrics prefer,
per the brief ("garder celle que l'évaluation préfère").
"""
from imblearn.over_sampling import SMOTE
import pandas as pd


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Oversample the minority (default) class with SMOTE. Fit on TRAIN ONLY —
    never call this on data that includes the test/holdout split, or the
    evaluation will be optimistically biased by synthetic neighbors leaking
    across the split."""
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


def class_weights(y: pd.Series) -> dict:
    """Inverse-frequency class weights, for models that accept class_weight /
    scale_pos_weight instead of resampling the data."""
    counts = y.value_counts()
    total = len(y)
    return {cls: total / (len(counts) * count) for cls, count in counts.items()}


def scale_pos_weight(y: pd.Series) -> float:
    """XGBoost/LightGBM-style single scalar: negative/positive ratio."""
    counts = y.value_counts()
    return counts[0] / counts[1]
