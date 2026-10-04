"""
Banking-standard evaluation metrics: AUC-ROC, Gini, KS.
Used identically for the baseline and the advanced models so M5/M6 are a fair
head-to-head comparison.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def gini_coefficient(y_true, y_proba) -> float:
    """Gini = 2*AUC - 1, the standard credit-scoring transform of AUC-ROC."""
    auc = roc_auc_score(y_true, y_proba)
    return 2 * auc - 1


def ks_statistic(y_true, y_proba) -> float:
    """Kolmogorov-Smirnov: max separation between the cumulative distribution
    of scores for defaulters vs non-defaulters. Higher = better separation.
    This is the metric risk teams tend to trust most alongside Gini."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(np.abs(tpr - fpr)))


def evaluate(y_true, y_proba, model_name: str = "") -> dict:
    auc = roc_auc_score(y_true, y_proba)
    gini = gini_coefficient(y_true, y_proba)
    ks = ks_statistic(y_true, y_proba)

    result = {"model": model_name, "auc_roc": auc, "gini": gini, "ks": ks}
    print(
        f"[{model_name}] AUC-ROC={auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}"
    )
    return result


def results_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).set_index("model").round(4)
