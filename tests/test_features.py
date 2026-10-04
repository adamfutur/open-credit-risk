"""
Tests for the feature-engineering and imbalance-handling logic. Run:
    pytest tests/
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import gini_coefficient, ks_statistic  # noqa: E402
from imbalance import apply_smote, class_weights, scale_pos_weight  # noqa: E402


def test_gini_perfect_separation():
    y_true = [0, 0, 0, 1, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
    assert gini_coefficient(y_true, y_proba) == pytest.approx(1.0, abs=1e-6)


def test_gini_random_is_near_zero():
    y_true = [0, 1, 0, 1, 0, 1, 0, 1]
    y_proba = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # Constant scores -> undefined ranking; roc_auc_score returns 0.5 -> gini 0
    assert gini_coefficient(y_true, y_proba) == pytest.approx(0.0, abs=1e-6)


def test_ks_statistic_perfect_separation():
    y_true = [0, 0, 0, 1, 1, 1]
    y_proba = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
    assert ks_statistic(y_true, y_proba) == pytest.approx(1.0, abs=1e-6)


def test_scale_pos_weight_matches_imbalance():
    y = pd.Series([0] * 90 + [1] * 10)
    assert scale_pos_weight(y) == pytest.approx(9.0)


def test_class_weights_inverse_frequency():
    y = pd.Series([0] * 80 + [1] * 20)
    weights = class_weights(y)
    # Minority class (1) should get a larger weight than the majority (0).
    assert weights[1] > weights[0]


def test_smote_balances_classes():
    X = pd.DataFrame(
        {
            "a": list(range(100)),
            "b": [x * 2 for x in range(100)],
        }
    )
    y = pd.Series([0] * 90 + [1] * 10)
    X_res, y_res = apply_smote(X, y)
    counts = y_res.value_counts()
    assert counts[0] == counts[1]
    assert len(X_res) == len(y_res)
