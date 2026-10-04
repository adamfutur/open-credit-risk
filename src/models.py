"""
M5/M6 — modelisation.

Trains and compares:
  1) Logistic Regression (traditional scorecard-style baseline)
  2) XGBoost (advanced)
  3) LightGBM (advanced)

Each is run twice — once with SMOTE-resampled training data, once with plain
class-weighting on the original imbalance — so imbalance.py's two strategies
are compared honestly rather than picking one upfront.

Tracked in MLflow (params, metrics, model artifact) per the brief's MLOps
requirement.

Usage:
    python src/models.py
"""
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from evaluate import evaluate, results_table
from imbalance import apply_smote, scale_pos_weight

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
MLFLOW_DIR = ROOT / "mlflow"

FEATURE_COLS = [
    "dti",
    "credit_utilization_ratio",
    "payment_history_score",
    "severe_delinquencies_90d",
    "monthly_income",
    "age",
    "open_credit_lines",
    "real_estate_lines",
    "dependents",
]
TARGET_COL = "defaulted"


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


def make_logreg_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def run() -> pd.DataFrame:
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("credit-risk-default-prediction")

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Impute here too (outside the sklearn Pipeline) so XGBoost/LightGBM and
    # the SMOTE step see the same imputed matrix as Logistic Regression.
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=FEATURE_COLS, index=X_test.index
    )

    X_train_smote, y_train_smote = apply_smote(X_train_imp, y_train)
    spw = scale_pos_weight(y_train)

    results = []

    # --- 1) Logistic Regression baseline, class-weighted (no SMOTE) ---
    with mlflow.start_run(run_name="logreg_class_weighted"):
        pipe = make_logreg_pipeline()
        pipe.set_params(clf__class_weight="balanced")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        res = evaluate(y_test, proba, "LogReg (class-weighted)")
        mlflow.log_params({"model": "logreg", "balancing": "class_weight"})
        mlflow.log_metrics({k: v for k, v in res.items() if k != "model"})
        mlflow.sklearn.log_model(pipe, "model", serialization_format="pickle")
        results.append(res)

    # --- 2) Logistic Regression baseline, SMOTE ---
    with mlflow.start_run(run_name="logreg_smote"):
        pipe = make_logreg_pipeline()
        pipe.fit(X_train_smote, y_train_smote)
        proba = pipe.predict_proba(X_test_imp)[:, 1]
        res = evaluate(y_test, proba, "LogReg (SMOTE)")
        mlflow.log_params({"model": "logreg", "balancing": "smote"})
        mlflow.log_metrics({k: v for k, v in res.items() if k != "model"})
        mlflow.sklearn.log_model(pipe, "model", serialization_format="pickle")
        results.append(res)

    # --- 3) XGBoost, scale_pos_weight (no SMOTE) ---
    with mlflow.start_run(run_name="xgboost_weighted"):
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=spw,
            eval_metric="auc",
            random_state=42,
        )
        clf.fit(X_train_imp, y_train)
        proba = clf.predict_proba(X_test_imp)[:, 1]
        res = evaluate(y_test, proba, "XGBoost (weighted)")
        mlflow.log_params(
            {"model": "xgboost", "balancing": "scale_pos_weight", "n_estimators": 300}
        )
        mlflow.log_metrics({k: v for k, v in res.items() if k != "model"})
        mlflow.xgboost.log_model(clf, "model")
        results.append(res)

    # --- 4) XGBoost, SMOTE ---
    with mlflow.start_run(run_name="xgboost_smote"):
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            eval_metric="auc",
            random_state=42,
        )
        clf.fit(X_train_smote, y_train_smote)
        proba = clf.predict_proba(X_test_imp)[:, 1]
        res = evaluate(y_test, proba, "XGBoost (SMOTE)")
        mlflow.log_params(
            {"model": "xgboost", "balancing": "smote", "n_estimators": 300}
        )
        mlflow.log_metrics({k: v for k, v in res.items() if k != "model"})
        mlflow.xgboost.log_model(clf, "model")
        results.append(res)

    # --- 5) LightGBM, class-weighted (no SMOTE) ---
    with mlflow.start_run(run_name="lightgbm_weighted"):
        clf = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train_imp, y_train)
        proba = clf.predict_proba(X_test_imp)[:, 1]
        res = evaluate(y_test, proba, "LightGBM (weighted)")
        mlflow.log_params(
            {"model": "lightgbm", "balancing": "class_weight", "n_estimators": 300}
        )
        mlflow.log_metrics({k: v for k, v in res.items() if k != "model"})
        mlflow.sklearn.log_model(clf, "model", serialization_format="pickle")
        results.append(res)

    table = results_table(results)
    print("\n=== Model comparison ===")
    print(table)
    table.to_csv(ROOT / "outputs" / "model_comparison.csv")

    return table


if __name__ == "__main__":
    run()
