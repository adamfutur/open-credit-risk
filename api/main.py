"""
M9 — FastAPI service.

POST /score with a client's financial profile -> risk score + top SHAP
contributing features, in the "why" format described in the brief.

Loads the best model from outputs/ (saved by src/models.py after comparing
the runs in MLflow — see README for how the winning run is selected).

Run locally:
    uvicorn api.main:app --reload --app-dir .

Run via Docker: see api/Dockerfile.
"""
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

app = FastAPI(
    title="Credit Risk Assessment API",
    description="Predicts probability of default and explains the decision with SHAP.",
    version="0.1.0",
)

_model = None
_explainer = None


class ClientProfile(BaseModel):
    dti: float = Field(..., description="Debt-to-income ratio")
    credit_utilization_ratio: float = Field(..., ge=0, le=1)
    payment_history_score: float = Field(..., ge=0, description="Late-payment incidents, capped")
    severe_delinquencies_90d: float = Field(..., ge=0)
    monthly_income: float = Field(..., ge=0)
    age: int = Field(..., ge=18)
    open_credit_lines: int = Field(..., ge=0)
    real_estate_lines: int = Field(..., ge=0)
    dependents: int = Field(..., ge=0)


class ScoreResponse(BaseModel):
    probability_of_default: float
    decision: str
    top_reasons: list[dict]


def _build_explainer(model):
    """XGBoost/LightGBM are tree models -> TreeExplainer (fast, exact).
    Logistic Regression is wrapped in a sklearn Pipeline (impute+scale+clf),
    which TreeExplainer can't handle, so fall back to the generic
    shap.Explainer with a small background sample instead."""
    is_tree_model = not isinstance(model, Pipeline)
    if is_tree_model:
        return shap.TreeExplainer(model)

    if not FEATURES_PATH.exists():
        raise RuntimeError(
            f"Winning model is a Logistic Regression pipeline, which needs a "
            f"background sample for SHAP but {FEATURES_PATH} doesn't exist. "
            "Run src/build_features.py first."
        )
    all_features = pd.read_parquet(FEATURES_PATH)[FEATURE_COLS]
    background = all_features.sample(n=min(100, len(all_features)), random_state=42)
    return shap.Explainer(model.predict_proba, background)


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


def _load_model():
    global _model, _explainer
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"No model found at {MODEL_PATH}. Run src/models.py, then "
                "src/select_best_model.py to save the winning run here."
            )
        _model = joblib.load(MODEL_PATH)
        _explainer = _build_explainer(_model)
    return _model, _explainer


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
def score(profile: ClientProfile):
    model, explainer = _load_model()

    X = pd.DataFrame([profile.model_dump()])
    proba = float(model.predict_proba(X)[0, 1])

    shap_values = explainer(X)
    contributions = list(zip(X.columns, shap_values.values[0]))
    contributions.sort(key=lambda c: abs(c[1]), reverse=True)

    top_reasons = [
        {
            "feature": name,
            "value": float(X.iloc[0][name]),
            "shap_contribution": float(value),
            "direction": "increases risk" if value > 0 else "decreases risk",
        }
        for name, value in contributions[:5]
    ]

    decision = "reject" if proba >= 0.5 else "accept"

    return ScoreResponse(
        probability_of_default=round(proba, 4),
        decision=decision,
        top_reasons=top_reasons,
    )
