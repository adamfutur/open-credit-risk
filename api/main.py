"""
FastAPI service for credit-risk scoring and local SHAP explanations.

Run locally:
    uvicorn api.main:app --reload

The model artifact must exist at outputs/best_model.joblib. This API is a
portfolio demonstration, not a production lending decision system.
"""
import os
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

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

try:
    DEFAULT_THRESHOLD = float(os.getenv("CREDIT_RISK_THRESHOLD", "0.5"))
except ValueError as exc:
    raise RuntimeError("CREDIT_RISK_THRESHOLD must be a number between 0 and 1.") from exc

if not 0 < DEFAULT_THRESHOLD < 1:
    raise RuntimeError("CREDIT_RISK_THRESHOLD must be greater than 0 and less than 1.")

app = FastAPI(
    title="Credit Risk Assessment API",
    description="Predicts probability of default and explains predictions with SHAP.",
    version="0.2.0",
)

_model = None
_explainer = None


class ClientProfile(BaseModel):
    """Input features in the same schema used by the training pipeline."""

    model_config = ConfigDict(extra="forbid")

    dti: float = Field(..., ge=0, le=5, allow_inf_nan=False, description="Debt-to-income ratio, clipped to 5")
    credit_utilization_ratio: float = Field(..., ge=0, le=1, allow_inf_nan=False)
    payment_history_score: float = Field(..., ge=0, le=10, allow_inf_nan=False)
    severe_delinquencies_90d: float = Field(..., ge=0, allow_inf_nan=False)
    monthly_income: float = Field(..., ge=0, allow_inf_nan=False)
    age: int = Field(..., ge=18, le=120)
    open_credit_lines: int = Field(..., ge=0)
    real_estate_lines: int = Field(..., ge=0)
    dependents: int = Field(..., ge=0)


class RiskReason(BaseModel):
    feature: str
    value: float
    shap_contribution: float
    direction: Literal["increases risk", "decreases risk", "neutral"]


class ScoreResponse(BaseModel):
    probability_of_default: float = Field(..., ge=0, le=1)
    decision: Literal["accept", "reject"]
    decision_threshold: float = Field(..., gt=0, lt=1)
    top_reasons: list[RiskReason]


def _build_explainer(model):
    """Build an appropriate explainer for tree models or sklearn pipelines."""
    if not isinstance(model, Pipeline):
        return shap.TreeExplainer(model)

    if not FEATURES_PATH.exists():
        raise RuntimeError(
            "The selected model is a sklearn Pipeline and needs a feature "
            f"background sample, but {FEATURES_PATH} does not exist. "
            "Run src/build_features.py first."
        )

    all_features = pd.read_parquet(FEATURES_PATH)[FEATURE_COLS]
    if all_features.empty:
        raise RuntimeError(f"No feature rows available in {FEATURES_PATH}.")
    background = all_features.sample(n=min(100, len(all_features)), random_state=42)
    return shap.Explainer(model.predict_proba, background)


def _load_model():
    """Load the model and explainer once per process."""
    global _model, _explainer
    if _model is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError(
                f"No model found at {MODEL_PATH}. Run src/models.py and "
                "src/select_best_model.py to create the winning model artifact."
            )
        try:
            model = joblib.load(MODEL_PATH)
            explainer = _build_explainer(model)
        except Exception as exc:
            raise RuntimeError(f"Could not load model or initialize SHAP: {exc}") from exc
        _model, _explainer = model, explainer
    return _model, _explainer


def _positive_class_contributions(shap_output, feature_count: int) -> np.ndarray:
    """Normalize SHAP output shapes across SHAP versions and classifier types."""
    values = np.asarray(shap_output.values)

    # Current SHAP versions may return (rows, features, classes).
    if values.ndim == 3:
        if values.shape[0] != 1 or values.shape[1] != feature_count:
            raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}.")
        class_index = 1 if values.shape[2] > 1 else 0
        values = values[0, :, class_index]
    elif values.ndim == 2:
        if values.shape != (1, feature_count):
            raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}.")
        values = values[0]
    else:
        raise RuntimeError(f"Unexpected SHAP output shape: {values.shape}.")

    if not np.isfinite(values).all():
        raise RuntimeError("SHAP returned non-finite feature contributions.")
    return values.astype(float)


@app.get("/health")
def health():
    """Liveness probe: confirms the API process is running."""
    return {"status": "ok"}


@app.get("/ready")
def readiness():
    """Readiness probe: confirms the model and explainer can be loaded."""
    try:
        _load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "model_loaded": True}


@app.post("/score", response_model=ScoreResponse)
def score(profile: ClientProfile):
    try:
        model, explainer = _load_model()
        X = pd.DataFrame([profile.model_dump()], columns=FEATURE_COLS)
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.shape != (1, 2):
            raise RuntimeError(
                f"Expected binary-class predict_proba output with shape (1, 2); "
                f"received {probabilities.shape}."
            )

        probability = float(probabilities[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("The model returned an invalid probability.")

        contributions = _positive_class_contributions(explainer(X), len(FEATURE_COLS))
        ranked = sorted(
            zip(FEATURE_COLS, X.iloc[0].to_numpy(dtype=float), contributions),
            key=lambda item: abs(item[2]),
            reverse=True,
        )

        reasons = [
            RiskReason(
                feature=name,
                value=float(value),
                shap_contribution=float(contribution),
                direction=(
                    "increases risk"
                    if contribution > 0
                    else "decreases risk"
                    if contribution < 0
                    else "neutral"
                ),
            )
            for name, value, contribution in ranked[:5]
        ]

        return ScoreResponse(
            probability_of_default=round(probability, 4),
            decision="reject" if probability >= DEFAULT_THRESHOLD else "accept",
            decision_threshold=DEFAULT_THRESHOLD,
            top_reasons=reasons,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
