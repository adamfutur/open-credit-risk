"""API contract and reliability tests. Run with: pytest tests/ -v"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import main


client = TestClient(main.app)

VALID_PROFILE = {
    "dti": 0.52,
    "credit_utilization_ratio": 0.8,
    "payment_history_score": 2,
    "severe_delinquencies_90d": 0,
    "monthly_income": 4500,
    "age": 34,
    "open_credit_lines": 6,
    "real_estate_lines": 1,
    "dependents": 1,
}


class FakeModel:
    def __init__(self, probability=0.18):
        self.probability = probability
        self.received = None

    def predict_proba(self, frame):
        self.received = frame.copy()
        return np.array([[1 - self.probability, self.probability]])


class FakeExplainer:
    def __init__(self, values):
        self.values = values
        self.received = None

    def __call__(self, frame):
        self.received = frame.copy()
        return SimpleNamespace(values=self.values)


@pytest.fixture(autouse=True)
def reset_model_cache(monkeypatch):
    monkeypatch.setattr(main, "_model", None)
    monkeypatch.setattr(main, "_explainer", None)


def test_health_is_liveness_probe():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("credit_utilization_ratio", 1.1),
        ("credit_utilization_ratio", -0.1),
        ("dti", 6),
        ("payment_history_score", 11),
        ("monthly_income", -1),
        ("age", 17),
        ("open_credit_lines", -1),
        ("dependents", -1),
        ("monthly_income", float("inf")),
    ],
)
def test_score_rejects_invalid_profile(field, value):
    payload = {**VALID_PROFILE, field: value}
    response = client.post("/score", json=payload)
    assert response.status_code == 422


def test_score_rejects_unknown_fields():
    response = client.post("/score", json={**VALID_PROFILE, "untrusted_extra": 1})
    assert response.status_code == 422


def test_score_returns_ranked_reasons_and_accept_decision(monkeypatch):
    model = FakeModel(probability=0.18)
    contributions = np.array([[0.12, -0.01, 0.03, 0.02, 0.01, -0.2, 0.0, 0.0, 0.0]])
    explainer = FakeExplainer(contributions)
    monkeypatch.setattr(main, "_load_model", lambda: (model, explainer))

    response = client.post("/score", json=VALID_PROFILE)

    assert response.status_code == 200
    body = response.json()
    assert body["probability_of_default"] == 0.18
    assert body["decision"] == "accept"
    assert body["decision_threshold"] == main.DEFAULT_THRESHOLD
    assert len(body["top_reasons"]) == 5
    assert body["top_reasons"][0]["feature"] == "age"
    assert body["top_reasons"][0]["direction"] == "decreases risk"
    assert list(model.received.columns) == main.FEATURE_COLS
    assert list(explainer.received.columns) == main.FEATURE_COLS


def test_score_rejects_when_probability_reaches_threshold(monkeypatch):
    model = FakeModel(probability=main.DEFAULT_THRESHOLD)
    explainer = FakeExplainer(np.zeros((1, len(main.FEATURE_COLS))))
    monkeypatch.setattr(main, "_load_model", lambda: (model, explainer))

    response = client.post("/score", json=VALID_PROFILE)

    assert response.status_code == 200
    assert response.json()["decision"] == "reject"


def test_score_supports_multiclass_axis_shap_shape(monkeypatch):
    model = FakeModel(probability=0.7)
    # Shape: rows, features, classes. Use class 1 contributions.
    values = np.zeros((1, len(main.FEATURE_COLS), 2))
    values[0, 2, 1] = 0.4
    explainer = FakeExplainer(values)
    monkeypatch.setattr(main, "_load_model", lambda: (model, explainer))

    response = client.post("/score", json=VALID_PROFILE)

    assert response.status_code == 200
    assert response.json()["decision"] == "reject"
    assert response.json()["top_reasons"][0]["feature"] == "payment_history_score"


def test_score_returns_service_unavailable_when_model_cannot_load(monkeypatch):
    def fail_load():
        raise RuntimeError("model artifact missing")

    monkeypatch.setattr(main, "_load_model", fail_load)
    response = client.post("/score", json=VALID_PROFILE)

    assert response.status_code == 503
    assert "model artifact missing" in response.json()["detail"]


def test_readiness_fails_when_model_cannot_load(monkeypatch):
    def fail_load():
        raise RuntimeError("model artifact missing")

    monkeypatch.setattr(main, "_load_model", fail_load)
    response = client.get("/ready")

    assert response.status_code == 503
