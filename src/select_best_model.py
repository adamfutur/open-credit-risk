"""
Picks the winning run from the MLflow experiment (by KS statistic, the metric
risk teams trust most — see evaluate.py docstring) and saves its model to
outputs/best_model.joblib, where api/main.py expects to find it.

Run this after src/models.py has logged all runs.

Usage:
    python src/select_best_model.py
    python src/select_best_model.py --metric auc_roc   # pick a different metric
"""
import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_DIR = ROOT / "mlflow"
OUT_PATH = ROOT / "outputs" / "best_model.joblib"
EXPERIMENT_NAME = "credit-risk-default-prediction"


def main(metric: str = "ks") -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"No MLflow experiment '{EXPERIMENT_NAME}' found under {MLFLOW_DIR}. "
            "Run src/models.py first."
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found in the experiment.")

    best_run = runs[0]
    run_id = best_run.info.run_id
    run_name = best_run.data.tags.get("mlflow.runName", run_id)
    print(f"Best run by {metric}: {run_name} ({run_id})")
    for m in ("auc_roc", "gini", "ks"):
        val = best_run.data.metrics.get(m)
        if val is not None:
            print(f"  {m} = {val:.4f}")

    model_uri = f"runs:/{run_id}/model"
    model_flavor = best_run.data.params.get("model", "")
    # XGBoost runs were logged with mlflow.xgboost.log_model; everything else
    # (Logistic Regression, LightGBM) with mlflow.sklearn.log_model — see
    # models.py. Loading with the wrong flavor silently returns a pyfunc
    # wrapper whose .predict_proba the API can't call, so pick explicitly.
    if model_flavor == "xgboost":
        model = mlflow.xgboost.load_model(model_uri)
    else:
        model = mlflow.sklearn.load_model(model_uri)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_PATH)
    print(f"Saved winning model to {OUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="ks",
        choices=["ks", "auc_roc", "gini"],
        help="Metric to rank runs by (default: ks).",
    )
    args = parser.parse_args()
    main(args.metric)
