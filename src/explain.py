"""
M7 — explainability: SHAP (primary) + LIME (secondary, per the brief).

Produces:
- A global SHAP summary plot (which features drive risk overall).
- Per-decision SHAP waterfall plots for a handful of accept/reject cases, with
  a human-readable sentence like the brief's example
  ("Le DTI superieur a 45% a contribue a +35% dans la decision de refus").
- A LIME explanation for the same handful of cases, as a second, independent
  explanation method — useful to show in the README that SHAP and LIME broadly
  agree on the top drivers (or to discuss honestly if they don't).
"""
from pathlib import Path

import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "shap_plots"

FEATURE_LABELS = {
    "dti": "Debt-to-Income ratio",
    "credit_utilization_ratio": "Credit utilization ratio",
    "payment_history_score": "Payment history (late incidents)",
    "severe_delinquencies_90d": "90+ day delinquencies",
    "monthly_income": "Monthly income",
    "age": "Age",
    "open_credit_lines": "Open credit lines",
    "real_estate_lines": "Real estate loans/lines",
    "dependents": "Number of dependents",
}


def explain_global(model, X_background: pd.DataFrame, out_dir: Path = OUT_DIR):
    """SHAP summary plot across a background sample — the 'what drives risk
    overall' view for the README / presentation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)

    plt.figure()
    shap.summary_plot(shap_values, X_background, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary_global.png", dpi=150)
    plt.close()

    return explainer, shap_values


def explain_instance(explainer, X_row: pd.DataFrame, out_dir: Path = OUT_DIR, name: str = "instance"):
    """SHAP waterfall for a single client, plus the sentence format from the
    brief's example."""
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_values = explainer(X_row)

    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_waterfall_{name}.png", dpi=150)
    plt.close()

    contributions = list(zip(X_row.columns, shap_values.values[0]))
    contributions.sort(key=lambda c: abs(c[1]), reverse=True)
    top = contributions[0]
    label = FEATURE_LABELS.get(top[0], top[0])
    direction = "increased" if top[1] > 0 else "decreased"
    sentence = (
        f"{label} = {X_row.iloc[0][top[0]]:.2f} {direction} the predicted risk "
        f"by {abs(top[1]):.3f} (SHAP value), the largest single contribution "
        f"for this application."
    )
    return sentence, contributions


def make_lime_explainer(X_train: pd.DataFrame, class_names=("no default", "default")):
    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        class_names=list(class_names),
        mode="classification",
    )


def explain_instance_lime(lime_explainer, predict_fn, X_row: pd.DataFrame, out_dir: Path = OUT_DIR, name: str = "instance"):
    out_dir.mkdir(parents=True, exist_ok=True)
    exp = lime_explainer.explain_instance(
        X_row.iloc[0].values, predict_fn, num_features=len(X_row.columns)
    )
    exp.save_to_file(str(out_dir / f"lime_{name}.html"))
    return exp.as_list()


def main() -> None:
    import joblib

    FEATURE_COLS = list(FEATURE_LABELS.keys())
    MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
    FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"

    model = joblib.load(MODEL_PATH)
    df = pd.read_parquet(FEATURES_PATH)
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    proba = model.predict_proba(X)[:, 1]
    df = df.assign(proba=proba)

    # Background sample for the global SHAP summary + as LIME's training reference.
    background = X.sample(n=min(500, len(X)), random_state=42)

    print("=== Global SHAP summary ===")
    explainer, _ = explain_global(model, background)
    print(f"Saved {OUT_DIR / 'shap_summary_global.png'}")

    # Pick one clear reject (highest predicted risk) and one clear accept
    # (lowest predicted risk) as the worked examples for the README.
    reject_idx = df["proba"].idxmax()
    accept_idx = df["proba"].idxmin()

    lime_explainer = make_lime_explainer(background)

    print("\n=== Instance explanations ===")
    for name, idx in (("reject_example", reject_idx), ("accept_example", accept_idx)):
        X_row = X.loc[[idx]]
        p = df.loc[idx, "proba"]
        print(f"\n--- {name} (client_id={df.loc[idx, 'client_id']}, "
              f"probability_of_default={p:.4f}) ---")

        sentence, _ = explain_instance(explainer, X_row, name=name)
        print("SHAP:", sentence)

        lime_list = explain_instance_lime(
            lime_explainer, model.predict_proba, X_row, name=name
        )
        top_lime = lime_list[0]
        print(f"LIME: top factor '{top_lime[0]}' with weight {top_lime[1]:.4f}")

    print(f"\nAll plots and LIME HTML reports saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
