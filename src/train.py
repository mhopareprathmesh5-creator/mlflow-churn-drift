"""Train churn models and track every run in MLflow.

Run:  python src/train.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (CATEGORICAL, FEATURES, FIGURES, NUMERIC, SEED, TARGET,
                    TRACKING_URI, load_data, make_splits)

EXPERIMENT = "Telco-Churn-Training"
MODEL_NAME = "telco-churn-classifier"

CANDIDATES = {
    "logistic_regression": (LogisticRegression, {"C": 1.0, "max_iter": 1000, "class_weight": "balanced"}),
    "random_forest": (RandomForestClassifier, {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 5,
                                               "class_weight": "balanced", "random_state": SEED}),
    "gradient_boosting": (GradientBoostingClassifier, {"n_estimators": 200, "learning_rate": 0.05,
                                                       "max_depth": 3, "random_state": SEED}),
}


def build_pipeline(estimator):
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("clf", estimator)])


def evaluate(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train, reference, _ = make_splits(df)
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xref, yref = reference[FEATURES], reference[TARGET]

    results = {}
    for name, (cls, params) in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"stage": "training", "domain": "telecom-customer-churn",
                             "data_window": "established customers (tenure > 12)"})
            mlflow.log_param("model_type", name)
            mlflow.log_params(params)
            mlflow.log_params({"train_rows": len(train), "reference_rows": len(reference),
                               "n_features": len(FEATURES)})

            model = build_pipeline(cls(**params)).fit(Xtr, ytr)
            metrics = evaluate(model, Xref, yref)
            mlflow.log_metrics(metrics)

            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            ConfusionMatrixDisplay.from_estimator(model, Xref, yref, ax=ax, cmap="Blues",
                                                  display_labels=["Stay", "Churn"])
            ax.set_title(f"{name} - reference window")
            fig.tight_layout()
            mlflow.log_figure(fig, f"confusion_matrix_{name}.png")
            plt.close(fig)

            mlflow.sklearn.log_model(model, name="model",
                                     signature=infer_signature(Xref.head(50), model.predict_proba(Xref.head(50))),
                                     input_example=Xref.head(3),
                                     # model files are produced by this script, so tree node storage is trusted
                                     skops_trusted_types=["sklearn.tree._tree.Tree"])
            results[name] = {"run_id": run.info.run_id, **metrics}
            print(f"{name:20s} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))

    best = max(results, key=lambda k: results[k]["roc_auc"])
    best_run = results[best]["run_id"]
    mv = mlflow.register_model(f"runs:/{best_run}/model", MODEL_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    client.set_model_version_tag(MODEL_NAME, mv.version, "selected_by", "highest ROC-AUC on reference window")

    summary = {"best_model": best, "registered_model": MODEL_NAME, "version": mv.version, "runs": results}
    (FIGURES.parent / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nBest model: {best} -> registered as {MODEL_NAME} v{mv.version} (alias 'champion')")


if __name__ == "__main__":
    main()
