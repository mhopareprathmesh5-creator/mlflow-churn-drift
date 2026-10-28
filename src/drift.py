"""Data-drift and prediction-drift monitoring for the champion churn model,
with every scenario logged as an MLflow run.

Scenarios
  A  control           reference hold-out vs. a fresh hold-out from the same population (expect no drift)
  B  new_customer_cohort  established customers vs. customers with tenure <= 12 months (natural drift)
  C  price_revision    reference vs. same customers after a simulated 18% tariff hike and
                       payment-method shift (simulated drift; labels not yet available)

Run:  python src/drift.py      (after src/train.py)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, recall_score, roc_auc_score

from common import (CATEGORICAL, FEATURES, FIGURES, NUMERIC, REPORTS, SEED, TARGET,
                    TRACKING_URI, load_data, make_splits, simulate_price_revision)

EXPERIMENT = "Telco-Churn-Drift-Monitoring"
MODEL_URI = "models:/telco-churn-classifier@champion"

P_THRESHOLD = 0.05        # statistical significance
PSI_MODERATE = 0.10       # PSI 0.10-0.25 = moderate shift
PSI_MAJOR = 0.25          # PSI > 0.25 = major shift
DATASET_DRIFT_SHARE = 0.30  # dataset flagged if >= 30% of features drift
AUC_DROP_ALERT = 0.05     # concept / performance alert
EPS = 1e-4


def psi(ref_pct, cur_pct):
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def numeric_psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return psi(r, c)


def categorical_psi(ref, cur):
    cats = sorted(set(ref) | set(cur))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).values
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).values
    return psi(r, c)


def feature_drift_table(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURES:
        if f in NUMERIC:
            test = "KS"
            stat, p = stats.ks_2samp(ref[f], cur[f])
            value = numeric_psi(ref[f].values, cur[f].values)
        else:
            test = "Chi-square"
            cats = sorted(set(ref[f]) | set(cur[f]))
            table = np.vstack([ref[f].value_counts().reindex(cats, fill_value=0),
                               cur[f].value_counts().reindex(cats, fill_value=0)])
            stat, p, _, _ = stats.chi2_contingency(table)
            value = categorical_psi(ref[f], cur[f])
        drift = (p < P_THRESHOLD) and (value >= PSI_MODERATE)
        severity = "major" if value > PSI_MAJOR else "moderate" if value >= PSI_MODERATE else "none"
        rows.append({"feature": f, "type": "numeric" if f in NUMERIC else "categorical", "test": test,
                     "statistic": stat, "p_value": p, "psi": value, "severity": severity, "drift": drift})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def plot_psi(table, title, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#c0392b" if s == "major" else "#e67e22" if s == "moderate" else "#7f8c8d"
              for s in table["severity"]]
    ax.barh(table["feature"][::-1], table["psi"][::-1], color=colors[::-1])
    ax.axvline(PSI_MODERATE, ls="--", color="#e67e22", lw=1, label="PSI 0.10 (moderate)")
    ax.axvline(PSI_MAJOR, ls="--", color="#c0392b", lw=1, label="PSI 0.25 (major)")
    ax.set_xlabel("Population Stability Index")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_features(ref, cur, table, title, path):
    top = table["feature"].head(4).tolist()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, f in zip(axes, top):
        if f in NUMERIC:
            bins = np.histogram_bin_edges(np.concatenate([ref[f], cur[f]]), bins=30)
            ax.hist(ref[f], bins=bins, alpha=0.55, density=True, label="reference", color="#2c7fb8")
            ax.hist(cur[f], bins=bins, alpha=0.55, density=True, label="current", color="#d95f0e")
        else:
            cats = sorted(set(ref[f]) | set(cur[f]))
            r = ref[f].value_counts(normalize=True).reindex(cats, fill_value=0)
            c = cur[f].value_counts(normalize=True).reindex(cats, fill_value=0)
            x = np.arange(len(cats))
            ax.bar(x - 0.2, r, 0.4, label="reference", color="#2c7fb8")
            ax.bar(x + 0.2, c, 0.4, label="current", color="#d95f0e")
            ax.set_xticks(x, [str(k)[:14] for k in cats], rotation=30, ha="right", fontsize=7)
        psi_v = table.set_index("feature").loc[f, "psi"]
        ax.set_title(f"{f}  (PSI={psi_v:.2f})", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_predictions(p_ref, p_cur, title, path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 26)
    ax.hist(p_ref, bins=bins, alpha=0.55, density=True, label=f"reference (mean {p_ref.mean():.2f})", color="#2c7fb8")
    ax.hist(p_cur, bins=bins, alpha=0.55, density=True, label=f"current (mean {p_cur.mean():.2f})", color="#d95f0e")
    ax.set_xlabel("Predicted churn probability")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def html_report(name, desc, table, summary, path):
    tbl = table.copy()
    tbl["p_value"] = tbl["p_value"].map(lambda v: f"{v:.2e}")
    tbl["statistic"] = tbl["statistic"].map(lambda v: f"{v:.3f}")
    tbl["psi"] = tbl["psi"].map(lambda v: f"{v:.3f}")
    rows = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in summary.items())
    path.write_text(f"""<html><head><meta charset='utf-8'><title>Drift report - {name}</title>
<style>body{{font-family:Arial;margin:24px}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}</style>
</head><body><h2>Drift report - {name}</h2><p>{desc}</p><ul>{rows}</ul>{tbl.to_html(index=False)}</body></html>""")


def run_scenario(model, name, desc, drift_source, ref, cur, baseline, labels_available):
    out = FIGURES / name
    out.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"stage": "monitoring", "scenario": name, "drift_source": drift_source,
                         "model_uri": MODEL_URI})
        mlflow.log_params({"reference_rows": len(ref), "current_rows": len(cur),
                           "p_value_threshold": P_THRESHOLD, "psi_moderate": PSI_MODERATE,
                           "psi_major": PSI_MAJOR, "dataset_drift_share": DATASET_DRIFT_SHARE,
                           "numeric_test": "Kolmogorov-Smirnov", "categorical_test": "Chi-square"})

        table = feature_drift_table(ref[FEATURES], cur[FEATURES])
        n_drift = int(table["drift"].sum())
        share = n_drift / len(FEATURES)
        dataset_drift = share >= DATASET_DRIFT_SHARE

        p_ref = model.predict_proba(ref[FEATURES])[:, 1]
        p_cur = model.predict_proba(cur[FEATURES])[:, 1]
        pred_psi = numeric_psi(p_ref, p_cur)
        pred_ks, pred_p = stats.ks_2samp(p_ref, p_cur)
        pred_drift = (pred_p < P_THRESHOLD) and (pred_psi >= PSI_MODERATE)

        metrics = {"n_drifted_features": n_drift, "share_drifted_features": share,
                   "dataset_drift": int(dataset_drift), "prediction_psi": pred_psi,
                   "prediction_ks_stat": pred_ks, "prediction_ks_pvalue": pred_p,
                   "prediction_drift": int(pred_drift),
                   "mean_pred_churn_reference": float(p_ref.mean()),
                   "mean_pred_churn_current": float(p_cur.mean())}
        for _, r in table.iterrows():
            metrics[f"psi_{r.feature}"] = r.psi

        if labels_available:
            y = cur[TARGET]
            auc = roc_auc_score(y, p_cur)
            metrics.update({"current_roc_auc": auc, "current_recall": recall_score(y, p_cur >= 0.5),
                            "current_f1": f1_score(y, p_cur >= 0.5),
                            "actual_churn_rate_reference": float(ref[TARGET].mean()),
                            "actual_churn_rate_current": float(y.mean()),
                            "roc_auc_drop": baseline["roc_auc"] - auc,
                            "performance_alert": int(baseline["roc_auc"] - auc > AUC_DROP_ALERT)})
        mlflow.log_metrics(metrics)

        csv = out / "feature_drift.csv"
        table.to_csv(csv, index=False)
        plot_psi(table, f"Feature PSI - {name}", out / "psi_by_feature.png")
        plot_top_features(ref, cur, table, f"Top shifted features - {name}", out / "top_features.png")
        plot_predictions(p_ref, p_cur, f"Prediction distribution - {name}", out / "prediction_drift.png")
        summary = {"dataset_drift": dataset_drift, "drifted_features": f"{n_drift}/{len(FEATURES)}",
                   "prediction_psi": round(pred_psi, 3), "prediction_drift": pred_drift}
        if labels_available:
            summary["current ROC-AUC"] = round(metrics["current_roc_auc"], 3)
        html_report(name, desc, table, summary, out / "drift_report.html")
        mlflow.log_artifacts(str(out), artifact_path="drift_report")

        result = {"description": desc, "drift_source": drift_source, **metrics,
                  "drifted_features": table.loc[table["drift"], "feature"].tolist(),
                  "top5": table.head(5)[["feature", "test", "p_value", "psi", "severity"]].to_dict("records")}
        print(f"\n[{name}] dataset_drift={dataset_drift} drifted={n_drift}/{len(FEATURES)} "
              f"pred_psi={pred_psi:.3f} pred_drift={pred_drift}")
        print(table.head(6).to_string(index=False))
        return result


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    model = mlflow.sklearn.load_model(MODEL_URI)

    df = load_data()
    _, reference, new_cohort = make_splits(df)
    # Control: split the reference window into two random halves of the same population.
    ref_a = reference.sample(frac=0.5, random_state=SEED)
    cur_a = reference.drop(ref_a.index)

    p = model.predict_proba(reference[FEATURES])[:, 1]
    baseline = {"roc_auc": roc_auc_score(reference[TARGET], p)}

    results = {
        "A_control_no_drift": run_scenario(
            model, "A_control_no_drift",
            "Two random halves of the established-customer hold-out (same population). Sanity check for false alarms.",
            "none (control)", ref_a, cur_a, baseline, True),
        "B_new_customer_cohort": run_scenario(
            model, "B_new_customer_cohort",
            "Reference = established customers (tenure > 12 m). Current = newly acquired customers (tenure <= 12 m).",
            "naturally observed", reference, new_cohort, baseline, True),
        "C_price_revision": run_scenario(
            model, "C_price_revision",
            "Reference hold-out vs. the same customers after a simulated 18% tariff hike and 30% of auto-pay "
            "customers moving to electronic cheque. Ground-truth churn not yet observed.",
            "simulated", reference, simulate_price_revision(reference), baseline, False),
    }
    results["baseline_reference_roc_auc"] = baseline["roc_auc"]
    (REPORTS / "drift_summary.json").write_text(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
