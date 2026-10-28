# MLflow Experiment Tracking & Drift Analysis — Telecom Customer Churn

ML DevOps (UAMPC0703) · Theory ISE-2 · A.Y. 2026-27  
Prathmesh Digvijay Mhopare · Roll No. 56 · B.Tech (Data Science), KIT's College of Engineering, Kolhapur

## What this project does
1. **Experiment tracking (MLflow).** Trains three churn classifiers (Logistic Regression, Random Forest,
   Gradient Boosting) on the IBM Telco Customer Churn dataset. Each run logs parameters, metrics,
   a confusion-matrix figure and the model with its signature. The best model by ROC-AUC is registered
   as `telco-churn-classifier` and given the alias `champion`.
2. **Drift monitoring (MLflow).** Loads `models:/telco-churn-classifier@champion` and compares a
   reference window against a current window in three scenarios. Each scenario is a separate MLflow run:

| Scenario | Reference | Current | Drift source |
|---|---|---|---|
| A_control_no_drift | random half of the hold-out | the other half | none (control) |
| B_new_customer_cohort | established customers (tenure > 12 m) | new customers (tenure ≤ 12 m) | naturally observed |
| C_price_revision | hold-out | same customers after +18% tariff and 30% of auto-pay customers moved to e-cheque | simulated |

**Detection method:** Kolmogorov–Smirnov test (numeric), Chi-square test (categorical) and the
Population Stability Index (PSI) for every feature. A feature counts as drifted when p < 0.05 **and**
PSI ≥ 0.10. A dataset is drifted when ≥ 30% of features drift. Prediction drift uses PSI and KS on the
predicted churn probabilities. When labels exist, a performance alert fires if ROC-AUC drops by more than 0.05.

## Results
| Scenario | Drifted features | Dataset drift | Prediction PSI | ROC-AUC (baseline 0.845) |
|---|---|---|---|---|
| A control | 0 / 19 | No | 0.034 | 0.832 |
| B new cohort | 14 / 19 | **Yes** | **1.313** | **0.765** (alert) |
| C price revision | 2 / 19 (MonthlyCharges, PaymentMethod) | No | 0.019 | labels not yet available |

Plots and per-feature tables are in `reports/figures/<scenario>/`. MLflow UI screenshots are in `screenshots/`.

## How to run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/train.py      # 3 tracked runs + model registry
python src/drift.py      # 3 drift-monitoring runs
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # open http://127.0.0.1:5000
```
The dataset ships in `data/`. Source: https://github.com/IBM/telco-customer-churn-on-icp4d

## Structure
```
data/Telco-Customer-Churn.csv   dataset (7,043 customers, 21 columns)
src/common.py                   loading, cleaning, reference/current splits, drift simulation
src/train.py                    MLflow experiment tracking + model registry
src/drift.py                    KS / Chi-square / PSI drift detection logged to MLflow
reports/                        training_summary.json, drift_summary.json, figures/
screenshots/                    MLflow UI evidence
```
