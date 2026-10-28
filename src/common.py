"""Shared helpers: data loading, cleaning, reference/current splits."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
SEED = 42

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Churn"


def load_data() -> pd.DataFrame:
    """Load the IBM Telco churn CSV and fix types.

    TotalCharges is stored as text and is blank for 11 brand-new customers
    (tenure = 0); those are set to 0 because nothing has been billed yet.
    """
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def make_splits(df: pd.DataFrame):
    """Return (train, reference, current_cohort).

    * Established customers (tenure > 12 months) are what the model was built on.
      They are split 70/30 into train and a held-out *reference* window.
    * New customers (tenure <= 12 months) form the *current* production window -
      a naturally observed shift in who the model is scoring.
    """
    established = df[df["tenure"] > 12].sample(frac=1.0, random_state=SEED)
    new_cohort = df[df["tenure"] <= 12].reset_index(drop=True)
    cut = int(len(established) * 0.7)
    train = established.iloc[:cut].reset_index(drop=True)
    reference = established.iloc[cut:].reset_index(drop=True)
    return train, reference, new_cohort


def simulate_price_revision(ref: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Simulated drift: an 18% tariff hike plus a move of 30% of auto-pay
    customers to electronic cheque (e.g. after an auto-debit mandate change)."""
    rng = np.random.default_rng(seed)
    cur = ref.copy()
    cur["MonthlyCharges"] = (cur["MonthlyCharges"] * 1.18).round(2)
    cur["TotalCharges"] = (cur["TotalCharges"] * 1.18).round(2)
    auto = cur["PaymentMethod"].str.contains("automatic")
    switch = auto & (rng.random(len(cur)) < 0.30)
    cur.loc[switch, "PaymentMethod"] = "Electronic check"
    return cur
