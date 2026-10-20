"""Shared data helpers for ChurnLens (Team Visionaries)."""
import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join("data", "telco.csv")
URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
]

# gender is NOT a model feature (fairness audit only)
RAW_FEATURES = [
    "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]
BIN = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTI = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
NUM = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data():
    """Load Telco CSV from data/telco.csv, else try to download it."""
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        last = None
        for u in URLS:
            try:
                print("Downloading dataset:", u)
                pd.read_csv(u).to_csv(DATA_PATH, index=False)
                break
            except Exception as e:  # noqa
                last = e
        if not os.path.exists(DATA_PATH):
            raise SystemExit(
                "Dataset download failed (%s).\nManually download the IBM Telco Customer Churn CSV "
                "and save it as data/telco.csv, then run again." % last
            )
    return pd.read_csv(DATA_PATH)


def clean(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    df["Churn"] = (df["Churn"].astype(str).str.strip() == "Yes").astype(int)
    return df.drop(columns=[c for c in ["customerID"] if c in df.columns])


def encode(df, columns=None):
    """Raw customer rows -> model matrix. `columns` fixes the final column order."""
    d = df[RAW_FEATURES].copy()
    for c in BIN:
        d[c] = (d[c] == "Yes").astype(int)
    d = pd.get_dummies(d, columns=MULTI, dtype=int)
    if columns is not None:
        d = d.reindex(columns=columns, fill_value=0)
    return d
