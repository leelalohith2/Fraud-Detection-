"""
Data Pipeline Module
=====================
Ingests financial transaction data (CSV upload, Kaggle-style datasets,
or a Kafka stream via streaming.py), then cleans, validates, and
feature-engineers it into the shape the Hybrid AI Engine expects.

All functions are pure and reusable so they can be called from the
FastAPI backend, the Streamlit frontend, or the test suite alike.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from backend.config import get_logger, settings

logger = get_logger(__name__)

REQUIRED_FIELDS = [
    "transaction_id", "customer_id", "transaction_amount", "timestamp",
    "merchant", "device_type", "location", "payment_method",
]


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a transactions CSV from disk (upload or Kaggle export)."""
    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def load_uploaded_csv(file_obj) -> pd.DataFrame:
    """Load a transactions CSV from a Streamlit UploadedFile object."""
    df = pd.read_csv(file_obj)
    logger.info("Loaded %d rows from uploaded file", len(df))
    return df


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Check that all required fields are present. Returns (is_valid, missing_fields)."""
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    return (len(missing) == 0, missing)


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode = df[col].mode()
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    dedup_key = "transaction_id" if "transaction_id" in df.columns else None
    df = df.drop_duplicates(subset=dedup_key) if dedup_key else df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate transactions", removed)
    return df.reset_index(drop=True)


def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    return df


# ----------------------------------------------------------------------
# Feature Engineering
# ----------------------------------------------------------------------

def engineer_features(df: pd.DataFrame, customers: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Derive the behavioral / temporal / velocity features the Hybrid AI
    Engine relies on. Safe to call repeatedly (idempotent)."""
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour_of_day"] = df["timestamp"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
    df["is_night_transaction"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h >= 23) else 0)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    if "is_foreign" not in df.columns:
        df["is_foreign"] = 0
    df["is_foreign"] = df["is_foreign"].fillna(False).astype(int)

    # Per-customer velocity features
    df = df.sort_values(["customer_id", "timestamp"])
    df["prev_txn_time"] = df.groupby("customer_id")["timestamp"].shift(1)
    df["minutes_since_last_txn"] = (
        (df["timestamp"] - df["prev_txn_time"]).dt.total_seconds() / 60
    )
    df["minutes_since_last_txn"] = df["minutes_since_last_txn"].fillna(999999)
    df["is_rapid_transaction"] = (df["minutes_since_last_txn"] < 5).astype(int)

    df["prev_device"] = df.groupby("customer_id")["device_type"].shift(1)
    df["device_switch"] = (
        (df["prev_device"].notna()) & (df["prev_device"] != df["device_type"])
    ).astype(int)

    df["txn_count_last_24h"] = (
        df.groupby("customer_id")["timestamp"]
        .transform(lambda s: s.apply(lambda t: ((s >= t - pd.Timedelta(hours=24)) & (s <= t)).sum()))
    )

    # Customer-level spend baseline
    cust_stats = df.groupby("customer_id")["transaction_amount"].agg(["mean", "std"]).rename(
        columns={"mean": "cust_avg_amount", "std": "cust_std_amount"}
    )
    cust_stats["cust_std_amount"] = cust_stats["cust_std_amount"].fillna(cust_stats["cust_avg_amount"] * 0.5 + 1)
    df = df.merge(cust_stats, on="customer_id", how="left")
    df["amount_zscore"] = (
        (df["transaction_amount"] - df["cust_avg_amount"]) / df["cust_std_amount"].replace(0, 1)
    ).fillna(0)

    if customers is not None and not customers.empty:
        df = df.merge(
            customers[["customer_id", "home_city", "account_age_days", "avg_monthly_spend", "risk_segment"]],
            on="customer_id", how="left",
        )
        df["is_new_device_location"] = (df["location"] != df["home_city"]).astype(int)
        df["account_age_days"] = df["account_age_days"].fillna(365)
    else:
        df["is_new_device_location"] = 0
        df["account_age_days"] = 365

    df = df.drop(columns=["prev_txn_time", "prev_device"], errors="ignore")
    df["is_high_amount"] = (df["transaction_amount"] > df["transaction_amount"].quantile(0.95)).astype(int)

    return df.reset_index(drop=True)


def run_full_pipeline(df: pd.DataFrame, customers: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Convenience wrapper: validate -> clean -> feature engineer."""
    is_valid, missing = validate_schema(df)
    if not is_valid:
        raise ValueError(f"Dataset missing required fields: {missing}")
    df = clean_pipeline(df)
    df = engineer_features(df, customers=customers)
    logger.info("Pipeline complete: %d transactions ready for scoring", len(df))
    return df
