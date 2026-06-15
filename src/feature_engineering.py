"""
Behavioural feature engineering for the Umba fraud detection solution.

The feature layer converts raw transaction and identity attributes into compact,
interpretable fraud signals that can be reused consistently during training,
batch scoring, API scoring, and dashboarding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data"
OUT = PROJECT_ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"
DROP_ALWAYS = ["flagged_for_review"]
C_COLS = [f"C{i}" for i in range(1, 9)]
M_COLS = [f"M{i}" for i in range(1, 7)]
RAW_MATURITY_COLS = ["sender_prev_txn_count", "recipient_account_age_days"]


def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the assignment datasets from the data directory."""
    return (
        pd.read_csv(DATA / "train.csv"),
        pd.read_csv(DATA / "test.csv"),
        pd.read_csv(DATA / "identity.csv"),
    )


def aggregate_identity(identity: pd.DataFrame) -> pd.DataFrame:
    """Aggregate many identity rows to one transaction-level row."""
    id_numeric = [c for c in identity.columns if c.startswith("id_")]
    out = identity.groupby(ID_COL).size().rename("session_count").to_frame()

    for col in ["DeviceType", "DeviceInfo"]:
        if col in identity.columns:
            out[f"{col.lower()}_unique_count"] = identity.groupby(ID_COL)[col].nunique(dropna=True)
            out[f"{col.lower()}_missing_rate"] = identity.groupby(ID_COL)[col].apply(lambda s: s.isna().mean())

    if id_numeric:
        tmp = identity[[ID_COL] + id_numeric].copy()
        tmp["id_numeric_missing_rate_row"] = tmp[id_numeric].isna().mean(axis=1)
        tmp["id_numeric_present_count_row"] = tmp[id_numeric].notna().sum(axis=1)
        id_summary = tmp.groupby(ID_COL).agg(
            id_numeric_missing_rate=("id_numeric_missing_rate_row", "mean"),
            id_numeric_present_count=("id_numeric_present_count_row", "mean"),
        )
        id_means = tmp.groupby(ID_COL)[id_numeric].mean().add_prefix("agg_")
        out = out.join(id_summary).join(id_means)

    return out.reset_index()


def fit_references(train: pd.DataFrame) -> dict:
    """Fit training-only references used for feature construction."""
    cat_cols = train.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET] + M_COLS + DROP_ALWAYS]
    freq_maps = {col: train[col].value_counts(normalize=True, dropna=False).to_dict() for col in cat_cols}

    return {
        "global_amount_median": float(train["TransactionAmt"].median()),
        "country_amount_median": train.groupby("country")["TransactionAmt"].median().to_dict(),
        "channel_amount_median": train.groupby("channel")["TransactionAmt"].median().to_dict(),
        "freq_maps": freq_maps,
        "cat_cols": cat_cols,
    }


def create_features(df: pd.DataFrame, identity_agg: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """Create compact behavioural features for training or scoring."""
    x = df.copy().merge(identity_agg, how="left", on=ID_COL)

    global_median = refs["global_amount_median"]
    country_median = x["country"].map(refs["country_amount_median"]).fillna(global_median)
    channel_median = x["channel"].map(refs["channel_amount_median"]).fillna(global_median)

    x["log_transaction_amount"] = np.log1p(x["TransactionAmt"].clip(lower=0))
    x["amount_vs_country_median"] = x["TransactionAmt"] / country_median.replace(0, np.nan)
    x["amount_vs_channel_median"] = x["TransactionAmt"] / channel_median.replace(0, np.nan)

    sender_history = x["sender_prev_txn_count"].fillna(0).clip(lower=0)
    recipient_age = x["recipient_account_age_days"].fillna(0).clip(lower=0)
    x["sender_maturity_score"] = np.log1p(sender_history)
    x["recipient_maturity_score"] = np.log1p(recipient_age)
    x["recipient_risk"] = 1 / (recipient_age + 1)

    c_cols = [c for c in C_COLS if c in x.columns]
    velocity_total_raw = x[c_cols].fillna(0).sum(axis=1) if c_cols else pd.Series(0, index=x.index)
    x["velocity_mean"] = x[c_cols].fillna(0).mean(axis=1) if c_cols else 0
    x["velocity_max"] = x[c_cols].fillna(0).max(axis=1) if c_cols else 0
    x["log_velocity_total"] = np.log1p(velocity_total_raw.clip(lower=0))

    m_cols = [c for c in M_COLS if c in x.columns]
    if m_cols:
        m_available = x[m_cols].notna().sum(axis=1).replace(0, np.nan)
        m_false = (x[m_cols] == "F").sum(axis=1)
        x["m_false_count"] = m_false
        x["m_available_count"] = m_available.fillna(0)
        x["identity_false_rate"] = (m_false / m_available).fillna(0)
        x["identity_completeness"] = (m_available / len(m_cols)).fillna(0)
    else:
        x["m_false_count"] = 0
        x["m_available_count"] = 0
        x["identity_false_rate"] = 0
        x["identity_completeness"] = 0

    fill_defaults = {
        "session_count": 0,
        "devicetype_unique_count": 0,
        "deviceinfo_unique_count": 0,
        "devicetype_missing_rate": 1,
        "deviceinfo_missing_rate": 1,
        "id_numeric_missing_rate": 1,
        "id_numeric_present_count": 0,
    }
    for col, val in fill_defaults.items():
        if col in x.columns:
            x[col] = x[col].fillna(val)

    x["has_identity_record"] = (x.get("session_count", pd.Series(0, index=x.index)) > 0).astype(int)
    x["identity_table_completeness"] = 1 - x.get("id_numeric_missing_rate", pd.Series(1, index=x.index)).fillna(1)

    x["amount_x_recipient_risk"] = x["amount_vs_country_median"] * x["recipient_risk"]
    x["amount_x_identity_risk"] = x["amount_vs_country_median"] * x["identity_false_rate"]
    x["velocity_x_identity_risk"] = x["log_velocity_total"] * x["identity_false_rate"]
    x["amount_x_velocity"] = x["amount_vs_country_median"] * x["log_velocity_total"]

    for col in refs["cat_cols"]:
        if col in x.columns:
            x[f"{col}_freq"] = x[col].map(refs["freq_maps"].get(col, {})).fillna(0)

    drop_cols = [ID_COL] + DROP_ALWAYS + c_cols + m_cols + refs["cat_cols"] + RAW_MATURITY_COLS
    return x.drop(columns=[c for c in drop_cols if c in x.columns], errors="ignore")


def chronological_split(raw_train: pd.DataFrame, valid_size: float = 0.2):
    """Create a production-style validation split using later transactions as validation."""
    ordered = raw_train.sort_values(TIME_COL).index.to_numpy()
    cut = int(len(ordered) * (1 - valid_size))
    return ordered[:cut], ordered[cut:]


def engineered_feature_list(columns: list[str]) -> list[str]:
    engineered = [
        "log_transaction_amount", "amount_vs_country_median", "amount_vs_channel_median",
        "sender_maturity_score", "recipient_maturity_score", "recipient_risk",
        "velocity_mean", "velocity_max", "log_velocity_total",
        "identity_false_rate", "identity_completeness", "m_false_count", "m_available_count",
        "session_count", "has_identity_record", "identity_table_completeness",
        "amount_x_recipient_risk", "amount_x_identity_risk", "velocity_x_identity_risk", "amount_x_velocity",
    ]
    return [c for c in engineered if c in columns]
