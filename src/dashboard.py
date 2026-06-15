"""
Light Streamlit dashboard for reviewing fraud model outputs.

Run from project root:
    streamlit run src/dashboard.py --server.port 8501 --server.address 0.0.0.0
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_outputs():
    submission_path = OUT / "submission.csv"
    importance_path = OUT / "xgboost_feature_importance_gain.csv"
    threshold_path = OUT / "threshold_screen.csv"
    engineered_path = OUT / "engineered_features_used.csv"

    submission = pd.read_csv(submission_path) if submission_path.exists() else pd.DataFrame()
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    threshold = pd.read_csv(threshold_path) if threshold_path.exists() else pd.DataFrame()
    engineered = pd.read_csv(engineered_path) if engineered_path.exists() else pd.DataFrame()
    summary = load_json(OUT / "model_summary.json")
    return submission, importance, threshold, engineered, summary


def risk_band(score: float) -> str:
    if score >= 0.90:
        return "Critical"
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


def main() -> None:
    st.set_page_config(page_title="Umba Fraud Dashboard", layout="wide")
    st.title("Umba Fraud Detection Dashboard")
    st.caption("Light operational view of model scores, risk bands, high-risk transactions and feature drivers.")

    submission, importance, threshold, engineered, summary = load_outputs()

    if submission.empty:
        st.error("No submission.csv found. Run `python src/train_model.py` first.")
        return

    score_col = "isFraud"
    best_block = summary.get("best_threshold_by_f1", {})
    threshold_value = float(best_block.get("threshold", summary.get("best_threshold", 0.75)))
    submission = submission.copy()
    submission["risk_band"] = submission[score_col].apply(risk_band)
    submission["recommended_action"] = submission[score_col].apply(
        lambda x: "Review" if x >= threshold_value else "Approve"
    )

    total_txns = len(submission)
    avg_score = submission[score_col].mean()
    review_count = int((submission[score_col] >= threshold_value).sum())
    critical_count = int((submission[score_col] >= 0.90).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scored Transactions", f"{total_txns:,}")
    c2.metric("Average Fraud Score", f"{avg_score:.3f}")
    c3.metric("Review Queue", f"{review_count:,}")
    c4.metric("Critical Risk", f"{critical_count:,}")

    st.divider()

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Fraud Score Distribution")
        hist = pd.cut(submission[score_col], bins=[0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0], include_lowest=True)
        hist_df = hist.value_counts().sort_index().rename_axis("score_band").reset_index(name="transactions")
        hist_df["score_band"] = hist_df["score_band"].astype(str)
        st.bar_chart(hist_df, x="score_band", y="transactions")

    with right:
        st.subheader("Risk Band Breakdown")
        band_order = ["Low", "Medium", "High", "Critical"]
        band_df = submission["risk_band"].value_counts().reindex(band_order, fill_value=0).reset_index()
        band_df.columns = ["risk_band", "transactions"]
        st.dataframe(band_df, use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("High-Risk Transaction Queue")
    top_n = st.slider("Transactions to show", min_value=10, max_value=100, value=25, step=5)
    queue_cols = [c for c in ["TransactionID", score_col, "risk_band", "recommended_action"] if c in submission.columns]
    high_risk = submission.sort_values(score_col, ascending=False).head(top_n)[queue_cols]
    st.dataframe(high_risk, use_container_width=True, hide_index=True)

    st.divider()

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Top Model Drivers")
        if not importance.empty:
            imp = importance.head(15).copy()
            st.bar_chart(imp, x="feature", y="gain_importance")
        else:
            st.info("Feature importance file not found.")

    with right:
        st.subheader("Validation Snapshot")
        metric_rows = []
        if "validation_pr_auc" in summary:
            metric_rows.append({"metric": "PR-AUC", "value": round(float(summary["validation_pr_auc"]), 4)})
        if "validation_roc_auc" in summary:
            metric_rows.append({"metric": "ROC-AUC", "value": round(float(summary["validation_roc_auc"]), 4)})
        best_block = summary.get("best_threshold_by_f1", {})
        for key, label in {
            "threshold": "Best F1 Threshold",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }.items():
            if key in best_block:
                metric_rows.append({"metric": label, "value": round(float(best_block[key]), 4)})
        if metric_rows:
            st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        if not threshold.empty:
            st.caption("Threshold screen available in outputs/threshold_screen.csv")

    st.divider()

    with st.expander("Engineered behavioural features used"):
        if not engineered.empty:
            st.dataframe(engineered, use_container_width=True, hide_index=True)
        else:
            st.info("Engineered feature list not found.")


if __name__ == "__main__":
    main()
