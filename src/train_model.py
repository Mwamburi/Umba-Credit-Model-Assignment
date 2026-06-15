"""
Train the behavioural XGBoost fraud model and generate validation outputs.

Run from project root:
    python src/train_model.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier
import numpy as np

from src.feature_engineering import (
    ID_COL,
    TARGET,
    OUT,
    aggregate_identity,
    chronological_split,
    create_features,
    engineered_feature_list,
    fit_references,
    read_data,
)

warnings.filterwarnings("ignore")


def threshold_screen(y_true, proba):
    rows = []
    for t in np.linspace(0.05, 0.95, 91):
        pred = (proba >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        rows.append({"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "threshold_screen.csv", index=False)
    return tab.sort_values(["f1", "precision"], ascending=False).iloc[0].to_dict()


def train() -> dict:
    train_raw, test_raw, identity_raw = read_data()
    refs = fit_references(train_raw)
    identity_agg = aggregate_identity(identity_raw)

    train_fe = create_features(train_raw, identity_agg, refs)
    test_fe = create_features(test_raw, identity_agg, refs)

    y = train_fe.pop(TARGET).astype(int)
    test_ids = test_raw[ID_COL].copy()
    if TARGET in test_fe.columns:
        test_fe = test_fe.drop(columns=[TARGET])

    train_idx, valid_idx = chronological_split(train_raw)
    X_train, X_valid = train_fe.loc[train_idx], train_fe.loc[valid_idx]
    y_train, y_valid = y.loc[train_idx], y.loc[valid_idx]
    test_fe = test_fe.reindex(columns=train_fe.columns, fill_value=np.nan)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_valid_imp = imputer.transform(X_valid)
    X_test_imp = imputer.transform(test_fe)

    scale_pos_weight = float((y_train == 0).sum() / max(y_train.sum(), 1))
    model = XGBClassifier(
        n_estimators=160,
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=2,
    )
    model.fit(X_train_imp, y_train)

    valid_proba = model.predict_proba(X_valid_imp)[:, 1]
    best = threshold_screen(y_valid.values, valid_proba)
    valid_pred = (valid_proba >= best["threshold"]).astype(int)

    importance = pd.DataFrame({"feature": train_fe.columns, "gain_importance": model.feature_importances_})
    importance = importance.sort_values("gain_importance", ascending=False)
    importance.to_csv(OUT / "xgboost_feature_importance_gain.csv", index=False)

    test_proba = model.predict_proba(X_test_imp)[:, 1]
    pd.DataFrame({ID_COL: test_ids, "isFraud": test_proba}).to_csv(OUT / "submission.csv", index=False)

    pd.DataFrame({"feature": engineered_feature_list(train_fe.columns.tolist())}).to_csv(
        OUT / "engineered_features_used.csv", index=False
    )

    artifact = {
        "model": model,
        "imputer": imputer,
        "refs": refs,
        "identity_agg": identity_agg,
        "feature_columns": train_fe.columns.tolist(),
        "best_threshold": float(best["threshold"]),
    }
    joblib.dump(artifact, OUT / "model_artifact.joblib")

    summary = {
        "rows_train": int(len(train_raw)),
        "rows_test": int(len(test_raw)),
        "train_fraud_rate": float(y.mean()),
        "validation_fraud_rate": float(y_valid.mean()),
        "features_before_imputation": int(train_fe.shape[1]),
        "scale_pos_weight": scale_pos_weight,
        "validation_pr_auc": float(average_precision_score(y_valid, valid_proba)),
        "validation_roc_auc": float(roc_auc_score(y_valid, valid_proba)),
        "best_threshold_by_f1": best,
        "confusion_matrix_at_best_threshold": confusion_matrix(y_valid, valid_pred).tolist(),
        "top_25_features": importance.head(25).to_dict(orient="records"),
    }
    with open(OUT / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    summary = train()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
