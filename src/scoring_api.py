"""
Minimal FastAPI scoring service for the fraud model.

Start locally after training:
    uvicorn src.scoring_api:app --host 0.0.0.0 --port 8000

Example request:
    POST /predict
    {"records": [{"TransactionID": 1, "TransactionAmt": 1000, ...}]}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.feature_engineering import create_features, ID_COL, TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = PROJECT_ROOT / "outputs" / "model_artifact.joblib"

app = FastAPI(title="Umba Fraud Detection API", version="1.0")


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]


@app.get("/health")
def health():
    return {"status": "ok", "artifact_exists": ARTIFACT_PATH.exists()}


@app.post("/predict")
def predict(req: PredictRequest):
    if not ARTIFACT_PATH.exists():
        raise HTTPException(status_code=500, detail="Model artifact not found. Run src/train_model.py first.")

    artifact = joblib.load(ARTIFACT_PATH)
    df = pd.DataFrame(req.records)
    if df.empty:
        raise HTTPException(status_code=400, detail="No records supplied.")

    features = create_features(df, artifact["identity_agg"], artifact["refs"])
    if TARGET in features.columns:
        features = features.drop(columns=[TARGET])
    features = features.reindex(columns=artifact["feature_columns"], fill_value=None)
    features_imp = artifact["imputer"].transform(features)
    scores = artifact["model"].predict_proba(features_imp)[:, 1]

    threshold = artifact.get("best_threshold", 0.75)
    ids = df[ID_COL].tolist() if ID_COL in df.columns else list(range(len(df)))
    return {
        "threshold": threshold,
        "predictions": [
            {
                "TransactionID": ids[i],
                "fraud_probability": float(scores[i]),
                "recommended_action": "review" if float(scores[i]) >= threshold else "approve",
            }
            for i in range(len(scores))
        ],
    }
