# Umba Fraud Detection Solution

## Overview

This repository contains a compact fraud detection solution for the Umba take-home assignment. It moves from behavioural EDA to feature engineering, XGBoost modelling, batch scoring, API deployment, and a lightweight Streamlit dashboard.

The solution is intentionally simple and interview-friendly: the feature layer is interpretable, the model is a single XGBoost classifier, and the deployment components are small enough to inspect quickly.

## Architecture

```text
Transaction Data + Identity Data
            ↓
Behavioural Feature Engineering
            ↓
XGBoost Fraud Model
            ↓
Fraud Probability Score
            ↓
Submission CSV / API / Dashboard
```

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .dockerignore
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── identity.csv
│   ├── sample_submission.csv
│   └── .gitkeep
│
├── docs/
│   ├── ASSIGNMENT_README.md
│   ├── DATA_DICTIONARY.md
│   ├── PROJECT_SUMMARY.md
│   └── MODEL_EVALUATION_RECOMMENDATIONS.md
│
├── notebooks/
│   └── 01_eda_behavioural_signals.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── scoring_api.py
│   ├── dashboard.py
│   ├── utils.py
│   └── fraud_model.py
│
└── outputs/
    ├── model_artifact.joblib
    ├── model_summary.json
    ├── submission.csv
    ├── threshold_screen.csv
    ├── engineered_features_used.csv
    ├── xgboost_feature_importance_gain.csv
    └── .gitkeep
```

`src/fraud_model.py` is retained as a backward-compatible wrapper, while the main implementation is split into focused modules.

## Method Summary

The model is built around behavioural fraud signals identified during EDA.

| Feature Group | Examples |
|---|---|
| Transaction Behaviour | amount vs country median, amount vs channel median |
| Sender & Recipient Maturity | sender maturity score, recipient maturity score, recipient risk |
| Identity Confidence | identity false rate, identity completeness |
| Behavioural Intensity | velocity mean, velocity max, log velocity total |
| Behavioural Interactions | amount × identity risk, velocity × identity risk, amount × recipient risk |

The model uses XGBoost with class weighting and a chronological validation split.

## Validation Results

| Metric | Result |
|---|---:|
| PR-AUC | 0.168 |
| ROC-AUC | 0.788 |
| Best F1 threshold | 0.75 |
| Precision | 18.7% |
| Recall | 34.9% |
| F1 | 24.4% |

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model and generate validation outputs plus the submission file:

```bash
python src/train_model.py
```

For backward compatibility, this also works:

```bash
python src/fraud_model.py
```

Batch score the test file using the saved model artifact:

```bash
python src/predict.py --input data/test.csv --output outputs/submission.csv
```

## Run the API Locally

After training:

```bash
uvicorn src.scoring_api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"records": [{"TransactionID": 1, "TransactionAmt": 1000}]}'
```

## Run the Dashboard Locally

The dashboard provides a light operational view of model scores, risk bands, high-risk transactions, validation metrics, and top model drivers.

```bash
streamlit run src/dashboard.py --server.port 8501
```

Open:

```text
http://localhost:8501
```

## Docker Deployment

Build the image:

```bash
docker build -t umba-fraud .
```

Run the API:

```bash
docker run -p 8000:8000 umba-fraud
```

Run the dashboard using the same image:

```bash
docker run -p 8501:8501 umba-fraud streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Run both API and dashboard together:

```bash
docker compose up --build
```

Endpoints:

```text
API:        http://localhost:8000
Dashboard:  http://localhost:8501
```

## Main Files to Review

| File | Purpose |
|---|---|
| `docs/PROJECT_SUMMARY.md` | Succinct project scope, approach, results, and recommendations |
| `notebooks/01_eda_behavioural_signals.ipynb` | Behavioural EDA and observed fraud patterns |
| `src/feature_engineering.py` | Reusable behavioural feature logic |
| `src/train_model.py` | Model training, validation, feature importance, and submission generation |
| `src/predict.py` | Batch scoring entry point |
| `src/scoring_api.py` | FastAPI scoring service |
| `src/dashboard.py` | Lightweight Streamlit fraud monitoring dashboard |
| `docs/MODEL_EVALUATION_RECOMMENDATIONS.md` | Evaluation interpretation and next-step recommendations |
| `outputs/submission.csv` | Final scored test-set predictions |

## Git Notes

The repository includes a `.gitignore` that keeps the folder structure while avoiding accidental commits of private data, generated outputs, and model artifacts. For assignment submission, the packaged zip includes the required data and outputs. For a public GitHub version, keep only `.gitkeep` files inside `data/` and `outputs/`.
