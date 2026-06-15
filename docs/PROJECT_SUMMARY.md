# Umba Fraud Detection Take-Home — Project Summary

## Scope

This project builds a fraud detection baseline for an online-facing financial services system. The work covers exploratory analysis, behavioural feature engineering, supervised model training, validation, test-set scoring, API deployment, and a lightweight operational dashboard.

The solution is intentionally simple and production-minded: a compact behavioural feature layer feeding an XGBoost classifier, with outputs that can support approve, review, or block decisions.

## Approach

The EDA focused on identifying behavioural patterns that separate fraudulent and legitimate transactions. The strongest observed themes were transaction deviation, recipient maturity, identity confidence, and behavioural intensity.

The feature-engineering module converts those observations into engineered features. Amounts are contextualised using country and channel medians. Sender and recipient history are transformed into maturity scores. The C-series variables are summarised into velocity indicators. The M-series variables are summarised into identity confidence indicators. A small number of interaction features then capture compounded risk, such as large transactions with weak identity signals or elevated activity.

A chronological validation split was used to mirror live deployment, where the model is trained on historical transactions and evaluated on later incoming transactions.

## Results

The final model achieved the following validation performance:

| Metric | Result |
|---|---:|
| PR-AUC | 0.168 |
| ROC-AUC | 0.788 |
| Best F1 threshold | 0.75 |
| Precision | 18.7% |
| Recall | 34.9% |
| F1 | 24.4% |

Given the portfolio fraud rate of approximately 3.4%, the model concentrates fraud into a substantially higher-risk population. The strongest features include V-series engineered signals, amount × identity risk, identity false rate, velocity × identity risk, amount × recipient risk, recipient maturity, and amount deviation.

## Operational Layer

The package includes two lightweight deployment components.

The FastAPI service exposes a `/predict` endpoint for scoring incoming transaction records and returning fraud probabilities with recommended actions.

The Streamlit dashboard provides a light analyst-facing view of model outputs, including portfolio score summaries, risk-band distribution, a high-risk transaction queue, validation metrics, and top model drivers.

## Recommendations

The current solution is a strong first-pass behavioural model. The next improvement areas should focus on expanding behavioural coverage rather than adding model complexity.

Recommended enhancements are device intelligence, entity reputation, network-based fraud signals, rolling-window velocity, and post-review feedback loops. These would help detect fraud types that are not fully visible from transaction-level behaviour alone.

## Deliverables

The package includes an EDA notebook, reusable feature-engineering module, model training script, batch prediction script, model outputs, submission file, scoring API, dashboard app, Dockerfile, Docker Compose file, README, and concise recommendations document.
