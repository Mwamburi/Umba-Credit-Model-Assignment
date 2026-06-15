"""
Batch scoring entry point.

By default this scores data/test.csv using outputs/model_artifact.joblib and writes
outputs/submission.csv.

Run from project root:
    python src/predict.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.feature_engineering import ID_COL, TARGET, DATA, OUT, create_features


def score(input_path: Path, output_path: Path, artifact_path: Path) -> pd.DataFrame:
    artifact = joblib.load(artifact_path)
    raw = pd.read_csv(input_path)
    features = create_features(raw, artifact["identity_agg"], artifact["refs"])
    if TARGET in features.columns:
        features = features.drop(columns=[TARGET])
    features = features.reindex(columns=artifact["feature_columns"], fill_value=None)
    features_imp = artifact["imputer"].transform(features)
    scores = artifact["model"].predict_proba(features_imp)[:, 1]
    ids = raw[ID_COL] if ID_COL in raw.columns else pd.Series(range(len(raw)), name=ID_COL)
    out = pd.DataFrame({ID_COL: ids, "isFraud": scores})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score transactions with the trained fraud model.")
    parser.add_argument("--input", default=str(DATA / "test.csv"), help="CSV file to score")
    parser.add_argument("--output", default=str(OUT / "submission.csv"), help="Output CSV path")
    parser.add_argument("--artifact", default=str(OUT / "model_artifact.joblib"), help="Trained model artifact")
    args = parser.parse_args()

    scored = score(Path(args.input), Path(args.output), Path(args.artifact))
    print(f"Wrote {len(scored):,} scored rows to {args.output}")


if __name__ == "__main__":
    main()
