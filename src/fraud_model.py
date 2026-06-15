"""
Backward-compatible training wrapper.

The implementation has been split into:
- src/feature_engineering.py
- src/train_model.py
- src/predict.py

Run either:
    python src/fraud_model.py
or:
    python src/train_model.py
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train_model import main, train

if __name__ == "__main__":
    main()
