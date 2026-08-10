from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model folder Path
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Model Files
MODEL_PATH = ARTIFACTS_DIR / "logistic_regression_model.pkl"
TFIDF_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "GFG_FINAL.csv"
